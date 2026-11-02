"""
exp_logit_trim.py -- 在 int8(+逐层偏置校正) 之上做 **logit 级整体 trim**, 看能否把 FPR 拉回 FP32 的 5.91%.

背景 (exp_hybrid_fix.py)
------------------------
  base (FP32)      部署区 F1 0.8671 / FPR 0.0591
  int8 (现状)      F1 0.8663 / FPR 0.0689   <- 权重量化的系统性上偏
  int8+bc(线性)     F1 0.8656 / FPR 0.0649   <- 逐通道偏置校正后仍差 0.58pp
  int16            F1 0.8671 / FPR 0.0591   <- 与 FP32 完全一致 (0 翻转)

  残差: 逐层校正只覆盖到 SE fc2, **分类头 (fc1 / fc2) 的 bias 没动**,
        所以最终 logit 还残留一个常数偏移. 这一步补上.

三种 trim 口径 (全部只用 train 区标定, 部署区只做评估)
-------------------------------------------------------
  mean    : b_fc2 += mean(logit_fp32) - mean(logit_q)          (标定集上)
  median  : b_fc2 += median(logit_fp32) - median(logit_q)      (对离群点稳健)
  fpr     : 选 b_fc2 的偏移, 使标定集上的 **FPR 与 FP32 模型一致**
            (1-D 单调求根; 直接对齐对外报的 FPR 指标)
  none    : 不 trim (对照)

用法: python exp_logit_trim.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hybrid_variants import bias_correct, forward, forward_hooks
from sim_hybrid_kernel import load_weights

WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
DEPLOY_FROM = 200001
N_CALIB = 2000
OUT_JSON = os.path.join(HERE, "logit_trim_results.json")


def metrics(det, atk):
    tp = int((det & atk).sum())
    fp = int((det & ~atk).sum())
    fn = int((~det & atk).sum())
    n_norm = int((~atk).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return dict(TP=tp, FP=fp, FN=fn, TN=n_norm - fp, P=P, R=R, F1=F1,
                FPR=fp / max(n_norm, 1), n=int(len(det)))


def trim_mean(W32, Wq, Xcal):
    W = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Wq.items()}
    h32 = forward_hooks(W32, Xcal)
    hq = forward_hooks(W, Xcal)
    d = float(h32["logit"].mean() - hq["logit"].mean())
    W["b_fc2"] = W["b_fc2"] + d
    return W, d


def trim_median(W32, Wq, Xcal):
    W = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Wq.items()}
    h32 = forward_hooks(W32, Xcal)
    hq = forward_hooks(W, Xcal)
    d = float(np.median(h32["logit"]) - np.median(hq["logit"]))
    W["b_fc2"] = W["b_fc2"] + d
    return W, d


def trim_fpr(W32, Wq, Xcal, atk_cal):
    """选 b_fc2 偏移使标定集 FPR 与 FP32 模型一致 (1-D 单调二分)."""
    W = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Wq.items()}
    h32 = forward_hooks(W32, Xcal)
    hq = forward_hooks(W, Xcal)
    lg32, lgq = h32["logit"], hq["logit"]
    norm = ~atk_cal.astype(bool)

    def fpr_of(delta):
        return float(((lgq + delta >= 0) & norm).sum()) / max(int(norm.sum()), 1)

    target = fpr_of(0.0)                       # FP32 模型在标定集上的 FPR
    lo, hi = -10.0, 10.0
    flo, fhi = fpr_of(lo) - target, fpr_of(hi) - target
    if flo > 0:
        delta = lo
    elif fhi < 0:
        delta = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if fpr_of(mid) - target < 0:
                lo = mid
            else:
                hi = mid
        delta = 0.5 * (lo + hi)
    W["b_fc2"] = W["b_fc2"] + delta
    return W, delta, target


def main():
    d = np.load(WIN_NPZ)
    X = d["X"].astype(np.float32)
    atk = d["atk"].astype(bool)
    frames = d["frames"]
    tr = frames < DEPLOY_FROM
    dep = frames >= DEPLOY_FROM

    W32 = load_weights("fp32")
    Why = load_weights("int8")

    rng = np.random.default_rng(20261027)
    cal_idx = rng.choice(np.where(tr)[0], size=min(N_CALIB, int(tr.sum())), replace=False)
    Xcal, atk_cal = X[cal_idx], atk[cal_idx]
    print("[标定] %d 个 train 区窗口; 部署区 %d 窗仅用于评估"
          % (len(cal_idx), int(dep.sum())))

    # 基础变体
    print("[构建] int8 + 逐通道偏置校正 (se2=skip) ...")
    Wbc, _ = bias_correct(W32, Why, Xcal, verbose=False, se2_mode="skip")

    bases = {"int8 原始": Why, "int8+bc": Wbc}
    trims = {"none": None, "mean": trim_mean, "median": trim_median}

    rows = []
    print("\n" + "=" * 100)
    print("%-22s | %-30s | %-9s | %-9s | %s"
          % ("变体", "部署区 F1/FPR/P/R", "ΔFPR", "翻转", "trim δ"))
    print("-" * 100)

    # FP32 基线
    p32d, lg32d = forward(W32, X[dep])
    m32 = metrics(p32d >= 0.5, atk[dep])
    print("%-22s | %.4f  %.4f  %.4f  %.4f | %-9s | %-9s | %s"
          % ("base (FP32)", m32["F1"], m32["FPR"], m32["P"], m32["R"],
             "-", "-", "-"))

    results = {"base (FP32)": {"deploy": m32}}
    for bname, W0 in bases.items():
        for tname, fn in trims.items():
            if fn is None:
                W, delta = W0, 0.0
            else:
                W, delta = fn(W32, W0, Xcal)
            pd, lgd = forward(W, X[dep])
            m = metrics(pd >= 0.5, atk[dep])
            flip = int(((pd >= 0.5) != (p32d >= 0.5)).sum())
            name = "%s + %s" % (bname, tname)
            print("%-22s | %.4f  %.4f  %.4f  %.4f | %+.4f | %-9d | %+.4f"
                  % (name, m["F1"], m["FPR"], m["P"], m["R"],
                     m["FPR"] - m32["FPR"], flip, delta))
            rows.append((name, m, flip, delta))
            results[name] = {"deploy": m, "flip": flip, "trim_delta": delta}

        # FPR 匹配 trim (需要标定集标签)
        W, delta, target = trim_fpr(W32, W0, Xcal, atk_cal)
        pd, lgd = forward(W, X[dep])
        m = metrics(pd >= 0.5, atk[dep])
        flip = int(((pd >= 0.5) != (p32d >= 0.5)).sum())
        name = "%s + fpr匹配" % bname
        print("%-22s | %.4f  %.4f  %.4f  %.4f | %+.4f | %-9d | %+.4f  (标定集目标 FPR=%.4f)"
              % (name, m["F1"], m["FPR"], m["P"], m["R"],
                 m["FPR"] - m32["FPR"], flip, delta, target))
        results[name] = {"deploy": m, "flip": flip, "trim_delta": delta,
                         "calib_target_fpr": target}

    # train 区也要看一眼 (trim 是在 train 区标的, 不能过拟合部署区)
    print("\n[train 区 (frame<200001) 对照]")
    for name in results:
        pass

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n已写入 %s" % OUT_JSON)


if __name__ == "__main__":
    main()
