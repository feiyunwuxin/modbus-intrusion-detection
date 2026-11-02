"""
exp_hybrid_fix.py -- 针对 "权重量化 -> FPR 上升" 的修正方案 PC 侧实验.

方案:
  base   : Route E FP32 (部署真值)
  int8   : Hybrid-INT8  (现有 hybrid header, absmax/127)          -> FPR 0.0689 (已知)
  int16  : Hybrid-INT16 (absmax/32767)                             -> 期望 ≈ FP32
  mse    : Hybrid-INT8  + MSE 最优裁剪 (per-channel 搜截断阈值)      -> 期望比 int8 好
  bcorr  : Hybrid-INT8  + 逐层偏置校正 (train 区标定)                -> 期望抵消系统性上偏

标定纪律: mse 的裁剪阈值 / bcorr 的 bias 全部 **只在 train 区 (frame<200001) 上拟合**;
          部署区 (frame>=200001) 只做最终评估.

用法: python exp_hybrid_fix.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hybrid_variants import (bias_correct, build_variant, forward,
                              quantize_dequant)
from sim_hybrid_kernel import load_weights

WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
DEPLOY_FROM = 200001
N_CALIB = 2000
OUT_JSON = os.path.join(HERE, "hybrid_fix_variants.json")


def metrics(det, atk):
    tp = int((det & atk).sum())
    fp = int((det & ~atk).sum())
    fn = int((~det & atk).sum())
    n_norm = int((~atk).sum())
    tn = n_norm - fp
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=P, R=R, F1=F1, FPR=fp / max(n_norm, 1),
                n=int(len(det)))


def main():
    d = np.load(WIN_NPZ)
    X = d["X"].astype(np.float32)
    atk = d["atk"].astype(bool)
    frames = d["frames"]
    tr = frames < DEPLOY_FROM
    dep = frames >= DEPLOY_FROM
    print("train 区 %d 窗 / 部署区 %d 窗" % (int(tr.sum()), int(dep.sum())))

    W32 = load_weights("fp32")
    Why = load_weights("int8")           # 现有 hybrid header (absmax/127)

    # 一致性检查: header 里的 int8 是否等于从 FP32 头重算的 absmax/127 量化
    wq, _ = quantize_dequant(W32["w_tcn_0_conv1"], bits=8)
    same = np.allclose(wq, Why["w_tcn_0_conv1"], atol=1e-7)
    print("[一致性] hybrid header 的 int8 == 从 FP32 头重算的 absmax/127 量化: %s" % same)

    variants = {}
    variants["base (FP32)"] = W32
    variants["int8 (现状)"] = Why
    t0 = time.time()
    variants["int16"] = build_variant(W32, bits=16)
    print("[构建] int16 变体 %.1fs" % (time.time() - t0))
    t0 = time.time()
    variants["mse"] = build_variant(W32, bits=8, mse_search=True)
    print("[构建] mse 变体 %.1fs" % (time.time() - t0))

    # --- 偏置校正: 只用 train 区标定 ---
    rng = np.random.default_rng(20261027)
    cal_idx = np.where(tr)[0]
    cal_idx = rng.choice(cal_idx, size=min(N_CALIB, len(cal_idx)), replace=False)
    Xcal = X[cal_idx]
    print("[标定] 偏置校正用 %d 个 train 区窗口 (部署区完全不参与)" % len(cal_idx))

    for tag, kwargs in (
        ("int8+bc(线性)", dict(se2_mode="skip")),
        ("int8+bc(线性+pre)", dict(se2_mode="pre")),
        ("int8+bc(线性+out)", dict(se2_mode="out")),
    ):
        t0 = time.time()
        Wbc, adj = bias_correct(W32, Why, Xcal, verbose=False, **kwargs)
        variants[tag] = Wbc
        print("[构建] %s  %.1fs" % (tag, time.time() - t0))

    # --- 评估 ---
    print("\n" + "=" * 104)
    print("%-18s | %-30s | %-30s" % ("变体", "部署区 (frame>=200001)", "train 区 (frame<200001)"))
    print("%-18s | %-30s | %-30s" % ("", "F1      FPR     P      R", "F1      FPR     P      R"))
    print("-" * 104)
    results = {}
    probs = {}
    for name, W in variants.items():
        p, lg = forward(W, X)
        probs[name] = p
        md = metrics(p[dep] >= 0.5, atk[dep])
        mt = metrics(p[tr] >= 0.5, atk[tr])
        results[name] = {"deploy": md, "train": mt}
        print("%-18s | %.4f  %.4f  %.4f  %.4f | %.4f  %.4f  %.4f  %.4f"
              % (name, md["F1"], md["FPR"], md["P"], md["R"], mt["F1"], mt["FPR"], mt["P"], mt["R"]))

    # --- logit 偏移 ---
    print("\n[logit 偏移 vs FP32 (部署区)]")
    p0, lg0 = forward(W32, X[dep])
    for name in variants:
        if name == "base (FP32)":
            continue
        _, lg = forward(variants[name], X[dep])
        d = lg - lg0
        flip = ((probs[name][dep] >= 0.5) != (p0 >= 0.5)).sum()
        print("  %-18s Δlogit 均值=%+.4f  中位数=%+.4f  决策翻转=%d" % (name, d.mean(), np.median(d), flip))

    # --- 阈值扫描 ---
    print("\n[阈值扫描 (部署区)]")
    print("  %-18s %s" % ("变体", "  ".join("thr=%.2f" % t for t in (0.4, 0.45, 0.5, 0.55, 0.6))))
    for name in variants:
        row = []
        for t in (0.4, 0.45, 0.5, 0.55, 0.6):
            m = metrics(probs[name][dep] >= t, atk[dep])
            row.append("%.4f/%.4f" % (m["F1"], m["FPR"]))
        print("  %-18s %s" % (name, "  ".join(row)))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print("\n已写入 %s" % OUT_JSON)


if __name__ == "__main__":
    main()
