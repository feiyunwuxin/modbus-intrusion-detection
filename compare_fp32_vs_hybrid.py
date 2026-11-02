"""
compare_fp32_vs_hybrid.py -- FP32 (部署真值) vs Hybrid (INT8 权重 / FP32 激活) vs 全量 INT8.

要回答的问题
------------
全量 INT8 的 logit 相对偏移 +4.2% 把工作点整体推向"更爱报警"(FPR +1.07pp),
58 个分歧窗全部落在 |logit| < 2.0 的边界带.  Hybrid 只量化权重、激活保持 FP32,
问: **能不能把 FPR 拉回 FP32 的 5.91% 而不牺牲 Flash/RAM?**

同时输出一个 2x2 误差归因表:
              激活 FP32          激活 INT8(fake-quant)
    权重 FP32   Route E 真值       只量化激活
    权重 INT8   <-- Hybrid         全量 INT8 (近似)

口径 (跟项目 4-metrics 口径一致, 见 compare_fp32_vs_int8.py)
------------------------------------------------------------
  部署区   frame >= 200001   (对外报这个)
  官方test frame >= 219701
  train区  frame <  200001
  Binary 指标, thr = 0.5

用法:
    python compare_fp32_vs_hybrid.py                # 全量对比
    python compare_fp32_vs_hybrid.py --no-sim       # 只重算指标 (读 fullsim_hybrid.npz)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
FP32_NPZ = os.path.join(HERE, "fullsim_f32.npz")
INT8_NPZ = os.path.join(HERE, "fullsim_int8.npz")
HYB_NPZ = os.path.join(HERE, "fullsim_hybrid.npz")
OUT_JSON = os.path.join(HERE, "hybrid_vs_fp32_metrics.json")

DEPLOY_FROM = 200001
TEST_FROM = 219701
N_BOOT = 2000

ZONES = [
    ("全量", lambda fr: np.ones(len(fr), dtype=bool)),
    ("部署区 (frame>=200001)", lambda fr: fr >= DEPLOY_FROM),
    ("官方 test 区 (frame>=219701)", lambda fr: fr >= TEST_FROM),
    ("train/val 区 (frame<200001)", lambda fr: fr < DEPLOY_FROM),
]


# ---------------------------------------------------------------------------
# 指标 (Binary)
# ---------------------------------------------------------------------------


def metrics(det, atk):
    tp = int((det & atk).sum())
    fp = int((det & ~atk).sum())
    fn = int((~det & atk).sum())
    n_norm = int((~atk).sum())
    tn = n_norm - fp
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    fpr = fp / max(n_norm, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=P, R=R, F1=F1, FPR=fpr,
                n=int(len(det)), n_att=int(atk.sum()))


def fmt_row(name, m):
    return ("  %-24s n=%5d TP=%5d FP=%4d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
            % (name, m["n"], m["TP"], m["FP"], m["FN"], m["P"], m["R"], m["F1"], m["FPR"]))


# ---------------------------------------------------------------------------
# 统计检验
# ---------------------------------------------------------------------------


def _f1fpr(det, atk, n_att, n_norm):
    tp = int((det & atk).sum())
    fp = int((det & ~atk).sum())
    fn = n_att - tp
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return F1, fp / max(n_norm, 1)


def bootstrap_delta(p_a, p_b, atk, thr=0.5, n_boot=N_BOOT, seed=20261027):
    """按窗重采样, 返回 (ΔF1 CI, ΔFPR CI, P(a 更好)).

    Δ = variant_a - variant_b (a = 被检验变体, b = FP32 基线).
    """
    rng = np.random.default_rng(seed)
    n = len(p_a)
    da = (p_a >= thr).astype(bool)
    db = (p_b >= thr).astype(bool)
    a = atk.astype(bool)
    n_att_all = int(a.sum())

    d_f1 = np.empty(n_boot)
    d_fpr = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aa = a[idx]
        n_att = int(aa.sum())
        n_norm = n - n_att
        f1a, fpr_a = _f1fpr(da[idx], aa, n_att, n_norm)
        f1b, fpr_b = _f1fpr(db[idx], aa, n_att, n_norm)
        d_f1[i] = f1a - f1b
        d_fpr[i] = fpr_a - fpr_b
    return (float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5)),
            float(np.percentile(d_fpr, 2.5)), float(np.percentile(d_fpr, 97.5)),
            float((d_f1 > 0).mean()))


def mcnemar_exact(p_a, p_b, atk, thr=0.5):
    """决策分歧的精确 McNemar 检验. 返回 (b, c, p)."""
    from scipy.stats import binomtest
    da = (p_a >= thr)
    db = (p_b >= thr)
    b = int((da & ~db).sum())       # 变体独正
    c = int((~da & db).sum())       # 基线独正
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_windows():
    d = np.load(WIN_NPZ)
    X = d["X"].astype(np.float32)
    atk = d["atk"].astype(bool)
    frames = d["frames"]
    d32 = np.load(FP32_NPZ)
    assert np.array_equal(frames, d32["frames"]), "窗口网格跟 FP32 基线不一致"
    assert np.array_equal(atk, d32["atk"].astype(bool)), "标签不一致"
    return X, atk, frames


def run_sim(X, weights_mode, act_mode):
    from sim_hybrid_kernel import run_all
    t0 = time.time()
    p = run_all(X, weights_mode=weights_mode, act_mode=act_mode, verbose=False)
    print("  [%s/%s] %d 窗, %.1fs" % (weights_mode, act_mode, len(X), time.time() - t0), flush=True)
    return p


def flip_breakdown(p_var, p_base, atk, thr=0.5):
    dv = p_var >= thr
    db = p_base >= thr
    return {
        "TP->FN (变差:漏报)": int((db & atk & ~dv).sum()),
        "TN->FP (变差:误报)": int((~db & ~atk & dv).sum()),
        "FP->TN (变好)": int((db & ~atk & ~dv).sum()),
        "FN->TP (变好)": int((~db & atk & dv).sum()),
    }


def main():
    args = sys.argv[1:]
    no_sim = "--no-sim" in args

    X, atk, frames = load_windows()
    masks = {name: fn(frames) for name, fn in ZONES}
    print("窗口 %d 个 (部署区 %d 个)" % (len(X), int(masks["部署区 (frame>=200001)"].sum())))

    if no_sim:
        variants = {"fp32/fp32": np.load(FP32_NPZ)["probs"],
                    "int8/fp32": np.load(HYB_NPZ)["probs"],
                    "int8/int8": np.load(INT8_NPZ)["probs"]}
    else:
        variants = {}
        for wm, am in (("fp32", "fp32"), ("int8", "fp32"), ("int8", "int8"), ("fp32", "int8")):
            variants["%s/%s" % (wm, am)] = run_sim(X, wm, am)
        np.savez(HYB_NPZ, probs=variants["int8/fp32"], atk=atk, frames=frames,
                 variant="int8_weights_fp32_act")
        np.savez(os.path.join(HERE, "fullsim_actonly.npz"),
                 probs=variants["fp32/int8"], atk=atk, frames=frames,
                 variant="fp32_weights_int8_act")

    p32 = variants["fp32/fp32"]
    p_hyb = variants["int8/fp32"]
    p_i8 = variants["int8/int8"]
    p_act = variants.get("fp32/int8")
    dep = masks["部署区 (frame>=200001)"]

    print("=" * 96)
    print("FP32 (Route E 部署真值)  vs  Hybrid (INT8 权重 / FP32 激活)  vs  全量 INT8")
    print("=" * 96)

    # ---- 1. 分区域指标 ----
    print("\n[1] 分区域指标 (thr=0.5, Binary)")
    all_m = {}
    for zname, m in masks.items():
        print("  --- %s ---" % zname)
        for tag, p in (("FP32 (真值)", p32), ("Hybrid (i8w/f32a)", p_hyb),
                       ("全量INT8 (参考)", p_i8)):
            mm = metrics(p[m] >= 0.5, atk[m])
            all_m[(tag, zname)] = mm
            print(fmt_row(tag, mm))

    # ---- 2. 阈值扫描 (部署区) ----
    print("\n[2] 阈值扫描 (部署区, frame>=200001)")
    print("    thr      FP32 F1/FPR       Hybrid F1/FPR      全量INT8 F1/FPR")
    for thr in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        a = metrics(p32[dep] >= thr, atk[dep])
        b = metrics(p_hyb[dep] >= thr, atk[dep])
        c = metrics(p_i8[dep] >= thr, atk[dep])
        print("    %.2f     %.4f / %.4f      %.4f / %.4f      %.4f / %.4f"
              % (thr, a["F1"], a["FPR"], b["F1"], b["FPR"], c["F1"], c["FPR"]))

    # ---- 3. 误差归因 (2x2) ----
    print("\n[3] 2x2 误差归因 (部署区, thr=0.5)")
    print("    %-24s %-9s %-9s %-9s" % ("", "F1", "FPR", "R"))
    for tag, p in (("权重FP32/激活FP32", p32), ("权重INT8/激活FP32", p_hyb),
                   ("权重FP32/激活INT8", p_act), ("权重INT8/激活INT8", p_i8)):
        if p is None:
            continue
        m = metrics(p[dep] >= 0.5, atk[dep])
        print("    %-24s %-9.4f %-9.4f %-9.4f" % (tag, m["F1"], m["FPR"], m["R"]))

    # ---- 4. 漂移 ----
    print("\n[4] 概率漂移 (部署区)")
    for tag, p in (("Hybrid", p_hyb), ("全量INT8", p_i8)):
        d = p[dep] - p32[dep]
        ad = np.abs(d)
        flip = (p[dep] >= 0.5) != (p32[dep] >= 0.5)
        logratio = np.log(np.clip(p[dep], 1e-12, 1)) - np.log(np.clip(p32[dep], 1e-12, 1))
        print("  %-8s MAE=%.6f  max|Δ|=%.6f  r=%.6f  决策翻转=%d/%d (%.3f%%)"
              % (tag, ad.mean(), ad.max(), np.corrcoef(p32[dep], p[dep])[0, 1],
                 flip.sum(), len(flip), 100 * flip.mean()))
        print("           log 概率比偏移: 均值=%+.4f  中位数=%+.4f"
              % (logratio.mean(), np.median(logratio)))

    # ---- 5. 统计检验 ----
    print("\n[5] 统计检验 (部署区 %d 窗, thr=0.5)" % int(dep.sum()))
    for tag, p in (("Hybrid", p_hyb), ("全量INT8", p_i8)):
        f1lo, f1hi, fprlo, fprhi, pbet = bootstrap_delta(p[dep], p32[dep], atk[dep])
        b, c, pv = mcnemar_exact(p[dep], p32[dep], atk[dep])
        print("  %-8s ΔF1  95%%CI = [%+.4f, %+.4f]   P(该变体更好)=%.3f" % (tag, f1lo, f1hi, pbet))
        print("           ΔFPR 95%%CI = [%+.4f, %+.4f]" % (fprlo, fprhi))
        print("           McNemar: 变体独正=%d, 基线独正=%d, p=%s"
              % (b, c, ("<1e-4" if pv < 1e-4 else "%.4f" % pv)))

    # ---- 6. 逐窗翻转分解 ----
    print("\n[6] 决策翻转分解 (部署区)")
    print("  Hybrid vs FP32:")
    for k, v in flip_breakdown(p_hyb[dep], p32[dep], atk[dep]).items():
        print("    %-22s %d" % (k, v))
    print("  全量INT8 vs FP32 (对照):")
    for k, v in flip_breakdown(p_i8[dep], p32[dep], atk[dep]).items():
        print("    %-22s %d" % (k, v))

    # ---- 落盘 ----
    out = {"zones": {"%s | %s" % (t, z): v for (t, z), v in all_m.items()}}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n指标已写入 %s" % OUT_JSON)


if __name__ == "__main__":
    main()
