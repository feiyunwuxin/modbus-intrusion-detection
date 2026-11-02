"""
compare_fp32_vs_int8.py -- FP32 (部署真值) vs 全量 INT8 的逐窗对比.

对照口径:
  FP32  : fullsim_f32.npz  (由 parity_fp32_c_vs_onnx.c_forward 生成,
          c_forward 直接从 MCU/Core/Src/ch32_inference.c 源码解析 t_in 公式 -> 源绑定)
  INT8  : sim_int8_kernel.Int8Model (numpy 镜像, 语义照抄 CMSIS-NN arm_nn_requantize)

区域口径 (跟 analyze_20261021_final.py 一致):
  部署区   frame >= 200001   (对外报这个)
  官方test frame >= 219701
  train区  frame <  200001

用法:
    python compare_fp32_vs_int8.py                # 主对比
    python compare_fp32_vs_int8.py --se-fp32      # A/B: SE 走 FP32
    python compare_fp32_vs_int8.py --no-sim       # 只重算指标 (读 fullsim_int8.npz)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FP32_NPZ = os.path.join(HERE, "fullsim_f32.npz")
INT8_NPZ = os.path.join(HERE, "fullsim_int8.npz")
QLOG = os.path.join(HERE, "int8_qparams_s42.json")
OUT_JSON = os.path.join(HERE, "int8_vs_fp32_metrics.json")

DEPLOY_FROM = 200001
TEST_FROM = 219701


# ---------------------------------------------------------------------------
# 指标 (Binary, 跟项目 4-metrics 口径一致)
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


def zone_report(probs, atk, frames, thr=0.5, tag=""):
    zones = {
        "全量": np.ones(len(probs), dtype=bool),
        "部署区 (frame>=200001)": frames >= DEPLOY_FROM,
        "官方 test 区 (frame>=219701)": frames >= TEST_FROM,
        "train/val 区 (frame<200001)": frames < DEPLOY_FROM,
    }
    out = {}
    for name, m in zones.items():
        out[name] = metrics(probs[m] >= thr, atk[m])
    return out


def fmt_row(name, m):
    return ("  %-30s n=%5d TP=%5d FP=%4d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
            % (name, m["n"], m["TP"], m["FP"], m["FN"], m["P"], m["R"], m["F1"], m["FPR"]))


# ---------------------------------------------------------------------------
# INT8 全量回放
# ---------------------------------------------------------------------------


def run_int8(se_fp32=False, batch=512, verbose=True):
    from sim_int8_kernel import run_all
    os.environ["H743_CSV"] = "IanArffDatasetF32.csv"
    from h743_windows import build_windows, window_labels

    t0 = time.time()
    wins, labels = build_windows(start=0, n=274628, phase=0)
    if verbose:
        print("[int8] build_windows %d 窗, 用时 %.1fs" % (len(wins), time.time() - t0), flush=True)
    X = np.stack([w[2] for w in wins], axis=0).astype(np.float32)
    frames = np.array([w[0] for w in wins])
    atk = np.array(window_labels(wins, labels), dtype=bool)

    t1 = time.time()
    probs = run_all(X, batch=batch, se_fp32=se_fp32, verbose=False)
    if verbose:
        print("[int8] 前向 %d 窗, 用时 %.1fs" % (len(X), time.time() - t1), flush=True)
    return probs, atk, frames


# ---------------------------------------------------------------------------


def main():
    args = sys.argv[1:]
    se_fp32 = "--se-fp32" in args
    no_sim = "--no-sim" in args

    d32 = np.load(FP32_NPZ)
    p32, atk32, fr32 = d32["probs"], d32["atk"].astype(bool), d32["frames"]

    if no_sim:
        d8 = np.load(INT8_NPZ.replace(".npz", "_sefp32.npz") if se_fp32 else INT8_NPZ)
        p8, atk8, fr8 = d8["probs"], d8["atk"].astype(bool), d8["frames"]
        assert len(p8) == len(p32), (len(p8), len(p32))
    else:
        p8, atk8, fr8 = run_int8(se_fp32=se_fp32)
        out_npz = INT8_NPZ.replace(".npz", "_sefp32.npz") if se_fp32 else INT8_NPZ
        np.savez(out_npz, probs=p8, atk=atk8, frames=fr8,
                 variant="se_fp32" if se_fp32 else "int8")
        print("[int8] 已写 %s" % out_npz)
        # FP32 侧也落一份 (同一网格, 保证逐窗对齐)
        if not np.array_equal(fr8, fr32):
            raise SystemExit("窗口网格跟 FP32 基线不一致, 拒绝比较")

    assert np.array_equal(atk8, atk32), "标签不一致"
    assert np.array_equal(fr8, fr32), "frame 网格不一致"

    print("=" * 92)
    print("FP32 (部署真值) vs 全量 INT8   [变体: %s]"
          % ("SE 走 FP32" if se_fp32 else "全 INT8 (SE 定点 Q12)"))
    print("=" * 92)

    m32 = zone_report(p32, atk32, fr32)
    m8 = zone_report(p8, atk8, fr8)

    print("\n[1] 分区域指标 (thr=0.5, Binary)")
    for z in m32:
        print("  --- %s ---" % z)
        print(fmt_row("FP32", m32[z]))
        print(fmt_row("INT8", m8[z]))
        print("      ΔF1=%+.4f  ΔFPR=%+.4f (%+.2f%%)  ΔR=%+.4f  ΔP=%+.4f"
              % (m8[z]["F1"] - m32[z]["F1"],
                 m8[z]["FPR"] - m32[z]["FPR"], 100 * (m8[z]["FPR"] - m32[z]["FPR"]),
                 m8[z]["R"] - m32[z]["R"], m8[z]["P"] - m32[z]["P"]))

    print("\n[2] 阈值扫描")
    for zname, msk in (("部署区", fr32 >= DEPLOY_FROM), ("全量", np.ones(len(p32), bool))):
        print("  --- %s ---" % zname)
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            a = metrics(p32[msk] >= thr, atk32[msk])
            b = metrics(p8[msk] >= thr, atk8[msk])
            print("    thr=%.2f  FP32: F1=%.4f FPR=%.4f R=%.4f  |  INT8: F1=%.4f FPR=%.4f R=%.4f  |  ΔF1=%+.4f"
                  % (thr, a["F1"], a["FPR"], a["R"], b["F1"], b["FPR"], b["R"], b["F1"] - a["F1"]))

    print("\n[3] 概率漂移 (FP32 vs INT8)")
    diff = p8 - p32
    ad = np.abs(diff)
    print("  MAE=%.6f  max|Δ|=%.6f  P99|Δ|=%.6f  RMSE=%.6f"
          % (ad.mean(), ad.max(), np.percentile(ad, 99), np.sqrt((diff ** 2).mean())))
    print("  相关系数 r=%.6f" % np.corrcoef(p32, p8)[0, 1])
    for thr in (0.5,):
        f32 = p32 >= thr
        f8 = p8 >= thr
        flip = f32 != f8
        tp32 = f32 & atk32
        print("  thr=%.2f 决策翻转率=%.4f (%d/%d 窗)" % (thr, flip.mean(), flip.sum(), len(flip)))
        print("    FP32 TP=%d -> INT8 TP=%d (丢 %d), FP32 FP=%d -> INT8 FP=%d (%+d)"
              % (tp32.sum(), (f8 & atk32).sum(), (tp32 & ~f8).sum(),
                 (f32 & ~atk32).sum(), (f8 & ~atk8).sum(),
                 (f8 & ~atk8).sum() - (f32 & ~atk32).sum()))
    # 只看 FP32 判为 attack 的窗里的漂移 (更贴近"漏报从哪来")
    sel32 = p32 >= 0.5
    if sel32.any():
        print("  FP32 判 attack 的窗内: MAE=%.6f  max|Δ|=%.6f  被 INT8 翻成 normal 的比例=%.4f"
              % (ad[sel32].mean(), ad[sel32].max(), (p8[sel32] < 0.5).mean()))

    print("\n[4] 汇总 (部署区, thr=0.5)")
    print("  FP32: F1=%.4f FPR=%.4f R=%.4f P=%.4f"
          % (m32["部署区 (frame>=200001)"]["F1"], m32["部署区 (frame>=200001)"]["FPR"],
             m32["部署区 (frame>=200001)"]["R"], m32["部署区 (frame>=200001)"]["P"]))
    print("  INT8: F1=%.4f FPR=%.4f R=%.4f P=%.4f"
          % (m8["部署区 (frame>=200001)"]["F1"], m8["部署区 (frame>=200001)"]["FPR"],
             m8["部署区 (frame>=200001)"]["R"], m8["部署区 (frame>=200001)"]["P"]))

    out = {"variant": "se_fp32" if se_fp32 else "int8",
           "fp32": {k: v for k, v in m32.items()},
           "int8": {k: v for k, v in m8.items()},
           "drift": {"mae": float(ad.mean()), "max": float(ad.max()),
                     "p99": float(np.percentile(ad, 99)),
                     "rmse": float(np.sqrt((diff ** 2).mean())),
                     "r": float(np.corrcoef(p32, p8)[0, 1]),
                     "flip_rate_0.5": float(((p32 >= 0.5) != (p8 >= 0.5)).mean())}}
    with open(OUT_JSON.replace(".json", "_%s.json" % out["variant"]), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n指标已写入 %s" % OUT_JSON.replace(".json", "_%s.json" % out["variant"]))


if __name__ == "__main__":
    main()
