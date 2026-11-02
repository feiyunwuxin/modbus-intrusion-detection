"""F 路线: ARFF train/val/test 23-dim 分布诊断

目的: 看 ARFF 内部 (TRAIN 0..192k) vs (VAL 192k..219k) vs (TEST 219k..274k)
       在 23 个 channel 上的分布漂移.
       这是数据流诊断的 baseline — 找出 ARFF 自身哪些 channel 时序漂移.

输出:
  - diag_f_route_results.md  (markdown 报告)
  - diag_f_route_drift.png    (per-channel Wasserstein bar chart)
  - diag_f_route_dist.png     (per-channel train vs test KDE / hist)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
import json, sys

OUT_MD = "C:/work/Claude/Issue/diag_f_route_results.md"
OUT_PNG_DRIFT = "C:/work/Claude/Issue/diag_f_route_drift.png"
OUT_PNG_DIST = "C:/work/Claude/Issue/diag_f_route_dist.png"

# ARFF 17 列名 (model 23-dim 由 v2 preprocessing 派生, 这里给原始 ARFF 列名映射)
# keep_idx = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
# 这里只看 model 直接吃的 23-dim 已窗口化数据 (在 X_*.npy 里)
# npys 是 (N_windows, T=16, C=23), 我们 flatten 成 (N_windows*T, C) 再做 channel-wise 分布

DATA_DIR = "C:/work/Claude/Issue"
# 27-dim raw -> 23-dim model input (按 gen_mock_swdbg2.py keep_idx)
KEEP_IDX = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
print("STEP 1: load 27-dim npy, slice -> 23-dim via keep_idx ...")
X_tr = np.load(f"{DATA_DIR}/X_train_binary_v2_scada_window16.npy")[..., KEEP_IDX]  # (12014, 16, 23)
X_va = np.load(f"{DATA_DIR}/X_val_binary_v2_scada_window16.npy")[..., KEEP_IDX]    # (1716, 16, 23)
X_te = np.load(f"{DATA_DIR}/X_test_binary_v2_scada_window16.npy")[..., KEEP_IDX]   # (3432, 16, 23)
print(f"  X_tr  {X_tr.shape}  X_va  {X_va.shape}  X_te  {X_te.shape}")

# Flatten (N, T, C) -> (N*T, C) 方便 channel-wise 分布
X_tr_f = X_tr.reshape(-1, 23)
X_va_f = X_va.reshape(-1, 23)
X_te_f = X_te.reshape(-1, 23)
print(f"  flattened: tr={X_tr_f.shape} va={X_va_f.shape} te={X_te_f.shape}")

# Channel 名 (23-dim model 输入, 顺序按 gen_mock_swdbg2.py keep_idx)
# keep_idx = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
# 这 23 个 index 来自 v2 preprocessing, 不是原始 ARFF 列;
# 取有意义的描述: address, function, gain, reset rate, deadband, cycle time,
# rate, system mode, control scheme, pump, solenoid, pressure, crc rate,
# cmd response, time, gap_prev, dt (time diff), is_attack, attack_type,
# func1_count, func3_count, fc4_count, ...  (23 个)
# 但更稳的是用 ch00..ch22 然后在报告里逐个对照 v2 pipeline 输出.

CH_NAMES = [f"ch{c:02d}" for c in range(23)]

print()
print("STEP 2: per-channel stats (train / val / test) ...")
stats = {}
for c in range(23):
    a_tr = X_tr_f[:, c]
    a_va = X_va_f[:, c]
    a_te = X_te_f[:, c]
    stats[c] = {
        "tr_mean": float(a_tr.mean()), "tr_std": float(a_tr.std()),
        "tr_p01": float(np.percentile(a_tr, 1)), "tr_p99": float(np.percentile(a_tr, 99)),
        "va_mean": float(a_va.mean()), "va_std": float(a_va.std()),
        "te_mean": float(a_te.mean()), "te_std": float(a_te.std()),
    }
    print(f"  ch{c:02d}  tr_mean={a_tr.mean():+.3f} std={a_tr.std():.3f}   "
          f"te_mean={a_te.mean():+.3f} std={a_te.std():.3f}")

print()
print("STEP 3: drift metrics (Wasserstein, KS) ...")
drift = []
for c in range(23):
    a_tr = X_tr_f[:, c]
    a_te = X_te_f[:, c]
    a_va = X_va_f[:, c]
    w_tr_te = wasserstein_distance(a_tr, a_te)
    w_tr_va = wasserstein_distance(a_tr, a_va)
    w_va_te = wasserstein_distance(a_va, a_te)
    ks_tr_va, _ = ks_2samp(a_tr, a_va)
    ks_tr_te, _ = ks_2samp(a_tr, a_te)
    ks_va_te, _ = ks_2samp(a_va, a_te)
    drift.append({
        "ch": c,
        "w_tr_te": w_tr_te,
        "w_tr_va": w_tr_va,
        "w_va_te": w_va_te,
        "ks_tr_te": ks_tr_te,
        "ks_tr_va": ks_tr_va,
        "ks_va_te": ks_va_te,
        "mean_diff_tr_te": abs(stats[c]["te_mean"] - stats[c]["tr_mean"]),
    })

# Rank by Wasserstein train↔test (主指标)
drift_sorted = sorted(drift, key=lambda x: x["w_tr_te"], reverse=True)
print(f"  Top 5 drift (Wasserstein train↔test):")
for r in drift_sorted[:5]:
    c = r["ch"]
    s = stats[c]
    print(f"    ch{c:02d}  W_tr_te={r['w_tr_te']:.4f}  KS={r['ks_tr_te']:.4f}  "
          f"|Δmean|={r['mean_diff_tr_te']:.4f}  "
          f"tr=[{s['tr_mean']:+.3f}±{s['tr_std']:.3f}]  te=[{s['te_mean']:+.3f}±{s['te_std']:.3f}]")

print()
print("STEP 4: save plot (drift bar chart) ...")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
chs = [d["ch"] for d in drift_sorted]
w_tr_te = [d["w_tr_te"] for d in drift_sorted]
ks_tr_te = [d["ks_tr_te"] for d in drift_sorted]

ax[0].bar(range(23), w_tr_te, color="steelblue")
ax[0].set_xticks(range(23))
ax[0].set_xticklabels([f"ch{c:02d}" for c in chs], rotation=45)
ax[0].set_ylabel("Wasserstein dist")
ax[0].set_title("Per-channel drift ARFF train ↔ test (sorted desc)")

ax[1].bar(range(23), ks_tr_te, color="darkorange")
ax[1].set_xticks(range(23))
ax[1].set_xticklabels([f"ch{c:02d}" for c in chs], rotation=45)
ax[1].set_ylabel("KS statistic")
ax[1].set_title("Per-channel KS ARFF train ↔ test (sorted desc)")

plt.tight_layout()
plt.savefig(OUT_PNG_DRIFT, dpi=110)
print(f"  saved {OUT_PNG_DRIFT}")

print()
print("STEP 5: save distribution plot (top-6 channels train vs test KDE) ...")
top6 = [d["ch"] for d in drift_sorted[:6]]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for i, c in enumerate(top6):
        r = i // 3
        col = i % 3
        ax = axes[r][col]
        bins = 60
        ax.hist(X_tr_f[:, c], bins=bins, density=True, alpha=0.5, label="train", color="steelblue")
        ax.hist(X_te_f[:, c], bins=bins, density=True, alpha=0.5, label="test", color="darkorange")
        ax.set_title(f"ch{c:02d} (W={drift_sorted[i]['w_tr_te']:.4f})")
        ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG_DIST, dpi=110)
print(f"  saved {OUT_PNG_DIST}")

print()
print("STEP 6: write markdown report ...")
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("# F 路线 — ARFF 23-dim 分布诊断报告\n\n")
    f.write(f"## 数据规模\n")
    f.write(f"- train windows: {X_tr.shape[0]} ({X_tr.shape[0]*X_tr.shape[1]} samples flattened)\n")
    f.write(f"- val windows:   {X_va.shape[0]}\n")
    f.write(f"- test windows:  {X_te.shape[0]}\n\n")
    f.write("## 数据切分\n")
    f.write("- ARFF 行顺序切分 (70/10/20)\n")
    f.write("- TRAIN 行 0..192239 (12014 windows after W=16)\n")
    f.write("- VAL 行 192239..219701 (1716 windows)\n")
    f.write("- TEST 行 219701..274628 (3432 windows)\n")
    f.write("- 中间 192239..219701 约 27k 行是 gap\n\n")

    f.write("## Top 10 drift channels (Wasserstein train↔test)\n\n")
    f.write("| Rank | ch | Wasserstein | KS | |Δmean| | tr_mean | te_mean |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for i, d in enumerate(drift_sorted[:10]):
        c = d["ch"]
        s = stats[c]
        f.write(f"| {i+1} | ch{c:02d} | {d['w_tr_te']:.4f} | {d['ks_tr_te']:.4f} | "
                f"{d['mean_diff_tr_te']:.4f} | {s['tr_mean']:+.3f} | {s['te_mean']:+.3f} |\n")

    f.write("\n## 决策依据\n\n")
    top5 = drift_sorted[:5]
    f.write("- 若 Top-5 都是 **时序类** (time, dt, gap_prev, is_attack_window) -> 真因是 ARFF 时序结构, 走 **路线 D (random split retrain)**\n")
    f.write("- 若 Top-5 都是 **数值范围类** (pressure, gain, setpoint 等) -> 真因是传感器分布漂移, 走 **路线 E (data augmentation 加 noise)**\n")
    f.write("- 若 Top-5 都是 **类别 one-hot** (function code, control scheme) -> 真因是 Modbus traffic profile 不同, 走 **路线 E (加 traffic mix-up)**\n\n")

    f.write("## 全部 channel 全表 (Wasserstein train↔test)\n\n")
    f.write("| ch | W | KS | tr_std | te_std |\n")
    f.write("|---|---|---|---|---|\n")
    for d in drift_sorted:
        c = d["ch"]
        s = stats[c]
        f.write(f"| ch{c:02d} | {d['w_tr_te']:.4f} | {d['ks_tr_te']:.4f} | "
                f"{s['tr_std']:.3f} | {s['te_std']:.3f} |\n")

    f.write("\n## 下一步\n")
    f.write("- 把 H743 实际 [SW-DBG2] dump 也对比, 看是否同样的 channel 漂移\n")
    f.write("- 对照 `[[project-2026-10-17-fpr-root-cause-final.md]]`: 数据漂移 6%→57% (TRAIN 0..192k vs TEST 219k+) 单独贡献 8 FP\n")

print(f"  saved {OUT_MD}")
print()
print("DONE.")