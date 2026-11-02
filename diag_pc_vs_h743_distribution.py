#!/usr/bin/env python3
"""
PC forward vs H743 distribution 对比.
判断 0.999 饱和是 H743 Hybrid INT8 特有还是 TCN+SE 模型本身特性.

数据:
  - PC test set: X_test_binary_v2_scada_window16.npy (3432 windows, 1627 normal / 1805 attack)
  - H743 test:   logs/test_20261013-003345 (17165 windows, 8343 normal / 8822 attack)
  - 模型: 同一 seed=42, 同一架构 (ch=32, 23-dim, TCN+SE)
  - H743 路径: Hybrid INT8 (init 时 INT8→FP32 dequant, 推理纯 FP32)
"""
import sys, os, collections
import numpy as np
import torch

sys.path.insert(0, r"C:\work\Claude\Issue")
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEED = 42

# Load PC test
X = np.load("X_test_binary_v2_scada_window16.npy").astype(np.float32)
y = np.load("y_test_binary_v2_scada_window16.npy").astype(np.int64)
X = np.clip(X[:, :, KEEP_23], -10.0, 10.0).transpose(0, 2, 1)  # (N,F,T)
print(f"PC X_test: {X.shape}, y={y.shape}, normal={np.sum(y==0)}, attack={np.sum(y==1)}")

# Load PC model (FP32, no quantization)
ckpt = torch.load(f"model_v4_se_23dim_b64_ch32_do01_window16_s{SEED}.pt", map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23)
model.load_state_dict(ckpt['state_dict'])
model.eval()

with torch.no_grad():
    logits = model(torch.from_numpy(X)).numpy().squeeze()
print(f"PC logits range: [{logits.min():.3f}, {logits.max():.3f}], mean={logits.mean():.3f}, std={logits.std():.3f}")

probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
print(f"PC probs range: [{probs.min():.6f}, {probs.max():.6f}]")

# Distribution buckets
print()
print("=" * 60)
print("PC forward output probability distribution (N=3432):")
print("=" * 60)
buckets = [
    (0.000, 0.001), (0.001, 0.100), (0.100, 0.300), (0.300, 0.500),
    (0.500, 0.700), (0.700, 0.900), (0.900, 0.950), (0.950, 0.990),
    (0.990, 0.995), (0.995, 0.998), (0.998, 0.999), (0.999, 1.000),
]
print(f'{"range":>20} {"count":>7} {"TP(at)":>7} {"FP(no)":>7} {"prec%":>7}')
for lo, hi in buckets:
    mask = (probs >= lo) & (probs < hi)
    n = mask.sum()
    n_att = ((mask) & (y == 1)).sum()
    n_norm = ((mask) & (y == 0)).sum()
    prec = n_att / max(n, 1) * 100
    print(f'[{lo:.3f}, {hi:.3f}): {n:>7d} {n_att:>7d} {n_norm:>7d} {prec:>6.1f}%')

# Compare with H743 distribution
print()
print("=" * 60)
print("H743 (test_20261013-003345) distribution (N=7525 alarms):")
print("=" * 60)
print("0.999      : 5681 (75.5%)  prec=99.0%")
print("0.998-0.999:  525 (7.0%)   prec=~90%")
print("0.995-0.998:  497 (6.6%)   prec=~89%")
print("0.990-0.995:  443 (5.9%)   prec=~88%")
print("0.950-0.990:  720 (9.6%)   prec=~75%")
print("0.900-0.950:  299 (4.0%)   prec=~50%")

# Threshold sweep on PC
print()
print("=" * 60)
print("PC forward threshold sweep:")
print("=" * 60)
print(f'{"thr":>6} {"TP":>5} {"FP":>5} {"FN":>5} {"TN":>5} {"Rec":>6} {"FPR":>6} {"Prec":>6} {"F1":>6}')
for thr in [0.10, 0.30, 0.40, 0.45, 0.49, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]:
    yp = (probs >= thr).astype(int)
    TP = ((yp == 1) & (y == 1)).sum()
    FP = ((yp == 1) & (y == 0)).sum()
    FN = ((yp == 0) & (y == 1)).sum()
    TN = ((yp == 0) & (y == 0)).sum()
    rec = TP / max(TP + FN, 1) * 100
    fpr = FP / max(FP + TN, 1) * 100
    prec = TP / max(TP + FP, 1) * 100
    f1 = 2 * prec * rec / max(prec + rec, 1)
    print(f'{thr:>6.3f} {TP:>5d} {FP:>5d} {FN:>5d} {TN:>5d} {rec:>5.2f}% {fpr:>5.2f}% {prec:>5.2f}% {f1:>5.2f}%')

# Save probs for offline analysis
np.savez("pc_vs_h743_probs.npz",
         pc_probs=probs, pc_y=y, pc_logits=logits,
         h743_alarm_dist="see analysis output")
print()
print("Saved to pc_vs_h743_probs.npz")
