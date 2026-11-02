#!/usr/bin/env python3
"""
在 PC 上验证 temperature scaling 的效果.
预测: T>1 把饱和 logits 压回边界区间, 恢复阈值调优空间, 降低 FPR.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

# Load PC forward results (full 274k)
d = np.load("pc_full274k_probs.npz")
logits = d['logits']
yw = d['y']
print(f"Loaded: {len(logits)} windows, attack={np.sum(yw==1)}, normal={np.sum(yw==0)}")
print(f"Original logits: range [{logits.min():.2f}, {logits.max():.2f}], mean={logits.mean():.2f}")

print()
print("=" * 90)
print("Temperature scaling sweep — 预测 H743 修复效果")
print("=" * 90)
print(f'{"T":>5} {"thr":>6} {"TP":>5} {"FP":>5} {"Recall":>7} {"FPR":>6} {"Prec":>6} {"F1":>6}  vs H743 (T=1.0, thr=0.9): F1=85.59% FPR=6.34%')
n_att = np.sum(yw == 1)
n_norm = np.sum(yw == 0)
print()

# Try multiple temperatures and thresholds
for T in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    scaled_logits = logits / T
    probs_T = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -50, 50)))

    # Find best threshold for this T
    best_f1 = 0
    best_thr = 0.5
    best_metrics = None
    for thr100 in range(10, 100, 1):
        thr = thr100 / 100.0
        yp = (probs_T >= thr).astype(int)
        TP = ((yp == 1) & (yw == 1)).sum()
        FP = ((yp == 1) & (yw == 0)).sum()
        FN = ((yp == 0) & (yw == 1)).sum()
        TN = ((yp == 0) & (yw == 0)).sum()
        rec = TP / max(TP + FN, 1) * 100
        fpr = FP / max(FP + TN, 1) * 100
        prec = TP / max(TP + FP, 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = (TP, FP, FN, TN, rec, fpr, prec, f1)

    TP, FP, FN, TN, rec, fpr, prec, f1 = best_metrics
    print(f'{T:>5.1f} {best_thr:>6.3f} {TP:>5d} {FP:>5d} {rec:>6.2f}% {fpr:>5.2f}% {prec:>5.2f}% {f1:>5.2f}%   (best for this T)')

# Also: detailed distribution with T=2.0
print()
print("=" * 90)
print("Distribution shape: T=1 (current) vs T=2.0 (proposed)")
print("=" * 90)
buckets = [
    (0.000, 0.100), (0.100, 0.300), (0.300, 0.500),
    (0.500, 0.700), (0.700, 0.900), (0.900, 0.950),
    (0.950, 0.990), (0.990, 0.995), (0.995, 0.998), (0.998, 0.999), (0.999, 1.000),
]
print(f'{"range":>20} {"T=1 count":>10} {"T=2 count":>10}  delta')
probs_T1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
probs_T2 = 1.0 / (1.0 + np.exp(-np.clip(logits/2.0, -50, 50)))
for lo, hi in buckets:
    n1 = ((probs_T1 >= lo) & (probs_T1 < hi)).sum()
    n2 = ((probs_T2 >= lo) & (probs_T2 < hi)).sum()
    delta = n2 - n1
    sign = '+' if delta > 0 else ''
    print(f'[{lo:.3f}, {hi:.3f}): {n1:>10d} {n2:>10d}  {sign}{delta}')

# Threshold landscape at T=2.0 (more granular)
print()
print("=" * 90)
print("Threshold sweep at T=2.0 (predicted H743 with temperature scaling):")
print("=" * 90)
scaled_logits = logits / 2.0
probs_T2 = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -50, 50)))
print(f'{"thr":>6} {"TP":>5} {"FP":>5} {"Recall":>7} {"FPR":>6} {"Prec":>6} {"F1":>6}')
for thr in [0.30, 0.40, 0.45, 0.49, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
    yp = (probs_T2 >= thr).astype(int)
    TP = ((yp == 1) & (yw == 1)).sum()
    FP = ((yp == 1) & (yw == 0)).sum()
    FN = ((yp == 0) & (yw == 1)).sum()
    TN = ((yp == 0) & (yw == 0)).sum()
    rec = TP / max(TP + FN, 1) * 100
    fpr = FP / max(FP + TN, 1) * 100
    prec = TP / max(TP + FP, 1) * 100
    f1 = 2 * prec * rec / max(prec + rec, 1)
    print(f'{thr:>6.3f} {TP:>5d} {FP:>5d} {rec:>6.2f}% {fpr:>5.2f}% {prec:>5.2f}% {f1:>5.2f}%')
