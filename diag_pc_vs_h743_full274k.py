#!/usr/bin/env python3
"""
PC forward vs H743 完整 274k 数据对比.

关键: 跑 PC forward 在跟 H743 测试完全相同的 274k 数据上,
看 PC 的概率分布是否也有 [0.5, 0.9) 空白带 (这决定是 H743 drift 还是数据特性).

H743 test: logs/test_20261013-003345_auto-40ms.csv (17165 windows, 274k frames / 16)
PC test:   same data via preprocess_v2_scada.py pipeline + KEEP_23
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import collections

sys.path.insert(0, r"C:\work\Claude\Issue")
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEED = 42
WINDOW = 16

# 1. Load row-level preprocessed data (19-dim)
X_train = np.load("X_train_binary_v2_scada.npy")  # (192239, 19)
X_val   = np.load("X_val_binary_v2_scada.npy")     # (27462, 19)
X_test  = np.load("X_test_binary_v2_scada.npy")   # (54927, 19)

# Concatenate to get full 274k
X_full = np.concatenate([X_train, X_val, X_test], axis=0).astype(np.float32)
print(f"Full X shape: {X_full.shape}")

# Need binary labels too
y_train = np.load("y_train_binary_v2_scada.npy")
y_val   = np.load("y_val_binary_v2_scada.npy")
y_test  = np.load("y_test_binary_v2_scada.npy")
y_full  = np.concatenate([y_train, y_val, y_test], axis=0).astype(np.int64)
print(f"Full y shape: {y_full.shape}, normal={np.sum(y_full==0)}, attack={np.sum(y_full==1)}")

# 2. Add 8 per-window aggregates (broadcast to 16 timesteps) — replicate preprocess_v2_scada.py:154-208
n = (len(X_full) // WINDOW) * WINDOW
Xw = X_full[:n].reshape(n // WINDOW, WINDOW, -1)  # (17165, 16, 19)
yw = y_full[:n].reshape(n // WINDOW, WINDOW).max(axis=1).astype(np.int64)
print(f"Windowed: X={Xw.shape}, y={yw.shape}, attack_windows={np.sum(yw==1)}, normal_windows={np.sum(yw==0)}")

# Per-window aggregates
IDX_PRESS, IDX_CRC, IDX_ISRESP, IDX_LENGTH, IDX_ISUNUSUAL = 13, 14, 18, 2, 17
press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
crc_mean_w   = Xw[:, :, IDX_CRC].mean(axis=1)
crc_max_w    = Xw[:, :, IDX_CRC].max(axis=1)
cmd_count_w  = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)
resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
balance_w    = (cmd_count_w - resp_count_w) / WINDOW
length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(n // WINDOW)], dtype=np.float32)
unusual_count_w  = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)
aggs = np.column_stack([press_mean_w, crc_mean_w, crc_max_w, cmd_count_w, resp_count_w, balance_w, length_nunique_w, unusual_count_w]).astype(np.float32)
aggs_tiled = np.repeat(aggs[:, np.newaxis, :], WINDOW, axis=1)  # (17165, 16, 8)
Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)  # (17165, 16, 27)
print(f"With aggregates: {Xw_full.shape}")

# 3. KEEP_23 + transpose to (N, F, T) for PyTorch
Xw_23 = np.clip(Xw_full[:, :, KEEP_23], -10.0, 10.0).transpose(0, 2, 1)  # (17165, 23, 16)
print(f"KEEP_23 + transpose: {Xw_23.shape}")

# 4. PC forward (FP32, seed=42)
ckpt = torch.load(f"model_v4_se_23dim_b64_ch32_do01_window16_s{SEED}.pt", map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23)
model.load_state_dict(ckpt['state_dict'])
model.eval()

with torch.no_grad():
    # Process in batches to avoid OOM
    BATCH = 1024
    all_probs = []
    all_logits = []
    for i in range(0, len(Xw_23), BATCH):
        batch = torch.from_numpy(Xw_23[i:i+BATCH])
        logits = model(batch).numpy().squeeze()
        all_logits.append(logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        all_probs.append(probs)
logits = np.concatenate(all_logits)
probs = np.concatenate(all_probs)
print(f"PC logits range: [{logits.min():.3f}, {logits.max():.3f}], mean={logits.mean():.3f}, std={logits.std():.3f}")
print(f"PC probs range: [{probs.min():.6f}, {probs.max():.6f}]")

# 5. Compare distributions
print()
print("=" * 70)
print("PC forward on FULL 274k (17165 windows) — DIRECT COMPARE WITH H743")
print("=" * 70)
buckets = [
    (0.000, 0.001), (0.001, 0.100), (0.100, 0.300), (0.300, 0.500),
    (0.500, 0.700), (0.700, 0.900), (0.900, 0.950), (0.950, 0.990),
    (0.990, 0.995), (0.995, 0.998), (0.998, 0.999), (0.999, 1.000),
]
print(f'{"range":>20} {"PC_count":>9} {"PC_TP":>7} {"PC_FP":>7} {"PC_prec%":>9}  |  H743 (alarms only)')
h743_dist = {
    (0.001, 0.100): '?',  # H743 doesn't log non-alarm probs
    (0.100, 0.300): '?',
    (0.300, 0.500): '?',
    (0.500, 0.700): '0 alarms (flat landscape)',
    (0.700, 0.900): '299 alarms (4%) prec=50%',
    (0.900, 0.950): '?',
    (0.950, 0.990): '720 alarms prec=75%',
    (0.990, 0.995): '443 alarms prec=88%',
    (0.995, 0.998): '497 alarms prec=89%',
    (0.998, 0.999): '525 alarms prec=90%',
    (0.999, 1.000): '5681 alarms (75%) prec=99%',
}
for lo, hi in buckets:
    mask = (probs >= lo) & (probs < hi)
    n = mask.sum()
    n_att = ((mask) & (yw == 1)).sum()
    n_norm = ((mask) & (yw == 0)).sum()
    prec = n_att / max(n, 1) * 100
    h = h743_dist.get((lo, hi), '?')
    print(f'[{lo:.3f}, {hi:.3f}): {n:>9d} {n_att:>7d} {n_norm:>7d} {prec:>8.1f}%  |  {h}')

# 6. Threshold sweep
print()
print("=" * 70)
print("PC vs H743 threshold sweep on FULL 274k:")
print("=" * 70)
print(f'{"thr":>6} {"PC_TP":>6} {"PC_FP":>6} {"PC_R":>7} {"PC_FPR":>7} {"PC_F1":>7} | {"H743_TP":>7} {"H743_FP":>6} {"H743_R":>7} {"H743_FPR":>7} {"H743_F1":>7}')
# H743 at thr=0.9 (same as thr=0.5 due to flat): TP=6996, FP=529, Recall=79.30%, FPR=6.34%, F1=85.59%
# (8822 attack, 8343 normal)
for thr in [0.30, 0.40, 0.49, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]:
    yp = (probs >= thr).astype(int)
    TP = ((yp == 1) & (yw == 1)).sum()
    FP = ((yp == 1) & (yw == 0)).sum()
    FN = ((yp == 0) & (yw == 1)).sum()
    TN = ((yp == 0) & (yw == 0)).sum()
    rec = TP / max(TP + FN, 1) * 100
    fpr = FP / max(FP + TN, 1) * 100
    prec = TP / max(TP + FP, 1) * 100
    f1 = 2 * prec * rec / max(prec + rec, 1)
    # H743 values (constant across [0.5, 0.9])
    h_TP, h_FP, h_R, h_FPR, h_F1 = (6996, 529, 79.30, 6.34, 85.59) if thr < 0.9 else (
        {0.95: (6849, 377, 77.64, 4.52, 85.36),
         0.99: (6426, 153, 72.84, 1.83, 83.45)}.get(thr, ('?',)*5)
    )
    if thr == 0.50:
        h_TP, h_FP, h_R, h_FPR, h_F1 = 6996, 529, 79.30, 6.34, 85.59
    h_str = f'{h_TP:>7} {h_FP:>6} {h_R:>6.2f}% {h_FPR:>6.2f}% {h_F1:>6.2f}%' if isinstance(h_TP, int) else '   same as 0.5'
    print(f'{thr:>6.3f} {TP:>6d} {FP:>6d} {rec:>6.2f}% {fpr:>6.2f}% {f1:>6.2f}% | {h_str}')

# Save for later analysis
np.savez("pc_full274k_probs.npz", probs=probs, y=yw, logits=logits)
print()
print("Saved pc_full274k_probs.npz")
