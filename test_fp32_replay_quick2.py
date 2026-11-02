"""Run trained TCN+SE model on first 918 windows of training set, compare with H743 result.

Use ytr_w (training labels) for ground truth, same as H743 test.
"""
import sys
import csv
import numpy as np
import torch

sys.path.insert(0, '.')
import train_quantize_3ch as tq

# Load model
ckpt = torch.load('model_v4_se_23dim_b64_ch32_do01_window16_s42.pt', map_location='cpu', weights_only=False)
drop_features = ckpt['config']['drop_features']
print(f"drop_features: {drop_features}")

model = tq.TCNClassifierSE(in_ch=23, channels=32, n_blocks=3, dropout=0.1)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print("Model loaded")

# Load 27-dim windowed TRAINING data
Xtr_w = np.load('X_train_binary_v2_scada_window16.npy')
ytr_w = np.load('y_train_binary_v2_scada_window16.npy')

drop_idx = sorted(drop_features.keys())
keep_idx = [i for i in range(27) if i not in drop_idx]

n_train_win = 918
X = Xtr_w[:n_train_win, ...][..., keep_idx]  # (918, 16, 23)
y = ytr_w[:n_train_win]
print(f"X: {X.shape}, y sum: {y.sum()}")

# Run model
def run_model(X_w, model, batch_size=64, clip_val=10.0):
    with torch.no_grad():
        if clip_val:
            X_w = np.clip(X_w, -clip_val, clip_val)
        Xt = torch.from_numpy(X_w).float().transpose(1, 2)
        n = Xt.shape[0]
        probs = []
        for i in range(0, n, batch_size):
            batch = Xt[i:i+batch_size]
            out = model(batch)
            probs.append(torch.sigmoid(out).numpy().flatten())
        return np.concatenate(probs)

probs = run_model(X, model)
print(f"probs: range [{probs.min():.4f}, {probs.max():.4f}]")

# Load H743 test results
h743_alarm_w = set()
with open('logs/test_20261018-223848_auto-40ms.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        if row['event'] == 'ALARM':
            W = (int(row['frame_idx']) - 1) // 16
            h743_alarm_w.add(W)
print(f"H743 alarm windows: {len(h743_alarm_w)}")

# PC sim predictions at thr=0.5
pc_pred = set(i for i in range(n_train_win) if probs[i] >= 0.5)
print(f"PC sim predictions (thr=0.5): {len(pc_pred)}")

# H743 predictions
h743_pred = h743_alarm_w
print(f"H743 predictions: {len(h743_pred)}")

# Common windows
common = h743_pred & pc_pred
print(f"\\nCommon (both predict attack): {len(common)}")
print(f"Only H743 (H743 alarm, PC sim normal): {len(h743_pred - pc_pred)}")
print(f"Only PC sim (PC sim attack, H743 silent): {len(pc_pred - h743_pred)}")

# H743 vs PC sim probs for H743-only windows
h743_only = sorted(h743_pred - pc_pred)
pc_only = sorted(pc_pred - h743_pred)
print(f"\\nH743-only windows (H743 thinks attack, PC sim thinks normal):")
print(f"  count: {len(h743_only)}")
for w in h743_only[:10]:
    print(f"  W={w}: H743=fired, PC sim prob={probs[w]:.4f}, y={y[w]}")

print(f"\\nPC-only windows (PC sim thinks attack, H743 silent):")
print(f"  count: {len(pc_only)}")
for w in pc_only[:10]:
    print(f"  W={w}: H743=silent, PC sim prob={probs[w]:.4f}, y={y[w]}")

# Compare using both H743 and PC sim ground truth (should be same)
win_atk = set(i for i in range(n_train_win) if y[i] == 1)
print(f"\\nGround truth attack windows: {len(win_atk)}")

# Metrics using GT
for name, pred in [('H743', h743_pred), ('PC sim', pc_pred)]:
    tp = len(pred & win_atk)
    fp = len(pred - win_atk)
    fn = len(win_atk - pred)
    tn = n_train_win - tp - fp - fn
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    print(f"  {name}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Recall={rec:.4f}, FPR={fpr:.4f}, F1={f1:.4f}")

# Show prob distribution
print(f"\\nProb distribution:")
for thr in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    n_pred = (probs >= thr).sum()
    tp = ((probs >= thr) & (y == 1)).sum()
    fp = ((probs >= thr) & (y == 0)).sum()
    fn = ((probs < thr) & (y == 1)).sum()
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    print(f"  thr={thr}: {n_pred} alarms, TP={tp}, FP={fp}, FN={fn}, F1={f1:.4f}")
