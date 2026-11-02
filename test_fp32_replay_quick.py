"""Run trained TCN+SE model on first 918 windows of training set, compare with H743 result.

The model expects 23-dim input. Training data has 27-dim (19 base + 8 aggregations).
drop_features = {2, 3, 20, 22} drops 4 to get 23.
"""
import sys
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

# Load 27-dim windowed data
Xte_w = np.load('X_test_binary_v2_scada_window16.npy')
yte_w = np.load('y_test_binary_v2_scada_window16.npy')
print(f"Xte_w shape: {Xte_w.shape}, yte_w sum: {yte_w.sum()}")

# Drop the 4 features to get 23-dim
drop_idx = sorted(drop_features.keys())
keep_idx = [i for i in range(27) if i not in drop_idx]
print(f"  drop_idx={drop_idx}, keep_idx={keep_idx}")

Xte_23 = Xte_w[..., keep_idx]
print(f"  Xte_23 shape: {Xte_23.shape}")

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

probs_te = run_model(Xte_23, model)
preds_te = (probs_te >= 0.5).astype(int)
tp = ((preds_te == 1) & (yte_w == 1)).sum()
fp = ((preds_te == 1) & (yte_w == 0)).sum()
fn = ((preds_te == 0) & (yte_w == 1)).sum()
tn = ((preds_te == 0) & (yte_w == 0)).sum()
print(f"\nTest set (X_test, PC sim, on trained model):")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Recall={tp/(tp+fn):.4f}, FPR={fp/(fp+tn):.4f}, F1={2*tp/(2*tp+fp+fn):.4f}")
print(f"  prob range: [{probs_te.min():.4f}, {probs_te.max():.4f}]")

# First 918 windows of training set
Xtr_w = np.load('X_train_binary_v2_scada_window16.npy')
ytr_w = np.load('y_train_binary_v2_scada_window16.npy')
print(f"\nXtr_w shape: {Xtr_w.shape}, ytr_w sum: {ytr_w.sum()}")

n_train_win = 918
Xtr_918 = Xtr_w[:n_train_win, ...][..., keep_idx]
ytr_918 = ytr_w[:n_train_win]
print(f"  First 918 windows: ytr_918 sum: {ytr_918.sum()}")

probs_tr = run_model(Xtr_918, model)
preds_tr = (probs_tr >= 0.5).astype(int)
tp = ((preds_tr == 1) & (ytr_918 == 1)).sum()
fp = ((preds_tr == 1) & (ytr_918 == 0)).sum()
fn = ((preds_tr == 0) & (ytr_918 == 1)).sum()
tn = ((preds_tr == 0) & (ytr_918 == 0)).sum()
print(f"\nFirst 918 windows of TRAINING set (PC sim):")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Recall={tp/(tp+fn):.4f}, FPR={fp/(fp+tn):.4f}, F1={2*tp/(2*tp+fp+fn):.4f}")
print(f"  prob range: [{probs_tr.min():.4f}, {probs_tr.max():.4f}]")
print(f"  prob distribution:")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    n_alarm = (probs_tr >= thr).sum()
    print(f"    thr={thr}: {n_alarm} alarms, {n_alarm/(tp+fn)*100:.1f}% of attack windows detected")

print()
print("=" * 60)
print("H743 test result for comparison (from logs/test_20261018-223848):")
print(f"  918 windows, TP=298, FP=0, FN=175, TN=445")
print(f"  Recall=0.6300, FPR=0.0000, F1=0.7730")
print("=" * 60)
