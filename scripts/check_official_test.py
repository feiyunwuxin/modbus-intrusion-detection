"""V9: Load ACTUAL pre-built test windows from disk and run model."""
import os, sys
sys.path.insert(0, r'C:\work\Claude\Issue')
import numpy as np
import torch
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE

BASE = r'C:\work\Claude\Issue'
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]

X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
y_test = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy"))
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'X_test dtype: {X_test.dtype}, y_test dtype: {y_test.dtype}')
print(f'y_test unique: {np.unique(y_test, return_counts=True)}')

X_23 = X_test[:, :, KEEP_23]
X_23 = np.clip(X_23, -10.0, 10.0)

ckpt = torch.load(os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"),
                  map_location="cpu", weights_only=False)
print(f'\nCheckpoint keys: {list(ckpt.keys())}')
if 'state_dict' in ckpt:
    print(f'State dict keys (first 5): {list(ckpt["state_dict"].keys())[:5]}')

model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt["state_dict"])
model.eval()

X_t = torch.from_numpy(X_23.transpose(0, 2, 1).astype(np.float32))
with torch.no_grad():
    logits = model(X_t).numpy().squeeze()
probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

print(f'\n=== ON OFFICIAL TEST SET (KEEP_23 applied) ===')
print(f'Total windows: {len(probs)}')
print(f'Attack labels: {y_test.sum()}/{len(y_test)}')
print(f'Predicted attack: {(probs >= 0.5).sum()}/{len(probs)}')

from sklearn.metrics import confusion_matrix, f1_score
preds = (probs >= 0.5).astype(int)
print(f'Confusion matrix:')
print(confusion_matrix(y_test, preds))
print(f'F1: {f1_score(y_test, preds, average="binary"):.4f}')

# Try threshold 0.49
preds_49 = (probs >= 0.49).astype(int)
print(f'At thr=0.49: F1={f1_score(y_test, preds_49, average="binary"):.4f}')