#!/usr/bin/env python3
"""Full test set evaluation to check if model is broken or my subset is unrepresentative."""
import sys, time
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"
sys.path.insert(0, BASE)
from retrain_tcn_23dim_b64_ch32_do01_savept import (
    TCNClassifierSE, SEEDS, N_FEATURES, WINDOW, KEEP_23, CLIP_VAL,
)

# Load test data (channels-last, NOT transposed)
X = np.load(f"{BASE}/X_test_binary_v2_scada_window16.npy").astype(np.float32)  # (3432, 16, 27)
y = np.load(f"{BASE}/y_test_binary_v2_scada_window16.npy").astype(np.int64)
X23 = X[:, :, KEEP_23]  # (3432, 16, 23)
print(f"X23 shape: {X23.shape}, y_test: y0={sum(y==0)}, y1={sum(y==1)}")

# Transpose to channels-first (batch, 23, 16) like training script does
X23_cf = np.clip(X23, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)  # (3432, 23, 16)
print(f"X23_cf (channels-first) shape: {X23_cf.shape}")

# Load all 5 models
models = []
for s in SEEDS:
    path = f"{BASE}/model_v4_se_23dim_b64_ch32_do01_window16_s{s}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = TCNClassifierSE(in_ch=N_FEATURES)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    models.append((s, m))

# Inference per seed
print("\n=== Per-seed Full test set eval ===")
print(f"{'seed':<6} | {'Acc':<8} | {'P':<8} | {'R':<8} | {'F1':<8} | {'PR-AUC':<8} | "
      f"{'nrm_prob':<14} | {'atk_prob':<14}")
print("-" * 96)
y_probs_all = []
for s, m in models:
    with torch.no_grad():
        x = torch.from_numpy(X23_cf)
        logits = m(x).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_probs_all.append(probs)
    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    p = precision_score(y, y_pred, zero_division=0)
    r = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred)
    pr_auc = average_precision_score(y, probs)
    nrm_mean = probs[y == 0].mean()
    atk_mean = probs[y == 1].mean()
    print(f"{s:<6} | {acc:.4f}  | {p:.4f}  | {r:.4f}  | {f1:.4f}  | {pr_auc:.4f}  | "
          f"{nrm_mean:.4f}+/-{probs[y==0].std():.4f} | {atk_mean:.4f}+/-{probs[y==1].std():.4f}")

# 5-seed ensemble
print("\n=== 5-seed ENSEMBLE ===")
ens_probs = np.mean(y_probs_all, axis=0)  # (3432,)
y_pred_ens = (ens_probs >= 0.5).astype(int)
acc = accuracy_score(y, y_pred_ens)
p = precision_score(y, y_pred_ens, zero_division=0)
r = recall_score(y, y_pred_ens, zero_division=0)
f1 = f1_score(y, y_pred_ens)
pr_auc = average_precision_score(y, ens_probs)
nrm_mean = ens_probs[y == 0].mean()
atk_mean = ens_probs[y == 1].mean()
print(f"5-ENS | Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f} PR-AUC={pr_auc:.4f}")
print(f"       nrm={nrm_mean:.4f}+/-{ens_probs[y==0].std():.4f}, atk={atk_mean:.4f}+/-{ens_probs[y==1].std():.4f}")

# Probability histogram
print("\n=== Probability distribution ===")
print(f"nrm: min={ens_probs[y==0].min():.4f} q25={np.percentile(ens_probs[y==0], 25):.4f} "
      f"med={np.median(ens_probs[y==0]):.4f} q75={np.percentile(ens_probs[y==0], 75):.4f} max={ens_probs[y==0].max():.4f}")
print(f"atk: min={ens_probs[y==1].min():.4f} q25={np.percentile(ens_probs[y==1], 25):.4f} "
      f"med={np.median(ens_probs[y==1]):.4f} q75={np.percentile(ens_probs[y==1], 75):.4f} max={ens_probs[y==1].max():.4f}")

# Show what's wrong: count at-thr samples
print(f"\nnrm prob >= 0.5: {(ens_probs[y==0] >= 0.5).sum()}/{sum(y==0)} = "
      f"{(ens_probs[y==0] >= 0.5).mean()*100:.1f}% (false positive rate)")
print(f"atk prob <  0.5: {(ens_probs[y==1] <  0.5).sum()}/{sum(y==1)} = "
      f"{(ens_probs[y==1] <  0.5).mean()*100:.1f}% (false negative rate)")
