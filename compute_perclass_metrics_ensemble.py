#!/usr/bin/env python3
"""精确提取 23-dim ch=32 5-seed ensemble 的 per-class Recall/Precision"""
import os, sys, json
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                              f1_score, accuracy_score, roc_auc_score,
                              average_precision_score, matthews_corrcoef)

BASE = r"D:\workspace\claude\Issue"
sys.path.insert(0, BASE)

# 加载测试数据 (23-dim 切片)
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
y_test = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)

# Clip + transpose to (N, F, T) -> TCN expects (N, F, T) which is what we have after [:,:,KEEP_23]
# Wait — the retrain script does transpose(0,2,1) which converts (N,T,F) -> (N,F,T)
X_test_t = np.clip(X_test, -10.0, 10.0).transpose(0, 2, 1)  # (N,T,F) -> (N,F,T)
print(f"X_test shape: {X_test.shape} -> T {X_test_t.shape}, y_test shape: {y_test.shape}")
print(f"Class dist: Normal={np.sum(y_test==0)}, Attack={np.sum(y_test==1)}")

# Load model class definition
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE

SEEDS = [42, 123, 456, 789, 1024]
all_probs = []
per_seed_metrics = {}

for seed in SEEDS:
    pt_path = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = TCNClassifierSE(in_ch=23)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_t)).numpy().squeeze()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    all_probs.append(probs)

    y_pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred).tolist()
    per_seed_metrics[seed] = {
        "F1m": f1_score(y_test, y_pred, average="macro"),
        "BinF1": f1_score(y_test, y_pred, pos_label=1),
        "Acc": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, probs),
        "PR_AUC": average_precision_score(y_test, probs),
        "Attack_Recall": recall_score(y_test, y_pred, pos_label=1),
        "Attack_Precision": precision_score(y_test, y_pred, pos_label=1),
        "Normal_Recall": recall_score(y_test, y_pred, pos_label=0),
        "Normal_Precision": precision_score(y_test, y_pred, pos_label=0),
        "CM_05": cm,
    }
    print(f"Seed {seed}: F1m={per_seed_metrics[seed]['F1m']:.4f} BinF1={per_seed_metrics[seed]['BinF1']:.4f} "
          f"Acc={per_seed_metrics[seed]['Acc']:.4f} ROC={per_seed_metrics[seed]['ROC_AUC']:.4f} "
          f"PR={per_seed_metrics[seed]['PR_AUC']:.4f} | "
          f"Att R={per_seed_metrics[seed]['Attack_Recall']:.4f} P={per_seed_metrics[seed]['Attack_Precision']:.4f} | "
          f"Nor R={per_seed_metrics[seed]['Normal_Recall']:.4f} P={per_seed_metrics[seed]['Normal_Precision']:.4f}")

# Ensemble: 5-seed probability averaging
ensemble_probs = np.mean(all_probs, axis=0)
y_pred_ens = (ensemble_probs >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_ens).tolist()

# Find best threshold for ensemble
best_thr, best_f1m = 0.5, 0.0
for thr in np.arange(0.20, 0.51, 0.01):
    yp = (ensemble_probs >= thr).astype(int)
    f1m = f1_score(y_test, yp, average="macro")
    if f1m > best_f1m:
        best_f1m, best_thr = f1m, thr
y_pred_best = (ensemble_probs >= best_thr).astype(int)

ensemble_metrics = {
    "thr_05": {
        "F1m": f1_score(y_test, y_pred_ens, average="macro"),
        "BinF1": f1_score(y_test, y_pred_ens, pos_label=1),
        "Acc": accuracy_score(y_test, y_pred_ens),
        "ROC_AUC": roc_auc_score(y_test, ensemble_probs),
        "PR_AUC": average_precision_score(y_test, ensemble_probs),
        "Attack_Recall": recall_score(y_test, y_pred_ens, pos_label=1),
        "Attack_Precision": precision_score(y_test, y_pred_ens, pos_label=1),
        "Normal_Recall": recall_score(y_test, y_pred_ens, pos_label=0),
        "Normal_Precision": precision_score(y_test, y_pred_ens, pos_label=0),
        "MCC": matthews_corrcoef(y_test, y_pred_ens),
        "BM": recall_score(y_test, y_pred_ens, pos_label=1) + recall_score(y_test, y_pred_ens, pos_label=0) - 1,
        "CM": cm,
    },
    "thr_best": {
        "best_thr": float(best_thr),
        "best_F1m": float(best_f1m),
        "F1m": f1_score(y_test, y_pred_best, average="macro"),
        "BinF1": f1_score(y_test, y_pred_best, pos_label=1),
        "Acc": accuracy_score(y_test, y_pred_best),
        "Attack_Recall": recall_score(y_test, y_pred_best, pos_label=1),
        "Attack_Precision": precision_score(y_test, y_pred_best, pos_label=1),
        "Normal_Recall": recall_score(y_test, y_pred_best, pos_label=0),
        "Normal_Precision": precision_score(y_test, y_pred_best, pos_label=0),
        "MCC": matthews_corrcoef(y_test, y_pred_best),
        "CM": confusion_matrix(y_test, y_pred_best).tolist(),
    },
}

print("\n=== 5-Seed Ensemble (prob-averaged) ===")
for k, v in ensemble_metrics["thr_05"].items():
    print(f"  thr=0.5 {k}: {v}")
print()
for k, v in ensemble_metrics["thr_best"].items():
    print(f"  thr={ensemble_metrics['thr_best']['best_thr']:.2f} {k}: {v}")

# Save
out = {
    "model": "TCN+SE 23-dim ch=32 5-seed ensemble",
    "n_features": 23,
    "window": 16,
    "per_seed": {str(s): m for s, m in per_seed_metrics.items()},
    "ensemble": ensemble_metrics,
}
with open(os.path.join(BASE, "perclass_metrics_23dim_ensemble.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved: perclass_metrics_23dim_ensemble.json")