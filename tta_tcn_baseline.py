#!/usr/bin/env python3
"""Test-Time Augmentation (TTA) on TCN baseline.

Train-time augmentation hurt F1 (sub-project #3). But TTA is different:
only applied at INFERENCE (no learning distortion). Idea: add mild Gaussian
noise to test inputs, run inference N times, average predictions → smoother
decision boundary.

Setup:
  - Load 5 saved TCN baseline models (model_tcn_23dim_w16_s{seed}.pt)
  - For each model, run inference on:
      1. Original test
      2. N=10 augmented versions (Gaussian noise σ=0.02×feature_std)
      3. Average all 11 predictions
  - 5-seed prob_mean ensemble of (TTA-averaged per-seed predictions)

Compare:
  - TCN baseline (no TTA): F1=0.8795
  - TCN + TTA: F1=?
  - Stacking M3@0.390: F1=0.8904 (ceiling)
"""
import os
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from _common_train import load_data_23dim_w16, BASE_PATH, SEEDS
from train_tcn_23dim_w16_5seed import TCNClassifier

DEVICE = "cpu"
N_TTA = 10
TTA_SIGMA = 0.02  # 1/2 of train-time augmentation's sigma (which was 0.05)


def predict_with_tta(model, X_test, feature_std, n_tta=N_TTA, sigma=TTA_SIGMA, batch=64):
    """Run inference on original X_test + N augmented versions, return averaged probs."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(n_tta + 1):
            if i == 0:
                X_aug = X_test
            else:
                noise = np.random.randn(*X_test.shape).astype(np.float32) * (sigma * feature_std).reshape(1, -1, 1)
                X_aug = X_test + noise
            loader = DataLoader(torch.from_numpy(X_aug), batch_size=batch, shuffle=False)
            probs = []
            for xb in loader:
                probs.append(torch.sigmoid(model(xb)).numpy())
            all_probs.append(np.concatenate(probs))
    return np.mean(all_probs, axis=0)


def predict_no_tta(model, X_test, batch=64):
    """Standard inference on original X_test (no augmentation)."""
    model.eval()
    with torch.no_grad():
        loader = DataLoader(torch.from_numpy(X_test), batch_size=batch, shuffle=False)
        probs = []
        for xb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(probs)


def four_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def sweep_threshold(p_oof, y):
    from threshold_tuning_stacking import THR_RANGE, four_metrics as fm2
    rows = []
    for t in THR_RANGE:
        y_pred = (p_oof >= t).astype(int)
        rows.append({"threshold": float(t), **fm2(y, y_pred)})
    rows_sorted = sorted(rows, key=lambda r: -r["F1-Score"])
    return rows_sorted[0], next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9)


def main():
    print("=== TTA Experiment on TCN Baseline ===\n")
    print(f"  N_TTA={N_TTA}, σ={TTA_SIGMA}×feature_std\n")

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data_23dim_w16()
    feature_std = X_tr.std(axis=(0, 2)).astype(np.float32)

    print(f"Loaded data: X_test={X_te.shape}, y_test={y_te.shape}\n")

    # Load 5 models + per-seed predictions
    per_seed_no_tta = {}
    per_seed_tta = {}
    for seed in SEEDS:
        ckpt_path = os.path.join(BASE_PATH, f"model_tcn_23dim_w16_s{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: {ckpt_path} not found, skipping seed {seed}")
            continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model = TCNClassifier()
        model.load_state_dict(sd)

        np.random.seed(seed + 10000)  # ensure TTA noise is reproducible
        p_no_tta = predict_no_tta(model, X_te)
        p_tta = predict_with_tta(model, X_te, feature_std)

        per_seed_no_tta[seed] = p_no_tta
        per_seed_tta[seed] = p_tta

        f1_no = f1_score(y_te, (p_no_tta >= 0.5).astype(int))
        f1_tta = f1_score(y_te, (p_tta >= 0.5).astype(int))
        print(f"  seed={seed}: F1_no_tta={f1_no:.4f}  F1_tta={f1_tta:.4f}  Δ={f1_tta-f1_no:+.4f}")

    if not per_seed_no_tta:
        print("ERROR: no models loaded")
        return

    # 5-seed prob_mean ensemble
    seeds = sorted(per_seed_no_tta.keys())
    p_ens_no_tta = np.mean([per_seed_no_tta[s] for s in seeds], axis=0)
    p_ens_tta = np.mean([per_seed_tta[s] for s in seeds], axis=0)

    # Sweep threshold on each
    best_no, cur_no = sweep_threshold(p_ens_no_tta, y_te)
    best_tta, cur_tta = sweep_threshold(p_ens_tta, y_te)

    print("\n" + "=" * 70)
    print(f"{'Config':<40s} {'Thr':>6s} {'Acc':>8s} {'P':>8s} {'R':>8s} {'F1':>8s}")
    print("-" * 70)
    print(f"{'TCN baseline (no TTA) @ best thr':<40s} {best_no['threshold']:>6.3f} "
          f"{best_no['Accuracy']:>8.4f} {best_no['Precision']:>8.4f} "
          f"{best_no['Recall']:>8.4f} {best_no['F1-Score']:>8.4f}")
    print(f"{'TCN + TTA @ best thr':<40s} {best_tta['threshold']:>6.3f} "
          f"{best_tta['Accuracy']:>8.4f} {best_tta['Precision']:>8.4f} "
          f"{best_tta['Recall']:>8.4f} {best_tta['F1-Score']:>8.4f}")
    print(f"{'TCN baseline (no TTA) @ thr=0.5':<40s} {0.5:>6.3f} "
          f"{cur_no['Accuracy']:>8.4f} {cur_no['Precision']:>8.4f} "
          f"{cur_no['Recall']:>8.4f} {cur_no['F1-Score']:>8.4f}")
    print(f"{'TCN + TTA @ thr=0.5':<40s} {0.5:>6.3f} "
          f"{cur_tta['Accuracy']:>8.4f} {cur_tta['Precision']:>8.4f} "
          f"{cur_tta['Recall']:>8.4f} {cur_tta['F1-Score']:>8.4f}")

    delta_pp = (best_tta["F1-Score"] - best_no["F1-Score"]) * 100
    print(f"\nDelta (TTA vs no-TTA, both at best thr): {delta_pp:+.3f}pp")
    print(f"\nReference: Stacking M3@0.390 → F1=0.8904")
    print(f"           TCN + TTA best thr → F1={best_tta['F1-Score']:.4f}")
    delta_stack = (best_tta["F1-Score"] - 0.8904) * 100
    print(f"           Delta vs stacking: {delta_stack:+.3f}pp")

    if delta_pp > 0.005:
        print("\n>>> TTA HELPS TCN baseline <<<")
    elif delta_pp < -0.005:
        print("\n>>> TTA HURTS TCN baseline <<<")
    else:
        print("\n>>> TTA NEUTRAL on TCN baseline <<<")

    # Save outputs
    out_dir = BASE_PATH
    save_path = os.path.join(out_dir, "tta_tcn_baseline.json")
    save_data = {
        "task": "TTA on TCN baseline (N=10 augmented + 1 original, σ=0.02×feature_std)",
        "n_tta": N_TTA, "tta_sigma": TTA_SIGMA,
        "per_seed_f1_no_tta": {str(s): float(f1_score(y_te, (per_seed_no_tta[s] >= 0.5).astype(int)))
                              for s in seeds},
        "per_seed_f1_tta": {str(s): float(f1_score(y_te, (per_seed_tta[s] >= 0.5).astype(int)))
                            for s in seeds},
        "ensemble_no_tta": {"best_threshold": best_no["threshold"],
                            "best_metrics": best_no,
                            "thr_0.5_metrics": cur_no},
        "ensemble_tta": {"best_threshold": best_tta["threshold"],
                         "best_metrics": best_tta,
                         "thr_0.5_metrics": cur_tta},
        "delta_f1_pp": delta_pp,
        "reference": {"stacking_m3_thr_0.390_f1": 0.8904,
                      "tcn_baseline_thr_0.5_f1": 0.8795},
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {save_path}")


if __name__ == "__main__":
    main()