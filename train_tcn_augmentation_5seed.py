#!/usr/bin/env python3
"""TCN baseline + 4 data augmentation methods × 5 seeds.

Self-contained: reuses TCNClassifier from train_tcn_23dim_w16_5seed.py
and data loader from _common_train.py.

5 augmentation settings × 5 seeds = 25 model trainings:
  - "none":      baseline (no augmentation)
  - "noise":     Gaussian noise σ=0.05 × feature_std
  - "feat_mask": random 10% feature columns zeroed per sample
  - "time_mask": random 10% timesteps zeroed per sample
  - "combined":  noise → feat_mask → time_mask

Per-feature std precomputed from train set; val/test untouched.
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS
from train_tcn_23dim_w16_5seed import TCNClassifier

TAG = "tcn_augmentation"
SETTINGS = ["none", "noise", "feat_mask", "time_mask", "combined"]
NOISE_SIGMA = 0.05
MASK_RATE = 0.1
EPOCHS = 20
BATCH = 64
LR = 4e-3
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5


def add_gaussian_noise(x: torch.Tensor, feature_std: torch.Tensor, sigma: float):
    """x: (B, 23, 16). Per-feature std precomputed (23,). Adds σ·std·N(0,1)."""
    noise = torch.randn_like(x) * (sigma * feature_std).view(1, -1, 1)
    return x + noise


def feature_mask(x: torch.Tensor, mask_rate: float):
    """x: (B, 23, 16). Random mask_rate fraction of features (over all 23) → 0.
    A feature is masked for the whole window if selected (deterministic per sample).
    """
    B, C, T = x.shape
    n_mask = max(1, int(round(C * mask_rate)))
    mask_idx = torch.randint(0, C, (B, n_mask), device=x.device)
    out = x.clone()
    out.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, T), 0.0)
    return out


def time_mask(x: torch.Tensor, mask_rate: float):
    """x: (B, 23, 16). Random mask_rate fraction of timesteps (over 16) → 0.
    A timestep is masked across all features if selected.
    """
    B, C, T = x.shape
    n_mask = max(1, int(round(T * mask_rate)))
    mask_idx = torch.randint(0, T, (B, n_mask), device=x.device)
    out = x.clone()
    out.scatter_(2, mask_idx.unsqueeze(1).expand(-1, C, -1), 0.0)
    return out


def apply_augmentation(x: torch.Tensor, method: str, feature_std: torch.Tensor):
    """Dispatcher. Order for 'combined': noise → feat_mask → time_mask."""
    if method == "none":
        return x
    if method in ("noise", "combined"):
        x = add_gaussian_noise(x, feature_std, NOISE_SIGMA)
    if method in ("feat_mask", "combined"):
        x = feature_mask(x, MASK_RATE)
    if method in ("time_mask", "combined"):
        x = time_mask(x, MASK_RATE)
    return x


def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one_seed_aug(seed: int, method: str, feature_std: torch.Tensor):
    """Train one seed with one augmentation setting. Save prediction CSV."""
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data_23dim_w16()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                            batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                             batch_size=BATCH, shuffle=False, num_workers=0)

    model = TCNClassifier()
    n_params = sum(p.numel() for p in model.parameters())

    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=2)

    best_f1m, best_state, best_epoch = -1.0, None, -1
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            if method != "none":
                xb = apply_augmentation(xb, method, feature_std)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        val_prob, val_lbl = predict_probs(model, val_loader)
        val_f1m = f1_score(val_lbl, (val_prob >= 0.5).astype(int), average="macro")
        scheduler.step(val_f1m)

        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    train_time = time.time() - t0

    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)

    pred_path = os.path.join(BASE_PATH, f"predictions_test_tcn_aug_{method}_s{seed}.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack"])
        for y, p in zip(test_lbl.tolist(), test_prob.tolist()):
            w.writerow([y, p])

    metrics = {
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float(accuracy_score(test_lbl, test_pred)),
        "test_precision": float(precision_score(test_lbl, test_pred, zero_division=0)),
        "test_recall": float(recall_score(test_lbl, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
    }

    print(f"  method={method:9s}  seed={seed}: "
          f"F1={metrics['test_binary_f1']:.4f}  "
          f"best_ep={best_epoch}  time={train_time:.1f}s")

    return {
        "method": method, "seed": int(seed),
        "n_params": n_params, "best_epoch": int(best_epoch),
        "train_time_s": float(train_time), **metrics,
    }


def main():
    print(f"=== TCN + 4 Augmentation Methods × 5 seeds ===\n"
          f"  Noise σ={NOISE_SIGMA}×feature_std, mask_rate={MASK_RATE}\n")

    # Precompute per-feature std from train data
    X_tr, _, _, _, _, _ = load_data_23dim_w16()
    feature_std = torch.from_numpy(X_tr.std(axis=(0, 2)).astype(np.float32))

    results = []
    for method in SETTINGS:
        print(f"\n--- method={method} ---")
        for seed in SEEDS:
            r = train_one_seed_aug(seed, method, feature_std)
            results.append(r)

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="TCN + 4 Augmentation Settings", tag=TAG,
                   config={"epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
                           "noise_sigma": NOISE_SIGMA, "mask_rate": MASK_RATE,
                           "settings": SETTINGS, "n_settings": len(SETTINGS),
                           "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()