#!/usr/bin/env python3
"""TCN baseline with Snapshot Ensemble: cosine annealing warm restarts.

Strategy:
  - Same TCN architecture + hyperparams as baseline
  - Replace ReduceLROnPlateau with CosineAnnealingWarmRestarts(T_0=4)
  - Save model snapshot at end of each cycle (epochs 4, 8, 12, 16, 20)
  - 5 snapshots per seed → per-seed 5-snapshot avg → 5-seed ensemble
  - Compare to baseline (1 model per seed × 5 seeds)

Hypothesis: cyclical LR forces the optimizer to explore different local minima;
averaging snapshots gives a cheap ensemble effect.

~5 min on CPU.
"""
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS
from train_tcn_23dim_w16_5seed import TCNClassifier

EPOCHS = 20
BATCH = 64
LR = 4e-3
WD = 1e-5
GRAD_CLIP = 0.5
SNAPSHOT_EPOCHS = [4, 8, 12, 16, 20]  # 5 snapshots per run
TAG_PRED = "tcn_snap"


def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one_seed_snap(seed: int):
    from sklearn.metrics import f1_score

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
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1)

    snapshots = {}  # epoch -> state_dict
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()  # cosine warm restart

        if epoch in SNAPSHOT_EPOCHS:
            snapshots[epoch] = {k: v.clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    # Per-snapshot predictions
    per_snap_probs = {}
    for ep, sd in snapshots.items():
        model.load_state_dict(sd)
        p, _ = predict_probs(model, test_loader)
        per_snap_probs[ep] = p
        f1_at_05 = f1_score(y_te, (p >= 0.5).astype(int))
        print(f"  seed={seed}  snap_ep={ep:>2}  F1@0.5={f1_at_05:.4f}")

    # Snapshot ensemble per seed: average of all snapshot probs
    snap_arr = np.stack(list(per_snap_probs.values()), axis=0)
    p_snap_ens = snap_arr.mean(axis=0)
    f1_snap_ens = f1_score(y_te, (p_snap_ens >= 0.5).astype(int))
    print(f"  seed={seed}  snap_ens(5) F1@0.5={f1_snap_ens:.4f}  time={train_time:.1f}s")

    # Save per-snapshot AND per-seed-snap-ensemble predictions
    for ep, p in per_snap_probs.items():
        path = os.path.join(BASE_PATH, f"predictions_test_{TAG_PRED}_s{seed}_ep{ep}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true", "prob_attack"])
            for y, pp in zip(y_te.tolist(), p.tolist()):
                w.writerow([y, pp])
    path = os.path.join(BASE_PATH, f"predictions_test_{TAG_PRED}_s{seed}_ens.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack"])
        for y, pp in zip(y_te.tolist(), p_snap_ens.tolist()):
            w.writerow([y, pp])

    return {"seed": int(seed), "n_params": n_params,
            "train_time_s": float(train_time),
            "snap_epochs": SNAPSHOT_EPOCHS,
            "snap_ens_f1_at_0.5": float(f1_snap_ens)}


def main():
    print(f"=== TCN Snapshot Ensemble (cosine warm restarts) × 5 seeds ===\n"
          f"  Snapshots at epochs: {SNAPSHOT_EPOCHS}\n")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_snap(seed))

    save_meta_json(results, os.path.join(BASE_PATH, "processed_meta_tcn_snapshot.json"),
                   model_name="TCN baseline + Snapshot Ensemble (cosine warm restarts)",
                   tag="tcn_snap",
                   config={"snapshots_epochs": SNAPSHOT_EPOCHS, "epochs": EPOCHS,
                           "batch": BATCH, "lr": LR, "wd": WD, "grad_clip": GRAD_CLIP,
                           "scheduler": "CosineAnnealingWarmRestarts(T_0=4)"})
    print(f"\n[saved meta] processed_meta_tcn_snapshot.json")


if __name__ == "__main__":
    main()
