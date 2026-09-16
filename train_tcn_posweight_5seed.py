#!/usr/bin/env python3
"""TCN baseline with custom pos_weight (e.g., 2.0 to bias toward Attack class).

Current baseline uses pos_weight = n_neg/n_pos = 1.026 (essentially balanced).
This experiment tries pos_weight=2.0 to push more probability mass toward Attack,
which should boost Recall (the current bottleneck at thr=0.390).

Hypothesis: pos_weight=2.0 → at thr=0.5 should give similar F1 to TCN @ thr=0.380.

5 seeds × 1 setting. ~3 min.
"""
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common_train import (
    load_data_23dim_w16, predict_probs, save_predictions_csv,
    save_meta_json, BASE_PATH, SEEDS,
)
from train_tcn_23dim_w16_5seed import TCNClassifier

EPOCHS = 20
BATCH = 64
LR = 4e-3
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5
POS_WEIGHT = 2.0
TAG_PRED = "tcn_pw2"


def train_one_seed_pw(seed: int):
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

    pw = torch.tensor([POS_WEIGHT], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=PATIENCE)

    best_f1m, best_state, best_epoch = -1.0, None, -1
    epochs_no_improve = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
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

    pred_path = os.path.join(BASE_PATH, f"predictions_test_{TAG_PRED}_s{seed}.csv")
    save_predictions_csv(test_lbl, test_prob, pred_path)

    metrics = {
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float(accuracy_score(test_lbl, test_pred)),
        "test_precision": float(precision_score(test_lbl, test_pred, zero_division=0)),
        "test_recall": float(recall_score(test_lbl, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
    }

    print(f"  seed={seed}: F1={metrics['test_binary_f1']:.4f}  "
          f"R={metrics['test_recall']:.4f}  P={metrics['test_precision']:.4f}  "
          f"best_ep={best_epoch}  time={train_time:.1f}s")
    return {"seed": int(seed), "n_params": n_params, "best_epoch": int(best_epoch),
            "train_time_s": float(train_time), **metrics}


def main():
    print(f"=== TCN baseline + pos_weight={POS_WEIGHT} × 5 seeds ===\n")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_pw(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_tcn_posweight{POS_WEIGHT}.json"),
                   model_name=f"TCN baseline + pos_weight={POS_WEIGHT}", tag=f"tcn_pw{int(POS_WEIGHT)}",
                   config={"pos_weight": POS_WEIGHT, "epochs": EPOCHS, "batch": BATCH,
                           "lr": LR, "wd": WD, "patience": PATIENCE, "grad_clip": GRAD_CLIP})
    print(f"\n[saved] processed_meta_tcn_posweight{POS_WEIGHT}.json")


if __name__ == "__main__":
    main()