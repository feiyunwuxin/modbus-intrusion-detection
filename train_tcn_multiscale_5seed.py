#!/usr/bin/env python3
"""Multi-Scale TCN: 3 blocks × parallel kernels=[1,3,5,7] × dilations=[1,2,4].

Each block: 4 parallel Conv1d branches → Concat → 1x1 Conv reduce → residual add.

5 seeds. Same hyperparams as TCN baseline (lr=4e-3, wd=1e-5, EPOCHS=20, BATCH=64).
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS

TAG = "tcn_multiscale"
TAG_PRED = "tcn_ms"
KERNELS = (1, 3, 5, 7)
DILATIONS = (1, 2, 4)
N_BLOCKS = 3
CH = 32
DROPOUT = 0.1
EPOCHS = 20
BATCH = 64
LR = 4e-3
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5


class MultiScaleTCNBlock(nn.Module):
    """Parallel Conv1d branches with kernels=[1,3,5,7] → concat → 1x1 conv → residual."""
    def __init__(self, ch: int, dilation: int, kernels=KERNELS, dropout=DROPOUT):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            pad = (k - 1) * dilation // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.reduce = nn.Sequential(
            nn.Conv1d(ch * len(kernels), ch, kernel_size=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        outs = []
        target_len = x.shape[-1]
        for branch in self.branches:
            o = branch(x)
            if o.shape[-1] > target_len:
                o = o[..., :target_len]
            elif o.shape[-1] < target_len:
                pad = target_len - o.shape[-1]
                o = nn.functional.pad(o, (0, pad))
            outs.append(o)
        cat = torch.cat(outs, dim=1)
        out = self.reduce(cat)
        return x + out


class MultiScaleTCN(nn.Module):
    def __init__(self, n_features=23, window=16):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, CH, kernel_size=1)
        self.blocks = nn.Sequential(*[
            MultiScaleTCNBlock(CH, dilation=d) for d in DILATIONS
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(CH, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one_seed_ms(seed: int):
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

    model = MultiScaleTCN()
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

    print(f"  seed={seed}: F1={metrics['test_binary_f1']:.4f}  "
          f"best_ep={best_epoch}  time={train_time:.1f}s  params={n_params}")
    return {
        "seed": int(seed), "n_params": n_params, "best_epoch": int(best_epoch),
        "train_time_s": float(train_time), **metrics,
    }


def main():
    print(f"=== Multi-Scale TCN × 5 seeds ===\n"
          f"  Kernels={KERNELS}  Dilations={DILATIONS}  Ch={CH}  Blocks={N_BLOCKS}\n")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_ms(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="Multi-Scale TCN (kernels=1,3,5,7)", tag=TAG,
                   config={"kernels": list(KERNELS), "dilations": list(DILATIONS),
                           "ch": CH, "n_blocks": N_BLOCKS, "dropout": DROPOUT,
                           "epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
                           "patience": PATIENCE, "grad_clip": GRAD_CLIP,
                           "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()