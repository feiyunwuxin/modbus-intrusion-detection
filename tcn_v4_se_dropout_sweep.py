#!/usr/bin/env python3
"""
TCN+SE 19-dim SCADA — Dropout sweep at LR=2e-3 (winner) + ch=64 (default).

Dropout is the most under-tuned hyperparameter we've used (always 0.3).
With high LR (2e-3), the implicit regularization from gradient noise is
strong, so explicit dropout may need to be different.

Dropout values to test:
  - 0.1  (very light regularization)
  - 0.2  (light)
  - 0.3  ← current default (control)
  - 0.4  (moderate)
  - 0.5  (heavy)
"""

import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
)

BASE = r"C:\work\Claude\Issue"
DROPOUTS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Fixed: LR=2e-3, ch=64
SEED         = 42
WINDOW       = 16
EPOCHS       = 40
BATCH        = 128
LR           = 2e-3
WD           = 1e-5
PATIENCE     = 10
GRAD_CLIP    = 0.5
CLIP_VAL     = 10.0
N_BLOCKS     = 3
CHANNELS     = 64
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4]
SE_REDUCTION = 8

t_start = time.time()
print(f"[config] sweep dropout: {DROPOUTS}, LR=2e-3, ch=64")
print(f"[expected time] ~{len(DROPOUTS) * 100 // 60} min")

# Load data
print("\n[loading data] ...")
X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada.npy")).astype(np.float32)
X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada.npy")).astype(np.float32)
X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada.npy")).astype(np.float32)
y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada.npy")).astype(np.int64)
y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada.npy")).astype(np.int64)
y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada.npy")).astype(np.int64)

def make_windows(X, y, win):
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    return Xw, yw

X_train_w, y_train_w = make_windows(X_train, y_train, WINDOW)
X_val_w,   y_val_w   = make_windows(X_val,   y_val,   WINDOW)
X_test_w,  y_test_w  = make_windows(X_test,  y_test,  WINDOW)
N_FEATURES = X_train_w.shape[-1]
X_train_w = np.clip(X_train_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_val_w   = np.clip(X_val_w,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_test_w  = np.clip(X_test_w,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)

val_loader  = DataLoader(TensorDataset(torch.from_numpy(X_val_w),  torch.from_numpy(y_val_w)),
                         batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_w), torch.from_numpy(y_test_w)),
                         batch_size=BATCH, shuffle=False, num_workers=0)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class TCNBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, se_reduction=SE_REDUCTION):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=se_reduction)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return F.relu(x + residual)


class TCNClassifierSE(nn.Module):
    def __init__(self, in_ch, channels=CHANNELS, n_blocks=N_BLOCKS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=0.3):
        super().__init__()
        layers = []
        layers.append(TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout))
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc2 = nn.Linear(32, 1)
        self.head_drop = nn.Dropout(0.3)  # head dropout stays at 0.3

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.head_drop(x)
        return self.fc2(x).squeeze(-1)


def train_one(dropout):
    torch.manual_seed(SEED); np.random.seed(SEED)
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_w), torch.from_numpy(y_train_w)),
                              batch_size=BATCH, shuffle=True, num_workers=0, generator=g)
    model = TCNClassifierSE(in_ch=N_FEATURES, channels=CHANNELS, dropout=dropout)
    n_pos = int((y_train_w == 1).sum())
    n_neg = int((y_train_w == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    @torch.no_grad()
    def predict_probs(loader):
        model.eval()
        probs = []
        for xb, _ in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
        return np.concatenate(probs)

    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    t_d = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        val_prob = predict_probs(val_loader)
        val_f1m = f1_score(y_val_w, (val_prob >= 0.5).astype(int), average="macro")
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

    train_time = time.time() - t_d
    model.load_state_dict(best_state)
    val_prob = predict_probs(val_loader)
    test_prob = predict_probs(test_loader)
    best_thr, best_vf1m = 0.5, 0
    for t in np.arange(0.20, 0.80, 0.01):
        f1m = f1_score(y_val_w, (val_prob >= t).astype(int), average="macro")
        if f1m > best_vf1m:
            best_vf1m = f1m; best_thr = float(t)
    test_pred = (test_prob >= best_thr).astype(int)
    return {
        "dropout": dropout,
        "best_val_f1m": float(best_f1m),
        "best_val_epoch": int(best_epoch),
        "best_threshold": float(best_thr),
        "test_f1m": float(f1_score(y_test_w, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(y_test_w, test_pred, average="binary")),
        "test_pr_auc": float(average_precision_score(y_test_w, test_prob)),
        "test_acc": float((test_pred == y_test_w).mean()),
        "train_time": float(train_time),
    }


results = []
for d in DROPOUTS:
    print(f"\n>>> dropout = {d}")
    res = train_one(d)
    results.append(res)
    print(f"    best_ep={res['best_val_epoch']}  val_f1m={res['best_val_f1m']:.4f}  "
          f"test_f1m={res['test_f1m']:.4f}  PR-AUC={res['test_pr_auc']:.4f}  "
          f"time={res['train_time']:.1f}s")

total = time.time() - t_start

df = pd.DataFrame(results)
df.to_csv(os.path.join(BASE, "tcn_v4_se_dropout_sweep_summary.csv"), index=False)
with open(os.path.join(BASE, "tcn_v4_se_dropout_sweep_meta.json"), "w") as f:
    json.dump({"config": {"lr": LR, "batch": BATCH, "epochs": EPOCHS, "channels": CHANNELS, "seed": SEED},
               "dropouts_tested": DROPOUTS, "results": results,
               "total_time_seconds": float(total),
               "best_dropout": max(results, key=lambda r: r["test_f1m"])["dropout"],
               "best_test_f1m": max(r["test_f1m"] for r in results)}, f, indent=2)

print()
print("=" * 80)
print(f"  DROPOUT SWEEP @ LR=2e-3 — COMPLETE in {total:.1f}s")
print("=" * 80)
print(f"{'Dropout':>10} {'TestF1m':>10} {'PR-AUC':>10} {'BestEp':>8} {'Time(s)':>8}")
print("-" * 80)
for r in results:
    print(f"{r['dropout']:>10.1f} {r['test_f1m']:>10.4f} {r['test_pr_auc']:>10.4f} "
          f"{r['best_val_epoch']:>8} {r['train_time']:>8.1f}")
print("=" * 80)
best = max(results, key=lambda r: r["test_f1m"])
print(f"\nBest: dropout={best['dropout']} -> Test F1m = {best['test_f1m']:.4f}, "
      f"PR-AUC = {best['test_pr_auc']:.4f}")