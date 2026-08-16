#!/usr/bin/env python3
"""
TCN+SE 19-dim SCADA — Learning Rate sweep on B=128 + ep=40 base.

Hypothesis: with smaller batch (B=128 vs original B=512), the optimal LR
may differ from the default 5e-4. Smaller batch = more updates = larger
effective LR, so we may need to LOWER LR.

LR values to test:
  - 1e-4   (very conservative)
  - 3e-4   (slightly lower than current)
  - 5e-4   ← current default (control)
  - 1e-3   (2× current)
  - 2e-3   (4× current, may diverge)

Each LR is run with the same seed=42 for fair comparison.
"""

import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report

BASE = r"C:\work\Claude\Issue"
TAG_PREFIX = "v4_se_b128_lr"
LRS = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]

# Hyperparameters (B=128 base, vary LR)
SEED         = 42
WINDOW       = 16
EPOCHS       = 40
BATCH        = 128
WD           = 1e-5
PATIENCE     = 10
GRAD_CLIP    = 0.5
DROPOUT      = 0.3
CLIP_VAL     = 10.0
N_BLOCKS     = 3
CHANNELS     = 64
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4]
SE_REDUCTION = 8

t_start = time.time()

# ─────────────────────────────────────────────
# Load data once (shared across all LRs)
# ─────────────────────────────────────────────
print(f"[config] sweep LRs: {LRS}, total expected ~{len(LRS) * 2} min")
print("[loading data] 19-dim SCADA ...")
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

# Fixed val/test loaders
val_loader  = DataLoader(TensorDataset(torch.from_numpy(X_val_w),  torch.from_numpy(y_val_w)),
                         batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_w), torch.from_numpy(y_test_w)),
                         batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────
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
    def __init__(self, in_ch, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = []
        layers.append(TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout))
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc2(x).squeeze(-1)


def train_one_lr(lr_value):
    """Train one model with the given LR and return metrics."""
    # Reset RNG
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_w), torch.from_numpy(y_train_w)),
                              batch_size=BATCH, shuffle=True, num_workers=0, generator=g)

    model = TCNClassifierSE(in_ch=N_FEATURES)
    n_params = sum(p.numel() for p in model.parameters())

    n_pos = int((y_train_w == 1).sum())
    n_neg = int((y_train_w == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_value, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    @torch.no_grad()
    def predict_probs(loader):
        model.eval()
        probs = []
        for xb, _ in loader:
            logits = model(xb)
            probs.append(torch.sigmoid(logits).numpy())
        return np.concatenate(probs)

    best_f1m, best_state, best_epoch = -1, None, -1
    history = []
    epochs_no_improve = 0
    t_lr = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_loss_sum / train_n

        val_prob = predict_probs(val_loader)
        val_pred_05 = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(y_val_w, val_pred_05, average="macro")
        val_auc = roc_auc_score(y_val_w, val_prob)
        history.append((epoch, train_loss, val_f1m, val_auc))

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

    train_time = time.time() - t_lr
    model.load_state_dict(best_state)
    val_prob = predict_probs(val_loader)
    test_prob = predict_probs(test_loader)

    # Find best threshold on val
    best_thr, best_vf1m = 0.5, 0
    for t in np.arange(0.30, 0.70, 0.01):
        pred = (val_prob >= t).astype(int)
        f1m = f1_score(y_val_w, pred, average="macro")
        if f1m > best_vf1m:
            best_vf1m = f1m
            best_thr = float(t)

    test_pred = (test_prob >= best_thr).astype(int)
    test_f1m = f1_score(y_test_w, test_pred, average="macro")
    test_binary_f1 = f1_score(y_test_w, test_pred, average="binary")
    test_pr_auc = average_precision_score(y_test_w, test_prob)
    test_acc = (test_pred == y_test_w).mean()

    return {
        "lr": lr_value,
        "best_val_f1m": float(best_f1m),
        "best_val_epoch": int(best_epoch),
        "best_threshold": float(best_thr),
        "test_f1m": float(test_f1m),
        "test_binary_f1": float(test_binary_f1),
        "test_pr_auc": float(test_pr_auc),
        "test_acc": float(test_acc),
        "train_time": float(train_time),
        "history": [{"epoch": h[0], "loss": h[1], "val_f1m": h[2], "val_auc": h[3]} for h in history],
    }


# ─────────────────────────────────────────────
# Sweep all LRs
# ─────────────────────────────────────────────
results = []
print()
print("=" * 70)
print(f"  LR SWEEP — B=128, ep=40, 5 values")
print("=" * 70)
for lr in LRS:
    print(f"\n>>> Training LR={lr:.0e}")
    res = train_one_lr(lr)
    results.append(res)
    print(f"    best_val_ep={res['best_val_epoch']}  best_val_f1m={res['best_val_f1m']:.4f}  "
          f"test_f1m={res['test_f1m']:.4f}  test_pr_auc={res['test_pr_auc']:.4f}  "
          f"thr={res['best_threshold']:.2f}  time={res['train_time']:.1f}s")

total_time = time.time() - t_start

# ─────────────────────────────────────────────
# Save summary
# ─────────────────────────────────────────────
summary_df = pd.DataFrame([
    {
        "lr": r["lr"],
        "best_val_f1m": r["best_val_f1m"],
        "best_val_epoch": r["best_val_epoch"],
        "best_threshold": r["best_threshold"],
        "test_f1m": r["test_f1m"],
        "test_binary_f1": r["test_binary_f1"],
        "test_pr_auc": r["test_pr_auc"],
        "test_acc": r["test_acc"],
        "train_time_s": r["train_time"],
    }
    for r in results
])
summary_df.to_csv(os.path.join(BASE, "tcn_v4_se_lr_sweep_summary.csv"), index=False)

# Save full meta
meta = {
    "experiment": "LR sweep on B=128 base",
    "base_config": "TCN+SE 19-dim SCADA, B=128, ep=40, seed=42",
    "lrs_tested": LRS,
    "best_lr": max(results, key=lambda r: r["test_f1m"])["lr"],
    "best_test_f1m": max(r["test_f1m"] for r in results),
    "best_test_pr_auc": max(r["test_pr_auc"] for r in results),
    "per_lr": results,
    "total_time_seconds": float(total_time),
}
with open(os.path.join(BASE, "tcn_v4_se_lr_sweep_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# Final summary
print()
print("=" * 80)
print(f"  LR SWEEP COMPLETE in {total_time:.1f}s")
print("=" * 80)
print(f"{'LR':>10} {'BestValF1m':>12} {'BestEp':>8} {'TestF1m':>10} {'BinF1':>10} {'PR-AUC':>10} {'Thr':>6} {'Time(s)':>8}")
print("-" * 80)
for r in results:
    print(f"{r['lr']:>10.0e} {r['best_val_f1m']:>12.4f} {r['best_val_epoch']:>8} "
          f"{r['test_f1m']:>10.4f} {r['test_binary_f1']:>10.4f} {r['test_pr_auc']:>10.4f} "
          f"{r['best_threshold']:>6.2f} {r['train_time']:>8.1f}")
print("=" * 80)
best = max(results, key=lambda r: r["test_f1m"])
print(f"\nBest LR: {best['lr']:.0e}  →  Test F1m = {best['test_f1m']:.4f}  "
      f"PR-AUC = {best['test_pr_auc']:.4f}  (vs default LR=5e-4: ΔF1m = "
      f"{best['test_f1m'] - next(r['test_f1m'] for r in results if r['lr']==5e-4):+.4f})")