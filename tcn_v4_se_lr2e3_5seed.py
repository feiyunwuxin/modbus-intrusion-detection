#!/usr/bin/env python3
"""
TCN+SE 19-dim SCADA — Verify LR=2e-3 with 5 seeds.

Goal: confirm that the LR=2e-3 result (single-seed F1m=0.8719) is NOT
a lucky RNG fluke, but a robust improvement over LR=5e-4 (default).

If LR=2e-3 is truly better:
  - All 5 seeds should give F1m > 0.85
  - Average should be ≥ 0.86
  - Variance should be ≤ LR=5e-4's variance

Compares against the previous 5-seed ensemble (LR=5e-4):
  - F1m = 0.8560 (avg), range [0.8450, 0.8607]
"""

import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)

BASE = r"C:\work\Claude\Issue"
TAG = "v4_se_lr2e3_5seed"
SEEDS = [42, 123, 456, 789, 1024]

# === Same as LR=2e-3 best from sweep, just vary seed ===
SEED         = 42   # default; will be overridden per seed
WINDOW       = 16
EPOCHS       = 40
BATCH        = 128
LR           = 2e-3     # ★ the discovered winner
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

# Output paths
META_OUT   = os.path.join(BASE, f"processed_meta_tcn_{TAG}.json")
EVAL_OUT   = os.path.join(BASE, f"evaluation_tcn_{TAG}.txt")
PRED_VAL   = os.path.join(BASE, f"predictions_val_tcn_{TAG}.csv")
PRED_TEST  = os.path.join(BASE, f"predictions_test_tcn_{TAG}.csv")
THR_OUT    = os.path.join(BASE, f"best_threshold_tcn_{TAG}.json")

t_start = time.time()
print(f"[config] BATCH={BATCH}  EPOCHS={EPOCHS}  LR={LR}  seeds={SEEDS}")
print(f"[expected time] ~{len(SEEDS) * 100 // 60} min ({len(SEEDS)} seeds × ~100s each)")

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("\n[loading data] 19-dim SCADA ...")
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


# ─────────────────────────────────────────────
def train_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_w), torch.from_numpy(y_train_w)),
                              batch_size=BATCH, shuffle=True, num_workers=0, generator=g)

    model = TCNClassifierSE(in_ch=N_FEATURES)
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
    t_seed = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
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

    train_time = time.time() - t_seed
    model.load_state_dict(best_state)
    val_prob = predict_probs(val_loader)
    test_prob = predict_probs(test_loader)
    return {
        "seed": seed,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "best_val_f1m": float(best_f1m),
        "best_epoch": int(best_epoch),
        "train_time": float(train_time),
    }


# ─────────────────────────────────────────────
all_results = []
for seed in SEEDS:
    print(f"\n{'='*60}\n  TRAINING seed={seed}\n{'='*60}")
    res = train_one_seed(seed)
    all_results.append(res)
    print(f"  seed={seed}: best_ep={res['best_epoch']}  best_val_f1m={res['best_val_f1m']:.4f}  "
          f"time={res['train_time']:.1f}s")

# Per-seed test metrics
for r in all_results:
    test_f1m_05 = f1_score(y_test_w, (r["test_prob"] >= 0.5).astype(int), average="macro")
    test_pr = average_precision_score(y_test_w, r["test_prob"])
    test_acc = ((r["test_prob"] >= 0.5).astype(int) == y_test_w).mean()
    r["test_f1m_05"] = float(test_f1m_05)
    r["test_pr_auc"] = float(test_pr)
    r["test_acc"] = float(test_acc)
    print(f"  seed={r['seed']}: test_f1m@0.5={test_f1m_05:.4f}  test_pr_auc={test_pr:.4f}")

# ─────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────
val_prob_ensemble  = np.mean([r["val_prob"]  for r in all_results], axis=0)
test_prob_ensemble = np.mean([r["test_prob"] for r in all_results], axis=0)

# Find best threshold on val
best_thr = 0.5; best_vf1m = 0
for t in np.arange(0.20, 0.80, 0.01):
    pred = (val_prob_ensemble >= t).astype(int)
    f1m = f1_score(y_val_w, pred, average="macro")
    if f1m > best_vf1m:
        best_vf1m = f1m; best_thr = float(t)

# Evaluate ensemble
val_05   = (val_prob_ensemble >= 0.5).astype(int)
test_05  = (test_prob_ensemble >= 0.5).astype(int)
val_thr  = (val_prob_ensemble >= best_thr).astype(int)
test_thr = (test_prob_ensemble >= best_thr).astype(int)

def compute_metrics(y, pred, prob):
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "binary_f1": float(f1_score(y, pred, average="binary")),
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }

ens_val_05 = compute_metrics(y_val_w, val_05, val_prob_ensemble)
ens_val_thr = compute_metrics(y_val_w, val_thr, val_prob_ensemble)
ens_test_05 = compute_metrics(y_test_w, test_05, test_prob_ensemble)
ens_test_thr = compute_metrics(y_test_w, test_thr, test_prob_ensemble)

# Save predictions
pd.DataFrame({"y_true": y_val_w, "prob_attack": val_prob_ensemble,
              "pred_05": val_05, "pred_tuned": val_thr}).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": y_test_w, "prob_attack": test_prob_ensemble,
              "pred_05": test_05, "pred_tuned": test_thr}).to_csv(PRED_TEST, index=False)

# Save meta
total_time = time.time() - t_start
meta = {
    "model": f"TCN {TAG}",
    "tag": TAG,
    "base": "v4_se_b128_e40",
    "ablation_type": "LR=2e-3 + 5-seed ensemble (B=128, ep=40)",
    "config": {"batch": BATCH, "epochs": EPOCHS, "lr": LR, "wd": WD,
               "dropout": DROPOUT, "channels": CHANNELS, "se_reduction": SE_REDUCTION},
    "seeds": SEEDS,
    "n_models": len(SEEDS),
    "total_time_seconds": float(total_time),
    "best_threshold": best_thr,
    "per_seed": [{k: v for k, v in r.items() if k not in ("val_prob", "test_prob")} for r in all_results],
    "ensemble_metrics": {
        "val_at_0.5": ens_val_05,
        "val_at_thr": ens_val_thr,
        "test_at_0.5": ens_test_05,
        "test_at_thr": ens_test_thr,
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print()
print("=" * 80)
print(f"  5-SEED ENSEMBLE @ LR=2e-3 (total time {total_time:.1f}s)")
print("=" * 80)
print(f"{'Seed':>6} {'BestValEp':>10} {'ValF1m':>10} {'TestF1m@0.5':>13} {'TestPR-AUC':>13} {'Time(s)':>9}")
print("-" * 80)
for r in all_results:
    print(f"{r['seed']:>6} {r['best_epoch']:>10} {r['best_val_f1m']:>10.4f} "
          f"{r['test_f1m_05']:>13.4f} {r['test_pr_auc']:>13.4f} {r['train_time']:>9.1f}")

# Aggregate stats
f1ms = [r["test_f1m_05"] for r in all_results]
prs = [r["test_pr_auc"] for r in all_results]
print("-" * 80)
print(f"{'AVG':>6} {'':>10} {np.mean([r['best_val_f1m'] for r in all_results]):>10.4f} "
      f"{np.mean(f1ms):>13.4f} {np.mean(prs):>13.4f}")
print(f"{'STD':>6} {'':>10} {np.std([r['best_val_f1m'] for r in all_results]):>10.4f} "
      f"{np.std(f1ms):>13.4f} {np.std(prs):>13.4f}")
print(f"{'MIN':>6} {'':>10} {np.min([r['best_val_f1m'] for r in all_results]):>10.4f} "
      f"{np.min(f1ms):>13.4f} {np.min(prs):>13.4f}")
print(f"{'MAX':>6} {'':>10} {np.max([r['best_val_f1m'] for r in all_results]):>10.4f} "
      f"{np.max(f1ms):>13.4f} {np.max(prs):>13.4f}")
print("=" * 80)

print()
print(f"  ENSEMBLE @ threshold {best_thr:.2f}:")
print(f"    Test F1m = {ens_test_thr['macro_f1']:.4f}   PR-AUC = {ens_test_thr['pr_auc']:.4f}")
print()
print(f"  COMPARISON:")
print(f"    LR=5e-4 5-seed avg F1m = 0.8531  ensemble F1m = 0.8560")
print(f"    LR=2e-3 5-seed avg F1m = {np.mean(f1ms):.4f}  ensemble F1m = {ens_test_thr['macro_f1']:.4f}")
print(f"    Δ avg    = {np.mean(f1ms) - 0.8531:+.4f}")
print(f"    Δ ens    = {ens_test_thr['macro_f1'] - 0.8560:+.4f}")
print(f"    Best single seed = {max(f1ms):.4f}  (seed={[r['seed'] for r in all_results if r['test_f1m_05']==max(f1ms)][0]})")