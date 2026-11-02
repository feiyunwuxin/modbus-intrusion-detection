#!/usr/bin/env python3
"""
TCN hyperparameter sweep on the 17-dim Phase 2 feature set + window=8.
Reuses the data pipeline from tcn_binary_v2_window8.py (17-dim + 28 one-hot function
= 44 dim/step). Architecture: TCN v2 (3 blocks, dilations=[1,2,4], kernel=3).

Usage:
    python train_tcn_p2_sweep.py --seed 42 --batch 512 --channels 64 --dropout 0.3
    python train_tcn_p2_sweep.py --seed 42 --batch 512 --channels 64 --dropout 0.3 --use_se

Sweep dims (default values match TCN v2):
    - channels  ∈ {16, 32, 64}     (default 64)
    - dropout   ∈ {0.1, 0.2, 0.3}  (default 0.3)
    - batch     ∈ {64, 128, 256}   (default 512 for sanity check, 64/128/256 for sweep)

Fixed: window=8, lr=5e-4, epochs=30, n_blocks=3, dilations=[1,2,4], kernel=3.
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)

BASE = r"C:\work\Claude\Issue"

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--batch",       type=int, default=512)
    p.add_argument("--channels",    type=int, default=64)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--patience",    type=int, default=6)
    p.add_argument("--use_se",      action="store_true", help="add Squeeze-Excitation block")
    p.add_argument("--se_reduction",type=int, default=8)
    p.add_argument("--window",      type=int, default=8)
    p.add_argument("--clip_val",    type=float, default=10.0)
    p.add_argument("--grad_clip",   type=float, default=0.5)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--save_preds",  action="store_true", default=True)
    p.add_argument("--out_dir",     type=str, default=BASE)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Data pipeline (17-dim + window=8 + one-hot function)
# ────────────────────────────────────────────────────────────────────────────
def load_data(window):
    X_tr = np.load(os.path.join(BASE, "X_train_binary.npy")).astype(np.float32)
    X_va = np.load(os.path.join(BASE, "X_val_binary.npy")).astype(np.float32)
    X_te = np.load(os.path.join(BASE, "X_test_binary.npy")).astype(np.float32)
    y_tr = np.load(os.path.join(BASE, "y_train_binary.npy")).astype(np.int64)
    y_va = np.load(os.path.join(BASE, "y_val_binary.npy")).astype(np.int64)
    y_te = np.load(os.path.join(BASE, "y_test_binary.npy")).astype(np.int64)

    def make_windows(X, y):
        n = (len(X) // window) * window
        Xw = X[:n].reshape(n // window, window, -1)
        yw = (y[:n].reshape(n // window, window).max(axis=1)).astype(np.int64)
        return Xw, yw

    Xtr_w, ytr_w = make_windows(X_tr, y_tr)
    Xva_w, yva_w = make_windows(X_va, y_va)
    Xte_w, yte_w = make_windows(X_te, y_te)

    # one-hot function at each timestep
    fn_tr = Xtr_w[:, :, 1].astype(np.int64).ravel()
    fn_va = Xva_w[:, :, 1].astype(np.int64).ravel()
    fn_te = Xte_w[:, :, 1].astype(np.int64).ravel()
    all_fn = np.unique(np.concatenate([fn_tr, fn_va, fn_te]))
    fn_map = {v: i for i, v in enumerate(all_fn)}
    n_cats = len(all_fn)

    def onehot_window(fn_codes, n):
        out = np.zeros((len(fn_codes), n), dtype=np.float32)
        out[np.arange(len(fn_codes)), fn_codes] = 1.0
        return out

    oh_tr = onehot_window(np.vectorize(fn_map.get)(fn_tr), n_cats).reshape(Xtr_w.shape[0], window, n_cats)
    oh_va = onehot_window(np.vectorize(fn_map.get)(fn_va), n_cats).reshape(Xva_w.shape[0], window, n_cats)
    oh_te = onehot_window(np.vectorize(fn_map.get)(fn_te), n_cats).reshape(Xte_w.shape[0], window, n_cats)

    keep_idx = [0] + list(range(2, 17))  # drop col 1 (function), keep 16 numeric
    Xtr = np.concatenate([Xtr_w[:, :, keep_idx], oh_tr], axis=-1)
    Xva = np.concatenate([Xva_w[:, :, keep_idx], oh_va], axis=-1)
    Xte = np.concatenate([Xte_w[:, :, keep_idx], oh_te], axis=-1)

    Xtr = np.clip(Xtr, -10.0, 10.0).transpose(0, 2, 1)
    Xva = np.clip(Xva, -10.0, 10.0).transpose(0, 2, 1)
    Xte = np.clip(Xte, -10.0, 10.0).transpose(0, 2, 1)
    n_features = Xtr.shape[1]
    return Xtr, ytr_w, Xva, yva_w, Xte, yte_w, n_features, n_cats, all_fn.tolist()


# ────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
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


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, use_se=False, se_reduction=8):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch, reduction=se_reduction) if use_se else None

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        if self.se is not None:
            x = self.se(x)
        return F.relu(x + residual)


class TCNClassifier(nn.Module):
    def __init__(self, in_ch, n_blocks=3, channels=64, kernel_size=3,
                 dilations=(1, 2, 4), dropout=0.3, use_se=False, se_reduction=8):
        super().__init__()
        layers = [TCNBlock(in_ch, channels, kernel_size, dilations[0], dropout, use_se, se_reduction)]
        for d in dilations[1:]:
            layers.append(TCNBlock(channels, channels, kernel_size, d, dropout, use_se, se_reduction))
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


# ────────────────────────────────────────────────────────────────────────────
# Single run
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        logits = model(xb)
        probs.append(torch.sigmoid(logits).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_one(args):
    """Train one TCN config, return metrics dict."""
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # 1) data
    Xtr, ytr, Xva, yva, Xte, yte, n_feat, n_cats, fn_values = load_data(args.window)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                              batch_size=args.batch * 2, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                              batch_size=args.batch * 2, shuffle=False, num_workers=0)

    # 2) model
    model = TCNClassifier(in_ch=n_feat, n_blocks=3, channels=args.channels,
                          kernel_size=3, dilations=(1, 2, 4), dropout=args.dropout,
                          use_se=args.use_se, se_reduction=args.se_reduction)
    n_params = sum(p.numel() for p in model.parameters())

    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    # 3) train
    t0 = time.time()
    best_f1m, best_state, best_epoch, no_imp = -1, None, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = loss_sum / n_seen

        val_prob, val_lbl = predict_probs(model, val_loader)
        val_f1m = f1_score(val_lbl, (val_prob >= 0.5).astype(int), average="macro")
        scheduler.step(val_f1m)
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # 4) eval
    val_prob, val_lbl = predict_probs(model, val_loader)
    test_prob, test_lbl = predict_probs(model, test_loader)

    val_pred_05 = (val_prob >= 0.5).astype(int)
    test_pred_05 = (test_prob >= 0.5).astype(int)

    metrics = {
        "n_params":       n_params,
        "best_epoch":     best_epoch,
        "train_time_s":   round(train_time, 1),
        "best_val_f1m":   float(best_f1m),
        "val_acc":        float(accuracy_score(val_lbl, val_pred_05)),
        "val_f1m":        float(f1_score(val_lbl, val_pred_05, average="macro")),
        "val_bin_f1":     float(f1_score(val_lbl, val_pred_05, average="binary")),
        "val_roc_auc":    float(roc_auc_score(val_lbl, val_prob)),
        "val_pr_auc":     float(average_precision_score(val_lbl, test_prob if False else val_prob)),
        "test_acc":       float(accuracy_score(test_lbl, test_pred_05)),
        "test_prec":      float(precision_score(test_lbl, test_pred_05)),
        "test_rec":       float(recall_score(test_lbl, test_pred_05)),
        "test_f1":        float(f1_score(test_lbl, test_pred_05, average="binary")),
        "test_f1m":       float(f1_score(test_lbl, test_pred_05, average="macro")),
        "test_roc_auc":   float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc":    float(average_precision_score(test_lbl, test_prob)),
        "test_cm":        confusion_matrix(test_lbl, test_pred_05).tolist(),
    }

    # 5) save artifacts
    if args.save_preds:
        pd.DataFrame({"y_true": test_lbl, "prob_attack": test_prob, "pred_05": test_pred_05}).to_csv(
            os.path.join(args.out_dir, f"predictions_test_tcn_p2_sweep_s{args.seed}.csv"), index=False)

    return metrics, test_prob, test_lbl, model


def main():
    args = parse_args()
    print(f"[run] seed={args.seed} batch={args.batch} channels={args.channels} "
          f"dropout={args.dropout} use_se={args.use_se} lr={args.lr} epochs={args.epochs}",
          flush=True)

    metrics, _, _, _ = run_one(args)

    print()
    print(f"=== TCN p2 sweep result ===")
    print(f"  n_params      = {metrics['n_params']:,}")
    print(f"  best_epoch    = {metrics['best_epoch']}")
    print(f"  train_time_s  = {metrics['train_time_s']}")
    print(f"  best_val_f1m  = {metrics['best_val_f1m']:.4f}")
    print(f"  --- 4 Binary metrics (test @ thr=0.5) ---")
    print(f"  Accuracy      = {metrics['test_acc']:.4f}")
    print(f"  Precision     = {metrics['test_prec']:.4f}")
    print(f"  Recall        = {metrics['test_rec']:.4f}")
    print(f"  F1-Score      = {metrics['test_f1']:.4f}")
    print(f"  Macro-F1      = {metrics['test_f1m']:.4f}")
    print(f"  PR-AUC        = {metrics['test_pr_auc']:.4f}")
    print(f"  ROC-AUC       = {metrics['test_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
