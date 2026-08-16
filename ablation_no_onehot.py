#!/usr/bin/env python3
"""
17 维 v1 raw (无 one-hot 整数编码) TCN+SE 实验 — 验证距离幻觉.

输入: 17 维 v1 raw 数据 (function 是整数 1 列, in_ch=17)
对比: 17 dim + one-hot (44 dim) / 19 dim / 27 dim
"""

import os, time, json, sys
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
EPOCHS = 20
SEED = 42
BATCH = 512
LR = 5e-4
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5
DROPOUT = 0.3
CLIP_VAL = 10.0
N_BLOCKS = 3
CHANNELS = 64
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
SE_REDUCTION = 8
WINDOW = 16


def load_17v1_raw():
    """17 维 v1 raw, function 整数编码, in_ch=17."""
    X_train = np.load(os.path.join(BASE, "X_train_binary.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(BASE, "X_val_binary.npy")).astype(np.float32)
    X_test  = np.load(os.path.join(BASE, "X_test_binary.npy")).astype(np.float32)
    y_train = np.load(os.path.join(BASE, "y_train_binary.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary.npy")).astype(np.int64)

    def make_windows(X, y, win):
        n = (len(X) // win) * win
        return (X[:n].reshape(n // win, win, -1),
                (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64))

    X_train_w, y_train_w = make_windows(X_train, y_train, WINDOW)
    X_val_w,   y_val_w   = make_windows(X_val,   y_val,   WINDOW)
    X_test_w,  y_test_w  = make_windows(X_test,  y_test,  WINDOW)

    X_train_w = np.clip(X_train_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val_w   = np.clip(X_val_w,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test_w  = np.clip(X_test_w,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    return X_train_w, X_val_w, X_test_w, y_train_w, y_val_w, y_test_w


def load_v2(n_features):
    X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)
    X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
    y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
    X_train = np.clip(X_train[:, :, :n_features], -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val   = np.clip(X_val[:,   :, :n_features],   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test  = np.clip(X_test[:,  :, :n_features],  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_17v1_with_onehot():
    """17 维 + one-hot (44 dim/步)."""
    X_train = np.load(os.path.join(BASE, "X_train_binary.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(BASE, "X_val_binary.npy")).astype(np.float32)
    X_test  = np.load(os.path.join(BASE, "X_test_binary.npy")).astype(np.float32)
    y_train = np.load(os.path.join(BASE, "y_train_binary.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary.npy")).astype(np.int64)

    def make_windows(X, y, win):
        n = (len(X) // win) * win
        return (X[:n].reshape(n // win, win, -1),
                (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64))

    X_train_w, y_train_w = make_windows(X_train, y_train, WINDOW)
    X_val_w,   y_val_w   = make_windows(X_val,   y_val,   WINDOW)
    X_test_w,  y_test_w  = make_windows(X_test,  y_test,  WINDOW)

    fn_train = X_train_w[:, :, 1].astype(np.int64).ravel()
    fn_val   = X_val_w[:,   :, 1].astype(np.int64).ravel()
    fn_test  = X_test_w[:,  :, 1].astype(np.int64).ravel()
    all_fn   = np.unique(np.concatenate([fn_train, fn_val, fn_test]))
    fn_map   = {v: i for i, v in enumerate(all_fn)}
    n_cats   = len(all_fn)

    def onehot_window(fn_codes, n):
        out = np.zeros((len(fn_codes), n), dtype=np.float32)
        out[np.arange(len(fn_codes)), fn_codes] = 1.0
        return out

    oh_tr = onehot_window(np.vectorize(fn_map.get)(fn_train), n_cats).reshape(X_train_w.shape[0], WINDOW, n_cats)
    oh_va = onehot_window(np.vectorize(fn_map.get)(fn_val),   n_cats).reshape(X_val_w.shape[0],   WINDOW, n_cats)
    oh_te = onehot_window(np.vectorize(fn_map.get)(fn_test),  n_cats).reshape(X_test_w.shape[0],  WINDOW, n_cats)

    keep_idx = [0] + list(range(2, 17))
    num_train = X_train_w[:, :, keep_idx]
    num_val   = X_val_w[:,   :, keep_idx]
    num_test  = X_test_w[:,  :, keep_idx]
    X_train_cnn = np.concatenate([num_train, oh_tr], axis=-1)
    X_val_cnn   = np.concatenate([num_val,   oh_va], axis=-1)
    X_test_cnn  = np.concatenate([num_test,  oh_te], axis=-1)
    X_train_cnn = np.clip(X_train_cnn, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val_cnn   = np.clip(X_val_cnn,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test_cnn  = np.clip(X_test_cnn,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    return X_train_cnn, X_val_cnn, X_test_cnn, y_train_w, y_val_w, y_test_w


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
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
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


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        logits = model(xb)
        probs.append(torch.sigmoid(logits).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_one(name, X_train, X_val, X_test, y_train, y_val, y_test, results):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"  >> {name}  (in_ch={X_train.shape[1]})")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=BATCH, shuffle=False, num_workers=0)

    model = TCNClassifierSE(in_ch=X_train.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    t0 = time.time()
    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        val_prob, val_lbl = predict_probs(model, val_loader)
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(val_lbl, val_pred, average="macro")
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

    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)
    elapsed = time.time() - t0

    result = {
        "name": name,
        "n_features_per_step": int(X_train.shape[1]),
        "n_params": int(n_params),
        "best_epoch": int(best_epoch),
        "train_time_s": round(elapsed, 1),
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
    }
    results.append(result)
    log(f"     F1m={result['test_macro_f1']:.4f}  PR-AUC={result['test_pr_auc']:.4f}  "
        f"Params={n_params:,}  BestEp={best_epoch}  Time={elapsed:.0f}s")
    return result


t0_global = time.time()
def log(msg):
    print(f"[{time.time()-t0_global:7.1f}s] {msg}", flush=True)


def main():
    log("Loading data ...")
    log("  - 17 v1 raw (in_ch=17, function 整数)")
    Xa_tr, Xa_va, Xa_te, ya_tr, ya_va, ya_te = load_17v1_raw()
    log(f"    shape: {Xa_tr.shape}")
    log("  - 17 v1 + one-hot (in_ch=44)")
    Xb_tr, Xb_va, Xb_te, yb_tr, yb_va, yb_te = load_17v1_with_onehot()
    log(f"    shape: {Xb_tr.shape}")
    log("  - 19 v2 row-level (in_ch=19)")
    Xc_tr, Xc_va, Xc_te, yc_tr, yc_va, yc_te = load_v2(19)
    log(f"    shape: {Xc_tr.shape}")
    log("  - 27 v2 full (in_ch=27)")
    Xd_tr, Xd_va, Xd_te, yd_tr, yd_va, yd_te = load_v2(27)
    log(f"    shape: {Xd_tr.shape}")

    runs = [
        ("A_17v1_raw_int",     Xa_tr, Xa_va, Xa_te, ya_tr, ya_va, ya_te),
        ("B_17v1_44_onehot",   Xb_tr, Xb_va, Xb_te, yb_tr, yb_va, yb_te),
        ("C_19v2_protocol",    Xc_tr, Xc_va, Xc_te, yc_tr, yc_va, yc_te),
        ("D_27v2_full",        Xd_tr, Xd_va, Xd_te, yd_tr, yd_va, yd_te),
    ]

    results = []
    for name, Xtr, Xva, Xte, ytr, yva, yte in runs:
        run_one(name, Xtr, Xva, Xte, ytr, yva, yte, results)

    df = pd.DataFrame(results)
    out_csv = os.path.join(BASE, "ablation_no_onehot_results.csv")
    df.to_csv(out_csv, index=False)

    log(f"\n{'='*95}")
    log(f"{'encoding':<22} {'in_ch':>6} {'F1m':>7} {'Bin-F1':>7} {'Acc':>6} {'ROC':>6} {'PR-AUC':>7} {'Params':>8}  Time")
    log(f"{'-'*95}")
    for r in results:
        log(f"{r['name']:<22} {r['n_features_per_step']:>6} {r['test_macro_f1']:>7.4f} "
            f"{r['test_binary_f1']:>7.4f} {r['test_accuracy']:>6.4f} {r['test_roc_auc']:>6.4f} "
            f"{r['test_pr_auc']:>7.4f} {r['n_params']:>8,}  {r['train_time_s']:>4.0f}s")
    log(f"{'='*95}")
    log(f"Results saved: {out_csv}")


if __name__ == "__main__":
    main()
