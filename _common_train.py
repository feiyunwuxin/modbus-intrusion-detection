#!/usr/bin/env python3
"""Shared utilities for cross-architecture 23-dim x window=16 training scripts.

Used by:
  - train_tcn_23dim_w16_5seed.py
  - train_lstm_23dim_w16_5seed.py
  - train_gru_23dim_w16_5seed.py
  - train_cnnlstm_23dim_w16_5seed.py

The TCN+SE training uses an independent retrain script (BASE path fix only).
"""
import os
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_PATH = r"D:\workspace\claude\Issue\Issue"
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEEDS = [42, 123, 456, 789, 1024]
CLIP_VAL = 10.0


def load_data_23dim_w16(base: str = BASE_PATH):
    """Load preprocessed 23-dim x window=16 numpy arrays.

    Returns X with shape (N, 23, 16) - features as channels-first for Conv1d.
    Returns y with shape (N,) - int64 labels.
    """
    X_tr = np.load(os.path.join(base, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)
    X_va = np.load(os.path.join(base, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)
    X_te = np.load(os.path.join(base, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
    y_tr = np.load(os.path.join(base, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_va = np.load(os.path.join(base, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_te = np.load(os.path.join(base, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)

    # Select 23 features and clip outliers
    X_tr = np.clip(X_tr[:, :, KEEP_23], -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_va = np.clip(X_va[:, :, KEEP_23], -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_te = np.clip(X_te[:, :, KEEP_23], -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)

    # Assert file presence on first call (idempotent)
    assert X_tr.shape[1] == 23, f"Expected 23 features, got {X_tr.shape[1]}"
    assert X_tr.shape[2] == 16, f"Expected window=16, got {X_tr.shape[2]}"

    return X_tr, y_tr, X_va, y_va, X_te, y_te


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader):
    """Run model over loader; return sigmoid probabilities and labels."""
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb)).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def save_predictions_csv(y_true: np.ndarray, probs: np.ndarray, path: str) -> None:
    """Save (y_true, prob_attack) CSV - header included."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack"])
        for y, p in zip(y_true.tolist(), probs.tolist()):
            w.writerow([y, p])


def save_meta_json(results: list, path: str, model_name: str, tag: str, config: dict) -> None:
    """Save 5-seed summary meta JSON."""
    f1m_mean = float(np.mean([r["test_macro_f1"] for r in results]))
    f1m_std = float(np.std([r["test_macro_f1"] for r in results], ddof=0))
    meta = {
        "model": model_name,
        "tag": tag,
        "config": config,
        "seeds": SEEDS,
        "n_seeds": len(results),
        "n_models": len(results),
        "per_seed": results,
        "summary": {"f1m_mean": f1m_mean, "f1m_std": f1m_std},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def train_one_seed(
    model_class,
    seed: int,
    base: str = BASE_PATH,
    tag: str = "arch",
    epochs: int = 20,
    batch: int = 64,
    lr: float = 4e-3,
    wd: float = 1e-5,
    patience: int = 5,
    grad_clip: float = 0.5,
    pos_weight: bool = True,
) -> dict:
    """Train one model on one seed, save .pt + predictions CSV, return metrics dict.

    The model_class must be a callable that returns an nn.Module with forward(x) returning logits.
    X is shape (B, 23, 16) - channels first for Conv1d, raw sequence for LSTM/GRU.
    """
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data_23dim_w16(base)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                            batch_size=batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                             batch_size=batch, shuffle=False, num_workers=0)

    model = model_class()
    n_params = sum(p.numel() for p in model.parameters())

    if pos_weight:
        n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
        pw = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    else:
        pw = None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    t0 = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        val_prob, val_lbl = predict_probs(model, val_loader)
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(val_lbl, val_pred, average="macro")
        scheduler.step(val_f1m)
        history.append({"epoch": epoch, "val_f1m": float(val_f1m),
                        "current_lr": float(optimizer.param_groups[0]["lr"])})
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)

    pt_path = os.path.join(base, f"model_{tag}_s{seed}.pt")
    torch.save({"state_dict": best_state, "seed": int(seed), "n_params": n_params,
                "best_epoch": int(best_epoch), "actual_epochs_run": len(history),
                "best_val_f1m": float(best_f1m), "train_time_s": float(train_time),
                "test_metrics": {
                    "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
                    "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
                    "test_accuracy": float((test_pred == test_lbl).mean()),
                    "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
                    "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
                }},
               pt_path)

    pred_path = os.path.join(base, f"predictions_test_{tag}_s{seed}.csv")
    save_predictions_csv(test_lbl, test_prob, pred_path)

    return {
        "seed": int(seed), "n_params": n_params,
        "best_epoch": int(best_epoch), "actual_epochs_run": len(history),
        "early_stopped": bool(epochs_no_improve >= patience),
        "best_val_f1m": float(best_f1m), "train_time_s": float(train_time),
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        "pt_path": pt_path, "pred_path": pred_path,
    }
