#!/usr/bin/env python3
"""TCN baseline + SWA + EMA on 23-dim x window=16, 5 seeds.

Self-contained: reuses TCNClassifier from train_tcn_23dim_w16_5seed.py
and data loader from _common_train.py. Trains 20 epochs with:
  - SWA snapshots from epoch >= 10, averaged + BN re-update at end
  - EMA (alpha=0.999) updated every epoch

Outputs 3 prediction CSVs per seed (raw / SWA / EMA) + meta JSON.
"""
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS
from train_tcn_23dim_w16_5seed import TCNClassifier

TAG_PREFIX = "tcn_swa_ema"
SWA_START_EPOCH = 10
EPOCHS = 20
EMA_DECAY = 0.999
BATCH = 64
LR = 4e-3
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5


def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def average_state_dicts(state_dicts):
    """Element-wise mean of N state_dicts. Returns new state_dict."""
    avg = copy.deepcopy(state_dicts[0])
    for k in avg.keys():
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        avg[k] = stacked.mean(dim=0).to(avg[k].dtype)
    return avg


def bn_re_update(model, train_loader):
    """Recompute BatchNorm running stats with one forward pass over train data."""
    model.train()
    with torch.no_grad():
        for xb, _ in train_loader:
            model(xb)
    model.eval()


def train_one_seed_swa_ema(seed):
    """Train one seed, evaluate 3 views, save CSVs. Returns metrics dict."""
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

    swa_snapshots = []
    ema_state = None
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

        # SWA snapshot
        if epoch >= SWA_START_EPOCH:
            swa_snapshots.append({k: v.clone() for k, v in model.state_dict().items()})

        # EMA update
        cur_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ema_state is None:
            ema_state = cur_state
        else:
            for k in ema_state.keys():
                ema_state[k] = EMA_DECAY * cur_state[k] + (1 - EMA_DECAY) * ema_state[k]

        if epochs_no_improve >= PATIENCE:
            break

    train_time = time.time() - t0
    n_swa_snapshots = len(swa_snapshots)
    swa_warning = ""
    if n_swa_snapshots == 0:
        swa_snapshots = [best_state]
        swa_warning = "FALLBACK: no SWA snapshots (early stop), used best_state"

    # View 1: raw best
    model.load_state_dict(best_state)
    raw_prob, test_lbl = predict_probs(model, test_loader)

    # View 2: SWA averaged + BN re-update
    swa_state = average_state_dicts(swa_snapshots)
    model.load_state_dict(swa_state)
    bn_re_update(model, train_loader)
    swa_prob, _ = predict_probs(model, test_loader)

    # View 3: EMA
    model.load_state_dict(ema_state)
    ema_prob, _ = predict_probs(model, test_loader)

    def four_metrics(probs):
        pred = (probs >= 0.5).astype(int)
        return {
            "test_macro_f1": float(f1_score(test_lbl, pred, average="macro")),
            "test_binary_f1": float(f1_score(test_lbl, pred, average="binary")),
            "test_accuracy": float(accuracy_score(test_lbl, pred)),
            "test_precision": float(precision_score(test_lbl, pred, zero_division=0)),
            "test_recall": float(recall_score(test_lbl, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(test_lbl, probs)),
        }

    raw_metrics = four_metrics(raw_prob)
    swa_metrics = four_metrics(swa_prob)
    ema_metrics = four_metrics(ema_prob)

    # Save 3 prediction CSVs
    for view, probs in [("raw", raw_prob), ("swa", swa_prob), ("ema", ema_prob)]:
        with open(os.path.join(BASE_PATH, f"predictions_test_tcn_{view}_s{seed}.csv"), "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["y_true", "prob_attack"])
            for y, p in zip(test_lbl.tolist(), probs.tolist()):
                w.writerow([y, p])

    print(f"  seed={seed}: "
          f"raw F1={raw_metrics['test_binary_f1']:.4f}  "
          f"SWA F1={swa_metrics['test_binary_f1']:.4f}  "
          f"EMA F1={ema_metrics['test_binary_f1']:.4f}  "
          f"swa_snaps={n_swa_snapshots}  time={train_time:.1f}s"
          + (f"  [{swa_warning}]" if swa_warning else ""))

    return {
        "seed": int(seed),
        "n_params": n_params,
        "best_epoch": int(best_epoch),
        "n_swa_snapshots": n_swa_snapshots,
        "swa_warning": swa_warning,
        "train_time_s": float(train_time),
        "raw": raw_metrics,
        "swa": swa_metrics,
        "ema": ema_metrics,
    }


def main():
    print(f"=== TCN + SWA + EMA: 23-dim x window=16 x 5 seeds ===\n"
          f"  SWA start epoch: {SWA_START_EPOCH}, EMA decay: {EMA_DECAY}\n")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_swa_ema(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG_PREFIX}.json"),
                   model_name="TCN + SWA + EMA", tag=TAG_PREFIX,
                   config={"epochs": EPOCHS, "swa_start_epoch": SWA_START_EPOCH,
                           "ema_decay": EMA_DECAY, "batch": BATCH, "lr": LR, "wd": WD,
                           "patience": PATIENCE, "n_archs": 3,
                           "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG_PREFIX}.json")


if __name__ == "__main__":
    main()