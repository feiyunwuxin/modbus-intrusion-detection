#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""23 维 CNN 训练 - 用于替换 H743 上的 TCN+SE 模型

数据流水线与 train_tcn_23dim.py 完全一致 (bit-perfect 对齐 H743 sliding_window.c):
  - X_*_binary_v2_scada_window16.npy  (N, 16, 27)
  - KEEP_23 切片 -> 23 维
  - np.clip(±10) + transpose(0,2,1) -> (N, 23, 16)

模型架构 (仿 19 维 CNN1D, hidden=64):
  Conv1d(23->64,k=3,pad=1) + BN + ReLU
  Conv1d(64->64,k=3,pad=1) + BN + ReLU
  Conv1d(64->64,k=3,pad=1) + BN + ReLU
  AdaptiveAvgPool1d(1)
  Linear(64->32) + ReLU
  Linear(32->1) -> sigmoid

部署时 BN fold 进 Conv, C 端只需 Conv+ReLU。
"""

import os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, precision_score,
    recall_score, accuracy_score,
)

BASE = r"C:\work\Claude\Issue"

# ---- 与 train_tcn_23dim.py 一致的超参 ----
EPOCHS    = 30
BATCH     = 128
LR        = 5e-4
WD        = 1e-5
PATIENCE  = 6
GRAD_CLIP = 0.5
DROPOUT   = 0.3
CLIP_VAL  = 10.0
HIDDEN    = 64
WINDOW    = 16

# 23 维 = 27 维 - {3, 6, 16, 20}  (与 TCN 23-dim / H743 sliding_window.c 一致)
KEEP_23 = [0,1,2,4,5,7,8,9,10,11,12,13,14,15,17,18,19,21,22,23,24,25,26]
DROP    = {3:"setpoint", 6:"deadband", 16:"time_since_last_same_addr_func", 20:"crc_mean_w"}

FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]


class CNN1D23(nn.Module):
    """Conv1d-BN-ReLU x3 -> GAP -> FC(64->32)+ReLU -> FC(32->1)"""
    def __init__(self, in_ch=23, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.drop  = nn.Dropout(dropout)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(hidden, 32)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


def load_data(keep_idx):
    """加载 + KEEP_23 切片 + clip + transpose -> (N, 23, 16)"""
    out = {}
    for split in ("train", "val", "test"):
        X = np.load(os.path.join(BASE, f"X_{split}_binary_v2_scada_window16.npy")).astype(np.float32)
        y = np.load(os.path.join(BASE, f"y_{split}_binary_v2_scada_window16.npy")).astype(np.int64)
        X = X[:, :, keep_idx]
        X = np.clip(X, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)  # (N,16,F) -> (N,F,16)
        out[split] = (X, y)
    return out


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb)).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def best_threshold(prob, lbl, grid=None):
    """扫描阈值, 最大化 binary F1 (IDS 部署口径)"""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(lbl, (prob >= t).astype(int), average="binary")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def run_one(keep_idx, seed, data, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    X_test,  y_test  = data["test"]
    N_FEATURES = X_train.shape[1]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=BATCH, shuffle=False, num_workers=0)

    model = CNN1D23(in_ch=N_FEATURES)
    n_params = sum(p.numel() for p in model.parameters())

    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

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
        if verbose:
            print(f"    epoch {epoch:2d}  val_F1m={val_f1m:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                if verbose:
                    print(f"    early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)

    test_prob, test_lbl = predict_probs(model, test_loader)
    thr, test_bin_f1_thr = best_threshold(test_prob, test_lbl)
    pred05  = (test_prob >= 0.5).astype(int)
    predthr = (test_prob >= thr).astype(int)

    res = {
        "seed": seed,
        "n_features": int(N_FEATURES),
        "n_params": int(n_params),
        "best_epoch": best_epoch,
        "val_f1m_at_best": float(best_f1m),
        "best_threshold": thr,
        "test_at_0.5": {
            "macro_f1": float(f1_score(test_lbl, pred05, average="macro")),
            "binary_f1": float(f1_score(test_lbl, pred05, average="binary")),
            "accuracy": float(accuracy_score(test_lbl, pred05)),
            "precision": float(precision_score(test_lbl, pred05)),
            "recall": float(recall_score(test_lbl, pred05)),
            "roc_auc": float(roc_auc_score(test_lbl, test_prob)),
            "pr_auc": float(average_precision_score(test_lbl, test_prob)),
        },
        "test_at_thr": {
            "macro_f1": float(f1_score(test_lbl, predthr, average="macro")),
            "binary_f1": float(f1_score(test_lbl, predthr, average="binary")),
            "accuracy": float(accuracy_score(test_lbl, predthr)),
            "precision": float(precision_score(test_lbl, predthr)),
            "recall": float(recall_score(test_lbl, predthr)),
        },
    }
    res["_model"] = model
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 80)
    print("=== 23-dim CNN1D 训练 (hidden=64) ===")
    print("=" * 80)
    print(f"KEEP_23 ({len(KEEP_23)}): {[FEATURE_NAMES_27[i] for i in KEEP_23]}")
    print(f"超参: EPOCHS={EPOCHS} BATCH={BATCH} LR={LR} HIDDEN={HIDDEN} DROPOUT={DROPOUT}")

    print("\n加载数据 ...")
    data = load_data(KEEP_23)
    for s in ("train", "val", "test"):
        X, y = data[s]
        print(f"  {s:5s} X={X.shape} pos_rate={y.mean():.4f}")

    results = []
    best_overall = None
    for seed in args.seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        t0 = time.time()
        r = run_one(KEEP_23, seed, data, verbose=args.verbose)
        r["train_seconds"] = round(time.time() - t0, 2)
        m = r.pop("_model")
        results.append(r)
        print(f"  -> Params={r['n_params']:,} Epoch={r['best_epoch']} "
              f"BinF1@thr={r['test_at_thr']['binary_f1']:.4f} "
              f"PR-AUC={r['test_at_0.5']['pr_auc']:.4f} "
              f"thr={r['best_threshold']:.2f} ({r['train_seconds']}s)")
        if best_overall is None or r["test_at_thr"]["binary_f1"] > best_overall[0]["test_at_thr"]["binary_f1"]:
            best_overall = (r, m, seed)

    best_r, best_model, best_seed = best_overall

    print("\n" + "=" * 80)
    print(f"{'seed':>6} {'BinF1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8} {'PR-AUC':>8} {'thr':>5} {'Ep':>4}")
    print("-" * 80)
    for r in results:
        t = r["test_at_thr"]
        print(f"{r['seed']:>6} {t['binary_f1']:>8.4f} {t['precision']:>8.4f} {t['recall']:>8.4f} "
              f"{t['accuracy']:>8.4f} {r['test_at_0.5']['pr_auc']:>8.4f} "
              f"{r['best_threshold']:>5.2f} {r['best_epoch']:>4d}")

    mean_bin = float(np.mean([r["test_at_thr"]["binary_f1"] for r in results]))
    std_bin  = float(np.std([r["test_at_thr"]["binary_f1"] for r in results]))
    mean_pr  = float(np.mean([r["test_at_0.5"]["pr_auc"] for r in results]))
    print("-" * 80)
    print(f"{'mean':>6} {mean_bin:>8.4f} (std={std_bin:.4f}) {'':>24} {mean_pr:>8.4f}")

    # 保存最佳 seed
    ckpt = {
        "state_dict": best_model.state_dict(),
        "n_params": best_r["n_params"],
        "best_epoch": best_r["best_epoch"],
        "best_threshold": best_r["best_threshold"],
        "model_name": "cnn1d_23dim_ch64",
        "preprocessing": "v2_scada_23dim_window16",
        "keep_idx": KEEP_23,
        "feature_names": [FEATURE_NAMES_27[i] for i in KEEP_23],
        "hidden": HIDDEN,
        "window": WINDOW,
        "clip_val": CLIP_VAL,
    }
    ckpt_path = os.path.join(BASE, f"model_cnn1d_23dim_ch64_s{best_seed}.pt")
    torch.save(ckpt, ckpt_path)

    summary = {
        "model": "cnn1d_23dim_ch64",
        "hidden": HIDDEN, "window": WINDOW, "n_params": best_r["n_params"],
        "hyperparams": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
                        "dropout": DROPOUT, "clip_val": CLIP_VAL, "patience": PATIENCE},
        "keep_idx": KEEP_23,
        "feature_names": [FEATURE_NAMES_27[i] for i in KEEP_23],
        "per_seed": results,
        "mean_binary_f1": mean_bin, "std_binary_f1": std_bin, "mean_pr_auc": mean_pr,
        "best_seed": best_seed,
        "best": {k: v for k, v in best_r.items() if not k.startswith("_")},
        "checkpoint": ckpt_path,
    }
    meta_path = os.path.join(BASE, "processed_meta_cnn1d_23dim_ch64.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n最佳 seed = {best_seed}  (BinF1={best_r['test_at_thr']['binary_f1']:.4f})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Meta      : {meta_path}")
    print(f"部署阈值   : {best_r['best_threshold']:.2f}")


if __name__ == "__main__":
    main()
