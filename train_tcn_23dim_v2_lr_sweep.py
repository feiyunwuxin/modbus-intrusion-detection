#!/usr/bin/env python3
"""TCN+SE 23-dim (64-combos 冠军版) + B=64 (本工作 batch 扫描甜蜜点) × LR 扫描 + 5-seed

23-dim = 27-dim - {length(2), setpoint(3), crc_mean_w(20), cmd_count_w(22)}
BATCH = 64 (本次 batch sweep 验证的 23-dim 甜蜜点 F1m=0.8454)
扫描 LR ∈ {2e-4, 5e-4, 1e-3, 2e-3, 4e-3}  × 5 seeds = 25 runs

历史依据:
  - [[project-batch-size-breakthrough]] B=128 sweet spot on 19-dim
  - [[project-hparam-sweep-comprehensive]] LR=2e-3 sweet spot +0.027 F1m on 27-dim + B=128
  - 本工作 B=64 sweet spot on 23-dim F1m=0.8454 (default LR=5e-4)

目标: 在 23-dim + B=64 新基础上看 LR=2e-3 是否仍最佳,或在新配置下甜蜜点左移/右移

用法 (单跑):
    python train_tcn_23dim_v2_lr_sweep.py --lr 2e-3 --seed 42
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"
EPOCHS = 20
BATCH = 64             # 本工作 batch sweep 验证的 23-dim 甜蜜点
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

# 23-dim (64-combos 冠军版) — drops {2, 3, 20, 22}
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
DROP = {2: "length", 3: "setpoint", 20: "crc_mean_w", 22: "cmd_count_w"}
N_FEATURES = len(KEEP_23)

# 5 个 LR 档 (宽扫 20x 范围)
LR_GRID = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3]
SEEDS = [42, 123, 456, 789, 1024]


def lr_tag(lr):
    """把 LR 数字转为文件名 tag: 0.0002 -> 2e-4"""
    return f"{lr:.0e}".replace("e-0", "e-")


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
        probs.append(torch.sigmoid(model(xb)).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_one(lr: float, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val   = np.clip(X_val, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test  = np.clip(X_test, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)

    n_windows = len(X_train)
    iters_per_ep = (n_windows + BATCH - 1) // BATCH

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=BATCH, shuffle=False, num_workers=0)

    model = TCNClassifierSE(in_ch=N_FEATURES)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1m, best_state, best_epoch, best_val_f1m = -1, None, -1, -1
    epochs_no_improve = 0
    t0 = time.time()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_t0 = time.time()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        ep_t = time.time() - ep_t0
        val_prob, val_lbl = predict_probs(model, val_loader)
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(val_lbl, val_pred, average="macro")
        scheduler.step(val_f1m)
        history.append({"epoch": epoch, "val_f1m": float(val_f1m), "epoch_time_s": ep_t})
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_val_f1m = val_f1m
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

    return {
        "lr": float(lr),
        "batch": int(BATCH),
        "iters_per_ep": int(iters_per_ep),
        "seed": int(seed),
        "n_features": N_FEATURES,
        "n_params": n_params,
        "best_epoch": int(best_epoch),
        "best_val_f1m": float(best_val_f1m),
        "train_time_s": float(train_time),
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True, choices=LR_GRID)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        # 文件名: 2e-4 -> lr2e-4, 5e-4 -> lr5e-4, 1e-3 -> lr1e-3, 2e-3 -> lr2e-3, 4e-3 -> lr4e-3
        lr_tag = f"{args.lr:.0e}".replace("e-0", "e-")
        args.out = os.path.join(BASE, f"train_tcn_23dim_v2_lr_sweep_partial_lr{lr_tag}_s{args.seed}.json")

    print(f"[run] lr={args.lr}  seed={args.seed}", flush=True)
    t0 = time.time()
    r = run_one(args.lr, args.seed)
    print(f"[done] F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
          f"Epoch={r['best_epoch']}  TrainT={r['train_time_s']:.1f}s  "
          f"WallT={time.time()-t0:.1f}s", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
