#!/usr/bin/env python3
"""TCN+SE 23-dim (64-combos 冠军版) × batch size 扫描 + 5-seed

23-dim = 27-dim - {length(2), setpoint(3), crc_mean_w(20), cmd_count_w(22)}
       (依据 [[project-64combos-results]] 5-seed bit-perfect 冠军 F1m=0.8287)

扫描网格: BATCH ∈ {64, 96, 128, 192, 256}  × 5 seeds = 25 runs
固定超参: LR=5e-4, WD=1e-5, Dropout=0.3, EPOCHS=20, Patience=5
         ch=64, k=3, dilations=[1,2,4], SE r=8, WINDOW=16

复用 [[project-batch-size-breakthrough]] 历史: B=128 是 19/27 维甜蜜点 (+0.021 F1m),
现在在 23 维上验证 batch 影响是否仍稳健。

用法 (单跑):
    python train_tcn_23dim_v2_batch_sweep.py --batch 128 --seed 42
用法 (平行 5 worker):
    配 train_tcn_23dim_v2_batch_sweep_runner.py 自动调度
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
)

BASE = r"C:\work\Claude\Issue"
EPOCHS = 20
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

# 23-dim (64-combos 冠军版) — drops {2, 3, 20, 22}
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
DROP = {2: "length", 3: "setpoint", 20: "crc_mean_w", 22: "cmd_count_w"}
N_FEATURES = len(KEEP_23)  # 23

# 5 个 batch 档
BATCH_GRID = [64, 96, 128, 192, 256]
SEEDS = [42, 123, 456, 789, 1024]


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


def run_one(batch_size: int, seed: int):
    """跑单次 (batch_size, seed) 训练,返回测试集指标 + 训练时间。"""
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
    iters_per_ep = (n_windows + batch_size - 1) // batch_size

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    model = TCNClassifierSE(in_ch=N_FEATURES)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
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
        "batch": int(batch_size),
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
    parser.add_argument("--batch", type=int, required=True, choices=BATCH_GRID)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--out", type=str, default=None,
                        help="partial JSON 路径,默认 train_tcn_23dim_v2_batch_sweep_partial_b{B}_s{seed}.json")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(BASE, f"train_tcn_23dim_v2_batch_sweep_partial_b{args.batch}_s{args.seed}.json")

    print(f"[run] batch={args.batch}  seed={args.seed}", flush=True)
    t0 = time.time()
    r = run_one(args.batch, args.seed)
    print(f"[done] F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
          f"Epoch={r['best_epoch']}  TrainT={r['train_time_s']:.1f}s  "
          f"WallT={time.time()-t0:.1f}s", flush=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
