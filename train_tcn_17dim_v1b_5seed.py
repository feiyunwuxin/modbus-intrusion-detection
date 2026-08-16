#!/usr/bin/env python3
"""原始 17 维 TCN+SE v1b 5-seed retrain

数据源: X_*_binary.npy (16 raw + time_diff, 无 one-hot)
历史 (2026-06-19): single seed=42 F1m=0.8002 PR-AUC=0.8968

窗口化: 每 16 行组合成一个样本 (与 27 维 window16 一致)
超参: B=512 / LR=5e-4 / WD=1e-5 / Dropout=0.3 / Patience=7 / Epoch<=35
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"
EPOCHS = 35; BATCH = 512; LR = 5e-4; WD = 1e-5; PATIENCE = 7
GRAD_CLIP = 0.5; DROPOUT = 0.3; CLIP_VAL = 10.0
N_BLOCKS = 3; CHANNELS = 64; KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]; SE_REDUCTION = 8; WINDOW = 16
SEEDS = [42, 123, 456, 789, 1024]

# 17 维特征名
FEATURE_NAMES_17 = [
    "address",                       # 0
    "function",                      # 1  (NO one-hot)
    "length",                        # 2
    "setpoint",                      # 3
    "gain",                          # 4
    "reset rate",                    # 5
    "deadband",                      # 6
    "cycle time",                    # 7
    "rate",                          # 8
    "system mode",                   # 9
    "control scheme",                # 10
    "pump",                          # 11
    "solenoid",                      # 12
    "pressure measurement",          # 13
    "crc rate",                      # 14
    "command response",              # 15
    "time_diff",                     # 16
]

print(f"=== TCN+SE 17-dim v1b (原始 17 维) Retrain 5-seed ===")
print(f"数据源: X_*_binary.npy (行级)")
print(f"特征 ({len(FEATURE_NAMES_17)}): {FEATURE_NAMES_17}")

# 加载行级数据
print("\n=== 加载 17 维行级数据 ===")
X_train_raw = np.load(os.path.join(BASE, "X_train_binary.npy")).astype(np.float32)
X_val_raw   = np.load(os.path.join(BASE, "X_val_binary.npy")).astype(np.float32)
X_test_raw  = np.load(os.path.join(BASE, "X_test_binary.npy")).astype(np.float32)
y_train_raw = np.load(os.path.join(BASE, "y_train_binary.npy")).astype(np.int64)
y_val_raw   = np.load(os.path.join(BASE, "y_val_binary.npy")).astype(np.int64)
y_test_raw  = np.load(os.path.join(BASE, "y_test_binary.npy")).astype(np.int64)
print(f"  train rows: {X_train_raw.shape} val: {X_val_raw.shape} test: {X_test_raw.shape}")

# 窗口化
print(f"\n=== 窗口化 (每 {WINDOW} 行) ===")
def make_windows(X, y, win):
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    return Xw, yw

X_train, y_train = make_windows(X_train_raw, y_train_raw, WINDOW)
X_val,   y_val   = make_windows(X_val_raw,   y_val_raw,   WINDOW)
X_test,  y_test  = make_windows(X_test_raw,  y_test_raw,  WINDOW)
print(f"  train windows: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")
print(f"  window-label dist: train {y_train.mean()*100:.2f}% val {y_val.mean()*100:.2f}% test {y_test.mean()*100:.2f}%")

X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_val   = np.clip(X_val,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_test  = np.clip(X_test,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
print(f"  TCN input shape: {X_train.shape}")


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


def run_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    model = TCNClassifierSE(in_ch=17)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
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
    return {
        "n_features": 17, "n_params": n_params, "best_epoch": best_epoch,
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
    }


print(f"\n=== 17-dim v1b 5-seed ===")
results = []
for seed in SEEDS:
    r = run_one(seed)
    print(f"  seed={seed}  F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  Epoch={r['best_epoch']}  Params={r['n_params']:,}")
    results.append(r)

print("\n" + "="*80)
print("=== 17-dim v1b 5-seed 汇总 ===")
print("="*80)
print(f"{'seed':>5} {'F1m':>8} {'BinF1':>8} {'Acc':>8} {'ROC':>8} {'PR':>8} {'Epoch':>5}")
for i, r in enumerate(results):
    print(f"{SEEDS[i]:>5} {r['test_macro_f1']:>8.4f} {r['test_binary_f1']:>8.4f} "
          f"{r['test_accuracy']:>8.4f} {r['test_roc_auc']:>8.4f} {r['test_pr_auc']:>8.4f} {r['best_epoch']:>5d}")

mean_f1m = np.mean([r['test_macro_f1'] for r in results])
std_f1m = np.std([r['test_macro_f1'] for r in results])
mean_pr = np.mean([r['test_pr_auc'] for r in results])
std_pr = np.std([r['test_pr_auc'] for r in results])
print(f"\n{'mean':>5} {mean_f1m:>8.4f} (std={std_f1m:.4f}) {mean_pr:>8.4f} (std={std_pr:.4f})")

# 历史对照
print(f"\n=== 历史对照 (memory project-tcn-v3-19dim-results.md) ===")
print(f"  17 维 v1 (有 one-hot, 44 Conv1D dims)   single seed: F1m=0.8340  PR-AUC=0.9025")
print(f"  17 维 v1b (无 one-hot, 17 Conv1D dims)  single seed: F1m=0.8002  PR-AUC=0.8968")
print(f"  19 维 v3 (无 one-hot)                   single seed: F1m=0.8444  PR-AUC=0.8892")
print(f"  27 维 v2 (无 one-hot)                   single seed: F1m=0.8367  PR-AUC=0.8990")
print()
print(f"  本次 17-dim v1b 5-seed mean: F1m={mean_f1m:.4f}  PR-AUC={mean_pr:.4f}")
print(f"  本次 17-dim v1b seed=42:   F1m={results[0]['test_macro_f1']:.4f}  PR-AUC={results[0]['test_pr_auc']:.4f}")

# 检查 outlier
outliers = [(SEEDS[i], r['test_macro_f1']) for i, r in enumerate(results) if r['test_macro_f1'] < mean_f1m - 2*std_f1m]
print(f"\nOutlier seeds (< mean-2std): {outliers if outliers else 'None'}")

out_json = os.path.join(BASE, "train_tcn_17dim_v1b_5seed_results.json")
with open(out_json, "w") as f:
    json.dump({"results_17dim": results,
               "mean_f1m": mean_f1m, "std_f1m": std_f1m,
               "mean_pr_auc": mean_pr, "std_pr_auc": std_pr,
               "feature_names": FEATURE_NAMES_17}, f, indent=2)
print(f"\nSaved: {out_json}")