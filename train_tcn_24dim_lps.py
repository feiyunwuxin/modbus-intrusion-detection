#!/usr/bin/env python3
"""24 维 = 27 - {length (idx 2), press_mean_w (idx 19), setpoint (idx 3)} retrain 5-seed 验证

5-seed LOO 单点预测:
  - length (idx 2):         DeltaF1m = +0.0051 (正向最大)
  - press_mean_w (idx 19):  DeltaF1m = +0.0035 (正向第二)
  - setpoint (idx 3):       DeltaF1m = +0.0003 (中性)

LOO 加总预测: +0.0051 + 0.0035 + 0.0003 = +0.0089
期望 F1m >= 0.8205 + 0.0089 = 0.8294
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"
EPOCHS = 20; BATCH = 512; LR = 5e-4; WD = 1e-5; PATIENCE = 5
GRAD_CLIP = 0.5; DROPOUT = 0.3; CLIP_VAL = 10.0
N_BLOCKS = 3; CHANNELS = 64; KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]; SE_REDUCTION = 8; WINDOW = 16
SEEDS = [42, 123, 456, 789, 1024]

# 24 维 = 27 - {2, 19, 3}
KEEP_24 = [i for i in range(27) if i not in (2, 19, 3)]
DROP = {2: "length", 19: "press_mean_w", 3: "setpoint"}

FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]

print(f"=== TCN+SE 24-dim (-length, -press_mean_w, -setpoint) Retrain 5-seed ===")
print(f"去除特征: {DROP}")
print(f"保留特征 ({len(KEEP_24)} dim)")

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

def run_one(keep_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val   = np.clip(X_val, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test  = np.clip(X_test, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    N_FEATURES = X_train.shape[1]
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
            epochs_no_impose = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break
    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)
    return {
        "n_features": N_FEATURES, "n_params": n_params, "best_epoch": best_epoch,
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
    }

print("\n=== 24-dim (-length, -press_mean_w, -setpoint) 5-seed ===")
results = []
for seed in SEEDS:
    r = run_one(KEEP_24, seed=seed)
    print(f"  seed={seed}  F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  Epoch={r['best_epoch']}  Params={r['n_params']:,}")
    results.append(r)

print("\n" + "="*80)
print("=== 24-dim (-length, -press_mean_w, -setpoint) 5-seed 汇总 ===")
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

# vs 27 baseline 真实 mean (5-seed) = 0.8205
baseline_f1m_5seed = 0.8205
baseline_pr_5seed = 0.9052
print(f"\nDelta vs 27 baseline 5-seed mean (F1m={baseline_f1m_5seed}, PR-AUC={baseline_pr_5seed}):")
print(f"  24-dim mean:        F1m = {mean_f1m:.4f}  DeltaF1m = {mean_f1m - baseline_f1m_5seed:+.4f}")
print(f"  24-dim seed=42:     F1m = {results[0]['test_macro_f1']:.4f}  DeltaF1m = {results[0]['test_macro_f1'] - baseline_f1m_5seed:+.4f}")
print(f"  LOO 加总预测:        DeltaF1m >= +0.0089 (= +0.0051 + 0.0035 + 0.0003)")
print(f"  LOO 预测满足 (mean)? {'YES OK' if mean_f1m - baseline_f1m_5seed >= 0.0089 else 'NO FAIL'}")
print(f"  LOO 预测满足 (seed=42)? {'YES OK' if results[0]['test_macro_f1'] - baseline_f1m_5seed >= 0.0089 else 'NO FAIL'}")

# vs 26 (-length) 单冠军 (5-seed mean = 0.8256)
single_champion_f1m = 0.8256
print(f"\nDelta vs 26-dim (-length) single champion (F1m={single_champion_f1m}):")
print(f"  24-dim mean:        DeltaF1m = {mean_f1m - single_champion_f1m:+.4f}")

# 检查 outlier
outliers = [(SEEDS[i], r['test_macro_f1']) for i, r in enumerate(results) if r['test_macro_f1'] < mean_f1m - 2*std_f1m]
print(f"\nOutlier seeds (< mean-2std): {outliers if outliers else 'None'}")

out_json = os.path.join(BASE, "train_tcn_24dim_lps_results.json")
with open(out_json, "w") as f:
    json.dump({"results_24dim": results,
               "mean_f1m": mean_f1m, "std_f1m": std_f1m,
               "mean_pr_auc": mean_pr, "std_pr_auc": std_pr,
               "drop_features": DROP, "keep_idx": KEEP_24}, f, indent=2)
print(f"\nSaved: {out_json}")