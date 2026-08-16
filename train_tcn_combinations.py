#!/usr/bin/env python3
"""6 个特征 (function, setpoint, cmd_count_w, crc_mean_w, press_mean_w, length)
选 3 个的所有 20 种组合的 LOO 加总预测 + 实际 5-seed 验证

已跑过 (失败案例):
  - length + crc_mean_w + press_mean_w  → F1m 0.8129 (-0.0076)
  - length + press_mean_w + cmd_count_w → F1m 0.8022 (-0.0183)

新跑 3 个候选 (避免 press_mean_w + cmd_count_w 危险组合):
  A. length + setpoint + crc_mean_w  (LOO +0.0064)
  B. length + setpoint + function   (LOO +0.0050)
  C. length + function + crc_mean_w (LOO +0.0057)
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

# 6 个特征的 idx 映射
FEATURE_IDX = {
    "function":     1,
    "length":       2,
    "setpoint":     3,
    "press_mean_w": 19,
    "crc_mean_w":   20,
    "cmd_count_w":  22,
}
# LOO 5-seed 单点预测
LOO_DELTA = {
    "function":     -0.0004,
    "length":       +0.0051,
    "setpoint":     +0.0003,
    "press_mean_w": +0.0035,
    "crc_mean_w":   +0.0010,
    "cmd_count_w":  +0.0006,
}

# 候选组合 (避开已知失败)
CANDIDATES = [
    ("length", "setpoint", "crc_mean_w"),
    ("length", "setpoint", "function"),
    ("length", "function", "crc_mean_w"),
]

FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]

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
            epochs_no_improve = 0
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


print("=" * 100)
print("=== 6 个特征选 3 个的所有 20 种组合 LOO 加总预测排序 ===")
print("=" * 100)
all_combos = []
from itertools import combinations
for combo in combinations(FEATURE_IDX.keys(), 3):
    loo_sum = sum(LOO_DELTA[f] for f in combo)
    all_combos.append((combo, loo_sum))

# 按 LOO 加总预测排序
all_combos.sort(key=lambda x: x[1], reverse=True)
print(f"\n{'#':>3} {'组合 (3 特征)':<60} {'LOO 加总':>10}")
for i, (combo, loo_sum) in enumerate(all_combos, 1):
    print(f"{i:>3} {str(combo):<60} {loo_sum:>+10.4f}")

# 列出候选
print(f"\n=== 候选 (避开 press_mean_w + cmd_count_w 危险组合) ===")
for i, (combo, loo_sum) in enumerate(all_combos, 1):
    if "press_mean_w" not in combo and "cmd_count_w" not in combo:
        print(f"  候选 #{i}: {combo} (LOO +{loo_sum:.4f})")

print("\n" + "=" * 100)
print("=== 跑 3 个候选组合 5-seed ===")
print("=" * 100)

baseline_f1m_5seed = 0.8205
all_results = {}

for combo in CANDIDATES:
    drop_idx = [FEATURE_IDX[f] for f in combo]
    keep_idx = [i for i in range(27) if i not in drop_idx]
    loo_pred = sum(LOO_DELTA[f] for f in combo)
    print(f"\n=== 组合: {combo} ===")
    print(f"  drop idx: {drop_idx}")
    print(f"  keep idx ({len(keep_idx)} dim): {keep_idx}")
    print(f"  LOO 加总预测: +{loo_pred:.4f}")

    results = []
    for seed in SEEDS:
        r = run_one(keep_idx, seed)
        print(f"    seed={seed}  F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  Epoch={r['best_epoch']}  Params={r['n_params']:,}")
        results.append(r)

    mean_f1m = np.mean([r['test_macro_f1'] for r in results])
    std_f1m = np.std([r['test_macro_f1'] for r in results])
    mean_pr = np.mean([r['test_pr_auc'] for r in results])
    std_pr = np.std([r['test_pr_auc'] for r in results])
    print(f"  mean: F1m={mean_f1m:.4f} (±{std_f1m:.4f})  PR-AUC={mean_pr:.4f} (±{std_pr:.4f})")
    print(f"  DeltaF1m vs baseline (0.8205): {mean_f1m - baseline_f1m_5seed:+.4f}")
    print(f"  DeltaF1m vs LOO 预测 (+{loo_pred:.4f}): {mean_f1m - baseline_f1m_5seed - loo_pred:+.4f}")

    all_results[",".join(combo)] = {
        "combo": list(combo),
        "drop_idx": drop_idx,
        "keep_idx": keep_idx,
        "loo_pred": loo_pred,
        "results": results,
        "mean_f1m": mean_f1m,
        "std_f1m": std_f1m,
        "mean_pr": mean_pr,
        "std_pr": std_pr,
    }

# 汇总对比
print("\n" + "=" * 100)
print("=== 3 个候选组合对比汇总 ===")
print("=" * 100)
print(f"{'组合':<55} {'LOO 加总':>10} {'F1m mean':>10} {'std':>7} {'PR-AUC':>9} {'ΔF1m':>8} {'LOO-实际':>9}")
print("-" * 100)
for combo_str, r in all_results.items():
    delta_actual = r['mean_f1m'] - baseline_f1m_5seed
    delta_loo_actual = delta_actual - r['loo_pred']
    print(f"{combo_str:<55} {r['loo_pred']:>+10.4f} {r['mean_f1m']:>10.4f} {r['std_f1m']:>7.4f} {r['mean_pr']:>9.4f} {delta_actual:>+8.4f} {delta_loo_actual:>+9.4f}")

# 保存
out_json = os.path.join(BASE, "train_tcn_combinations_results.json")
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_json}")