#!/usr/bin/env python3
"""
TCN+SE v2 SCADA 特征有效性消融实验.

输入: 27 维 v2 SCADA 数据 (X_*_binary_v2_scada_window16.npy)
模型: TCN v4+SE (与 tcn_binary_v2_v4se_window16.py 相同架构)
epoch: 20 (加速)
评估: Macro-F1 / Binary-F1 / Acc / ROC-AUC / PR-AUC

用法:
  python ablation_tcn_v2_features.py              # 跑全部 6 个消融
  python ablation_tcn_v2_features.py baseline    # 单跑基线
"""

import os, time, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
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

# 27 维特征的语义分组 (基于 processed_meta_v2_scada.json)
# Index 0-15:  v1 基础 (16 raw - time + time_diff)
# Index 16:    time_since_last_same_addr_func  [v2 行级-时序]
# Index 17:    is_unusual_fc                   [v2 行级-异常FC]
# Index 18:    is_response                     [v2 行级-主从]
# Index 19:    press_mean_w                    [v2 窗口-工艺基线]
# Index 20:    crc_mean_w                      [v2 窗口-CRC]
# Index 21:    crc_max_w                       [v2 窗口-CRC]
# Index 22:    cmd_count_w                     [v2 窗口-命令]
# Index 23:    resp_count_w                    [v2 窗口-响应]
# Index 24:    cmd_resp_balance_w              [v2 窗口-主从配对]
# Index 25:    length_nunique_w                [v2 窗口-payload]
# Index 26:    unusual_count_w                 [v2 窗口-异常FC]

ALL_27 = list(range(27))

# 消融实验配置
# 名字 -> 保留的特征索引
ABLATIONS = {
    "01_baseline_27dim":        ALL_27,
    "02_minus_3_row_scada":     list(range(0, 16)) + list(range(19, 27)),   # 去掉 16,17,18
    "03_minus_8_window_scada":  list(range(0, 19)),                          # 去掉 19-26
    "04_minus_cmd_resp_5":      [i for i in ALL_27 if i not in {16, 18, 22, 23, 24}],  # 去时序间隔+主从角色+3窗口cmd
    "05_minus_process_3":       [i for i in ALL_27 if i not in {19, 20, 21}],  # 去压力+CRC均值+CRC最大
    "06_minus_unusual_fc_2":    [i for i in ALL_27 if i not in {17, 26}],     # 去异常FC标志+异常FC密度
    "07_minus_length_nunique":  [i for i in ALL_27 if i != 25],               # 去 payload 长度多样性
    "08_minus_is_response":     [i for i in ALL_27 if i != 18],               # 单去 is_response
    "09_only_v1_17dim":         list(range(0, 15)) + [15] + [18],              # 17 v1 + is_response (近似 17 dim v1)
    "10_only_v1_16dim":         list(range(0, 16)),                            # 纯 16 维 v1 (无 time_diff, 无 command_response)
}

# 架构定义 (与 tcn_binary_v2_v4se_window16.py 一致)
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
        layers = []
        layers.append(TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout))
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


def run_one(name, keep_idx, results_list):
    """跑一个消融实验"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log(f"  >> {name}  (keep {len(keep_idx)} dim)")
    X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)

    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val   = np.clip(X_val,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test  = np.clip(X_test,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    N_FEATURES = X_train.shape[1]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=BATCH, shuffle=False, num_workers=0)

    model = TCNClassifierSE(in_ch=N_FEATURES)
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
        "n_features": len(keep_idx),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "train_time_s": round(elapsed, 1),
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        "kept_idx": keep_idx,
    }
    results_list.append(result)
    log(f"     F1m={result['test_macro_f1']:.4f}  PR-AUC={result['test_pr_auc']:.4f}  "
        f"Params={n_params:,}  Epoch={best_epoch}  Time={elapsed:.0f}s")
    return result


t0_global = time.time()
def log(msg):
    print(f"[{time.time()-t0_global:7.1f}s] {msg}", flush=True)


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "all":
            runs = ABLATIONS
        else:
            runs = {target: ABLATIONS[target]}
    else:
        runs = ABLATIONS

    log(f"Ablation: {len(runs)} variants, epoch={EPOCHS}")
    results = []
    for name, keep_idx in runs.items():
        run_one(name, keep_idx, results)

    df = pd.DataFrame([{
        "name": r["name"],
        "n_features": r["n_features"],
        "n_params": r["n_params"],
        "best_epoch": r["best_epoch"],
        "test_macro_f1": r["test_macro_f1"],
        "test_binary_f1": r["test_binary_f1"],
        "test_accuracy": r["test_accuracy"],
        "test_roc_auc": r["test_roc_auc"],
        "test_pr_auc": r["test_pr_auc"],
        "train_time_s": r["train_time_s"],
    } for r in results])
    out_csv = os.path.join(BASE, "ablation_tcn_v2_features_results.csv")
    df.to_csv(out_csv, index=False)
    out_json = os.path.join(BASE, "ablation_tcn_v2_features_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nResults saved: {out_csv}")
    log(f"\n{'='*90}")
    log(f"{'name':<28} {'Dim':>5} {'F1m':>7} {'PR-AUC':>8} {'Params':>8} {'deltaF1m':>8}  time")
    log(f"{'-'*90}")
    base_f1m = results[0]["test_macro_f1"]
    for r in results:
        delta = r["test_macro_f1"] - base_f1m
        log(f"{r['name']:<28} {r['n_features']:>5} {r['test_macro_f1']:>7.4f} "
            f"{r['test_pr_auc']:>8.4f} {r['n_params']:>8,} {delta:>+8.4f}  {r['train_time_s']:>4.0f}s")
    log(f"{'='*90}")


if __name__ == "__main__":
    main()
