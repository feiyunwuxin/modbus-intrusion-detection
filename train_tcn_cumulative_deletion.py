#!/usr/bin/env python3
"""TCN+SE 累积删除实验: 依次删除 1/2/3/4/5/6 特征

删除顺序 (按用户要求):
  step 1: function          (idx 1)
  step 2: setpoint          (idx 3)
  step 3: cmd_count_w       (idx 22)
  step 4: crc_mean_w        (idx 20)
  step 5: press_mean_w      (idx 19)
  step 6: length            (idx 2)

7 档累积配置 (含 27 baseline):
  27  baseline (none)                                                 -- F1m=0.8205 (LOO baseline)
  26  -function                                                       -- NEW
  25  -function,-setpoint                                             -- NEW
  24  -function,-setpoint,-cmd_count_w                                -- NEW
  23  -function,-setpoint,-cmd_count_w,-crc_mean_w                    -- NEW
  22  -function,-setpoint,-cmd_count_w,-crc_mean_w,-press_mean_w       -- NEW
  21  -function,-setpoint,-cmd_count_w,-crc_mean_w,-press_mean_w,-length -- 已有 = 之前 21-dim 测试

依据 [[project-tcn-v2-loo-5seed-correction]] 5-seed LOO 单点 (basis 排序):
  idx 1 function:         ΔF1m = -0.0004   (基本中性)
  idx 3 setpoint:         ΔF1m = +0.0003   (基本中性)
  idx 22 cmd_count_w:     ΔF1m = +0.0006   (基本中性)
  idx 20 crc_mean_w:      ΔF1m = +0.0010   (小正)
  idx 19 press_mean_w:    ΔF1m = +0.0035   (正)
  idx 2 length:           ΔF1m = +0.0051   (单冠军)

LOO 加性预测 (累加): -0.0004 + 0.0003 + 0.0006 + 0.0010 + 0.0035 + 0.0051 = +0.0101
但 [[project-tcn-21dim-results]] 已证: ≥2 同时移除 LOO 加性失效 (实测 -0.022).

设置 (与所有 LOO 5-seed 历史配置一致):
  B=512, LR=5e-4, WD=1e-5, Dropout=0.3, Epoch=20, Patience=5
  ch=64, k=3, dilations=[1,2,4], SE r=8, window=16
  seeds = [42, 123, 456, 789, 1024]
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

FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]

# 累积删除顺序 (按用户要求)
CUMULATIVE_DROPS = [
    ("27_baseline",           []),
    ("26_-function",          [1]),
    ("25_-function,-setpoint",[1, 3]),
    ("24_-function,-setpoint,-cmd_count_w", [1, 3, 22]),
    ("23_-function,-setpoint,-cmd_count_w,-crc_mean_w", [1, 3, 22, 20]),
    ("22_-function,-setpoint,-cmd_count_w,-crc_mean_w,-press_mean_w", [1, 3, 22, 20, 19]),
    ("21_-function,-setpoint,-cmd_count_w,-crc_mean_w,-press_mean_w,-length", [1, 3, 22, 20, 19, 2]),
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
        "val_macro_f1": float(val_f1m),
    }


t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:8.1f}s] {msg}", flush=True)


all_results = {}
total_runs = sum(len(SEEDS) for _ in CUMULATIVE_DROPS)
run_idx = 0
for name, drop_ids in CUMULATIVE_DROPS:
    keep_idx = [i for i in range(27) if i not in drop_ids]
    n_feat = len(keep_idx)
    log(f"=== {name} | drop={drop_ids} | n_features={n_feat} ===")
    seeds_results = []
    for seed in SEEDS:
        run_idx += 1
        r = run_one(keep_idx, seed)
        seeds_results.append(r)
        log(f"  [{run_idx}/{total_runs}] seed={seed}  F1m={r['test_macro_f1']:.4f}  "
            f"PR-AUC={r['test_pr_auc']:.4f}  Epoch={r['best_epoch']}  Params={r['n_params']:,}")
    f1ms = np.array([r['test_macro_f1'] for r in seeds_results])
    prs = np.array([r['test_pr_auc']  for r in seeds_results])
    bfs = np.array([r['test_binary_f1'] for r in seeds_results])
    accs = np.array([r['test_accuracy'] for r in seeds_results])
    rocs = np.array([r['test_roc_auc'] for r in seeds_results])
    eps = np.array([r['best_epoch'] for r in seeds_results])
    n_collapsed = int((eps < 5).sum())
    all_results[name] = {
        "n_features": n_feat,
        "drop_idxs": drop_ids,
        "drop_features": [FEATURE_NAMES_27[i] for i in drop_ids],
        "n_params": seeds_results[0]['n_params'],
        "seeds": [r for r in seeds_results],
        "mean_f1m": float(f1ms.mean()),
        "std_f1m": float(f1ms.std(ddof=0)),
        "median_f1m": float(np.median(f1ms)),
        "min_f1m": float(f1ms.min()),
        "max_f1m": float(f1ms.max()),
        "mean_pr_auc": float(prs.mean()),
        "std_pr_auc": float(prs.std(ddof=0)),
        "mean_binary_f1": float(bfs.mean()),
        "mean_accuracy": float(accs.mean()),
        "mean_roc_auc": float(rocs.mean()),
        "n_collapsed_seeds": n_collapsed,
        "median_best_epoch": int(np.median(eps)),
        "loo_cumulative_predicted_delta": None,  # filled below
    }


# LOO 单点 ΔF1m (5-seed LOO)
LOO_5SEED = {
    1: -0.000401,   # function
    2: +0.005134,   # length
    3: +0.000266,   # setpoint
    19: +0.003517,  # press_mean_w
    20: +0.001035,  # crc_mean_w
    22: +0.000563,  # cmd_count_w
}

# 每档 LOO 加性预测
loo_cum = 0.0
for name, drop_ids in CUMULATIVE_DROPS:
    delta = sum(LOO_5SEED[i] for i in drop_ids)
    if name != "27_baseline":  # 不算累加, 只算本档的边际
        loo_cum = delta  # 累积加性 = 前几档之和 (本档 drop 集合的 LOO 加总)
    all_results[name]["loo_cumulative_predicted_delta"] = delta  # 本档 drop 集合的 LOO 加性预测 (独立 vs base 27)
    # 真正的累积 LOO 是 sum of all drops so far
    if name != "27_baseline":
        # 累计: sum of all LOO_5SEED values for current drop_ids
        cum_so_far = 0.0
        seen_so_far = set()
        for n2, dids2 in CUMULATIVE_DROPS:
            if n2 == name:
                break
            seen_so_far.update(dids2)
        all_results[name]["loo_cumulative_predicted_delta"] = sum(LOO_5SEED[i] for i in seen_so_far)


# ===== 输出对比表 =====
print("\n" + "=" * 130)
print("=== 7 档累积删除对比表 (5-seed) ===")
print("=" * 130)
header = f"{'config':<60} {'n':>3}  {'F1m_mean':>9} {'F1m_std':>8} {'median':>7} {'PR_mean':>8} {'LOO_cum_Δ':>10} {'实测ΔF1m':>10} {'Δ(actual-LOO)':>13} {'塌缩':>5}"
print(header)
print("-" * 130)

BASE_F1M = all_results["27_baseline"]["mean_f1m"]
for name, _ in CUMULATIVE_DROPS:
    r = all_results[name]
    n = r["n_features"]
    f1m = r["mean_f1m"]
    std = r["std_f1m"]
    med = r["median_f1m"]
    pr = r["mean_pr_auc"]
    loo = r["loo_cumulative_predicted_delta"] if n < 27 else 0.0
    delta_f = f1m - BASE_F1M
    diff = delta_f - loo if n < 27 else 0.0
    ncol = r["n_collapsed_seeds"]
    print(f"{name:<60} {n:>3}  {f1m:>9.4f} {std:>8.4f} {med:>7.4f} {pr:>8.4f} "
          f"{loo:>+10.4f} {delta_f:>+10.4f} {diff:>+13.4f} {ncol:>5}")
print("=" * 130)


# 完整配置文件 (JSON)
out_json = os.path.join(BASE, "train_tcn_cumulative_deletion_results.json")
out_data = {
    "configs": [{k: v for k, v in r.items() if k != "seeds"} for r in all_results.values()],
    "configs_full": all_results,
    "baseline_27_f1m": BASE_F1M,
    "loo_5seed_deltas": LOO_5SEED,
    "cumulative_drop_order": [
        {"step": i+1, "drop_idx": did, "drop_feature": FEATURE_NAMES_27[did], "loo_5seed_delta": LOO_5SEED[did]}
        for i, did in enumerate([1, 3, 22, 20, 19, 2])
    ],
}
with open(out_json, "w") as f:
    json.dump(out_data, f, indent=2, default=str)
log(f"\nSaved: {out_json}")
log(f"Total time: {(time.time()-t0)/60:.1f} minutes")


# CSV (单行 per config)
import csv
out_csv = os.path.join(BASE, "train_tcn_cumulative_deletion_results.csv")
with open(out_csv, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["config", "n_features", "drop_idxs", "drop_features",
                 "mean_f1m", "std_f1m", "median_f1m", "min_f1m", "max_f1m",
                 "mean_pr_auc", "std_pr_auc", "mean_binary_f1", "mean_accuracy", "mean_roc_auc",
                 "n_collapsed_seeds", "median_best_epoch",
                 "loo_cumulative_predicted_delta_f1m",
                 "delta_f1m_vs_27baseline", "actual_minus_loo"])
    for name, _ in CUMULATIVE_DROPS:
        r = all_results[name]
        n = r["n_features"]
        delta_f = r["mean_f1m"] - BASE_F1M
        loo = r["loo_cumulative_predicted_delta"]
        diff = delta_f - loo
        wr.writerow([
            name, n,
            ";".join(str(i) for i in r["drop_idxs"]),
            ";".join(r["drop_features"]),
            r["mean_f1m"], r["std_f1m"], r["median_f1m"], r["min_f1m"], r["max_f1m"],
            r["mean_pr_auc"], r["std_pr_auc"], r["mean_binary_f1"], r["mean_accuracy"], r["mean_roc_auc"],
            r["n_collapsed_seeds"], r["median_best_epoch"],
            loo, delta_f, diff,
        ])
log(f"CSV: {out_csv}")
