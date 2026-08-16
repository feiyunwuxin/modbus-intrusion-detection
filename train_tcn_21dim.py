#!/usr/bin/env python3
"""21 维 = 27 - {function, length, setpoint, press_mean_w, crc_mean_w, cmd_count_w}

依据 [[project-tcn-v2-loo-5seed-correction]] 5-seed LOO 单点结果:
  drop idx 1  function         DeltaF1m = -0.000401 (essentially neutral)
  drop idx 2  length           DeltaF1m = +0.005134 ✅ single biggest positive LOO
  drop idx 3  setpoint         DeltaF1m = +0.000266 (essentially neutral)
  drop idx 19 press_mean_w     DeltaF1m = +0.003517 ✅
  drop idx 20 crc_mean_w       DeltaF1m = +0.001035
  drop idx 22 cmd_count_w      DeltaF1m = +0.000563 (essentially neutral)

LOO 单点加总预测: DeltaF1m ≈ +0.010
但 LOO 加总预测系统性高估 +0.008~+0.035 (见 [[project-2026-07-05-daily-report]])
真实期望: DeltaF1m 略高于 26 维 (-length) 单冠军 (+0.0051, F1m=0.8256)

5 seeds = [42, 123, 456, 789, 1024]
设置 100% 沿用 LOO 5-seed baseline:
  B=512, LR=5e-4, WD=1e-5, Dropout=0.3, Epoch=20, Patience=5
  ch=64, k=3, dilations=[1,2,4], SE r=8, window=16
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

# 21 维 = 27 维 - {1 function, 2 length, 3 setpoint, 19 press_mean_w, 20 crc_mean_w, 22 cmd_count_w}
DROP = {1: "function", 2: "length", 3: "setpoint",
        19: "press_mean_w", 20: "crc_mean_w", 22: "cmd_count_w"}
KEEP_21 = [i for i in range(27) if i not in DROP]

FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]

assert len(KEEP_21) == 21, f"Expected 21 features, got {len(KEEP_21)}"
print(f"=== TCN+SE 21-dim Retrain 5-seed ===")
print(f"Drop ({len(DROP)} features): {list(DROP.values())}")
print(f"Keep ({len(KEEP_21)}): {[FEATURE_NAMES_27[i] for i in KEEP_21]}")


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
        "test_pred": test_pred.tolist(),
        "test_prob": test_prob.tolist(),
    }


t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


results = []
for seed in SEEDS:
    r = run_one(KEEP_21, seed=seed)
    log(f"  seed={seed}  F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
        f"Epoch={r['best_epoch']}  Params={r['n_params']:,}")
    results.append(r)

print("\n" + "=" * 80)
print("=== 21-dim (-function,-length,-setpoint,-press,-crc,-cmd_count) 5-seed 汇总 ===")
print("=" * 80)
print(f"{'seed':>5} {'F1m':>8} {'BinF1':>8} {'Acc':>8} {'ROC':>8} {'PR':>8} {'Epoch':>5}")
for i, r in enumerate(results):
    print(f"{SEEDS[i]:>5} {r['test_macro_f1']:>8.4f} {r['test_binary_f1']:>8.4f} "
          f"{r['test_accuracy']:>8.4f} {r['test_roc_auc']:>8.4f} {r['test_pr_auc']:>8.4f} {r['best_epoch']:>5d}")

mean_f1m = float(np.mean([r['test_macro_f1'] for r in results]))
std_f1m  = float(np.std([r['test_macro_f1'] for r in results]))
mean_pr  = float(np.mean([r['test_pr_auc'] for r in results]))
std_pr   = float(np.std([r['test_pr_auc'] for r in results]))
mean_bf1 = float(np.mean([r['test_binary_f1'] for r in results]))
mean_acc = float(np.mean([r['test_accuracy'] for r in results]))
mean_roc = float(np.mean([r['test_roc_auc'] for r in results]))
print(f"\n{'mean':>5} {mean_f1m:>8.4f} (std={std_f1m:.4f}) {mean_pr:>8.4f} (std={std_pr:.4f})")

# ===== Comparison =====
# 5-seed baseline 27-dim: F1m = 0.8205, PR-AUC = 0.9052 (from ablation_tcn_v2_loo_5seed)
# 5-seed 26-dim (-length): F1m = 0.8256, PR-AUC = 0.9079 (from train_tcn_26dim_length)
baseline_27 = {"f1m": 0.8205, "pr": 0.9052}
champion_26 = {"f1m": 0.8256, "pr": 0.9079}

print(f"\n=== 与历史对比 ===")
print(f"27-dim baseline (5-seed):    F1m={baseline_27['f1m']:.4f}  PR-AUC={baseline_27['pr']:.4f}")
print(f"26-dim (-length) champion:   F1m={champion_26['f1m']:.4f}  PR-AUC={champion_26['pr']:.4f}")
print(f"21-dim (本次):               F1m={mean_f1m:.4f}  PR-AUC={mean_pr:.4f}")
print(f"\nDeltaF1m vs 27-dim baseline: {mean_f1m - baseline_27['f1m']:+.4f}")
print(f"DeltaF1m vs 26-dim champion: {mean_f1m - champion_26['f1m']:+.4f}")
print(f"DeltaPR  vs 27-dim baseline: {mean_pr  - baseline_27['pr']:+.4f}")
print(f"DeltaPR  vs 26-dim champion: {mean_pr  - champion_26['pr']:+.4f}")

# Per-seed deltas
print(f"\n=== Per-seed vs 27-dim baseline (F1m) ===")
for i, r in enumerate(results):
    delta = r['test_macro_f1'] - baseline_27['f1m']
    print(f"  seed={SEEDS[i]}: F1m={r['test_macro_f1']:.4f}  Delta={delta:+.4f}")

# LOO 单点加总预测 vs 实际
loo_pred_delta = -0.000401 + 0.005134 + 0.000266 + 0.003517 + 0.001035 + 0.000563
print(f"\n=== LOO 单点加总预测检验 ===")
print(f"LOO 单点加总预测 DeltaF1m: {loo_pred_delta:+.4f}")
print(f"实际 DeltaF1m (mean 5-seed): {mean_f1m - baseline_27['f1m']:+.4f}")
print(f"LOO 高估: {loo_pred_delta - (mean_f1m - baseline_27['f1m']):+.4f}")

# 保存
out_json = os.path.join(BASE, "train_tcn_21dim_results.json")
with open(out_json, "w") as f:
    json.dump({"results_21dim": results,
               "mean_f1m": mean_f1m, "std_f1m": std_f1m,
               "mean_pr_auc": mean_pr, "std_pr_auc": std_pr,
               "mean_binary_f1": mean_bf1, "mean_accuracy": mean_acc, "mean_roc_auc": mean_roc,
               "drop_features": DROP, "keep_idx": KEEP_21,
               "baseline_27dim_mean_f1m": baseline_27['f1m'],
               "baseline_27dim_mean_pr": baseline_27['pr'],
               "champion_26dim_mean_f1m": champion_26['f1m'],
               "champion_26dim_mean_pr": champion_26['pr'],
               "loo_predicted_delta_f1m": loo_pred_delta}, f, indent=2)
log(f"\nSaved: {out_json}")
log(f"Total time: {(time.time()-t0)/60:.1f} minutes")
