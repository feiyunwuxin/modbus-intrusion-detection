#!/usr/bin/env python3
"""TCN+SE v2 SCADA 27 维 **5-seed Leave-One-Out (LOO)** 特征消融实验

每个变体跑 5 seeds (42, 123, 456, 789, 1024) 取平均
28 变体 x 5 seeds = 140 次训练 ~80 分钟

用法:
  python ablation_tcn_v2_loo_5seed.py              # 跑全部 28 x 5 = 140
  python ablation_tcn_v2_loo_5seed.py baseline     # 只跑 baseline 27 维 5-seed
  python ablation_tcn_v2_loo_5seed.py 0,5,10       # 只跑 LOO index 0/5/10 (各 5-seed)
"""

import os, time, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
)

BASE = r"C:\work\Claude\Issue"
EPOCHS = 20
SEEDS = [42, 123, 456, 789, 1024]   # 5 seeds for stability
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

FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]

GROUPS_27 = ([0]*16 + [1]*3 + [2]*8)


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


def run_one(name, keep_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
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

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
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

    result = {
        "name": name, "seed": seed,
        "drop_idx": (None if len(keep_idx) == 27
                     else sorted(set(range(27)) - set(keep_idx))[0]),
        "n_features": len(keep_idx),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        "kept_idx": list(keep_idx),
    }
    return result


t0_global = time.time()
def log(msg):
    print(f"[{time.time()-t0_global:8.1f}s] {msg}", flush=True)


def main():
    # 全特征 baseline (idx=27 个,保留 0..26)
    runs = {"baseline_27dim": list(range(27))}
    # LOO:每次去掉 idx i
    for i in range(27):
        keep = [k for k in range(27) if k != i]
        runs[f"loo_{i:02d}_drop_{FEATURE_NAMES_27[i].replace(' ', '_')}"] = keep

    # 子集筛选
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        target = sys.argv[1]
        if target == "baseline":
            runs = {"baseline_27dim": runs["baseline_27dim"]}
        else:
            idx_list = [int(x) for x in target.split(",")]
            runs = {"baseline_27dim": runs["baseline_27dim"]}
            runs.update({k: v for k, v in runs.items() if k.startswith("loo_") and any(f"loo_{i:02d}_" in k for i in idx_list)})

    log(f"TCN+SE 5-seed LOO Ablation: {len(runs)} variants x {len(SEEDS)} seeds = {len(runs)*len(SEEDS)} runs, epoch={EPOCHS}")
    log(f"Seeds: {SEEDS}")
    log(f"预计耗时: ~{len(runs)*len(SEEDS)*35//60} 分钟")
    raw_results = []
    n_done = 0
    for name, keep_idx in runs.items():
        for seed in SEEDS:
            n_done += 1
            log(f"  >> [{n_done}/{len(runs)*len(SEEDS)}] {name} seed={seed}")
            r = run_one(name, keep_idx, seed)
            raw_results.append(r)
            log(f"     F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
                f"Epoch={r['best_epoch']}  Time={(time.time()-t0_global):.0f}s")

    # 5-seed 聚合
    log("\n=== 5-seed 聚合 (按变体取 mean ± std) ===")
    aggregated = []
    variants = {}
    for r in raw_results:
        nm = r['name']
        if nm not in variants:
            variants[nm] = []
        variants[nm].append(r)

    for name, rs in variants.items():
        f1ms = np.array([r['test_macro_f1'] for r in rs])
        prs = np.array([r['test_pr_auc'] for r in rs])
        agg = {
            "name": name,
            "drop_idx": rs[0]['drop_idx'],
            "drop_feature": (None if rs[0]['drop_idx'] is None else FEATURE_NAMES_27[rs[0]['drop_idx']]),
            "group": (None if rs[0]['drop_idx'] is None else ["v1_raw", "v2_row", "v2_window"][GROUPS_27[rs[0]['drop_idx']]]),
            "n_features": rs[0]['n_features'],
            "n_params": rs[0]['n_params'],
            "n_seeds": len(rs),
            "mean_macro_f1": float(f1ms.mean()),
            "std_macro_f1": float(f1ms.std()),
            "min_macro_f1": float(f1ms.min()),
            "max_macro_f1": float(f1ms.max()),
            "mean_pr_auc": float(prs.mean()),
            "std_pr_auc": float(prs.std()),
            "all_f1ms": f1ms.tolist(),
            "all_pr_aucs": prs.tolist(),
        }
        aggregated.append(agg)

    # 排序 (按 mean F1m 升序, 看哪些 drop 让 F1m 升)
    base_agg = next(a for a in aggregated if a['name'] == 'baseline_27dim')
    base_f1m = base_agg['mean_macro_f1']
    base_pr  = base_agg['mean_pr_auc']
    log(f"\nBaseline 27-dim 5-seed mean: F1m={base_f1m:.4f} (±{base_agg['std_macro_f1']:.4f}) PR-AUC={base_pr:.4f}")
    log(f"\n{'='*120}")
    log(f"{'name':<48} {'drop':<28} {'group':<11} {'F1m mean±std':<18} {'ΔF1m':<8} {'PR-AUC mean':<14} {'ΔPR':<8}")
    log(f"{'-'*120}")
    for a in sorted(aggregated, key=lambda x: x['mean_macro_f1']):
        delta_f = a['mean_macro_f1'] - base_f1m
        delta_p = a['mean_pr_auc'] - base_pr
        log(f"{a['name']:<48} {(a['drop_feature'] or '-'):<28} {(a['group'] or '-'):<11} "
            f"{a['mean_macro_f1']:.4f}±{a['std_macro_f1']:.4f}      {delta_f:>+7.4f} "
            f"{a['mean_pr_auc']:.4f}         {delta_p:>+7.4f}")
    log(f"{'='*120}")

    # 保存
    out_csv = os.path.join(BASE, "ablation_tcn_v2_loo_5seed_results.csv")
    df = pd.DataFrame([{
        "name": a['name'],
        "drop_idx": a['drop_idx'],
        "drop_feature": a['drop_feature'],
        "group": a['group'],
        "n_features": a['n_features'],
        "n_params": a['n_params'],
        "n_seeds": a['n_seeds'],
        "mean_macro_f1": a['mean_macro_f1'],
        "std_macro_f1": a['std_macro_f1'],
        "min_macro_f1": a['min_macro_f1'],
        "max_macro_f1": a['max_macro_f1'],
        "mean_pr_auc": a['mean_pr_auc'],
        "std_pr_auc": a['std_pr_auc'],
        "delta_f1m_vs_baseline": a['mean_macro_f1'] - base_f1m,
        "delta_pr_vs_baseline": a['mean_pr_auc'] - base_pr,
    } for a in aggregated])
    df.to_csv(out_csv, index=False)

    out_json = os.path.join(BASE, "ablation_tcn_v2_loo_5seed_results.json")
    with open(out_json, "w") as f:
        json.dump({"raw": raw_results, "aggregated": aggregated,
                   "baseline_mean_f1m": base_f1m, "baseline_mean_pr": base_pr,
                   "seeds": SEEDS}, f, indent=2, default=str)

    log(f"\nResults saved:")
    log(f"  CSV: {out_csv}")
    log(f"  JSON: {out_json}")
    log(f"Total time: {(time.time()-t0_global)/60:.1f} minutes")


if __name__ == "__main__":
    main()