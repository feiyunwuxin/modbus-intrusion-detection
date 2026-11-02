#!/usr/bin/env python3
"""重训 TCN+SE 23-dim 冠军配置 (B=64 + LR=4e-3 + ep=20 + ch=32 + do=0.1) × 5 seeds + 保存 .pt

6 轮 sweep 总冠军架构导出:
  - 23-dim (drops length, setpoint, crc_mean_w, cmd_count_w)
  - B=64, LR=4e-3, ep=20, ch=32 (Pareto 拐点), do=0.1
  - 3 TCN blocks, dilations=[1,2,4], SE r=8, k=3
  - 20,877 params (vs ch=64 73,945, -71.7%)
  - ch=32 F1m 0.8647 +/- 0.0077 (vs ch=64 0.8732, -0.0085)

输出:
  - model_tcn_23dim_b64_ch32_do01_window16_s{seed}.pt × 5 (每个 seed 单独 .pt)
  - processed_meta_tcn_23dim_b64_ch32_do01_window16.json (5-seed 汇总)
  - predictions_test_tcn_23dim_b64_ch32_do01_window16.csv (test 集预测)

.pt 格式与历史 model_tcn_v4_se_*.pt 完全一致:
  {
    "state_dict": {60 个 tensors},
    "n_features": 23,
    "window": 16,
    "n_blocks": 3,
    "channels": 32,
    "config": {...},  # 完整超参
    "seed": int,
    "test_metrics": {...}
  }
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"D:\workspace\claude\Issue\Issue"
BATCH = 64
LR = 4e-3
EPOCHS = 20
CHANNELS = 32                # 改为 32 (vs 本工作 6 轮默认 ch=64)
DROPOUT = 0.1
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5
CLIP_VAL = 10.0
N_BLOCKS = 3
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
SE_REDUCTION = 8
WINDOW = 16

# 23-dim (64-combos 冠军版) — drops {2, 3, 20, 22}
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
N_FEATURES = len(KEEP_23)
SEEDS = [42, 123, 456, 789, 1024]

TAG = "v4_se_23dim_b64_ch32_do01_window16"


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
    def __init__(self, in_ch, channels=CHANNELS, n_blocks=N_BLOCKS, kernel_size=KERNEL_SIZE,
                 dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb)).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_one(seed: int):
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
        history.append({"epoch": epoch, "val_f1m": float(val_f1m), "epoch_time_s": ep_t,
                        "current_lr": float(optimizer.param_groups[0]["lr"])})
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
    val_best_prob, val_best_lbl = predict_probs(model, val_loader)

    # 准备 .pt 保存 (与历史 model_tcn_v4_se_*.pt 格式一致)
    pt_payload = {
        "state_dict": best_state,
        "n_features": N_FEATURES,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "channels": CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dilations": DILATIONS,
        "se_reduction": SE_REDUCTION,
        "dropout": DROPOUT,
        "config": {
            "batch": BATCH, "lr": LR, "epochs": EPOCHS, "wd": WD,
            "patience": PATIENCE, "grad_clip": GRAD_CLIP, "clip_val": CLIP_VAL,
            "drop_features": {2: "length", 3: "setpoint", 20: "crc_mean_w", 22: "cmd_count_w"},
        },
        "seed": int(seed),
        "n_params": n_params,
        "best_epoch": int(best_epoch),
        "actual_epochs_run": len(history),
        "early_stopped": bool(epochs_no_improve >= PATIENCE),
        "best_val_f1m": float(best_val_f1m),
        "train_time_s": float(train_time),
        "test_metrics": {
            "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
            "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
            "test_accuracy": float((test_pred == test_lbl).mean()),
            "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
            "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        },
        "history": history,
    }

    # 保存 .pt
    pt_path = os.path.join(BASE, f"model_{TAG}_s{seed}.pt")
    torch.save(pt_payload, pt_path)
    print(f"[saved pt] {pt_path}  ({n_params:,} params, {os.path.getsize(pt_path):,} bytes)")

    # 保存测试集预测 CSV (与历史 processed_meta 模式一致)
    pred_df_path = os.path.join(BASE, f"predictions_test_{TAG}_s{seed}.csv")
    import csv
    with open(pred_df_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack"])
        for y, p in zip(test_lbl.tolist(), test_prob.tolist()):
            w.writerow([y, p])
    print(f"[saved pred] {pred_df_path}")

    return {
        "seed": int(seed),
        "n_params": n_params,
        "best_epoch": int(best_epoch),
        "actual_epochs_run": len(history),
        "early_stopped": bool(epochs_no_improve >= PATIENCE),
        "best_val_f1m": float(best_val_f1m),
        "train_time_s": float(train_time),
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        "pt_path": pt_path,
        "pt_size_bytes": os.path.getsize(pt_path),
    }


def main():
    print(f"=== Retrain TCN+SE 23-dim (B=64 + LR=4e-3 + ep=20 + ch={CHANNELS} + do={DROPOUT}) × 5 seeds ===")
    print(f"    保存到 model_{TAG}_s{{seed}}.pt")

    all_results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        r = run_one(seed)
        print(f"  F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
              f"BestEp={r['best_epoch']}/{r['actual_epochs_run']}  Params={r['n_params']:,}  "
              f"TrainT={r['train_time_s']:.1f}s")
        all_results.append(r)

    # 汇总
    f1m_mean = float(np.mean([r["test_macro_f1"] for r in all_results]))
    f1m_std  = float(np.std([r["test_macro_f1"] for r in all_results], ddof=0))
    pr_mean  = float(np.mean([r["test_pr_auc"] for r in all_results]))
    pr_std   = float(np.std([r["test_pr_auc"] for r in all_results], ddof=0))

    # 汇总 JSON (与历史 processed_meta_tcn_*.json 格式一致)
    meta = {
        "model": "TCN+SE",
        "tag": TAG,
        "config": {
            "batch": BATCH, "lr": LR, "epochs": EPOCHS, "channels": CHANNELS,
            "dropout": DROPOUT, "wd": WD, "patience": PATIENCE,
            "n_blocks": N_BLOCKS, "kernel_size": KERNEL_SIZE,
            "dilations": DILATIONS, "se_reduction": SE_REDUCTION,
            "window": WINDOW, "n_features": N_FEATURES,
            "drop_features": {2: "length", 3: "setpoint", 20: "crc_mean_w", 22: "cmd_count_w"},
        },
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "n_models": len(SEEDS),
        "per_seed": all_results,
        "summary": {
            "f1m_mean": f1m_mean, "f1m_std": f1m_std,
            "pr_mean": pr_mean, "pr_std": pr_std,
        },
    }
    meta_path = os.path.join(BASE, f"processed_meta_{TAG}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[saved meta] {meta_path}")

    # 5-seed 概率平均 ensemble
    print("\n=== 5-seed prob mean ensemble ===")
    test_probs_all = []
    test_lbls = None
    for r in all_results:
        # 重新加载 .pt 跑 test
        ckpt = torch.load(r["pt_path"], map_location="cpu", weights_only=False)
        model = TCNClassifierSE(in_ch=N_FEATURES)
        model.load_state_dict(ckpt["state_dict"])
        X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
        X_test = np.clip(X_test, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
        y_test = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                            batch_size=BATCH, shuffle=False, num_workers=0)
        probs, lbls = predict_probs(model, loader)
        test_probs_all.append(probs)
        test_lbls = lbls
    ensemble_prob = np.mean(test_probs_all, axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    ens_f1m = float(f1_score(test_lbls, ensemble_pred, average="macro"))
    ens_pr = float(average_precision_score(test_lbls, ensemble_prob))
    ens_f1b = float(f1_score(test_lbls, ensemble_pred, average="binary"))
    print(f"  5-seed ensemble F1m = {ens_f1m:.4f}  PR-AUC = {ens_pr:.4f}  BinF1 = {ens_f1b:.4f}")

    # 更新 meta 加 ensemble 指标
    meta["ensemble_metrics"] = {
        "test_at_0.5": {
            "macro_f1": ens_f1m, "binary_f1": ens_f1b,
            "pr_auc": ens_pr,
            "roc_auc": float(roc_auc_score(test_lbls, ensemble_prob)),
            "accuracy": float((ensemble_pred == test_lbls).mean()),
        },
        "n_models": 5,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # 5-seed ensemble .pt
    print(f"\n=== 5-seed ensemble F1m = {ens_f1m:.4f} ===")
    print(f"    vs single seed mean = {f1m_mean:.4f}, delta = {ens_f1m - f1m_mean:+.4f}")
    print(f"    单文件大小: {[r['pt_size_bytes'] for r in all_results]}")


if __name__ == "__main__":
    main()
