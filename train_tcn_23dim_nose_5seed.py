#!/usr/bin/env python3
"""23 维 TCN 23-dim × {SE vs NoSE} × 5-seed 公平消融

完全复用 train_tcn_27dim_nose_5seed.py 的架构与超参,只切换 KEEP_27 -> KEEP_23
23-dim = 27-dim - {setpoint, deadband, time_since_last_same_addr_func, crc_mean_w}
        = 删除索引 [3, 6, 16, 20]
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

# 23 维: 删除 setpoint(3), deadband(6), time_since_last_same_addr_func(16), crc_mean_w(20)
KEEP_23 = [0,1,2,4,5,7,8,9,10,11,12,13,14,15,17,18,19,21,22,23,24,25,26]
DROP = {3: "setpoint", 6: "deadband", 16: "time_since_last_same_addr_func", 20: "crc_mean_w"}
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


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout,
                 use_se=True, se_reduction=SE_REDUCTION):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, se_reduction) if use_se else nn.Identity()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return F.relu(x + residual)


class TCNClassifier(nn.Module):
    def __init__(self, in_ch, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT,
                 use_se=True):
        super().__init__()
        layers = [TCNBlock(in_ch, channels, kernel_size, dilations[0], dropout, use_se=use_se)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlock(channels, channels, kernel_size, d, dropout, use_se=use_se))
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


def run_one(keep_idx, seed, use_se):
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

    model = TCNClassifier(in_ch=N_FEATURES, use_se=use_se)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    t0 = time.time()
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
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)
    return {
        "n_features": N_FEATURES, "n_params": n_params, "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1),
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
    }


def run_variant(label, use_se):
    print(f"\n{'='*80}\n=== {label} (use_se={use_se}) 5-seed ===\n{'='*80}")
    results = []
    for seed in SEEDS:
        r = run_one(KEEP_23, seed=seed, use_se=use_se)
        print(f"  seed={seed:>4}  F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
              f"Epoch={r['best_epoch']:>2}  Params={r['n_params']:,}  Time={r['train_time_s']}s", flush=True)
        results.append(r)
    return results


def summarize(label, results):
    print(f"\n--- {label} ---")
    print(f"{'seed':>5} {'F1m':>8} {'BinF1':>8} {'Acc':>8} {'ROC':>8} {'PR':>8} {'Epoch':>5}")
    for i, r in enumerate(results):
        print(f"{SEEDS[i]:>5} {r['test_macro_f1']:>8.4f} {r['test_binary_f1']:>8.4f} "
              f"{r['test_accuracy']:>8.4f} {r['test_roc_auc']:>8.4f} {r['test_pr_auc']:>8.4f} {r['best_epoch']:>5d}")
    mf1 = np.mean([r['test_macro_f1'] for r in results])
    sf1 = np.std([r['test_macro_f1'] for r in results])
    mpr = np.mean([r['test_pr_auc'] for r in results])
    spr = np.std([r['test_pr_auc'] for r in results])
    print(f"{'mean':>5} {mf1:>8.4f} (std={sf1:.4f}) {mpr:>8.4f} (std={spr:.4f})")
    return {"mean_f1m": float(mf1), "std_f1m": float(sf1),
            "mean_pr_auc": float(mpr), "std_pr_auc": float(spr)}


if __name__ == "__main__":
    print(f"=== TCN 23-dim × SE/NoSE 5-seed 公平消融 ===")
    print(f"23-dim 删除: {DROP}")
    print(f"WINDOW=16, blocks=3, ch=64, dilations=[1,2,4], LR=5e-4, WD=1e-5, dropout=0.3")
    print(f"Seeds: {SEEDS}")

    res_se    = run_variant("23-dim WITH SE",   use_se=True)
    res_nose  = run_variant("23-dim WITHOUT SE", use_se=False)

    sum_se   = summarize("23-dim WITH SE",    res_se)
    sum_nose = summarize("23-dim WITHOUT SE", res_nose)

    delta = {
        "delta_f1m_mean":    sum_se["mean_f1m"]    - sum_nose["mean_f1m"],
        "delta_pr_auc_mean": sum_se["mean_pr_auc"] - sum_nose["mean_pr_auc"],
        "delta_f1m_std":     sum_se["std_f1m"]     - sum_nose["std_f1m"],
    }
    print("\n" + "="*80)
    print("=== 23-dim: SE 模块的边际贡献 ===")
    print("="*80)
    print(f"  Δ F1m   = {delta['delta_f1m_mean']:+.4f}  (SE vs NoSE)")
    print(f"  Δ PR-AUC= {delta['delta_pr_auc_mean']:+.4f}")
    print(f"  Δ F1m_std= {delta['delta_f1m_std']:+.4f}  (负值=更稳定)")
    print(f"  Params  = {res_se[0]['n_params']:,} (SE) vs {res_nose[0]['n_params']:,} (NoSE)")
    print(f"  Δ Params = {res_se[0]['n_params'] - res_nose[0]['n_params']:+,}")

    out = {
        "config": {
            "data": "23-dim SCADA v2 (drop idx 3,6,16,20), window=16",
            "drop_features": DROP,
            "model": "TCN(3 blocks, ch=64, k=3, dilations=[1,2,4])",
            "hparams": {"LR": LR, "WD": WD, "dropout": DROPOUT,
                        "batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE},
            "seeds": SEEDS,
        },
        "with_se":    {"runs": res_se,    **sum_se,    "params": res_se[0]['n_params']},
        "without_se": {"runs": res_nose,  **sum_nose,  "params": res_nose[0]['n_params']},
        "delta_se_vs_nose": delta,
    }
    out_path = os.path.join(BASE, "train_tcn_23dim_nose_5seed_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")