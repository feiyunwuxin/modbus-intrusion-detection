#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCN (no SE) 23-dim 超参扫描 (5 维度 × 5 seeds = 125 runs)

用法:
    # 单跑
    python train_tcn_nose_sweep.py --dim dropout --value 0.2 --seed 42

    # 全跑 (一维度)
    python train_tcn_nose_sweep.py --dim dropout --sweep

    # 全跑 (5 维度)
    for dim in batch ch dropout epochs lr; do
        python train_tcn_nose_sweep.py --dim $dim --sweep
    done

    # 全跑 (一键)
    python train_tcn_nose_sweep.py --all  # 25 runs per dim × 5 dim = 125 runs

固定 baseline (默认),可用 --override 调整:
    BATCH=64, LR=5e-4, EPOCHS=20, CHANNELS=64, DROPOUT=0.3

5 维度扫描:
  - batch:   [64, 96, 128, 192, 256]
  - ch:      [16, 32, 48, 64, 96]   # channels
  - dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
  - epochs:  [20, 30, 40, 50, 60]
  - lr:      [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3]

每个 (dim, value, seed) 一次训练,产出 metrics dict → JSON
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"

# ==== Fixed baseline (与 TCN+SE sweep 一致,便于直接对比) ====
BATCH = 64
LR = 5e-4
EPOCHS = 20
CHANNELS = 64
DROPOUT = 0.3
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5
CLIP_VAL = 10.0
N_BLOCKS = 3
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
WINDOW = 16

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
N_FEATURES = len(KEEP_23)

# ==== Sweep grids ====
GRIDS = {
    'batch':   [64, 96, 128, 192, 256],
    'ch':      [16, 32, 48, 64, 96],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'epochs':  [20, 30, 40, 50, 60],
    'lr':      [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3],
}
SEEDS = [42, 123, 456, 789, 1024]


# ==== TCN (NoSE) 模型 ====
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        # ★ No SE Block (与 train_tcn_23dim_nose_5seed.py 一致)
        return F.relu(x + residual)


class TCNClassifier(nn.Module):
    def __init__(self, in_ch, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = [TCNBlock(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlock(channels, channels, kernel_size, d, dropout))
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


def load_data():
    X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val   = np.clip(X_val, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test  = np.clip(X_test, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_one(seed, batch=BATCH, lr=LR, epochs=EPOCHS, channels=CHANNELS, dropout=DROPOUT):
    torch.manual_seed(seed); np.random.seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=batch, shuffle=False, num_workers=0)

    model = TCNClassifier(in_ch=N_FEATURES, channels=channels, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
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


def save_partial(dim, value, seed, result):
    """保存单次运行结果 (partial)"""
    val_str = str(value).replace('.', '_').replace('e-0', 'e_').replace('-', '_')
    path = os.path.join(BASE, f"train_tcn_nose_sweep_partial_{dim}_{val_str}_s{seed}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path


def summarize(dim):
    """读所有 partial,聚合 summary_by_<dim>"""
    val_strs = {k: [str(v).replace('.', '_').replace('e-0', 'e_').replace('-', '_') for v in vs] for k, vs in GRIDS.items()}
    summary_key = f"summary_by_{dim}"
    summary = {}
    all_runs = []
    for val in GRIDS[dim]:
        runs = []
        for seed in SEEDS:
            v_str = str(val).replace('.', '_').replace('e-0', 'e_').replace('-', '_')
            path = os.path.join(BASE, f"train_tcn_nose_sweep_partial_{dim}_{v_str}_s{seed}.json")
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    r = json.load(f)
                r['seed'] = seed
                if dim == 'batch': r['batch'] = val
                elif dim == 'ch': r['channels'] = val
                elif dim == 'dropout': r['dropout'] = val
                elif dim == 'epochs': r['epochs'] = val
                elif dim == 'lr': r['lr'] = val
                runs.append(r)
                all_runs.append(r)
        if runs:
            f1m = [r['test_macro_f1'] for r in runs]
            binf1 = [r['test_binary_f1'] for r in runs]
            pr = [r['test_pr_auc'] for r in runs]
            acc = [r['test_accuracy'] for r in runs]
            roc = [r['test_roc_auc'] for r in runs]
            ep = [r['best_epoch'] for r in runs]
            t = [r['train_time_s'] for r in runs]
            summary[str(val)] = {
                'n_seeds': len(runs),
                'f1m_mean': float(np.mean(f1m)), 'f1m_std': float(np.std(f1m)),
                'binf1_mean': float(np.mean(binf1)), 'binf1_std': float(np.std(binf1)),
                'pr_mean': float(np.mean(pr)), 'pr_std': float(np.std(pr)),
                'acc_mean': float(np.mean(acc)),
                'roc_mean': float(np.mean(roc)),
                'best_epoch_mean': float(np.mean(ep)),
                'train_time_s_mean': float(np.mean(t)),
                'n_params': runs[0].get('n_params'),
            }
    out_path = os.path.join(BASE, f"train_tcn_nose_sweep_results_{dim}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'n_features': 23,
                'use_se': False,  # ★ NoSE
                'baseline': {'B': 64, 'ch': 64, 'do': 0.3, 'ep': 20, 'lr': '5e-4'},
                'dim': dim,
                'grid': GRIDS[dim],
                'seeds': SEEDS,
            },
            summary_key: summary,
            'all_runs': all_runs,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', choices=list(GRIDS.keys()) + ['all'])
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()

    if args.all:
        for dim in GRIDS.keys():
            print(f"\n{'='*80}\n=== Sweeping {dim} ===\n{'='*80}")
            for val in GRIDS[dim]:
                for seed in SEEDS:
                    kwargs = {dim if dim != 'ch' else 'channels': val}
                    print(f"  dim={dim} val={val} seed={seed} ...", end=' ', flush=True)
                    r = run_one(seed, **kwargs)
                    save_partial(dim, val, seed, r)
                    print(f"F1m={r['test_macro_f1']:.4f} BinF1={r['test_binary_f1']:.4f}")
            summarize(dim)
        return

    if args.summarize:
        summarize(args.dim)
        return

    if args.sweep:
        for val in GRIDS[args.dim]:
            for seed in SEEDS:
                kwargs = {args.dim if args.dim != 'ch' else 'channels': val}
                print(f"  dim={args.dim} val={val} seed={seed} ...", end=' ', flush=True)
                r = run_one(seed, **kwargs)
                save_partial(args.dim, val, seed, r)
                print(f"F1m={r['test_macro_f1']:.4f} BinF1={r['test_binary_f1']:.4f}")
        summarize(args.dim)
        return

    # 单跑
    val = float(args.value) if args.value.replace('.','').replace('-','').isdigit() or '.' in args.value else args.value
    if args.dim == 'lr':
        val = float(args.value)
    elif args.dim in ('batch', 'epochs'):
        val = int(args.value)
    elif args.dim == 'ch':
        val = int(args.value)
    elif args.dim == 'dropout':
        val = float(args.value)
    kwargs = {args.dim if args.dim != 'ch' else 'channels': val}
    r = run_one(args.seed, **kwargs)
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()