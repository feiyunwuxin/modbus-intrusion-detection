#!/usr/bin/env python3
"""TCN+SE 23-dim ch=32 × window in {4, 8} × 5-seed sweep.

完全复用 retrain_tcn_23dim_b64_ch32_do01_savept.py 的架构/超参:
  - 23-dim (drops length, setpoint, crc_mean_w, cmd_count_w)
  - B=64, LR=4e-3, EPOCHS=20, CHANNELS=32, DROPOUT=0.1
  - 3 TCN blocks, dilations=[1,2,4], SE r=8, k=3
  - Seeds [42, 123, 456, 789, 1024]

唯一变化: WINDOW ∈ {4, 8} (vs 原 window=16)
数据: make_window4_window8.py 重新生成的 _window{4,8}.npy

输出:
  - per-seed .pt 与 predictions_test_*.csv
  - tcn_se_ch32_w4_w8_5seed_summary.json (含 4 指标 + 5-seed prob-mean ensemble)
"""
import os, sys, time, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"
BATCH = 64
LR = 4e-3
EPOCHS = 20
CHANNELS = 32
DROPOUT = 0.1
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5
CLIP_VAL = 10.0
N_BLOCKS = 3
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
SE_REDUCTION = 8

# 跟 retrain_tcn_23dim_b64_ch32_do01_savept.py 一致 (drops {2, 3, 20, 22})
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
N_FEATURES = len(KEEP_23)
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


def load_data(W):
    Xtr = np.load(os.path.join(BASE, f"X_train_binary_v2_scada_window{W}.npy")).astype(np.float32)[:, :, KEEP_23]
    Xva = np.load(os.path.join(BASE, f"X_val_binary_v2_scada_window{W}.npy")).astype(np.float32)[:, :, KEEP_23]
    Xte = np.load(os.path.join(BASE, f"X_test_binary_v2_scada_window{W}.npy")).astype(np.float32)[:, :, KEEP_23]
    ytr = np.load(os.path.join(BASE, f"y_train_binary_v2_scada_window{W}.npy")).astype(np.int64)
    yva = np.load(os.path.join(BASE, f"y_val_binary_v2_scada_window{W}.npy")).astype(np.int64)
    yte = np.load(os.path.join(BASE, f"y_test_binary_v2_scada_window{W}.npy")).astype(np.int64)
    Xtr = np.clip(Xtr, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    Xva = np.clip(Xva, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    Xte = np.clip(Xte, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    return Xtr, ytr, Xva, yva, Xte, yte


def per_class_prf(y_true, y_pred, cls):
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == cls) & (y_pred == cls)))
    fp = int(np.sum((y_true != cls) & (y_pred == cls)))
    fn = int(np.sum((y_true == cls) & (y_pred != cls)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, int(np.sum(y_true == cls))


def all_metrics(y_true, y_pred):
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    p_n, r_n, f1_n, sup_n = per_class_prf(y_true, y_pred, 0)
    p_a, r_a, f1_a, sup_a = per_class_prf(y_true, y_pred, 1)
    return {"Accuracy": acc, "Precision": p_a, "Recall": r_a, "F1-Score": f1_a,
            "Macro_F1": (f1_n + f1_a) / 2,
            "Normal_Support": sup_n, "Attack_Support": sup_a}


def run_one(W, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xva, yva, Xte, yte = load_data(W)
    n_windows = len(Xtr); iters_per_ep = (n_windows + BATCH - 1) // BATCH

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                              batch_size=BATCH, shuffle=False, num_workers=0)

    model = TCNClassifierSE(in_ch=N_FEATURES)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    t0 = time.time()
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
        print(f"    [W={W} s={seed} ep={epoch:>2}] val_f1m={val_f1m:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  ep_t={ep_t:.1f}s", flush=True)
        if val_f1m > best_f1m:
            best_f1m = val_f1m; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    [early-stop] epoch {epoch}", flush=True); break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)
    m4 = all_metrics(test_lbl, test_pred)

    # save predictions CSV (跟历史 predictions_test_*.csv 格式一致)
    tag = f"v4_se_23dim_b64_ch32_do01_window{W}"
    pred_path = os.path.join(BASE, f"predictions_test_{tag}_s{seed}.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack", "pred_05"])
        for y, p, yp in zip(test_lbl.tolist(), test_prob.tolist(), test_pred.tolist()):
            w.writerow([y, p, yp])

    # save .pt
    pt_payload = {
        "state_dict": best_state, "n_features": N_FEATURES, "window": W,
        "n_blocks": N_BLOCKS, "channels": CHANNELS, "kernel_size": KERNEL_SIZE,
        "dilations": DILATIONS, "se_reduction": SE_REDUCTION, "dropout": DROPOUT,
        "config": {"batch": BATCH, "lr": LR, "epochs": EPOCHS, "wd": WD,
                   "patience": PATIENCE, "grad_clip": GRAD_CLIP, "clip_val": CLIP_VAL,
                   "drop_features": {2: "length", 3: "setpoint", 20: "crc_mean_w", 22: "cmd_count_w"}},
        "seed": int(seed), "n_params": n_params, "best_epoch": int(best_epoch),
        "train_time_s": float(train_time),
        "test_metrics": {
            "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
            "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
            "test_accuracy": float((test_pred == test_lbl).mean()),
            "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
            "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        },
    }
    pt_path = os.path.join(BASE, f"model_{tag}_s{seed}.pt")
    torch.save(pt_payload, pt_path)

    return {"seed": int(seed), "window": int(W), "n_params": n_params,
            "best_epoch": int(best_epoch), "train_time_s": float(train_time),
            "iters_per_ep": int(iters_per_ep),
            "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
            "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
            "test_accuracy": float((test_pred == test_lbl).mean()),
            "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
            "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
            "4metrics": m4, "pred_csv": pred_path, "pt_path": pt_path}


def run_window(W, seeds=SEEDS):
    print(f"\n{'='*80}\n=== TCN+SE 23-dim ch=32 × WINDOW={W} × seeds={seeds} ===\n{'='*80}", flush=True)
    results = []
    for seed in seeds:
        print(f"\n--- W={W} seed={seed} ---", flush=True)
        r = run_one(W, seed)
        m = r["4metrics"]
        print(f"  [done] seed={seed}  F1={m['F1-Score']:.4f}  P={m['Precision']:.4f}  "
              f"R={m['Recall']:.4f}  Acc={m['Accuracy']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
              f"Time={r['train_time_s']:.1f}s", flush=True)
        results.append(r)
    return results


def ensemble_5seed(W, all_results):
    """5-seed prob-mean ensemble (跟 rerun_4metrics_5seed.json 一致)."""
    tag = f"v4_se_23dim_b64_ch32_do01_window{W}"
    test_probs = []
    test_lbls = None
    for r in all_results:
        ckpt = torch.load(r["pt_path"], map_location="cpu", weights_only=False)
        model = TCNClassifierSE(in_ch=N_FEATURES)
        model.load_state_dict(ckpt["state_dict"])
        Xte, yte = load_data(W)[4:6]
        loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                            batch_size=BATCH, shuffle=False, num_workers=0)
        probs, lbls = predict_probs(model, loader)
        test_probs.append(probs); test_lbls = lbls
    ens_prob = np.mean(test_probs, axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    m = all_metrics(test_lbls, ens_pred)
    return {
        "ensemble_prob": ens_prob, "test_lbls": test_lbls, "metrics": m,
        "n_models": len(all_results),
        "test_at_0.5": {
            "macro_f1": float(f1_score(test_lbls, ens_pred, average="macro")),
            "binary_f1": float(f1_score(test_lbls, ens_pred, average="binary")),
            "pr_auc": float(average_precision_score(test_lbls, ens_prob)),
            "roc_auc": float(roc_auc_score(test_lbls, ens_prob)),
            "accuracy": float((ens_pred == test_lbls).mean()),
        }
    }


def summarize_window(W, results, ens):
    print(f"\n=== W={W} 5-seed summary ===")
    print(f"{'seed':>5} {'F1m':>8} {'BinF1':>8} {'Acc':>8} {'ROC':>8} {'PR':>8} {'Epoch':>5}")
    for r in results:
        m = r["4metrics"]
        print(f"{r['seed']:>5} {r['test_macro_f1']:>8.4f} {r['test_binary_f1']:>8.4f} "
              f"{m['Accuracy']:>8.4f} {r['test_roc_auc']:>8.4f} {r['test_pr_auc']:>8.4f} "
              f"{r['best_epoch']:>5d}")
    f1m = [r["test_macro_f1"] for r in results]
    f1b = [r["test_binary_f1"] for r in results]
    print(f"{'mean':>5} {np.mean(f1m):>8.4f} (std={np.std(f1m):.4f}) {np.mean(f1b):>8.4f}")
    m = ens["metrics"]
    print(f"{'ens5':>5} {ens['test_at_0.5']['macro_f1']:>8.4f} {ens['test_at_0.5']['binary_f1']:>8.4f} "
          f"{m['Accuracy']:>8.4f} {ens['test_at_0.5']['roc_auc']:>8.4f} "
          f"{ens['test_at_0.5']['pr_auc']:>8.4f}")
    return {"per_seed_mean_f1m": float(np.mean(f1m)),
            "per_seed_std_f1m": float(np.std(f1m)),
            "ensemble": ens["test_at_0.5"],
            "ensemble_4metrics": ens["metrics"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="seeds to run (default: 5 seeds)")
    parser.add_argument("--windows", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--out", type=str, default="tcn_se_ch32_w4_w8_5seed_summary.json")
    args = parser.parse_args()

    summary = {}
    for W in args.windows:
        results = run_window(W, seeds=args.seeds)
        ens = ensemble_5seed(W, results)
        summ = summarize_window(W, results, ens)
        summary[f"window{W}"] = {
            "window": W,
            "n_seeds": len(args.seeds),
            "seeds": list(args.seeds),
            "per_seed": results,
            "summary": summ,
            "n_test_windows": int(len(results[0]["4metrics"]) and 0) or int(results[0]["4metrics"]["Attack_Support"] + results[0]["4metrics"]["Normal_Support"]),
            "test_attack_pct": float(results[0]["4metrics"]["Attack_Support"] / (results[0]["4metrics"]["Attack_Support"] + results[0]["4metrics"]["Normal_Support"])),
        }
        # save partial
        with open(os.path.join(BASE, args.out), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n[partial-saved] window={W} -> {args.out}")

    print(f"\n[done] {args.out}")
    return summary


if __name__ == "__main__":
    main()
