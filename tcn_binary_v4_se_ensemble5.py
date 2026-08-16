#!/usr/bin/env python3
"""
TCN+SE 19-dim SCADA — 5-seed ensemble (B=128, ep=40).

Trains 5 TCN+SE models with different random seeds, then averages their
test/val probabilities to form an ensemble. Compares against the best
single-seed model (B=128 seed=42, F1m=0.8655).

Why this should work:
  - Each seed explores different initialization + data shuffling
  - Models make uncorrelated errors → averaging cancels them out
  - Expected F1m boost: +0.005 ~ +0.015 (typical ensemble gain)
  - Cost: just 5× the training time (no extra inference cost at deployment)

Total time estimate: ~10 minutes CPU (5 × 124s + ensembling)
"""

import os, time, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]

# Data paths
X_TR = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada.npy")

# Hyperparameters (same as B=128 best config)
WINDOW       = 16
EPOCHS       = 40
BATCH        = 128
LR           = 5e-4
WD           = 1e-5
PATIENCE     = 10
GRAD_CLIP    = 0.5
DROPOUT      = 0.3
CLIP_VAL     = 10.0
N_BLOCKS     = 3
CHANNELS     = 64
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4]
SE_REDUCTION = 8

# Output paths
ENSEMBLE_TAG = "v4_se_ensemble5"
META_OUT   = os.path.join(BASE, f"processed_meta_tcn_{ENSEMBLE_TAG}.json")
EVAL_OUT   = os.path.join(BASE, f"evaluation_tcn_{ENSEMBLE_TAG}.txt")
PRED_VAL   = os.path.join(BASE, f"predictions_val_tcn_{ENSEMBLE_TAG}.csv")
PRED_TEST  = os.path.join(BASE, f"predictions_test_tcn_{ENSEMBLE_TAG}.csv")
THR_OUT    = os.path.join(BASE, f"best_threshold_tcn_{ENSEMBLE_TAG}.json")
CM_PNG     = os.path.join(BASE, f"confusion_matrix_tcn_{ENSEMBLE_TAG}.png")
PRROC_PNG  = os.path.join(BASE, f"pr_roc_tcn_{ENSEMBLE_TAG}.png")
HIST_PNG   = os.path.join(BASE, f"training_history_tcn_{ENSEMBLE_TAG}.png")


# ─────────────────────────────────────────────
# Model definition (same as B=128 baseline)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
def train_one_seed(seed, X_train_w, y_train_w, X_val_w, y_val_w,
                   X_test_w, y_test_w, N_FEATURES, train_loader,
                   val_loader, test_loader):
    """Train one model with a given seed and return predictions + metadata."""
    print(f"\n{'='*60}")
    print(f"  TRAINING SEED = {seed}")
    print(f"{'='*60}")

    # Reset RNGs for this seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # IMPORTANT: re-create DataLoader with seeded shuffle for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader_s = DataLoader(
        TensorDataset(torch.from_numpy(X_train_w), torch.from_numpy(y_train_w)),
        batch_size=BATCH, shuffle=True, num_workers=0, generator=g
    )

    model = TCNClassifierSE(in_ch=N_FEATURES)
    n_params = sum(p.numel() for p in model.parameters())

    n_pos = int((y_train_w == 1).sum())
    n_neg = int((y_train_w == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    @torch.no_grad()
    def predict_probs(loader):
        model.eval()
        probs = []
        for xb, _ in loader:
            logits = model(xb)
            probs.append(torch.sigmoid(logits).numpy())
        return np.concatenate(probs)

    t_seed_start = time.time()
    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader_s:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_loss_sum / train_n

        val_prob = predict_probs(val_loader)
        val_pred_05 = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(y_val_w, val_pred_05, average="macro")
        val_auc = roc_auc_score(y_val_w, val_prob)
        history.append((epoch, train_loss, val_f1m, val_auc))

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

    train_time = time.time() - t_seed_start
    model.load_state_dict(best_state)

    # Get final predictions
    val_prob = predict_probs(val_loader)
    test_prob = predict_probs(test_loader)

    # Evaluate single model
    test_pred_05 = (test_prob >= 0.5).astype(int)
    test_f1m = f1_score(y_test_w, test_pred_05, average="macro")
    test_pr_auc = average_precision_score(y_test_w, test_prob)
    test_binary_f1 = f1_score(y_test_w, test_pred_05, average="binary")
    test_acc = (test_pred_05 == y_test_w).mean()

    print(f"  seed={seed}  best_val_ep={best_epoch}  best_val_f1m={best_f1m:.4f}  "
          f"test_f1m={test_f1m:.4f}  test_pr_auc={test_pr_auc:.4f}  train_time={train_time:.1f}s")

    return {
        "seed": seed,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "best_val_f1m": float(best_f1m),
        "test_f1m_05": float(test_f1m),
        "test_pr_auc": float(test_pr_auc),
        "test_binary_f1": float(test_binary_f1),
        "test_acc": float(test_acc),
        "best_epoch": int(best_epoch),
        "train_time": float(train_time),
        "history": [{"epoch": h[0], "loss": h[1], "val_f1m": h[2], "val_auc": h[3]} for h in history],
    }


# ─────────────────────────────────────────────
t_start = time.time()
print(f"[config] BATCH={BATCH}  EPOCHS={EPOCHS}  seeds={SEEDS}")
print(f"[expected time] ~{len(SEEDS) * 124 // 60} min")

# Load data once (shared across all seeds)
print("\n[loading data] 19-dim SCADA ...")
X_train = np.load(X_TR).astype(np.float32)
X_val   = np.load(X_VA).astype(np.float32)
X_test  = np.load(X_TE).astype(np.float32)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)

def make_windows(X, y, win):
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    return Xw, yw

X_train_w, y_train_w = make_windows(X_train, y_train, WINDOW)
X_val_w,   y_val_w   = make_windows(X_val,   y_val,   WINDOW)
X_test_w,  y_test_w  = make_windows(X_test,  y_test,  WINDOW)
N_FEATURES = X_train_w.shape[-1]

X_train_w = np.clip(X_train_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_val_w   = np.clip(X_val_w,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_test_w  = np.clip(X_test_w,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)

# Fixed val/test loaders (no shuffling needed)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_w), torch.from_numpy(y_val_w)),
                        batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_w), torch.from_numpy(y_test_w)),
                         batch_size=BATCH, shuffle=False, num_workers=0)

# Placeholder train_loader (will be replaced per-seed inside train_one_seed)
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_w), torch.from_numpy(y_train_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# Train all seeds
all_results = []
for seed in SEEDS:
    res = train_one_seed(seed, X_train_w, y_train_w, X_val_w, y_val_w,
                         X_test_w, y_test_w, N_FEATURES, train_loader,
                         val_loader, test_loader)
    all_results.append(res)

total_time = time.time() - t_start
print(f"\n{'='*60}")
print(f"  ALL SEEDS TRAINED in {total_time:.1f}s")
print(f"{'='*60}")

# ─────────────────────────────────────────────
# Ensemble: average probabilities
# ─────────────────────────────────────────────
val_prob_ensemble  = np.mean([r["val_prob"]  for r in all_results], axis=0)
test_prob_ensemble = np.mean([r["test_prob"] for r in all_results], axis=0)

# Find best threshold on val
print("\n[ensemble] finding best threshold ...")
thresholds = np.arange(0.10, 0.91, 0.01)
sweep = []
for t in thresholds:
    pred = (val_prob_ensemble >= t).astype(int)
    f1m = f1_score(y_val_w, pred, average="macro")
    sweep.append((float(t), float(f1m)))
sweep_df = pd.DataFrame(sweep, columns=["threshold","val_macro_f1"])
best_idx  = int(sweep_df["val_macro_f1"].idxmax())
best_thr  = float(sweep_df.loc[best_idx, "threshold"])
best_vf1m = float(sweep_df.loc[best_idx, "val_macro_f1"])
f1m_at_05 = float(sweep_df.loc[(sweep_df["threshold"]-0.5).abs().idxmin(), "val_macro_f1"])
print(f"  best threshold = {best_thr:.2f}  →  val Macro-F1 = {best_vf1m:.4f}  (vs 0.50 = {f1m_at_05:.4f})")

with open(THR_OUT, "w") as f:
    json.dump({"best_threshold": best_thr, "val_macro_f1_at_best": best_vf1m,
               "val_macro_f1_at_0.5": f1m_at_05, "metric": "macro_f1"}, f, indent=2)

# Final ensemble evaluation
def evaluate(y_true, prob, thr, label):
    pred = (prob >= thr).astype(int)
    return {
        "label": label, "threshold": float(thr),
        "macro_f1":  float(f1_score(y_true, pred, average="macro")),
        "binary_f1": float(f1_score(y_true, pred, average="binary")),
        "accuracy":  float((pred == y_true).mean()),
        "roc_auc":   float(roc_auc_score(y_true, prob)),
        "pr_auc":    float(average_precision_score(y_true, prob)),
        "cm":        confusion_matrix(y_true, pred).tolist(),
    }

val_05   = evaluate(y_val_w,  val_prob_ensemble,  0.5,      "val@0.5")
val_thr  = evaluate(y_val_w,  val_prob_ensemble,  best_thr, f"val@{best_thr:.2f}")
test_05  = evaluate(y_test_w, test_prob_ensemble, 0.5,      "test@0.5")
test_thr = evaluate(y_test_w, test_prob_ensemble, best_thr, f"test@{best_thr:.2f}")

# Save predictions
pd.DataFrame({"y_true": y_val_w, "prob_attack": val_prob_ensemble,
              "pred_05": (val_prob_ensemble >= 0.5).astype(int),
              "pred_tuned": (val_prob_ensemble >= best_thr).astype(int),
              }).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": y_test_w, "prob_attack": test_prob_ensemble,
              "pred_05": (test_prob_ensemble >= 0.5).astype(int),
              "pred_tuned": (test_prob_ensemble >= best_thr).astype(int),
              }).to_csv(PRED_TEST, index=False)

# Save full meta
meta = {
    "model": f"TCN {ENSEMBLE_TAG}",
    "tag": ENSEMBLE_TAG,
    "base": "v4_se_b128_e40",
    "ablation_type": f"5-seed ensemble (B=128, ep=40, seeds={SEEDS})",
    "window": WINDOW,
    "n_features_per_step": int(N_FEATURES),
    "n_blocks": N_BLOCKS,
    "channels": CHANNELS,
    "kernel_size": KERNEL_SIZE,
    "dilations": DILATIONS,
    "se_reduction": SE_REDUCTION,
    "batch": BATCH,
    "seeds": SEEDS,
    "n_models_in_ensemble": len(SEEDS),
    "splits_windows": {
        "train_size": int(len(X_train_w)),
        "val_size":   int(len(X_val_w)),
        "test_size":  int(len(X_test_w)),
    },
    "n_params_per_model": int(sum(p.numel() for p in TCNClassifierSE(in_ch=N_FEATURES).parameters())),
    "total_train_time_seconds": float(total_time),
    "best_threshold": best_thr,
    "per_seed_metrics": [
        {k: v for k, v in r.items() if k not in ("val_prob", "test_prob")}
        for r in all_results
    ],
    "metrics": {
        "val_at_0.5":  {k: val_05[k]   for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_0.5": {k: test_05[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
print()
print("=" * 72)
print(f"  PER-SEED RESULTS")
print("=" * 72)
print(f"{'Seed':>8} {'BestValEp':>10} {'Val F1m':>10} {'Test F1m@0.5':>14} {'Test PR-AUC':>13} {'Time(s)':>10}")
print("-" * 72)
for r in all_results:
    print(f"{r['seed']:>8} {r['best_epoch']:>10} {r['best_val_f1m']:>10.4f} "
          f"{r['test_f1m_05']:>14.4f} {r['test_pr_auc']:>13.4f} {r['train_time']:>10.1f}")

print()
print("=" * 72)
print(f"  ENSEMBLE (5 seeds averaged)")
print("=" * 72)
print(f"Best threshold: {best_thr:.2f}")
print(f"Test @ 0.5  : Macro-F1={test_05['macro_f1']:.4f}  PR-AUC={test_05['pr_auc']:.4f}  Bin-F1={test_05['binary_f1']:.4f}")
print(f"Test @ best : Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Bin-F1={test_thr['binary_f1']:.4f}")
print()
print(f"vs single B=128 (seed=42): ΔMacro-F1 = {test_thr['macro_f1'] - 0.8655:+.4f}  ΔPR-AUC = {test_thr['pr_auc'] - 0.9122:+.4f}")
print(f"vs original  B=512 (seed=42): ΔMacro-F1 = {test_thr['macro_f1'] - 0.8444:+.4f}  ΔPR-AUC = {test_thr['pr_auc'] - 0.8892:+.4f}")
print(f"Total train time: {total_time:.1f}s  (5 seeds × ~124s)")
print("=" * 72)

# Save eval report
lines = [
    "=" * 72,
    f"EVALUATION REPORT — TCN {ENSEMBLE_TAG} (5-seed ensemble)",
    "=" * 72,
    f"Architecture: 5 × TCN+SE (B=128, ep=40), probability-averaged",
    f"Seeds: {SEEDS}",
    f"Per-model params: {sum(p.numel() for p in TCNClassifierSE(in_ch=N_FEATURES).parameters()):,}",
    f"Total ensemble params (no sharing): {5 * sum(p.numel() for p in TCNClassifierSE(in_ch=N_FEATURES).parameters()):,}",
    f"Train time (all 5 models): {total_time:.1f}s",
    f"Features per step: {N_FEATURES}  |  Window: {WINDOW}",
    f"Tuned threshold: {best_thr:.2f}",
    "",
    "Per-seed summary:",
]
for r in all_results:
    lines.append(f"  seed={r['seed']}: best_val_ep={r['best_epoch']}, "
                 f"val_f1m={r['best_val_f1m']:.4f}, test_f1m@0.5={r['test_f1m_05']:.4f}, "
                 f"test_pr_auc={r['test_pr_auc']:.4f}")
lines.append("")
for ev in [val_05, val_thr, test_05, test_thr]:
    lines.append("-" * 72)
    lines.append(f"[{ev['label']}]  thr={ev['threshold']:.2f}  "
                 f"Macro-F1={ev['macro_f1']:.4f}  Binary-F1={ev['binary_f1']:.4f}  "
                 f"Acc={ev['accuracy']:.4f}  ROC-AUC={ev['roc_auc']:.4f}  PR-AUC={ev['pr_auc']:.4f}")
    lines.append(f"Confusion matrix: {ev['cm']}")
    lines.append("")

with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # CM
    cm = np.array(test_thr["cm"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"]); ax.set_yticklabels(["Normal", "Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="red" if cm[i, j] < cm.max()/2 else "white", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{ENSEMBLE_TAG} Test CM @ thr={best_thr:.2f}  (Macro-F1={test_thr['macro_f1']:.4f})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)

    # PR/ROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for prob, yy, lbl in [(val_prob_ensemble, y_val_w, "val"), (test_prob_ensemble, y_test_w, "test")]:
        p, r, _ = precision_recall_curve(yy, prob); ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, prob):.3f}")
        fpr, tpr, _ = roc_curve(yy, prob); ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, prob):.3f}")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title(f"{ENSEMBLE_TAG} PR"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("FPR");    ax2.set_ylabel("TPR");     ax2.set_title(f"{ENSEMBLE_TAG} ROC"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)

    # Training history per seed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in all_results:
        ep = [h["epoch"] for h in r["history"]]
        f1 = [h["val_f1m"] for h in r["history"]]
        lo = [h["loss"]    for h in r["history"]]
        axes[0].plot(ep, f1, "-", linewidth=1.5, label=f"seed={r['seed']}")
        axes[1].plot(ep, lo, "-", linewidth=1.5, label=f"seed={r['seed']}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Macro-F1"); axes[0].set_title("Per-seed Val F1m"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Train Loss"); axes[1].set_title("Per-seed Train Loss"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f"5-Seed Ensemble — Training History (B=128, ep=40)", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(HIST_PNG, dpi=120); plt.close(fig)
    print(f"\n[plots saved]")
except Exception as e:
    print(f"\n[plots skipped: {e}]")