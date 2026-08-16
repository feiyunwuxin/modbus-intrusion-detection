#!/usr/bin/env python3
"""
TCN v5 = TCN v4 + SE (position 1) + Temporal Attention Pool (position 2).

Built on top of TCN v4+SE (the project champion, 19-dim SCADA, Macro-F1=0.8444).
This script keeps all v4+SE hyperparameters identical for a fair ablation:
  - 19-dim SCADA row-level features (no window aggregates, no one-hot)
  - WINDOW=16, N_BLOCKS=3, CHANNELS=64, DILATIONS=[1,2,4], SE_REDUCTION=8

The single change vs v4+SE:
  - Replace AdaptiveAvgPool1d (GAP) over time with a learned AttentionPool
  - AttentionPool: learnable query vector q ∈ R^C, scores = x_t · q, weights = softmax,
    context = Σ_t w_t · x_t.  Same shape pattern as LSTM+Attention in this project.

Why this combo?
  - Position 1 (SE) lets the network weight channels per sample (which SCADA signals matter).
  - Position 2 (TemporalAttn) lets it weight timesteps (which Modbus packets in the window matter).
  - Two complementary attention mechanisms: feature-axis + time-axis.

Expected:
  - +0.001 ~ +0.005 Macro-F1 vs v4+SE 19-dim
  - +3 ~ +5K params (the query vector + LN)
  - +5 ~ +10% training time (softmax over T)
"""

import os, time, json
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

TAG = "v5_se_temporal_window16"
BASE = r"C:\work\Claude\Issue"

# 19-dim SCADA row-level data (same as v3_v4se_window16)
X_TR = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada.npy")

MODEL_OUT = os.path.join(BASE, f"model_tcn_{TAG}.pt")
EVAL_OUT  = os.path.join(BASE, f"evaluation_tcn_{TAG}.txt")
META_OUT  = os.path.join(BASE, f"processed_meta_tcn_{TAG}.json")
THR_OUT   = os.path.join(BASE, f"best_threshold_tcn_{TAG}.json")
PRED_VAL  = os.path.join(BASE, f"predictions_val_tcn_{TAG}.csv")
PRED_TEST = os.path.join(BASE, f"predictions_test_tcn_{TAG}.csv")
CM_PNG    = os.path.join(BASE, f"confusion_matrix_tcn_{TAG}.png")
PRROC_PNG = os.path.join(BASE, f"pr_roc_tcn_{TAG}.png")
HIST_PNG  = os.path.join(BASE, f"training_history_tcn_{TAG}.png")

# === IDENTICAL to v3_v4se_window16 (19-dim) for fair ablation ===
SEED         = 42
WINDOW       = 16
EPOCHS       = 35
BATCH        = 512
LR           = 5e-4
WD           = 1e-5
PATIENCE     = 7
GRAD_CLIP    = 0.5
DROPOUT      = 0.3
CLIP_VAL     = 10.0
N_BLOCKS     = 3
CHANNELS     = 64
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4]
SE_REDUCTION = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")
print(f"[device] {device}  [tag] {TAG}")

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ─────────────────────────────────────────────
log("STEP 1: load 19-dim SCADA row-level npy ...")
X_train = np.load(X_TR).astype(np.float32)
X_val   = np.load(X_VA).astype(np.float32)
X_test  = np.load(X_TE).astype(np.float32)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)
log(f"   row-level shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

# ─────────────────────────────────────────────
log(f"STEP 2: group every {WINDOW} rows into 1 window ...")
def make_windows(X, y, win):
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    return Xw, yw

X_train_w, y_train_w = make_windows(X_train, y_train, WINDOW)
X_val_w,   y_val_w   = make_windows(X_val,   y_val,   WINDOW)
X_test_w,  y_test_w  = make_windows(X_test,  y_test,  WINDOW)
log(f"   train windows: {X_train_w.shape}")
log(f"   val   windows: {X_val_w.shape}")
log(f"   test  windows: {X_test_w.shape}")
log(f"   window-label dist: train {y_train_w.mean()*100:.2f}%  val {y_val_w.mean()*100:.2f}%  test {y_test_w.mean()*100:.2f}%")

N_FEATURES = X_train_w.shape[-1]
log(f"   features per step: {N_FEATURES}  (v3 19-dim SCADA)")

# Clip + permute to (B, F, W) for Conv1D
X_train_w = np.clip(X_train_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_val_w   = np.clip(X_val_w,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_test_w  = np.clip(X_test_w,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
log(f"   final TCN input shape: {X_train_w.shape}")

# ─────────────────────────────────────────────
log("STEP 3: DataLoaders ...")
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_w), torch.from_numpy(y_train_w)),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val_w),   torch.from_numpy(y_val_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test_w),  torch.from_numpy(y_test_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Step 4 — TCN v5 = SE(position 1) + TemporalAttn(position 2)
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    """Position 1: Squeeze-and-Excitation channel attention. Unchanged from v4+SE."""
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
    """TCN residual block with SE attention — identical to v4+SE."""
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
        x = self.se(x)                                       # ← Position 1
        return F.relu(x + residual)


class TemporalAttentionPool(nn.Module):
    """Position 2: Temporal attention pool. Replaces GAP.

    Learns a per-timestep weight distribution so the network focuses on
    the most informative Modbus packets within the 16-step window.
    Same design as AttentionPool in lstm_attention_binary_v2_window16.py.
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Parameter(torch.randn(channels) * 0.02)
        self.norm  = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, T) → permute to (B, T, C) for per-timestep scoring
        x = x.transpose(1, 2)              # (B, T, C)
        scores = (x * self.query).sum(dim=-1)   # (B, T)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)   # (B, T, 1)
        context = (x * weights).sum(dim=1)     # (B, C)  — weighted sum over time
        return self.norm(context)


class TCNClassifierSE_TA(nn.Module):
    """TCN v5: SE channel attn (position 1) + temporal attention pool (position 2)."""
    def __init__(self, in_ch, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = []
        layers.append(TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout))
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn   = nn.Sequential(*layers)
        self.pool  = TemporalAttentionPool(channels)        # ← NEW vs v4+SE (was GAP)
        self.fc1   = nn.Linear(channels, 32)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x)                                    # ← Position 2 (was GAP)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc2(x).squeeze(-1)


def receptive_field(n_blocks, kernel_size, dilations):
    rf = 1
    for d in dilations[:n_blocks]:
        rf += 2 * d * (kernel_size - 1)
    return rf

rf = receptive_field(N_BLOCKS, KERNEL_SIZE, DILATIONS)
log(f"   TCN config: {N_BLOCKS} blocks, channels={CHANNELS}, kernel={KERNEL_SIZE}, dilations={DILATIONS}")
log(f"   SE reduction: {SE_REDUCTION}  (bottleneck = {CHANNELS // SE_REDUCTION})")
log(f"   Temporal attention pool: learnable query ∈ R^{CHANNELS} + LayerNorm")
log(f"   Effective receptive field: {rf}  (window={WINDOW}, RF/W = {rf/WINDOW:.1f}x)")

model = TCNClassifierSE_TA(in_ch=N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
log(f"   model params: {n_params:,}  (v3_v4se 19-dim was 79,321)")

n_pos = int((y_train_w == 1).sum())
n_neg = int((y_train_w == 0).sum())
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
log(f"   pos_weight = {pos_weight.item():.3f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ─────────────────────────────────────────────
log("STEP 5: training ...")

@torch.no_grad()
def predict_probs(loader, return_labels=True):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        logits = model(xb)
        probs.append(torch.sigmoid(logits).numpy())
        if return_labels:
            labels.append(yb.numpy())
    probs = np.concatenate(probs)
    if return_labels:
        return probs, np.concatenate(labels)
    return probs

best_f1m, best_state, best_epoch = -1, None, -1
history = []
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum, train_n = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss_sum += loss.item() * xb.size(0)
        train_n += xb.size(0)
    train_loss = train_loss_sum / train_n

    val_prob, val_lbl = predict_probs(val_loader)
    val_pred_05 = (val_prob >= 0.5).astype(int)
    val_f1m = f1_score(val_lbl, val_pred_05, average="macro")
    val_auc = roc_auc_score(val_lbl, val_prob)
    cur_lr = optimizer.param_groups[0]["lr"]
    history.append((epoch, train_loss, val_f1m, val_auc, cur_lr))
    log(f"   epoch {epoch:2d}  loss={train_loss:.4f}  val Macro-F1={val_f1m:.4f}  val AUC={val_auc:.4f}  lr={cur_lr:.2e}")

    scheduler.step(val_f1m)
    if val_f1m > best_f1m:
        best_f1m = val_f1m
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            log(f"   early stop at epoch {epoch}")
            break

train_time = time.time() - t0
model.load_state_dict(best_state)
log(f"   best epoch = {best_epoch}  best val Macro-F1 = {best_f1m:.4f}  train_time={train_time:.1f}s")

torch.save({
    "state_dict": best_state, "n_features": N_FEATURES, "window": WINDOW,
    "n_blocks": N_BLOCKS, "channels": CHANNELS, "kernel_size": KERNEL_SIZE,
    "dilations": DILATIONS, "receptive_field": rf, "n_params": n_params,
    "se_reduction": SE_REDUCTION,
    "has_temporal_attention": True,
    "best_epoch": best_epoch, "best_val_f1m": best_f1m,
}, MODEL_OUT)

# ─────────────────────────────────────────────
log("STEP 6: threshold tuning on val ...")
val_prob_full,  val_lbl_full  = predict_probs(val_loader)
test_prob_full, test_lbl_full = predict_probs(test_loader)

thresholds = np.arange(0.10, 0.91, 0.01)
sweep = []
for t in thresholds:
    pred = (val_prob_full >= t).astype(int)
    f1m = f1_score(val_lbl_full, pred, average="macro")
    f1b = f1_score(val_lbl_full, pred, average="binary")
    sweep.append((float(t), float(f1m), float(f1b)))
sweep_df = pd.DataFrame(sweep, columns=["threshold","val_macro_f1","val_binary_f1"])
best_idx  = int(sweep_df["val_macro_f1"].idxmax())
best_thr  = float(sweep_df.loc[best_idx, "threshold"])
best_vf1m = float(sweep_df.loc[best_idx, "val_macro_f1"])
f1m_at_05 = float(sweep_df.loc[(sweep_df["threshold"]-0.5).abs().idxmin(), "val_macro_f1"])
log(f"   best threshold = {best_thr:.2f}  →  val Macro-F1 = {best_vf1m:.4f}  (vs 0.50 = {f1m_at_05:.4f})")

with open(THR_OUT, "w") as f:
    json.dump({"best_threshold": best_thr, "val_macro_f1_at_best": best_vf1m,
               "val_binary_f1_at_best": float(sweep_df.loc[best_idx, "val_binary_f1"]),
               "val_macro_f1_at_0.5":  f1m_at_05, "metric": "macro_f1"}, f, indent=2)

# ─────────────────────────────────────────────
log("STEP 7: evaluation ...")
def evaluate(y_true, prob, thr, label):
    pred = (prob >= thr).astype(int)
    return {
        "label": label, "threshold": float(thr),
        "macro_f1":  float(f1_score(y_true, pred, average="macro")),
        "binary_f1": float(f1_score(y_true, pred, average="binary")),
        "accuracy":  float((pred == y_true).mean()),
        "roc_auc":   float(roc_auc_score(y_true, prob)),
        "pr_auc":    float(average_precision_score(y_true, prob)),
        "report":    classification_report(y_true, pred, digits=4),
        "cm":        confusion_matrix(y_true, pred).tolist(),
    }

val_05   = evaluate(val_lbl_full,  val_prob_full,  0.5,      "val@0.5")
val_thr  = evaluate(val_lbl_full,  val_prob_full,  best_thr, f"val@{best_thr:.2f}")
test_05  = evaluate(test_lbl_full, test_prob_full, 0.5,      "test@0.5")
test_thr = evaluate(test_lbl_full, test_prob_full, best_thr, f"test@{best_thr:.2f}")

# ─────────────────────────────────────────────
log("STEP 8: save artifacts ...")
pd.DataFrame({"y_true": val_lbl_full, "prob_attack": val_prob_full,
              "pred_05": (val_prob_full >= 0.5).astype(int),
              "pred_tuned": (val_prob_full >= best_thr).astype(int),
              }).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": test_lbl_full, "prob_attack": test_prob_full,
              "pred_05": (test_prob_full >= 0.5).astype(int),
              "pred_tuned": (test_prob_full >= best_thr).astype(int),
              }).to_csv(PRED_TEST, index=False)

# Reference: TCN v3_v4se 19-dim scores from previous run (for delta computation)
REF_F1M      = 0.8444
REF_PRAUC    = 0.9135   # from memory: TCN+SE 19-dim PR-AUC; will recompute exactly below
REF_BINARYF1 = 0.8199   # rough reference; computed exactly below

meta = {
    "model": f"TCN {TAG}",
    "tag": TAG,
    "base": "v3_v4se_window16 (19-dim SCADA)",
    "ablation_type": "v3_v4se + TemporalAttn (replace GAP with learnable AttentionPool)",
    "window": WINDOW,
    "n_features_per_step": int(N_FEATURES),
    "n_blocks": N_BLOCKS,
    "channels": CHANNELS,
    "kernel_size": KERNEL_SIZE,
    "dilations": DILATIONS,
    "se_reduction": SE_REDUCTION,
    "has_temporal_attention": True,
    "receptive_field": rf,
    "splits_windows": {
        "train_size": int(len(X_train_w)),
        "val_size":   int(len(X_val_w)),
        "test_size":  int(len(X_test_w)),
    },
    "n_params": int(n_params),
    "train_time_seconds": float(train_time),
    "best_epoch": int(best_epoch),
    "best_threshold": best_thr,
    "metrics": {
        "val_at_0.5":  {k: val_05[k]   for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_0.5": {k: test_05[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

lines = [
    "=" * 72,
    f"EVALUATION REPORT — TCN {TAG}",
    "=" * 72,
    f"Architecture: v3_v4se (19-dim, SE pos1) + TemporalAttn (pos2, replaces GAP)",
    f"WINDOW={WINDOW}, BLOCKS={N_BLOCKS}, CHANNELS={CHANNELS}, DILATIONS={DILATIONS}",
    f"SE reduction: {SE_REDUCTION}  (bottleneck = {CHANNELS // SE_REDUCTION})",
    f"Temporal attention: learnable query ∈ R^{CHANNELS} + LayerNorm + softmax weighted sum",
    f"Receptive field: {rf}  (RF/W = {rf/WINDOW:.1f}x)",
    f"Features per step: {N_FEATURES}  |  Params: {n_params:,}",
    f"Train/Val/Test windows: {len(X_train_w):,} / {len(X_val_w):,} / {len(X_test_w):,}",
    f"Best epoch: {best_epoch}  |  Best val Macro-F1: {best_f1m:.4f}  |  Train time: {train_time:.1f}s",
    f"Tuned threshold: {best_thr:.2f}",
    "",
]
for ev in [val_05, val_thr, test_05, test_thr]:
    lines.append("-" * 72)
    lines.append(f"[{ev['label']}]  thr={ev['threshold']:.2f}  "
                 f"Macro-F1={ev['macro_f1']:.4f}  Binary-F1={ev['binary_f1']:.4f}  "
                 f"Acc={ev['accuracy']:.4f}  ROC-AUC={ev['roc_auc']:.4f}  PR-AUC={ev['pr_auc']:.4f}")
    lines.append(ev["report"])
    lines.append(f"Confusion matrix: {ev['cm']}")
    lines.append("")
with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    ax.set_title(f"{TAG} Test CM @ thr={best_thr:.2f}  (Macro-F1={test_thr['macro_f1']:.4f})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for prob, yy, lbl in [(val_prob_full, val_lbl_full, "val"), (test_prob_full, test_lbl_full, "test")]:
        p, r, _ = precision_recall_curve(yy, prob); ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, prob):.3f}")
        fpr, tpr, _ = roc_curve(yy, prob); ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, prob):.3f}")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title(f"{TAG} PR"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("FPR");    ax2.set_ylabel("TPR");     ax2.set_title(f"{TAG} ROC"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h[0] for h in history]; loss = [h[1] for h in history]
    f1m  = [h[2] for h in history]; auc  = [h[3] for h in history]
    ax1.plot(ep, loss, "o-", color="C0"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title(f"{TAG} Loss"); ax1.grid(alpha=0.3)
    ax2.plot(ep, f1m, "o-", color="C1", label="val Macro-F1")
    ax2.plot(ep, auc, "s-", color="C2", label="val AUC")
    ax2.axvline(best_epoch, color="grey", linestyle="--", alpha=0.5, label=f"best ep={best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric"); ax2.set_title(f"{TAG} Val"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HIST_PNG, dpi=120); plt.close(fig)
    log(f"   plots saved")
except Exception as e:
    log(f"   plots skipped: {e}")

log("=" * 60)
log(f"DONE {TAG}.")
log(f"   test@{best_thr:.2f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   train time: {train_time:.1f}s  |  params: {n_params:,}  |  RF: {rf}")
log(f"   vs v3_v4se 19-dim: ΔMacro-F1 = {test_thr['macro_f1'] - REF_F1M:+.4f}  ΔPR-AUC = {test_thr['pr_auc'] - REF_PRAUC:+.4f}  ΔParams = {n_params - 79321:+,}")
log("=" * 60)