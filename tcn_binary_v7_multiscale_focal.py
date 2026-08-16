#!/usr/bin/env python3
"""
TCN v7 = TCN v4+SE + Multi-Scale Feature Fusion + Focal Loss + EMA.

Built on top of v3_v4se_window16 (19-dim SCADA, PR-AUC=0.8892, Macro-F1=0.8444).

Three changes vs v4+SE:
  1. Multi-Scale Feature Fusion (Deep Supervision):
     - Each TCN block contributes its GAP output (B, 64)
     - Concatenate all 3: (B, 64 × 3 = 192)
     - Pass through a small MLP: FC(192→64) → ReLU → Drop → FC(64→1)
     - This lets the classifier see features at all receptive fields (RF=5, 13, 29)

  2. Focal Loss (replace BCE):
     - Focal(α=0.25, γ=2.0)
     - Down-weights easy samples → focuses on borderline cases
     - PR-AUC directly measures ranking quality → focal loss refines the ranking

  3. EMA model (Exponential Moving Average):
     - Maintain EMA of model weights with decay=0.999
     - Use EMA weights for evaluation (smoother, less noisy)
     - Common in modern training pipelines (e.g. timm, MAE)

Expected:
  - PR-AUC: 0.8892 → 0.91 ~ 0.92 (target: beat V19's 0.9133)
  - Macro-F1: 0.8444 → ~0.83 (likely slight drop; F1 + PR-AUC trade-off)
  - Params: 72,921 → ~85K (+12K from MLP head on 192-dim concat)
  - Training time: +10 ~ +15%
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

TAG = "v7_multiscale_focal_window16"
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

# === NEW: Focal Loss + EMA + Fusion MLP ===
FOCAL_ALPHA  = 0.25    # weight for positive class
FOCAL_GAMMA  = 2.0     # focusing parameter (higher = more focus on hard examples)
EMA_DECAY    = 0.99    # EMA decay (was 0.999 → too slow for ~24 steps/epoch)
FUSION_HIDDEN = 64     # MLP hidden dim after multi-scale concat (concat = 3*64=192)
HEAD_DROPOUT = 0.3

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

N_FEATURES = X_train_w.shape[-1]
log(f"   features per step: {N_FEATURES}  (v3 19-dim SCADA)")

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
# Step 4 — TCN v7 = v4+SE + Multi-Scale Fusion
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    """Same as v4+SE."""
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
    """Same as v4+SE."""
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


class TCNClassifierMultiScale(nn.Module):
    """TCN v7: SE per block + multi-scale feature fusion.

    Returns BOTH per-block GAP outputs and the final logit, so we can:
      - use the final logit for loss / inference
      - (optionally) use intermediate features for auxiliary losses (not used here)
    """
    def __init__(self, in_ch, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT,
                 fusion_hidden=FUSION_HIDDEN, head_dropout=HEAD_DROPOUT):
        super().__init__()
        # Build TCN blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout))
        for d in dilations[1:n_blocks]:
            self.blocks.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        # Per-block GAPs
        self.gaps = nn.ModuleList([nn.AdaptiveAvgPool1d(1) for _ in range(n_blocks)])
        # Multi-scale fusion head: concat (B, 3*64=192) → FC → ReLU → Drop → FC → logit
        fusion_dim = channels * n_blocks
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x):
        feats = []
        for blk, gap in zip(self.blocks, self.gaps):
            x = blk(x)
            feats.append(gap(x).squeeze(-1))        # (B, C) per block
        cat = torch.cat(feats, dim=-1)              # (B, 3*C)
        return self.head(cat).squeeze(-1)            # (B,)


# ─────────────────────────────────────────────
# Focal Loss — replaces BCEWithLogitsLoss
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Binary Focal Loss (Lin et al. 2017).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
      alpha: weight for positive class (0.25 ~ 0.5 typical)
      gamma: focusing parameter (2.0 typical; 0 = BCE)
    """
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # targets: float (B,)
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)   # p if y=1 else 1-p
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────
# EMA model — exponential moving average of weights
# ─────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype in (torch.float32, torch.float64):
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    # buffers like num_batches_tracked: copy as-is
                    self.shadow[k].copy_(v.detach())

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


# ─────────────────────────────────────────────
def receptive_field(n_blocks, kernel_size, dilations):
    rf = 1
    for d in dilations[:n_blocks]:
        rf += 2 * d * (kernel_size - 1)
    return rf


# ─────────────────────────────────────────────
log("STEP 4: build model ...")
model = TCNClassifierMultiScale(in_ch=N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
log(f"   TCN v7 Multi-Scale + Focal + EMA on 19-dim input, params: {n_params:,}")

n_pos = int((y_train_w == 1).sum())
n_neg = int((y_train_w == 0).sum())
log(f"   class counts: n_neg={n_neg:,}, n_pos={n_pos:,}")

# Focal Loss (no class weight needed since α handles it)
loss_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
log(f"   Focal Loss: α={FOCAL_ALPHA}, γ={FOCAL_GAMMA}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
ema = EMA(model, decay=EMA_DECAY)


# ─────────────────────────────────────────────
log("STEP 5: training ...")

@torch.no_grad()
def predict_probs(loader, model_to_use):
    model_to_use.eval()
    probs, labels = [], []
    for xb, yb in loader:
        logits = model_to_use(xb)
        probs.append(torch.sigmoid(logits).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


best_f1m, best_ema_state, best_epoch = -1, None, -1
history = []
epochs_no_improve = 0

# We evaluate BOTH the live model and the EMA model on val each epoch
# Track best EMA val F1m (which is what we'll use for final eval)

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
        ema.update(model)
        train_loss_sum += loss.item() * xb.size(0)
        train_n += xb.size(0)
    train_loss = train_loss_sum / train_n

    # Evaluate live model
    val_prob, val_lbl = predict_probs(val_loader, model)
    val_f1m = f1_score(val_lbl, (val_prob >= 0.5).astype(int), average="macro")

    # Evaluate EMA model
    ema_model = copy.deepcopy(model)
    ema.apply_to(ema_model)
    val_prob_ema, val_lbl_ema = predict_probs(val_loader, ema_model)
    val_f1m_ema = f1_score(val_lbl_ema, (val_prob_ema >= 0.5).astype(int), average="macro")
    val_auc_ema = roc_auc_score(val_lbl_ema, val_prob_ema)

    cur_lr = optimizer.param_groups[0]["lr"]
    history.append((epoch, train_loss, val_f1m, val_f1m_ema, val_auc_ema, cur_lr))
    log(f"   epoch {epoch:2d}  loss={train_loss:.4f}  val F1m(live)={val_f1m:.4f}  val F1m(EMA)={val_f1m_ema:.4f}  val AUC(EMA)={val_auc_ema:.4f}  lr={cur_lr:.2e}")

    scheduler.step(val_f1m_ema)   # use EMA F1 for LR scheduling
    if val_f1m_ema > best_f1m:
        best_f1m = val_f1m_ema
        best_ema_state = {k: v.clone() for k, v in ema.shadow.items()}
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            log(f"   early stop at epoch {epoch}")
            break

train_time = time.time() - t0
log(f"   best epoch = {best_epoch}  best val Macro-F1 (EMA) = {best_f1m:.4f}  train_time={train_time:.1f}s")

# Load best EMA weights
ema.apply_to(model)
torch.save({
    "state_dict": best_ema_state, "n_features": N_FEATURES, "window": WINDOW,
    "n_blocks": N_BLOCKS, "channels": CHANNELS, "kernel_size": KERNEL_SIZE,
    "dilations": DILATIONS, "receptive_field": receptive_field(N_BLOCKS, KERNEL_SIZE, DILATIONS),
    "n_params": n_params,
    "se_reduction": SE_REDUCTION,
    "fusion_hidden": FUSION_HIDDEN,
    "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA,
    "ema_decay": EMA_DECAY,
    "best_epoch": best_epoch, "best_val_f1m": best_f1m,
}, MODEL_OUT)

# ─────────────────────────────────────────────
log("STEP 6: threshold tuning on val (EMA model) ...")
val_prob_full,  val_lbl_full  = predict_probs(val_loader, model)
test_prob_full, test_lbl_full = predict_probs(test_loader, model)

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

# Reference: TCN v3_v4se 19-dim
REF_F1M      = 0.8444
REF_PRAUC    = 0.8892
REF_BINARYF1 = 0.8347
REF_PARAMS   = 72921

meta = {
    "model": f"TCN {TAG}",
    "tag": TAG,
    "base": "v3_v4se_window16 (19-dim SCADA)",
    "ablation_type": "v4+SE + Multi-Scale Feature Fusion + Focal Loss + EMA",
    "window": WINDOW,
    "n_features_per_step": int(N_FEATURES),
    "n_blocks": N_BLOCKS,
    "channels": CHANNELS,
    "kernel_size": KERNEL_SIZE,
    "dilations": DILATIONS,
    "se_reduction": SE_REDUCTION,
    "fusion_hidden": FUSION_HIDDEN,
    "focal_alpha": FOCAL_ALPHA,
    "focal_gamma": FOCAL_GAMMA,
    "ema_decay": EMA_DECAY,
    "receptive_field": receptive_field(N_BLOCKS, KERNEL_SIZE, DILATIONS),
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
    f"Architecture: v4+SE (19-dim) + Multi-Scale Fusion + Focal Loss + EMA",
    f"WINDOW={WINDOW}, BLOCKS={N_BLOCKS}, CHANNELS={CHANNELS}, DILATIONS={DILATIONS}",
    f"Multi-Scale Fusion: concat GAP from all 3 blocks → FC({3*CHANNELS}→{FUSION_HIDDEN}) → FC({FUSION_HIDDEN}→1)",
    f"Focal Loss: α={FOCAL_ALPHA}, γ={FOCAL_GAMMA}",
    f"EMA decay: {EMA_DECAY}",
    f"Features per step: {N_FEATURES}  |  Params: {n_params:,}",
    f"Train/Val/Test windows: {len(X_train_w):,} / {len(X_val_w):,} / {len(X_test_w):,}",
    f"Best epoch: {best_epoch}  |  Best val Macro-F1 (EMA): {best_f1m:.4f}  |  Train time: {train_time:.1f}s",
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
    ep = [h[0] for h in history]
    loss = [h[1] for h in history]
    f1m_live = [h[2] for h in history]
    f1m_ema  = [h[3] for h in history]
    auc_ema  = [h[4] for h in history]
    ax1.plot(ep, loss, "o-", color="C0"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title(f"{TAG} Loss"); ax1.grid(alpha=0.3)
    ax2.plot(ep, f1m_live, "o-", color="C1", label="val F1m (live)", alpha=0.5)
    ax2.plot(ep, f1m_ema,  "o-", color="C3", label="val F1m (EMA)")
    ax2.plot(ep, auc_ema,  "s-", color="C2", label="val AUC (EMA)")
    ax2.axvline(best_epoch, color="grey", linestyle="--", alpha=0.5, label=f"best ep={best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric"); ax2.set_title(f"{TAG} Val"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HIST_PNG, dpi=120); plt.close(fig)
    log(f"   plots saved")
except Exception as e:
    log(f"   plots skipped: {e}")

log("=" * 60)
log(f"DONE {TAG}.")
log(f"   test@{best_thr:.2f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   train time: {train_time:.1f}s  |  params: {n_params:,}")
log(f"   vs v3_v4se 19-dim: ΔMacro-F1 = {test_thr['macro_f1'] - REF_F1M:+.4f}  ΔPR-AUC = {test_thr['pr_auc'] - REF_PRAUC:+.4f}  ΔParams = {n_params - REF_PARAMS:+,}")
log("=" * 60)