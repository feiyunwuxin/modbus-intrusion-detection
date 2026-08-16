#!/usr/bin/env python3
"""
1D MobileNetV3-Small binary classifier on the 17-feature v2 dataset, sliding window=8.

MobileNetV3 adds three new building blocks on top of MobileNetV1/V2:
  1. h-swish activation:  x * ReLU6(x + 3) / 6       (cheap swish approximation)
  2. h-sigmoid:           ReLU6(x + 3) / 6            (cheap sigmoid approximation)
  3. Squeeze-and-Excitation (SE) channel-attention blocks

MobileNetV3-Small uses 11 NAS-optimised bottleneck blocks with mixed (RE/HS) activations
and selective SE. The "Small" variant favours low latency over accuracy and uses 16/24/40/48/96
output channels, smaller expansion ratios (typically 3-6x) and shorter kernels (3 or 5).

This script adapts V3-Small to 1D sequences:
  - 2D convs become 1D convs
  - Strides are set to 1 throughout to preserve the 8-timestep window length
  - SE uses 1D adaptive average pool + 1x1 conv 1D (equivalent to FC)

References:
  Howard et al., "Searching for MobileNetV3", ICCV 2019.
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

BASE      = r"C:\work\Claude\Issue"
X_TR      = os.path.join(BASE, "X_train_binary.npy")
X_VA      = os.path.join(BASE, "X_val_binary.npy")
X_TE      = os.path.join(BASE, "X_test_binary.npy")
Y_TR      = os.path.join(BASE, "y_train_binary.npy")
Y_VA      = os.path.join(BASE, "y_val_binary.npy")
Y_TE      = os.path.join(BASE, "y_test_binary.npy")

MODEL_OUT = os.path.join(BASE, "model_mobile_net_v3_binary_v2_window8.pt")
EVAL_OUT  = os.path.join(BASE, "evaluation_mobile_net_v3_binary_v2_window8.txt")
META_OUT  = os.path.join(BASE, "processed_meta_mobile_net_v3_binary_window8.json")
THR_OUT   = os.path.join(BASE, "best_threshold_mobile_net_v3_window8.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_mobile_net_v3_binary_v2_window8.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_mobile_net_v3_binary_v2_window8.csv")
CM_PNG    = os.path.join(BASE, "confusion_matrix_mobile_net_v3_binary_v2_window8.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_mobile_net_v3_binary_v2_window8.png")
HIST_PNG  = os.path.join(BASE, "training_history_mobile_net_v3_binary_v2_window8.png")

# Hyperparameters
SEED        = 42
WINDOW      = 8
EPOCHS      = 30
BATCH       = 512
LR          = 5e-4
WD          = 1e-5
PATIENCE    = 6
GRAD_CLIP   = 0.5
DROPOUT     = 0.3
CLIP_VAL    = 10.0
SE_RATIO    = 0.25      # SE bottleneck ratio (4x downsample)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")
print(f"[device] {device} (cuda available: {torch.cuda.is_available()})")

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ─────────────────────────────────────────────
# Step 1 — Load npy
# ─────────────────────────────────────────────
log("STEP 1: load npy ...")
X_train = np.load(X_TR).astype(np.float32)
X_val   = np.load(X_VA).astype(np.float32)
X_test  = np.load(X_TE).astype(np.float32)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)
log(f"   raw shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

# ─────────────────────────────────────────────
# Step 2 — Group every WINDOW rows into 1 window
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
log(f"   train windows: {X_train_w.shape} (dropped {len(X_train) - len(X_train_w)*WINDOW} tail rows)")
log(f"   val   windows: {X_val_w.shape}   (dropped {len(X_val)   - len(X_val_w)*WINDOW} tail rows)")
log(f"   test  windows: {X_test_w.shape}  (dropped {len(X_test)  - len(X_test_w)*WINDOW} tail rows)")
log(f"   window-label dist: train 0={int((y_train_w==0).sum()):,} 1={int((y_train_w==1).sum()):,} ({y_train_w.mean()*100:.2f}% attack)")
log(f"                       val   0={int((y_val_w  ==0).sum()):,} 1={int((y_val_w  ==1).sum()):,} ({y_val_w.mean()*100:.2f}% attack)")
log(f"                       test  0={int((y_test_w ==0).sum()):,} 1={int((y_test_w ==1).sum()):,} ({y_test_w.mean()*100:.2f}% attack)")

# ─────────────────────────────────────────────
# Step 3 — One-hot encode function at each timestep
# ─────────────────────────────────────────────
log("STEP 3: one-hot encode function column at each timestep ...")
fn_train = X_train_w[:, :, 1].astype(np.int64).ravel()
fn_val   = X_val_w[:,   :, 1].astype(np.int64).ravel()
fn_test  = X_test_w[:,  :, 1].astype(np.int64).ravel()
all_fn   = np.unique(np.concatenate([fn_train, fn_val, fn_test]))
n_cats   = len(all_fn)
log(f"   function unique codes: {n_cats}")

fn_map = {v: i for i, v in enumerate(all_fn)}
def onehot_window(fn_codes, n):
    out = np.zeros((len(fn_codes), n), dtype=np.float32)
    out[np.arange(len(fn_codes)), fn_codes] = 1.0
    return out

oh_tr = onehot_window(np.vectorize(fn_map.get)(fn_train), n_cats).reshape(X_train_w.shape[0], WINDOW, n_cats)
oh_va = onehot_window(np.vectorize(fn_map.get)(fn_val),   n_cats).reshape(X_val_w.shape[0],   WINDOW, n_cats)
oh_te = onehot_window(np.vectorize(fn_map.get)(fn_test),  n_cats).reshape(X_test_w.shape[0],  WINDOW, n_cats)

# Drop col 1 (function) and concat one-hot at each timestep
keep_idx = [0] + list(range(2, 17))   # 16 numeric cols
num_train = X_train_w[:, :, keep_idx]
num_val   = X_val_w[:,   :, keep_idx]
num_test  = X_test_w[:,  :, keep_idx]
X_train_cnn = np.concatenate([num_train, oh_tr], axis=-1)
X_val_cnn   = np.concatenate([num_val,   oh_va], axis=-1)
X_test_cnn  = np.concatenate([num_test,  oh_te], axis=-1)
N_FEATURES  = X_train_cnn.shape[-1]
log(f"   per-step feature dim: {N_FEATURES} (16 numeric + {n_cats} one-hot)")

# Clip + permute to (B, F, W) for Conv1D
X_train_cnn = np.clip(X_train_cnn, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_val_cnn   = np.clip(X_val_cnn,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_test_cnn  = np.clip(X_test_cnn,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
log(f"   final MobileNetV3 input shape: {X_train_cnn.shape}  → Conv1D(F={N_FEATURES}, W={WINDOW})")

# ─────────────────────────────────────────────
# Step 4 — DataLoaders
# ─────────────────────────────────────────────
log("STEP 4: DataLoaders ...")
train_ds = TensorDataset(torch.from_numpy(X_train_cnn), torch.from_numpy(y_train_w))
val_ds   = TensorDataset(torch.from_numpy(X_val_cnn),   torch.from_numpy(y_val_w))
test_ds  = TensorDataset(torch.from_numpy(X_test_cnn),  torch.from_numpy(y_test_w))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Step 5 — MobileNetV3-Small 1D
# ─────────────────────────────────────────────
class HSwish(nn.Module):
    """h-swish:  x * ReLU6(x + 3) / 6   — cheap swish approximation.
    Accepts (and ignores) `inplace=` for nn.ReLU-style API compatibility.
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * F.relu6(x + 3) / 6.0


class HSigmoid(nn.Module):
    """h-sigmoid: ReLU6(x + 3) / 6 — cheap sigmoid approximation."""
    def forward(self, x):
        return F.relu6(x + 3) / 6.0


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block (1D version).
    Channel attention via global avg pool → 1x1 conv (FC) reduce → ReLU → 1x1 conv expand → h-sigmoid → scale.
    """
    def __init__(self, channels, se_ratio=SE_RATIO):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1, bias=True)
        self.hsigmoid = HSigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1)        # squeeze
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.hsigmoid(s)                   # excite
        return x * s                           # scale


class MobileNetV3Bottleneck1D(nn.Module):
    """MobileNetV3 bottleneck (1D): expansion 1x1 → depthwise kxk → SE → projection 1x1.

    Layout (in_ch, exp_ch, out_ch, kernel, stride, use_se, use_hs):
      - if exp_ch != in_ch:    1x1 conv (expand) + BN + (HSwish | ReLU)
      - depthwise conv kxk + BN + (HSwish | ReLU)
      - if use_se:             SEBlock1D(exp_ch)
      - 1x1 conv (project) + BN  (no activation)
      - residual: add x if in_ch == out_ch and stride == 1
    """
    def __init__(self, in_ch, exp_ch, out_ch, kernel, stride, use_se, use_hs):
        super().__init__()
        self.use_residual = (in_ch == out_ch) and (stride == 1)
        act = HSwish if use_hs else nn.ReLU

        layers = []
        # 1) Expansion 1x1 (skip if in == exp)
        if exp_ch != in_ch:
            layers += [
                nn.Conv1d(in_ch, exp_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(exp_ch),
                act(inplace=True),
            ]
        # 2) Depthwise kxk
        pad = (kernel - 1) // 2
        layers += [
            nn.Conv1d(exp_ch, exp_ch, kernel_size=kernel, stride=stride,
                      padding=pad, groups=exp_ch, bias=False),
            nn.BatchNorm1d(exp_ch),
            act(inplace=True),
        ]
        # 3) SE
        if use_se:
            layers.append(SEBlock1D(exp_ch))
        # 4) Projection 1x1 (no activation)
        layers += [
            nn.Conv1d(exp_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


# MobileNetV3-Small configuration (Howard et al., 2019).
# Tuples: (exp_ch, out_ch, kernel, stride, use_se, use_hs)
# Strides are forced to 1 here to preserve the 8-timestep window length.
MNV3_SMALL_CONFIG = [
    ( 16,  16, 3, 1, True,  False),   # SE, RE
    ( 72,  24, 3, 1, False, False),   # RE
    ( 88,  24, 3, 1, False, False),   # RE
    ( 96,  40, 5, 1, True,  True),    # SE, HS
    (240,  40, 5, 1, True,  True),    # SE, HS
    (240,  40, 5, 1, True,  True),    # SE, HS
    (120,  48, 5, 1, True,  True),    # SE, HS
    (144,  48, 5, 1, True,  True),    # SE, HS
    (288,  96, 5, 1, True,  True),    # SE, HS
    (576,  96, 5, 1, True,  True),    # SE, HS
    (576,  96, 5, 1, True,  True),    # SE, HS
]


class MobileNetV3Small1D(nn.Module):
    """MobileNetV3-Small adapted for 1D sequences."""
    def __init__(self, in_ch, dropout=DROPOUT, se_ratio=SE_RATIO):
        super().__init__()
        # 1) Stem: 3x3 conv stride 1 (preserve length), 16 output channels, h-swish
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            HSwish(),
        )
        # 2) Bottleneck stack
        bottlenecks = []
        in_c = 16
        for exp_c, out_c, k, s, se, hs in MNV3_SMALL_CONFIG:
            bottlenecks.append(
                MobileNetV3Bottleneck1D(in_c, exp_c, out_c, k, s, se, hs)
            )
            in_c = out_c
        self.bottlenecks = nn.Sequential(*bottlenecks)
        # 3) Final 1x1 expand to 576 channels + h-swish
        self.final_expand = nn.Sequential(
            nn.Conv1d(in_c, 576, kernel_size=1, bias=False),
            nn.BatchNorm1d(576),
            HSwish(),
        )
        # 4) Classifier head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(576, 1)

    def forward(self, x):
        # x: (B, F, T)
        x = self.stem(x)                            # (B, 16, T)
        x = self.bottlenecks(x)                     # (B, 96, T)
        x = self.final_expand(x)                    # (B, 576, T)
        x = self.gap(x).squeeze(-1)                 # (B, 576)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)                # logits (B,)


model = MobileNetV3Small1D(in_ch=N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
log(f"   model params: {n_params:,}")
log(f"   architecture: Stem({N_FEATURES}→16, k=3, h-swish)")
log(f"                 → 11 MobileNetV3 bottlenecks:")
for i, (e, o, k, s, se, hs) in enumerate(MNV3_SMALL_CONFIG):
    nl = "HS" if hs else "RE"
    se_str = "SE" if se else "  "
    log(f"                    b{i+1:2d}: exp={e:>3} out={o:>3} k={k} stride={s} {se_str} {nl}")
log(f"                 → Final 1x1 expand 96→576 (h-swish)")
log(f"                 → GAP → Dropout({DROPOUT}) → FC(576→1)")

# Show block-wise parameter breakdown
breakdown = []
hook_handles = []
def hook_factory(name):
    def h(_m, _i, o):
        pass
    return h
# Quick per-module param count
for name, m in model.named_modules():
    if isinstance(m, (nn.Conv1d, nn.BatchNorm1d, nn.Linear)):
        pass
total_stem     = sum(p.numel() for p in model.stem.parameters())
total_bottleneck = sum(p.numel() for p in model.bottlenecks.parameters())
total_expand   = sum(p.numel() for p in model.final_expand.parameters())
total_head     = sum(p.numel() for p in model.gap.parameters()) + \
                 sum(p.numel() for p in model.dropout.parameters()) + \
                 sum(p.numel() for p in model.fc.parameters())
log(f"   param breakdown: stem={total_stem:,}  bottlenecks={total_bottleneck:,}  "
    f"final_expand={total_expand:,}  head={total_head:,}")

n_pos = int((y_train_w == 1).sum())
n_neg = int((y_train_w == 0).sum())
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
log(f"   pos_weight = {pos_weight.item():.3f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2)

# ─────────────────────────────────────────────
# Step 6 — Train
# ─────────────────────────────────────────────
log("STEP 6: training ...")

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

model.load_state_dict(best_state)
log(f"   best epoch = {best_epoch}  best val Macro-F1 = {best_f1m:.4f}")
torch.save({
    "state_dict": best_state,
    "n_features": N_FEATURES,
    "window": WINDOW,
    "config": MNV3_SMALL_CONFIG,
    "se_ratio": SE_RATIO,
    "n_params": n_params,
    "best_epoch": best_epoch,
    "best_val_f1m": best_f1m,
    "n_cats_function": n_cats,
    "function_values": all_fn.tolist(),
    "param_breakdown": {
        "stem": total_stem, "bottlenecks": total_bottleneck,
        "final_expand": total_expand, "head": total_head,
    },
}, MODEL_OUT)

# ─────────────────────────────────────────────
# Step 7 — Threshold tuning on val
# ─────────────────────────────────────────────
log("STEP 7: threshold tuning on val ...")
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
    json.dump({
        "best_threshold": best_thr,
        "val_macro_f1_at_best": best_vf1m,
        "val_binary_f1_at_best": float(sweep_df.loc[best_idx, "val_binary_f1"]),
        "val_macro_f1_at_0.5":  f1m_at_05,
        "metric": "macro_f1",
    }, f, indent=2)

# ─────────────────────────────────────────────
# Step 8 — Evaluate
# ─────────────────────────────────────────────
log("STEP 8: evaluation ...")
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
# Step 9 — Save artifacts
# ─────────────────────────────────────────────
log("STEP 9: save artifacts ...")

pd.DataFrame({
    "y_true": val_lbl_full, "prob_attack": val_prob_full,
    "pred_05": (val_prob_full >= 0.5).astype(int),
    "pred_tuned": (val_prob_full >= best_thr).astype(int),
}).to_csv(PRED_VAL, index=False)

pd.DataFrame({
    "y_true": test_lbl_full, "prob_attack": test_prob_full,
    "pred_05": (test_prob_full >= 0.5).astype(int),
    "pred_tuned": (test_prob_full >= best_thr).astype(int),
}).to_csv(PRED_TEST, index=False)

meta = {
    "model": f"MobileNetV3-Small 1D v2 window={WINDOW}",
    "window": WINDOW,
    "n_features_per_step": int(N_FEATURES),
    "n_features_raw": 17,
    "n_cats_function": int(n_cats),
    "config": list(MNV3_SMALL_CONFIG),
    "se_ratio": SE_RATIO,
    "param_breakdown": {
        "stem": total_stem, "bottlenecks": total_bottleneck,
        "final_expand": total_expand, "head": total_head,
    },
    "splits_windows": {
        "train_size": int(len(X_train_w)),
        "val_size":   int(len(X_val_w)),
        "test_size":  int(len(X_test_w)),
    },
    "class_dist_windows": {
        "train": {"0": int((y_train_w==0).sum()), "1": int((y_train_w==1).sum())},
        "val":   {"0": int((y_val_w  ==0).sum()), "1": int((y_val_w  ==1).sum())},
        "test":  {"0": int((y_test_w ==0).sum()), "1": int((y_test_w ==1).sum())},
    },
    "n_params": int(n_params),
    "best_epoch": int(best_epoch),
    "best_threshold": best_thr,
    "hyperparams": {
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
        "patience": PATIENCE, "grad_clip": GRAD_CLIP, "dropout": DROPOUT,
        "clip_val": CLIP_VAL, "pos_weight": float(pos_weight.item()),
    },
    "metrics": {
        "val_at_0.5":  {k: val_05[k]   for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_0.5": {k: test_05[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

# Text report
lines = []
lines.append("=" * 72)
lines.append("EVALUATION REPORT — MobileNetV3-Small 1D Binary (v2, sliding window=8)")
lines.append("=" * 72)
lines.append(f"Window: {WINDOW} rows  |  Features per step: {N_FEATURES} (16 raw + {n_cats} one-hot function)")
lines.append("Architecture: MobileNetV3-Small adapted to 1D (h-swish + SE + NAS-optimised blocks)")
lines.append("Bottleneck config (exp_ch, out_ch, kernel, stride, use_se, use_hs):")
for i, (e, o, k, s, se, hs) in enumerate(MNV3_SMALL_CONFIG):
    nl = "HS" if hs else "RE"
    se_str = "SE" if se else "  "
    lines.append(f"  b{i+1:2d}: exp={e:>3} out={o:>3} k={k} stride={s} {se_str} {nl}")
lines.append(f"SE ratio: {SE_RATIO}  |  Strides forced to 1 to preserve window length")
lines.append(f"Train windows: {len(X_train_w):,}  |  Val windows: {len(X_val_w):,}  |  Test windows: {len(X_test_w):,}")
lines.append(f"Model params: {n_params:,}  (stem={total_stem:,} + bottlenecks={total_bottleneck:,} + "
             f"final_expand={total_expand:,} + head={total_head:,})")
lines.append(f"Best epoch: {best_epoch}  |  Best val Macro-F1: {best_f1m:.4f}")
lines.append(f"Tuned threshold (chosen on val, optimizing Macro-F1): {best_thr:.2f}")
lines.append("")
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
    ax.set_title(f"Test CM @ thr={best_thr:.2f}  (Macro-F1={test_thr['macro_f1']:.4f})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for prob, yy, lbl in [(val_prob_full, val_lbl_full, "val"),
                          (test_prob_full, test_lbl_full, "test")]:
        p, r, _ = precision_recall_curve(yy, prob)
        ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, prob):.3f}")
        fpr, tpr, _ = roc_curve(yy, prob)
        ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, prob):.3f}")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title("PR"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("FPR");    ax2.set_ylabel("TPR");     ax2.set_title("ROC"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h[0] for h in history]
    loss = [h[1] for h in history]
    f1m  = [h[2] for h in history]
    auc  = [h[3] for h in history]
    ax1.plot(ep, loss, "o-", color="C0"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("Loss"); ax1.grid(alpha=0.3)
    ax2.plot(ep, f1m, "o-", color="C1", label="val Macro-F1")
    ax2.plot(ep, auc, "s-", color="C2", label="val AUC")
    ax2.axvline(best_epoch, color="grey", linestyle="--", alpha=0.5, label=f"best ep={best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric"); ax2.set_title("Validation"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HIST_PNG, dpi=120); plt.close(fig)
    log(f"   plots saved: {CM_PNG}, {PRROC_PNG}, {HIST_PNG}")
except Exception as e:
    log(f"   plots skipped: {e}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
log("=" * 60)
log("DONE.")
log(f"   window        : {WINDOW} rows per window")
log(f"   features/step : {N_FEATURES} (16 raw + {n_cats} one-hot)")
log(f"   MobileNetV3-S : 11 bottlenecks (NAS-config) + h-swish + SE")
log(f"   train/val/test windows: {len(X_train_w):,} / {len(X_val_w):,} / {len(X_test_w):,}")
log(f"   params        : {n_params:,}")
log(f"   best_epoch    : {best_epoch}  (val Macro-F1 = {best_f1m:.4f})")
log(f"   best_thr      : {best_thr:.2f}")
log(f"   val@0.5       : Macro-F1={val_05['macro_f1']:.4f}  Binary-F1={val_05['binary_f1']:.4f}")
log(f"   val@{best_thr:.2f}      : Macro-F1={val_thr['macro_f1']:.4f}  Binary-F1={val_thr['binary_f1']:.4f}")
log(f"   test@0.5      : Macro-F1={test_05['macro_f1']:.4f}  Binary-F1={test_05['binary_f1']:.4f}")
log(f"   test@{best_thr:.2f}     : Macro-F1={test_thr['macro_f1']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log("=" * 60)
