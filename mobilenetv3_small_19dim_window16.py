#!/usr/bin/env python3
"""
MobileNetV3-Small (Howard et al. 2019) — 1D adaptation on 19-dim SCADA, window=16.

Key innovations vs MobileNetV2:
  1. h-swish activation: x * ReLU6(x+3)/6 — smoother than ReLU, helps quantization
  2. Built-in Squeeze-and-Excitation (SE) in every block
  3. NAS-optimized block configuration: varying expansion ratios (3, 3, 3, 6, 6, 6, 6)
  4. Different activation per block: early blocks use ReLU, later use h-swish

Block structure:
  Input  ─→ 1x1 Conv (expand t=variable)  ─→ BN ─→ activation
         ─→ 3x3 Depthwise (stride=1)        ─→ BN ─→ activation
         ─→ SE (channel attention)           ─→ 1x1 Conv (project) ─→ BN
         ─→ ⊕ residual (if stride=1 & shape match)

h-swish implementation:
  hswish(x) = x * ReLU6(x + 3) / 6
  Equivalent to x * sigmoid(x) in spirit, but cheaper and quantization-friendly.

Reference: https://arxiv.org/abs/1905.02244 (MobileNetV3)
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

TAG = "mobilenetv3_small_19dim_window16"
BASE = r"C:\work\Claude\Issue"

X_TR = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada.npy")

MODEL_OUT = os.path.join(BASE, f"model_{TAG}.pt")
EVAL_OUT  = os.path.join(BASE, f"evaluation_{TAG}.txt")
META_OUT  = os.path.join(BASE, f"processed_meta_{TAG}.json")
THR_OUT   = os.path.join(BASE, f"best_threshold_{TAG}.json")
PRED_VAL  = os.path.join(BASE, f"predictions_val_{TAG}.csv")
PRED_TEST = os.path.join(BASE, f"predictions_test_{TAG}.csv")
CM_PNG    = os.path.join(BASE, f"confusion_matrix_{TAG}.png")
PRROC_PNG = os.path.join(BASE, f"pr_roc_{TAG}.png")
HIST_PNG  = os.path.join(BASE, f"training_history_{TAG}.png")

SEED         = 42
WINDOW       = 16
EPOCHS       = 35
BATCH        = 512
LR           = 5e-4
WD           = 1e-5
PATIENCE     = 7
GRAD_CLIP    = 0.5
DROPOUT      = 0.2
CLIP_VAL     = 10.0
KERNEL_SIZE  = 3
# MobileNetV3-Small (adapted to 1D, window=16):
#   (out_channels, exp_ratio, use_se, activation)
#   - First 3 blocks: ReLU (no h-swish) — matches paper
#   - Later blocks: h-swish (smoother gradients)
#   - All blocks: SE + stride=1 (sequence too short for downsampling)
MNV3S_BLOCKS = [
    # c_out, exp, SE,  activation
    (16,  1, False, "relu"),    # first block no expansion, no SE (matches paper)
    (24,  3, False, "relu"),
    (32,  3, True,  "relu"),
    (48,  6, True,  "hswish"),
    (64,  6, True,  "hswish"),
    (96,  6, True,  "hswish"),
]
INIT_CHANNELS = 16
SE_REDUCTION  = 4   # SE bottleneck (paper uses 4 for V3-small)
HEAD_HIDDEN   = 32

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
log(f"   features per step: {N_FEATURES}  (19-dim SCADA)")

X_train_w = np.clip(X_train_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_val_w   = np.clip(X_val_w,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_test_w  = np.clip(X_test_w,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
log(f"   final input shape: {X_train_w.shape}")

# ─────────────────────────────────────────────
log("STEP 3: DataLoaders ...")
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_w), torch.from_numpy(y_train_w)),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val_w),   torch.from_numpy(y_val_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test_w),  torch.from_numpy(y_test_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Step 4 — MobileNetV3-Small 1D
# ─────────────────────────────────────────────
class HSwish(nn.Module):
    """h-swish: x * ReLU6(x+3)/6 — smoother than ReLU, quantization-friendly."""
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


def get_activation(name):
    return nn.ReLU(inplace=True) if name == "relu" else HSwish()


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for MobileNetV3."""
    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden, channels, 1, bias=True)

    def forward(self, x):
        s = self.gap(x)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class InvertedResidualV3(nn.Module):
    """MobileNetV3 bottleneck: expand -> depthwise -> SE -> project [+ residual]."""
    def __init__(self, in_ch, out_ch, stride, expansion, kernel_size, use_se, activation_name):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.use_residual = (stride == 1) and (in_ch == out_ch)
        hidden = in_ch * expansion
        layers = []
        if expansion != 1:
            layers += [
                nn.Conv1d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm1d(hidden),
                get_activation(activation_name),
            ]
        # depthwise
        pad = (kernel_size - 1) // 2
        layers += [
            nn.Conv1d(hidden, hidden, kernel_size, stride=stride, padding=pad, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            get_activation(activation_name),
        ]
        # SE (only in some blocks per paper)
        if use_se:
            layers.append(SEBlock(hidden, reduction=SE_REDUCTION))
        # project
        layers += [
            nn.Conv1d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV3Small_1D(nn.Module):
    def __init__(self, in_ch, init_channels=INIT_CHANNELS, blocks=MNV3S_BLOCKS,
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT, head_hidden=HEAD_HIDDEN):
        super().__init__()
        # Stem
        pad = (kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, init_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(init_channels),
            HSwish(),   # V3 uses h-swish in stem
        )
        # Inverted residual blocks (variable expansion, optional SE, mixed activations)
        layers = []
        in_c = init_channels
        for c_out, exp, use_se, act in blocks:
            layers.append(InvertedResidualV3(in_c, c_out, stride=1, expansion=exp,
                                             kernel_size=kernel_size, use_se=use_se,
                                             activation_name=act))
            in_c = c_out
        self.blocks = nn.Sequential(*layers)
        # Head: V3 uses 1x1 conv to expand channels before GAP (paper: Conv2d 1x1 -> 1024)
        self.head_conv = nn.Sequential(
            nn.Conv1d(in_c, head_hidden * 4, 1, bias=False),
            nn.BatchNorm1d(head_hidden * 4),
            HSwish(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(head_hidden * 4, head_hidden)
        self.fc2 = nn.Linear(head_hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


# ─────────────────────────────────────────────
log("STEP 4: build model ...")
model = MobileNetV3Small_1D(in_ch=N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
log(f"   MobileNetV3-Small 1D params: {n_params:,}")
log(f"   blocks: {len(MNV3S_BLOCKS)} bottleneck, channels: {INIT_CHANNELS} → {' → '.join(str(c) for c, _, _, _ in MNV3S_BLOCKS)}")
log(f"   SE blocks: {sum(1 for _, _, se, _ in MNV3S_BLOCKS if se)}  h-swish blocks: {sum(1 for _, _, _, a in MNV3S_BLOCKS if a == 'hswish')}")

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
    "n_params": n_params, "blocks": MNV3S_BLOCKS,
    "init_channels": INIT_CHANNELS, "kernel_size": KERNEL_SIZE,
    "se_reduction": SE_REDUCTION,
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

meta = {
    "model": f"MobileNetV3Small_1D_{TAG}",
    "tag": TAG,
    "family": "Mobile",
    "base_arch": "MobileNetV3-Small (Howard 2019)",
    "key_innovations": ["h-swish", "built-in SE", "NAS-optimized blocks", "variable expansion"],
    "window": WINDOW,
    "n_features_per_step": int(N_FEATURES),
    "init_channels": INIT_CHANNELS,
    "blocks": [list(b) for b in MNV3S_BLOCKS],
    "kernel_size": KERNEL_SIZE,
    "se_reduction": SE_REDUCTION,
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
    f"EVALUATION REPORT — MobileNetV3-Small_1D ({TAG})",
    "=" * 72,
    f"Architecture: MobileNetV3-Small 1D (Howard 2019) — h-swish + SE + NAS blocks",
    f"Stem: Conv1d({N_FEATURES}→{INIT_CHANNELS}, k={KERNEL_SIZE}) → BN → h-swish",
    f"Blocks: {' → '.join(f'IR(c={c},e={e},SE={se},act={a})' for c, e, se, a in MNV3S_BLOCKS)}",
    f"Head: 1x1 Conv → BN → h-swish → GAP → FC({HEAD_HIDDEN*4}→{HEAD_HIDDEN}) → ReLU → Drop → FC({HEAD_HIDDEN}→1)",
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
log(f"   train time: {train_time:.1f}s  |  params: {n_params:,}")
log("=" * 60)