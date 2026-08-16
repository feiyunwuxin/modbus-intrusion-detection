#!/usr/bin/env python3
"""
MobileViT (1D) on 19-dim SCADA features.

MobileViT (Mehta & Rastegari, 2021) = MobileNet conv blocks + Transformer blocks + MLP.
For 1D tabular sequences, we adapt:
  - MobileNet block: depthwise-separable conv 1D
  - Transformer block: multi-head self-attention on spatial dimension (time)
  - Inverted residual + linear bottleneck (IRBN)

Adapted to 1D:
  Input (B, F=19, T=16)
  → MobileNet conv blocks (stride 2 for downsampling)
  → Transformer block (treat time dim as sequence)
  → MobileNet conv blocks
  → GAP → FC → logit
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
)

BASE = r"C:\work\Claude\Issue"

X_TR = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada.npy")

EVAL_OUT  = os.path.join(BASE, "evaluation_mobilevit_v3_19dim.txt")
META_OUT  = os.path.join(BASE, "processed_meta_mobilevit_v3_19dim.json")
THR_OUT   = os.path.join(BASE, "best_threshold_mobilevit_v3_19dim.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_mobilevit_v3_19dim.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_mobilevit_v3_19dim.csv")
MODEL_OUT = os.path.join(BASE, "model_mobilevit_v3_19dim.pt")
CM_PNG    = os.path.join(BASE, "confusion_matrix_mobilevit_v3_19dim.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_mobilevit_v3_19dim.png")
HIST_PNG  = os.path.join(BASE, "training_history_mobilevit_v3_19dim.png")

SEED = 42
WINDOW = 16
EPOCHS = 30
BATCH = 512
LR = 5e-4
WD = 1e-5
PATIENCE = 7
GRAD_CLIP = 0.5
DROPOUT = 0.1
CLIP_VAL = 10.0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

# ─────────────────────────────────────────────
# MobileViT 1D Building Blocks
# ─────────────────────────────────────────────
class InvertedResidual(nn.Module):
    """MobileNet-v2 style: 1x1 expand → 3x3 dw → 1x1 project (linear)."""
    def __init__(self, in_ch, out_ch, stride=1, expand=4):
        super().__init__()
        hidden = in_ch * expand
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand != 1:
            layers.append(nn.Conv1d(in_ch, hidden, 1, bias=False))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU6(inplace=True))
        layers += [
            nn.Conv1d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv1d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class TransformerBlock1D(nn.Module):
    """Multi-head self-attention on temporal dimension."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, C, T) → (B, T, C) for attention
        x_t = x.transpose(1, 2)
        x_norm = self.norm1(x_t)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_t = x_t + attn_out
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t.transpose(1, 2)  # back to (B, C, T)


class MobileViTBlock1D(nn.Module):
    """One MobileViT block: local conv (IRBN) + global transformer."""
    def __init__(self, in_ch, out_ch, num_heads=4, mlp_ratio=2.0, dropout=DROPOUT):
        super().__init__()
        # Local representation: IRBN
        self.local = InvertedResidual(in_ch, out_ch, stride=1, expand=2)
        # Global representation: transformer
        self.global_block = TransformerBlock1D(out_ch, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        x = self.local(x)
        x = self.global_block(x)
        x = self.fusion(x)
        return x


class MobileViT1D(nn.Module):
    def __init__(self, in_ch=19, dims=(32, 64, 96), num_heads=4, dropout=DROPOUT):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, dims[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU6(inplace=True),
        )
        # Stage 1: MobileNet
        self.stage1 = InvertedResidual(dims[0], dims[0], stride=1, expand=2)
        # Stage 2: MobileViT block
        self.stage2 = MobileViTBlock1D(dims[0], dims[1], num_heads=num_heads)
        # Stage 3: MobileNet downsample
        self.stage3 = InvertedResidual(dims[1], dims[2], stride=1, expand=2)
        # Head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(dims[2], 32),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)


# ─────────────────────────────────────────────
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log("STEP 1: load + window 19-dim ...")
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
X_tr_w, y_tr_w = make_windows(X_train, y_train, WINDOW)
X_va_w, y_va_w = make_windows(X_val,   y_val,   WINDOW)
X_te_w, y_te_w = make_windows(X_test,  y_test,  WINDOW)
log(f"   windowed: train={X_tr_w.shape} val={X_va_w.shape} test={X_te_w.shape}")

X_tr_w = np.clip(X_tr_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_va_w = np.clip(X_va_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_te_w = np.clip(X_te_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
log(f"   input: (B, F={X_tr_w.shape[1]}, T={X_tr_w.shape[2]})")

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_w), torch.from_numpy(y_tr_w)),
                          batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_va_w), torch.from_numpy(y_va_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_te_w), torch.from_numpy(y_te_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
log("STEP 2: build MobileViT-1D ...")
model = MobileViT1D(in_ch=19, dims=(32, 64, 96), num_heads=4, dropout=DROPOUT)
n_params = sum(p.numel() for p in model.parameters())
log(f"   params: {n_params:,}")

n_pos = int((y_tr_w == 1).sum()); n_neg = int((y_tr_w == 0).sum())
pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ─────────────────────────────────────────────
log("STEP 3: training ...")
@torch.no_grad()
def predict(loader):
    model.eval()
    probs = []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(probs)

best_f1m, best_state, best_epoch = -1, None, -1
epochs_no_improve = 0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_n = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        train_n += xb.size(0)
    train_loss /= train_n

    val_prob = predict(val_loader)
    val_f1m = f1_score(y_va_w, (val_prob >= 0.5).astype(int), average="macro")
    cur_lr = optimizer.param_groups[0]["lr"]
    history.append((epoch, train_loss, val_f1m))
    log(f"   ep{epoch:2d}  loss={train_loss:.4f}  val f1m={val_f1m:.4f}  lr={cur_lr:.2e}")
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
log(f"   best epoch={best_epoch}  best val f1m={best_f1m:.4f}  train_time={train_time:.1f}s")

torch.save({"state_dict": best_state, "n_params": n_params, "best_epoch": best_epoch,
            "preprocessing": "v3_19dim"}, MODEL_OUT)

# ─────────────────────────────────────────────
log("STEP 4: threshold tuning ...")
val_prob  = predict(val_loader)
test_prob = predict(test_loader)
thresholds = np.arange(0.10, 0.91, 0.01)
sweep = [(float(t), float(f1_score(y_va_w, (val_prob >= t).astype(int), average="macro"))) for t in thresholds]
best_idx = int(np.argmax([s[1] for s in sweep]))
best_thr = sweep[best_idx][0]
log(f"   best threshold = {best_thr:.2f}")

with open(THR_OUT, "w") as f:
    json.dump({"best_threshold": best_thr, "val_macro_f1_at_best": sweep[best_idx][1],
               "metric": "macro_f1"}, f, indent=2)

# ─────────────────────────────────────────────
log("STEP 5: evaluation ...")
def evaluate(prob, y, thr, label):
    pred = (prob >= thr).astype(int)
    return {
        "label": label, "threshold": float(thr),
        "macro_f1":  float(f1_score(y, pred, average="macro")),
        "binary_f1": float(f1_score(y, pred, average="binary")),
        "accuracy":  float((pred == y).mean()),
        "roc_auc":   float(roc_auc_score(y, prob)),
        "pr_auc":    float(average_precision_score(y, prob)),
        "report":    classification_report(y, pred, digits=4),
        "cm":        confusion_matrix(y, pred).tolist(),
    }

val_thr  = evaluate(val_prob,  y_va_w, best_thr, f"val@{best_thr:.2f}")
test_thr = evaluate(test_prob, y_te_w, best_thr, f"test@{best_thr:.2f}")

pd.DataFrame({"y_true": y_va_w, "prob": val_prob,
              "pred": (val_prob >= best_thr).astype(int)}).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": y_te_w, "prob": test_prob,
              "pred": (test_prob >= best_thr).astype(int)}).to_csv(PRED_TEST, index=False)

meta = {
    "model": "MobileViT 1D v3 19-dim", "n_features": 19, "n_params": int(n_params),
    "window": WINDOW, "best_epoch": best_epoch, "best_threshold": best_thr,
    "train_time_seconds": float(train_time),
    "metrics": {
        "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

lines = [
    "=" * 72, "EVALUATION REPORT — MobileViT 1D v3 19-dim SCADA",
    "=" * 72,
    f"Window: {WINDOW}  |  Params: {n_params:,}  |  Train time: {train_time:.1f}s",
    f"Tuned threshold: {best_thr:.2f}", "",
]
for ev in [val_thr, test_thr]:
    lines.append("-" * 72)
    lines.append(f"[{ev['label']}]  thr={ev['threshold']:.2f}  "
                 f"Macro-F1={ev['macro_f1']:.4f}  Binary-F1={ev['binary_f1']:.4f}  "
                 f"Acc={ev['accuracy']:.4f}  ROC-AUC={ev['roc_auc']:.4f}  PR-AUC={ev['pr_auc']:.4f}")
    lines.append(ev["report"])
    lines.append(f"CM: {ev['cm']}")
    lines.append("")
with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(test_thr["cm"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Attack"]); ax.set_yticklabels(["Normal","Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center", fontsize=14,
                    color="red" if cm[i,j] < cm.max()/2 else "white")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)

    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for prob, yy, lbl in [(val_prob, y_va_w, "val"), (test_prob, y_te_w, "test")]:
        p, r, _ = precision_recall_curve(yy, prob); ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, prob):.3f}")
        fpr, tpr, _ = roc_curve(yy, prob); ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, prob):.3f}")
    ax1.legend(); ax1.grid(alpha=0.3); ax2.legend(); ax2.grid(alpha=0.3); ax2.plot([0,1],[0,1],"k--",alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h[0] for h in history]; loss = [h[1] for h in history]; f1m = [h[2] for h in history]
    ax1.plot(ep, loss, "o-", color="C0"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3)
    ax2.plot(ep, f1m, "o-", color="C1"); ax2.axvline(best_epoch, color="grey", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Macro-F1"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HIST_PNG, dpi=120); plt.close(fig)
    log("   plots saved")
except Exception as e:
    log(f"   plots skipped: {e}")

log("=" * 60)
log(f"DONE MobileViT v3 19-dim.")
log(f"   test@{best_thr:.2f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Bin-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   params: {n_params:,}  |  train time: {train_time:.1f}s")
log("=" * 60)