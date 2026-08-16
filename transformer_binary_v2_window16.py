#!/usr/bin/env python3
"""
1D Transformer Encoder for IDS — pure self-attention stack on v2 data.

Architecture:
  Input: (B, T=16, F=27)         # 27 SCADA-aware features
  Input proj: Linear(F → d_model) + LayerNorm + Dropout
  Positional encoding: learned (T, d_model)
  N Transformer encoder layers:
    - Multi-Head Self-Attention
    - FFN (d_model → 4·d_model → d_model) with GELU
    - Pre-LN residual + Dropout
  Attention pool: learnable query → weighted sum
  Classifier: Linear(d_model → 32 → 1)

This is the Transformer analog of LSTM+Attention.
"""

import os, time, json, math
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

TAG = "transformer_v2_window16"
BASE = r"C:\work\Claude\Issue"

X_TR = os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")

MODEL_OUT = os.path.join(BASE, f"model_{TAG}.pt")
EVAL_OUT = os.path.join(BASE, f"evaluation_{TAG}.txt")
META_OUT = os.path.join(BASE, f"processed_meta_{TAG}.json")
THR_OUT = os.path.join(BASE, f"best_threshold_{TAG}.json")
PRED_VAL = os.path.join(BASE, f"predictions_val_{TAG}.csv")
PRED_TEST = os.path.join(BASE, f"predictions_test_{TAG}.csv")
CM_PNG = os.path.join(BASE, f"confusion_matrix_{TAG}.png")
PRROC_PNG = os.path.join(BASE, f"pr_roc_{TAG}.png")
HIST_PNG = os.path.join(BASE, f"training_history_{TAG}.png")

# Hyperparameters
SEED         = 42
WINDOW       = 16
EPOCHS       = 40
BATCH        = 256                  # smaller batch for transformer stability
LR           = 3e-4                 # lower than LSTM/TCN
WD           = 1e-4
PATIENCE     = 8
GRAD_CLIP    = 1.0
CLIP_VAL     = 10.0
LABEL_SMOOTH = 0.05
# Transformer config
D_MODEL      = 128
N_HEADS      = 4
HEAD_DIM     = D_MODEL // N_HEADS   # = 32
N_LAYERS     = 3
FFN_DIM      = D_MODEL * 4          # 512
DROPOUT      = 0.3
ATTN_DROPOUT = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")
print(f"[device] {device}  [tag] {TAG}")

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ─────────────────────────────────────────────
log("STEP 1: load v2 windowed npy ...")
X_train = np.load(X_TR).astype(np.float32)
X_val   = np.load(X_VA).astype(np.float32)
X_test  = np.load(X_TE).astype(np.float32)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)
log(f"   shapes: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
N_FEATURES = X_train.shape[-1]
log(f"   features per step: {N_FEATURES}  (v2 data)")

X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL)
X_val   = np.clip(X_val,   -CLIP_VAL, CLIP_VAL)
X_test  = np.clip(X_test,  -CLIP_VAL, CLIP_VAL)

# ─────────────────────────────────────────────
log("STEP 2: DataLoaders ...")
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Step 3 — 1D Transformer
# ─────────────────────────────────────────────
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv   = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj  = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # each (B, H, T, Dh)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block."""
    def __init__(self, d_model, n_heads, ffn_dim, attn_dropout, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, x):
        scores = (x * self.query).sum(dim=-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return self.norm(context)


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 ffn_dim=FFN_DIM, attn_dropout=ATTN_DROPOUT, dropout=DROPOUT, max_len=WINDOW):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)
        self.pos = LearnedPositionalEncoding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, attn_dropout, dropout)
            for _ in range(n_layers)
        ])
        self.pool = AttentionPool(d_model)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_drop(x)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


log("STEP 3: build model ...")
model = TransformerClassifier(in_dim=N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
log(f"   d_model={D_MODEL}  n_heads={N_HEADS}  n_layers={N_LAYERS}  ffn_dim={FFN_DIM}")
log(f"   dropout={DROPOUT}  attn_dropout={ATTN_DROPOUT}  label_smoothing={LABEL_SMOOTH}")
log(f"   model params: {n_params:,}")

n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
log(f"   pos_weight = {pos_weight.item():.3f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

# Label-smoothed positive weight: targets are 1-LABEL_SMOOTH/2 and LABEL_SMOOTH/2
def smooth_bce_loss(logits, targets, pos_weight, label_smoothing=LABEL_SMOOTH):
    targets = targets.float()
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    # weighted BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # pos_weight applied to positives
    weight = torch.where(targets >= 0.5, pos_weight, torch.ones_like(pos_weight))
    return (bce * weight).mean()


loss_fn = lambda logits, yb: smooth_bce_loss(logits, yb, pos_weight, LABEL_SMOOTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
# Warmup + cosine schedule
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(0.1 * total_steps)
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ─────────────────────────────────────────────
log("STEP 4: training ...")

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
        loss = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
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

torch.save({"state_dict": best_state, "n_features": N_FEATURES, "window": WINDOW,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "ffn_dim": FFN_DIM, "n_params": n_params,
            "best_epoch": best_epoch, "best_val_f1m": best_f1m,
            "preprocessing": "v2_scada", "label_smooth": LABEL_SMOOTH}, MODEL_OUT)

# ─────────────────────────────────────────────
log("STEP 5: threshold tuning on val ...")
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
log(f"   best threshold = {best_thr:.2f}  →  val Macro-F1 = {best_vf1m:.4f}")

with open(THR_OUT, "w") as f:
    json.dump({"best_threshold": best_thr, "val_macro_f1_at_best": best_vf1m,
               "val_binary_f1_at_best": float(sweep_df.loc[best_idx, "val_binary_f1"]),
               "val_macro_f1_at_0.5":  f1m_at_05, "metric": "macro_f1"}, f, indent=2)

# ─────────────────────────────────────────────
log("STEP 6: evaluation ...")
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
log("STEP 7: save artifacts ...")
pd.DataFrame({"y_true": val_lbl_full, "prob_attack": val_prob_full,
              "pred_05": (val_prob_full >= 0.5).astype(int),
              "pred_tuned": (val_prob_full >= best_thr).astype(int),
              }).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": test_lbl_full, "prob_attack": test_prob_full,
              "pred_05": (test_prob_full >= 0.5).astype(int),
              "pred_tuned": (test_prob_full >= best_thr).astype(int),
              }).to_csv(PRED_TEST, index=False)

meta = {
    "model": f"1D Transformer on v2", "tag": TAG, "preprocessing": "v2_scada",
    "window": WINDOW, "n_features_per_step": int(N_FEATURES),
    "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
    "ffn_dim": FFN_DIM, "label_smooth": LABEL_SMOOTH,
    "n_params": int(n_params), "train_time_seconds": float(train_time),
    "best_epoch": int(best_epoch), "best_threshold": best_thr,
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
    "=" * 72, f"EVALUATION REPORT — 1D Transformer Encoder on v2 (SCADA features)",
    "=" * 72,
    f"Preprocessing: v2 (27 dim)",
    f"Architecture: Linear(27→{D_MODEL}) + LearnedPos + {N_LAYERS}x Pre-LN Transformer blocks (heads={N_HEADS}, ffn={FFN_DIM}) + AttentionPool + FC",
    f"Training: AdamW lr={LR} warmup+cosine, label_smoothing={LABEL_SMOOTH}, batch={BATCH}",
    f"Features per step: {N_FEATURES}  |  Params: {n_params:,}",
    f"Train/Val/Test windows: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}",
    f"Best epoch: {best_epoch}  |  Best val Macro-F1: {best_f1m:.4f}  |  Train time: {train_time:.1f}s",
    f"Tuned threshold: {best_thr:.2f}", "",
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
log(f"   test@{best_thr:.2f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Bin-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   vs TCN v4 v2 baseline (0.8367):  ΔMacro-F1 = {test_thr['macro_f1'] - 0.8367:+.4f}")
log(f"   vs LSTM+Att v2 baseline (0.8168): ΔMacro-F1 = {test_thr['macro_f1'] - 0.8168:+.4f}")
log("=" * 60)
