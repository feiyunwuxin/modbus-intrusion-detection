#!/usr/bin/env python3
"""
LSTM + Multi-Head Self-Attention on v2 (SCADA protocol features) data.

Same architecture as the v1 LSTM+Attention, but input is 27-dim.
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

TAG = "lstm_attention_v2_window16"
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
HIDDEN       = 64
LSTM_LAYERS  = 1
BIDIR        = True
N_HEADS      = 4
HEAD_DIM     = 32
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
log(f"   features per step: {N_FEATURES}  (was 44 in v1)")
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
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv   = nn.Linear(dim, dim * 3, bias=True)
        self.proj  = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.residual_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.proj(out)
        out = self.residual_drop(out)
        return self.norm(x + out)


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        scores = (x * self.query).sum(dim=-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return self.norm(context)


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, lstm_layers=LSTM_LAYERS, bidir=BIDIR,
                 n_heads=N_HEADS, head_dim=HEAD_DIM, attn_dropout=ATTN_DROPOUT, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidir, dropout=0.0)
        out_dim = hidden * (2 if bidir else 1)
        self.proj_in = nn.Linear(out_dim, n_heads * head_dim)
        self.norm_in = nn.LayerNorm(n_heads * head_dim)
        self.attn = MultiHeadSelfAttention(dim=n_heads * head_dim, n_heads=n_heads, dropout=attn_dropout)
        self.pool = AttentionPool(dim=n_heads * head_dim)
        self.fc1 = nn.Linear(n_heads * head_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.proj_in(h)
        h = self.norm_in(h)
        h = self.attn(h)
        h = self.pool(h)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)


log("STEP 3: build model ...")
model = LSTMAttentionClassifier(in_dim=N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
log(f"   BiLSTM hidden={HIDDEN} bidir={BIDIR}")
log(f"   Multi-Head Self-Attn: n_heads={N_HEADS} head_dim={HEAD_DIM} (total dim={N_HEADS*HEAD_DIM})")
log(f"   model params: {n_params:,}")

n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
log(f"   pos_weight = {pos_weight.item():.3f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

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

torch.save({"state_dict": best_state, "n_features": N_FEATURES, "window": WINDOW,
            "hidden": HIDDEN, "lstm_layers": LSTM_LAYERS, "bidir": BIDIR,
            "n_heads": N_HEADS, "head_dim": HEAD_DIM, "n_params": n_params,
            "best_epoch": best_epoch, "best_val_f1m": best_f1m,
            "preprocessing": "v2_scada"}, MODEL_OUT)

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
    "model": f"LSTM+Attention on v2", "tag": TAG, "preprocessing": "v2_scada",
    "window": WINDOW, "n_features_per_step": int(N_FEATURES),
    "lstm_hidden": HIDDEN, "lstm_layers": LSTM_LAYERS, "bidirectional": BIDIR,
    "n_heads": N_HEADS, "head_dim": HEAD_DIM,
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
    "=" * 72, f"EVALUATION REPORT — LSTM+Multi-Head Self-Attention on v2 (SCADA features)",
    "=" * 72,
    f"Preprocessing: v2 (27 dim: 19 row-level + 8 window-aggregates)",
    f"Architecture: BiLSTM (h={HIDDEN}, layers={LSTM_LAYERS}, bidir={BIDIR}) + "
    f"Multi-Head Self-Attention ({N_HEADS} heads x {HEAD_DIM} dim) + LayerNorm + Attention Pool + FC",
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
log(f"   vs LSTM+Att v1 baseline (0.8159): ΔMacro-F1 = {test_thr['macro_f1'] - 0.8159:+.4f}")
log(f"   vs LSTM+Att v1 baseline (0.9088): ΔPR-AUC   = {test_thr['pr_auc'] - 0.9088:+.4f}")
log("=" * 60)
