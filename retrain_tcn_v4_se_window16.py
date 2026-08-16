#!/usr/bin/env python3
"""Re-train TCN v4 +SE on the v1 (44-dim) data.

Pipeline (matches original tcn_v4_se training, recovered from
tcn_v4_train_log.txt + evaluation_tcn_v4_se_window16.txt):
  1. Load 17-dim row-level v1 data
  2. Window into 16-step sequences  (12014/1716/3432 train/val/test windows)
  3. Drop function col (col 1), one-hot 28 categories (no drop_first) -> 44 dim
  4. Train TCN v4 +SE (3 blocks, channels=64, kernel=3, dilations=[1,2,4], SE r=8)
  5. Threshold tuning, full evaluation, save all artifacts
  6. Print full per-layer parameter breakdown
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

# ── paths & config ───────────────────────────────────────────────────────
TAG = "v4_se_window16_retrain"
BASE = r"C:\work\Claude\Issue"

X_TR = os.path.join(BASE, "X_train_binary.npy")
X_VA = os.path.join(BASE, "X_val_binary.npy")
X_TE = os.path.join(BASE, "X_test_binary.npy")
Y_TR = os.path.join(BASE, "y_train_binary.npy")
Y_VA = os.path.join(BASE, "y_val_binary.npy")
Y_TE = os.path.join(BASE, "y_test_binary.npy")

MODEL_OUT = os.path.join(BASE, f"model_tcn_{TAG}.pt")
EVAL_OUT  = os.path.join(BASE, f"evaluation_tcn_{TAG}.txt")
META_OUT  = os.path.join(BASE, f"processed_meta_tcn_{TAG}.json")
THR_OUT   = os.path.join(BASE, f"best_threshold_tcn_{TAG}.json")
PRED_VAL  = os.path.join(BASE, f"predictions_val_tcn_{TAG}.csv")
PRED_TEST = os.path.join(BASE, f"predictions_test_tcn_{TAG}.csv")
CM_PNG    = os.path.join(BASE, f"confusion_matrix_tcn_{TAG}.png")
PRROC_PNG = os.path.join(BASE, f"pr_roc_tcn_{TAG}.png")
HIST_PNG  = os.path.join(BASE, f"training_history_tcn_{TAG}.png")
LOG_OUT   = os.path.join(BASE, f"tcn_{TAG}_train_log.txt")

# TCN v4 hyperparameters (identical to v4 baseline)
SEED, WINDOW, EPOCHS, BATCH = 42, 16, 35, 512
LR, WD, PATIENCE = 5e-4, 1e-5, 7
GRAD_CLIP, DROPOUT, CLIP_VAL = 0.5, 0.3, 10.0
N_BLOCKS, CHANNELS, KERNEL_SIZE, DILATIONS, SE_REDUCTION = 3, 64, 3, [1, 2, 4], 8

FUNCTION_COL = 1  # col 1 in v1 = function (28 unique values)

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cpu")
print(f"[device] {device}  [tag] {TAG}")

t0 = time.time()
def log(msg):
    s = f"[{time.time()-t0:7.1f}s] {msg}"
    print(s, flush=True)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(s + "\n")

# wipe log
open(LOG_OUT, "w").close()

# ─────────────────────────────────────────────
log("STEP 1: load v1 row-level npy (17-dim) ...")
X_train_raw = np.load(X_TR).astype(np.float32)
X_val_raw   = np.load(X_VA).astype(np.float32)
X_test_raw  = np.load(X_TE).astype(np.float32)
y_train     = np.load(Y_TR).astype(np.int64)
y_val       = np.load(Y_VA).astype(np.int64)
y_test      = np.load(Y_TE).astype(np.int64)
log(f"   raw shapes: train={X_train_raw.shape} val={X_val_raw.shape} test={X_test_raw.shape}")
N_RAW_FEATURES = X_train_raw.shape[-1]
log(f"   raw features: {N_RAW_FEATURES}")

# ─────────────────────────────────────────────
log("STEP 2: window every 16 rows into 1 sequence ...")
def make_windows(X, y, w=WINDOW):
    n = (len(X) // w) * w
    X = X[:n]
    y = y[:n]
    Xw = X.reshape(-1, w, X.shape[-1])
    yw = y.reshape(-1, w)
    # window label = "any row in window is positive" (matches original tcn_v4)
    # attacks cluster in time, so 51% of windows contain >=1 attack row
    yw_lab = (yw.sum(axis=1) >= 1).astype(np.int64)
    return Xw, yw_lab

X_train_w, y_train_w = make_windows(X_train_raw, y_train)
X_val_w,   y_val_w   = make_windows(X_val_raw,   y_val)
X_test_w,  y_test_w  = make_windows(X_test_raw,  y_test)
log(f"   train windows: {X_train_w.shape}  val: {X_val_w.shape}  test: {X_test_w.shape}")
log(f"   window-label dist: train {y_train_w.mean():.2%}  val {y_val_w.mean():.2%}  test {y_test_w.mean():.2%}")

# ─────────────────────────────────────────────
log("STEP 3: one-hot encode function column at each timestep ...")
# extract function code at each row
func_train = X_train_w[..., FUNCTION_COL].astype(np.int64).ravel()
func_val   = X_val_w[...,   FUNCTION_COL].astype(np.int64).ravel()
func_test  = X_test_w[...,  FUNCTION_COL].astype(np.int64).ravel()
all_funcs  = np.unique(np.concatenate([func_train, func_val, func_test]))
n_cats     = len(all_funcs)
log(f"   function unique codes: {n_cats}  values: {sorted(int(x) for x in all_funcs)}")
assert n_cats == 28, f"expected 28 function codes, got {n_cats}"

# map raw function code → dense index 0..27
code2idx = {int(c): i for i, c in enumerate(all_funcs)}
idx_train = np.array([code2idx[int(c)] for c in func_train], dtype=np.int64)
idx_val   = np.array([code2idx[int(c)] for c in func_val],   dtype=np.int64)
idx_test  = np.array([code2idx[int(c)] for c in func_test],  dtype=np.int64)

# one-hot (no drop_first → 28 columns)
def onehot(idx, n=n_cats):
    out = np.zeros((len(idx), n), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out

oh_train = onehot(idx_train).reshape(X_train_w.shape[0], WINDOW, n_cats)
oh_val   = onehot(idx_val).reshape(X_val_w.shape[0],   WINDOW, n_cats)
oh_test  = onehot(idx_test).reshape(X_test_w.shape[0],  WINDOW, n_cats)

# drop function col, concat one-hot
def build(Xw, oh):
    other = np.delete(Xw, FUNCTION_COL, axis=-1)            # (N, 16, 16)
    out  = np.concatenate([other, oh], axis=-1)             # (N, 16, 44)
    return out

X_train_44 = build(X_train_w, oh_train)
X_val_44   = build(X_val_w,   oh_val)
X_test_44  = build(X_test_w,  oh_test)
log(f"   per-step feature dim: {X_train_44.shape[-1]}  (16 numeric + 28 one-hot)")
log(f"   raw label dist: train {(y_train==1).mean():.2%} val {(y_val==1).mean():.2%} test {(y_test==1).mean():.2%}")

# clip + permute to (B, F, W) for Conv1D
X_train = np.clip(X_train_44, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1).astype(np.float32)
X_val   = np.clip(X_val_44,   -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1).astype(np.float32)
X_test  = np.clip(X_test_44,  -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1).astype(np.float32)
log(f"   final TCN input shape: {X_train.shape}")

# ─────────────────────────────────────────────
log("STEP 4: DataLoaders ...")
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_w)),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

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
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
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
log("STEP 5: build model ...")
model = TCNClassifierSE(in_ch=44)
n_params = sum(p.numel() for p in model.parameters())
log(f"   TCN config: 3 blocks, channels=64, kernel=3, dilations=[1,2,4]")
log(f"   SE reduction: 8  (bottleneck = 8)")
log(f"   Effective receptive field: 29  (window=16, RF/W = 1.8x)")
log(f"   model params: {n_params:,}")

# print full per-layer parameter table
log("")
log("FULL PARAMETER BREAKDOWN:")
log(f"   {'name':<55} {'shape':<25} {'numel':>10}")
log("   " + "-"*93)
total_check = 0
for name, p in model.named_parameters():
    log(f"   {name:<55} {str(list(p.shape)):<25} {p.numel():>10,}")
    total_check += p.numel()
log("   " + "-"*93)
log(f"   {'TOTAL':<55} {'':<25} {total_check:>10,}")
log("")

n_pos = int((y_train_w == 1).sum())
n_neg = int((y_train_w == 0).sum())
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
log(f"   pos_weight = {pos_weight.item():.3f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ─────────────────────────────────────────────
log("STEP 6: training ...")
@torch.no_grad()
def predict_probs(loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb)).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)

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
        best_f1m, best_epoch = val_f1m, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            log(f"   early stop at epoch {epoch}")
            break

train_time = time.time() - t0
model.load_state_dict(best_state)
log(f"   best epoch = {best_epoch}  best val Macro-F1 = {best_f1m:.4f}  train_time={train_time:.1f}s")

torch.save({"state_dict": best_state, "n_features": 44, "window": WINDOW,
            "n_blocks": N_BLOCKS, "channels": CHANNELS, "kernel_size": KERNEL_SIZE,
            "dilations": DILATIONS, "receptive_field": 29, "n_params": n_params,
            "se_reduction": SE_REDUCTION, "best_epoch": best_epoch,
            "best_val_f1m": best_f1m}, MODEL_OUT)

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
    json.dump({"best_threshold": best_thr, "val_macro_f1_at_best": best_vf1m,
               "val_binary_f1_at_best": float(sweep_df.loc[best_idx, "val_binary_f1"]),
               "val_macro_f1_at_0.5":  f1m_at_05, "metric": "macro_f1"}, f, indent=2)

# ─────────────────────────────────────────────
log("STEP 8: evaluation ...")
def evaluate(y_true, prob, thr, label):
    pred = (prob >= thr).astype(int)
    return {"label": label, "threshold": float(thr),
            "macro_f1":  float(f1_score(y_true, pred, average="macro")),
            "binary_f1": float(f1_score(y_true, pred, average="binary")),
            "accuracy":  float((pred == y_true).mean()),
            "roc_auc":   float(roc_auc_score(y_true, prob)),
            "pr_auc":    float(average_precision_score(y_true, prob)),
            "report":    classification_report(y_true, pred, digits=4),
            "cm":        confusion_matrix(y_true, pred).tolist()}

val_05   = evaluate(val_lbl_full,  val_prob_full,  0.5,      "val@0.5")
val_thr  = evaluate(val_lbl_full,  val_prob_full,  best_thr, f"val@{best_thr:.2f}")
test_05  = evaluate(test_lbl_full, test_prob_full, 0.5,      "test@0.5")
test_thr = evaluate(test_lbl_full, test_prob_full, best_thr, f"test@{best_thr:.2f}")

# ─────────────────────────────────────────────
log("STEP 9: save artifacts ...")
pd.DataFrame({"y_true": val_lbl_full, "prob_attack": val_prob_full,
              "pred_05": (val_prob_full >= 0.5).astype(int),
              "pred_tuned": (val_prob_full >= best_thr).astype(int)}).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": test_lbl_full, "prob_attack": test_prob_full,
              "pred_05": (test_prob_full >= 0.5).astype(int),
              "pred_tuned": (test_prob_full >= best_thr).astype(int)}).to_csv(PRED_TEST, index=False)

meta = {"model": f"TCN v4 +SE on v1 (44-dim) data",
        "tag": TAG, "preprocessing": "v1", "window": WINDOW,
        "n_features_per_step": 44,
        "n_params": int(n_params), "train_time_seconds": float(train_time),
        "best_epoch": int(best_epoch), "best_threshold": best_thr,
        "metrics": {
            "val_at_0.5":  {k: val_05[k]   for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
            "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
            "test_at_0.5": {k: test_05[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
            "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]}}}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

lines = ["="*72, f"EVALUATION REPORT — TCN v4 +SE on v1 data (RETRAIN)",
         "="*72,
         f"Architecture: TCN v3b (W=16, BLOCKS=3, DILATIONS=[1,2,4]) + SE attention",
         f"SE reduction: 8  (bottleneck = 8)",
         f"Receptive field: 29  (RF/W = 1.8x)",
         f"Features per step: 44 (16 raw + 28 one-hot function)  |  Params: {n_params:,}",
         f"Train/Val/Test windows: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}",
         f"Best epoch: {best_epoch}  |  Best val Macro-F1: {best_f1m:.4f}  |  Train time: {train_time:.1f}s",
         f"Tuned threshold: {best_thr:.2f}", ""]
for ev in [val_05, val_thr, test_05, test_thr]:
    lines.append("-"*72)
    lines.append(f"[{ev['label']}]  thr={ev['threshold']:.2f}  "
                 f"Macro-F1={ev['macro_f1']:.4f}  Binary-F1={ev['binary_f1']:.4f}  "
                 f"Acc={ev['accuracy']:.4f}  ROC-AUC={ev['roc_auc']:.4f}  PR-AUC={ev['pr_auc']:.4f}")
    lines.append(ev["report"])
    lines.append(f"Confusion matrix: {ev['cm']}"); lines.append("")
with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# plots
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(test_thr["cm"])
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Attack"]); ax.set_yticklabels(["Normal","Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center",
                    color="red" if cm[i,j]<cm.max()/2 else "white", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{TAG} Test CM @ thr={best_thr:.2f}  (Macro-F1={test_thr['macro_f1']:.4f})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for prob, yy, lbl in [(val_prob_full, val_lbl_full, "val"), (test_prob_full, test_lbl_full, "test")]:
        p, r, _ = precision_recall_curve(yy, prob); ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, prob):.3f}")
        fpr, tpr, _ = roc_curve(yy, prob); ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, prob):.3f}")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title("PR"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.plot([0,1],[0,1],"k--",alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h[0] for h in history]; loss = [h[1] for h in history]
    f1m = [h[2] for h in history]; auc = [h[3] for h in history]
    ax1.plot(ep, loss, "o-", color="C0"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.grid(alpha=0.3)
    ax2.plot(ep, f1m, "o-", color="C1", label="val Macro-F1")
    ax2.plot(ep, auc, "s-", color="C2", label="val AUC")
    ax2.axvline(best_epoch, color="grey", linestyle="--", alpha=0.5, label=f"best ep={best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HIST_PNG, dpi=120); plt.close(fig)
    log("   plots saved")
except Exception as e:
    log(f"   plots skipped: {e}")

log("="*60)
log(f"DONE {TAG}.")
log(f"   test@{best_thr:.2f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Bin-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   params: {n_params:,}  train_time: {train_time:.1f}s  best_epoch: {best_epoch}")
log("="*60)
