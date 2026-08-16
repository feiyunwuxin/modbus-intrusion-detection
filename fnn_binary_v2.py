#!/usr/bin/env python3
"""
FNN (Feedforward Neural Network / MLP) binary classifier on the 17-feature v2 dataset,
row-level (no windowing).

FNN is the most basic deep learning architecture: stacked fully-connected layers with
non-linear activations. No convolution, no recurrence.

Purpose: clean comparison with tree models on the same row-level tabular data.
  Single DT (just done):  0.7935 Macro-F1, 0.4s
  RF (just done):         0.8330 Macro-F1, 11s
  LGB v2:                 0.8337 Macro-F1, <5s
  FNN (this):             ?

Key questions:
  - Can a basic MLP beat trees on tabular data? (classic Kaggle debate)
  - How does FNN compare to window-based CNNs on the same data structure?
  - What's the right hidden layer size for 17 features?
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

MODEL_OUT = os.path.join(BASE, "model_fnn_binary_v2.pt")
EVAL_OUT  = os.path.join(BASE, "evaluation_fnn_binary_v2.txt")
META_OUT  = os.path.join(BASE, "processed_meta_fnn_binary.json")
THR_OUT   = os.path.join(BASE, "best_threshold_fnn.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_fnn_binary_v2.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_fnn_binary_v2.csv")
CM_PNG    = os.path.join(BASE, "confusion_matrix_fnn_binary_v2.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_fnn_binary_v2.png")
HIST_PNG  = os.path.join(BASE, "training_history_fnn_binary_v2.png")

# Hyperparameters
SEED         = 42
EPOCHS       = 30
BATCH        = 512
LR           = 5e-4
WD           = 1e-5
PATIENCE     = 6
GRAD_CLIP    = 0.5
DROPOUT      = 0.3
CLIP_VAL     = 10.0
FEATURE_NAMES = [
    "address", "function", "length", "setpoint", "gain", "reset rate",
    "deadband", "cycle time", "rate", "system mode", "control scheme",
    "pump", "solenoid", "pressure measurement", "crc rate", "command response",
    "time_diff",
]

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
log(f"   class dist: train 0={int((y_train==0).sum()):,} 1={int((y_train==1).sum()):,}")
log(f"               val   0={int((y_val  ==0).sum()):,} 1={int((y_val  ==1).sum()):,}")
log(f"               test  0={int((y_test ==0).sum()):,} 1={int((y_test ==1).sum()):,}")

# ─────────────────────────────────────────────
# Step 2 — DataLoaders
# ─────────────────────────────────────────────
log("STEP 2: DataLoaders ...")
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Step 3 — FNN architecture (with hidden_dim search)
# ─────────────────────────────────────────────
class FNNClassifier(nn.Module):
    """Feedforward NN with configurable hidden layers.

    Each block: Linear → BatchNorm1d → ReLU → Dropout
    """
    def __init__(self, in_features, hidden_dims, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        # Output head
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# Hidden layer configurations to try
HIDDEN_CHOICES = [
    [64, 32],
    [128, 64],
    [256, 128, 64],
    [128, 64, 32],
    [256, 128],
]

N_FEATURES = X_train.shape[1]
log(f"   n_features: {N_FEATURES}")

# ─────────────────────────────────────────────
# Step 4 — Search over hidden_dims
# ─────────────────────────────────────────────
log("STEP 4: search over hidden layer configurations ...")
results = []
best_val_f1 = -1
best_model = None
best_config = None
best_config_name = None

n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
log(f"   pos_weight = {pos_weight.item():.3f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

for hidden in HIDDEN_CHOICES:
    log(f"   trying hidden_dims={hidden} ...")
    model = FNNClassifier(N_FEATURES, hidden, dropout=DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    @torch.no_grad()
    def predict_probs(loader):
        model.eval()
        probs, labels = [], []
        for xb, yb in loader:
            logits = model(xb)
            probs.append(torch.sigmoid(logits).numpy())
            labels.append(yb.numpy())
        return np.concatenate(probs), np.concatenate(labels)

    best_epoch_f1 = -1
    best_state = None
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        val_prob, val_lbl = predict_probs(val_loader)
        val_pred_05 = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(val_lbl, val_pred_05, average="macro")
        scheduler.step(val_f1m)
        if val_f1m > best_epoch_f1:
            best_epoch_f1 = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    train_time = time.time() - train_start
    model.load_state_dict(best_state)
    val_prob, val_lbl = predict_probs(val_loader)
    val_pred_05 = (val_prob >= 0.5).astype(int)
    val_f1m = f1_score(val_lbl, val_pred_05, average="macro")
    val_auc = roc_auc_score(val_lbl, val_prob)
    log(f"     n_params={n_params:,}  train={train_time:.1f}s  val Macro-F1={val_f1m:.4f}  val AUC={val_auc:.4f}")

    results.append({
        "hidden_dims": list(hidden),
        "n_params": int(n_params),
        "val_macro_f1": float(val_f1m),
        "val_auc": float(val_auc),
        "train_time": float(train_time),
    })
    if val_f1m > best_val_f1:
        best_val_f1 = val_f1m
        best_model = model
        best_config = hidden
        best_n_params = n_params

log(f"   best hidden_dims = {best_config}  (val Macro-F1 = {best_val_f1:.4f}, n_params = {best_n_params:,})")

# ─────────────────────────────────────────────
# Step 5 — Threshold tuning on val
# ─────────────────────────────────────────────
log("STEP 5: threshold tuning on val ...")

@torch.no_grad()
def predict_probs_full(loader):
    best_model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        logits = best_model(xb)
        probs.append(torch.sigmoid(logits).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)

val_prob_full,  val_lbl_full  = predict_probs_full(val_loader)
test_prob_full, test_lbl_full = predict_probs_full(test_loader)

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
        "best_hidden_dims": best_config,
    }, f, indent=2)

# ─────────────────────────────────────────────
# Step 6 — Evaluate on test
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
# Step 7 — Save model
# ─────────────────────────────────────────────
log("STEP 7: save model ...")
torch.save({
    "state_dict": best_model.state_dict(),
    "n_features": N_FEATURES,
    "hidden_dims": best_config,
    "dropout": DROPOUT,
    "n_params": best_n_params,
    "best_threshold": best_thr,
    "feature_names": FEATURE_NAMES,
}, MODEL_OUT)

# ─────────────────────────────────────────────
# Step 8 — Save predictions and metadata
# ─────────────────────────────────────────────
log("STEP 8: save predictions and metadata ...")
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
    "model": "FNN (Feedforward NN / MLP)",
    "n_features": int(N_FEATURES),
    "feature_names": FEATURE_NAMES,
    "splits": {
        "train_size": int(len(X_train)),
        "val_size":   int(len(X_val)),
        "test_size":  int(len(X_test)),
    },
    "class_dist": {
        "train": {"0": int((y_train==0).sum()), "1": int((y_train==1).sum())},
        "val":   {"0": int((y_val  ==0).sum()), "1": int((y_val  ==1).sum())},
        "test":  {"0": int((y_test ==0).sum()), "1": int((y_test ==1).sum())},
    },
    "n_params": int(best_n_params),
    "best_hidden_dims": list(best_config),
    "dropout": DROPOUT,
    "best_threshold": best_thr,
    "config_search": results,
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

# ─────────────────────────────────────────────
# Step 9 — Save text report
# ─────────────────────────────────────────────
log("STEP 9: save text report ...")
lines = []
lines.append("=" * 72)
lines.append("EVALUATION REPORT — FNN (Feedforward NN / MLP) Binary v2 IDS")
lines.append("=" * 72)
lines.append(f"Features: {N_FEATURES} (16 raw + time_diff)")
lines.append(f"Splits: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
lines.append(f"Hyperparams: dropout={DROPOUT}, lr={LR}, batch={BATCH}, pos_weight={pos_weight.item():.3f}")
lines.append(f"Best hidden_dims: {best_config}")
lines.append(f"n_params: {best_n_params:,}")
lines.append(f"Best val Macro-F1: {best_val_f1:.4f}")
lines.append(f"Tuned threshold: {best_thr:.2f}")
lines.append("")
lines.append("Hidden layer search:")
lines.append(f"  {'hidden_dims':<25}  {'n_params':>10}  {'val_F1m':>8}  {'val_AUC':>8}  {'train_time':>10}")
for r in results:
    lines.append(f"  {str(r['hidden_dims']):<25}  {r['n_params']:>10,}  {r['val_macro_f1']:8.4f}  {r['val_auc']:8.4f}  {r['train_time']:10.1f}")
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

# ─────────────────────────────────────────────
# Step 10 — Plots
# ─────────────────────────────────────────────
log("STEP 10: plots ...")
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
    p, r, _ = precision_recall_curve(y_val, val_prob_full)
    ax1.plot(r, p, label=f"val AP={average_precision_score(y_val, val_prob_full):.3f}")
    p, r, _ = precision_recall_curve(y_test, test_prob_full)
    ax1.plot(r, p, label=f"test AP={average_precision_score(y_test, test_prob_full):.3f}")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title("PR"); ax1.legend(); ax1.grid(alpha=0.3)

    fpr, tpr, _ = roc_curve(y_val, val_prob_full)
    ax2.plot(fpr, tpr, label=f"val AUC={roc_auc_score(y_val, val_prob_full):.3f}")
    fpr, tpr, _ = roc_curve(y_test, test_prob_full)
    ax2.plot(fpr, tpr, label=f"test AUC={roc_auc_score(y_test, test_prob_full):.3f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)

    log(f"   plots saved: {CM_PNG}, {PRROC_PNG}")
except Exception as e:
    log(f"   plots skipped: {e}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
log("=" * 60)
log("DONE.")
log(f"   features      : {N_FEATURES}")
log(f"   splits        : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
log(f"   best config   : hidden_dims={best_config}  n_params={best_n_params:,}")
log(f"   best_thr      : {best_thr:.2f}")
log(f"   val@0.5       : Macro-F1={val_05['macro_f1']:.4f}  Binary-F1={val_05['binary_f1']:.4f}")
log(f"   val@{best_thr:.2f}      : Macro-F1={val_thr['macro_f1']:.4f}  Binary-F1={val_thr['binary_f1']:.4f}")
log(f"   test@0.5      : Macro-F1={test_05['macro_f1']:.4f}  Binary-F1={test_05['binary_f1']:.4f}")
log(f"   test@{best_thr:.2f}     : Macro-F1={test_thr['macro_f1']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log("=" * 60)
