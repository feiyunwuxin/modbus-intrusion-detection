#!/usr/bin/env python3
"""
Unidirectional GRU binary classifier on the 17-feature v2 dataset, sliding window=8.

Goal: ablate the bidirectional contribution by training an otherwise-identical unidirectional
GRU on the same window=8 dataset. Direct comparison with BiGRU w=8 (same hyperparams, only
bidirectional flag flipped) to quantify how much the backward pass helps.

Architecture:
  GRU(input=44, hidden=64, num_layers=1)  → take last hidden state
  → Dropout(0.3) → Linear(64 → 32) → ReLU → Linear(32 → 1) → logits

vs BiGRU w=8:
  BiGRU forward + backward = 2x hidden, but only final h is used → out_dim = 2*hidden
  GRU unidirectional  = 1x hidden, out_dim = hidden
  So params ~ 1/3 of BiGRU.
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

# Output naming — distinguish from BiGRU artifacts
MODEL_OUT = os.path.join(BASE, "model_gru_uni_binary_v2_window8.pt")
EVAL_OUT  = os.path.join(BASE, "evaluation_gru_uni_binary_v2_window8.txt")
META_OUT  = os.path.join(BASE, "processed_meta_gru_uni_binary_window8.json")
THR_OUT   = os.path.join(BASE, "best_threshold_gru_uni_window8.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_gru_uni_binary_v2_window8.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_gru_uni_binary_v2_window8.csv")
CM_PNG    = os.path.join(BASE, "confusion_matrix_gru_uni_binary_v2_window8.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_gru_uni_binary_v2_window8.png")
HIST_PNG  = os.path.join(BASE, "training_history_gru_uni_binary_v2_window8.png")

# Hyperparameters — IDENTICAL to BiGRU w=8 (only BIDIRECTIONAL changes)
SEED        = 42
WINDOW      = 8                       # 2 Modbus cycles
EPOCHS      = 30
BATCH       = 512
LR          = 5e-4
WD          = 1e-5
PATIENCE    = 5
GRAD_CLIP   = 0.5
DROPOUT     = 0.3
CLIP_VAL    = 10.0
HIDDEN      = 64
NUM_LAYERS  = 1
BIDIRECTIONAL = False                # <-- only difference from BiGRU script

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

# Clip
X_train_cnn = np.clip(X_train_cnn, -CLIP_VAL, CLIP_VAL)
X_val_cnn   = np.clip(X_val_cnn,   -CLIP_VAL, CLIP_VAL)
X_test_cnn  = np.clip(X_test_cnn,  -CLIP_VAL, CLIP_VAL)
log(f"   final GRU input shape: {X_train_cnn.shape}  → GRU(T={WINDOW}, F={N_FEATURES}, unidirectional)")

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
# Step 5 — Unidirectional GRU
# ─────────────────────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, bidirectional=BIDIRECTIONAL):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(out_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, h = self.gru(x)
        # h: (num_layers * num_directions, B, hidden)
        if self.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h_last = h[-1]                              # unidirectional: just last layer's final h
        x = F.relu(self.fc1(h_last))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc2(x).squeeze(-1)


model = GRUClassifier(N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
log(f"   model params: {n_params:,}")
log(f"   architecture: 1-layer UNIDIRECTIONAL GRU, hidden={HIDDEN}, dropout={DROPOUT}")
log(f"                  → h[-1] (last timestep, last layer) → FC({HIDDEN}→32) → FC(32→1)")

# Reference: BiGRU has out_dim = 128, here out_dim = 64
# Approx param reduction: 3x fewer than BiGRU (46K)

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
    "hidden": HIDDEN,
    "num_layers": NUM_LAYERS,
    "bidirectional": BIDIRECTIONAL,
    "n_params": n_params,
    "best_epoch": best_epoch,
    "best_val_f1m": best_f1m,
    "n_cats_function": n_cats,
    "function_values": all_fn.tolist(),
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
    "model": f"GRU (unidirectional) v2 window={WINDOW}",
    "window": WINDOW,
    "n_features_per_step": int(N_FEATURES),
    "n_features_raw": 17,
    "n_cats_function": int(n_cats),
    "hidden_size": HIDDEN,
    "num_layers": NUM_LAYERS,
    "bidirectional": BIDIRECTIONAL,
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
lines.append("EVALUATION REPORT — Unidirectional GRU Binary (v2, sliding window=8)")
lines.append("=" * 72)
lines.append(f"Window: {WINDOW} rows  |  Features per step: {N_FEATURES} (16 raw + {n_cats} one-hot function)")
lines.append(f"Architecture: 1-layer UNIDIRECTIONAL GRU, hidden={HIDDEN}, dropout={DROPOUT}")
lines.append(f"vs BiGRU: same hyperparams, only bidirectional=False (out_dim {2*HIDDEN} → {HIDDEN})")
lines.append(f"Train windows: {len(X_train_w):,}  |  Val windows: {len(X_val_w):,}  |  Test windows: {len(X_test_w):,}")
lines.append(f"Model params: {n_params:,}  |  Best epoch: {best_epoch}  |  Best val Macro-F1: {best_f1m:.4f}")
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
log(f"   architecture  : 1-layer unidirectional GRU, hidden={HIDDEN}")
log(f"   train/val/test windows: {len(X_train_w):,} / {len(X_val_w):,} / {len(X_test_w):,}")
log(f"   params        : {n_params:,}  (BiGRU: 46,401)")
log(f"   best_epoch    : {best_epoch}  (val Macro-F1 = {best_f1m:.4f})")
log(f"   best_thr      : {best_thr:.2f}")
log(f"   val@0.5       : Macro-F1={val_05['macro_f1']:.4f}  Binary-F1={val_05['binary_f1']:.4f}")
log(f"   val@{best_thr:.2f}      : Macro-F1={val_thr['macro_f1']:.4f}  Binary-F1={val_thr['binary_f1']:.4f}")
log(f"   test@0.5      : Macro-F1={test_05['macro_f1']:.4f}  Binary-F1={test_05['binary_f1']:.4f}")
log(f"   test@{best_thr:.2f}     : Macro-F1={test_thr['macro_f1']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log("=" * 60)
