#!/usr/bin/env python3
"""
Single Decision Tree (sklearn) binary classifier on the 17-feature v2 dataset,
row-level (no windowing — DT doesn't naturally model sequences).

Purpose: ablation against LightGBM to quantify the value of gradient boosting
on this tabular IDS task. Same training/val/test split, same 17 features.

Comparison:
  LGB v2  (ensemble of 284 trees, learning rate 0.05):  Test Macro-F1 = 0.8337
  Single DT (one tree, max_depth tuned):              Test Macro-F1 = ?

Single DT is expected to be:
  - Easier to interpret (one tree vs hundreds)
  - Less accurate (high variance, no ensembling)
  - Fast to train (no boosting iterations)
  - May overfit without proper depth control
"""

import os, time, json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
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

# Output naming — distinguish from LGB
MODEL_OUT = os.path.join(BASE, "model_decision_tree_binary_v2.joblib")
EVAL_OUT  = os.path.join(BASE, "evaluation_decision_tree_binary_v2.txt")
META_OUT  = os.path.join(BASE, "processed_meta_decision_tree_binary.json")
THR_OUT   = os.path.join(BASE, "best_threshold_decision_tree.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_decision_tree_binary_v2.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_decision_tree_binary_v2.csv")
CM_PNG    = os.path.join(BASE, "confusion_matrix_decision_tree_binary_v2.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_decision_tree_binary_v2.png")

# Hyperparameters
SEED      = 42
# We try a small grid and pick best by val Macro-F1
MAX_DEPTH_CHOICES    = [8, 12, 16, 20, None]
MIN_SAMPLES_LEAF     = 10   # prevent overfitting (default=1 would likely overfit)
CRITERION             = "gini"
CLASS_WEIGHT         = "balanced"   # match LGB setup
FEATURE_NAMES         = [
    "address", "function", "length", "setpoint", "gain", "reset rate",
    "deadband", "cycle time", "rate", "system mode", "control scheme",
    "pump", "solenoid", "pressure measurement", "crc rate", "command response",
    "time_diff",
]

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ─────────────────────────────────────────────
# Step 1 — Load npy
# ─────────────────────────────────────────────
log("STEP 1: load npy ...")
X_train = np.load(X_TR)
X_val   = np.load(X_VA)
X_test  = np.load(X_TE)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)
log(f"   raw shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
log(f"   class dist: train 0={int((y_train==0).sum()):,} 1={int((y_train==1).sum()):,}")
log(f"               val   0={int((y_val  ==0).sum()):,} 1={int((y_val  ==1).sum()):,}")
log(f"               test  0={int((y_test ==0).sum()):,} 1={int((y_test ==1).sum()):,}")

# ─────────────────────────────────────────────
# Step 2 — Try multiple max_depth, pick best by val Macro-F1
# ─────────────────────────────────────────────
log("STEP 2: hyperparameter search over max_depth ...")

results = []
best_val_f1 = -1
best_model = None
best_depth = None

for depth in MAX_DEPTH_CHOICES:
    log(f"   trying max_depth={depth} ...")
    dt = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        criterion=CRITERION,
        class_weight=CLASS_WEIGHT,
        random_state=SEED,
    )
    t_train = time.time()
    dt.fit(X_train, y_train)
    train_time = time.time() - t_train

    val_prob = dt.predict_proba(X_val)[:, 1]
    val_pred_05 = (val_prob >= 0.5).astype(int)
    val_f1m = f1_score(y_val, val_pred_05, average="macro")
    val_auc = roc_auc_score(y_val, val_prob)
    n_nodes = dt.tree_.node_count
    n_leaves = dt.get_n_leaves()
    log(f"     train={train_time:.2f}s  val Macro-F1={val_f1m:.4f}  val AUC={val_auc:.4f}  "
        f"n_nodes={n_nodes:,}  n_leaves={n_leaves:,}")

    results.append({
        "max_depth": depth,
        "val_macro_f1": val_f1m,
        "val_auc": val_auc,
        "train_time": train_time,
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
    })
    if val_f1m > best_val_f1:
        best_val_f1 = val_f1m
        best_model = dt
        best_depth = depth

log(f"   best max_depth = {best_depth}  (val Macro-F1 = {best_val_f1:.4f})")
log(f"   model params (n_leaves): {best_model.get_n_leaves():,}")

# ─────────────────────────────────────────────
# Step 3 — Threshold tuning on val
# ─────────────────────────────────────────────
log("STEP 3: threshold tuning on val ...")
val_prob_full  = best_model.predict_proba(X_val)[:, 1]
test_prob_full = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)
sweep = []
for t in thresholds:
    pred = (val_prob_full >= t).astype(int)
    f1m = f1_score(y_val, pred, average="macro")
    f1b = f1_score(y_val, pred, average="binary")
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
        "best_max_depth": best_depth,
    }, f, indent=2)

# ─────────────────────────────────────────────
# Step 4 — Evaluate on test
# ─────────────────────────────────────────────
log("STEP 4: evaluation ...")
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

val_05   = evaluate(y_val,  val_prob_full,  0.5,      "val@0.5")
val_thr  = evaluate(y_val,  val_prob_full,  best_thr, f"val@{best_thr:.2f}")
test_05  = evaluate(y_test, test_prob_full, 0.5,      "test@0.5")
test_thr = evaluate(y_test, test_prob_full, best_thr, f"test@{best_thr:.2f}")

# ─────────────────────────────────────────────
# Step 5 — Save model
# ─────────────────────────────────────────────
log("STEP 5: save model ...")
import joblib
joblib.dump(best_model, MODEL_OUT)
log(f"   saved → {MODEL_OUT}")

# ─────────────────────────────────────────────
# Step 6 — Save predictions
# ─────────────────────────────────────────────
log("STEP 6: save predictions ...")
pd.DataFrame({
    "y_true": y_val, "prob_attack": val_prob_full,
    "pred_05": (val_prob_full >= 0.5).astype(int),
    "pred_tuned": (val_prob_full >= best_thr).astype(int),
}).to_csv(PRED_VAL, index=False)

pd.DataFrame({
    "y_true": y_test, "prob_attack": test_prob_full,
    "pred_05": (test_prob_full >= 0.5).astype(int),
    "pred_tuned": (test_prob_full >= best_thr).astype(int),
}).to_csv(PRED_TEST, index=False)

# ─────────────────────────────────────────────
# Step 7 — Save metadata
# ─────────────────────────────────────────────
log("STEP 7: save metadata ...")
meta = {
    "model": "DecisionTreeClassifier (sklearn)",
    "n_features": int(X_train.shape[1]),
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
    "n_params": int(best_model.get_n_leaves()),
    "best_max_depth": best_depth,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "criterion": CRITERION,
    "class_weight": CLASS_WEIGHT,
    "n_nodes": int(best_model.tree_.node_count),
    "n_leaves": int(best_model.get_n_leaves()),
    "best_threshold": best_thr,
    "depth_search": [
        {k: (int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (np.floating,)) else v))
         for k, v in r.items()}
        for r in results
    ],
    "metrics": {
        "val_at_0.5":  {k: val_05[k]   for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_0.5": {k: test_05[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

# Top features from the tree
log("   top features (by impurity decrease):")
feat_imp = best_model.feature_importances_
sorted_idx = np.argsort(feat_imp)[::-1]
for i in sorted_idx[:10]:
    if feat_imp[i] > 0:
        log(f"     {FEATURE_NAMES[i]:>22}: {feat_imp[i]:.4f}")

# ─────────────────────────────────────────────
# Step 8 — Save text report
# ─────────────────────────────────────────────
log("STEP 8: save text report ...")
lines = []
lines.append("=" * 72)
lines.append("EVALUATION REPORT — Single Decision Tree (sklearn, v2 binary IDS)")
lines.append("=" * 72)
lines.append(f"Features: {X_train.shape[1]} (16 raw + time_diff)")
lines.append(f"Splits: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
lines.append(f"Hyperparams: max_depth={best_depth}  min_samples_leaf={MIN_SAMPLES_LEAF}  "
             f"criterion={CRITERION}  class_weight={CLASS_WEIGHT}")
lines.append(f"Tree size: n_nodes={best_model.tree_.node_count:,}  n_leaves={best_model.get_n_leaves():,}")
lines.append(f"Best val Macro-F1: {best_val_f1:.4f}  (best max_depth = {best_depth})")
lines.append(f"Tuned threshold: {best_thr:.2f}")
lines.append("")
lines.append("Depth search:")
lines.append(f"  {'max_depth':>10}  {'val_F1m':>8}  {'val_AUC':>8}  {'n_leaves':>9}  {'train_time':>10}")
for r in results:
    lines.append(f"  {str(r['max_depth']):>10}  {r['val_macro_f1']:8.4f}  {r['val_auc']:8.4f}  "
                 f"{r['n_leaves']:9,}  {r['train_time']:10.2f}")
lines.append("")
lines.append("Top 10 features (impurity-based importance):")
for i in sorted_idx[:10]:
    if feat_imp[i] > 0:
        lines.append(f"  {FEATURE_NAMES[i]:>22}: {feat_imp[i]:.4f}")
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

# Also dump a small text view of the tree structure (first 200 lines)
try:
    tree_text = export_text(best_model, feature_names=FEATURE_NAMES, max_depth=5)
    with open(EVAL_OUT, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 72 + "\n")
        f.write("TREE STRUCTURE (first 5 levels, full tree is much larger)\n")
        f.write("=" * 72 + "\n")
        f.write(tree_text)
except Exception as e:
    log(f"   tree structure dump skipped: {e}")

# ─────────────────────────────────────────────
# Step 9 — Plots
# ─────────────────────────────────────────────
log("STEP 9: plots ...")
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
log(f"   features      : {X_train.shape[1]}")
log(f"   splits        : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
log(f"   best max_depth: {best_depth}")
log(f"   n_leaves      : {best_model.get_n_leaves():,}")
log(f"   best_thr      : {best_thr:.2f}")
log(f"   val@0.5       : Macro-F1={val_05['macro_f1']:.4f}  Binary-F1={val_05['binary_f1']:.4f}")
log(f"   val@{best_thr:.2f}      : Macro-F1={val_thr['macro_f1']:.4f}  Binary-F1={val_thr['binary_f1']:.4f}")
log(f"   test@0.5      : Macro-F1={test_05['macro_f1']:.4f}  Binary-F1={test_05['binary_f1']:.4f}")
log(f"   test@{best_thr:.2f}     : Macro-F1={test_thr['macro_f1']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log("=" * 60)
