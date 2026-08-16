#!/usr/bin/env python3
"""
Random Forest (sklearn) binary classifier on the 17-feature v2 dataset, row-level.

Purpose: complete the tree-ensemble family comparison:
  Single DT (just done):  0.7935 Macro-F1, 0.4s, 92 leaves, 0.6795 PR-AUC
  RF (this):               ?   Macro-F1, ?s, ?trees,    ?   PR-AUC
  LGB v2 (already done):   0.8337 Macro-F1, <5s, ~few K, 0.8166 PR-AUC

Random Forest:
  - Bagging ensemble (each tree on a bootstrap sample)
  - Feature randomness (only consider random sqrt(F) features at each split)
  - Reduces variance vs single DT (averaging many trees)
  - Sequential trees, no boosting, no gradient
  - Expected to outperform single DT, may be competitive with LGB on Macro-F1
  - Should be MUCH better than single DT on PR-AUC (smoothed leaf probabilities)
"""

import os, time, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

MODEL_OUT = os.path.join(BASE, "model_random_forest_binary_v2.joblib")
EVAL_OUT  = os.path.join(BASE, "evaluation_random_forest_binary_v2.txt")
META_OUT  = os.path.join(BASE, "processed_meta_random_forest_binary.json")
THR_OUT   = os.path.join(BASE, "best_threshold_random_forest.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_random_forest_binary_v2.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_random_forest_binary_v2.csv")
CM_PNG    = os.path.join(BASE, "confusion_matrix_random_forest_binary_v2.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_random_forest_binary_v2.png")

# Hyperparameters — search a small grid
SEED         = 42
N_ESTIMATORS_CHOICES = [100, 200, 500]
MAX_DEPTH_CHOICES    = [12, 16, 20]   # not None (would overfit like DT)
MIN_SAMPLES_LEAF     = 5
MAX_FEATURES         = "sqrt"          # default for classification (good for variance reduction)
CLASS_WEIGHT         = "balanced"
N_JOBS               = -1              # use all cores
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
# Step 2 — Grid search over n_estimators × max_depth
# ─────────────────────────────────────────────
log("STEP 2: grid search n_estimators × max_depth ...")

results = []
best_val_f1 = -1
best_model = None
best_n = None
best_d = None

for n_est in N_ESTIMATORS_CHOICES:
    for depth in MAX_DEPTH_CHOICES:
        log(f"   trying n_estimators={n_est}, max_depth={depth} ...")
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            class_weight=CLASS_WEIGHT,
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        t_train = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - t_train

        val_prob = rf.predict_proba(X_val)[:, 1]
        val_pred_05 = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(y_val, val_pred_05, average="macro")
        val_auc = roc_auc_score(y_val, val_prob)
        # total leaves across all trees
        total_leaves = sum(est.get_n_leaves() for est in rf.estimators_)
        log(f"     train={train_time:.2f}s  val Macro-F1={val_f1m:.4f}  val AUC={val_auc:.4f}  "
            f"total_leaves={total_leaves:,}")

        results.append({
            "n_estimators": n_est,
            "max_depth": depth,
            "val_macro_f1": val_f1m,
            "val_auc": val_auc,
            "train_time": train_time,
            "total_leaves": total_leaves,
        })
        if val_f1m > best_val_f1:
            best_val_f1 = val_f1m
            best_model = rf
            best_n = n_est
            best_d = depth

log(f"   best (n_estimators, max_depth) = ({best_n}, {best_d})  (val Macro-F1 = {best_val_f1:.4f})")
total_leaves_best = sum(est.get_n_leaves() for est in best_model.estimators_)
log(f"   total leaves across {best_n} trees: {total_leaves_best:,}")
log(f"   avg leaves per tree: {total_leaves_best/best_n:.0f}")

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
        "best_n_estimators": best_n,
        "best_max_depth": best_d,
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
    "model": "RandomForestClassifier (sklearn)",
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
    "n_params_total_leaves": int(total_leaves_best),
    "n_params_avg_leaves_per_tree": int(total_leaves_best / best_n),
    "n_estimators": int(best_n),
    "best_max_depth": int(best_d),
    "min_samples_leaf": int(MIN_SAMPLES_LEAF),
    "max_features": MAX_FEATURES,
    "class_weight": CLASS_WEIGHT,
    "n_jobs": N_JOBS,
    "best_threshold": best_thr,
    "grid_search": [
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

# Top features from the ensemble (averaged importance)
log("   top features (mean impurity decrease across trees):")
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
lines.append("EVALUATION REPORT — Random Forest (sklearn, v2 binary IDS)")
lines.append("=" * 72)
lines.append(f"Features: {X_train.shape[1]} (16 raw + time_diff)")
lines.append(f"Splits: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
lines.append(f"Hyperparams: n_estimators={best_n}  max_depth={best_d}  min_samples_leaf={MIN_SAMPLES_LEAF}  "
             f"max_features={MAX_FEATURES}  class_weight={CLASS_WEIGHT}")
lines.append(f"Forest size: {best_n} trees  ×  {total_leaves_best/best_n:.0f} avg leaves = {total_leaves_best:,} total leaves")
lines.append(f"Best val Macro-F1: {best_val_f1:.4f}  (best n_est={best_n}, max_depth={best_d})")
lines.append(f"Tuned threshold: {best_thr:.2f}")
lines.append("")
lines.append("Grid search:")
lines.append(f"  {'n_est':>5}  {'max_d':>5}  {'val_F1m':>8}  {'val_AUC':>8}  {'total_leaves':>12}  {'train_time':>10}")
for r in results:
    lines.append(f"  {r['n_estimators']:5d}  {r['max_depth']:5d}  {r['val_macro_f1']:8.4f}  {r['val_auc']:8.4f}  "
                 f"{r['total_leaves']:12,}  {r['train_time']:10.2f}")
lines.append("")
lines.append("Top 10 features (mean impurity-based importance across all trees):")
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
log(f"   features       : {X_train.shape[1]}")
log(f"   splits         : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
log(f"   best n_est     : {best_n}")
log(f"   best max_depth : {best_d}")
log(f"   total leaves   : {total_leaves_best:,}")
log(f"   best_thr       : {best_thr:.2f}")
log(f"   val@0.5        : Macro-F1={val_05['macro_f1']:.4f}  Binary-F1={val_05['binary_f1']:.4f}")
log(f"   val@{best_thr:.2f}      : Macro-F1={val_thr['macro_f1']:.4f}  Binary-F1={val_thr['binary_f1']:.4f}")
log(f"   test@0.5       : Macro-F1={test_05['macro_f1']:.4f}  Binary-F1={test_05['binary_f1']:.4f}")
log(f"   test@{best_thr:.2f}     : Macro-F1={test_thr['macro_f1']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log("=" * 60)
