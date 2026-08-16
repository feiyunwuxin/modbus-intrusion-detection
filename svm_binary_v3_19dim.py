#!/usr/bin/env python3
"""
SVM binary classifier on the 19-feature v2_scada dataset, row-level.

Strategy for 192K samples:
- LinearSVC: fast linear SVM, scaled by StandardScaler
- CalibratedClassifierCV wraps LinearSVC to get probabilities
- RBF kernel SVM: too slow on 192K (skip or subsample)
"""

import os, time, json
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)

BASE      = r"C:\work\Claude\Issue"

X_TR_NPY  = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA_NPY  = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE_NPY  = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR_NPY  = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA_NPY  = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE_NPY  = os.path.join(BASE, "y_test_binary_v2_scada.npy")

MODEL_OUT = os.path.join(BASE, "model_svm_binary_v3_19dim.joblib")
EVAL_OUT  = os.path.join(BASE, "evaluation_svm_binary_v3_19dim.txt")
META_OUT  = os.path.join(BASE, "processed_meta_svm_binary_v3_19dim.json")
THR_OUT   = os.path.join(BASE, "best_threshold_svm_binary_v3_19dim.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_svm_binary_v3_19dim.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_svm_binary_v3_19dim.csv")
CM_PNG    = os.path.join(BASE, "confusion_matrix_svm_binary_v3_19dim.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_svm_binary_v3_19dim.png")

SEED = 42
np.random.seed(SEED)

FEATURE_NAMES = [
    "address", "function", "length",
    "setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
    "system mode", "control scheme", "pump", "solenoid",
    "pressure measurement", "crc rate",
    "time_diff", "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
]

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log("STEP 1: load 19-dim npy ...")
X_train = np.load(X_TR_NPY).astype(np.float32)
X_val   = np.load(X_VA_NPY).astype(np.float32)
X_test  = np.load(X_TE_NPY).astype(np.float32)
y_train = np.load(Y_TR_NPY).astype(np.int64)
y_val   = np.load(Y_VA_NPY).astype(np.int64)
y_test  = np.load(Y_TE_NPY).astype(np.int64)
log(f"   shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

log("STEP 2: StandardScaler fit on train ...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
log(f"   scaled")

log("STEP 3: LinearSVC + CalibratedClassifierCV ...")
# LinearSVC is much faster than SVC(kernel='linear') for large data
base_svc = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=SEED,
    dual="auto",
)
# Wrap to get probabilities (slower but needed for PR-AUC / threshold tuning)
clf = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3, n_jobs=-1)
clf.fit(X_train, y_train)

val_prob  = clf.predict_proba(X_val)[:, 1]
test_prob = clf.predict_proba(X_test)[:, 1]
log(f"   training done")

log("STEP 4: threshold tuning ...")
thresholds = np.arange(0.10, 0.91, 0.01)
sweep = []
for t in thresholds:
    pred = (val_prob >= t).astype(int)
    f1m  = f1_score(y_val, pred, average="macro")
    sweep.append((float(t), float(f1m)))
sweep_df = pd.DataFrame(sweep, columns=["threshold","val_macro_f1"])
best_idx  = int(sweep_df["val_macro_f1"].idxmax())
best_thr  = float(sweep_df.loc[best_idx, "threshold"])
best_f1m  = float(sweep_df.loc[best_idx, "val_macro_f1"])
f1m_at_05 = float(sweep_df.loc[(sweep_df["threshold"]-0.5).abs().idxmin(), "val_macro_f1"])
log(f"   best threshold = {best_thr:.2f}  →  val Macro-F1 = {best_f1m:.4f}  (vs 0.50 = {f1m_at_05:.4f})")

with open(THR_OUT, "w") as f:
    json.dump({"best_threshold": best_thr, "val_macro_f1_at_best": best_f1m,
               "val_macro_f1_at_0.5": f1m_at_05, "metric": "macro_f1"}, f, indent=2)

log("STEP 5: evaluation ...")
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

val_05   = evaluate(y_val,  val_prob,  0.5,         "val@0.5")
val_thr  = evaluate(y_val,  val_prob,  best_thr,    f"val@{best_thr:.2f}")
test_05  = evaluate(y_test, test_prob, 0.5,         "test@0.5")
test_thr = evaluate(y_test, test_prob, best_thr,    f"test@{best_thr:.2f}")

log("STEP 6: save artifacts ...")
import joblib
joblib.dump({"clf": clf, "scaler": scaler}, MODEL_OUT)

pd.DataFrame({"y_true": y_val, "prob_attack": val_prob,
              "pred_05": (val_prob >= 0.5).astype(int),
              "pred_tuned": (val_prob >= best_thr).astype(int)}).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": y_test, "prob_attack": test_prob,
              "pred_05": (test_prob >= 0.5).astype(int),
              "pred_tuned": (test_prob >= best_thr).astype(int)}).to_csv(PRED_TEST, index=False)

meta = {
    "model": "LinearSVC + Calibration v3 19-dim", "n_features": 19, "feature_names": FEATURE_NAMES,
    "kernel": "linear", "C": 1.0, "calibration": "sigmoid(cv=3)",
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
    "=" * 72, "EVALUATION REPORT — LinearSVM v3 19-dim SCADA",
    "=" * 72,
    f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}",
    f"Features: 19  |  Kernel: linear  |  C: 1.0",
    f"Tuned threshold: {best_thr:.2f}", "",
]
for ev in [val_05, val_thr, test_05, test_thr]:
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for prob, yy, lbl in [(val_prob, y_val, "val"), (test_prob, y_test, "test")]:
        p, r, _ = precision_recall_curve(yy, prob); ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, prob):.3f}")
        fpr, tpr, _ = roc_curve(yy, prob); ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, prob):.3f}")
    ax1.legend(); ax1.grid(alpha=0.3); ax2.legend(); ax2.grid(alpha=0.3); ax2.plot([0,1],[0,1],"k--",alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)
    log("   plots saved")
except Exception as e:
    log(f"   plots skipped: {e}")

log("=" * 60)
log(f"DONE SVM v3 19-dim.")
log(f"   test@{best_thr:.2f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Bin-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   train time: {time.time()-t0:.1f}s")
log("=" * 60)