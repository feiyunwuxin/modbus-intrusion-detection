#!/usr/bin/env python3
"""
Binary Intrusion Detection — minimal pipeline (v2)
Source: IanArffDataset.csv (raw, 20 cols)
Final features: 17 = 16 raw (minus time) + 1 time_diff
Split: temporal 70/10/20 (train/val/test)
Model: LightGBM, val-based threshold tuning
"""

import os, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)

BASE       = r"C:\work\Claude\Issue"
CSV_IN     = os.path.join(BASE, "IanArffDataset.csv")

# Outputs (v2 suffix, no overwrite)
MODEL_OUT  = os.path.join(BASE, "model_binary_v2.joblib")
SCALER_OUT = os.path.join(BASE, "scaler_binary_v2.joblib")
EVAL_OUT   = os.path.join(BASE, "evaluation_binary_v2.txt")
IMP_OUT    = os.path.join(BASE, "feature_importance_binary_v2.csv")
META_OUT   = os.path.join(BASE, "processed_meta_binary.json")
THR_OUT    = os.path.join(BASE, "best_threshold.json")
PRED_VAL   = os.path.join(BASE, "predictions_val_binary_v2.csv")
PRED_TEST  = os.path.join(BASE, "predictions_test_binary_v2.csv")
X_TR_NPY   = os.path.join(BASE, "X_train_binary.npy")
X_VA_NPY   = os.path.join(BASE, "X_val_binary.npy")
X_TE_NPY   = os.path.join(BASE, "X_test_binary.npy")
Y_TR_NPY   = os.path.join(BASE, "y_train_binary.npy")
Y_VA_NPY   = os.path.join(BASE, "y_val_binary.npy")
Y_TE_NPY   = os.path.join(BASE, "y_test_binary.npy")
CM_PNG     = os.path.join(BASE, "confusion_matrix_binary_v2.png")
PRROC_PNG  = os.path.join(BASE, "pr_roc_binary_v2.png")

SEED = 42
np.random.seed(SEED)

CATEGORICAL = ["function"]   # will be re-set below by index after we know column order
CAT_COL_NAME = "function"
LABEL_COL   = "binary result"
DROP_LABEL_COLS = ["binary result", "categorized result", "specific result"]

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ─────────────────────────────────────────────
# Step 1 — Load + type convert + sort by time
# ─────────────────────────────────────────────
log("STEP 1: load + sort ...")
df = pd.read_csv(CSV_IN)
n_raw = len(df)
log(f"   loaded {n_raw:,} rows × {df.shape[1]} cols")

df = df.replace("?", np.nan)

float_cols = ["address","function","length","setpoint","gain","reset rate",
              "deadband","cycle time","rate","system mode","control scheme",
              "pump","solenoid","pressure measurement","crc rate","time"]
for c in float_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["command response"] = pd.to_numeric(df["command response"], errors="coerce").astype("Int64")
for c in DROP_LABEL_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

df = df.sort_values("time").reset_index(drop=True)
log(f"   sorted by time; range = {pd.to_datetime(df['time'].min(), unit='s')} → {pd.to_datetime(df['time'].max(), unit='s')}")

# ─────────────────────────────────────────────
# Step 2 — Drop time, add time_diff
# ─────────────────────────────────────────────
log("STEP 2: drop time, add time_diff ...")
df["time_diff"] = df["time"].diff().fillna(0)
df = df.drop(columns=["time"])

# ─────────────────────────────────────────────
# Step 3 — Structural NaN → 0 + physical sanity
# ─────────────────────────────────────────────
log("STEP 3: fill structural NaN ...")
fillna_cols = ["setpoint","gain","reset rate","deadband","cycle time",
               "rate","system mode","control scheme","pump","solenoid",
               "pressure measurement"]
df[fillna_cols] = df[fillna_cols].fillna(0)
df["pressure measurement"] = df["pressure measurement"].clip(lower=0, upper=100)
df.loc[df["pressure measurement"].abs() < 1e-30, "pressure measurement"] = 0
df.loc[df["crc rate"].abs() < 1e-30, "crc rate"] = 0
log(f"   remaining NaNs: {df.isna().sum().sum()}")

# ─────────────────────────────────────────────
# Step 4 — Separate y, drop label columns
# ─────────────────────────────────────────────
log("STEP 4: extract y, drop label cols ...")
y = df[LABEL_COL].astype(int).values
df = df.drop(columns=DROP_LABEL_COLS)
log(f"   feature cols ({df.shape[1]}): {df.columns.tolist()}")

# Convert function to int for LGB categorical handling
df["function"] = df["function"].astype("Int64").astype("int32")
# Cast as pandas category so LGB treats it as categorical
df["function"] = df["function"].astype("category")

# Sanity: address should be ~constant
log(f"   address unique values: {df['address'].nunique()} (top 3: {df['address'].value_counts().head(3).to_dict()})")
log(f"   function unique values: {df['function'].nunique()} (top 5: {df['function'].value_counts().head(5).to_dict()})")

# ─────────────────────────────────────────────
# Step 5 — Temporal 70/10/20 split
# ─────────────────────────────────────────────
log("STEP 5: temporal 70/10/20 split ...")
n = len(df)
n_train = int(n * 0.70)            # 192,239
n_val   = int(n * 0.10)            #  27,462
n_test  = n - n_train - n_val      #  54,927

X_train = df.iloc[:n_train].copy()
X_val   = df.iloc[n_train:n_train + n_val].copy()
X_test  = df.iloc[n_train + n_val:].copy()
y_train = y[:n_train]
y_val   = y[n_train:n_train + n_val]
y_test  = y[n_train + n_val:]

log(f"   train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}  sum={len(X_train)+len(X_val)+len(X_test):,}")
for name, yy in [("train",y_train),("val",y_val),("test",y_test)]:
    pos = int((yy==1).sum())
    log(f"   {name} dist: 0={len(yy)-pos:,}  1={pos:,}  attack%={pos/len(yy)*100:.2f}")

# ─────────────────────────────────────────────
# Step 6 — RobustScaler (fit on train only)
# ─────────────────────────────────────────────
log("STEP 6: RobustScaler fit on train ...")
numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL]
scaler = RobustScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols]   = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
joblib.dump(scaler, SCALER_OUT)

# Save arrays
np.save(X_TR_NPY, X_train.values.astype(np.float32))
np.save(X_VA_NPY, X_val.values.astype(np.float32))
np.save(X_TE_NPY, X_test.values.astype(np.float32))
np.save(Y_TR_NPY, y_train.astype(np.int8))
np.save(Y_VA_NPY, y_val.astype(np.int8))
np.save(Y_TE_NPY, y_test.astype(np.int8))
log(f"   saved scaler + 6 npy arrays")

# ─────────────────────────────────────────────
# Step 7 — LightGBM training
# ─────────────────────────────────────────────
log("STEP 7: LightGBM training ...")
clf = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)
# LGB auto-detects pandas category dtype, so we just need to pass via fit
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
)
log(f"   best_iteration = {clf.best_iteration_}")
joblib.dump(clf, MODEL_OUT)

val_prob  = clf.predict_proba(X_val)[:, 1]
test_prob = clf.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# Step 8 — Threshold tuning on val (no test leak)
# ─────────────────────────────────────────────
log("STEP 8: threshold tuning on val ...")
thresholds = np.arange(0.10, 0.91, 0.01)
sweep = []
for t in thresholds:
    pred = (val_prob >= t).astype(int)
    f1m  = f1_score(y_val, pred, average="macro")
    f1b  = f1_score(y_val, pred, average="binary")
    sweep.append((float(t), float(f1m), float(f1b)))

sweep_df = pd.DataFrame(sweep, columns=["threshold","val_macro_f1","val_binary_f1"])
best_idx   = int(sweep_df["val_macro_f1"].idxmax())
best_thr   = float(sweep_df.loc[best_idx, "threshold"])
best_f1m   = float(sweep_df.loc[best_idx, "val_macro_f1"])
f1m_at_05  = float(sweep_df.loc[(sweep_df["threshold"]-0.5).abs().idxmin(), "val_macro_f1"])
log(f"   best threshold = {best_thr:.2f}  →  val Macro-F1 = {best_f1m:.4f}  (vs 0.50 = {f1m_at_05:.4f})")

with open(THR_OUT, "w") as f:
    json.dump({
        "best_threshold": best_thr,
        "val_macro_f1_at_best": best_f1m,
        "val_binary_f1_at_best": float(sweep_df.loc[best_idx, "val_binary_f1"]),
        "val_macro_f1_at_0.5":  f1m_at_05,
        "metric": "macro_f1",
        "sweep_grid": "0.10..0.90 step 0.01",
    }, f, indent=2)

# ─────────────────────────────────────────────
# Step 9 — Evaluate at both thresholds (val + test)
# ─────────────────────────────────────────────
log("STEP 9: evaluation ...")
def evaluate(y_true, prob, thr, label):
    pred = (prob >= thr).astype(int)
    return {
        "label": label,
        "threshold": float(thr),
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

# ─────────────────────────────────────────────
# Step 10 — Save artifacts
# ─────────────────────────────────────────────
log("STEP 10: save artifacts ...")

# Feature importance
imp_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False)
imp_df.to_csv(IMP_OUT, index=False)

# Predictions CSVs
pd.DataFrame({
    "y_true": y_val, "prob_attack": val_prob,
    "pred_05": (val_prob >= 0.5).astype(int),
    "pred_tuned": (val_prob >= best_thr).astype(int),
}).to_csv(PRED_VAL, index=False)

pd.DataFrame({
    "y_true": y_test, "prob_attack": test_prob,
    "pred_05": (test_prob >= 0.5).astype(int),
    "pred_tuned": (test_prob >= best_thr).astype(int),
}).to_csv(PRED_TEST, index=False)

# Meta JSON
meta = {
    "n_raw": int(n_raw),
    "n_features": int(X_train.shape[1]),
    "feature_names": X_train.columns.tolist(),
    "categorical_features": CATEGORICAL,
    "splits": {
        "train_size": int(len(X_train)),
        "val_size":   int(len(X_val)),
        "test_size":  int(len(X_test)),
        "split_rule": "temporal 70/10/20, no shuffle; sort by time first",
    },
    "class_dist": {
        "train": {"0": int((y_train==0).sum()), "1": int((y_train==1).sum())},
        "val":   {"0": int((y_val  ==0).sum()), "1": int((y_val  ==1).sum())},
        "test":  {"0": int((y_test ==0).sum()), "1": int((y_test ==1).sum())},
    },
    "best_threshold": best_thr,
    "best_iteration": int(clf.best_iteration_),
    "hyperparams": {
        "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
        "class_weight": "balanced", "random_state": SEED,
        "categorical_feature": CATEGORICAL,
    },
    "time_range": {
        "start": str(pd.to_datetime(df["time"].min() if "time" in df.columns else 0, unit="s")) if False else "see raw CSV",
        "end":   "see raw CSV",
    },
    "metrics": {
        "val_at_0.5":  {k: val_05[k]   for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_0.5": {k: test_05[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2, default=str)

# Text report
lines = []
lines.append("=" * 72)
lines.append("EVALUATION REPORT — IanArffDataset Binary Intrusion Detection (v2, minimal)")
lines.append("=" * 72)
lines.append(f"Total records: {n_raw:,}  |  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
lines.append(f"Features used: {X_train.shape[1]}  |  Categorical: {CATEGORICAL}")
lines.append(f"Best LGB iteration: {clf.best_iteration_}")
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
lines.append("-" * 72)
lines.append("[TOP-17 FEATURE IMPORTANCE]")
lines.append("-" * 72)
for _, r in imp_df.iterrows():
    bar = "█" * int(r["importance"] / max(imp_df["importance"].max(), 1) * 40)
    lines.append(f"  {r['feature']:<35s} {r['importance']:>6d}  {bar}")
lines.append("")
with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Plots (optional, fail-safe)
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
    for prob, yy, lbl in [(val_prob, y_val, "val"), (test_prob, y_test, "test")]:
        p, r, _ = precision_recall_curve(yy, prob)
        ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, prob):.3f}")
        fpr, tpr, _ = roc_curve(yy, prob)
        ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, prob):.3f}")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title("PR curve"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("FPR");    ax2.set_ylabel("TPR");     ax2.set_title("ROC curve"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)
    log(f"   plots saved: {CM_PNG}, {PRROC_PNG}")
except Exception as e:
    log(f"   plots skipped: {e}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
log("=" * 60)
log("DONE.")
log(f"   features    : {X_train.shape[1]}")
log(f"   train/val/test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
log(f"   best_thr    : {best_thr:.2f}  (val Macro-F1 = {best_f1m:.4f})")
log(f"   val@0.5     : Macro-F1={val_05['macro_f1']:.4f}  Binary-F1={val_05['binary_f1']:.4f}")
log(f"   val@{best_thr:.2f}    : Macro-F1={val_thr['macro_f1']:.4f}  Binary-F1={val_thr['binary_f1']:.4f}")
log(f"   test@0.5    : Macro-F1={test_05['macro_f1']:.4f}  Binary-F1={test_05['binary_f1']:.4f}")
log(f"   test@{best_thr:.2f}   : Macro-F1={test_thr['macro_f1']:.4f}  Binary-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   report      : {EVAL_OUT}")
log(f"   artifacts   : {BASE}")
log("=" * 60)
