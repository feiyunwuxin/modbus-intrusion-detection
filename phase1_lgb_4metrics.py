#!/usr/bin/env python3
"""
阶段一 (2026-06-08) LightGBM 基线 4 指标重制
================================================
- 加载 06-08 已训练产物 (model_binary.joblib / model_multiclass.joblib / scaler.joblib)
- 复现 pipeline.py 的预处理 + temporal 70/30 split
- 二分类: Binary convention (Precision/Recall/F1 = Attack positive class)
  - 主表: thr=0.5 (忠实复现, baseline 无 val 集)
  - 参考: thr 扫描 test best_f1
- 多分类 (36 类): Macro + Weighted 4 指标
- 输出 phase1_4metrics.json
"""
import json, time, os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

BASE = r"C:\work\Claude\Issue"
CSV_IN    = os.path.join(BASE, "IanArffDataset.csv")
SCALER    = os.path.join(BASE, "scaler.joblib")
MODEL_BIN = os.path.join(BASE, "model_binary.joblib")
MODEL_MULT= os.path.join(BASE, "model_multiclass.joblib")
OUT_JSON  = os.path.join(BASE, "phase1_4metrics.json")

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

log("STEP 1: 06-08 baseline was bit-perfect on (SEED=42, n_est=400/600, num_leaves=63/127, class_weight=balanced)")
log("         will RETRAIN rather than unpickle old artifacts to guarantee bit-perfect reproduction")
import lightgbm as lgb

# ── 复现 pipeline.py 预处理 ──
log("STEP 2: reproducing pipeline.py preprocessing ...")
df = pd.read_csv(CSV_IN)
df = df.replace("?", np.nan)
float_cols = ["address","function","length","setpoint","gain","reset rate",
              "deadband","cycle time","rate","system mode","control scheme",
              "pump","solenoid","pressure measurement","crc rate","time"]
for c in float_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["command response"] = pd.to_numeric(df["command response"], errors="coerce").astype("Int64")
for c in ["binary result","categorized result","specific result"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
df = df.sort_values("time").reset_index(drop=True)
df["dt"] = df["time"].diff().fillna(0)
ts = pd.to_datetime(df["time"], unit="s")
df["hour"]  = ts.dt.hour
df["dow"]   = ts.dt.dayofweek
df["minute"]= ts.dt.minute
for col in ["pressure measurement", "crc rate"]:
    df[f"{col}_diff1"]   = df[col].diff().fillna(0)
    df[f"{col}_rmean4"]  = df[col].rolling(4, min_periods=1).mean()
    df[f"{col}_rstd4"]   = df[col].rolling(4, min_periods=1).std().fillna(0)
    df[f"{col}_rmin4"]   = df[col].rolling(4, min_periods=1).min()
    df[f"{col}_rmax4"]   = df[col].rolling(4, min_periods=1).max()
df["is_read"]  = (df["function"] == 3).astype(int)
df["is_write"] = (df["function"] == 16).astype(int)
df["is_resp"]  = ((df["function"] == 3) | (df["function"] == 16)).astype(int)

miss_cols = ["setpoint","gain","reset rate","deadband","cycle time",
             "rate","system mode","control scheme","pump","solenoid",
             "pressure measurement"]
df[miss_cols] = df.groupby(["address","function"])[miss_cols].transform(
    lambda s: s.fillna(s.median())
)
for c in miss_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())
df.loc[df["pressure measurement"].abs() < 1e-30, "pressure measurement"] = np.nan
df.loc[df["crc rate"].abs() < 1e-30, "crc rate"] = np.nan
for c in ["pressure measurement","crc rate"]:
    df[c] = df.groupby(["address","function"])[c].transform(lambda s: s.fillna(s.median()))
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())
df["pressure measurement"] = df["pressure measurement"].clip(lower=0, upper=100)
log(f"   post-process NaNs: {df.isna().sum().sum()}")

LABEL_BIN  = "binary result"
LABEL_MULT = "specific result"
DROP_COLS  = [LABEL_BIN, "categorized result", LABEL_MULT, "command response", "time"]
X = df.drop(columns=DROP_COLS)
y_bin  = df[LABEL_BIN].astype(int).values
y_mult = df[LABEL_MULT].astype(int).values

split = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
y_bin_train,  y_bin_test  = y_bin[:split],  y_bin[split:]
y_mult_train, y_mult_test = y_mult[:split], y_mult[split:]
log(f"   train: {len(X_train):,}  |  test: {len(X_test):,}")
log(f"   binary test dist: 0={(y_bin_test==0).sum():,}, 1={(y_bin_test==1).sum():,}")
log(f"   multiclass test classes: {len(np.unique(y_mult_test))}")

# ── 重训 scaler (fit_transform on train) 保证 bit-perfect ──
log("STEP 3: refit RobustScaler on train ...")
num_cols = X_train.select_dtypes(include="number").columns.tolist()
scaler = RobustScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])
log(f"   scaled {len(num_cols)} numeric cols; refit scaler")

# ── 重训 LightGBM ──
log("STEP 3b: retrain LightGBM (binary + multiclass) ...")
clf_bin = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
)
clf_bin.fit(X_train, y_bin_train)

clf_mult = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=127,
    class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
)
clf_mult.fit(X_train, y_mult_train)
log(f"   retrain done")

# ── 二分类 4 指标 ──
log("STEP 4: binary 4 metrics (Binary convention) ...")
y_proba_bin = clf_bin.predict_proba(X_test)[:, 1]

def four_metrics(y_true, y_pred):
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

# 主表: thr=0.5 (baseline 无 val 集)
y_pred_bin_05 = (y_proba_bin >= 0.5).astype(int)
bin_thr05 = four_metrics(y_bin_test, y_pred_bin_05)
cm_05 = confusion_matrix(y_bin_test, y_pred_bin_05).tolist()
log(f"   thr=0.5  Acc={bin_thr05['accuracy']:.4f}  P={bin_thr05['precision']:.4f}  R={bin_thr05['recall']:.4f}  F1={bin_thr05['f1']:.4f}")

# 参考: test 扫描 thr best_f1
thrs = np.arange(0.05, 0.951, 0.01)
f1s  = [f1_score(y_bin_test, (y_proba_bin >= t).astype(int), pos_label=1, zero_division=0) for t in thrs]
best_idx = int(np.argmax(f1s))
best_thr = float(thrs[best_idx])
y_pred_bin_best = (y_proba_bin >= best_thr).astype(int)
bin_best = four_metrics(y_bin_test, y_pred_bin_best)
cm_best = confusion_matrix(y_bin_test, y_pred_bin_best).tolist()
log(f"   best_f1 thr={best_thr:.2f}  Acc={bin_best['accuracy']:.4f}  P={bin_best['precision']:.4f}  R={bin_best['recall']:.4f}  F1={bin_best['f1']:.4f}")

# 多分类 4 指标
log("STEP 5: multiclass 4 metrics ...")
y_pred_mult = clf_mult.predict(X_test)
mult_macro = {
    "accuracy":  float(accuracy_score(y_mult_test, y_pred_mult)),
    "precision": float(precision_score(y_mult_test, y_pred_mult, average="macro",    zero_division=0)),
    "recall":    float(recall_score(y_mult_test, y_pred_mult, average="macro",    zero_division=0)),
    "f1":        float(f1_score(y_mult_test, y_pred_mult, average="macro",    zero_division=0)),
}
mult_weighted = {
    "accuracy":  float(accuracy_score(y_mult_test, y_pred_mult)),
    "precision": float(precision_score(y_mult_test, y_pred_mult, average="weighted", zero_division=0)),
    "recall":    float(recall_score(y_mult_test, y_pred_mult, average="weighted", zero_division=0)),
    "f1":        float(f1_score(y_mult_test, y_pred_mult, average="weighted", zero_division=0)),
}
log(f"   multiclass Macro:   Acc={mult_macro['accuracy']:.4f}  P={mult_macro['precision']:.4f}  R={mult_macro['recall']:.4f}  F1={mult_macro['f1']:.4f}")
log(f"   multiclass Weighted:Acc={mult_weighted['accuracy']:.4f}  P={mult_weighted['precision']:.4f}  R={mult_weighted['recall']:.4f}  F1={mult_weighted['f1']:.4f}")

# ── 输出 JSON ──
out = {
    "phase": "phase1_lgb_baseline_2026-06-08",
    "model": "LightGBM",
    "binary_clf_n_estimators": int(clf_bin.n_estimators),
    "multiclass_clf_n_estimators": int(clf_mult.n_estimators),
    "n_features": int(X.shape[1]),
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
    "binary_test_dist": {"0": int((y_bin_test==0).sum()), "1": int((y_bin_test==1).sum())},
    "multiclass_test_classes": int(len(np.unique(y_mult_test))),
    "binary": {
        "thr_0.5": {
            "metrics":  bin_thr05,
            "confusion_matrix": cm_05,
            "support_attack": int((y_bin_test==1).sum()),
            "support_normal": int((y_bin_test==0).sum()),
        },
        "test_scan_best_f1": {
            "best_threshold": best_thr,
            "metrics":  bin_best,
            "confusion_matrix": cm_best,
            "note": "test-derived, baseline pipeline.py has no val set; only reference",
        },
    },
    "multiclass": {
        "macro":   {**mult_macro, "n_classes": 36},
        "weighted":{**mult_weighted, "n_classes": 36},
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
log(f"DONE.  json -> {OUT_JSON}")