#!/usr/bin/env python3
"""
Intrusion Detection Pipeline for IanArffDataset (SCADA Gas Pipeline)
Steps:
  1. Load + type conversion
  2. Feature engineering (time, rolling, protocol)
  3. Missing value handling + outlier cleanup
  4. Temporal 70/30 split + RobustScaler
  5. Train LightGBM (binary + multi-class) + evaluate
  6. Save processed data, models, evaluation, feature importance
"""

import time, json, sys, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import lightgbm as lgb
import joblib

BASE = r"C:\work\Claude\issue"
CSV_IN   = os.path.join(BASE, "IanArffDataset.csv")
PROC_OUT = os.path.join(BASE, "IanArffDataset_processed.csv")
META_OUT = os.path.join(BASE, "processed_meta.json")
EVAL_OUT = os.path.join(BASE, "evaluation_report.txt")
IMP_BIN = os.path.join(BASE, "feature_importance_binary.csv")
IMP_MULT = os.path.join(BASE, "feature_importance_multiclass.csv")
MODEL_BIN = os.path.join(BASE, "model_binary.joblib")
MODEL_MULT = os.path.join(BASE, "model_multiclass.joblib")
SCALER = os.path.join(BASE, "scaler.joblib")

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ──────────────────────────────────────────────────────────
# ① 加载 + 类型转换
# ──────────────────────────────────────────────────────────
log("STEP 1: loading data ...")
df = pd.read_csv(CSV_IN)
n_raw = len(df)
log(f"   loaded {n_raw:,} rows, {df.shape[1]} cols")

df = df.replace("?", np.nan)

float_cols = ["address","function","length","setpoint","gain","reset rate",
              "deadband","cycle time","rate","system mode","control scheme",
              "pump","solenoid","pressure measurement","crc rate","time"]
for c in float_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["command response"] = pd.to_numeric(df["command response"], errors="coerce").astype("Int64")
for c in ["binary result","categorized result","specific result"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# 按时间排序
df = df.sort_values("time").reset_index(drop=True)
log(f"   sorted by time, range {pd.to_datetime(df['time'].min(), unit='s')} -> {pd.to_datetime(df['time'].max(), unit='s')}")

# ──────────────────────────────────────────────────────────
# ② 特征工程
# ──────────────────────────────────────────────────────────
log("STEP 2: feature engineering ...")
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
log(f"   feature cols now: {df.shape[1]}")

# ──────────────────────────────────────────────────────────
# ③ 缺失值处理 + 异常值清洗
# ──────────────────────────────────────────────────────────
log("STEP 3: missing values + outliers ...")
miss_cols = ["setpoint","gain","reset rate","deadband","cycle time",
             "rate","system mode","control scheme","pump","solenoid",
             "pressure measurement"]

# 用 (address, function) 组内中位数填补
df[miss_cols] = df.groupby(["address","function"])[miss_cols].transform(
    lambda s: s.fillna(s.median())
)
# 兜底：整列中位数
for c in miss_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())
log(f"   missing filled; remaining NaNs: {df.isna().sum().sum()}")

# 极小值置 NaN 重填一次
df.loc[df["pressure measurement"].abs() < 1e-30, "pressure measurement"] = np.nan
df.loc[df["crc rate"].abs() < 1e-30, "crc rate"] = np.nan
for c in ["pressure measurement","crc rate"]:
    df[c] = df.groupby(["address","function"])[c].transform(lambda s: s.fillna(s.median()))
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# 物理裁剪
df["pressure measurement"] = df["pressure measurement"].clip(lower=0, upper=100)
log(f"   pressure clipped to [0,100]; remaining NaNs: {df.isna().sum().sum()}")

# ──────────────────────────────────────────────────────────
# ④ 时序切分 + 归一化
# ──────────────────────────────────────────────────────────
log("STEP 4: temporal split + scaling ...")
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
log(f"   train: {len(X_train):,} | test: {len(X_test):,}")
log(f"   binary train dist: 0={int((y_bin_train==0).sum()):,}, 1={int((y_bin_train==1).sum()):,}")
log(f"   multiclass train classes: {len(np.unique(y_mult_train))}, test classes: {len(np.unique(y_mult_test))}")

# 归一化
num_cols = X_train.select_dtypes(include="number").columns.tolist()
scaler = RobustScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])
joblib.dump(scaler, SCALER)
log(f"   scaled {len(num_cols)} numeric cols; scaler saved -> {SCALER}")

# 保存处理后的全量（标准化前的版本更有用）
processed_df = df.copy()
processed_df.to_csv(PROC_OUT, index=False)
log(f"   processed CSV saved -> {PROC_OUT}")

# 元信息
meta = {
    "n_raw": int(n_raw),
    "n_features": int(X.shape[1]),
    "feature_names": X.columns.tolist(),
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
    "binary_train_dist": {"0": int((y_bin_train==0).sum()), "1": int((y_bin_train==1).sum())},
    "multiclass_classes": int(len(np.unique(y_mult))),
    "time_range": {
        "start": str(pd.to_datetime(df['time'].min(), unit='s')),
        "end":   str(pd.to_datetime(df['time'].max(), unit='s')),
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)
log(f"   meta saved -> {META_OUT}")

# ──────────────────────────────────────────────────────────
# ⑤ 训练 + 评估
# ──────────────────────────────────────────────────────────
log("STEP 5: training LightGBM ...")
report_lines = []
report_lines.append("="*70)
report_lines.append("EVALUATION REPORT — IanArffDataset Intrusion Detection")
report_lines.append("="*70)
report_lines.append(f"Total records: {n_raw:,}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
report_lines.append(f"Features used: {X.shape[1]}")
report_lines.append("")

# ── 5a. 二分类 ──
log("   [5a] binary task (binary result) ...")
bin_clf = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
)
bin_clf.fit(X_train, y_bin_train)
y_pred_bin = bin_clf.predict(X_test)
f1_bin_macro = f1_score(y_bin_test, y_pred_bin, average="macro")
f1_bin_bi    = f1_score(y_bin_test, y_pred_bin, average="binary")
report_lines.append("─"*70)
report_lines.append("[A] BINARY CLASSIFICATION  (0=Normal, 1=Attack)")
report_lines.append("─"*70)
report_lines.append(classification_report(y_bin_test, y_pred_bin, digits=4))
report_lines.append(f"Macro-F1  : {f1_bin_macro:.4f}")
report_lines.append(f"Binary-F1 : {f1_bin_bi:.4f}")
log(f"   binary Macro-F1 = {f1_bin_macro:.4f}")
joblib.dump(bin_clf, MODEL_BIN)

# 特征重要性
imp_bin_df = pd.DataFrame({
    "feature": X.columns,
    "importance": bin_clf.feature_importances_
}).sort_values("importance", ascending=False)
imp_bin_df.to_csv(IMP_BIN, index=False)

# ── 5b. 多分类 ──
log("   [5b] multi-class task (specific result, 36 classes) ...")
mult_clf = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=127,
    class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
)
mult_clf.fit(X_train, y_mult_train)
y_pred_mult = mult_clf.predict(X_test)
f1_mult_macro = f1_score(y_mult_test, y_pred_mult, average="macro")
f1_mult_w     = f1_score(y_mult_test, y_pred_mult, average="weighted")
report_lines.append("")
report_lines.append("─"*70)
report_lines.append("[B] MULTI-CLASS CLASSIFICATION  (specific result, 36 classes)")
report_lines.append("─"*70)
report_lines.append(classification_report(y_mult_test, y_pred_mult, digits=4, zero_division=0))
report_lines.append(f"Macro-F1    : {f1_mult_macro:.4f}")
report_lines.append(f"Weighted-F1 : {f1_mult_w:.4f}")
log(f"   multi-class Macro-F1 = {f1_mult_macro:.4f}, Weighted-F1 = {f1_mult_w:.4f}")
joblib.dump(mult_clf, MODEL_MULT)

imp_mult_df = pd.DataFrame({
    "feature": X.columns,
    "importance": mult_clf.feature_importances_
}).sort_values("importance", ascending=False)
imp_mult_df.to_csv(IMP_MULT, index=False)

# ── 5c. 特征重要性 Top 15 ──
report_lines.append("")
report_lines.append("─"*70)
report_lines.append("[C] TOP-15 FEATURE IMPORTANCE (multi-class)")
report_lines.append("─"*70)
for i, row in imp_mult_df.head(15).iterrows():
    bar = "█" * int(row["importance"] / max(imp_mult_df["importance"].max(),1) * 40)
    report_lines.append(f"  {row['feature']:<35s} {row['importance']:>6d}  {bar}")
report_lines.append("")

with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"   evaluation saved -> {EVAL_OUT}")

# ──────────────────────────────────────────────────────────
# ⑥ 总结
# ──────────────────────────────────────────────────────────
log("STEP 6: summary ...")
log(f"   binary  Macro-F1 = {f1_bin_macro:.4f}")
log(f"   multi   Macro-F1 = {f1_mult_macro:.4f}, Weighted-F1 = {f1_mult_w:.4f}")
log("DONE.")
