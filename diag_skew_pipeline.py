"""
diag_skew_pipeline.py -- 定位 "官方 test FPR 4.4% vs 部署回放 FPR 15.6%" 的根因.

步骤:
  1) 用 preprocess_v2_scada.py 的完全相同流程, 在当前 CSV 上重建 test 段的 27-dim
  2) 与官方保存的 X_test_binary_v2_scada_window16.npy 逐元素比对
     -> 判定: 当前 CSV / 特征构造 是否与训练时一致
  3) 同一 window 网格上, 对比 "训练式特征" vs "部署式特征" (逐列差异)
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

BASE = r"C:\work\Claude\Issue"
CSV_PATH = os.path.join(BASE, "IanArffDataset.csv")
NUMERIC_COLS = ["setpoint", "gain", "reset rate", "deadband", "cycle time",
                "rate", "system mode", "control scheme", "pump", "solenoid",
                "pressure measurement"]
LABELS = ["binary result", "categorized result", "specific result"]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
WINDOW = 16

# ── 步骤 1: 完全复刻 preprocess_v2_scada.py ────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.replace("?", np.nan)
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
y_full = df[LABELS].copy()
df = df.drop(columns=LABELS)
df = df.sort_values("time").reset_index(drop=True)
y_full = y_full.loc[df.index]

df["time_diff"] = df["time"].diff().fillna(0)
key = df["address"].astype(int).values * 1000 + df["function"].astype(int).values
ts_l = np.zeros(len(df), dtype=np.float32)
last = {}
tarr = df["time"].values
for i in range(len(df)):
    k = key[i]
    ts_l[i] = (tarr[i] - last[k]) if k in last else 0.0
    last[k] = tarr[i]
df["time_since_last_same_addr_func"] = ts_l
df["is_unusual_fc"] = df["function"].apply(lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)
df["is_response"] = df["command response"].astype(np.float32)
df = df.drop(columns=["time"])
df = df.drop(columns=["command response"])

print("[1] 复刻后列数:", df.shape[1])
print("    列名:", list(df.columns))

# 只取 19 个 (训练时的 scaler 是 19 维); 末尾的 Unnamed 列是 CSV 后加的, 丢掉
COL19 = ["address", "function", "length", "setpoint", "gain", "reset rate", "deadband",
         "cycle time", "rate", "system mode", "control scheme", "pump", "solenoid",
         "pressure measurement", "crc rate", "time_diff",
         "time_since_last_same_addr_func", "is_unusual_fc", "is_response"]
df19 = df[COL19].copy()

N = len(df19)
n_train = int(N * 0.70)
n_val = int(N * 0.10)
d_tr, d_va, d_te = df19.iloc[:n_train], df19.iloc[n_train:n_train + n_val], df19.iloc[n_train + n_val:]
y_te = y_full["binary result"].values[n_train + n_val:]

print("[2] split: train=%d val=%d test=%d" % (len(d_tr), len(d_va), len(d_te)))

sc = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
for d in (d_tr, d_va, d_te):
    d["pressure measurement"] = d["pressure measurement"].clip(0, 100)
    for c in d.columns:
        d[c] = d[c].astype(np.float32)
    d.fillna(0, inplace=True)

X_te = sc.transform(d_te.values).astype(np.float32)
n = (len(X_te) // WINDOW) * WINDOW
Xw = X_te[:n].reshape(n // WINDOW, WINDOW, -1)
yw = y_te[:n].reshape(n // WINDOW, WINDOW).max(axis=1).astype(np.int64)

IDX_PRESS, IDX_CRC, IDX_ISRESP, IDX_LENGTH, IDX_ISUNUSUAL = 13, 14, 18, 2, 17
press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
crc_mean_w = Xw[:, :, IDX_CRC].mean(axis=1)
crc_max_w = Xw[:, :, IDX_CRC].max(axis=1)
cmd_count_w = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)
resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
balance_w = (cmd_count_w - resp_count_w) / WINDOW
length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(n // WINDOW)],
                             dtype=np.float32)
unusual_count_w = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)
aggs = np.column_stack([press_mean_w, crc_mean_w, crc_max_w, cmd_count_w,
                        resp_count_w, balance_w, length_nunique_w,
                        unusual_count_w]).astype(np.float32)
Xw_full = np.concatenate([Xw, np.repeat(aggs[:, None, :], WINDOW, axis=1)], axis=-1)
print("[3] 复刻的 27-dim shape:", Xw_full.shape)

# ── 步骤 2: 与官方 npy 比对 ───────────────────────────────────────────────
X_off = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
y_off = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
print("[4] 官方 npy shape:", X_off.shape, " 复刻 shape:", Xw_full.shape)
if X_off.shape == Xw_full.shape:
    d = np.abs(X_off - Xw_full)
    print("    逐元素 max|diff| = %.6g   mean|diff| = %.6g" % (d.max(), d.mean()))
    per_col = d.reshape(-1, 27).max(axis=0)
    bad = [(c, float(v)) for c, v in enumerate(per_col) if v > 1e-6]
    print("    有差异的列:", bad if bad else "无 (完全一致 ✅)")
    print("    标签一致率: %.6f" % (y_off == yw).mean())
else:
    print("    ⚠ shape 不同")
