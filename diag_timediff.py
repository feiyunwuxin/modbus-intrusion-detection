"""
diag_timediff.py -- 定位 train/serve skew 的具体来源 (col 15/16 = 两个时间特征).

对同一段 test 帧, 用三种方式算 time_diff / time_since:
  (a) 文件原始顺序 (部署端的方式: PC 直接读 CSV 行序, H743 从串口实时收)
  (b) df.sort_values("time") 之后 (训练管线 preprocess_v2_scada.py 的做法)
  (c) 官方训练 npy 里的实际值

再分别喂给部署的 C kernel, 看 FPR 差多少。
"""
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from parity_fp32_c_vs_onnx import c_forward, parse_h

BASE = r"C:\work\Claude\Issue"
CSV_PATH = os.path.join(BASE, "IanArffDataset.csv")
NUMERIC_COLS = ["setpoint", "gain", "reset rate", "deadband", "cycle time",
                "rate", "system mode", "control scheme", "pump", "solenoid",
                "pressure measurement"]
LABELS = ["binary result", "categorized result", "specific result"]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
WINDOW = 16
N_TRAIN = int(274628 * 0.70)
N_VAL = int(274628 * 0.10)
TEST_LO = N_TRAIN + N_VAL          # 0-indexed row of test split start = 219701

# ── 载入原始 CSV (保留文件顺序) ───────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.replace("?", np.nan)
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
y_full = df[LABELS].copy()
df = df.drop(columns=LABELS)
df["_af"] = df["address"].astype(int).values * 1000 + df["function"].astype(int).values
df["time_diff_raw"] = df["time"].diff().fillna(0)

# (b) 排序版
srt = df.sort_values("time", kind="quicksort").reset_index(drop=True)
sy = y_full.loc[srt.index]
srt["time_diff_sorted"] = srt["time"].diff().fillna(0)
# 排序后 time_since
ts_s = np.zeros(len(srt), dtype=np.float32)
last_s = {}
ta_s = srt["time"].values
af_s = srt["_af"].values
for i in range(len(srt)):
    k = af_s[i]
    ts_s[i] = (ta_s[i] - last_s[k]) if k in last_s else 0.0
    last_s[k] = ta_s[i]
srt["ts_since_sorted"] = ts_s

# (a) 文件顺序版 time_since
ts_f = np.zeros(len(df), dtype=np.float32)
last_f = {}
ta_f = df["time"].values
af_f = df["_af"].values
for i in range(len(df)):
    k = af_f[i]
    ts_f[i] = (ta_f[i] - last_f[k]) if k in last_f else 0.0
    last_f[k] = ta_f[i]
df["ts_since_fileorder"] = ts_f

print("[1] 排序是否改变行序:",
      "否 (相同)" if (srt["_af"].values == df["_af"].values).all() and
      (srt["time"].values == df["time"].values).all() else "是 (排序打乱了同 time 行的顺序)")
print("    文件顺序 vs 排序后 time_diff 不等的行数: %d / %d"
      % (int((df["time_diff_raw"].values != np.sort(df["time_diff_raw"].values, kind="stable")).sum()),
         len(df)))
n_perm = int((srt["_af"].values != df["_af"].values).sum())
print("    排序后地址/功能列与原顺序不同的行数: %d" % n_perm)

# ── 官方 npy ──────────────────────────────────────────────────────────────
X_off = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
y_off = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
sc = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
mu, sg = np.asarray(sc.center_), np.asarray(sc.scale_)

# 官方 npy 是 test 段按 sort 后的顺序切窗: row 219701..274612
# 对比三种来源在 test 段的 col15/col16 (缩放后)
def scaled_time_feats(frame_sorted: bool):
    d = srt if frame_sorted else df
    td = d["time_diff_sorted"].values if frame_sorted else d["time_diff_raw"].values
    ts = d["ts_since_sorted"].values if frame_sorted else d["ts_since_fileorder"].values
    seg_td = td[TEST_LO:TEST_LO + 54912].astype(np.float32)
    seg_ts = ts[TEST_LO:TEST_LO + 54912].astype(np.float32)
    w_td = ((seg_td.reshape(-1, 16) - mu[15]) / sg[15])
    w_ts = ((seg_ts.reshape(-1, 16) - mu[16]) / sg[16])
    return w_td, w_ts


td_s, ts_s = scaled_time_feats(True)
td_f, ts_f2 = scaled_time_feats(False)

off15 = X_off[:, :, 15]
off16 = X_off[:, :, 16]
print()
print("[2] col15 (time_diff) 与官方 npy 的 max|diff|:")
print("      排序版 (训练做法) : %.6g" % np.abs(off15 - td_s).max())
print("      文件序版 (部署做法): %.6g" % np.abs(off15 - td_f).max())
print("[3] col16 (time_since) 与官方 npy 的 max|diff|:")
print("      排序版 (训练做法) : %.6g" % np.abs(off16 - ts_s).max())
print("      文件序版 (部署做法): %.6g" % np.abs(off16 - ts_f2).max())
