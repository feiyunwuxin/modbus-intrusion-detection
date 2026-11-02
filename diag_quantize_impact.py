"""
diag_quantize_impact.py -- 量化 "time 取整到秒" 对 FPR 的影响.

在官方 test 网格 (3432 window) 上:
  A) 官方 npy 原样                                   (训练时的小数时间特征)
  B) 官方 npy, 但 col15/col16 换成 "秒级取整" 后的值  (当前 CSV / H743 实际拿到的)
  C) 官方 npy, 但 col15/col16 换成 "亚秒精度" 还原值  (从 ARFF 重建)

三者的 FPR / Recall 用同一个部署 C kernel 算。
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
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
WINDOW = 16
N_TRAIN = int(274628 * 0.70)
N_VAL = int(274628 * 0.10)
TEST_LO = N_TRAIN + N_VAL

# ── 部署权重 ──────────────────────────────────────────────────────────────
raw = parse_h("MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h")
W = {"w_tcn_0_conv1": raw["v4se23_ch32_s42_w_tcn_0_conv1_weight"],
     "b_tcn_0_conv1": raw["v4se23_ch32_s42_b_tcn_0_conv1_bias"],
     "w_tcn_0_conv2": raw["v4se23_ch32_s42_w_tcn_0_conv2_weight"],
     "b_tcn_0_conv2": raw["v4se23_ch32_s42_b_tcn_0_conv2_bias"],
     "w_tcn_0_residual": raw["v4se23_ch32_s42_w_tcn_0_residual_weight"],
     "b_tcn_0_residual": raw["v4se23_ch32_s42_b_tcn_0_residual_bias"],
     "fc1": raw["v4se23_ch32_s42_fc1_weight"], "b_fc1": raw["v4se23_ch32_s42_b_fc1_bias"],
     "fc2": raw["v4se23_ch32_s42_fc2_weight"], "b_fc2": raw["v4se23_ch32_s42_b_fc2_bias"]}
for bi in (0, 1, 2):
    for nm in ("conv1", "conv2", "se_fc1", "se_fc2"):
        W["w_tcn_%d_%s" % (bi, nm)] = raw[f"v4se23_ch32_s42_w_tcn_{bi}_{nm}_weight"]
        W["b_tcn_%d_%s" % (bi, nm)] = raw[f"v4se23_ch32_s42_b_tcn_{bi}_{nm}_bias"]


def forward_batch(X23):
    out = np.empty(len(X23))
    for i, x in enumerate(X23):
        out[i] = 1.0 / (1.0 + math.exp(-c_forward(W, x)))
    return out


def metrics(probs, y, thr):
    det = probs >= thr
    tp = int((det & (y == 1)).sum()); fp = int((det & (y == 0)).sum())
    fn = int((~det & (y == 1)).sum()); tn = int((~det & (y == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, P=P, R=R, F1=2 * P * R / max(P + R, 1e-9),
                FPR=fp / max(fp + tn, 1))


# ── 官方 npy (训练特征, 小数时间) ─────────────────────────────────────────
X_off = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
y_off = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
sc = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
mu, sg = np.asarray(sc.center_), np.asarray(sc.scale_)

# ── 从 ARFF 重建亚秒 time, 算 time_diff / time_since ─────────────────────
print("[1] 读 ARFF 的 time 列 (亚秒) ...")
arff_times = []
in_data = False
with open(os.path.join(BASE, "IanArffDataset.arff"), encoding="utf-8", errors="replace") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if s.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data:
            continue
        p = s.split(",")
        if len(p) == 20:
            arff_times.append((float(p[0]), float(p[1]), float(p[16])))
arff_times = np.array(arff_times, dtype=np.float64)
print("    ARFF 行数: %d (CSV 274628)" % len(arff_times))

t_arff = arff_times[:, 2]
af_arff = arff_times[:, 0] * 1000 + arff_times[:, 1]
td_a = np.zeros(len(t_arff)); ts_a = np.zeros(len(t_arff))
last = {}
for i in range(len(t_arff)):
    td_a[i] = (t_arff[i] - t_arff[i - 1]) if i > 0 else 0.0
    k = af_arff[i]
    ts_a[i] = (t_arff[i] - last[k]) if k in last else 0.0
    last[k] = t_arff[i]

# ── 当前 CSV (秒级取整) ──────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "IanArffDataset.csv"))
t_csv = df["time"].values.astype(np.float64)
af_csv = df["address"].values.astype(np.float64) * 1000 + df["function"].values.astype(np.float64)
td_c = np.zeros(len(t_csv)); ts_c = np.zeros(len(t_csv))
last = {}
for i in range(len(t_csv)):
    td_c[i] = (t_csv[i] - t_csv[i - 1]) if i > 0 else 0.0
    k = af_csv[i]
    ts_c[i] = (t_csv[i] - last[k]) if k in last else 0.0
    last[k] = t_csv[i]

print("    ARFF time_diff 前 12:", np.array2string(td_a[219701:219713], precision=3, max_line_width=200))
print("    CSV  time_diff 前 12:", np.array2string(td_c[219701:219713], precision=3, max_line_width=200))

SEG = 54912


def win_scaled(vals):
    seg = vals[TEST_LO:TEST_LO + SEG].astype(np.float32)
    w = (seg.reshape(-1, 16) - mu[15]) / sg[15]
    return w


def win_scaled16(vals):
    seg = vals[TEST_LO:TEST_LO + SEG].astype(np.float32)
    return (seg.reshape(-1, 16) - mu[16]) / sg[16]


variants = {
    "A 官方 npy (训练用, 亚秒 time)": X_off,
    "B 官方 npy + 秒级 time (当前 CSV/H743)": None,
    "C 官方 npy + ARFF 亚秒 time 重建": None,
}
Xb = X_off.copy()
Xb[:, :, 15] = win_scaled(td_c)
Xb[:, :, 16] = win_scaled16(ts_c)
variants["B 官方 npy + 秒级 time (当前 CSV/H743)"] = Xb

Xc = X_off.copy()
Xc[:, :, 15] = win_scaled(td_a)
Xc[:, :, 16] = win_scaled16(ts_a)
variants["C 官方 npy + ARFF 亚秒 time 重建"] = Xc

print()
print("=" * 100)
print("同一 test 网格 (3432 window), 部署 C kernel, thr=0.5")
print("=" * 100)
for tag, X in variants.items():
    X23 = np.clip(X[:, :, KEEP_23], -10.0, 10.0).transpose(0, 2, 1).astype(np.float32)
    pr = forward_batch(X23)
    m = metrics(pr, y_off, 0.5)
    print("  %-38s TP=%5d FP=%5d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (tag, m["TP"], m["FP"], m["FN"], m["P"], m["R"], m["F1"], m["FPR"]))
    np.save("probs_%s.npy" % tag[0], pr)
