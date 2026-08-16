#!/usr/bin/env python3
"""
Preprocessing v2 — adds SCADA protocol-derived features.

New features (10 total, beyond the original 17):
  Row-level (per row):
    1. time_since_last_same_addr_func (backward time)
    2. is_unusual_fc (binary)
    3. is_response (alias of command response)
  Window-level (per window, broadcasted to all 16 timesteps):
    4. cmd_count_w
    5. resp_count_w
    6. cmd_resp_balance_w = (cmd-resp)/16
    7. crc_mean_w
    8. crc_max_w
    9. press_mean_w
    10. length_nunique_w

Pipeline:
  1. Read raw CSV, sort by time
  2. Add ROW-LEVEL features (across the whole sorted dataset)
  3. 70/10/20 temporal split (no shuffle)
  4. RobustScaler fit on train only
  5. Window=16 grouping
  6. Per-window aggregation features (broadcasted to 16 timesteps)
  7. Save X_*/y_*_binary_v2.npy and meta JSON

Output column ordering: same as v1 (16 raw + time_diff) + 10 new = 27 total.
The new features go in cols 17-26.
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

BASE = r"C:\work\Claude\Issue"
CSV_PATH = os.path.join(BASE, "IanArffDataset.csv")

OUT_DIR  = BASE
OUT_PREFIX = "v2_scada"

UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}  # command-only FCs

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ─────────────────────────────────────────────
log("STEP 1: read raw CSV ...")
df = pd.read_csv(CSV_PATH)
log(f"   raw shape: {df.shape}")

# Replace ? with NaN and convert numeric cols
NUMERIC_COLS = ["setpoint", "gain", "reset rate", "deadband", "cycle time",
                "rate", "system mode", "control scheme", "pump", "solenoid",
                "pressure measurement"]
df = df.replace("?", np.nan)
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop label cols
LABELS = ["binary result", "categorized result", "specific result"]
y_full = df[LABELS].copy()
df = df.drop(columns=LABELS)

# Sort by time
df = df.sort_values("time").reset_index(drop=True)
y_full = y_full.loc[df.index]
log(f"   after sort: {df.shape}")

# ─────────────────────────────────────────────
log("STEP 2: add row-level features ...")

# time_diff
df["time_diff"] = df["time"].diff().fillna(0)

# time_since_last_same_addr_func (backward)
log("   computing time_since_last_same_addr_func ...")
df["_addr_func"] = df["address"].astype(int) * 1000 + df["function"].astype(int)
last_seen_time = {}
ts_l = np.zeros(len(df), dtype=np.float32)
addr_func_arr = df["_addr_func"].values
time_arr = df["time"].values
for i in range(len(df)):
    key = addr_func_arr[i]
    if key in last_seen_time:
        ts_l[i] = time_arr[i] - last_seen_time[key]
    else:
        ts_l[i] = 0.0
    last_seen_time[key] = time_arr[i]
df["time_since_last_same_addr_func"] = ts_l
df = df.drop(columns=["_addr_func"])

# is_unusual_fc
df["is_unusual_fc"] = df["function"].apply(lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)

# is_response (just the existing column, normalized)
df["is_response"] = df["command response"].astype(np.float32)

# Drop time
df = df.drop(columns=["time"])

# Drop command response (we have is_response now)
df = df.drop(columns=["command response"])

log(f"   row-level features added. new shape: {df.shape}")
log(f"   columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────
log("STEP 3: split 70/10/20 (temporal) ...")
N = len(df)
n_train = int(N * 0.70)
n_val   = int(N * 0.10)
df_train = df.iloc[:n_train].reset_index(drop=True)
df_val   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
df_test  = df.iloc[n_train + n_val:].reset_index(drop=True)
y_train  = y_full.iloc[:n_train].reset_index(drop=True)
y_val    = y_full.iloc[n_train:n_train + n_val].reset_index(drop=True)
y_test   = y_full.iloc[n_train + n_val:].reset_index(drop=True)
log(f"   train: {df_train.shape}  val: {df_val.shape}  test: {df_test.shape}")

# ─────────────────────────────────────────────
log("STEP 4: fill structural NaN with 0 ...")
# Original pipeline fills NaN with 0 (since `?` is structural absence)
df_train = df_train.fillna(0)
df_val   = df_val.fillna(0)
df_test  = df_test.fillna(0)

# Special clip on pressure
for d in [df_train, df_val, df_test]:
    d["pressure measurement"] = d["pressure measurement"].clip(0, 100)
    for c in d.columns:
        d[c] = d[c].astype(np.float32)

# ─────────────────────────────────────────────
log("STEP 5: RobustScaler fit on train ...")
scaler = RobustScaler()
X_train_raw = scaler.fit_transform(df_train.values)
X_val_raw   = scaler.transform(df_val.values)
X_test_raw  = scaler.transform(df_test.values)
N_FEATURES  = X_train_raw.shape[1]
log(f"   scaled shape: train={X_train_raw.shape}  features={N_FEATURES}")

# Save scaler
import joblib
SCALER_OUT = os.path.join(BASE, "scaler_binary_v2_scada.joblib")
joblib.dump(scaler, SCALER_OUT)

# ─────────────────────────────────────────────
log("STEP 6: window=16 + per-window aggregates ...")
WINDOW = 16

def make_windows_with_agg(X, y_bin, win):
    """Make windows AND add per-window aggregates (broadcast to all timesteps)."""
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y_bin[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    # Aggregates per window:
    # indices in the original feature order:
    # 0=address, 1=function, 2=length, 3..12=numeric setpoint..solenoid,
    # 13=pressure measurement, 14=crc rate, 15=is_response, 16=time_diff,
    # 17=time_since_last, 18=is_unusual_fc, 19=is_response_dup
    # Wait, the "command response" was dropped; is_response is at some position.
    # Let me re-derive the column order:
    # After "drop time, drop command response, add 3 row features", columns are:
    # 0=address, 1=function, 2=length,
    # 3=setpoint, 4=gain, 5=reset rate, 6=deadband, 7=cycle time, 8=rate,
    # 9=system mode, 10=control scheme, 11=pump, 12=solenoid,
    # 13=pressure measurement, 14=crc rate,
    # 15=time_diff, 16=time_since_last_same_addr_func, 17=is_unusual_fc, 18=is_response
    # = 19 columns total
    # Window-level aggregates: use the values from the actual Xw
    # Xw is (n_win, win, 19)
    n_win = Xw.shape[0]
    # Feature indices for aggregates
    IDX_PRESS   = 13
    IDX_CRC     = 14
    IDX_ISRESP  = 18
    IDX_LENGTH  = 2
    IDX_ISUNUSUAL = 17

    press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)  # (n_win,)
    crc_mean_w   = Xw[:, :, IDX_CRC].mean(axis=1)
    crc_max_w    = Xw[:, :, IDX_CRC].max(axis=1)
    # cmd_count: 1 - is_response
    cmd_count_w  = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)  # how many are commands
    resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
    balance_w    = (cmd_count_w - resp_count_w) / win
    length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(n_win)], dtype=np.float32)
    unusual_count_w  = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)

    # Broadcast back to (n_win, win) shape
    aggs = np.column_stack([
        press_mean_w,        # 19
        crc_mean_w,          # 20
        crc_max_w,           # 21
        cmd_count_w,         # 22
        resp_count_w,        # 23
        balance_w,           # 24
        length_nunique_w,    # 25
        unusual_count_w,     # 26
    ]).astype(np.float32)   # (n_win, 8)
    # tile to (n_win, win, 8)
    aggs_tiled = np.repeat(aggs[:, np.newaxis, :], win, axis=1)
    # Concatenate
    Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)  # (n_win, win, 19+8=27)
    return Xw_full, yw

y_train_bin = y_train["binary result"].values
y_val_bin   = y_val["binary result"].values
y_test_bin  = y_test["binary result"].values

X_train_w, y_train_w = make_windows_with_agg(X_train_raw, y_train_bin, WINDOW)
X_val_w,   y_val_w   = make_windows_with_agg(X_val_raw,   y_val_bin,   WINDOW)
X_test_w,  y_test_w  = make_windows_with_agg(X_test_raw,  y_test_bin,  WINDOW)
log(f"   final windowed shape: train={X_train_w.shape}  val={X_val_w.shape}  test={X_test_w.shape}")
log(f"   final feature dim per step: {X_train_w.shape[-1]}")

# ─────────────────────────────────────────────
log("STEP 7: save artifacts ...")

# Save the row-level (non-windowed) versions too, for row-level models
np.save(os.path.join(OUT_DIR, f"X_train_binary_{OUT_PREFIX}.npy"), X_train_raw)
np.save(os.path.join(OUT_DIR, f"X_val_binary_{OUT_PREFIX}.npy"),   X_val_raw)
np.save(os.path.join(OUT_DIR, f"X_test_binary_{OUT_PREFIX}.npy"),  X_test_raw)
np.save(os.path.join(OUT_DIR, f"y_train_binary_{OUT_PREFIX}.npy"), y_train_bin)
np.save(os.path.join(OUT_DIR, f"y_val_binary_{OUT_PREFIX}.npy"),   y_val_bin)
np.save(os.path.join(OUT_DIR, f"y_test_binary_{OUT_PREFIX}.npy"),  y_test_bin)

# Save the windowed versions
np.save(os.path.join(OUT_DIR, f"X_train_binary_{OUT_PREFIX}_window16.npy"), X_train_w)
np.save(os.path.join(OUT_DIR, f"X_val_binary_{OUT_PREFIX}_window16.npy"),   X_val_w)
np.save(os.path.join(OUT_DIR, f"X_test_binary_{OUT_PREFIX}_window16.npy"),  X_test_w)
np.save(os.path.join(OUT_DIR, f"y_train_binary_{OUT_PREFIX}_window16.npy"), y_train_w)
np.save(os.path.join(OUT_DIR, f"y_val_binary_{OUT_PREFIX}_window16.npy"),   y_val_w)
np.save(os.path.join(OUT_DIR, f"y_test_binary_{OUT_PREFIX}_window16.npy"),  y_test_w)

# Save meta
META_OUT = os.path.join(BASE, f"processed_meta_{OUT_PREFIX}.json")
meta = {
    "preprocessing_version": "v2_scada",
    "n_features_total": N_FEATURES,
    "n_features_row_level": 19,
    "n_features_window_aggregates": 8,
    "n_features_windowed_per_step": int(X_train_w.shape[-1]),
    "feature_order_row_level": [
        "address", "function", "length",
        "setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
        "system mode", "control scheme", "pump", "solenoid",
        "pressure measurement", "crc rate",
        "time_diff", "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
    ],
    "feature_order_windowed_per_step": [
        "address", "function", "length",
        "setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
        "system mode", "control scheme", "pump", "solenoid",
        "pressure measurement", "crc rate",
        "time_diff", "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
        "press_mean_w", "crc_mean_w", "crc_max_w",
        "cmd_count_w", "resp_count_w", "cmd_resp_balance_w",
        "length_nunique_w", "unusual_count_w",
    ],
    "window": WINDOW,
    "splits": {
        "train": int(len(X_train_raw)),
        "val":   int(len(X_val_raw)),
        "test":  int(len(X_test_raw)),
    },
    "splits_windows": {
        "train": int(len(X_train_w)),
        "val":   int(len(X_val_w)),
        "test":  int(len(X_test_w)),
    },
    "scaler": "RobustScaler",
    "unusual_fcs": sorted(UNUSUAL_FCS),
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

log(f"   meta saved: {META_OUT}")
log("DONE preprocessing v2.")
