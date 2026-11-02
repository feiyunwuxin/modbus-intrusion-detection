"""V6: FINAL comparison — training pipeline vs H743 pipeline on first 80 rows.
Goal: pinpoint the exact feature(s) that cause the model to fire attack on all normal frames.
"""
import os, sys
sys.path.insert(0, r'C:\work\Claude\Issue')
import numpy as np
import pandas as pd
import torch
import joblib
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE

BASE = r'C:\work\Claude\Issue'
CSV_PATH = os.path.join(BASE, 'IanArffDataset.csv')
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}

df = pd.read_csv(CSV_PATH)
df = df.rename(columns={
    'reset': 'reset rate', 'cycle': 'cycle time',
    'system': 'system mode', 'control': 'control scheme',
    'pressure': 'pressure measurement',
    'crc': 'crc rate', 'command': 'command response',
    'binary': 'binary result', 'categorized': 'categorized result',
    'specific': 'specific result'
})

NUMERIC_COLS = ["setpoint", "gain", "reset rate", "deadband", "cycle time",
                "rate", "system mode", "control scheme", "pump", "solenoid",
                "pressure measurement"]
df = df.replace("?", np.nan)
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# === Original 19-dim columns the scaler was fit on ===
# Per retrain_tcn_23dim script, the 19-dim CSV columns are:
ORIG_19 = ["address", "function", "length", "setpoint", "gain", "reset rate",
           "deadband", "cycle time", "rate", "system mode", "control scheme",
           "pump", "solenoid", "pressure measurement", "crc rate",
           "Unnamed: 20", "Unnamed: 21", "time_diff", "time_since_last_same_addr_func"]

# Compute time_diff and time_since properly
df["time_diff"] = df["time"].diff().fillna(0)
df["_addr_func"] = df["address"].astype(int) * 1000 + df["function"].astype(int)
last_seen = {}
ts_l = np.zeros(len(df), dtype=np.float32)
for i in range(len(df)):
    k = df["_addr_func"].iloc[i]
    if k in last_seen:
        ts_l[i] = df["time"].iloc[i] - last_seen[k]
    last_seen[k] = df["time"].iloc[i]
df["time_since_last_same_addr_func"] = ts_l
df = df.drop(columns=["_addr_func"])

df["is_unusual_fc"] = df["function"].apply(lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)
df["is_response"] = df["command response"].astype(np.float32)

LABELS = ["binary result", "categorized result", "specific result"]
y_full = df[LABELS].copy()
df = df.drop(columns=LABELS + ["time", "command response"])

N_TEST = 80
df_test = df.iloc[:N_TEST].copy()
y_test = y_full.iloc[:N_TEST].copy()
df_test = df_test.fillna(0)
df_test["pressure measurement"] = df_test["pressure measurement"].clip(0, 100)

# Apply scaler to original 19 columns
scaler = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
X_19 = scaler.transform(df_test[ORIG_19].values).astype(np.float32)

# Add the 2 extra cols (is_unusual_fc, is_response) — RAW 0/1
X_21 = np.column_stack([X_19,
                         df_test['is_unusual_fc'].values.astype(np.float32),
                         df_test['is_response'].values.astype(np.float32)])
print(f'X_21 shape (after scaler + 2 raw cols): {X_21.shape}')

# === Build windows and aggregates ===
WINDOW = 16
N_WIN = N_TEST // WINDOW
Xw = X_21[:N_WIN*WINDOW].reshape(N_WIN, WINDOW, -1)
yw = (y_test['binary result'].values[:N_WIN*WINDOW]
      .reshape(N_WIN, WINDOW).max(axis=1)).astype(np.int64)

# 21-dim indexes: 0..16=CSV, 17=time_diff, 18=time_since, 19=is_unusual_fc, 20=is_response
IDX_PRESS = 13
IDX_CRC = 14
IDX_ISRESP = 20  # RAW is_response (0 or 1)
IDX_LENGTH = 2
IDX_ISUNUSUAL = 19

press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
crc_max_w = Xw[:, :, IDX_CRC].max(axis=1)
cmd_count_w = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)  # RAW is_resp
resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)  # RAW is_resp ∈ [0, WINDOW]
balance_w = (cmd_count_w - resp_count_w) / WINDOW  # ∈ [-1, 1]
length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(N_WIN)], dtype=np.float32)
unusual_count_w = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)

aggs = np.column_stack([
    press_mean_w, crc_max_w,
    cmd_count_w, resp_count_w, balance_w,
    length_nunique_w, unusual_count_w,
]).astype(np.float32)
aggs_tiled = np.repeat(aggs[:, np.newaxis, :], WINDOW, axis=1)
Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)
print(f'Xw_full shape: {Xw_full.shape}')

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
X_23 = Xw_full[:, :, KEEP_23]
X_23 = np.clip(X_23, -10.0, 10.0)

ckpt = torch.load(os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"),
                  map_location="cpu", weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt["state_dict"])
model.eval()

X_t = torch.from_numpy(X_23.transpose(0, 2, 1).astype(np.float32))
with torch.no_grad():
    logits = model(X_t).numpy().squeeze()
probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

print()
print('=== GROUND TRUTH (raw is_response, balance ∈ [-1, +1]) ===')
print(f'{"win":>4} {"rows":>9} {"prob":>8} {"label":>5}')
for i in range(N_WIN):
    print(f'{i+1:>4} {i*WINDOW+1:>3}-{i*WINDOW+WINDOW:<5} {probs[i]:>8.4f} {yw[i]:>5}')
print()
print(f'Attack count: {(probs >= 0.5).sum()}/{N_WIN}')

# === Now test with normalized is_response (H743-style) ===
print()
print('=== H743-STYLE (normalized is_response, balance ∈ [1, 3]) ===')
Xw_h743 = Xw.copy().astype(np.float32)
# is_response normalized = (raw - 1) / 1 = raw - 1
Xw_h743[:, :, IDX_ISRESP] = Xw[:, :, IDX_ISRESP] - 1.0

resp_count_h743 = Xw_h743[:, :, IDX_ISRESP].sum(axis=1)  # ∈ [-WINDOW, 0]
cmd_count_h743 = (1 - Xw_h743[:, :, IDX_ISRESP]).sum(axis=1)  # ∈ [WINDOW, 2*WINDOW]
balance_h743 = (cmd_count_h743 - resp_count_h743) / WINDOW  # ∈ [1, 3]
length_nunique_h743 = np.array([len(np.unique(Xw_h743[i, :, IDX_LENGTH])) for i in range(N_WIN)], dtype=np.float32)
unusual_count_h743 = Xw_h743[:, :, IDX_ISUNUSUAL].sum(axis=1)  # unchanged (raw)

aggs_h743 = np.column_stack([
    press_mean_w, crc_max_w,
    cmd_count_h743, resp_count_h743, balance_h743,
    length_nunique_h743, unusual_count_h743,
]).astype(np.float32)
aggs_h743_tiled = np.repeat(aggs_h743[:, np.newaxis, :], WINDOW, axis=1)
Xw_h743_full = np.concatenate([Xw_h743, aggs_h743_tiled], axis=-1)
X_23_h743 = np.clip(Xw_h743_full[:, :, KEEP_23], -10.0, 10.0)

X_t_h743 = torch.from_numpy(X_23_h743.transpose(0, 2, 1).astype(np.float32))
with torch.no_grad():
    logits_h743 = model(X_t_h743).numpy().squeeze()
probs_h743 = 1.0 / (1.0 + np.exp(-np.clip(logits_h743, -50, 50)))

print(f'{"win":>4} {"prob":>8}')
for i in range(N_WIN):
    print(f'{i+1:>4} {probs_h743[i]:>8.4f}')
print(f'H743-style attack count: {(probs_h743 >= 0.5).sum()}/{N_WIN}')

# === Test ONLY with balance_w fix (resp_count in raw [0, WINDOW]) ===
print()
print('=== MIXED (raw is_response for resp_count, but normalized for balance) ===')
# Just fix the resp_count and balance to use raw is_response
aggs_mixed = np.column_stack([
    press_mean_w, crc_max_w,
    cmd_count_w, resp_count_w, balance_w,  # raw-based
    length_nunique_w, unusual_count_w,
]).astype(np.float32)
aggs_mixed_tiled = np.repeat(aggs_mixed[:, np.newaxis, :], WINDOW, axis=1)
Xw_mixed_full = np.concatenate([Xw, aggs_mixed_tiled], axis=-1)
X_23_mixed = np.clip(Xw_mixed_full[:, :, KEEP_23], -10.0, 10.0)

X_t_mixed = torch.from_numpy(X_23_mixed.transpose(0, 2, 1).astype(np.float32))
with torch.no_grad():
    logits_mixed = model(X_t_mixed).numpy().squeeze()
probs_mixed = 1.0 / (1.0 + np.exp(-np.clip(logits_mixed, -50, 50)))

print(f'{"win":>4} {"prob":>8}')
for i in range(N_WIN):
    print(f'{i+1:>4} {probs_mixed[i]:>8.4f}')
print(f'Mixed attack count: {(probs_mixed >= 0.5).sum()}/{N_WIN}')