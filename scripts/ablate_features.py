"""V7: Ablation study — change ONE feature at a time to find which triggers attack."""
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

ORIG_19 = ["address", "function", "length", "setpoint", "gain", "reset rate",
           "deadband", "cycle time", "rate", "system mode", "control scheme",
           "pump", "solenoid", "pressure measurement", "crc rate",
           "Unnamed: 20", "Unnamed: 21", "time_diff", "time_since_last_same_addr_func"]

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

scaler = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
X_19 = scaler.transform(df_test[ORIG_19].values).astype(np.float32)

X_21 = np.column_stack([X_19,
                         df_test['is_unusual_fc'].values.astype(np.float32),
                         df_test['is_response'].values.astype(np.float32)])

WINDOW = 16
N_WIN = N_TEST // WINDOW
Xw = X_21[:N_WIN*WINDOW].reshape(N_WIN, WINDOW, -1)

IDX_PRESS = 13; IDX_CRC = 14; IDX_ISRESP = 20; IDX_LENGTH = 2; IDX_ISUNUSUAL = 19

def build_and_run(Xw_in, label):
    press_mean_w = Xw_in[:, :, IDX_PRESS].mean(axis=1)
    crc_max_w = Xw_in[:, :, IDX_CRC].max(axis=1)
    cmd_count_w = (1 - Xw_in[:, :, IDX_ISRESP]).sum(axis=1)
    resp_count_w = Xw_in[:, :, IDX_ISRESP].sum(axis=1)
    balance_w = (cmd_count_w - resp_count_w) / WINDOW
    length_nunique_w = np.array([len(np.unique(Xw_in[i, :, IDX_LENGTH])) for i in range(N_WIN)], dtype=np.float32)
    unusual_count_w = Xw_in[:, :, IDX_ISUNUSUAL].sum(axis=1)
    aggs = np.column_stack([press_mean_w, crc_max_w, cmd_count_w, resp_count_w, balance_w, length_nunique_w, unusual_count_w]).astype(np.float32)
    aggs_tiled = np.repeat(aggs[:, np.newaxis, :], WINDOW, axis=1)
    Xw_full = np.concatenate([Xw_in, aggs_tiled], axis=-1)
    KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
    X_23 = np.clip(Xw_full[:, :, KEEP_23], -10.0, 10.0)
    X_t = torch.from_numpy(X_23.transpose(0, 2, 1).astype(np.float32))
    with torch.no_grad():
        logits = model(X_t).numpy().squeeze()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return probs

ckpt = torch.load(os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"),
                  map_location="cpu", weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt["state_dict"])
model.eval()

print('=== ABLATION: change one feature at a time, run model ===\n')

# Baseline
p_base = build_and_run(Xw, 'baseline')
print(f'Baseline (first 80 rows as-is): probs = {p_base.round(3)}, attacks = {(p_base >= 0.5).sum()}/{N_WIN}')

# Ablate time_diff to 1.0 (typical SCADA poll interval)
Xw_a = Xw.copy()
# time_diff is at index 17 (after scaler cols 0..18 = ORIG_19, but X_21 = X_19 + is_unusual + is_response)
# X_21 cols: 0..14 = CSV, 15..16=time_diff/time_since, 17=is_unusual, 18=is_response
# Wait, original 19 cols = 15 CSV + time_diff + time_since + 2 Unnamed
# Then add 2 more (is_unusual, is_response) = 21 total
# So time_diff = col 15, time_since = col 16, is_unusual = col 17, is_response = col 18
IDX_TIME_DIFF = 15
IDX_TIME_SINCE = 16

Xw_a[:, :, IDX_TIME_DIFF] = 1.0  # constant 1.0 sec
p = build_and_run(Xw_a, 'time_diff=1.0')
print(f'time_diff=1.0: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate time_diff to 0.5
Xw_a2 = Xw.copy()
Xw_a2[:, :, IDX_TIME_DIFF] = 0.5
p = build_and_run(Xw_a2, 'time_diff=0.5')
print(f'time_diff=0.5: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate time_since to 1.0
Xw_b = Xw.copy()
Xw_b[:, :, IDX_TIME_SINCE] = 1.0
p = build_and_run(Xw_b, 'time_since=1.0')
print(f'time_since=1.0: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate length to 16 (single value)
Xw_c = Xw.copy()
Xw_c[:, :, IDX_LENGTH] = 16.0
p = build_and_run(Xw_c, 'length=16')
print(f'length=16 (single value): probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate is_unusual to 0
Xw_d = Xw.copy()
Xw_d[:, :, IDX_ISUNUSUAL] = 0.0
p = build_and_run(Xw_d, 'is_unusual=0')
print(f'is_unusual=0: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate is_response to 0 (all masters)
Xw_e = Xw.copy()
Xw_e[:, :, IDX_ISRESP] = 0.0
p = build_and_run(Xw_e, 'is_response=0')
print(f'is_response=0: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate is_response to 1 (all slaves)
Xw_f = Xw.copy()
Xw_f[:, :, IDX_ISRESP] = 1.0
p = build_and_run(Xw_f, 'is_response=1')
print(f'is_response=1: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate pressure to 0
Xw_g = Xw.copy()
Xw_g[:, :, IDX_PRESS] = 0.0
p = build_and_run(Xw_g, 'pressure=0')
print(f'pressure=0: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate crc to 0
Xw_h = Xw.copy()
Xw_h[:, :, IDX_CRC] = 0.0
p = build_and_run(Xw_h, 'crc=0')
print(f'crc=0: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate function to 3 only
Xw_i = Xw.copy()
Xw_i[:, :, 1] = 3.0
p = build_and_run(Xw_i, 'function=3')
print(f'function=3 only: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate function to 16 only
Xw_j = Xw.copy()
Xw_j[:, :, 1] = 16.0
p = build_and_run(Xw_j, 'function=16')
print(f'function=16 only: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# Ablate address to 1
Xw_k = Xw.copy()
Xw_k[:, :, 0] = 1.0
p = build_and_run(Xw_k, 'address=1')
print(f'address=1: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')

# All zeros
Xw_zero = np.zeros_like(Xw)
p = build_and_run(Xw_zero, 'all zeros')
print(f'all zeros: probs = {p.round(3)}, attacks = {(p >= 0.5).sum()}/{N_WIN}')