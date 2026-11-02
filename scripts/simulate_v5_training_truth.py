"""V5: Run diag_pytorch_truth.py logic on first 80 rows to get ground truth
probabilities from the actual training pipeline + actual PyTorch model.
This will show whether the model itself is misfiring or my simulation has a bug.
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
df = df.drop(columns=["time", "command response"])

LABELS = ["binary result", "categorized result", "specific result"]
y_full = df[LABELS].copy()
df = df.drop(columns=LABELS)

N_TEST = 80
df_test = df.iloc[:N_TEST].copy()
y_test = y_full.iloc[:N_TEST].copy()
df_test = df_test.fillna(0)
df_test["pressure measurement"] = df_test["pressure measurement"].clip(0, 100)

scaler = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
# Scaler expects 19 features — the pre-aggregated row-level features BEFORE adding window aggs
# df has 21 cols (after time_diff + time_since). Drop the last 2 to match the original scaler input.
# Per scaler.n_features_in_=19, the original 19-dim was: original 17 cols + time_diff + time_since_last
# After model training, is_unusual_fc and is_response were ADDED to make 21.
# But scaler was fit on 19 features (the row-level pre-aggregates, including time_diff and time_since).
# So we drop is_unusual_fc and is_response from the scaler input.
SCALER_COLS = [c for c in df_test.columns if c not in ('is_unusual_fc', 'is_response')]
print(f'Scaler input cols ({len(SCALER_COLS)}): {SCALER_COLS}')
X_test = scaler.transform(df_test[SCALER_COLS].values).astype(np.float32)

# Add is_unusual_fc and is_response back as extra cols (cols 20, 21 of 27-dim)
X_test = np.column_stack([X_test,
                          df_test['is_unusual_fc'].values.astype(np.float32),
                          df_test['is_response'].values.astype(np.float32)])
print(f'X_test shape (after adding back 2 cols): {X_test.shape}')

WINDOW = 16
N_WIN = N_TEST // WINDOW
Xw = X_test[:N_WIN*WINDOW].reshape(N_WIN, WINDOW, -1)
yw = (y_test['binary result'].values[:N_WIN*WINDOW]
      .reshape(N_WIN, WINDOW).max(axis=1)).astype(np.int64)

IDX_PRESS = 13
IDX_CRC = 14
IDX_ISRESP = 20  # was 18, now is_unusual_fc at 18 + is_response at 19, is_response at 20
IDX_LENGTH = 2
IDX_ISUNUSUAL = 19  # was 17

press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
crc_mean_w   = Xw[:, :, IDX_CRC].mean(axis=1)
crc_max_w    = Xw[:, :, IDX_CRC].max(axis=1)
cmd_count_w  = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)
resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
balance_w    = (cmd_count_w - resp_count_w) / WINDOW
length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(N_WIN)], dtype=np.float32)
unusual_count_w  = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)

aggs = np.column_stack([
    press_mean_w, crc_mean_w, crc_max_w,
    cmd_count_w, resp_count_w, balance_w,
    length_nunique_w, unusual_count_w,
]).astype(np.float32)
aggs_tiled = np.repeat(aggs[:, np.newaxis, :], WINDOW, axis=1)
Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)

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

print('=== GROUND TRUTH (training pipeline + s42 PyTorch) ===')
print(f'{"win":>4} {"csv_rows":>10} {"prob":>8} {"label":>6}  {"predicted":>10}')
for i in range(N_WIN):
    label = yw[i]
    pred = 1 if probs[i] >= 0.5 else 0
    print(f'{i+1:>4} {i*WINDOW+1:>4}-{i*WINDOW+WINDOW:<5} {probs[i]:>8.4f} {label:>6}  {"ATTACK" if pred else "normal":>10}')

attack_count = (probs >= 0.5).sum()
print(f'\n Attack count: {attack_count}/{N_WIN} (all 5 windows correspond to first 80 rows = all normal)')
print()
print('=== AGGREGATES PER WINDOW (training pipeline) ===')
print(f'{"win":>4} {"press_mean":>10} {"crc_max":>10} {"resp":>5} {"bal":>6} {"nuniq":>6} {"unusual":>8}')
for i in range(N_WIN):
    print(f'{i+1:>4} {press_mean_w[i]:>10.4f} {crc_max_w[i]:>10.4f} {resp_count_w[i]:>5.0f} {balance_w[i]:>6.2f} {length_nunique_w[i]:>6.1f} {unusual_count_w[i]:>8.1f}')