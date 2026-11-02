"""V8: Test model on different parts of the dataset to see if it's globally broken."""
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

df = df.fillna(0)
df["pressure measurement"] = df["pressure measurement"].clip(0, 100)

scaler = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
X_19 = scaler.transform(df[ORIG_19].values).astype(np.float32)
X_21 = np.column_stack([X_19,
                         df['is_unusual_fc'].values.astype(np.float32),
                         df['is_response'].values.astype(np.float32)])

WINDOW = 16
N = len(X_21)
N_WIN = N // WINDOW
Xw = X_21[:N_WIN*WINDOW].reshape(N_WIN, WINDOW, -1)
yw = (y_full['binary result'].values[:N_WIN*WINDOW]
      .reshape(N_WIN, WINDOW).max(axis=1)).astype(np.int64)

IDX_PRESS = 13; IDX_CRC = 14; IDX_ISRESP = 20; IDX_LENGTH = 2; IDX_ISUNUSUAL = 19
press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
crc_max_w = Xw[:, :, IDX_CRC].max(axis=1)
cmd_count_w = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)
resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
balance_w = (cmd_count_w - resp_count_w) / WINDOW
length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(N_WIN)], dtype=np.float32)
unusual_count_w = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)
aggs = np.column_stack([press_mean_w, crc_max_w, cmd_count_w, resp_count_w, balance_w, length_nunique_w, unusual_count_w]).astype(np.float32)
aggs_tiled = np.repeat(aggs[:, np.newaxis, :], WINDOW, axis=1)
Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
X_23 = np.clip(Xw_full[:, :, KEEP_23], -10.0, 10.0)

ckpt = torch.load(os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"),
                  map_location="cpu", weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt["state_dict"])
model.eval()

X_t = torch.from_numpy(X_23.transpose(0, 2, 1).astype(np.float32))
with torch.no_grad():
    logits = model(X_t).numpy().squeeze()
probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

print(f'Total windows: {N_WIN}')
print(f'Attack windows (label): {yw.sum()}/{N_WIN}')
print(f'Predicted attack: {(probs >= 0.5).sum()}/{N_WIN}')
print()
print('=== Sample windows across dataset ===')
sample_indices = list(range(0, min(10, N_WIN))) + list(range(N_WIN//2-5, N_WIN//2+5)) + list(range(N_WIN-10, N_WIN))
print(f'{"win":>6} {"rows":>10} {"prob":>6} {"label":>5}')
for i in sample_indices:
    print(f'{i+1:>6} {i*WINDOW+1:>4}-{i*WINDOW+WINDOW:<5} {probs[i]:>6.3f} {yw[i]:>5}')

# === Confusion matrix ===
from sklearn.metrics import confusion_matrix
preds = (probs >= 0.5).astype(int)
print()
print(f'Confusion matrix on {N_WIN} windows:')
cm = confusion_matrix(yw, preds)
print(cm)
print(f'TP={cm[1,1]} FP={cm[0,1]} FN={cm[1,0]} TN={cm[0,0]}')