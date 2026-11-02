"""End-to-end verification: wrapper predictions vs X_test direct on 562-row prefix.

Goal: confirm the precision fix to extract_features_19 makes the wrapper
agree with X_test direct predictions on the same rows (both ~35-50%
attack rate). Before the fix, wrapper predicted 100% attack (562/562)
because pressure truncation gave it press_mean_w=0 for all windows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from ids.inference import (
    extract_features_19,
    keep_23_per_frame_indices,
    keep_23_aggregate_indices,
)
from ids.scaler_23dim import Scaler23
from ids.model_loader import _Torch3DStateDictWrapper23

CSV = Path(r"D:\workspace\claude\Issue\Issue\IanArffDataset_RAW.csv")
PT  = Path(r"D:\workspace\claude\Issue\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s1024.pt")
WINDOW = 16
N_ROWS = 562  # user's exact test length

# Load CSV
df = pd.read_csv(CSV).sort_values("time").reset_index(drop=True)
df["time_diff"] = df["time"].diff().fillna(0)
df["_k"] = df["address"].astype(int) * 1000 + df["function"].astype(int)
last_seen = {}
tsl = np.zeros(len(df), dtype=np.float32)
for i in range(len(df)):
    k = df["_k"].iloc[i]
    if k in last_seen:
        tsl[i] = float(df["time"].iloc[i]) - last_seen[k]
    last_seen[k] = float(df["time"].iloc[i])
df["time_since_last_same_addr_func"] = tsl
df["is_unusual_fc"] = df["function"].apply(
    lambda f: 1 if f in {136, 171, 139, 133, 137, 138, 140} else 0
).astype(np.float32)
df["is_response"] = df["command response"].astype(np.float32)

UNUSUAL = {136, 171, 139, 133, 137, 138, 140}

records = []
for _, row in df.head(N_ROWS).iterrows():
    records.append({
        "address": row["address"],
        "function": row["function"],
        "length": row["length"],
        "setpoint": row["setpoint"],
        "gain": row["gain"],
        "reset": row["reset rate"],
        "deadband": row["deadband"],
        "cycle": row["cycle time"],
        "rate": row["rate"],
        "system": row["system mode"],
        "control": row["control scheme"],
        "pump": row["pump"],
        "solenoid": row["solenoid"],
        "pressure": row["pressure measurement"],
        "crc": row["crc rate"],
        "command": row["command response"],
        "time": row["time"],
        "binary": row["binary result"],
    })

# Load model
ckpt = torch.load(PT, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
# Probe in_channels from first conv weight
probe = "tcn.0.conv1.weight"
in_ch = sd[probe].shape[1]  # e.g. 23
print(f"Conv1d in_channels = {in_ch} (expect 23)")
# Load via wrapper
wrapper = _Torch3DStateDictWrapper23(sd, WINDOW)

scaler = Scaler23()
wrapper_scaler = Scaler23()

# Also test against X_test directly
X_test = np.load(r"D:\workspace\claude\Issue\Issue\X_test_binary_v2_scada_window16.npy").astype(np.float32)
y_test = np.load(r"D:\workspace\claude\Issue\Issue\y_test_binary_v2_scada_window16.npy")
print(f"X_test shape={X_test.shape}, pos rate={y_test.mean():.3f}")

# Use wrapper.infer() directly (the production path)
wrapper_preds = []
wrapper_probs = []
truths = []
for rec in records:
    out = wrapper.infer(rec)
    if out is None:
        continue
    label, prob = out
    wrapper_preds.append(label)
    wrapper_probs.append(prob)
    truths.append(int(rec["binary"]))

wrapper_preds = np.array(wrapper_preds)
wrapper_probs = np.array(wrapper_probs)
truths = np.array(truths)
print()
print(f"Wrapper predictions on first {N_ROWS} records (post-fix):")
print(f"  total:   {len(wrapper_preds)}")
print(f"  attack:  {int(wrapper_preds.sum())}")
print(f"  normal:  {int((wrapper_preds == 0).sum())}")
print(f"  truth attack: {int(truths.sum())}, normal: {int((truths == 0).sum())}")
if len(truths) > 0:
    print(f"  accuracy: {(wrapper_preds == truths).mean():.3f}")
    print(f"  precision: {precision_score(truths, wrapper_preds, zero_division=0):.3f}")
    print(f"  recall:    {recall_score(truths, wrapper_preds, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(truths, wrapper_preds, zero_division=0):.3f}")

# Also check first 16 records' press_mean_w to confirm non-zero
print()
print(f"Pressure sample values from first 5 records:")
for i in range(5):
    p = records[i]["pressure"]
    p = float(p) if isinstance(p, (int, float)) else p
    print(f"  rec[{i}]: raw={p}")