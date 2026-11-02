"""Simulate H743 IDS path on first 40 frames of IanArffDataset.csv.

V2: Faithful to sliding_window.c — stores 16 historical frames, broadcasts
only f[17..22] aggregates. This is the model architecture that matches
the training data preprocessing.
"""
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import defaultdict
import os
os.chdir(r'C:\work\Claude\Issue')

# === Load scaler (matches scaler_export.h) ===
mu = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               14135.0, 1.426832, 3.116917, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
sigma = np.array([1.0, 13.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  4849.0, 1.521081, 3.241358, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}

# === Load first 40 rows ===
df = pd.read_csv('IanArffDataset.csv', encoding='utf-8', low_memory=False).head(40)
print(f'Loaded {len(df)} rows. binary distribution: {df["binary result"].value_counts().to_dict()}')

# === Load 5-seed ensemble ===
sess_list = [ort.InferenceSession(f'model_v4_se_23dim_ch32_s{s}.onnx') for s in [42, 123, 456, 789, 1024]]
input_name = sess_list[0].get_inputs()[0].name
print(f'Loaded {len(sess_list)} ONNX models. input shape={sess_list[0].get_inputs()[0].shape}')

# === Simulate H743 path ===
SW_SIZE = 16
THRESHOLD = 0.49
CLIP = 10.0

# H743 ring buffer of 16 normalized features (per sliding_window.c)
buf = np.zeros((23, SW_SIZE), dtype=np.float32)
write_idx = 0
frame_count = 0

last_ts_ms = 0
addr_func_ts = defaultdict(int)

alarm_count = 0
fps = []
results = []

for idx in range(len(df)):
    row = df.iloc[idx]
    slave = int(row['address'])
    func = int(row['function'])
    length = int(row['length'])
    pressure = float(row['pressure measurement']) if pd.notna(row['pressure measurement']) else 0.0
    crc_rate = float(row['crc rate'])
    is_resp = int(row['command response'])
    time_sec = float(row['time'])
    time_ms = int(time_sec * 1000)
    binary = int(row['binary result'])

    # === Window stats from current buf (BEFORE writing new row) ===
    # Mirrors sliding_window.c lines 78-114
    if frame_count >= SW_SIZE:
        # Read from buf in time order: oldest is at write_idx (line 61)
        press_sum = 0.0
        crc_max = -1e30
        resp_count_signed = 0
        unusual_count_signed = 0
        seen_lens = []
        for t in range(SW_SIZE):
            oldest = write_idx
            ti = (oldest + t) % SW_SIZE
            press_n = buf[11, ti]
            crc_n = buf[12, ti]
            is_resp_n = buf[16, ti]
            is_unusual_n = buf[15, ti]
            press_sum += press_n
            if crc_n > crc_max: crc_max = crc_n
            resp_count_signed += int(is_resp_n)
            unusual_count_signed += int(is_unusual_n)
            len_n = buf[2, ti]
            len_int = int(len_n + 0.5)
            if len_int not in seen_lens:
                seen_lens.append(len_int)
        press_mean_w = press_sum / SW_SIZE
        nunique_w = float(len(seen_lens))
        unusual_w = float(-unusual_count_signed)
    else:
        # Cold start: buf is zeros, stats from zeros
        press_mean_w = 0.0
        crc_max = 0.0
        resp_count_signed = 0
        unusual_w = 0.0
        nunique_w = 0.0

    # === Compute time_diff ===
    time_diff_sec = 0 if last_ts_ms == 0 else float(time_ms - last_ts_ms)
    last_ts_ms = time_ms
    key = (slave * 16 + func) % 256
    time_since_same = 0.0 if addr_func_ts[key] == 0 else (time_ms - addr_func_ts[key]) / 1000.0
    addr_func_ts[key] = time_ms

    # === Build 23-dim feature (raw, before normalize) ===
    feat_raw = np.array([
        slave,                                  # [ 0] address
        func,                                   # [ 1] function
        0.0,                                    # [ 2] gain (CSV col 4)
        0.0,                                    # [ 3] reset_rate (CSV col 5)
        0.0,                                    # [ 4] deadband (CSV col 6)
        0.0,                                    # [ 5] cycle_time (CSV col 7)
        0.0,                                    # [ 6] rate (CSV col 8)
        0.0,                                    # [ 7] system_mode (CSV col 9)
        0.0,                                    # [ 8] control_scheme (CSV col 10)
        0.0,                                    # [ 9] pump (CSV col 11)
        0.0,                                    # [10] solenoid (CSV col 12)
        np.clip(pressure, 0, 100),              # [11] pressure (clipped)
        crc_rate,                               # [12] crc_rate
        time_diff_sec,                          # [13] time_diff
        time_since_same,                        # [14] time_since_last_same
        0.0,                                    # [15] is_unusual_fc (will be updated)
        float(is_resp),                         # [16] is_response
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,           # [17..22] aggregates
    ], dtype=np.float32)

    is_unusual = 1 if func in UNUSUAL_FCS else 0
    feat_raw[15] = float(is_unusual)

    # === RobustScaler normalize + clip ===
    feat_norm = np.clip((feat_raw - mu) / sigma, -CLIP, CLIP).astype(np.float32)

    # === Write to sliding window buf (per sliding_window.c line 41-44) ===
    buf[:, write_idx] = feat_norm
    write_idx = (write_idx + 1) % SW_SIZE
    frame_count += 1

    if frame_count < SW_SIZE:
        results.append((idx+1, binary, 0.0, False, time_diff_sec, ''))
        continue

    # === Build window in time order (oldest to newest) ===
    # sliding_window.c line 71-75 reads in time order from win_at()
    oldest = write_idx  # next slot to overwrite = oldest
    win = np.zeros((23, SW_SIZE), dtype=np.float32)
    for t in range(SW_SIZE):
        ti = (oldest + t) % SW_SIZE
        win[:, t] = buf[:, ti]

    # === Recompute aggregates from 16 frames (sliding_window.c lines 78-114) ===
    press_sum = 0.0
    crc_max_v = -1e30
    resp_count_signed = 0
    unusual_count_signed = 0
    seen_lens = []
    for t in range(SW_SIZE):
        press_n = win[11, t]
        crc_n = win[12, t]
        is_resp_n = win[16, t]
        is_unusual_n = win[15, t]
        press_sum += press_n
        if crc_n > crc_max_v: crc_max_v = crc_n
        resp_count_signed += int(is_resp_n)
        unusual_count_signed += int(is_unusual_n)
        len_n = win[2, t]
        len_int = int(len_n + 0.5)
        if len_int not in seen_lens:
            seen_lens.append(len_int)
    press_mean_w = press_sum / SW_SIZE
    cmd_count = SW_SIZE - resp_count_signed
    balance_w = float(cmd_count - resp_count_signed) / SW_SIZE
    nunique_w = float(len(seen_lens))
    unusual_w = float(-unusual_count_signed)

    # === Broadcast aggregates to all 16 rows (sliding_window.c lines 117-124) ===
    for t in range(SW_SIZE):
        win[17, t] = press_mean_w
        win[18, t] = crc_max_v
        win[19, t] = float(resp_count_signed)
        win[20, t] = balance_w
        win[21, t] = nunique_w
        win[22, t] = unusual_w

    # === 5-seed ensemble inference (model output = logit, apply sigmoid) ===
    win_input = win[np.newaxis, :, :]  # (1, 23, 16)
    probs = []
    for s in sess_list:
        out = s.run(None, {input_name: win_input})[0]
        logit = float(out.flatten()[0])
        prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        probs.append(prob)
    prob = float(np.mean(probs))

    is_attack = prob >= THRESHOLD
    fp_marker = ''
    if is_attack:
        alarm_count += 1
        if binary == 0:
            fp_marker = 'FP!'
            fps.append((idx+1, prob))
        else:
            fp_marker = 'TP!'
    results.append((idx+1, binary, prob, is_attack, time_diff_sec, fp_marker))

print()
print(f'{"#":>3} | {"bin":>3} | {"dt":>4} | {"prob":>6} | {"alarm":>5} | note')
print('-' * 50)
for r in results:
    idx, binary, prob, is_atk, dt, marker = r
    print(f'{idx:>3} | {binary:>3} | {dt:>4.0f} | {prob:>6.3f} | {alarm_count if is_atk else "":>5} | {marker}')

print()
print('=== SUMMARY ===')
print(f'Total: {len(results)}')
print(f'Normal frames: {sum(1 for r in results if r[1]==0)}')
print(f'Alarms: {alarm_count}')
print(f'FPs: {len(fps)} (expected 0 for 40 all-normal frames)')
if fps:
    print(f'FP frames: {fps}')