"""Simulate H743 IDS path on first 40 frames of IanArffDataset.csv.

Uses the actual TCN+SE 23-dim 5-seed ensemble model (ONNX) and the
RobustScaler params from scaler_export.h. Reproduces the bug the user
saw: 7 FPs in 40 normal frames in manual mode.
"""
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import defaultdict
import sys, os
os.chdir(r'C:\work\Claude\Issue')

# === Load scaler (matches scaler_export.h) ===
mu = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               14135.0, 1.426832, 3.116917, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
sigma = np.array([1.0, 13.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  4849.0, 1.521081, 3.241358, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}

# === Load first 40 rows ===
df = pd.read_csv('IanArffDataset.csv', encoding='utf-8', low_memory=False).head(40)
print(f'Loaded {len(df)} rows. All binary={df["binary result"].unique()}')

# === Load 5-seed ensemble ===
sess_list = [ort.InferenceSession(f'model_v4_se_23dim_ch32_s{s}.onnx') for s in [42, 123, 456, 789, 1024]]
input_name = sess_list[0].get_inputs()[0].name
print(f'Loaded {len(sess_list)} ONNX models. input={input_name} shape={sess_list[0].get_inputs()[0].shape}')

# === Simulate H743 path ===
SW_SIZE = 16
THRESHOLD = 0.49
CLIP = 10.0

# Hist ring buffer (max 16 entries): each entry stores 23-dim normalized feature
hist = []
last_ts_ms = 0
addr_func_ts = defaultdict(int)

# Track alarms
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

    # === 1. window stats from hist (BEFORE writing current) ===
    n = len(hist)
    if n > 0:
        # press_mean_w from NORMALIZED pressure (in H743, hist stores (raw-mu)/sigma)
        # But the model expects press_mean_w as the RAW aggregated pressure
        # (per scaler_export.h comment: identity RAW, see feature_extractor.c:69)
        # Actually the model trained on the ORIGINAL normalized pressure mean.
        press_mean_w = sum(h['pressure_norm'] for h in hist) / n
        crc_max_w = max(h['crc_norm'] for h in hist)
        resp_count = sum(1 for h in hist if h['is_resp'])
        req_count = n - resp_count
        seen_lens = []
        for h in hist:
            if h['payload_len'] not in seen_lens:
                seen_lens.append(h['payload_len'])
        length_nunique = float(len(seen_lens))
        unusual_count = sum(1 for h in hist if h['is_unusual'])
    else:
        press_mean_w = 0.0
        crc_max_w = 0.0
        resp_count = 0
        req_count = 0
        length_nunique = 0.0
        unusual_count = 0
    cmd_resp_balance = (resp_count - req_count) / n if n > 0 else 0.0

    # === 2. write current row to hist (with normalized pressure/crc) ===
    pressure_norm = (pressure - mu[11]) / sigma[11]
    crc_norm = (crc_rate - mu[12]) / sigma[12]
    is_unusual = 1 if func in UNUSUAL_FCS else 0

    hist.append({
        'pressure_norm': pressure_norm,
        'crc_norm': crc_norm,
        'is_resp': is_resp,
        'payload_len': length,
        'is_unusual': is_unusual,
        'func': func,
        'slave': slave,
    })
    if len(hist) > SW_SIZE:
        hist.pop(0)

    # === 3. time_diff / time_since_last ===
    time_diff_sec = 0 if last_ts_ms == 0 else float(time_ms - last_ts_ms)
    last_ts_ms = time_ms

    key = (slave * 16 + func) % 256
    time_since_same = 0.0 if addr_func_ts[key] == 0 else (time_ms - addr_func_ts[key]) / 1000.0
    addr_func_ts[key] = time_ms

    # === 4. Build 23-dim feature ===
    feat = np.array([
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
        float(is_unusual),                      # [15] is_unusual_fc
        float(is_resp),                         # [16] is_response
        press_mean_w,                           # [17] press_mean_w
        crc_max_w,                              # [18] crc_max_w
        float(resp_count),                      # [19] resp_count_w
        cmd_resp_balance,                       # [20] cmd_resp_balance_w
        length_nunique,                         # [21] length_nunique_w
        float(unusual_count),                   # [22] unusual_count_w
    ], dtype=np.float32)

    # === 5. RobustScaler normalize + clip ===
    feat_norm = np.clip((feat - mu) / sigma, -CLIP, CLIP).astype(np.float32)

    # === 6. Sliding window ===
    if len(hist) < SW_SIZE:
        results.append((idx+1, binary, 0.0, False, time_diff_sec, ''))
        continue

    # Build window of 16 normalized features
    # In H743, sliding_window broadcasts: when current frame is the 17th,
    # all 16 slots in win[SW_SIZE] are filled with current feat_norm (broadcast)
    # OR fill from hist positions (depends on sliding_window.c implementation)
    # Per memory: "broadcast aggregation" - we'll test both modes

    # Mode A: broadcast (current feat replicated 16 times)
    # ONNX input shape is [batch, 23, 16] (channels-first TCN)
    win_a = np.tile(feat_norm, (SW_SIZE, 1)).T[np.newaxis, :, :]  # (1, 23, 16)

    # Mode B: historical feats from hist (current not yet included in hist)
    # hist stores normalized pressure/crc but NOT all 23 dims
    # So Mode B requires re-storing full feats in hist - H743 doesn't do this
    # Therefore Mode A (broadcast) is what H743 actually does

    # 5-seed ensemble
    probs_a = []
    probs_b = []  # Placeholder for mode B if needed
    for s in sess_list:
        out = s.run(None, {input_name: win_a})[0]
        probs_a.append(float(out.flatten()[0]))
    prob = float(np.mean(probs_a))

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