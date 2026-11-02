"""V4: Test hypothesis that nunique_w = 1.0 (gain=0) is the actual FP trigger.
In training, nunique_w comes from actual CSV length column (16, 46, 90) = 3 unique.
H743 reads f[2]=gain (always 0) → nunique_w = 1.0. Model has never seen 1.0 → FP.
"""
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import defaultdict
import os
os.chdir(r'C:\work\Claude\Issue')

mu = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               14135.0, 1.426832, 3.116917, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
sigma = np.array([1.0, 13.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  4849.0, 1.521081, 3.241358, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}

df = pd.read_csv('IanArffDataset.csv', encoding='utf-8', low_memory=False).head(40)
sess_list = [ort.InferenceSession(f'model_v4_se_23dim_ch32_s{s}.onnx') for s in [42, 123, 456, 789, 1024]]
input_name = sess_list[0].get_inputs()[0].name

SW_SIZE = 16
THRESHOLD = 0.49
CLIP = 10.0

# We need to track RAW length separately (not f[2]=gain)
raw_lengths = []

buf = np.zeros((23, SW_SIZE), dtype=np.float32)
write_idx = 0
frame_count = 0
last_ts_ms = 0
addr_func_ts = defaultdict(int)

results_v4 = []  # with CORRECT nunique_w from raw length

for idx in range(len(df)):
    row = df.iloc[idx]
    slave = int(row['address'])
    func = int(row['function'])
    length = int(row['length'])
    pressure = float(row['pressure measurement']) if pd.notna(row['pressure measurement']) else 0.0
    crc_rate = float(row['crc rate'])
    is_resp = int(row['command response'])
    time_ms = int(float(row['time']) * 1000)
    binary = int(row['binary result'])
    raw_lengths.append(length)

    # === window stats with H743 implementation (matching training) ===
    if frame_count >= SW_SIZE:
        press_sum = 0.0
        crc_max_v = -1e30
        resp_signed = 0
        unusual_signed = 0
        for t in range(SW_SIZE):
            oldest = write_idx
            ti = (oldest + t) % SW_SIZE
            press_sum += buf[11, ti]
            if buf[12, ti] > crc_max_v: crc_max_v = buf[12, ti]
            resp_signed += int(buf[16, ti])
            unusual_signed += int(buf[15, ti])
        press_mean_w = press_sum / SW_SIZE
        cmd_count = SW_SIZE - resp_signed
        balance_w = float(cmd_count - resp_signed) / SW_SIZE
        unusual_w = float(-unusual_signed)

        # CORRECT nunique_w from raw length (NOT f[2]=gain)
        recent_lens = raw_lengths[-SW_SIZE:]
        nunique_w = float(len(set(recent_lens)))
    else:
        press_mean_w = crc_max_v = balance_w = 0.0
        nunique_w = 0.0
        unusual_w = 0.0

    time_diff_sec = 0 if last_ts_ms == 0 else float(time_ms - last_ts_ms)
    last_ts_ms = time_ms
    key = (slave * 16 + func) % 256
    time_since_same = 0.0 if addr_func_ts[key] == 0 else (time_ms - addr_func_ts[key]) / 1000.0
    addr_func_ts[key] = time_ms

    feat_raw = np.array([
        slave, func, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        np.clip(pressure, 0, 100), crc_rate, time_diff_sec, time_since_same,
        float(1 if func in UNUSUAL_FCS else 0), float(is_resp),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    feat_norm = np.clip((feat_raw - mu) / sigma, -CLIP, CLIP).astype(np.float32)

    buf[:, write_idx] = feat_norm
    write_idx = (write_idx + 1) % SW_SIZE
    frame_count += 1

    if frame_count < SW_SIZE:
        continue

    oldest = write_idx
    win = np.zeros((23, SW_SIZE), dtype=np.float32)
    for t in range(SW_SIZE):
        ti = (oldest + t) % SW_SIZE
        win[:, t] = buf[:, ti]

    # Recompute aggregates from window
    press_sum2 = 0.0
    crc_max2 = -1e30
    resp_signed2 = 0
    unusual_signed2 = 0
    for t in range(SW_SIZE):
        press_sum2 += win[11, t]
        if win[12, t] > crc_max2: crc_max2 = win[12, t]
        resp_signed2 += int(win[16, t])
        unusual_signed2 += int(win[15, t])
    press_mean_w2 = press_sum2 / SW_SIZE
    cmd_count2 = SW_SIZE - resp_signed2
    balance_w2 = float(cmd_count2 - resp_signed2) / SW_SIZE
    unusual_w2 = float(-unusual_signed2)

    # Broadcast with corrected nunique
    for t in range(SW_SIZE):
        win[17, t] = press_mean_w2
        win[18, t] = crc_max2
        win[19, t] = float(resp_signed2)
        win[20, t] = balance_w2
        win[21, t] = nunique_w  # CORRECTED
        win[22, t] = unusual_w2

    win_input = win[np.newaxis, :, :]
    probs = []
    for s in sess_list:
        out = s.run(None, {input_name: win_input})[0]
        logit = float(out.flatten()[0])
        probs.append(1.0 / (1.0 + np.exp(-logit)))
    prob = float(np.mean(probs))

    results_v4.append((idx+1, binary, prob, nunique_w, len(set(raw_lengths[-SW_SIZE:]))))

print(f'{"#":>3} | {"bin":>3} | {"prob":>6} | {"nunique_fix":>11} | {"true_unique":>11}')
print('-' * 50)
for r in results_v4:
    idx, binary, prob, nuniq_fix, true_uniq = r
    print(f'{idx:>3} | {binary:>3} | {prob:>6.3f} | {nuniq_fix:>11.1f} | {true_uniq:>11}')

print()
fps = sum(1 for r in results_v4 if r[1] == 0 and r[2] >= THRESHOLD)
print(f'FPs (corrected nunique_w): {fps}/25')
print()
print('=== HYPOTHESIS TEST ===')
print('If FPs drop to 0 with nunique_w=3.0, then root cause is sliding_window.c')
print('line 97 using f[2]=gain (always 0) as proxy for length.')