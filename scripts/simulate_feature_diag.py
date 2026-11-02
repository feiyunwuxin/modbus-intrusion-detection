"""Diagnostic: log feature values per frame to find what triggers FP."""
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
sess = ort.InferenceSession('model_v4_se_23dim_ch32_s42.onnx')
input_name = sess.get_inputs()[0].name

SW_SIZE = 16
THRESHOLD = 0.49
CLIP = 10.0

buf = np.zeros((23, SW_SIZE), dtype=np.float32)
write_idx = 0
frame_count = 0
last_ts_ms = 0
addr_func_ts = defaultdict(int)

print('Frame | fc | len | is_r | crc | press | dt(s) | press_mean_w | crc_max_w | resp_cnt | bal_w | nunique | unusual | prob')
print('-' * 130)

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

    # window stats BEFORE writing
    if frame_count >= SW_SIZE:
        press_sum = 0.0
        crc_max_v = -1e30
        resp_count_signed = 0
        unusual_count_signed = 0
        seen_lens = []
        for t in range(SW_SIZE):
            oldest = write_idx
            ti = (oldest + t) % SW_SIZE
            press_sum += buf[11, ti]
            if buf[12, ti] > crc_max_v: crc_max_v = buf[12, ti]
            resp_count_signed += int(buf[16, ti])
            unusual_count_signed += int(buf[15, ti])
            len_n = buf[2, ti]
            len_int = int(len_n + 0.5)
            if len_int not in seen_lens:
                seen_lens.append(len_int)
        press_mean_w = press_sum / SW_SIZE
        nunique_w = float(len(seen_lens))
        unusual_w = float(-unusual_count_signed)
        cmd_count = SW_SIZE - resp_count_signed
        balance_w = float(cmd_count - resp_count_signed) / SW_SIZE
    else:
        press_mean_w = crc_max_v = 0.0
        resp_count_signed = 0
        balance_w = 0.0
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
        print(f'{idx+1:>3}   | {func:>2} | {length:>3} | {is_resp:>3} | {int(crc_rate):>5} | {pressure:>5.2f} | {time_diff_sec:>5.0f} | WARMING')
        continue

    # Build window
    oldest = write_idx
    win = np.zeros((23, SW_SIZE), dtype=np.float32)
    for t in range(SW_SIZE):
        ti = (oldest + t) % SW_SIZE
        win[:, t] = buf[:, ti]

    # Recompute aggregates
    press_sum = 0.0
    crc_max_v2 = -1e30
    resp_signed = 0
    unusual_signed = 0
    seen_lens = []
    for t in range(SW_SIZE):
        press_sum += win[11, t]
        if win[12, t] > crc_max_v2: crc_max_v2 = win[12, t]
        resp_signed += int(win[16, t])
        unusual_signed += int(win[15, t])
        len_int = int(win[2, t] + 0.5)
        if len_int not in seen_lens:
            seen_lens.append(len_int)
    press_mean_w = press_sum / SW_SIZE
    cmd_count = SW_SIZE - resp_signed
    balance_w = float(cmd_count - resp_signed) / SW_SIZE
    nunique_w = float(len(seen_lens))
    unusual_w = float(-unusual_signed)

    for t in range(SW_SIZE):
        win[17, t] = press_mean_w
        win[18, t] = crc_max_v2
        win[19, t] = float(resp_signed)
        win[20, t] = balance_w
        win[21, t] = nunique_w
        win[22, t] = unusual_w

    # Inference
    win_input = win[np.newaxis, :, :]
    out = sess.run(None, {input_name: win_input})[0]
    logit = float(out.flatten()[0])
    prob = 1.0 / (1.0 + np.exp(-logit))

    print(f'{idx+1:>3}   | {func:>2} | {length:>3} | {is_resp:>3} | {int(crc_rate):>5} | {pressure:>5.2f} | {time_diff_sec:>5.0f} | {press_mean_w:>11.4f} | {crc_max_v2:>9.4f} | {resp_signed:>8} | {balance_w:>5.2f} | {nunique_w:>7.2f} | {unusual_w:>7.2f} | {prob:>6.3f}')