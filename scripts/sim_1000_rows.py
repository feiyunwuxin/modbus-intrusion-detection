"""Reproduce the user's 1000-row test with detailed diagnostics."""
import os, sys
sys.path.insert(0, r'C:\work\Claude\Issue')
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import defaultdict

BASE = r'C:\work\Claude\Issue'

mu = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               14135.0, 1.426832, 3.116917, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
sigma = np.array([1.0, 13.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  4849.0, 1.521081, 3.241358, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}

# Load first 1000 rows
df = pd.read_csv(os.path.join(BASE, 'IanArffDataset.csv'), low_memory=False).head(1000)
print(f'Loaded {len(df)} rows')
print(f'Attack distribution: {df["binary result"].value_counts().to_dict()}')
print(f'  Sent (attack) = {int((df["binary result"]==1).sum())}')
print(f'  Normal = {int((df["binary result"]==0).sum())}')

# Load 5-seed ensemble
sess_list = [ort.InferenceSession(os.path.join(BASE, f'model_v4_se_23dim_ch32_s{s}.onnx')) for s in [42, 123, 456, 789, 1024]]
input_name = sess_list[0].get_inputs()[0].name

SW_SIZE = 16
THRESHOLD = 0.49
CLIP = 10.0

# === H743 simulation (correct sliding window) ===
buf = np.zeros((23, SW_SIZE), dtype=np.float32)
write_idx = 0
frame_count = 0
last_ts_ms = 0
addr_func_ts = defaultdict(int)

alarm_count = 0
gap_counter = 0  # for rate limiting simulation
GAP = 4

tp = fp = fn = tn = 0
results = []
prob_sum_attack = 0  # for stats on attack frames
prob_sum_normal = 0

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

    # === window stats ===
    if frame_count >= SW_SIZE:
        press_sum = 0.0
        crc_max_v = -1e30
        resp_signed = 0
        unusual_signed = 0
        seen_lens = []
        for t in range(SW_SIZE):
            oldest = write_idx
            ti = (oldest + t) % SW_SIZE
            press_sum += buf[11, ti]
            if buf[12, ti] > crc_max_v: crc_max_v = buf[12, ti]
            resp_signed += int(buf[16, ti])
            unusual_signed += int(buf[15, ti])
            len_int = int(buf[2, ti] + 0.5)
            if len_int not in seen_lens:
                seen_lens.append(len_int)
        press_mean_w = press_sum / SW_SIZE
        nunique_w = float(len(seen_lens))
        unusual_w = float(-unusual_signed)
        cmd_count = SW_SIZE - resp_signed
        balance_w = float(cmd_count - resp_signed) / SW_SIZE
    else:
        press_mean_w = crc_max_v = 0.0
        resp_signed = 0
        nunique_w = unusual_w = balance_w = 0.0

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

    # Build window + recompute aggregates
    oldest = write_idx
    win = np.zeros((23, SW_SIZE), dtype=np.float32)
    for t in range(SW_SIZE):
        ti = (oldest + t) % SW_SIZE
        win[:, t] = buf[:, ti]

    press_sum2 = 0.0
    crc_max2 = -1e30
    resp_signed2 = 0
    unusual_signed2 = 0
    seen_lens2 = []
    for t in range(SW_SIZE):
        press_sum2 += win[11, t]
        if win[12, t] > crc_max2: crc_max2 = win[12, t]
        resp_signed2 += int(win[16, t])
        unusual_signed2 += int(win[15, t])
        len_int = int(win[2, t] + 0.5)
        if len_int not in seen_lens2:
            seen_lens2.append(len_int)
    press_mean_w2 = press_sum2 / SW_SIZE
    nunique_w2 = float(len(seen_lens2))
    unusual_w2 = float(-unusual_signed2)
    cmd_count2 = SW_SIZE - resp_signed2
    balance_w2 = float(cmd_count2 - resp_signed2) / SW_SIZE

    for t in range(SW_SIZE):
        win[17, t] = press_mean_w2
        win[18, t] = crc_max2
        win[19, t] = float(resp_signed2)
        win[20, t] = balance_w2
        win[21, t] = nunique_w2
        win[22, t] = unusual_w2

    win_input = win[np.newaxis, :, :]
    probs = []
    for s in sess_list:
        out = s.run(None, {input_name: win_input})[0]
        logit = float(out.flatten()[0])
        probs.append(1.0 / (1.0 + np.exp(-logit)))
    prob = float(np.mean(probs))

    is_attack_pred = prob >= THRESHOLD

    # Per-frame KPI tracking (PC sent frame type)
    if binary == 1:
        prob_sum_attack += prob
        if is_attack_pred:
            tp += 1
        else:
            fn += 1
    else:
        prob_sum_normal += prob
        if is_attack_pred:
            fp += 1
        else:
            tn += 1

    # H743 alarm (rate-limited)
    if is_attack_pred:
        alarm_count += 1
        if gap_counter == 0:
            # Print alarm
            gap_counter = GAP

    gap_counter = max(0, gap_counter - 1)  # decrement every frame
    if not is_attack_pred:
        gap_counter = 0

    results.append((idx+1, binary, prob, is_attack_pred))

print()
print('=== PER-FRAME KPI (model output, NO rate-limiting) ===')
print(f'TP = {tp}')
print(f'FP = {fp}')
print(f'FN = {fn}')
print(f'TN = {tn}')
print(f'Total inferences: {tp+fp+fn+tn} (frames 17-1000 = {1000-16})')
print()
sent = 177
normal = 823
recall = tp / sent if sent > 0 else 0
fpr = fp / normal if normal > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f'Recall (TP/Sent) = {recall:.4f} = {recall*100:.1f}%')
print(f'FPR    (FP/Normal) = {fpr:.4f} = {fpr*100:.1f}%')
print(f'Precision (TP/(TP+FP)) = {precision:.4f} = {precision*100:.1f}%')
print(f'F1 = {f1:.4f}')
print()
print(f'H743 detected (model fires, regardless of rate-limit): {alarm_count}')
print(f'Note: H743 visible alarms = rate-limited @ gap={GAP} → ~{alarm_count // GAP}')

# Probability stats
attack_probs = [r[2] for r in results if r[1] == 1]
normal_probs = [r[2] for r in results if r[1] == 0]
print()
print(f'Attack frames mean prob: {np.mean(attack_probs):.3f} (n={len(attack_probs)})')
print(f'Normal frames mean prob: {np.mean(normal_probs):.3f} (n={len(normal_probs)})')