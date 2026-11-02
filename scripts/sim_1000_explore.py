"""V12: Explore why my ONNX sim gives TP=177/FP=808 but user reports TP=56/FP=185.

Possibilities to test:
1. H743 hybrid_s42 model weights differ from regular s42 ONNX
2. Different threshold (H743 uses 0.49, maybe effective was higher)
3. DMA frame loss (only some frames reach inference)
4. Some features in H743 differ from my sim

Test: vary threshold to see what threshold gives TP=56, FP=185
"""
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

df = pd.read_csv(os.path.join(BASE, 'IanArffDataset.csv'), low_memory=False).head(1000)
sess_list = [ort.InferenceSession(os.path.join(BASE, f'model_v4_se_23dim_ch32_s{s}.onnx')) for s in [42, 123, 456, 789, 1024]]
input_name = sess_list[0].get_inputs()[0].name

SW_SIZE = 16
CLIP = 10.0

buf = np.zeros((23, SW_SIZE), dtype=np.float32)
write_idx = 0
frame_count = 0
last_ts_ms = 0
addr_func_ts = defaultdict(int)

all_probs = []
all_binaries = []

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

    if frame_count >= SW_SIZE:
        press_sum = 0.0; crc_max_v = -1e30; resp_signed = 0; unusual_signed = 0; seen_lens = []
        for t in range(SW_SIZE):
            oldest = write_idx; ti = (oldest + t) % SW_SIZE
            press_sum += buf[11, ti]
            if buf[12, ti] > crc_max_v: crc_max_v = buf[12, ti]
            resp_signed += int(buf[16, ti])
            unusual_signed += int(buf[15, ti])
            len_int = int(buf[2, ti] + 0.5)
            if len_int not in seen_lens: seen_lens.append(len_int)
        press_mean_w = press_sum / SW_SIZE
        nunique_w = float(len(seen_lens))
        unusual_w = float(-unusual_signed)
        cmd_count = SW_SIZE - resp_signed
        balance_w = float(cmd_count - resp_signed) / SW_SIZE
    else:
        press_mean_w = crc_max_v = 0.0; resp_signed = 0; nunique_w = unusual_w = balance_w = 0.0

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

    press_sum2 = 0.0; crc_max2 = -1e30; resp_signed2 = 0; unusual_signed2 = 0; seen_lens2 = []
    for t in range(SW_SIZE):
        press_sum2 += win[11, t]
        if win[12, t] > crc_max2: crc_max2 = win[12, t]
        resp_signed2 += int(win[16, t])
        unusual_signed2 += int(win[15, t])
        len_int = int(win[2, t] + 0.5)
        if len_int not in seen_lens2: seen_lens2.append(len_int)
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

    all_probs.append(prob)
    all_binaries.append(binary)

all_probs = np.array(all_probs)
all_binaries = np.array(all_binaries)
attack_probs = all_probs[all_binaries == 1]
normal_probs = all_probs[all_binaries == 0]

print(f'Total inferences: {len(all_probs)}')
print(f'Attack frame probs: min={attack_probs.min():.4f} max={attack_probs.max():.4f} mean={attack_probs.mean():.4f}')
print(f'Normal frame probs: min={normal_probs.min():.4f} max={normal_probs.max():.4f} mean={normal_probs.mean():.4f}')
print()
print(f'{"threshold":>10} {"TP":>5} {"FP":>5} {"FN":>5} {"TN":>5} {"Recall":>7} {"FPR":>7} {"Precision":>10} {"F1":>7}')
for thr in [0.10, 0.30, 0.49, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]:
    tp = ((all_binaries == 1) & (all_probs >= thr)).sum()
    fp = ((all_binaries == 0) & (all_probs >= thr)).sum()
    fn = ((all_binaries == 1) & (all_probs < thr)).sum()
    tn = ((all_binaries == 0) & (all_probs < thr)).sum()
    sent = (all_binaries == 1).sum()
    normal = (all_binaries == 0).sum()
    recall = tp / sent if sent else 0
    fpr = fp / normal if normal else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2*prec*recall / (prec+recall) if (prec+recall) else 0
    print(f'{thr:>10.2f} {tp:>5} {fp:>5} {fn:>5} {tn:>5} {recall:>7.3f} {fpr:>7.3f} {prec:>10.3f} {f1:>7.3f}')

print()
print('User observed: TP=56 FP=185 Recall=31.6% FPR=22.4%')
print('Looking for threshold that matches user metrics:')
print('  Recall = 56/177 = 0.316 → need threshold that captures 56 of 177 attacks')
print('  FPR = 185/823 = 0.225 → need threshold that fires on 185 of 823 normal')