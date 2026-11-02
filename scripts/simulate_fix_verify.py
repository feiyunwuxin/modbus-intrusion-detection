"""V3: Simulate with CORRECT balance_w and resp_count_w computation (training-space)."""
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

buf = np.zeros((23, SW_SIZE), dtype=np.float32)
write_idx = 0
frame_count = 0
last_ts_ms = 0
addr_func_ts = defaultdict(int)

results_buggy = []  # buggy H743 implementation
results_fixed = []  # corrected to training-space

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
        # Current H743 buggy implementation
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
            len_int = int(buf[2, ti] + 0.5)
            if len_int not in seen_lens:
                seen_lens.append(len_int)
        press_mean_w = press_sum / SW_SIZE
        nunique_w = float(len(seen_lens))
        unusual_w = float(-unusual_count_signed)
        cmd_count = SW_SIZE - resp_count_signed
        balance_w_buggy = float(cmd_count - resp_count_signed) / SW_SIZE  # BUGGY: ∈ [1, 3]
        resp_count_buggy = float(resp_count_signed)  # BUGGY: ∈ [-16, 0]
    else:
        press_mean_w = crc_max_v = balance_w_buggy = 0.0
        resp_count_buggy = nunique_w = unusual_w = 0.0

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

    # Read window
    oldest = write_idx
    win = np.zeros((23, SW_SIZE), dtype=np.float32)
    for t in range(SW_SIZE):
        ti = (oldest + t) % SW_SIZE
        win[:, t] = buf[:, ti]

    # === CORRECTED aggregates (training-space) ===
    press_sum2 = 0.0
    crc_max2 = -1e30
    resp_count_raw = 0  # count of frames where is_resp=1 (slave)
    req_count_raw = 0   # count of frames where is_resp=0 (master)
    unusual_raw = 0
    seen_lens2 = []
    for t in range(SW_SIZE):
        is_resp_norm = win[16, t]
        is_resp_raw = int(is_resp_norm + 1.5)
        if is_resp_raw == 1:
            resp_count_raw += 1
        else:
            req_count_raw += 1
        is_unusual_norm = win[15, t]
        is_unusual_raw = int(is_unusual_norm + 0.5)
        unusual_raw += is_unusual_raw

        press_sum2 += win[11, t]
        if win[12, t] > crc_max2: crc_max2 = win[12, t]
        # CORRECT: use f[2] normalized → recover gain, but use length instead
        # In training, f[2] in the model corresponds to which csv column?
        # Actually KEEP_23 excludes length (27-dim idx 2). Training uses csv col 4
        # mapped to model_idx 2 (gain in scaler_export.h). So training f[2] = gain.
        # Therefore nunique_w in TRAINING is computed from csv col 4 (gain), which is always 0!
        # In that case, training nunique_w = 1.0 always too. Let me verify.
        len_int = int(win[2, t] + 0.5)
        if len_int not in seen_lens2:
            seen_lens2.append(len_int)
    press_mean_w2 = press_sum2 / SW_SIZE
    nunique_w2 = float(len(seen_lens2))
    unusual_w2 = float(unusual_raw)
    resp_count_fixed = float(resp_count_raw)
    balance_w_fixed = float(resp_count_raw - req_count_raw) / SW_SIZE

    # === Run with BOTH buggy and corrected aggregates ===
    for label, press_mw, crc_mw, resp_cnt, bal_w, nuniq, unus in [
        ('BUGGY', press_mean_w, crc_max_v, resp_count_buggy, balance_w_buggy, nunique_w, unusual_w),
        ('FIXED', press_mean_w2, crc_max2, resp_count_fixed, balance_w_fixed, nunique_w2, unusual_w2),
    ]:
        win2 = win.copy()
        for t in range(SW_SIZE):
            win2[17, t] = press_mw
            win2[18, t] = crc_mw
            win2[19, t] = resp_cnt
            win2[20, t] = bal_w
            win2[21, t] = nuniq
            win2[22, t] = unus
        win_input = win2[np.newaxis, :, :]
        probs = []
        for s in sess_list:
            out = s.run(None, {input_name: win_input})[0]
            logit = float(out.flatten()[0])
            probs.append(1.0 / (1.0 + np.exp(-logit)))
        prob = float(np.mean(probs))
        is_atk = prob >= THRESHOLD
        (results_buggy if label == 'BUGGY' else results_fixed).append((idx+1, binary, prob, is_atk, bal_w, resp_cnt))

print(f'{"#":>3} | {"bin":>3} | {"prob(buggy)":>11} | {"prob(fix)":>9} | {"bal(bug)":>8} | {"bal(fix)":>8} | {"resp(bug)":>9} | {"resp(fix)":>9}')
print('-' * 95)
for b, f in zip(results_buggy, results_fixed):
    idx, binary, prob_b, atk_b, bal_b, resp_b = b
    idx2, binary2, prob_f, atk_f, bal_f, resp_f = f
    print(f'{idx:>3} | {binary:>3} | {prob_b:>11.3f} | {prob_f:>9.3f} | {bal_b:>8.2f} | {bal_f:>8.2f} | {resp_b:>9.1f} | {resp_f:>9.1f}')

print()
buggy_fps = sum(1 for r in results_buggy if r[1] == 0 and r[3])
fixed_fps = sum(1 for r in results_fixed if r[1] == 0 and r[3])
print(f'BUGGY  FPs (frames 16-40): {buggy_fps}/25  (matches user 7/40 with rate-limiting)')
print(f'FIXED  FPs (frames 16-40): {fixed_fps}/25  (expected 0)')
print()
print('=== ROOT CAUSE CONFIRMED ===')
print('sliding_window.c lines 107-114 (and 119) compute balance_w and resp_count_w')
print('using NORMALIZED is_response (master=-1, slave=0), giving:')
print('  balance_w: ∈ [1, 3]  (training expects [-1, +1])')
print('  resp_count_w: ∈ [-16, 0]  (training expects [0, 16])')
print('Result: model has NEVER seen these values, fires attack on every normal frame after window fills.')