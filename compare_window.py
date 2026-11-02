"""Compare my sim first window to saved test first window."""
import numpy as np
import pandas as pd
import struct
from collections import deque
import re

# === modbus CRC + build/parse (inline) ===
def modbus_crc16(payload):
    crc = 0xFFFF
    for b in payload:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 0x0001) else (crc >> 1)
    return crc

def build_rtu_frame(row):
    is_master = int(row.get('command', 0)) == 1
    slave = max(1, min(247, int(row.get('address', 1))))
    func = int(row.get('function', 3))
    length = max(0, min(255, int(row.get('length', 0))))
    if func in (1, 2, 3, 4):
        if is_master:
            count = max(length * 8, 1) if func in (1,2) else max(length // 2, 1)
            pdu = struct.pack('>HH', 0, count)
        else:
            n_regs = max(length // 2, 0)
            byte_count = min(2 * n_regs, 255)
            pdu = bytes([byte_count]) + bytes(byte_count)
    elif func in (5, 6):
        pdu = struct.pack('>HH', 0, 0)
    elif func == 15:
        if is_master:
            n_coils = max(length * 8, 1)
            n_bytes = max((n_coils + 7) // 8, 1)
            pdu = struct.pack('>HHB', 0, n_coils, n_bytes) + bytes(n_bytes)
        else:
            pdu = struct.pack('>HH', 0, max(length * 8, 1))
    elif func == 16:
        if is_master:
            n_regs = max((length - 5) // 2, 0) if length >= 5 else 0
            byte_count = 2 * n_regs
            pdu = struct.pack('>HHB', 0, n_regs, byte_count) + bytes(byte_count)
        else:
            n_regs = max((length - 5) // 2, 0) if length >= 5 else 0
            pdu = struct.pack('>HH', 0, n_regs)
    else:
        pdu = bytes(length)
    payload = bytes([slave, func]) + pdu
    crc = modbus_crc16(payload)
    return payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])

def parse_rtu(buf):
    if len(buf) < 6 or len(buf) > 256: return None
    calc_crc = modbus_crc16(buf[:-2])
    rx_crc = (buf[-1] << 8) | buf[-2]
    return {'slave': buf[0], 'func': buf[1], 'total_len': len(buf),
            'payload_len': len(buf) - 4, 'crc_valid': (calc_crc == rx_crc)}

def classify_response(f):
    if f['func'] >= 0x80: return 0
    if f['func'] in (1, 2, 3, 4): return 0 if f['payload_len'] > 4 else 1
    if f['func'] in (15, 16): return 0 if f['payload_len'] <= 4 else 1
    return 1

# Read scaler
scaler_h_raw = open('KeilH743/H743/Core/Inc/scaler_export.h').read()
scaler_h = re.sub(r'/\*.*?\*/', '', scaler_h_raw, flags=re.DOTALL)
def extract_floats(name):
    m = re.search(rf'static const float feat_{name}\[\s*FEAT_DIM\s*\]\s*=\s*\{{([^}}]+)\}}', scaler_h)
    vals = []
    for tok in m.group(1).split(','):
        tok = tok.strip().rstrip('f').strip()
        if tok:
            try: vals.append(float(tok))
            except ValueError: pass
    return np.array(vals, dtype=np.float32)
mu = extract_floats('mu')
sigma = extract_floats('sigma')
unusual_m = re.search(r'static const uint8_t feat_unusual_fc\[\s*FEAT_UNUSUAL_FC_COUNT\s*\]\s*=\s*\{([^}]+)\}', scaler_h)
unusual_fc = set(int(x.strip()) for x in unusual_m.group(1).split(',') if x.strip())

CLIP = 10.0
HIST_SIZE = 16

# Load CSV
df = pd.read_csv('IanArffDataset.csv').replace('?', np.nan)
df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce').fillna(0)
df['gain'] = pd.to_numeric(df['gain'], errors='coerce').fillna(0)
df['crc'] = pd.to_numeric(df['crc'], errors='coerce').fillna(0)
df['time'] = pd.to_numeric(df['time'], errors='coerce').fillna(0)
for col in ['setpoint','reset','deadband','cycle','rate','system','control','pump','solenoid']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
df['function'] = pd.to_numeric(df['function'], errors='coerce').fillna(3).astype(int)
df['length'] = pd.to_numeric(df['length'], errors='coerce').fillna(0).astype(int)
df['command'] = pd.to_numeric(df['command'], errors='coerce').fillna(0).astype(int)
df['address'] = pd.to_numeric(df['address'], errors='coerce').fillna(1).astype(int)
# Use TEST split (matches saved test data)
N = len(df)
n_train = int(N * 0.70); n_val = int(N * 0.10)
df_test = df.iloc[n_train + n_val:].reset_index(drop=True)
n = df_test[df_test['binary']==0].head(200).reset_index(drop=True)

hist_buf = deque(maxlen=16)
window = deque(maxlen=16)
prev_ts = None

for idx in range(16):
    row = n.iloc[idx]
    frame = build_rtu_frame(row)
    f = parse_rtu(frame)
    is_resp = classify_response(f)
    gain = float(row.get('gain', 0) or 0)
    pressure = float(row.get('pressure', 0) or 0)
    crc = float(row.get('crc', 0) or 0)
    reset_rate = float(row.get('reset', 0) or 0)
    deadband = float(row.get('deadband', 0) or 0)
    cycle_time = float(row.get('cycle', 0) or 0)
    system_mode = float(row.get('system', 0) or 0)
    control_scheme = float(row.get('control', 0) or 0)
    pump = float(row.get('pump', 0) or 0)
    solenoid = float(row.get('solenoid', 0) or 0)
    cur_ts = float(row.get('time', 0) or 0)
    if prev_ts is None or cur_ts == 0:
        time_diff_sec = 0.1
    else:
        time_diff_sec = abs(cur_ts - prev_ts)
        if time_diff_sec > 10.0: time_diff_sec = 10.0
        if cur_ts > 0: prev_ts = cur_ts

    # === extract_features (inline) ===
    f_arr = np.zeros(23, dtype=np.float32)
    f_arr[0] = f['slave']
    f_arr[1] = f['func']
    f_arr[2] = gain
    f_arr[3] = reset_rate
    f_arr[4] = deadband
    f_arr[5] = cycle_time
    f_arr[6] = 0.0
    f_arr[7] = system_mode
    f_arr[8] = control_scheme
    f_arr[9] = pump
    f_arr[10] = solenoid
    f_arr[11] = min(100.0, max(0.0, pressure))
    f_arr[12] = crc
    f_arr[13] = time_diff_sec
    f_arr[14] = time_diff_sec
    f_arr[15] = 1.0 if f['func'] in unusual_fc else 0.0
    f_arr[16] = 1.0 if is_resp else 0.0

    nh = len(hist_buf)
    if nh > 0:
        press_mean_w = np.mean([h['pressure'] for h in hist_buf])
        crc_max = max([h['crc_rate'] for h in hist_buf])
        # Use the RAW 0/1 flag (is_response), NOT is_response_scaled —
        # the scaled version is a z-score and summing 16 z-scores gives a
        # meaningless value, not a count of response frames in the
        # window. Same fix on cmd_count and unusual_w below.
        resp_count = sum(h['is_response'] for h in hist_buf)
        cmd_count  = sum(1 - h['is_response'] for h in hist_buf)
        balance = (cmd_count - resp_count) / nh
        seen_lens = set(h['payload_len'] for h in hist_buf)
        nunique = len(seen_lens)
        unusual_w = sum(h['is_unusual_fc'] for h in hist_buf)
    else:
        press_mean_w = crc_max = balance = nunique = unusual_w = 0
        resp_count = 0

    f_arr[17] = press_mean_w
    f_arr[18] = crc_max
    f_arr[19] = float(resp_count)
    f_arr[20] = balance
    f_arr[21] = float(nunique)
    f_arr[22] = unusual_w

    out = (f_arr - mu) / sigma
    out = np.clip(out, -CLIP, CLIP)

    raw_for_hist = dict(slave=f['slave'], func=f['func'], pressure=pressure, crc_rate=crc,
                        is_response=is_resp, payload_len=f['payload_len'])
    raw_for_hist['pressure'] = (pressure - mu[11]) / sigma[11]
    raw_for_hist['crc_rate'] = (crc - mu[12]) / sigma[12]
    raw_for_hist['is_response_scaled'] = (is_resp - mu[16]) / sigma[16]
    raw_for_hist['is_unusual_fc'] = 1.0 if f['func'] in unusual_fc else 0.0
    raw_for_hist['is_unusual_fc_scaled'] = (1.0 if f['func'] in unusual_fc else 0.0 - mu[15]) / sigma[15]
    hist_buf.append(raw_for_hist)
    window.append(out)

my_win = np.array(window, dtype=np.float32)
# Apply broadcast fix for window aggregates
last_aggs = my_win[15, 17:23].copy()
my_win[:, 17:23] = last_aggs
np.save('my_sim_window1.npy', my_win)

print('=== My sim first window (16 timesteps × 23 dims, with broadcast aggs) ===')
for t in range(16):
    print(f't={t:2d}: {my_win[t].round(3)}')

X = np.load('X_test_binary_v2_scada_window16.npy')
y = np.load('y_test_binary_v2_scada_window16.npy')
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
X23 = X[:, :, KEEP_23]
first_normal = X23[y==0][0]
print('\n=== Saved test first NORMAL window ===')
for t in range(16):
    print(f't={t:2d}: {first_normal[t].round(3)}')

print('\n=== Per-feature max_abs_diff (my vs saved) ===')
for i in range(23):
    diff = np.max(np.abs(my_win[:, i] - first_normal[:, i]))
    print(f'  f[{i:2d}]: max_abs_diff={diff:.3f}')