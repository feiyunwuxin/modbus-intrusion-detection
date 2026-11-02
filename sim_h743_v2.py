"""
Simulator that exactly mirrors the new H743 sliding_window.c + feature_extractor.c flow.
Tests: 1) broadcast aggregates from full 16-frame window, 2) synth_timing produces correct f[13]
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import deque

# === modbus CRC + build/parse (inline) ===
def modbus_crc16(payload):
    crc = 0xFFFF
    for b in payload:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 0x0001) else (crc >> 1)
    return crc

def build_rtu_frame(row):
    import struct
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

# === Load model ===
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
    def forward(self, x):
        s = x.mean(dim=-1); s = F.relu(self.fc1(s)); s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class TCNBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1, se_reduction=8):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=se_reduction)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x))); x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.drop(x); x = self.se(x)
        return F.relu(x + residual)

class TCNClassifierSE(nn.Module):
    def __init__(self, in_ch=23, channels=32, n_blocks=3, kernel_size=3, dilations=(1,2,4), dropout=0.1):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.tcn(x); x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x)); x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)

state = torch.load('model_v4_se_23dim_b64_ch32_do01_window16_s42.pt', map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, n_blocks=3, kernel_size=3, dilations=(1,2,4), dropout=0.1)
model.load_state_dict(state['state_dict'])
model.eval()

# === Simulate H743 pipeline exactly ===
HIST_SIZE = 16
hist_buf = deque(maxlen=HIST_SIZE)
sliding_buf = np.zeros((HIST_SIZE, 23), dtype=np.float32)  # 16×23 normalized
write_idx = 0
frame_count = 0

# pre-allocate new sliding_window.c logic
import struct

def extract_features_h743(row, hist):
    """Mirror H743 feature_extractor_update exactly."""
    frame = build_rtu_frame(row)
    f = parse_rtu(frame)
    is_resp = classify_response(f)
    is_master_now = int(row.get('command', 0)) == 1

    gain = float(row.get('gain', 0) or 0)
    pressure = float(row.get('pressure', 0) or 0)
    crc = float(row.get('crc', 0) or 0)
    reset_rate = float(row.get('reset', 0) or 0)
    deadband = float(row.get('deadband', 0) or 0)
    cycle_time = float(row.get('cycle', 0) or 0)
    rate = float(row.get('rate', 0) or 0)
    system_mode = float(row.get('system', 0) or 0)
    control_scheme = float(row.get('control', 0) or 0)
    pump = float(row.get('pump', 0) or 0)
    solenoid = float(row.get('solenoid', 0) or 0)

    # Synth timing
    if len(hist) == 0:
        time_diff_sec = 1.5 if is_master_now else 0.1
    elif is_master_now:
        time_diff_sec = 1.5 + 0.075 * (2 * (frame_count % 7) / 7 - 1)
    else:
        time_diff_sec = 0.1 + 0.005 * (2 * (frame_count % 7) / 7 - 1)

    # Clip pressure
    pressure_clipped = min(100.0, max(0.0, pressure))

    # f[0..16] in RAW values
    f_arr = np.zeros(23, dtype=np.float32)
    f_arr[0] = f['slave']
    f_arr[1] = f['func']
    f_arr[2] = gain
    f_arr[3] = reset_rate
    f_arr[4] = deadband
    f_arr[5] = cycle_time
    f_arr[6] = rate
    f_arr[7] = system_mode
    f_arr[8] = control_scheme
    f_arr[9] = pump
    f_arr[10] = solenoid
    f_arr[11] = pressure_clipped
    f_arr[12] = crc
    f_arr[13] = time_diff_sec
    f_arr[14] = time_diff_sec
    f_arr[15] = 1.0 if f['func'] in unusual_fc else 0.0
    f_arr[16] = 1.0 if is_resp else 0.0

    # Compute aggregates from OLD hist (incremental, like H743)
    nh = len(hist)
    if nh > 0:
        press_mean_w = np.mean([h['pressure_n'] for h in hist])  # already normalized
        crc_max = max([h['crc_n'] for h in hist])
        resp_count = sum(h['is_resp_n'] for h in hist)  # 0 for master, -1 for slave
        seen_lens = set(h['length'] for h in hist)
        unusual_w = sum(h['is_unusual_n'] for h in hist)
    else:
        press_mean_w = crc_max = 0.0
        resp_count = 0
        seen_lens = set()
        unusual_w = 0
    cmd_count = nh - resp_count
    balance_w = (cmd_count - resp_count) / nh if nh > 0 else 0

    f_arr[17] = press_mean_w
    f_arr[18] = crc_max
    f_arr[19] = float(resp_count)
    f_arr[20] = balance_w
    f_arr[21] = float(len(seen_lens))
    f_arr[22] = unusual_w

    # Normalize
    out = (f_arr - mu) / sigma
    out = np.clip(out, -10.0, 10.0)
    return out, f, is_resp

def h743_push_and_infer(row):
    """Mirror H743 modbus_ids_on_frame: extract → push to sliding → if ready, get & broadcast & infer."""
    global write_idx, frame_count

    feats, f, is_resp = extract_features_h743(row, hist_buf)

    # Build hist entry with normalized values
    hist_entry = {
        'pressure_n': (float(row.get('pressure', 0) or 0) - mu[11]) / sigma[11],
        'crc_n': (float(row.get('crc', 0) or 0) - mu[12]) / sigma[12],
        'is_resp_n': (is_resp - mu[16]) / sigma[16],
        'is_unusual_n': ((1.0 if f['func'] in unusual_fc else 0.0) - mu[15]) / sigma[15],
        'length': f['payload_len'],
    }
    hist_buf.append(hist_entry)

    # Push to sliding buffer
    sliding_buf[write_idx, :] = feats
    write_idx = (write_idx + 1) % HIST_SIZE
    frame_count += 1

    if frame_count < HIST_SIZE:
        return None, frame_count

    # === sliding_window_get with broadcast (new H743 logic) ===
    oldest = write_idx  # next to overwrite = oldest
    out = np.zeros((HIST_SIZE, 23), dtype=np.float32)
    for t in range(HIST_SIZE):
        out[t, :] = sliding_buf[(oldest + t) % HIST_SIZE, :]

    # Recompute aggregates from 16 frames (in normalized space)
    press_sum = 0.0
    crc_max = -1e30
    resp_count_signed = 0
    seen_lens = set()
    unusual_signed = 0
    for t in range(HIST_SIZE):
        press_n = out[t, 11]
        crc_n   = out[t, 12]
        is_resp_n = out[t, 16]
        is_unusual_n = out[t, 15]
        press_sum += press_n
        if crc_n > crc_max: crc_max = crc_n
        resp_count_signed += int(is_resp_n)  # 0 for master, -1 for slave
        unusual_signed += int(is_unusual_n)
        len_int = int(out[t, 2] + 0.5)
        seen_lens.add(len_int)

    press_mean_w = press_sum / HIST_SIZE
    cmd_count = HIST_SIZE - resp_count_signed
    balance_w = (cmd_count - resp_count_signed) / HIST_SIZE
    nunique_w = float(len(seen_lens))
    unusual_w = float(-unusual_signed)

    for t in range(HIST_SIZE):
        out[t, 17] = press_mean_w
        out[t, 18] = crc_max
        out[t, 19] = float(resp_count_signed)
        out[t, 20] = balance_w
        out[t, 21] = nunique_w
        out[t, 22] = unusual_w

    win_t = torch.from_numpy(np.clip(out, -10, 10)).unsqueeze(0).transpose(1, 2)
    with torch.no_grad():
        prob = torch.sigmoid(model(win_t)).item()
    return prob, frame_count


df = pd.read_csv('IanArffDataset.csv').replace('?', np.nan)
for col in ['pressure','gain','crc','time','setpoint','reset','deadband','cycle','rate','system','control','pump','solenoid']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
N = len(df)
df_test = df.iloc[int(N*0.80):].reset_index(drop=True)
normal_df = df_test[df_test['binary']==0]
for start_idx in range(len(normal_df)):
    if normal_df.iloc[start_idx]['command']==1 and normal_df.iloc[start_idx]['function']==3:
        break
normal_df = normal_df.iloc[start_idx:start_idx+200].reset_index(drop=True)
print(f'Total normal rows: {len(normal_df)}')

probs = []
for idx, row in normal_df.iterrows():
    prob, fc = h743_push_and_infer(row)
    if prob is not None:
        probs.append((fc, prob))

probs_arr = np.array([p for _, p in probs])
print(f'Total inferences: {len(probs)}')
print(f'Mean prob: {probs_arr.mean():.4f}')
print(f'P(>=0.49): {(probs_arr >= 0.49).mean():.2%}')
print(f'P(>=0.30): {(probs_arr >= 0.30).mean():.2%}')
print(f'\nFirst 10 inferences:')
for fc, p in probs[:10]:
    print(f'  frame_count={fc:3d} prob={p:.4f}{" <-- ATTACK" if p >= 0.49 else ""}')
print(f'\nTransition zone (fc 100-130):')
for fc, p in probs:
    if 100 <= fc <= 130:
        print(f'  frame_count={fc:3d} prob={p:.4f}{" <-- ATTACK" if p >= 0.49 else ""}')
