"""
仿真新 H743 modbus_ids.c 路径验证.
"""
import numpy as np
import pandas as pd
import struct
import re
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

# === CRC16 (matches modbus_crc16 in H743) ===
def modbus_crc16(payload: bytes) -> int:
    crc = 0xFFFF
    for b in payload:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc

# === 模拟 rtu_replay_gui.build_rtu_frame ===
def build_rtu_frame(row) -> bytes:
    # CSV column is 'command' (ARFF: 'command response')
    # command=1 → master (request), command=0 → slave (response)
    is_master = int(row.get('command', 0)) == 1
    slave = max(1, min(247, int(row.get('address', 1))))
    func = int(row.get('function', 3))
    length = max(0, min(255, int(row.get('length', 0))))
    if func in (1, 2, 3, 4):
        if is_master:  # master request: short PDU (addr+count)
            if func in (1, 2):
                count = max(length * 8, 1)
            else:
                count = max(length // 2, 1)
            pdu = struct.pack('>HH', 0, count)
        else:  # slave response: long PDU (byte_count + data)
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

# === 模拟 modbus_rtu_parse ===
def parse_rtu(buf):
    if len(buf) < 6 or len(buf) > 256:
        return None
    calc_crc = modbus_crc16(buf[:-2])
    rx_crc = (buf[-1] << 8) | buf[-2]
    crc_valid = (calc_crc == rx_crc)
    return {
        'slave': buf[0], 'func': buf[1],
        'total_len': len(buf), 'payload_len': len(buf) - 4,
        'crc_valid': crc_valid,
    }

# === 新 is_response 规则 (FLIPPED: master=1, slave=0, matches CSV command response) ===
def classify_response(f):
    if f['func'] >= 0x80:
        return 0  # exception → slave → is_resp=0
    if f['func'] in (1, 2, 3, 4):
        return 0 if f['payload_len'] > 4 else 1  # read: short payload=master=1
    if f['func'] in (15, 16):
        return 0 if f['payload_len'] <= 4 else 1  # write: long payload=master=1
    return 1  # default master=1

# === 读取 scaler 参数 (strip /* */ comments first) ===
scaler_h_raw = open('KeilH743/H743/Core/Inc/scaler_export.h').read()
scaler_h = re.sub(r'/\*.*?\*/', '', scaler_h_raw, flags=re.DOTALL)
def extract_floats(name):
    m = re.search(rf'static const float feat_{name}\[\s*FEAT_DIM\s*\]\s*=\s*\{{([^}}]+)\}}', scaler_h)
    body = m.group(1)
    vals = []
    for tok in body.split(','):
        tok = tok.strip().rstrip('f').strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return np.array(vals, dtype=np.float32)
mu = extract_floats('mu')
sigma = extract_floats('sigma')
print(f'mu len: {len(mu)}, sigma len: {len(sigma)}')
unusual_m = re.search(r'static const uint8_t feat_unusual_fc\[\s*FEAT_UNUSUAL_FC_COUNT\s*\]\s*=\s*\{([^}]+)\}', scaler_h)
unusual_fc = set(int(x.strip()) for x in unusual_m.group(1).split(',') if x.strip())
print(f'mu:    {mu[:23].round(3)}')
print(f'sigma: {sigma[:23].round(3)}')
print(f'unusual_fc: {sorted(unusual_fc)}')

CLIP = 10.0
HIST_SIZE = 16
hist_buf = deque(maxlen=HIST_SIZE)
last_ts_ms = 0

# === Process state mocks — use CSV actual values per row ===
# For master requests: CSV has gain=0, pressure=0 etc.
# For slave responses: CSV has actual SCADA process values
# H743 process_state stores LAST KNOWN from slave responses (or 0 if never seen)
TYPICAL = {
    'gain': 0.0, 'reset_rate': 0.0, 'deadband': 0.0,
    'cycle_time': 0.0, 'rate': 0.0, 'system_mode': 0.0,
    'control_scheme': 0.0, 'pump': 0.0, 'solenoid': 0.0,
    'pressure': 0.0, 'crc_rate': 14135.0,
}

def extract_features(raw, hist):
    f = np.zeros(23, dtype=np.float32)
    f[0] = raw['slave']
    f[1] = raw['func']
    f[2] = raw['gain']
    f[3] = raw['reset_rate']
    f[4] = raw['deadband']
    f[5] = raw['cycle_time']
    f[6] = raw['rate']
    f[7] = raw['system_mode']
    f[8] = raw['control_scheme']
    f[9] = raw['pump']
    f[10] = raw['solenoid']
    f[11] = min(100.0, max(0.0, raw['pressure']))
    f[12] = raw['crc_rate']
    f[13] = raw['time_diff']
    f[14] = raw['time_since_same']
    f[15] = 1.0 if raw['func'] in unusual_fc else 0.0
    f[16] = 1.0 if raw['is_response'] else 0.0

    n = len(hist)
    if n > 0:
        press_mean_w = np.mean([h['pressure'] for h in hist])
        crc_max = max([h['crc_rate'] for h in hist])
        # CRITICAL FIX: training uses SCALED is_response values (0 or -1) for sums
        # is_response=1 (master) -> scaled 0, is_response=0 (slave) -> scaled -1
        # So resp_count_w = sum of SCALED is_response (range -16..0)
        # And cmd_count_w  = sum of (1 - scaled is_response) (range 16..32)
        # Balance_w = (cmd - resp) / n (so 8 masters + 8 slaves -> balance = (16 - (-8))/16 = +1.5)
        resp_count = sum(h['is_response_scaled'] for h in hist)
        cmd_count  = sum(1 - h['is_response_scaled'] for h in hist)
        cmd_resp_balance_w = (cmd_count - resp_count) / n
        seen_lens = set(h['payload_len'] for h in hist)
        length_nunique_w = len(seen_lens)
        unusual_count_w = sum(h['is_unusual_fc_scaled'] for h in hist)
    else:
        press_mean_w = crc_max = cmd_resp_balance_w = length_nunique_w = unusual_count_w = 0
        resp_count = 0

    f[17] = press_mean_w
    f[18] = crc_max
    f[19] = float(resp_count)
    f[20] = cmd_resp_balance_w
    f[21] = float(length_nunique_w)
    f[22] = unusual_count_w

    out = (f - mu) / sigma
    out = np.clip(out, -CLIP, CLIP)
    return out

# === Load TCN+SE model (matches retrain_tcn_23dim_b64_ch32_do01_savept.py) ===
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
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
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)

state = torch.load('model_v4_se_23dim_b64_ch32_do01_window16_s42.pt', map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, n_blocks=3, kernel_size=3, dilations=(1,2,4), dropout=0.1)
model.load_state_dict(state['state_dict'])
model.eval()

# === Run sim on TEST split NORMAL rows (matches saved test data distribution) ===
df = pd.read_csv('IanArffDataset.csv')  # use ORIGINAL not F32
df = df.replace('?', np.nan)
df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce').fillna(0)
df['gain'] = pd.to_numeric(df['gain'], errors='coerce').fillna(0)
df['crc'] = pd.to_numeric(df['crc'], errors='coerce').fillna(0)
df['time'] = pd.to_numeric(df['time'], errors='coerce').fillna(0)
for col in ['setpoint','reset','deadband','cycle','rate','system','control','pump','solenoid']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
# Use TEST split (last 20%) per preprocess_v2_scada.py
N = len(df)
n_train = int(N * 0.70)
n_val = int(N * 0.10)
df_test = df.iloc[n_train + n_val:].reset_index(drop=True)
normal_df = df_test[df_test['binary'] == 0]
# 2026-09-19: saved test 第一窗从 master FC3 poll 开始 (saved f[16]=[0,-1,0,-1...])
# CSV 第一个 normal 行可能是 slave (cmd=0), 不匹配. 跳过到第一个 master FC3.
for start_idx in range(len(normal_df)):
    if normal_df.iloc[start_idx]['command']==1 and normal_df.iloc[start_idx]['function']==3:
        break
normal_df = normal_df.iloc[start_idx:start_idx+200].reset_index(drop=True)
print(f'\n=== Running sim on {len(normal_df)} NORMAL CSV rows (starting from idx {start_idx}, first master FC3) ===')

window = deque(maxlen=16)
all_probs = []
ts_ms = 1000
prev_ts = None

for idx, row in normal_df.iterrows():
    frame = build_rtu_frame(row)
    f = parse_rtu(frame)
    if f is None:
        continue
    is_resp = classify_response(f)

    # Use CSV's actual process values
    gain = float(row.get('gain', 0) or 0) if not pd.isna(row.get('gain', 0)) else 0.0
    pressure = float(row.get('pressure', 0) or 0) if not pd.isna(row.get('pressure', 0)) else 0.0
    crc = float(row.get('crc', 0) or 0) if not pd.isna(row.get('crc', 0)) else 0.0
    setpoint = float(row.get('setpoint', 0) or 0) if not pd.isna(row.get('setpoint', 0)) else 0.0
    reset_rate = float(row.get('reset', 0) or 0) if not pd.isna(row.get('reset', 0)) else 0.0
    deadband = float(row.get('deadband', 0) or 0) if not pd.isna(row.get('deadband', 0)) else 0.0
    cycle_time = float(row.get('cycle', 0) or 0) if not pd.isna(row.get('cycle', 0)) else 0.0
    rate = float(row.get('rate', 0) or 0) if not pd.isna(row.get('rate', 0)) else 0.0
    system_mode = float(row.get('system', 0) or 0) if not pd.isna(row.get('system', 0)) else 0.0
    control_scheme = float(row.get('control', 0) or 0) if not pd.isna(row.get('control', 0)) else 0.0
    pump = float(row.get('pump', 0) or 0) if not pd.isna(row.get('pump', 0)) else 0.0
    solenoid = float(row.get('solenoid', 0) or 0) if not pd.isna(row.get('solenoid', 0)) else 0.0
    cur_ts = float(row.get('time', 0) or 0) if not pd.isna(row.get('time', 0)) else 0.0
    # ⭐ SCADA 训练数据 time_diff 合成模式 (saved test f[13] 实测):
    #   全部 normal 窗: P(<0.5s)=50%, P(0.5-2.5s)=50% → master=1.5s±noise, slave=0.1s±noise 交替
    #   master: ~1.5s (μ=1.521, σ~0.07)
    #   slave:  ~0.1s (μ=0.099, σ~0.03)
    # CSV time 列整数精度 0/1/2s 与训练不符 → H743 f[13] 偏离 → 模型误判攻击.
    # 2026-09-19 第六轮: 模拟 rtu_replay_gui 的 synth_timing=True 路径.
    is_master_now = int(row.get('command', 0)) == 1
    if prev_ts is None or cur_ts == 0:
        # 第一帧: master 1.5s, slave 0.1s (对齐 saved test 第一行 master)
        time_diff_sec = 1.5 if is_master_now else 0.1
    elif is_master_now:
        # master 用 ±5% jitter: 1.425-1.575s
        time_diff_sec = 1.5 + 0.075 * (2 * (idx % 7) / 7 - 1)
    else:
        # slave 用 ±5% jitter: 0.095-0.105s
        time_diff_sec = 0.1 + 0.005 * (2 * (idx % 7) / 7 - 1)
    prev_ts = cur_ts

    # DEBUG: print first 16 parsed rows
    if idx < 16:
        print(f'idx={idx} func={f["func"]:3d} pl={f["payload_len"]:3d} is_resp={is_resp} gain={gain} pressure={pressure} crc={crc} dt={time_diff_sec:.3f}')

    raw = {
        'slave': f['slave'], 'func': f['func'],
        'gain': gain, 'reset_rate': reset_rate,
        'deadband': deadband, 'cycle_time': cycle_time,
        'rate': rate, 'system_mode': system_mode,
        'control_scheme': control_scheme, 'pump': pump,
        'solenoid': solenoid, 'pressure': pressure,
        'crc_rate': crc, 'is_response': is_resp,
        'payload_len': f['payload_len'], 'time_diff': time_diff_sec,
        'time_since_same': time_diff_sec,
    }

    feats = extract_features(raw, list(hist_buf))
    # === 4. 写入 hist (scale pressure/crc_rate/is_response/is_unusual_fc BEFORE hist to match training)
    # Training (preprocess_v2_scada.py) computes aggregates on SCALED is_response values.
    # is_response=1 -> scaled 0 (mu=1, sigma=1), is_response=0 -> scaled -1.
    # So resp_count_w = sum(scaled) range -16..0.
    # Same for is_unusual_fc (binary, no scaling, sum is just count).
    raw_for_hist = dict(raw)
    raw_for_hist['pressure'] = (raw['pressure'] - mu[11]) / sigma[11]
    raw_for_hist['crc_rate'] = (raw['crc_rate'] - mu[12]) / sigma[12]
    raw_for_hist['is_response_scaled'] = (raw['is_response'] - mu[16]) / sigma[16]
    raw_for_hist['is_unusual_fc_scaled'] = (1.0 if raw['func'] in unusual_fc else 0.0 - mu[15]) / sigma[15]
    hist_buf.append(raw_for_hist)
    window.append(feats)

    ts_ms += 100
    last_ts_ms = ts_ms

    if len(window) == 16:
        win = np.array(window, dtype=np.float32)  # (16, 23) = (T, F)
        # ⭐ FIX 2026-09-19: Training pipeline (preprocess_v2_scada.py line 205) BROADCASTS
        # the per-window aggregates (f[17..22]) to ALL 16 timesteps — all 16 rows in a
        # window have the SAME aggregate value (the FULL window aggregate).
        # H743 old code computes aggregates incrementally per row (first row=0, last row=full),
        # making the model see a "ramp" signal it was never trained on → prob=1.0.
        # FIX: replace f[17..22] in the just-filled window with the FULL window aggregate
        # (last row's aggregates), broadcast to all 16 rows.
        last_aggs = win[15, 17:23].copy()
        win[:, 17:23] = last_aggs  # broadcast
        if idx == 15:
            print(f'\n[DEBUG] idx=15 win[0..3] f[17..22]:')
            for t in range(4):
                print(f'  t={t}: {win[t, 17:23].round(3)}')
            print(f'  win[15] f[17..22]: {win[15, 17:23].round(3)}')
            print(f'  win[0] f[0..16]: {win[0, :17].round(3)}')
        win_clipped = np.clip(win, -10.0, 10.0)
        win_t = torch.from_numpy(win_clipped).unsqueeze(0).transpose(1, 2)  # (B=1, F=23, T=16)
        with torch.no_grad():
            logit = model(win_t)
            prob = torch.sigmoid(logit).item()
        if idx == 15:
            print(f'  prob={prob:.4f}')
        all_probs.append((idx, prob, is_resp, f['func'], f['payload_len'], feats.copy()))

print(f'\n=== Probabilities on NORMAL data ===')
normal_probs = np.array([p for _, p, _, _, _, _ in all_probs])
print(f'Total windows: {len(all_probs)}')
print(f'Mean prob: {normal_probs.mean():.4f}')
print(f'Max prob:  {normal_probs.max():.4f}')
print(f'P(>=0.49): {(normal_probs >= 0.49).mean():.2%}')
print(f'P(>=0.30): {(normal_probs >= 0.30).mean():.2%}')

print('\n=== First 5 inference windows with normalized features ===')
print(f'{"idx":>4} {"prob":>6} {"f[0..16]":<60}')
for idx, prob, ir, fn, pl, feats in all_probs[:5]:
    f_str = ' '.join(f'{feats[i]:+5.2f}' for i in range(17))
    print(f'{idx:>4} {prob:>6.3f}  {f_str}')

print('\n=== Window aggregates (f[17..22]) ===')
for idx, prob, ir, fn, pl, feats in all_probs[:5]:
    f_str = ' '.join(f'{feats[i]:+5.2f}' for i in range(17, 23))
    print(f'idx={idx} prob={prob:.3f}  aggs: {f_str}')

print('\n=== Last 5 windows ===')
for idx, prob, ir, fn, pl, feats in all_probs[-5:]:
    f_str = ' '.join(f'{feats[i]:+5.2f}' for i in range(23))
    print(f'idx={idx} prob={prob:.3f}  {f_str}')