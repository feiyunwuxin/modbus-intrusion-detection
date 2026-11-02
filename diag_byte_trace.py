"""
diag_byte_trace.py  --  模拟 GUI 完整发送链路 + H743 解析 + 完整推理

1. 读 CSV (filter=normal), 模拟 build_custom_frame
2. 模拟 frame_parser_decode 解出 raw_row_t
3. 模拟 feature_extractor_update 算 23 维特征
4. 模拟 ch32_forward (PyTorch)
5. 列出每个窗口的 prob + GUI 发出去的字节 + H743 解析后的字段值

如果第 5 步里 prob 全 high → 字段填错了
"""
import os, sys, struct
import numpy as np
import torch
sys.path.insert(0, r'C:\work\Claude\Issue')

# === 1. GUI build_custom_frame (从 rtu_replay_gui.py 完整复制) ===
def _safe_float(v, default=0.0):
    if v is None or v == '' or (isinstance(v, str) and v.strip() == '?'):
        return float(default)
    try:
        return float(v)
    except:
        return float(default)

def _safe_uint(v, default=0, vmin=0, vmax=0xFFFFFFFF):
    if v is None or v == '' or (isinstance(v, str) and v.strip() == '?'):
        return default
    try:
        x = int(float(v))
    except:
        return default
    return max(vmin, min(vmax, x))

CUSTOM_FRAME_SOP = 0xAA
FRAME_SIZE = 66

def build_custom_frame(row, seq=0, ts_ms=0):
    slave = _safe_uint(row.get('address'), default=1) or 1
    func  = _safe_uint(row.get('function'), default=3)
    length = _safe_uint(row.get('length'), default=4)
    is_response = 1 if _safe_uint(row.get('command')) == 1 else 0

    floats = (
        _safe_float(row.get('setpoint')),
        _safe_float(row.get('gain')),
        _safe_float(row.get('reset')),
        _safe_float(row.get('deadband')),
        _safe_float(row.get('cycle')),
        _safe_float(row.get('rate')),
        _safe_float(row.get('system')),
        _safe_float(row.get('control')),
        _safe_float(row.get('pump')),
        _safe_float(row.get('solenoid')),
        _safe_float(row.get('pressure')),
        _safe_float(row.get('crc'), default=14135.0),
    )
    time_ts = _safe_uint(row.get('time'), default=0)

    body = bytearray()
    body.append(CUSTOM_FRAME_SOP)
    body += struct.pack('<H', seq & 0xFFFF)
    body += struct.pack('<I', ts_ms & 0xFFFFFFFF)
    body.append(slave & 0xFF)
    body.append(func & 0xFF)
    body += struct.pack('<H', length & 0xFFFF)
    for f in floats:
        body += struct.pack('<f', f)
    body.append(is_response & 0xFF)
    body += struct.pack('<I', time_ts & 0xFFFFFFFF)

    # checksum
    csum = 0xA5A5
    for i in range(1, 64):
        csum ^= body[i]
    body += struct.pack('<H', csum & 0xFFFF)
    return bytes(body)


# === 2. H743 frame_parser_parse (从 frame_parser.c 完整复制) ===
def frame_parser_decode(buf):
    """返回 raw_row_t dict"""
    if len(buf) != 66: return None
    if buf[0] != 0xAA: return None
    csum = 0xA5A5
    for i in range(1, 64):
        csum ^= buf[i]
    rx_csum = struct.unpack_from('<H', buf, 64)[0]
    if csum != rx_csum: return None

    def rd_u16(p): return struct.unpack_from('<H', p, 0)[0]
    def rd_u32(p): return struct.unpack_from('<I', p, 0)[0]
    def rd_f32(p): return struct.unpack_from('<f', p, 0)[0]

    raw = {
        'timestamp_ms': rd_u32(buf[60:64]),
        'slave': buf[7],
        'func': buf[8],
        'payload_len': rd_u16(buf[9:11]),
        'gain': rd_f32(buf[11+1*4:11+2*4]),
        'reset_rate': rd_f32(buf[11+2*4:11+3*4]),
        'deadband': rd_f32(buf[11+3*4:11+4*4]),
        'cycle_time': rd_f32(buf[11+4*4:11+5*4]),
        'rate': rd_f32(buf[11+5*4:11+6*4]),
        'system_mode': int(rd_f32(buf[11+6*4:11+7*4])),
        'control_scheme': int(rd_f32(buf[11+7*4:11+8*4])),
        'pump': int(rd_f32(buf[11+8*4:11+9*4])),
        'solenoid': int(rd_f32(buf[11+9*4:11+10*4])),
        'pressure_measurement': rd_f32(buf[11+10*4:11+11*4]),
        'crc_rate': rd_f32(buf[11+11*4:11+12*4]),
        'is_response': buf[59],
    }
    return raw


# === 3. H743 feature_extractor ===
UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}
MU = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               14135.0, 1.426832, 3.116917, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
SIGMA = np.array([1.0, 13.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  4849.0, 1.521081, 3.241358, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

class H743FeatExtractor:
    def __init__(self):
        self.hist_pressure = [0.0]*16
        self.hist_crc_rate = [0.0]*16
        self.hist_is_resp = [0]*16
        self.hist_paylen = [0]*16
        self.hist_is_unusual = [0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_addr_func_ts = [0]*256
        self.last_ts = 0

    def update(self, raw):
        n = min(self.hist_count, 16)
        press_sum = sum(self.hist_pressure[:n])
        press_mean_w = press_sum / n if n > 0 else 0.0
        crc_max = max(self.hist_crc_rate[:n]) if n > 0 else 0.0
        resp_count = sum(self.hist_is_resp[:n])
        req_count = n - resp_count
        cmd_resp_balance = (resp_count - req_count) / n if n > 0 else 0.0
        seen_lens = set(self.hist_paylen[:n])
        length_nunique = float(len(seen_lens))
        unusual_count = float(sum(self.hist_is_unusual[:n]))

        slot = self.hist_idx
        press_clipped = max(0.0, min(100.0, raw['pressure_measurement']))
        self.hist_pressure[slot] = press_clipped
        self.hist_crc_rate[slot] = raw['crc_rate']
        self.hist_is_resp[slot] = raw['is_response']
        self.hist_paylen[slot] = raw['payload_len']
        is_unusual = 1 if raw['func'] in UNUSUAL_FCS else 0
        self.hist_is_unusual[slot] = is_unusual
        self.hist_idx = (self.hist_idx + 1) % 16
        if self.hist_count < 0xFFFFFFFF:
            self.hist_count += 1

        ts = raw['timestamp_ms']
        time_diff = 0.0 if self.last_ts == 0 else float(ts - self.last_ts)
        self.last_ts = ts
        key = (raw['slave']*16 + raw['func']) % 256
        time_since = 0.0 if self.last_addr_func_ts[key] == 0 else float(ts - self.last_addr_func_ts[key])
        self.last_addr_func_ts[key] = ts

        f = np.zeros(23, dtype=np.float32)
        f[0]  = raw['slave']
        f[1]  = raw['func']
        f[2]  = raw['gain']
        f[3]  = raw['reset_rate']
        f[4]  = raw['deadband']
        f[5]  = raw['cycle_time']
        f[6]  = raw['rate']
        f[7]  = float(raw['system_mode'])
        f[8]  = float(raw['control_scheme'])
        f[9]  = float(raw['pump'])
        f[10] = float(raw['solenoid'])
        f[11] = press_clipped
        f[12] = raw['crc_rate']
        f[13] = time_diff
        f[14] = time_since
        f[15] = float(is_unusual)
        f[16] = float(raw['is_response'])
        f[17] = press_mean_w
        f[18] = crc_max
        f[19] = float(resp_count)
        f[20] = cmd_resp_balance
        f[21] = length_nunique
        f[22] = unusual_count

        f_norm = (f - MU) / SIGMA
        f_norm = np.clip(f_norm, -10.0, 10.0)
        return f_norm


# === 4. 跑通 ===
import csv
with open(r'C:\work\Claude\Issue\IanArffDataset.csv', 'r', encoding='utf-8') as f:
    all_rows = list(csv.DictReader(f))

# filter=normal (跟 GUI 完全一致)
play_rows = [r for r in all_rows if int((r.get('categorized') or '0').strip()) == 0]
play_rows = play_rows[:48]  # 前 48 行 (3 个 window)
print(f'play_rows[:48], 全 normal')

extractor = H743FeatExtractor()
feats_list = []
for i, row in enumerate(play_rows):
    # GUI build frame
    frame = build_custom_frame(row, seq=i, ts_ms=0)
    # H743 parse
    raw = frame_parser_decode(frame)
    if raw is None:
        print(f'  frame {i+1}: PARSE FAIL')
        continue
    # H743 feature extract
    f_norm = extractor.update(raw)
    feats_list.append(f_norm)

feats = np.array(feats_list, dtype=np.float32)
print(f'feats shape: {feats.shape}')

# 加载模型
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                  map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# 推理 3 个 window
print('\n=== 推理结果 ===')
for w in range(3):
    win = feats[w*16:(w+1)*16]  # (16, 23)
    win_t = torch.from_numpy(win.T.astype(np.float32)).unsqueeze(0)  # (1, 23, 16)
    with torch.no_grad():
        logit = model(win_t).numpy().squeeze()
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    print(f'Window {w+1} (rows {w*16+1}-{(w+1)*16}): prob={prob:.4f}  '
          f'{"ATTACK" if prob>=0.49 else "normal"}')

# 关键 debug: 第 16 帧 (H743 第 1 次推理, 实际跳过 cold-start 但 prob 已有)
print('\n=== 关键字段归一化值 (frame 32-48) ===')
for i in range(31, 48):
    r = play_rows[i]
    f = feats[i]
    print(f'  frame {i+1:>2}: gain={float(r["gain"]):>5.1f}/f[2]={f[2]:+5.1f}  '
          f'reset={float(r["reset"]):>5.2f}/f[3]={f[3]:+5.1f}  '
          f'rate={float(r["rate"]):>6.3f}/f[6]={f[6]:+5.1f}  '
          f'crc={float(r["crc"]):>6.0f}/f[12]={f[12]:+5.1f}  '
          f'time_diff={f[13]:+5.1f}  '
          f'is_unusual={int(float(r["function"])) in UNUSUAL_FCS}/f[15]={f[15]:+5.1f}')

print('\n=== frame 32 字段详情 (H743 "frame 32 之后一直报" 的起点) ===')
i = 31
r = play_rows[i]
print(f'  CSV row {i+1}: time={r["time"]}  cat={r["categorized"]}  bin={r["binary"]}')
print(f'  function={r["function"]}  address={r["address"]}')
print(f'  gain={r["gain"]}  reset={r["reset"]}  cycle={r["cycle"]}')
print(f'  crc={r["crc"]}  pressure={r["pressure"]}')
print(f'  23维特征 f[0..22]: {feats[i]}')
