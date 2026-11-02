"""
diag_normal_500.py  --  跑 500 个 normal-only frame, 验证 H743 修复后误报率为 0
使用真 scaler (scaler_binary_v2_scada.joblib) 完整 19 维 RobustScaler
"""
import csv, sys
import numpy as np
import torch
import joblib
sys.path.insert(0, r'C:\work\Claude\Issue')

# 加载真 scaler (19 维 RobustScaler, 跟训练管线一致)
scaler = joblib.load(r'C:\work\Claude\Issue\scaler_binary_v2_scada.joblib')
MU19 = scaler.center_.astype(np.float32)   # 19 dim
SIGMA19 = scaler.scale_.astype(np.float32)  # 19 dim
print(f'MU19: {MU19}')
print(f'SIGMA19: {SIGMA19}')

UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}

class H743FeatExtractor:
    """复刻 H743 feature_extractor_update Python 版."""
    def __init__(self):
        self.hist_pressure = [0.0]*16   # 归一化后
        self.hist_crc = [0.0]*16        # 归一化后
        self.hist_is_resp = [0.0]*16    # 归一化后 (0→-1, 1→0)
        self.hist_paylen = [0]*16
        self.hist_is_unusual = [0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_ts = 0
        self.last_addr_func_ts = [0]*256

    def update(self, raw):
        # === 1. 先写 hist (当前帧), 跟训练管线 Xw.reshape(win=16) 对齐 ===
        slot = self.hist_idx
        press_clipped = max(0.0, min(100.0, raw['pressure_measurement']))
        press_norm = (press_clipped - MU19[13]) / SIGMA19[13]
        crc_norm = (raw['crc_rate'] - MU19[14]) / SIGMA19[14]
        is_resp_norm = 0.0 if raw['is_response'] else -1.0

        self.hist_pressure[slot] = press_norm
        self.hist_crc[slot] = crc_norm
        self.hist_is_resp[slot] = is_resp_norm
        self.hist_paylen[slot] = raw['payload_len']
        is_unusual = 1 if raw['func'] in UNUSUAL_FCS else 0
        self.hist_is_unusual[slot] = is_unusual
        self.hist_idx = (self.hist_idx + 1) % 16
        if self.hist_count < 0xFFFFFFFF:
            self.hist_count += 1

        # === 2. 再算窗口统计 (含当前帧), n 用 hist_count 算 16 ===
        n = min(self.hist_count, 16)
        press_sum = sum(self.hist_pressure[:n])
        press_mean_w = press_sum / n if n > 0 else 0.0
        crc_max = max(self.hist_crc[:n]) if n > 0 else 0.0
        is_resp_sum = sum(self.hist_is_resp[:n])
        resp_count_w = is_resp_sum if n > 0 else 0.0
        cmd_resp_balance_w = (1.0 - 2.0 * is_resp_sum / n) if n > 0 else 0.0
        seen_lens = set(self.hist_paylen[:n])
        length_nunique = float(len(seen_lens))
        unusual_count = float(sum(self.hist_is_unusual[:n]))

        ts = raw['timestamp_ms']
        time_diff = 0.0 if self.last_ts == 0 else float(ts - self.last_ts)
        self.last_ts = ts
        key = (raw['slave']*16 + raw['func']) % 256
        time_since = 0.0 if self.last_addr_func_ts[key] == 0 else float(ts - self.last_addr_func_ts[key])
        self.last_addr_func_ts[key] = ts

        # === 填 19 维 row-level (跟训练 preprocess_v2_scada.py 完全一致) ===
        # 列顺序: 0=address, 1=function, 2=length, 3=setpoint, 4=gain, 5=reset,
        # 6=deadband, 7=cycle, 8=rate, 9=system, 10=control, 11=pump, 12=solenoid,
        # 13=pressure, 14=crc, 15=time_diff, 16=time_since, 17=is_unusual, 18=is_response
        # ⭐ 2026-09-19: timestamp_ms 实际装 sec × 1000 (Raw1.csv f32 sec → u32 ms)
        #   time_diff/time_since 必须除以 1000 转 sec, 跟 feat_mu[13]=1.43 sec 对齐
        f19 = np.zeros(19, dtype=np.float32)
        f19[0]  = float(raw['slave'])
        f19[1]  = float(raw['func'])
        f19[2]  = float(raw['payload_len'])
        f19[3]  = float(raw['setpoint'])
        f19[4]  = float(raw['gain'])
        f19[5]  = float(raw['reset_rate'])
        f19[6]  = float(raw['deadband'])
        f19[7]  = float(raw['cycle_time'])
        f19[8]  = float(raw['rate'])
        f19[9]  = float(raw['system_mode'])
        f19[10] = float(raw['control_scheme'])
        f19[11] = float(raw['pump'])
        f19[12] = float(raw['solenoid'])
        f19[13] = press_clipped
        f19[14] = float(raw['crc_rate'])
        f19[15] = time_diff / 1000.0       # ms → sec
        f19[16] = time_since / 1000.0      # ms → sec
        f19[17] = float(is_unusual)
        f19[18] = float(raw['is_response'])

        # === RobustScaler 归一化 19 维 ===
        f19_norm = (f19 - MU19) / SIGMA19

        # === 拼 27 维 (19 row-level + 8 window agg) ===
        f27 = np.zeros(27, dtype=np.float32)
        f27[0:19] = f19_norm
        f27[19] = press_mean_w   # press_mean_w (已归一化)
        f27[20] = 0.0             # crc_mean_w dropped
        f27[21] = crc_max         # crc_max_w (已归一化)
        f27[22] = float(n) - is_resp_sum  # cmd_count_w
        f27[23] = resp_count_w    # resp_count_w (已归一化空间)
        f27[24] = cmd_resp_balance_w
        f27[25] = length_nunique
        f27[26] = unusual_count

        # === KEEP_23 提取 (跟 retrain 脚本完全一致) ===
        KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
        f23 = f27[KEEP_23]

        # === 必须 clip ±10σ ===
        f23 = np.clip(f23, -10.0, 10.0)
        return f23


def _safe_float(v, default=0.0):
    if v is None or v == '' or (isinstance(v, str) and v.strip() == '?'):
        return float(default)
    try:
        return float(v)
    except:
        return float(default)

def parse_row(row):
    return {
        'slave': int(_safe_float(row.get('address'), 0)),
        'func': int(_safe_float(row.get('function'), 0)),
        'payload_len': int(_safe_float(row.get('length'), 0)),
        'gain': _safe_float(row.get('gain'), 0),
        'reset_rate': _safe_float(row.get('reset'), 0),
        'deadband': _safe_float(row.get('deadband'), 0),
        'cycle_time': _safe_float(row.get('cycle'), 0),
        'rate': _safe_float(row.get('rate'), 0),
        'system_mode': int(_safe_float(row.get('system'), 0)),
        'control_scheme': int(_safe_float(row.get('control'), 0)),
        'pump': int(_safe_float(row.get('pump'), 0)),
        'solenoid': int(_safe_float(row.get('solenoid'), 0)),
        'pressure_measurement': _safe_float(row.get('pressure'), 0),
        'crc_rate': _safe_float(row.get('crc'), 14135.0),
        'is_response': int(_safe_float(row.get('command'), 0)),
        'timestamp_ms': _safe_float(row.get('time'), 0) * 1000.0,  # 整数 sec → ms (跟 H743 frame_parser 一致)
        'setpoint': _safe_float(row.get('setpoint'), 0),
    }


# 加载 normal 段 (用 ianArffDataset.csv 整数秒, 跟训练管线完全一致)
with open(r'C:\work\Claude\Issue\IanArffDataset.csv', 'r', encoding='utf-8') as f:
    all_rows = list(csv.DictReader(f))

normal_rows = []
for i, row in enumerate(all_rows):
    if int((row.get('categorized') or '0').strip()) == 0:
        normal_rows.append((i, row))
    if len(normal_rows) >= 500:
        break
print(f'Take {len(normal_rows)} normal rows from CSV rows {normal_rows[0][0]+1}-{normal_rows[-1][0]+1}')

extractor = H743FeatExtractor()
feats_list = []
for _, row in normal_rows:
    r = parse_row(row)
    feats_list.append(extractor.update(r))
feats = np.array(feats_list, dtype=np.float32)
print(f'feats shape: {feats.shape}')

# 加载模型
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                  map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# 跑 window
SW = 16
n_win = len(feats) // SW
print(f'\n=== {n_win} windows ===')
attack_count = 0
for w in range(n_win):
    win = feats[w*SW:(w+1)*SW]
    win_t = torch.from_numpy(win.T.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logit = model(win_t).numpy().squeeze()
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    if prob >= 0.49:
        attack_count += 1
        print(f'  Window {w+1} (rows {w*SW+1}-{(w+1)*SW}): prob={prob:.4f}  ATTACK  <-- FP!')
    elif w < 5 or w >= n_win - 3:
        print(f'  Window {w+1} (rows {w*SW+1}-{(w+1)*SW}): prob={prob:.4f}  normal')

print(f'\nFP rate: {attack_count}/{n_win} = {attack_count/n_win*100:.2f}%')
