"""
diag_after_fix.py  --  验证修复后的 feature_extractor 输出跟训练一致
"""
import csv
import numpy as np
import torch
import sys
sys.path.insert(0, r'C:\work\Claude\Issue')

MU = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               14135.0, 1.426832, 3.116917, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
SIGMA = np.array([1.0, 13.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  4849.0, 1.521081, 3.241358, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}

class H743FeatExtractor:
    def __init__(self):
        self.hist_pressure = [0.0]*16   # 归一化后
        self.hist_crc_rate = [0.0]*16   # 归一化后
        self.hist_is_resp  = [0.0]*16   # 归一化后
        self.hist_paylen = [0]*16
        self.hist_is_unusual = [0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_addr_func_ts = [0]*256
        self.last_ts = 0

    def update(self, raw):
        n = min(self.hist_count, 16)

        # 用旧 hist 算统计
        press_sum = sum(self.hist_pressure[:n])
        press_mean_w = press_sum / n if n > 0 else 0.0
        crc_max = max(self.hist_crc_rate[:n]) if n > 0 else 0.0
        is_resp_sum = sum(self.hist_is_resp[:n])
        resp_count_w = is_resp_sum if n > 0 else 0.0
        # cmd_resp_balance_w = 1 - 2*is_resp_sum/n  (训练管线公式)
        cmd_resp_balance_w = (1.0 - 2.0 * is_resp_sum / n) if n > 0 else 0.0
        seen_lens = set(self.hist_paylen[:n])
        length_nunique = float(len(seen_lens))
        unusual_count = float(sum(self.hist_is_unusual[:n]))

        # 写入新行 (归一化)
        slot = self.hist_idx
        press_clipped = max(0.0, min(100.0, raw['pressure_measurement']))
        press_norm = (press_clipped - MU[11]) / SIGMA[11]   # f[11] 的 mu/sigma
        crc_norm = (raw['crc_rate'] - MU[12]) / SIGMA[12]   # f[12]
        is_resp_norm = 0.0 if raw['is_response'] else -1.0  # 训练管线

        self.hist_pressure[slot] = press_norm
        self.hist_crc_rate[slot] = crc_norm
        self.hist_is_resp[slot] = is_resp_norm
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

        # 23 维特征
        f = np.zeros(23, dtype=np.float32)
        f[0] = raw['slave']
        f[1] = raw['func']
        f[2] = raw['gain']
        f[3] = raw['reset_rate']
        f[4] = raw['deadband']
        f[5] = raw['cycle_time']
        f[6] = raw['rate']
        f[7] = float(raw['system_mode'])
        f[8] = float(raw['control_scheme'])
        f[9] = float(raw['pump'])
        f[10] = float(raw['solenoid'])
        f[11] = press_clipped   # RAW, 下面归一化
        f[12] = raw['crc_rate'] # RAW, 下面归一化
        f[13] = time_diff
        f[14] = time_since
        f[15] = float(is_unusual)
        f[16] = float(raw['is_response'])  # 0/1, 下面归一化
        f[17] = press_mean_w
        f[18] = crc_max
        f[19] = resp_count_w
        f[20] = cmd_resp_balance_w
        f[21] = length_nunique
        f[22] = unusual_count

        f_norm = (f - MU) / SIGMA
        f_norm = np.clip(f_norm, -10.0, 10.0)
        return f_norm


def frame_parser_parse_from_csv(row):
    return {
        'slave': int(float(row.get('address', 0))),
        'func': int(float(row.get('function', 0))),
        'payload_len': int(float(row.get('length', 0))),
        'gain': float(row.get('gain', 0)),
        'reset_rate': float(row.get('reset', 0)),
        'deadband': float(row.get('deadband', 0)),
        'cycle_time': float(row.get('cycle', 0)),
        'rate': float(row.get('rate', 0)),
        'system_mode': int(float(row.get('system', 0))),
        'control_scheme': int(float(row.get('control', 0))),
        'pump': int(float(row.get('pump', 0))),
        'solenoid': int(float(row.get('solenoid', 0))),
        'pressure_measurement': float(row.get('pressure', 0)),
        'crc_rate': float(row.get('crc', 0)),
        'is_response': int(float(row.get('command', 0))),
        'timestamp_ms': int(float(row.get('time', 0))),
    }


# 加载训练数据作为对照
X_train = np.load(r'C:\work\Claude\Issue\X_train_binary_v2_scada_window16.npy')
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
X_23 = np.clip(X_train[:, :, KEEP_23], -10.0, 10.0)

# 加载 CSV (filter=normal)
with open(r'C:\work\Claude\Issue\IanArffDataset.csv', 'r', encoding='utf-8') as f:
    all_rows = list(csv.DictReader(f))
play_rows = [r for r in all_rows if int((r.get('categorized') or '0').strip()) == 0]
play_rows = play_rows[:48]  # 3 windows

extractor = H743FeatExtractor()
feats_list = []
for i, row in enumerate(play_rows):
    raw = frame_parser_parse_from_csv(row)
    f_norm = extractor.update(raw)
    feats_list.append(f_norm)
feats = np.array(feats_list, dtype=np.float32)

# 跟 X_23[0..2] 比对 (训练 window 0..2 对应 CSV rows 1-48)
print('=== H743 修复后 23 维特征 vs 训练数据 23 维特征 ===')
print(f'{"f":>3} {"name":<22} {"H743 frame 16":>13} {"X_train win0":>14} {"diff":>8}')
for j in range(23):
    h = feats[15, j]   # 第 16 帧 (window 0 最后)
    t = X_23[0, 15, j] # 训练数据 window 0 最后一帧
    diff = h - t
    print(f'{j:>3} {["addr","func","gain","reset","deadband","cycle","rate","sys","ctrl","pump","sol","press","crc","td","tsi","unusual","is_resp","press_mean_w","crc_max_w","resp_count_w","cmd_resp","len_nu","unusual_count"][j]:<22} '
          f'{h:>+13.4f} {t:>+14.4f} {diff:>+8.4f}')

# 关键: 跑推理
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                  map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt['state_dict'])
model.eval()

print('\n=== 推理 3 个 window ===')
for w in range(3):
    win = feats[w*16:(w+1)*16]
    win_t = torch.from_numpy(win.T.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logit = model(win_t).numpy().squeeze()
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    print(f'Window {w+1} (rows {w*16+1}-{(w+1)*16}): prob={prob:.4f}  '
          f'{"ATTACK" if prob>=0.49 else "normal"}')
