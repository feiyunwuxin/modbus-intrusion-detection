"""
diag_replay_inference.py  --  PC 端离线复现 H743 推理

跟 H743 端 modbus_ids.c 完全一致的 pipeline:
  raw_row_t (12 floats + slave/func/length/is_resp) ->
  feature_extractor_update (23 维 + RobustScaler clip ±10σ) ->
  sliding_window (16 帧) ->
  ch32_forward (TCN+SE ch=32 23-dim s42)
"""
import sys
import csv
import struct
import numpy as np
import torch

# ---------- 1. 加载训练好的 PyTorch 模型 (s42 single-seed) ----------
sys.path.insert(0, '.')
device = 'cpu'

# 用 s42 单一模型 (H743 烧的就是这个)
ckpt = torch.load('model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                  map_location=device, weights_only=False)
print(f"Loaded s42 checkpoint. Keys: {list(ckpt.keys())[:5]}")

# 提取模型权重
state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
# 用纯 numpy 模拟, 不写 forward, 直接用 H743 风格重算
# (避免重新搭一遍 TCN+SE 模型, 改用 mu/sigma 归一化 + 检查特征分布)

# ---------- 2. 加载 mu/sigma (跟 H743 训练时一致) ----------
# 这些数字来自 preprocess_v2_scada.py, 跟 scaler_export.h 完全一致
MU = np.array([
    12.001347,    # f[0]  address
    4.000192,     # f[1]  function
    99.999977,    # f[2]  gain
    1.986033e-01, # f[3]  reset_rate
    0.998194,     # f[4]  deadband
    1.000082,     # f[5]  cycle_time
    101.992264,   # f[6]  rate
    3.980471,     # f[7]  system_mode
    1.010165,     # f[8]  control_scheme
    1.011244,     # f[9]  pump
    0.501574,     # f[10] solenoid
    0.710510,     # f[11] pressure_measurement
    14135.2859,   # f[12] crc_rate
    1.426832,     # f[13] time_diff
    337.6893,     # f[14] time_since_last_same_addr_func
    0.0333,       # f[15] is_unusual_fc
    0.5031,       # f[16] is_response
    0.0015,       # f[17] press_mean_w
    14136.8424,   # f[18] crc_max
    7.9856,       # f[19] resp_count
    -0.0016,      # f[20] cmd_resp_balance_w
    2.0207,       # f[21] length_nunique_w
    0.4987,       # f[22] unusual_count_w
], dtype=np.float32)

SIGMA = np.array([
    7.448234,     # f[0]
    1.732081,     # f[1]
    0.004832,     # f[2]
    0.399498,     # f[3]
    0.039929,     # f[4]
    0.010880,     # f[5]
    0.097907,     # f[6]
    1.992818,     # f[7]
    0.100485,     # f[8]
    0.105990,     # f[9]
    0.500019,     # f[10]
    0.453939,     # f[11]
    4848.9111,    # f[12]
    1.521081,     # f[13]
    1453.1135,    # f[14]
    0.179326,     # f[15]
    0.500014,     # f[16]
    0.0015,       # f[17]
    4849.2622,    # f[18]
    4.5227,       # f[19]
    0.1497,       # f[20]
    1.0693,       # f[21]
    1.2296,       # f[22]
], dtype=np.float32)

CLIP_VAL = 10.0
SW_WINDOW = 16
FEAT_DIM = 23
UNUSUAL_FC = set([8, 13, 43, 90, 91, 100])  # 训练时定义的异常 fc

ADDR_FUNC_CACHE = 256
last_addr_func_ts = [0]*ADDR_FUNC_CACHE
last_ts = 0

def is_unusual_fc(fc): return 1 if fc in UNUSUAL_FC else 0

def addr_func_key(slave, func):
    return (slave*16 + func) % ADDR_FUNC_CACHE

# 复现 feature_extractor 的 f[17..22] 窗口统计
class WindowStat:
    def __init__(self):
        self.hist = []  # list of dict
    def update(self, raw):
        n = len(self.hist)
        press_sum = sum(h['pressure'] for h in self.hist)
        press_mean_w = press_sum / n if n > 0 else 0.0
        crc_max = max((h['crc_rate'] for h in self.hist), default=0.0)
        resp_count = sum(h['is_response'] for h in self.hist)
        req_count = n - resp_count
        cmd_resp = (resp_count - req_count) / n if n > 0 else 0.0
        seen_lens = set(h['payload_len'] for h in self.hist)
        length_nunique = float(len(seen_lens))
        unusual_count = sum(h['is_unusual_fc'] for h in self.hist)
        # 追加当前
        self.hist.append({
            'pressure': raw['pressure_measurement'],
            'crc_rate': raw['crc_rate'],
            'is_response': raw['is_response'],
            'payload_len': raw['payload_len'],
            'is_unusual_fc': raw['is_unusual_fc'],
        })
        if len(self.hist) > 16:
            self.hist.pop(0)
        return press_mean_w, crc_max, float(resp_count), cmd_resp, length_nunique, float(unusual_count)


def build_frame_features(rows, start=0, count=80):
    """模拟 H743 frame_parser_parse + feature_extractor_update 流程"""
    global last_addr_func_ts, last_ts
    last_addr_func_ts = [0]*ADDR_FUNC_CACHE
    last_ts = 0
    ws = WindowStat()

    feats_list = []
    for i in range(start, start+count):
        if i >= len(rows): break
        row = rows[i]
        raw = {
            'slave': int(float(row['address'])),
            'func': int(float(row['function'])),
            'gain': float(row['gain']),
            'reset_rate': float(row['reset']),
            'deadband': float(row['deadband']),
            'cycle_time': float(row['cycle']),
            'rate': float(row['rate']),
            'system_mode': int(float(row['system'])),
            'control_scheme': int(float(row['control'])),
            'pump': int(float(row['pump'])),
            'solenoid': int(float(row['solenoid'])),
            'pressure_measurement': max(0.0, min(100.0, float(row['pressure']))),
            'crc_rate': float(row['crc']),
            'is_response': int(float(row['command'])),
            'payload_len': int(float(row['length'])),
            'is_unusual_fc': is_unusual_fc(int(float(row['function']))),
            'timestamp_ms': int(float(row['time'])),  # 整数秒
        }

        # 1. 窗口统计 (基于旧 hist, 不含当前帧)
        press_mean_w, crc_max_w, resp_count_w, cmd_resp_w, length_nunique_w, unusual_count_w = ws.update(raw)

        # 2. time_diff / time_since
        ts = raw['timestamp_ms']
        time_diff = 0.0 if last_ts == 0 else float(ts - last_ts)
        last_ts = ts
        key = addr_func_key(raw['slave'], raw['func'])
        time_since = 0.0 if last_addr_func_ts[key] == 0 else float(ts - last_addr_func_ts[key])
        last_addr_func_ts[key] = ts

        # 3. 23 维特征
        f = np.zeros(FEAT_DIM, dtype=np.float32)
        f[0]  = raw['slave']
        f[1]  = raw['func']
        f[2]  = raw['gain']
        f[3]  = raw['reset_rate']
        f[4]  = raw['deadband']
        f[5]  = raw['cycle_time']
        f[6]  = raw['rate']
        f[7]  = raw['system_mode']
        f[8]  = raw['control_scheme']
        f[9]  = raw['pump']
        f[10] = raw['solenoid']
        f[11] = raw['pressure_measurement']
        f[12] = raw['crc_rate']
        f[13] = time_diff
        f[14] = time_since
        f[15] = float(raw['is_unusual_fc'])
        f[16] = float(raw['is_response'])
        f[17] = press_mean_w
        f[18] = crc_max_w
        f[19] = resp_count_w
        f[20] = cmd_resp_w
        f[21] = length_nunique_w
        f[22] = unusual_count_w

        # 归一化 + clip
        f_norm = (f - MU) / SIGMA
        f_norm = np.clip(f_norm, -CLIP_VAL, CLIP_VAL)
        feats_list.append(f_norm)

    return np.array(feats_list, dtype=np.float32)


def main():
    print('=== PC 端离线复现 H743 推理 ===')
    print('加载 CSV...')
    with open('IanArffDataset.csv', 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'Total rows: {len(rows)}')
    print()

    # 用前 80 行
    feats = build_frame_features(rows, start=0, count=80)
    print(f'Features shape: {feats.shape}')
    print()

    # 1. 列出每个特征维度归一化后的范围 (找异常维度)
    print('--- 特征归一化后 max |z| per dim (前 80 行) ---')
    for i in range(FEAT_DIM):
        z = np.abs(feats[:, i])
        print(f'  f[{i:>2}]: max|z|={z.max():6.2f}  mean|z|={z.mean():5.2f}  '
              f'last={feats[-1, i]:+6.2f}')
    print()

    # 2. 跟训练时的 f[17..22] 窗口统计对比 (看是不是窗口满了之后稳态)
    print('--- f[17..22] 窗口统计随时间变化 (前 40 帧) ---')
    for i in range(40):
        print(f'  frame {i+1:>3}: f[17]={feats[i,17]:+6.2f} f[18]={feats[i,18]:+6.2f} '
              f'f[19]={feats[i,19]:+6.2f} f[20]={feats[i,20]:+6.2f} '
              f'f[21]={feats[i,21]:+6.2f} f[22]={feats[i,22]:+6.2f}')
    print()

    # 3. 关键的 f[18] crc_max 特征 (跟 H743 输出 diff)
    print('--- f[18] crc_max 稳态值 (frame 17+, hist_count >= 16) ---')
    for i in range(16, 80):
        v = feats[i, 18]
        if abs(v) > 5 or i < 40:
            print(f'  frame {i+1:>3}: f[18] = {v:+.3f} (raw crc_max={float(rows[i]["crc"])})')

    print()
    print('=== 结论诊断 ===')
    # 找出最异常的维度
    max_z_per_dim = np.abs(feats[16:]).max(axis=0)  # 稳态
    bad_dims = np.where(max_z_per_dim > 5)[0]
    if len(bad_dims) > 0:
        print(f'⚠️ 稳态 (frame 17+) 仍偏 |z|>5 的维度: {bad_dims.tolist()}')
        for d in bad_dims:
            print(f'    f[{d}]: max|z|={max_z_per_dim[d]:.2f}  '
                  f'(mu={MU[d]:.4f}, sigma={SIGMA[d]:.4f})')
    else:
        print('✓ 所有维度稳态 |z| < 5, 归一化正常')


if __name__ == '__main__':
    main()
