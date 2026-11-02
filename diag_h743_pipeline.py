"""
diag_h743_pipeline.py  --  完全模拟 H743 frame_parser + feature_extractor + ch32_forward

跟 H743 端 byte-for-byte 一致:
  1. build_custom_frame(row) → 66 字节
  2. frame_parser_parse(buf) → raw_row_t
  3. feature_extractor_update(raw) → 23 维特征 (用 scaler_export.h 的 mu/sigma)
  4. sliding_window_push → window (16 帧)
  5. ch32_forward(win) → prob (用 PyTorch s42)

输入: IanArffDataset.csv (filter=normal)
输出: 每个 window 的 prob + 对应的 H743 浮点字段值
"""
import os, sys, csv, struct
import numpy as np
import torch
sys.path.insert(0, r'C:\work\Claude\Issue')

# ---- H743 mu/sigma (从 scaler_export.h 抄) ----
MU = np.array([
     4.0,    3.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
     0.0,    0.0,    0.0,    14135.0, 1.426832, 3.116917, 0.0,    1.0,
     0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
], dtype=np.float32)
SIGMA = np.array([
     1.0,   13.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,
     1.0,    1.0,    1.0,    4849.0, 1.521081, 3.241358, 1.0,    1.0,
     1.0,    1.0,    1.0,    1.0,    1.0,    1.0,
], dtype=np.float32)
CLIP = 10.0
SW_WINDOW = 16
FEAT_DIM = 23

# UNUSUAL_FCS (跟 H743 scaler_export.h 一致)
UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}

ADDR_FUNC_CACHE = 256

# ---- 跟 H743 同款 feature_extractor ----
class H743FeatExtractor:
    def __init__(self):
        # raw_history_t hist[FEAT_WIN_ROWS=16]
        self.hist_pressure = [0.0]*16
        self.hist_crc_rate = [0.0]*16
        self.hist_is_resp  = [0]*16
        self.hist_paylen   = [0]*16
        self.hist_is_unusual = [0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_addr_func_ts = [0]*ADDR_FUNC_CACHE
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

        # 写入新行
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
        key = (raw['slave']*16 + raw['func']) % ADDR_FUNC_CACHE
        time_since = 0.0 if self.last_addr_func_ts[key] == 0 else float(ts - self.last_addr_func_ts[key])
        self.last_addr_func_ts[key] = ts

        # 23 维特征 (跟 feature_extractor.c 完全一致)
        f = np.zeros(FEAT_DIM, dtype=np.float32)
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

        # RobustScaler 归一化 + clip ±10σ
        f_norm = (f - MU) / SIGMA
        f_norm = np.clip(f_norm, -CLIP, CLIP)
        return f_norm, raw


def frame_parser_parse_from_csv(row, seq=0):
    """跟 frame_parser_parse + frame_parser.c 的字段填充完全一致"""
    # 跳过 build_custom_frame, 直接从 row 填 raw_row_t (因为我们只看 raw 字段顺序)
    # frame_parser.c line 91-101: gain/reset_rate/deadband/cycle_time/rate/system_mode/control_scheme/pump/solenoid/pressure/crc_rate
    # 注意 system_mode/control_scheme/pump/solenoid 在 H743 是 cast uint8, 但 float 输入没差
    raw = {
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
        'timestamp_ms': int(float(row.get('time', 0))),  # 整数秒
    }
    return raw


def main():
    print('=== 用 H743 同款 feature_extractor 跑 filter=normal CSV ===')
    csv_path = r'C:\work\Claude\Issue\IanArffDataset.csv'
    with open(csv_path, 'r', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))

    # 模拟 GUI filter=normal
    play_rows = [r for r in all_rows if int((r.get('categorized') or '0').strip()) == 0]
    print(f'Total: {len(all_rows)}, normal: {len(play_rows)}')
    print()

    # 取前 80 行 normal
    play_rows = play_rows[:80]

    # 跑 feature_extractor
    extractor = H743FeatExtractor()
    feats_list = []
    raw_list = []
    for i, row in enumerate(play_rows):
        raw = frame_parser_parse_from_csv(row, seq=i)
        f_norm, raw = extractor.update(raw)
        feats_list.append(f_norm)
        raw_list.append(raw)

    feats = np.array(feats_list, dtype=np.float32)
    print(f'feats shape: {feats.shape}')

    # 看前 80 行每个 f[2] (gain) 的归一化值
    print('\n--- 关键维度归一化 (前 30 帧) ---')
    for i in range(30):
        r = play_rows[i]
        f = feats[i]
        print(f'  frame {i+1:>3}: gain(raw)={float(r["gain"]):>5.1f} → f[2]_norm={f[2]:+6.2f}   '
              f'rate(raw)={float(r["rate"]):.3f} → f[6]_norm={f[6]:+6.2f}   '
              f'crc(raw)={float(r["crc"]):>6.0f} → f[12]_norm={f[12]:+6.2f}   '
              f'press={float(r["pressure"]):.3f} → f[11]_norm={f[11]:+6.2f}')

    # 加载 PyTorch 模型
    from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
    ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                      map_location='cpu', weights_only=False)
    model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # 滑窗 + 推理 (16 帧)
    SW = 16
    n_win = len(feats) // SW
    print(f'\n=== 推理 (PyTorch s42, n_win={n_win}) ===')
    print(f'{"win":>4} {"csv_row":>10} {"prob":>8}  {"判决":>6}')
    for w in range(n_win):
        win = feats[w*SW:(w+1)*SW]  # (16, 23)
        win_t = torch.from_numpy(win.T.astype(np.float32)).unsqueeze(0)  # (1, 23, 16)
        with torch.no_grad():
            logit = model(win_t).numpy().squeeze()
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        verdict = 'ATTACK' if prob >= 0.49 else 'normal'
        print(f'{w+1:>4} {w*SW+1:>4}-{w*SW+SW:<5} {prob:>8.4f}  {verdict:>6}')

    # 跟 H743 frame 32 之后一直报的对比: H743 第 32 帧 = play_rows[31]
    print('\n=== 对应 H743 "frame 33 起每帧都报" ===')
    print('H743 实际: frame 32 (CSV play_rows[31]) + frame 33 = 进入第 2 个 window (rows 17-32)')
    print('模拟:')
    for w in range(min(3, n_win)):
        win = feats[w*SW:(w+1)*SW]
        win_t = torch.from_numpy(win.T.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logit = model(win_t).numpy().squeeze()
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        print(f'  Window {w+1} (rows {w*SW+1}-{w*SW+SW}): prob={prob:.4f}  → '
              f'{"会触发 [ALARM]" if prob >= 0.49 else "正常"}')


if __name__ == '__main__':
    main()
