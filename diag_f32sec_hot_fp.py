"""
diag_f32sec_hot_fp.py -- 只测热启动窗口 (frame 17+), 模拟 H743 实际工作状态
GUI 默认 CSV = IanArffDatasetF32.csv (含小数)
模拟 H743 Conv1d (in_ch=23, T=16) layout
"""
import csv, sys
import numpy as np
import torch
import joblib
sys.path.insert(0, r'C:\work\Claude\Issue')

scaler = joblib.load(r'C:\work\Claude\Issue\scaler_binary_v2_scada.joblib')
MU19 = scaler.center_.astype(np.float32)
SIGMA19 = scaler.scale_.astype(np.float32)

UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}


class H743FeatExtractorF32Sec:
    def __init__(self):
        self.hist_pressure = [0.0]*16
        self.hist_crc = [0.0]*16
        self.hist_is_resp = [0.0]*16
        self.hist_paylen = [0]*16
        self.hist_is_unusual = [0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_ts = 0.0
        self.last_addr_func_ts = [0.0]*256

    def update(self, row):
        slot = self.hist_idx
        press_clipped = max(0.0, min(100.0, float(row['pressure measurement']) if row['pressure measurement'] != '?' else 0.0))
        press_norm = (press_clipped - MU19[13]) / SIGMA19[13]
        crc_raw = float(row['crc rate']) if row['crc rate'] != '?' else 14135.0
        crc_norm = (crc_raw - MU19[14]) / SIGMA19[14]
        is_resp_raw = int(float(row['command response']))
        is_resp_norm = 0.0 if is_resp_raw else -1.0

        self.hist_pressure[slot] = press_norm
        self.hist_crc[slot] = crc_norm
        self.hist_is_resp[slot] = is_resp_norm
        self.hist_paylen[slot] = int(float(row['length']))
        is_unusual = 1 if int(float(row['function'])) in UNUSUAL_FCS else 0
        self.hist_is_unusual[slot] = is_unusual
        self.hist_idx = (self.hist_idx + 1) % 16
        if self.hist_count < 0xFFFFFFFF:
            self.hist_count += 1

        # ⭐ Hot state: n=16 强制用最近 16 帧 (跟训练 make_windows_with_agg 一致, broadcast 到 16 timestep)
        n = 16
        # 找最近 16 帧
        press_sum = 0.0; crc_max = 0.0; is_resp_sum = 0.0; unusual_count = 0.0
        seen_lens = set()
        for k in range(16):
            j = (slot + 1 + k) % 16
            press_sum += self.hist_pressure[j]
            if self.hist_crc[j] > crc_max:
                crc_max = self.hist_crc[j]
            is_resp_sum += self.hist_is_resp[j]
            seen_lens.add(self.hist_paylen[j])
            unusual_count += self.hist_is_unusual[j]

        press_mean_w = press_sum / n
        resp_count_w = is_resp_sum
        cmd_resp_balance_w = 1.0 - 2.0 * is_resp_sum / n
        length_nunique = float(len(seen_lens))

        ts = float(row['time'])
        time_diff = 0.0 if self.last_ts == 0 else (ts - self.last_ts)
        self.last_ts = ts
        key = (int(float(row['address']))*16 + int(float(row['function']))) % 256
        time_since = 0.0 if self.last_addr_func_ts[key] == 0 else (ts - self.last_addr_func_ts[key])
        self.last_addr_func_ts[key] = ts

        f19 = np.zeros(19, dtype=np.float32)
        f19[0]  = float(row['address'])
        f19[1]  = float(row['function'])
        f19[2]  = float(row['length'])
        f19[3]  = float(row['setpoint']) if row['setpoint'] != '?' else 0.0
        f19[4]  = float(row['gain']) if row['gain'] != '?' else 0.0
        f19[5]  = float(row['reset rate']) if row['reset rate'] != '?' else 0.0
        f19[6]  = float(row['deadband']) if row['deadband'] != '?' else 0.0
        f19[7]  = float(row['cycle time']) if row['cycle time'] != '?' else 0.0
        f19[8]  = float(row['rate']) if row['rate'] != '?' else 0.0
        f19[9]  = float(row['system mode']) if row['system mode'] != '?' else 0.0
        f19[10] = float(row['control scheme']) if row['control scheme'] != '?' else 0.0
        f19[11] = float(row['pump']) if row['pump'] != '?' else 0.0
        f19[12] = float(row['solenoid']) if row['solenoid'] != '?' else 0.0
        f19[13] = press_clipped
        f19[14] = crc_raw
        f19[15] = time_diff
        f19[16] = time_since
        f19[17] = float(is_unusual)
        f19[18] = float(is_resp_raw)

        f19_norm = (f19 - MU19) / SIGMA19

        f27 = np.zeros(27, dtype=np.float32)
        f27[0:19] = f19_norm
        f27[19] = press_mean_w
        f27[20] = 0.0
        f27[21] = crc_max
        f27[22] = float(n) - is_resp_sum
        f27[23] = resp_count_w
        f27[24] = cmd_resp_balance_w
        f27[25] = length_nunique
        f27[26] = unusual_count

        KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
        return np.clip(f27[KEEP_23], -10.0, 10.0)


def main():
    with open(r'C:\work\Claude\Issue\IanArffDatasetF32.csv', 'r', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))

    normal_rows = []
    for row in all_rows:
        if (row.get('categorized result') or '0').strip() == '0':
            normal_rows.append(row)
        if len(normal_rows) >= 1000:
            break
    print(f'Take {len(normal_rows)} normal rows')

    ext = H743FeatExtractorF32Sec()
    feats = []
    for row in normal_rows:
        feats.append(ext.update(row))
    feats = np.array(feats, dtype=np.float32)

    from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
    ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                      map_location='cpu', weights_only=False)
    model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    SW = 16
    # ⭐ 只跑 hot state (从 frame 17 开始, 每 16 帧算一次)
    n_hot_windows = (len(feats) - SW) // SW
    print(f'\n=== Hot-state windows: {n_hot_windows} (从 frame 17 起) ===')
    attack_count = 0
    for w in range(n_hot_windows):
        win = feats[(w+1)*SW:(w+2)*SW]  # 跳过第一窗 cold-start
        win_t = torch.from_numpy(win.T.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logit = model(win_t).numpy().squeeze()
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        if prob >= 0.5:
            attack_count += 1
            print(f'  Window {w+2} (rows {(w+1)*SW+1}-{(w+2)*SW}): prob={prob:.4f}  ATTACK  <-- FP!')
        elif w < 3 or w >= n_hot_windows - 3:
            print(f'  Window {w+2} (rows {(w+1)*SW+1}-{(w+2)*SW}): prob={prob:.4f}  normal')

    print(f'\nHot-state FP rate: {attack_count}/{n_hot_windows} = {attack_count/n_hot_windows*100:.2f}%')


if __name__ == '__main__':
    main()