"""
diag_f32sec_validation.py -- 用 f32 高精度 CSV 验证 H743 模拟 vs X_train bit-perfect

修复后预期:
- time_diff 从 CSV 算出来 = X_train 一致 (小数位 sec)
- H743 模拟 prob ≈ X_train prob (frame 16/32/48 各窗)
"""
import csv
import sys
import numpy as np
import joblib
import torch

sys.path.insert(0, r'C:\work\Claude\Issue')

# === 加载 scaler ===
scaler = joblib.load(r'C:\work\Claude\Issue\scaler_binary_v2_scada.joblib')
MU19 = scaler.center_.astype(np.float32)
SIGMA19 = scaler.scale_.astype(np.float32)

UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}


class H743FeatExtractorF32Sec:
    """复刻 H743 feature_extractor_update (f32 sec timestamp)."""

    def __init__(self):
        self.hist_pressure = [0.0]*16
        self.hist_crc = [0.0]*16
        self.hist_is_resp = [0.0]*16
        self.hist_paylen = [0]*16
        self.hist_is_unusual = [0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_ts = 0.0    # f32 sec (跟 H743 memcpy memset 一致)
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

        n = min(self.hist_count, 16)
        press_sum = sum(self.hist_pressure[:n])
        crc_max = max(self.hist_crc[:n]) if n > 0 else 0.0
        is_resp_sum = sum(self.hist_is_resp[:n])

        press_mean_w = press_sum / n if n > 0 else 0.0
        resp_count_w = is_resp_sum
        cmd_resp_balance_w = (1.0 - 2.0 * is_resp_sum / n) if n > 0 else 0.0
        seen_lens = set(self.hist_paylen[:n])
        length_nunique = float(len(seen_lens))
        unusual_count = float(sum(self.hist_is_unusual[:n]))

        # ⭐ f32 sec 时间戳 (跟 H743 一致)
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
        f27[22] = float(n) - is_resp_sum if n > 0 else 0.0
        f27[23] = resp_count_w
        f27[24] = cmd_resp_balance_w
        f27[25] = length_nunique
        f27[26] = unusual_count

        KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
        return np.clip(f27[KEEP_23], -10.0, 10.0)


def main():
    # === 读 f32 CSV ===
    with open(r'C:\work\Claude\Issue\IanArffDatasetF32.csv', 'r', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))

    normal_rows = []
    for row in all_rows:
        if (row.get('categorized result') or '0').strip() == '0':
            normal_rows.append(row)
        if len(normal_rows) >= 500:
            break
    print(f'Take {len(normal_rows)} normal rows from f32 CSV')

    extractor = H743FeatExtractorF32Sec()
    feats = []
    for row in normal_rows:
        feats.append(extractor.update(row))
    feats = np.array(feats, dtype=np.float32)

    # === 加载模型 ===
    from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
    ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                      map_location='cpu', weights_only=False)
    model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    SW = 16
    n_win = len(feats) // SW
    print(f'\n=== {n_win} windows (f32 high-precision CSV) ===')
    attack_count = 0
    for w in range(n_win):
        win = feats[w*SW:(w+1)*SW]   # (16, 23)
        # ⭐ 训练时 np.transpose(0, 2, 1) → (N, 23, 16), H743 Conv1d 也期待 (in_ch=23, T=16)
        win_t = torch.from_numpy(win.T.astype(np.float32)).unsqueeze(0)  # (1, 23, 16)
        with torch.no_grad():
            logit = model(win_t).numpy().squeeze()
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        if prob >= 0.49:
            attack_count += 1
            print(f'  Window {w+1} (rows {w*SW+1}-{(w+1)*SW}): prob={prob:.4f}  ATTACK  <-- FP!')
        elif w < 5 or w >= n_win - 3:
            print(f'  Window {w+1} (rows {w*SW+1}-{(w+1)*SW}): prob={prob:.4f}  normal')

    print(f'\nFP rate: {attack_count}/{n_win} = {attack_count/n_win*100:.2f}%')


if __name__ == '__main__':
    main()