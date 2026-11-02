"""
diag_index_mapping.py -- 严格对照 H743 模拟输出 vs X_train[0..2, :, KEEP_23]

H743 f[17..22] = 窗口统计的"归一化空间"值 (按训练 RobustScaler fit)
X_train col 17..22 (除 20,22 dropped) = 同样的"归一化空间"窗口统计
"""
import csv
import numpy as np
import sys
sys.path.insert(0, r'C:\work\Claude\Issue')

# 加载 scaler (跟训练管线一样的 19 维 RobustScaler)
import joblib
scaler = joblib.load(r'C:\work\Claude\Issue\scaler_binary_v2_scada.joblib')
print(f'Scaler center_ (19):  {scaler.center_}')
print(f'Scaler scale_  (19):  {scaler.scale_}')

# KEEP_23 跟训练管线一致 (去掉 [2, 3, 20, 22])
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]

# X_train 19 维 column names (从 preprocess_v2_scada.py 复制)
COL_NAMES_19 = [
    "address", "function", "length",
    "setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
    "system mode", "control scheme", "pump", "solenoid",
    "pressure measurement", "crc rate",
    "time_diff", "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
]

UNUSUAL_FCS = {7, 8, 11, 12, 17, 20, 21, 24}


class H743FeatExtractor:
    """复刻 H743 feature_extractor_update 的 Python 版."""

    def __init__(self):
        # hist 存 16 帧 (pressure_norm, crc_norm, is_resp_norm, payload_len, is_unusual, func, slave)
        self.hist_pressure = [0.0]*16
        self.hist_crc = [0.0]*16
        self.hist_is_resp = [0.0]*16
        self.hist_paylen = [0]*16
        self.hist_is_unusual = [0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_ts = 0
        self.last_addr_func_ts = [0]*256

        # scaler 19 维: 用跟训练管线一样的 RobustScaler
        self.mu19 = scaler.center_  # 19 dim
        self.sigma19 = scaler.scale_  # 19 dim

    def update(self, row):
        """row 是 CSV row dict, 返回 19 维归一化后的特征 (跟 X_train 19 维一致)"""
        n = min(self.hist_count, 16)

        # ---- 计算窗口统计 (用旧 hist 算, 都是归一化空间) ----
        press_sum = sum(self.hist_pressure[:n])
        crc_max = max(self.hist_crc[:n]) if n > 0 else 0.0
        is_resp_sum = sum(self.hist_is_resp[:n])

        press_mean_w = press_sum / n if n > 0 else 0.0
        resp_count_w = is_resp_sum  # sum 归一化后 is_response, ∈ [-n, 0]
        # cmd_count = (1 - is_response_norm).sum = n - is_resp_sum
        # balance = (cmd_count - resp_count) / n = (n - is_resp_sum - is_resp_sum) / n
        #                    = 1 - 2*is_resp_sum/n
        cmd_resp_balance_w = (1.0 - 2.0 * is_resp_sum / n) if n > 0 else 0.0
        seen_lens = set(self.hist_paylen[:n])
        length_nunique = float(len(seen_lens))
        unusual_count = float(sum(self.hist_is_unusual[:n]))

        # ---- 写入新行 ----
        slot = self.hist_idx
        # pressure RAW clip [0, 100], 然后归一化 (跟 f[11] 用同样 mu/sigma)
        press_raw = float(row['pressure'])
        press_clipped = max(0.0, min(100.0, press_raw))
        press_norm = (press_clipped - self.mu19[13]) / self.sigma19[13]

        # crc rate RAW, 然后归一化 (跟 f[14] 用同样 mu/sigma)
        crc_raw = float(row['crc']) if row['crc'] else 14135.0
        crc_norm = (crc_raw - self.mu19[14]) / self.sigma19[14]

        # is_response: 0→-1, 1→0 (跟训练 RobustScaler fit center=1, scale=1 一致)
        is_resp_raw = int(float(row['command']))
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

        # ---- 计算 time_diff / time_since ----
        ts = int(float(row['time']))
        time_diff = 0.0 if self.last_ts == 0 else float(ts - self.last_ts)
        self.last_ts = ts
        key = (int(float(row['address']))*16 + int(float(row['function']))) % 256
        time_since = 0.0 if self.last_addr_func_ts[key] == 0 else float(ts - self.last_addr_func_ts[key])
        self.last_addr_func_ts[key] = ts

        # ---- 填 19 维 row-level ----
        f19 = np.zeros(19, dtype=np.float32)
        f19[0]  = float(row['address'])
        f19[1]  = float(row['function'])
        f19[2]  = float(row['length'])
        f19[3]  = float(row['setpoint']) if row['setpoint'] else 0.0
        f19[4]  = float(row['gain'])
        f19[5]  = float(row['reset'])
        f19[6]  = float(row['deadband'])
        f19[7]  = float(row['cycle'])
        f19[8]  = float(row['rate'])
        f19[9]  = float(row['system'])
        f19[10] = float(row['control'])
        f19[11] = float(row['pump'])
        f19[12] = float(row['solenoid'])
        f19[13] = press_clipped
        f19[14] = crc_raw
        f19[15] = time_diff
        f19[16] = time_since
        f19[17] = float(is_unusual)
        f19[18] = float(is_resp_raw)  # 0/1 RAW (训练 RobustScaler fit 会归一化)

        # ---- RobustScaler 归一化 (19 维整体) ----
        f19_norm = (f19 - self.mu19) / self.sigma19

        # ---- 拼成 27 维 (19 row-level + 8 window agg, broadcast) ----
        f27 = np.zeros(27, dtype=np.float32)
        f27[0:19] = f19_norm
        f27[19] = press_mean_w  # 已归一化
        f27[20] = 0.0           # crc_mean_w dropped
        f27[21] = crc_max       # 已归一化
        # cmd_count_w = n - is_resp_sum (但训练时 n=win=16, 这里 n 跟 hist_count)
        # 训练用 win=16 固定. 这里 hist_count<16 时 n 跟训练不一样!
        # 见下面说明 ...
        if n > 0:
            f27[22] = float(n) - is_resp_sum  # cmd_count_w (但 n 可能是 <16!)
        else:
            f27[22] = 0.0
        f27[23] = resp_count_w
        f27[24] = cmd_resp_balance_w
        f27[25] = length_nunique
        f27[26] = unusual_count

        return np.clip(f27[KEEP_23], -10.0, 10.0)  # 23 维, 必须 clip


def parse_csv_row(row):
    """Parse CSV row to dict"""
    return row


# 读 CSV, 取前 48 行 (3 windows)
with open(r'C:\work\Claude\Issue\IanArffDataset.csv', 'r', encoding='utf-8') as f:
    all_rows = list(csv.DictReader(f))

extractor = H743FeatExtractor()
feats_list = []
for row in all_rows[:48]:
    feats_list.append(extractor.update(row))
feats_h743 = np.array(feats_list, dtype=np.float32)
print(f'\nfeats_h743 shape: {feats_h743.shape}')

# 跟训练 X_train[0:3, :, :] (前 3 个 window, 每窗 16 帧) 比对
X_train = np.load(r'C:\work\Claude\Issue\X_train_binary_v2_scada_window16.npy')
X_train_23 = X_train[0:3, :, KEEP_23]
print(f'X_train_23 shape: {X_train_23.shape}')

# 按训练管线: 第 16 帧时 hist_count=16, n=16
# 但 H743 Python 在第 16 帧时 n=15 (因为还没写入这一帧的 hist!)
# 训练管线: 先 make window, 算 Xw[:, :, IDX_PRESS].mean(axis=1) — 这是在 Xw 的 16 帧上算
# H743 Python 在第 16 帧时算 hist[:16] 的窗口统计 — 但 hist_count=15, n=15!

# 输出第 16 帧 (frame 16, window 0 最后一帧) 的 23 维
print('\n=== Frame 16 (window 0 最后一帧) ===')
print(f'{"f":>3} {"name":<22} {"H743":>10} {"X_train":>10} {"diff":>10}')
names23 = ['addr','func','gain','reset','deadband','cycle','rate','sys','ctrl','pump','sol',
           'press','crc','td','tsi','unusual','is_resp',
           'press_mean_w','crc_max_w','resp_count_w','cmd_resp','len_nu','unusual_count']
for j in range(23):
    h = feats_h743[15, j]
    t = X_train_23[0, 15, j]
    diff = h - t
    flag = '!' if abs(diff) > 0.5 else ''
    print(f'{j:>3} {names23[j]:<22} {h:>+10.4f} {t:>+10.4f} {diff:>+10.4f}  {flag}')

# 跑推理看 prob
import torch
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                  map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt['state_dict'])
model.eval()

print('\n=== H743 模拟 vs X_train 真值 prob 对比 ===')
for w in range(3):
    win_h = feats_h743[w*16:(w+1)*16]
    win_t = X_train_23[w]
    win_h_t = torch.from_numpy(win_h.T.astype(np.float32)).unsqueeze(0)
    win_t_t = torch.from_numpy(win_t.T.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logit_h = model(win_h_t).numpy().squeeze()
        logit_t = model(win_t_t).numpy().squeeze()
    prob_h = 1.0 / (1.0 + np.exp(-np.clip(logit_h, -50, 50)))
    prob_t = 1.0 / (1.0 + np.exp(-np.clip(logit_t, -50, 50)))
    print(f'Window {w+1}: H743 prob={prob_h:.4f}  vs  X_train prob={prob_t:.4f}  '
          f'{"OK" if abs(prob_h-prob_t)<0.01 else "DIFF"}')
