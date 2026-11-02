"""
diag_train_truth.py  --  严格按 preprocess_v2_scada.py 处理 CSV, 然后对比 H743 输入
"""
import os, sys, csv, struct
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import RobustScaler

BASE = r"C:\work\Claude\Issue"

# === preprocess_v2_scada.py STEP 1-4 ===
df = pd.read_csv(os.path.join(BASE, "IanArffDataset.csv"))
df = df.replace("?", np.nan)

NUMERIC_COLS = ["setpoint", "gain", "reset rate", "deadband", "cycle time",
                "rate", "system mode", "control scheme", "pump", "solenoid",
                "pressure measurement"]
# 老 CSV 列名无空格, preprocess_v2_scada.py 假设有空格
# 但 train_binary_v2_scada_window16.npy 存在, 说明训练是用带空格列名的 CSV
# 我重新生成 arff_with_space.csv 临时测试

# 直接用 train_binary_v2_scada_window16.npy 加载真实训练数据
X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy"))
y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy"))
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_train dtype: {X_train.dtype}')
print(f'X_train stats per col (前 19 维):')
for i in range(19):
    c = X_train[:, :, i].flatten()
    print(f'  27-dim col {i:>2}: mean={c.mean():+.3f}  std={c.std():.3f}  '
          f'min={c.min():+.3f}  max={c.max():+.3f}')

# 取前 80 个 window (对应 CSV 前 80*16=1280 行)
# 实际 play_rows[:80] (filter=normal) 是 CSV 整个文件的前 80 行 normal, 不是前 1280 行
# play_rows[:80] 在 CSV 里的位置: 前 80 行里全 normal (rows 0-44 都是 normal, rows 45+ 开始 attack)
# 但 filter=normal 时, play_rows[0] = CSV[0] (time=1418682163)
# H743 第 1 帧对应 play_rows[0], H743 第 16 帧对应 play_rows[15]
# H743 第 32 帧对应 play_rows[31] - 这是 CSV row 31 (time=1418682190, binary=0)

# 我先看训练数据 window[0] (rows 1-16) 的 f[2]=gain (gain 在 27-dim idx 4, 因为 KEEP_23 也跳过 length)
# KEEP_23 = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
# H743 f[2] = KEEP_23[2] = 4 = gain
# 训练数据已经用 scaler.transform 归一化过了, X_train[:, :, 4] 应该是归一化后的 gain
print(f'\n训练数据 window 0 (前 16 帧) 的 27-dim col 4 (gain, 已归一化):')
col = X_train[0, :, 4]
print(f'  values: {col[:16]}')

# KEEP_23 后 8 个是 window-level aggregates
# 27-dim idx 19 = press_mean_w (RAW, 不归一化)
# 27-dim idx 21 = crc_max_w (RAW)
# 27-dim idx 23 = resp_count_w (RAW)
# ...
print(f'\n训练数据 window 0 的 27-dim col 19 (press_mean_w, RAW):')
col = X_train[0, :, 19]
print(f'  values: {col[:16]}')
print(f'  这跟 H743 feature_extractor 算的 press_mean_w 应该一致')

# 现在对比: 我用 H743 feature_extractor 跑 play_rows[:80] 算出的 f[17] (press_mean_w, RAW)
# 应该跟 X_train[0..4, :, 19] 一致 (如果是同一段数据)
# 但 H743 第 1 帧对应 play_rows[0], 而训练 window[0] 对应 CSV rows 0-15
# play_rows[0] 是 CSV[0] (categorize=0), play_rows[1] 是 CSV[1]
# 但训练时 window[0] 是按 CSV 原顺序 rows 0-15 (含 rows 46+ 的 attack)
# 所以训练数据 window[0..4] 跟 H743 处理 play_rows[:80] 是不同数据段!

# 但我可以验证特征归一化值的统计分布是否一致
print('\n=== 验证: H743 处理 play_rows[:80] 跟 X_train 统计对比 ===')
# H743 处理 normal[:80]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}
ADDR_FUNC_CACHE = 256

class H743FeatExtractor:
    def __init__(self):
        self.hist_pressure = [0.0]*16
        self.hist_idx = 0
        self.hist_count = 0
        self.last_ts = 0

    def update(self, raw):
        n = min(self.hist_count, 16)
        press_sum = sum(self.hist_pressure[:n])
        press_mean_w = press_sum / n if n > 0 else 0.0

        slot = self.hist_idx
        press_clipped = max(0.0, min(100.0, raw['pressure_measurement']))
        self.hist_pressure[slot] = press_clipped
        self.hist_idx = (self.hist_idx + 1) % 16
        if self.hist_count < 0xFFFFFFFF:
            self.hist_count += 1

        time_diff = 0.0 if self.last_ts == 0 else float(raw['timestamp_ms'] - self.last_ts)
        self.last_ts = raw['timestamp_ms']

        return press_mean_w, time_diff, press_clipped, raw['crc_rate']

# 处理前 80 行 normal (filter=normal)
extractor = H743FeatExtractor()
h743_press_means = []
h743_time_diffs = []
h743_presses = []
h743_crcs = []
# 不直接读 IanArffDataset.csv, 改用训练管线顺序 (Raw1 CSV 带空格列名)
df_full = pd.read_csv(os.path.join(BASE, "IanArffDatasetRaw1.csv"))  # 带空格列名
df_full = df_full.iloc[:80]  # 前 80 行

for i in range(80):
    r = df_full.iloc[i]
    raw = {
        'pressure_measurement': float(r['pressure measurement']),
        'crc_rate': float(r['crc rate']),
        'timestamp_ms': int(float(r['time'])),
        'slave': int(float(r['address'])),
        'func': int(float(r['function'])),
        'payload_len': int(float(r['length'])),
        'is_response': int(float(r['command response'])),
        'gain': float(r['gain']),
        'reset_rate': float(r['reset rate']),
        'deadband': float(r['deadband']),
        'cycle_time': float(r['cycle time']),
        'rate': float(r['rate']),
        'system_mode': int(float(r['system mode'])),
        'control_scheme': int(float(r['control scheme'])),
        'pump': int(float(r['pump'])),
        'solenoid': int(float(r['solenoid'])),
    }
    press_mean, td, press_clipped, crc = extractor.update(raw)
    h743_press_means.append(press_mean)
    h743_time_diffs.append(td)
    h743_presses.append(press_clipped)
    h743_crcs.append(crc)

print(f'\nH743 处理前 80 行的统计:')
print(f'  press_mean_w  mean={np.mean(h743_press_means[16:]):.6f}  (应该 ≈ 训练集均值 0.0015 * 训练 μ 不一定)')
print(f'  time_diff     mean={np.mean(h743_time_diffs[1:]):.6f}  (训练 μ=1.426832)')
print(f'  pressure      mean={np.mean(h743_presses):.6f}  (训练 μ=0.7105)')
print(f'  crc_rate      mean={np.mean(h743_crcs):.6f}  (训练 μ=14135.29)')

# 训练数据 window[0..4] (前 5 个 window, 对应 CSV rows 0-79, 跟 H743 一致)
print(f'\n训练数据 window[0..4] 的 27-dim col 19 (press_mean_w) 统计:')
train_press_means = X_train[0:5, :, 19].flatten()
print(f'  press_mean_w  mean={train_press_means.mean():.6f}  std={train_press_means.std():.6f}')

# 关键差异: H743 的 press_mean_w 是 RAW (未归一化), 训练数据 col 19 也是 RAW
# 但 H743 f[17] = press_mean_w 是 RAW, 而训练数据 col 19 也是 RAW
# 所以应该一致
print('\n对比 H743 算的 press_mean_w 跟训练数据 col 19 (window 0):')
print(f'  H743 press_mean_w:   {h743_press_means[:16]}')
print(f'  Train col 19 (win0): {X_train[0, :, 19]}')
print(f'  Train col 19 (win2): {X_train[2, :, 19]}')
