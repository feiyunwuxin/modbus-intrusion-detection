#!/usr/bin/env python3
"""分析3个噪声特征的统计特性:分布/方差/与攻击label的相关性/特征间冗余度"""

import numpy as np
import os
import json

BASE = r"C:\work\Claude\Issue"

FEATURE_NAMES = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]

# 加载训练+测试数据
Xtr = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)
ytr = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
Xte = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
yte = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)

X = np.concatenate([Xtr, Xte], axis=0)
y = np.concatenate([ytr, yte], axis=0)
print(f"Data shape: X={X.shape}, y={y.shape}, attack_rate={y.mean():.4f}")

# 27 维
FEATURES_TO_ANALYZE = [3, 6, 16, 20]  # setpoint, deadband, time_since, crc_mean_w
NOISE_IDX = [3, 6, 16, 20]

print("\n=== 1. 全局统计 (Train+Test) ===")
print(f"{'idx':>3} {'feature':<35} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'n_unique':>10} {'zero%':>8}")
for i in NOISE_IDX:
    v = X[:, :, i].flatten()
    n_unique = len(np.unique(v))
    zero_pct = (v == 0).mean() * 100
    print(f"{i:>3} {FEATURE_NAMES[i]:<35} {v.mean():>10.4f} {v.std():>10.4f} {v.min():>10.4f} {v.max():>10.4f} {n_unique:>10d} {zero_pct:>7.2f}%")

print("\n=== 2. 攻击 vs 正常 分布对比 (Train) ===")
print(f"{'idx':>3} {'feature':<35} {'normal_mean':>12} {'normal_std':>12} {'attack_mean':>12} {'attack_std':>12} {'KS':>8}")
for i in NOISE_IDX:
    n_mean = Xtr[ytr==0, :, i].mean()
    n_std = Xtr[ytr==0, :, i].std()
    a_mean = Xtr[ytr==1, :, i].mean()
    a_std = Xtr[ytr==1, :, i].std()
    # 简化的分布差异
    diff = abs(a_mean - n_mean) / (n_std + 1e-8)
    print(f"{i:>3} {FEATURE_NAMES[i]:<35} {n_mean:>12.4f} {n_std:>12.4f} {a_mean:>12.4f} {a_std:>12.4f} {diff:>8.2f}")

print("\n=== 3. 与pressure_measurement(idx 13)的相关性 (Spearman) ===")
print(f"{'idx':>3} {'feature':<35} {'spearman_corr':>15}")
for i in NOISE_IDX:
    a = X[:, :, i].flatten()
    b = X[:, :, 13].flatten()
    # 取sample
    if len(a) > 100000:
        idx_s = np.random.RandomState(42).choice(len(a), 100000, replace=False)
        a = a[idx_s]
        b = b[idx_s]
    corr = np.corrcoef(a, b)[0, 1]
    print(f"{i:>3} {FEATURE_NAMES[i]:<35} {corr:>15.4f}")

print("\n=== 4. 冗余特征相关性 (setpoint-deadband, crc_mean_w-crc_max_w) ===")
def corr(a, b):
    if len(a) > 100000:
        idx_s = np.random.RandomState(42).choice(len(a), 100000, replace=False)
        return np.corrcoef(a[idx_s], b[idx_s])[0, 1]
    return np.corrcoef(a, b)[0, 1]

pairs = [
    (3, 6, "setpoint vs deadband"),
    (3, 13, "setpoint vs pressure_measurement"),
    (6, 13, "deadband vs pressure_measurement"),
    (20, 21, "crc_mean_w vs crc_max_w"),
    (20, 14, "crc_mean_w vs crc_rate"),
    (16, 1, "time_since vs function"),
    (16, 17, "time_since vs is_unusual_fc"),
]
for i, j, desc in pairs:
    c = corr(X[:, :, i].flatten(), X[:, :, j].flatten())
    print(f"  {desc:<40} corr={c:>+.4f}")

print("\n=== 5. 时间步内的方差 (setpoint/deadband 是否几乎不变) ===")
print(f"{'idx':>3} {'feature':<35} {'step_within_std_mean':>20} {'step_within_std_max':>20}")
for i in NOISE_IDX:
    # 每个样本在时间步上的标准差,然后求平均
    per_sample_std = X[:, :, i].std(axis=1)
    print(f"{i:>3} {FEATURE_NAMES[i]:<35} {per_sample_std.mean():>20.4f} {per_sample_std.max():>20.4f}")

print("\n=== 6. 与 label 的 Spearman 相关性 (Train) ===")
print(f"{'idx':>3} {'feature':<35} {'corr_with_label':>15}")
for i in NOISE_IDX:
    # 取每个样本的最后一时间步 (或者窗口聚合)
    feat = Xtr[:, :, i].mean(axis=1)  # 窗口聚合
    lbl = ytr.astype(float)
    corr_val = np.corrcoef(feat, lbl)[0, 1]
    print(f"{i:>3} {FEATURE_NAMES[i]:<35} {corr_val:>15.4f}")

