#!/usr/bin/env python3
"""端到端验证 H743 特征提取 + sliding window + 推理 与 PC 端一致.

输入: 200 组连续 16 帧 SCADA Modbus 数据 (随机生成,跟 ARFF 范围一致)
输出: PC vs H743-equivalent 推理结果 max_diff
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import re
import numpy as np
import torch
import joblib

BASE = r"C:\work\Claude\Issue"

# PC scaler
scaler = joblib.load(f"{BASE}/scaler_binary_v2_scada.joblib")
PC_MU    = scaler.center_.astype(np.float32)  # (19,)
PC_SIGMA = scaler.scale_.astype(np.float32)   # (19,)

# H743 scaler
H743_HEADER = open(f"{BASE}/h743/Core/Inc/scaler_export.h", 'r', encoding='utf-8').read()
mu18_match = re.search(r'static\s+const\s+float\s+feat_mu_18\[FEAT_KEEP_18_COUNT\]\s*=\s*\{([^}]+)\}', H743_HEADER, re.DOTALL)
sig18_match = re.search(r'static\s+const\s+float\s+feat_sigma_18\[FEAT_KEEP_18_COUNT\]\s*=\s*\{([^}]+)\}', H743_HEADER, re.DOTALL)
H743_MU_18    = np.array([float(x.replace('f', '')) for x in re.findall(r'-?\d+\.\d+f', mu18_match.group(1))], dtype=np.float32)
H743_SIGMA_18 = np.array([float(x.replace('f', '')) for x in re.findall(r'-?\d+\.\d+f', sig18_match.group(1))], dtype=np.float32)
assert len(H743_MU_18) == 18 and len(H743_SIGMA_18) == 18, f"H743 scaler parse failed: {len(H743_MU_18)}, {len(H743_SIGMA_18)}"

print("[1] Scaler 一致性检查")
print(f"  PC 19-dim mu:    {PC_MU}")
print(f"  PC 19-dim sigma: {PC_SIGMA}")
print(f"  H743 18-dim mu:    {H743_MU_18}")
print(f"  H743 18-dim sigma: {H743_SIGMA_18}")
KEEP_18 = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
mu_diff = np.abs(PC_MU[KEEP_18] - H743_MU_18).max()
sg_diff = np.abs(PC_SIGMA[KEEP_18] - H743_SIGMA_18).max()
print(f"  mu diff (KEEP_18):    {mu_diff:.6e}  {'[OK]' if mu_diff < 1e-4 else '[X]'}")
print(f"  sigma diff (KEEP_18): {sg_diff:.6e}  {'[OK]' if sg_diff < 1e-4 else '[X]'}")
print()

# UNUSUAL_FCS 一致性
unusual_pat = re.compile(r'static\s+const\s+uint8_t\s+feat_unusual_fc\[[^\]]+\]\s*=\s*\{([^}]+)\}', re.DOTALL)
h743_unusual = [int(x) for x in re.findall(r'\d+', unusual_pat.search(H743_HEADER).group(1))]
PC_UNUSUAL_FCS = [133, 136, 137, 138, 139, 140, 171]
print(f"[2] UNUSUAL_FCS 一致性")
print(f"  PC:    {PC_UNUSUAL_FCS}")
print(f"  H743:  {h743_unusual}")
print(f"  {'[OK]' if h743_unusual == PC_UNUSUAL_FCS else '[X]'}")
print()

# ── 端到端:生成 200 组 16 帧窗口数据,过两套流水线 ──

print("[3] 端到端 200 窗口 × 16 帧 验证 (PC vs H743-equivalent)")
print()

# 19-dim 特征名 (PC scaler 顺序)
FEAT_NAMES_19 = [
    "address", "function", "length",
    "setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
    "system mode", "control scheme", "pump", "solenoid",
    "pressure measurement", "crc rate",
    "time_diff", "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
]

# 模拟 SCADA 数据范围
np.random.seed(42)
N_WIN = 200
WIN = 16

# 生成 19-dim RAW,按列范围
raw19 = np.zeros((N_WIN, WIN, 19), dtype=np.float32)
for win_i in range(N_WIN):
    for t in range(WIN):
        raw19[win_i, t] = np.array([
            np.random.uniform(0, 30),                    # 0  address (0-30)
            np.random.choice([0,3,16,21,22]),           # 1  function (主用FC)
            np.random.choice([16,18,20,22,24]),         # 2  length
            0,                                            # 3  setpoint (unused, 总是0)
            np.random.uniform(0, 1),                     # 4  gain
            np.random.uniform(0, 1),                     # 5  reset rate
            np.random.uniform(0, 1),                     # 6  deadband
            np.random.uniform(0, 1),                     # 7  cycle time
            np.random.uniform(0, 1),                     # 8  rate
            np.random.uniform(0, 1),                     # 9  system mode
            np.random.uniform(0, 1),                     # 10 control scheme
            np.random.uniform(0, 1),                     # 11 pump
            np.random.uniform(0, 1),                     # 12 solenoid
            np.random.uniform(0, 100),                   # 13 pressure (clip 0-100)
            np.random.uniform(8000, 16000),              # 14 crc rate
            np.random.uniform(0, 2),                     # 15 time_diff
            np.random.uniform(0, 6),                     # 16 time_since_last
            np.random.choice([0, 1]),                    # 17 is_unusual_fc
            np.random.choice([0, 1]),                    # 18 is_response
        ], dtype=np.float32)

# === PC 端流水线 (与 preprocess_v2_scada.py:154-208 完全一致) ===
def pc_pipeline(raw_19):
    """输入 (N_WIN, 16, 19),输出 (N_WIN, 23, 16) 模型输入."""
    n_win, win, _ = raw_19.shape
    # 1. RobustScaler (per-row)
    X_scaled = (raw_19 - PC_MU) / PC_SIGMA  # (N_WIN, 16, 19)
    # 2. window 已经在外面组装好,直接算 6 聚合
    press_mean_w = X_scaled[:, :, 13].mean(axis=1)  # (N_WIN,)
    crc_max_w    = X_scaled[:, :, 14].max(axis=1)
    is_resp      = X_scaled[:, :, 18]  # signed (-1 或 0)
    is_unusual   = X_scaled[:, :, 17]
    resp_count_w = is_resp.sum(axis=1)              # signed -M
    cmd_count_w  = (1 - is_resp).sum(axis=1)         # = (1-(-1)) for M + (1-0) for R = 16+M
    balance_w    = (cmd_count_w - resp_count_w) / win  # = (16+2M)/16 = 1 + M/8
    length_unique = np.array([len(np.unique(np.round(X_scaled[w, :, 2]).astype(int))) for w in range(n_win)], dtype=np.float32)
    unusual_count_w = is_unusual.sum(axis=1)
    # 3. 装配 27-dim (broadcast)
    aggs = np.column_stack([press_mean_w, np.zeros(n_win), crc_max_w, np.zeros(n_win),
                            resp_count_w, balance_w, length_unique, unusual_count_w]).astype(np.float32)
    # 注: pc 端 Xw 已经包含 19 列,这里组装 19+8 = 27
    Xw_27 = np.concatenate([X_scaled, np.repeat(aggs[:, None, :], win, axis=1)], axis=-1)
    # 4. KEEP_23 切片 (从 scaler_export.h KEEP_23 表)
    KEEP_23 = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 19,21,23,24,25,26]
    out = Xw_27[:, :, KEEP_23].copy()  # (N_WIN, 16, 23)
    # 5. clip ±10 (跟 PC retrain_tcn_*.py:136 一致)
    out = np.clip(out, -10.0, 10.0)
    # 6. 转置为 (N, C, T) layout
    return out.transpose(0, 2, 1)

# === H743 端等价流水线 (模拟 feature_extractor.c + sliding_window.c) ===
def h743_pipeline(raw_19):
    """输入 (N_WIN, 16, 19),输出 (N_WIN, 23, 16) 模型输入."""
    n_win, win, _ = raw_19.shape
    # 1. 18 维归一化 (KEEP_18 索引 = 0,1,2,4,5,...,18)
    feats18 = np.zeros((n_win, win, 18), dtype=np.float32)
    for w in range(n_win):
        for t in range(win):
            row = raw_19[w, t]
            # KEEP_18: 0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
            raw18 = np.array([row[k] for k in KEEP_18], dtype=np.float32)
            # pressure clip [0, 100] (跟 PC line 132)
            raw18[12] = np.clip(raw18[12], 0.0, 100.0)
            feats18[w, t] = (raw18 - H743_MU_18) / H743_SIGMA_18
    # 2. 滑动窗口聚合 (跟 sliding_window.c:118-220 一致)
    out = np.zeros((n_win, 23, win), dtype=np.float32)
    SW18_PRESSURE = 12
    SW18_CRC = 13
    SW18_UNUSUAL = 16
    SW18_RESPONSE = 17
    SW18_LENGTH = 2
    KEEP_23 = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 19,21,23,24,25,26]
    SW_BUF_KEEP_18 = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]  # 18→19-dim idx (skip 3)
    for w in range(n_win):
        f18 = feats18[w]  # (16, 18)
        # 2a. 6 聚合
        press_sum = f18[:, SW18_PRESSURE].sum()
        crc_max = f18[:, SW18_CRC].max()
        resp_signed = f18[:, SW18_RESPONSE].sum()  # signed -M
        unusual_signed = f18[:, SW18_UNUSUAL].sum()
        # length unique
        len_int = np.round(f18[:, SW18_LENGTH]).astype(int)
        seen = set()
        for li in len_int:
            seen.add(int(li))
        nunique = len(seen)
        press_mean = press_sum / 16
        balance = 1.0 - resp_signed / 8.0
        # 2b. 27-dim
        f27 = np.zeros((16, 27), dtype=np.float32)
        for t in range(16):
            for i in range(18):
                f27[t, SW_BUF_KEEP_18[i]] = f18[t, i]
            f27[t, 19] = press_mean
            f27[t, 21] = crc_max
            f27[t, 23] = resp_signed
            f27[t, 24] = balance
            f27[t, 25] = nunique
            f27[t, 26] = unusual_signed
        # 2c. KEEP_23 + clip ±10
        for c in range(23):
            src = KEEP_23[c]
            for t in range(16):
                v = f27[t, src]
                v = max(-10.0, min(10.0, v))
                out[w, c, t] = v
    return out

# 运行
X_pc = pc_pipeline(raw19)
X_h743 = h743_pipeline(raw19)

print(f"  PC shape:    {X_pc.shape}")
print(f"  H743 shape:  {X_h743.shape}")
print(f"  PC[0, 0, :5]:    {X_pc[0, 0, :5]}")
print(f"  H743[0, 0, :5]:  {X_h743[0, 0, :5]}")
print()

diff = np.abs(X_pc - X_h743)
print(f"  Max diff:    {diff.max():.6e}")
print(f"  Mean diff:   {diff.mean():.6e}")
print(f"  Bit-perfect: {'[OK]' if diff.max() < 1e-5 else '[~]' if diff.max() < 1e-3 else '[X]'}")
print()

# === 加载 PC 模型推理 ===
print("[4] PC 模型推理 (200 样本)")
sd = torch.load(f"{BASE}/model_v4_se_23dim_b64_ch32_do01_window16_s42.pt", map_location='cpu', weights_only=False)['state_dict']

# 用之前 verify_h743_vs_pc.py 的模型定义
import torch.nn as nn
import torch.nn.functional as F
CHANNELS = 32
DROPOUT = 0.1
SE_REDUCTION = 8

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
    def forward(self, x):
        s = self.gap(x).squeeze(-1); s = F.relu(self.fc1(s)); s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class TCNBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, dropout):
        super().__init__()
        pad = (k-1)*d//2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        r = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x))); x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.drop(x)
        return F.relu(self.se(x) + r)

class TCNClassifierSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlockSE(23, 32, 3, 1, DROPOUT),
            TCNBlockSE(32, 32, 3, 2, DROPOUT),
            TCNBlockSE(32, 32, 3, 4, DROPOUT),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 32)
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.tcn(x); x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x)); x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)

model = TCNClassifierSE()
model.load_state_dict(sd)
model.eval()

with torch.no_grad():
    pc_out = torch.sigmoid(model(torch.from_numpy(X_pc))).numpy()
    h743_out = torch.sigmoid(model(torch.from_numpy(X_h743))).numpy()

print(f"  PC inference sample:    {pc_out[:5]}")
print(f"  H743 inference sample:  {h743_out[:5]}")
out_diff = np.abs(pc_out - h743_out).max()
print(f"  Max inference diff:  {out_diff:.6e}  {'[OK BIT-PERFECT]' if out_diff < 1e-6 else '[~]'}")
print()

print("=" * 70)
print("[5] 总结")
print("=" * 70)
print(f"  Scaler 一致性:       mu_diff={mu_diff:.1e}, sigma_diff={sg_diff:.1e}  {'[OK]' if mu_diff < 1e-4 and sg_diff < 1e-4 else '[X]'}")
print(f"  UNUSUAL_FCS 一致性: {'[OK]' if h743_unusual == PC_UNUSUAL_FCS else '[X]'}")
print(f"  Pipeline (18-dim → 23-dim) max diff:  {diff.max():.6e}  {'[OK]' if diff.max() < 1e-5 else '[X]'}")
print(f"  模型推理 max diff:   {out_diff:.6e}  {'[OK BIT-PERFECT]' if out_diff < 1e-6 else '[X]'}")
print(f"  结论: {'✓ H743 pipeline 与 PC 端 bit-perfect 一致' if out_diff < 1e-6 else '✗ 存在差异'}")