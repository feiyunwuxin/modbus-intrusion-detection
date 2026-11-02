#!/usr/bin/env python3
"""
analyze_h743_fc43_only_20260928.py
===================================
H743 只检 FC=43 (length=14 attack), 漏 FC=3 (length=46) 和 FC=171 (length=10).

此脚本:
  1. 模拟 H743 的 sliding_window 行为
  2. 对比 FC=43 / FC=3 / FC=171 攻击窗口的 23-dim 输入差异
  3. 跑 PC TCN+SE 模型, 验证哪个 FC 类型的 prob 应该 > 0.49
  4. 输出 H743 应该看到但没看到的攻击窗口
"""
import sys
import os

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "h743", "scripts"))
from verify_sliding_window_23dim import (
    pc_ground_truth_18, py_sliding_window_get, KEEP_18, SW_WINDOW, KEEP_23, UNUSUAL_FCS
)

CSV_PATH = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"
SCALER_PATH = r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib"
MODEL_PATH = r"C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"


class SEBlock(torch.nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Linear(channels, hidden)
        self.fc2 = torch.nn.Linear(hidden, channels)

    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class TCNBlockSE(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, se_reduction=8):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = torch.nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = torch.nn.BatchNorm1d(out_ch)
        self.conv2 = torch.nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = torch.nn.BatchNorm1d(out_ch)
        self.drop = torch.nn.Dropout(dropout)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.residual = torch.nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.dropout(x, p=self.drop.p, training=self.training)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.dropout(x, p=self.drop.p, training=self.training)
        x = self.se(x)
        return torch.relu(x + residual)


class TCNClassifierSE(torch.nn.Module):
    def __init__(self, in_ch=23, n_blocks=3, channels=32, kernel_size=3, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = torch.nn.Sequential(*layers)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Linear(channels, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def main():
    df = pd.read_csv(CSV_PATH, nrows=765)
    raw18 = pc_ground_truth_18(df)
    scaler = joblib.load(SCALER_PATH)
    mu_18 = scaler.center_[KEEP_18].astype(np.float32)
    sg_18 = scaler.scale_[KEEP_18].astype(np.float32)
    norm18 = (raw18 - mu_18) / sg_18

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
    model.load_state_dict(ckpt)
    model.eval()

    # 算所有 765 帧的 23-dim window + PC prob
    n = len(norm18)
    wins = []
    probs = []
    for t0 in range(SW_WINDOW - 1, n):
        win = py_sliding_window_get(norm18[t0 - SW_WINDOW + 1:t0 + 1])[0]
        wins.append(win)
        with torch.no_grad():
            xt = torch.tensor(win[None], dtype=torch.float32)
            p = model(xt).item()
        probs.append(p)

    print(f"=== Total {len(probs)} windows ===")
    print()

    # 检查每个 attack frame 对应的 window prob
    print("=== 攻击窗口 PC prob vs H743 表现 ===")
    print(f"{'#':<4} {'frame':<6} {'fc':<4} {'len':<4} {'binary':<7} {'type':<5} {'pc_prob':<8} {'h743?':<8}")
    for i in range(SW_WINDOW - 1, n):
        t0 = i
        win_idx = i - (SW_WINDOW - 1)
        p = probs[win_idx]
        bin_e = int(df["binary result"].iat[t0])
        fc = int(df["function"].iat[t0])
        ln = int(df["length"].iat[t0])
        spfc = int(df["specific result"].iat[t0])
        if bin_e == 1:  # 攻击
            h743_fires = (p >= 0.49)
            h743_str = "FIRE" if h743_fires else "MISS"
            print(f"{win_idx:<4} {t0+1:<6} {fc:<4} {ln:<4} {bin_e:<7} {spfc:<5} {p:<8.3f} {h743_str:<8}")

    print()
    # 按 FC 类型分组的 PC prob 分布
    print("=== 各 FC 类型攻击窗口的 PC prob 分布 ===")
    fc_probs = {}
    for i in range(SW_WINDOW - 1, n):
        t0 = i
        win_idx = i - (SW_WINDOW - 1)
        p = probs[win_idx]
        bin_e = int(df["binary result"].iat[t0])
        fc = int(df["function"].iat[t0])
        if bin_e == 1:
            fc_probs.setdefault(fc, []).append((p, t0 + 1))
    for fc in sorted(fc_probs.keys()):
        ps = fc_probs[fc]
        n_fire = sum(1 for p, _ in ps if p >= 0.49)
        print(f"  FC={fc:3d}: n_attack_windows={len(ps):3d}, n_fire(>={0.49})={n_fire:3d}, "
              f"prob range=[{min(p for p, _ in ps):.3f}, {max(p for p, _ in ps):.3f}]")

    print()
    # 重点: FC=43 vs FC=3 攻击窗口的特征对比 (frame 63 vs frame 50)
    print("=== 特征对比: FC=43 攻击窗口 (frame 63) vs FC=3 攻击窗口 (frame 50) ===")
    t0_43 = 62  # frame 63 (0-based 62)
    t0_3 = 49   # frame 50 (0-based 49)
    win_43 = wins[t0_43 - (SW_WINDOW - 1)]
    win_3 = wins[t0_3 - (SW_WINDOW - 1)]
    print(f"{'c':<3} {'cname':<22} {'win43_min':<10} {'win43_max':<10} {'win3_min':<10} {'win3_max':<10} {'max_diff':<10}")
    cname = ["slave", "func", "gain", "reset", "deadband", "cycle", "rate", "sysmode",
             "ctrl", "pump", "solenoid", "pressure", "crc_rate", "time_diff",
             "time_since", "is_unusual", "is_resp",
             "press_mean", "crc_max", "resp_signed", "balance", "nunique", "unusual"]
    for c in range(23):
        diff = float(np.abs(win_43[c] - win_3[c]).max())
        print(f"{c:<3} {cname[c]:<22} {win_43[c].min():<10.3f} {win_43[c].max():<10.3f} "
              f"{win_3[c].min():<10.3f} {win_3[c].max():<10.3f} {diff:<10.3f}")


if __name__ == "__main__":
    main()