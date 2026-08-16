#!/usr/bin/env python3
"""
实际保存 3 个 FP32 .pt 文件, 测出真实的磁盘大小 (含 pickle 元数据)
与 Hybrid .bin 大小做严格对比.
"""
import os, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE = r"C:\work\Claude\Issue"
WINDOW = 16
DROPOUT = 0.3
SE_REDUCTION = 8
DILATIONS = [1, 2, 4]


class SEBlock(nn.Module):
    def __init__(self, c, r=SE_REDUCTION):
        super().__init__()
        h = max(c // r, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c, h); self.fc2 = nn.Linear(h, c)
    def forward(self, x):
        s = F.relu(self.fc1(self.gap(x).squeeze(-1)))
        return x * torch.sigmoid(self.fc2(s)).unsqueeze(-1)


class TCNBlockSE(nn.Module):
    def __init__(self, ic, oc, k, d):
        super().__init__()
        p = (k-1)*d//2
        self.conv1 = nn.Conv1d(ic, oc, k, padding=p, dilation=d); self.bn1 = nn.BatchNorm1d(oc)
        self.conv2 = nn.Conv1d(oc, oc, k, padding=p, dilation=d); self.bn2 = nn.BatchNorm1d(oc)
        self.drop = nn.Dropout(DROPOUT); self.se = SEBlock(oc)
        self.residual = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        r = self.residual(x)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        return F.relu(self.se(x) + r)


class TCNClassifierSE(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, 3, DILATIONS[0])]
        for d in DILATIONS[1:3]:
            layers.append(TCNBlockSE(channels, channels, 3, d))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32); self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.tcn(x); x = self.gap(x).squeeze(-1)
        return self.fc2(F.dropout(F.relu(self.fc1(x)), 0.3, training=self.training)).squeeze(-1)


print(f"{'ch':>4} │ {'Params':>7} │ {'raw bytes':>10} │ {'.pt 文件':>10} │ {'Hybrid .bin':>12} │ {'节省':>6}")
print("─" * 75)

for ch in [8, 16, 32]:
    model = TCNClassifierSE(in_ch=19, channels=ch)
    n_params = sum(p.numel() for p in model.parameters())
    raw_bytes = n_params * 4  # FP32 = 4 bytes/param

    # 保存 FP32 .pt (用 torch.save 标准方式)
    pt_path = os.path.join(BASE, f"model_ch{ch}_fp32.pt")
    torch.save({"state_dict": model.state_dict(), "n_params": n_params, "channels": ch}, pt_path)
    pt_size = os.path.getsize(pt_path)

    # Hybrid .bin 大小 (从已有文件读)
    bin_path = os.path.join(BASE, f"model_ch{ch}_hybrid.bin")
    bin_size = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0

    saving = (1 - bin_size / pt_size) * 100 if bin_size else 0
    print(f"{ch:>4} │ {n_params:>7,} │ {raw_bytes:>9,} B │ {pt_size:>9,} B │ {bin_size:>11,} B │ {saving:>5.1f}%")
    print(f"     │         │ ({raw_bytes/1024:>6.2f} KB) │ ({pt_size/1024:>6.2f} KB) │ ({bin_size/1024:>6.2f} KB) │")

    # 删掉测试用 .pt, 不污染目录
    os.remove(pt_path)

print()
print("=" * 75)
print("对比说明:")
print("  - 'raw bytes' = n_params × 4 (纯 FP32 权重字节, 不含任何元数据)")
print("  - '.pt 文件'  = torch.save 实际磁盘大小 (含 PyTorch pickle 元数据)")
print("  - 'Hybrid .bin' = 混合量化后 .bin 文件大小 (含 INT8 + scale + FP32 bias)")
