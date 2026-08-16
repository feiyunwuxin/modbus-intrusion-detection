#!/usr/bin/env python3
"""Dump every named parameter of the trained TCN v4 +SE model with shape and numel.
Re-creates the architecture in_ch=44 (matches saved n_features), then loads state_dict.
No retraining needed: parameter count is deterministic from architecture + in_ch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Architecture (must match tcn_binary_v2_v4se_window16.py with W=16, channels=64) ──
N_FEATURES = 44   # saved n_features
N_BLOCKS = 3
CHANNELS = 64
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
DROPOUT = 0.3
SE_REDUCTION = 8


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class TCNBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, se_reduction=SE_REDUCTION):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=se_reduction)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return F.relu(x + residual)


class TCNClassifierSE(nn.Module):
    def __init__(self, in_ch, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = []
        layers.append(TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout))
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc2(x).squeeze(-1)


# ── Build and load ──
model = TCNClassifierSE(in_ch=N_FEATURES)
ck = torch.load("model_tcn_v4_se_window16.pt", map_location="cpu", weights_only=False)
state = ck["state_dict"]
missing, unexpected = model.load_state_dict(state, strict=True)
print(f"[load] strict=True  missing={missing}  unexpected={unexpected}")
print(f"[ckpt] saved n_params = {ck['n_params']:,}")
print()

# ── Per-parameter dump ──
total = 0
print(f"{'name':<55} {'shape':<25} {'numel':>10}")
print("-" * 95)
for name, p in model.named_parameters():
    n = p.numel()
    total += n
    print(f"{name:<55} {str(list(p.shape)):<25} {n:>10,}")
print("-" * 95)
print(f"{'TOTAL':<55} {'':<25} {total:>10,}")
print()

# ── Per-module dump ──
print("Per-module totals (nn.Module hierarchy):")
print(f"{'module':<55} {'numel':>10}")
print("-" * 70)
for name, m in model.named_modules():
    n = sum(p.numel() for p in m.parameters(recurse=False))
    if n > 0:
        print(f"{name:<55} {n:>10,}")

# ── Per-block SE block detail ──
print()
print("SE block internal params (per block, 3 blocks total):")
for i in range(N_BLOCKS):
    se = model.tcn[i].se
    n_se = sum(p.numel() for p in se.parameters())
    print(f"  block {i}.se: gap=0, fc1={list(se.fc1.weight.shape)}({se.fc1.weight.numel()})"
          f" + fc1.bias={se.fc1.bias.numel()}"
          f" + fc2={list(se.fc2.weight.shape)}({se.fc2.weight.numel()})"
          f" + fc2.bias={se.fc2.bias.numel()}"
          f"  → total {n_se}")
