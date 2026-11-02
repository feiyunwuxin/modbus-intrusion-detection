#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full INT8 PTQ (per-tensor activation) for STM32H743 deployment.

Forks quantize_v4se_23dim_ch32_to_h.py (Hybrid INT8) and adds:
  - Activation INT8 per-tensor asymmetric (NEW, with calibration)
  - Bias INT32 per-channel (aligned with input_scale * weight_scale)
  - Requant multiplier/shift per output channel (CMSIS-NN style)
  - fc1/fc2 weights INT8 per-channel + INT32 bias (so CMSIS-NN
    arm_fully_connected_s8 can be called)

FP32 escape hatches (handled in H743 C code by explicit dequant/quant):
  - SE fc2 (4->32, 128 MACs) - FP32 dot is cheaper
  - SE sigmoid - no INT8 kernel
  - Classifier fc1 + fc2 - FP32 escape
  - Final sigmoid - FP32

Outputs:
  - h743/Core/Inc/model_v4_se_23dim_ch32_fullint8_s{seed}.h x 5
  - PTQ log: fullint8_ptq_seed{seed}.json

Usage:
    python quantize_v4se_23dim_ch32_full_int8.py
"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
TAG_PT = "v4_se_23dim_b64_ch32_do01_window16"
TAG_H  = "v4_se_23dim_ch32_fullint8"
N_FEATURES = 23
WINDOW = 16
CHANNELS = 32
DROPOUT = 0.1
SE_REDUCTION = 8
N_BLOCKS = 3
DILATIONS = [1, 2, 4]
KERNEL_SIZE = 3

# 训练数据路径
X_TRAIN_NPY = os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")
SCALER_PATH = os.path.join(BASE, "scaler_binary_v2_scada.joblib")

# 输出路径
HEADER_DIR = os.path.join(BASE, "h743", "Core", "Inc")
LOG_DIR = BASE  # fullint8_ptq_seed{seed}.json 写到这里


# ═════════════════════════════════════════════════════════════════════
# 模型定义 (与 retrain 脚本一致,确保 load_state_dict 兼容)
# ═════════════════════════════════════════════════════════════════════

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
    """Standard TCN+SE classifier (matches retrain_tcn_23dim_*.py)."""

    def __init__(self, in_ch, channels=CHANNELS, n_blocks=N_BLOCKS, kernel_size=KERNEL_SIZE,
                 dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════
# 量化函数 (per-channel symmetric INT8 + per-tensor asymmetric)
# ═════════════════════════════════════════════════════════════════════

def quantize_per_channel_symmetric(w: np.ndarray):
    """Per-channel 对称 INT8.  形状 (out_ch, ...).  返回 (w_int8, scale_per_ch)."""
    assert w.ndim >= 2, f"weight must be 2D+, got {w.ndim}D"
    out_ch = w.shape[0]
    w_flat = w.reshape(out_ch, -1)
    absmax = np.max(np.abs(w_flat), axis=1)
    absmax = np.maximum(absmax, 1e-8)
    scale = absmax / 127.0
    w_q = np.clip(np.round(w_flat / scale[:, None]), -128, 127).astype(np.int8)
    return w_q.reshape(w.shape), scale.astype(np.float32)


def quantize_per_tensor_asymmetric(min_val, max_val):
    """Per-tensor 非对称 INT8.  返回 (scale, zero_point).

    scale = (max - min) / 255
    zp = round(-min / scale), clamp [-128, 127]
    若 max == min → (1.0, 0)
    """
    rng = max_val - min_val
    if rng < 1e-12:
        return 1.0, 0
    scale = rng / 255.0
    zp = int(np.clip(np.round(-min_val / scale), -128, 127))
    return float(scale), int(zp)


def quantize_per_tensor_symmetric(min_val, max_val):
    """Per-tensor 对称 INT8 (TFLite default for activations).

    scale = max(|min|, |max|) / 127
    zp = 0 (if min/max symmetric) OR clamped such that int8 covers the range.

    Returns (scale, zp). With zp=0, int8 range [-128, 127] maps to [-128*s, 127*s].
    If pre-ReLU activation is highly asymmetric (e.g. [-10, +0.1]), zp=0 symmetric
    int8 would map to [-10, 9.94] which is BETTER than the asymmetric clip.
    """
    absmax = max(abs(min_val), abs(max_val))
    if absmax < 1e-12:
        return 1.0, 0
    scale = absmax / 127.0
    return float(scale), 0


def compute_int32_bias(fp32_bias, input_scale, weight_scale_per_channel):
    """Per-channel INT32 bias.  b_q32[i] = round(b_fp32[i] / (input_scale * w_scale[i]))

    注意:不是 round-to-nearest-even, 只是 round.  CMSIS-NN 接受任意 int32.
    """
    eff = input_scale * weight_scale_per_channel  # (out_ch,)
    eff = np.maximum(np.abs(eff), 1e-12)  # 防 0
    b_q32 = np.round(fp32_bias / eff).astype(np.int32)
    return b_q32


def compute_requant_multiplier_shift(input_scale, weight_scale_per_channel, output_scale):
    """CMSIS-NN 风格 per-channel Requant M/N. Returns (M, N) per channel.

    Mathematical goal:
        y_q = saturate(((int32_val * M) + (1 << (N-1))) >> N) + out_zp
        where (M * 2^-N) ≈ (input_scale * weight_scale) / output_scale

    Algorithm (CMSIS-NN standard, from arm_nn_requantize_s8.c):
      target = (s_i * s_w) / s_o   (per-channel, FP32, can be tiny)
      Find largest N (>= 0) such that M = round(target * 2^N) fits in int32 [0, 2^31-1].
      Equivalently: N = floor(log2(INT32_MAX / target)), M = round(target * 2^N).
      Edge cases:
        target == 0     -> M = 0, N = 0
        target very tiny -> N = 0,  M = 0 (degenerate)
        target > 1.0    -> N = 0,  M = INT32_MAX (saturated; precision loss but valid)

    Returns (mult_per_ch int32, shift_per_ch int32) each shape (out_ch,).
    Shift is constrained to [0, 31] per CMSIS-NN convention (ARM_MATH_DSP path).
    """
    multiplier_fp = (input_scale * weight_scale_per_channel) / output_scale  # (out_ch,)
    n_ch = multiplier_fp.size
    mult = np.zeros(n_ch, dtype=np.int32)
    shift = np.zeros(n_ch, dtype=np.int32)
    INT32_MAX = (1 << 31) - 1
    for i, m in enumerate(multiplier_fp):
        if m <= 0.0:
            mult[i] = 0
            shift[i] = 0
            continue
        # Find largest N in [0, 31] such that M = round(m * 2^N) fits in int32.
        if m >= 1.0:
            # Saturate: target >= 1 means one channel's requant factor > 1, clamp
            N = 0
            M = INT32_MAX
        else:
            N_f = np.log2(INT32_MAX / m)  # may be < 0 for very large m, >= 31 for very small
            N = int(np.floor(N_f))
            if N > 31:
                N = 31
            elif N < 0:
                N = 0
            M = int(round(m * (1 << N)))
            if M > INT32_MAX:
                M = INT32_MAX
        if M == 0:
            # Edge: rounding produced 0, use 1 to keep channel alive
            M = 1
            N = 0
        mult[i] = np.int32(M)
        shift[i] = np.int32(N)
    return mult, shift


def quantize_tensor_int8_per_tensor(t_fp32, scale, zp):
    """Per-tensor INT8 量化.  返回 (t_int8,)"""
    q = np.round(t_fp32 / scale + zp)
    return np.clip(q, -128, 127).astype(np.int8)


def requantize_int8_per_channel(int32_val, mult, shift, out_zp):
    """Per-channel Requant: int32 → int8.
    y_q = saturate(((val * M) + (1<<(N-1))) >> N) + out_zp
    mult, shift 是 per-channel (out_ch,) 数组.
    """
    # 计算 (1 << (N-1)), 注意 N=0 时 (1 << -1) 不合法, 用 0
    round_bias = np.where(shift > 0, (1 << (shift - 1)).astype(np.int32), 0)
    val_with_bias = int32_val * mult + round_bias
    # 算术右移
    # 防止 shift=0 时退化为 >> 0 (无变化)
    y = np.where(shift > 0, val_with_bias >> shift, val_with_bias)
    y = np.clip(y, -128, 127)
    return (y + out_zp).astype(np.int8)


# ═════════════════════════════════════════════════════════════════════
# BN fold
# ═════════════════════════════════════════════════════════════════════

def fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """把 Conv1d + BN1d 折叠成一个等效 Conv1d (bias=True)."""
    w = conv.weight.data.numpy().copy()  # (out, in, k)
    b = conv.bias.data.numpy().copy() if conv.bias is not None else np.zeros(conv.out_channels)
    bn_w = bn.weight.data.numpy()
    bn_b = bn.bias.data.numpy()
    bn_mean = bn.running_mean.numpy()
    bn_var = bn.running_var.numpy()
    bn_eps = bn.eps
    scale = bn_w / np.sqrt(bn_var + bn_eps)
    folded_w = w * scale[:, None, None]
    folded_b = (b - bn_mean) * scale + bn_b
    return folded_w, folded_b.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════
# 激活范围 calibration (用 forward hook 抓 min/max)
# ═════════════════════════════════════════════════════════════════════

def calibrate_activation_ranges(model, X_calib):
    """对模型跑一遍 calibration, 收集每层激活的 (min, max).

    X_calib: (N, in_ch, T) float32, 已在 scaler 后 + transpose(0,2,1)

    返回 dict: {'<layer_name>': (min, max)} — 包含 input, conv1/conv2, residual (Block 0),
                se.fc1 (ReLU 后), se.fc2 (sigmoid 后), fc1 (ReLU 后), fc2 (sigmoid 后)
    """
    ranges = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out.detach().cpu().numpy()
            mn, mx = float(t.min()), float(t.max())
            if name in ranges:
                old_mn, old_mx = ranges[name]
                ranges[name] = (min(old_mn, mn), max(old_mx, mx))
            else:
                ranges[name] = (mn, mx)
        return hook

    # input: 直接从 X_calib 算
    ranges['input'] = (float(X_calib.min()), float(X_calib.max()))

    # 每一层的 conv (输出 = ReLU 之前)
    for i, block in enumerate(model.tcn):
        handles.append(block.conv1.register_forward_hook(make_hook(f'tcn.{i}.conv1')))
        handles.append(block.conv2.register_forward_hook(make_hook(f'tcn.{i}.conv2')))
        if isinstance(block.residual, nn.Conv1d):
            handles.append(block.residual.register_forward_hook(make_hook(f'tcn.{i}.residual')))
        # SE fc1 输出 = ReLU 后 (用 fc1 抓 + 实际是 ReLU 前)
        handles.append(block.se.fc1.register_forward_hook(make_hook(f'tcn.{i}.se.fc1')))
        # SE fc2 输出 = sigmoid 前
        handles.append(block.se.fc2.register_forward_hook(make_hook(f'tcn.{i}.se.fc2')))

    # Classifier
    handles.append(model.fc1.register_forward_hook(make_hook('fc1')))
    handles.append(model.fc2.register_forward_hook(make_hook('fc2')))

    # 前向传播 (calib batch 用 model 直接 forward)
    model.eval()
    with torch.no_grad():
        # 分批跑避免内存爆
        BATCH = 64
        for i in range(0, len(X_calib), BATCH):
            xb = torch.from_numpy(X_calib[i:i + BATCH])
            _ = model(xb)
            # 中断: hook 已经累计 min/max

    for h in handles:
        h.remove()
    return ranges


# ═════════════════════════════════════════════════════════════════════
# 数据准备 (复用 FT)
# ═════════════════════════════════════════════════════════════════════

def load_calib_data(n_samples=500):
    """Load training windows, apply KEEP_23 selection, clip ±10, transpose to (N, 23, 16).

    retrain_tcn_23dim_b64_ch32_do01_savept.py:130-138:
        X_train = np.load(X_train_binary_v2_scada_window16.npy)[:, :, KEEP_23]
        X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    The .npy file contains 27-dim pre-processed data (19 scaled + 8 aggregates),
    KEEP_23 selects 23 of them. CRITICAL: clip(±CLIP_VAL) is required because
    some aggregates (e.g. crc_max, length_nunique) can reach ±hundreds before clipping.
    Without clip, activation ranges explode and INT8 quant becomes useless.
    """
    import joblib
    KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
               16, 17, 18, 19, 21, 23, 24, 25, 26]
    X_train = np.load(X_TRAIN_NPY).astype(np.float32)[:, :, KEEP_23]
    CLIP_VAL = 10.0  # matches retrain_tcn_23dim_b64_ch32_do01_savept.py CLIP_VAL
    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL)
    scaler = joblib.load(SCALER_PATH)
    # Transpose to (N, 23, 16) for Conv1d
    X_calib = X_train[:n_samples].transpose(0, 2, 1).astype(np.float32)
    return X_calib, scaler


# ═════════════════════════════════════════════════════════════════════
# 完整 INT8 量化 (per seed)
# ═════════════════════════════════════════════════════════════════════

def full_int8_quantize(seed: int):
    """对一个 .pt 跑 Full INT8 PTQ, 返回 state dict 包含:
    - weights_int8 (per-channel) + scales (per-channel)
    - biases_int32 (per-channel)
    - activations_int8: per-tensor (scale, zp) per layer
    - requant: mult/shift per channel per layer
    - weights_fp32_int8 (fc1, fc2 量化后 per-channel) + scales
    - bias_int32_per_tensor: per-tensor bias (input scale only)
    """
    pt_path = os.path.join(BASE, f"model_{TAG_PT}_s{seed}.pt")
    print(f"\n=== seed={seed} 加载 {pt_path} ===")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model = TCNClassifierSE(in_ch=N_FEATURES)
    model.load_state_dict(state_sd)
    model.eval()

    # Calibration
    print("  calib 准备 500 样本 ...")
    X_calib, _ = load_calib_data(n_samples=500)
    print(f"  calib shape={X_calib.shape}")
    print("  跑 forward + hook 抓 min/max ...")
    ranges = calibrate_activation_ranges(model, X_calib)
    print(f"  ranges (层, min, max):")
    for k, (mn, mx) in ranges.items():
        print(f"    {k:20s}  min={mn:+.4f}  max={mx:+.4f}  range={mx - mn:.2f}")

    # 量化参数 (per-tensor SYMMETRIC for activations)
    # 关键设计: 用 SYMMETRIC (zp=0) 而不是 asymmetric.
    #   - 预 ReLU conv 输出 (BN 后) 范围不对称 (e.g. [-3, 0.5]), asymmetric zp=127 让正区间浪费
    #   - 对称 zp=0 让 int8 范围对称覆盖, 减少精度损失
    #   - 输入层可以保留 asymmetric (因为 scaler.transform 后不对称程度较低)
    act_q = {}  # {'<layer>': (scale, zp)}
    for k, (mn, mx) in ranges.items():
        act_q[k] = quantize_per_tensor_symmetric(mn, mx)
    print(f"\n  per-tensor SYMMETRIC activation quant (scale, zp):")
    for k, (s, z) in act_q.items():
        print(f"    {k:20s}  s={s:.6e}  zp={z:4d}")

    # 提取权重 + 折叠 bias
    print("\n  fold BN + per-channel quant 权重 ...")
    state = {
        "weights_int8": {},        # 13 INT8 权重 (per-channel)
        "scales": {},              # 13 per-channel FP32 scales
        "biases_fp32": {},         # 13 折叠后的 FP32 biases (用于算 int32 bias)
        "biases_int32": {},        # 13 per-channel int32 biases (用 input_scale + weight_scale 算)
        "activations": act_q,       # per-tensor scale/zp
        "requant": {},             # per-channel M/shift (用于 conv2 → conv2 输出)
        "weights_int8_fc": {},    # fc1, fc2 量化后的 per-channel int8 权重 + scale + int32 bias
        "biases_int32_fc": {},
        "scales_fc": {},
        "requant_fc": {},
    }

    # 13 个 conv/SE 权重
    layer_meta = [
        # (i, "name", in_ch, out_ch, k, dilation)
        (0, "tcn.0.conv1",   23, 32, 3, 1),
        (0, "tcn.0.conv2",   32, 32, 3, 1),
        (0, "tcn.0.residual", 23, 32, 1, 1),
        (0, "tcn.0.se.fc1",  32, 4,  1, 0),
        (0, "tcn.0.se.fc2",  4,  32, 1, 0),
        (1, "tcn.1.conv1",   32, 32, 3, 2),
        (1, "tcn.1.conv2",   32, 32, 3, 2),
        (1, "tcn.1.se.fc1",  32, 4,  1, 0),
        (1, "tcn.1.se.fc2",  4,  32, 1, 0),
        (2, "tcn.2.conv1",   32, 32, 3, 4),
        (2, "tcn.2.conv2",   32, 32, 3, 4),
        (2, "tcn.2.se.fc1",  32, 4,  1, 0),
        (2, "tcn.2.se.fc2",  4,  32, 1, 0),
    ]

    for bi, name, _, out_ch, _, _ in layer_meta:
        block_idx = int(name.split('.')[1])
        block = model.tcn[block_idx]
        # name = "tcn.<i>.conv1" / "tcn.<i>.conv2" / "tcn.<i>.residual" / "tcn.<i>.se.fc1" / "tcn.<i>.se.fc2"
        tail = name.split('.', 2)[2]  # "conv1" / "conv2" / "residual" / "se.fc1" / "se.fc2"
        if tail == "conv1":
            w_raw, bk_raw = fold_bn(block.conv1, block.bn1)
        elif tail == "conv2":
            w_raw, bk_raw = fold_bn(block.conv2, block.bn2)
        elif tail == "residual":
            assert isinstance(block.residual, nn.Conv1d)
            w_raw = block.residual.weight.data.numpy().copy()
            bk_raw = (block.residual.bias.data.numpy().copy()
                       if block.residual.bias is not None
                       else np.zeros(block.residual.out_channels))
        elif tail == "se.fc1":
            w_raw = block.se.fc1.weight.data.numpy().copy()
            bk_raw = block.se.fc1.bias.data.numpy().copy()
        elif tail == "se.fc2":
            w_raw = block.se.fc2.weight.data.numpy().copy()
            bk_raw = block.se.fc2.bias.data.numpy().copy()
        else:
            raise ValueError(f"unknown layer: {name}")
        w_q, s_w = quantize_per_channel_symmetric(w_raw)
        state["weights_int8"][f"{name}.weight"] = w_q
        state["scales"][f"{name}.weight"] = s_w
        state["biases_fp32"][f"{name}.bias"] = bk_raw.astype(np.float32)

    # fc1 + fc2 (量化, per-channel)
    fc_layers = [
        ("fc1", model.fc1),  # (32 → 32)
        ("fc2", model.fc2),  # (32 → 1)
    ]
    for name, fc in fc_layers:
        w_raw = fc.weight.data.numpy().copy()
        b_raw = fc.bias.data.numpy().copy()
        w_q, s_w = quantize_per_channel_symmetric(w_raw)
        state["weights_int8_fc"][f"{name}.weight"] = w_q
        state["scales_fc"][f"{name}.weight"] = s_w
        state["biases_fp32"][f"{name}.bias"] = b_raw.astype(np.float32)

    # ── bias int32 (per-channel, 依赖该层的 input_scale + weight_scale_per_channel)
    # 注意: 当前 input_scale 是 "激活图" 的输入, 但 int32 bias 算式是:
    #   y_acc = sum(in_q * w_q)  (int32)
    #   b_q32 = round(b_fp32 / (input_scale * weight_scale))
    # 所以 input_scale 必须是 *当前层* 的输入的 scale (即上一层的 output scale).
    # 我们的 ranges 字典 'act' key 是 "输出 = ReLU 之前", 所以:
    #   - tcn.0.conv1  的 input scale = 'input' 的 s
    #   - tcn.0.conv2  的 input scale = tcn.0.conv1 的 s (因为 ReLU 之前是 conv1 输出)
    # 等等.
    # Per-layer input scale lookup:
    def input_scale_of(layer_name: str):
        # layer_name: 'tcn.0.conv1', 'tcn.0.conv2', etc.
        if layer_name == "tcn.0.conv1":
            return act_q["input"][0]
        # conv2 of block i 的输入 = block{i-1 的 conv2 输出 + residual + ReLU}
        # 简化为上一层的 conv2
        if "conv2" in layer_name:
            # Block i conv2 的输入 = Block i conv1 输出
            return act_q[f"tcn.{layer_name.split('.')[1]}.conv1"][0]
        if "conv1" in layer_name and layer_name != "tcn.0.conv1":
            # Block i conv1 的输入 = Block i-1 输出
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i - 1}.conv2"][0]
        if "residual" in layer_name:
            return act_q["input"][0]
        if "se.fc1" in layer_name:
            # SE fc1 输入 = GAP 输出 (int8). 暂时用 conv2 输出 scale 近似.
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i}.conv2"][0]
        if "se.fc2" in layer_name:
            # SE fc2 输入 = ReLU(fc1) 输出 (int8).  用 fc1 的 output scale 近似
            return act_q[f"tcn.{layer_name.split('.')[1]}.se.fc1"][0]
        if layer_name == "fc1":
            return act_q["tcn.2.conv2"][0]  # 末层 GAP 后的
        if layer_name == "fc2":
            return act_q["fc1"][0]
        raise ValueError(f"unknown layer: {layer_name}")

    # ── 计算 int32 bias + requant mult/shift
    # 每层的 output 范围取下一层的 input (即当前层 output zp/scale 等于下一层 input zp/scale)
    def output_scale_of(layer_name: str):
        # 输出 scale 等于下一层的 input scale
        # ... (在 requant 里直接用下一层 input_scale)
        # 但 Requant 需要把当前层 int32 输出映射到下一层 int8 范围
        # 所以 out_scale 是"下一层输入 scale"
        # 简化: conv1 的 out = conv2 的 in, conv2 的 out = SE GAP 的 in (≈conv2 in), etc.
        if layer_name == "tcn.0.conv1":
            return act_q["tcn.0.conv1"][0]   # 这是 conv1 输出范围 (calib 里抓的)
        if "conv1" in layer_name and layer_name != "tcn.0.conv1":
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i}.conv1"][0]
        if "conv2" in layer_name:
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i}.conv2"][0]
        if "residual" in layer_name:
            return act_q[f"tcn.0.conv1"][0]   # residual 跟 conv1 同步
        if "se.fc1" in layer_name:
            return act_q[f"tcn.{layer_name.split('.')[1]}.se.fc1"][0]
        if "se.fc2" in layer_name:
            # SE fc2 输出 → sigmoid → FP32 escape (不 requant)
            return act_q[f"tcn.{layer_name.split('.')[1]}.se.fc2"][0]
        if layer_name == "fc1":
            return act_q["fc1"][0]
        if layer_name == "fc2":
            return act_q["fc2"][0]
        raise ValueError(f"unknown layer: {layer_name}")

    # 13 个 conv/SE 层 + fc1 + fc2 = 15 个 per-channel int32 bias + requant mult/shift
    all_layer_names = [name for _, name, _, _, _, _ in layer_meta] + ["fc1", "fc2"]
    for name in all_layer_names:
        if name in {"fc1", "fc2"}:
            w_q = state["weights_int8_fc"][f"{name}.weight"]
            s_w = state["scales_fc"][f"{name}.weight"]
        else:
            w_q = state["weights_int8"][f"{name}.weight"]
            s_w = state["scales"][f"{name}.weight"]
        b_fp32 = state["biases_fp32"][f"{name}.bias"]
        s_in = input_scale_of(name)
        s_out = output_scale_of(name)

        b_q32 = compute_int32_bias(b_fp32, s_in, s_w)
        mult, shift = compute_requant_multiplier_shift(s_in, s_w, s_out)

        if name in {"fc1", "fc2"}:
            state["biases_int32_fc"][f"{name}.bias"] = b_q32
            state["requant_fc"][f"{name}.weight"] = (mult, shift)
        else:
            state["biases_int32"][f"{name}.bias"] = b_q32
            state["requant"][f"{name}.weight"] = (mult, shift)

    # Sanity dump
    summary = {
        "seed": seed,
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "act_q": {k: [float(v[0]), int(v[1])] for k, v in act_q.items()},
        "layer_sizes": {
            "weights_int8": {k: int(v.size) for k, v in state["weights_int8"].items()},
            "weights_int8_fc": {k: int(v.size) for k, v in state["weights_int8_fc"].items()},
        },
    }
    return state, summary


# ═════════════════════════════════════════════════════════════════════
# C header writer
# ═════════════════════════════════════════════════════════════════════

def gen_int8_array_literal(name, arr):
    flat = arr.flatten()
    n = flat.size
    out = [f"static const int8_t {name}[{n}] = {{"]
    row = []
    for i, v in enumerate(flat):
        row.append(f"{int(v):4d}")
        if (i + 1) % 12 == 0 or i == n - 1:
            out.append("    " + ", ".join(row) + ("," if i != n - 1 else ""))
            row = []
    out.append("};")
    return "\n".join(out)


def gen_int32_array_literal(name, arr):
    flat = arr.flatten()
    n = flat.size
    out = [f"static const int32_t {name}[{n}] = {{"]
    row = []
    for i, v in enumerate(flat):
        row.append(f"{int(v):8d}")
        if (i + 1) % 8 == 0 or i == n - 1:
            out.append("    " + ", ".join(row) + ("," if i != n - 1 else ""))
            row = []
    out.append("};")
    return "\n".join(out)


def gen_float_array_literal(name, arr):
    flat = arr.flatten()
    n = flat.size
    out = [f"static const float {name}[{n}] = {{"]
    row = []
    for i, v in enumerate(flat):
        row.append(f"{float(v):.6e}f")
        if (i + 1) % 8 == 0 or i == n - 1:
            out.append("    " + ", ".join(row) + ("," if i != n - 1 else ""))
            row = []
    out.append("};")
    return "\n".join(out)


def write_full_int8_header(state, seed, output_path):
    """生成 Full INT8 C header."""
    safe = lambda k: k.replace(".", "_").replace(",", "_")
    lines = []
    lines.append(f"/* Auto-generated by quantize_v4se_23dim_ch32_full_int8.py")
    lines.append(f" * TCN+SE ch=32 (23-dim SCADA, dropout=0.1) - Full INT8 (per-tensor act + per-ch weight + int32 bias)")
    lines.append(f" * seed={seed} - 与 hybrid 的区别: activations/requant 全 INT8, fc1/fc2 也量化")
    lines.append(f" * H743 C 代码路径: ch32_inference_full_int8.c + arm_convolve_s8/fully_connected_s8 (CMSIS-NN)")
    lines.append(f" *")
    lines.append(f" * FP32 escape (在 C 推理里显式 dequant/quant):")
    lines.append(f" *   - SE fc2  (4→32): 128 MACs, FP32 dot 更便宜")
    lines.append(f" *   - SE sigmoid + scale: 无 INT8 实现")
    lines.append(f" *   - Classifier fc1 + fc2 + sigmoid: FP32 escape")
    lines.append(f" */")
    lines.append(f"#ifndef TCN_V4_SE_23DIM_CH32_FULLINT8_S{seed}_WEIGHTS_H")
    lines.append(f"#define TCN_V4_SE_23DIM_CH32_FULLINT8_S{seed}_WEIGHTS_H")
    lines.append(f"#include <stdint.h>")
    lines.append(f"#ifdef __cplusplus")
    lines.append(f'extern "C" {{')
    lines.append(f"#endif")
    lines.append(f"")

    act_q = state["activations"]

    # 计算汇总
    total_params = sum(v.size for v in state["weights_int8"].values()) + sum(v.size for v in state["weights_int8_fc"].values())
    total_bytes_int8 = sum(v.size for v in state["weights_int8"].values()) + sum(v.size for v in state["weights_int8_fc"].values())
    total_bytes_int32_bias = sum(v.size for v in state["biases_int32"].values()) + sum(v.size for v in state["biases_int32_fc"].values())
    total_bytes_int32_requant = sum(v[0].size + v[1].size for v in state["requant"].values()) + sum(v[0].size + v[1].size for v in state["requant_fc"].values())
    total_bytes_fp32_scales = sum(v.size for v in state["scales"].values()) + sum(v.size for v in state["scales_fc"].values()) * 4
    total_bytes_fp32_act = len(act_q) * 2 * 4  # 每个 act 有 1 个 float scale + 1 个 int32 zp, 但 zp 是 int32

    lines.append(f"#define TCN_V4_SE_23DIM_CH32_FULLINT8_S{seed}_N_QUANT_TENSORS 15")
    lines.append(f"#define TCN_V4_SE_23DIM_CH32_FULLINT8_S{seed}_TOTAL_PARAMS    {total_params}")
    lines.append(f"/* Flash ≈ {total_bytes_int8/1024:.1f}KB (int8 weights) + "
                 f"{total_bytes_int32_bias/1024:.1f}KB (int32 bias) + "
                 f"{total_bytes_int32_requant/1024:.1f}KB (int32 requant M/S) + "
                 f"{total_bytes_fp32_scales/1024:.1f}KB (FP32 weight scales) */")
    lines.append(f"")

    # Per-layer 顺序
    layer_order = [
        "tcn.0.conv1", "tcn.0.conv2", "tcn.0.residual",
        "tcn.0.se.fc1", "tcn.0.se.fc2",
        "tcn.1.conv1", "tcn.1.conv2",
        "tcn.1.se.fc1", "tcn.1.se.fc2",
        "tcn.2.conv1", "tcn.2.conv2",
        "tcn.2.se.fc1", "tcn.2.se.fc2",
        "fc1", "fc2",
    ]

    # 1) 每层: per-channel int8 weights
    for name in layer_order:
        if name in {"fc1", "fc2"}:
            w = state["weights_int8_fc"][f"{name}.weight"]
        else:
            w = state["weights_int8"][f"{name}.weight"]
        lines.append(f"/* {name}.weight  shape={list(w.shape)}  numel={w.size} */")
        lines.append(gen_int8_array_literal(f"v4se23_ch32_fullint8_s{seed}_w_{safe(name)}", w))
        lines.append("")

    # 2) 每层: per-channel int32 bias
    for name in layer_order:
        if name in {"fc1", "fc2"}:
            b = state["biases_int32_fc"][f"{name}.bias"]
        else:
            b = state["biases_int32"][f"{name}.bias"]
        lines.append(f"/* {name}.bias  numel={b.size}  int32 (per-channel) */")
        lines.append(gen_int32_array_literal(f"v4se23_ch32_fullint8_s{seed}_b_{safe(name)}", b))
        lines.append("")

    # 3) 每层: per-channel FP32 weight scale
    for name in layer_order:
        if name in {"fc1", "fc2"}:
            s_w = state["scales_fc"][f"{name}.weight"]
        else:
            s_w = state["scales"][f"{name}.weight"]
        lines.append(f"/* {name} weight scale  numel={s_w.size} */")
        lines.append(gen_float_array_literal(f"v4se23_ch32_fullint8_s{seed}_sw_{safe(name)}", s_w))
        lines.append("")

    # 4) Per-tensor activation scale + zp per layer
    lines.append(f"// ===== Per-tensor activation quantization (scale + zp) =====")
    for name in layer_order:
        # 用上一层的 output range (= current activation input scale)
        # 简化: 每层都列出它接收的输入 scale/zp
        if name == "tcn.0.conv1":
            s, zp = act_q["input"]
        elif "conv1" in name and name != "tcn.0.conv1":
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk-1}.conv2"]
        elif "conv2" in name:
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.conv1"]
        elif "residual" in name:
            s, zp = act_q["input"]
        elif "se.fc1" in name:
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.conv2"]  # GAP 输出 (近似)
        elif "se.fc2" in name:
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.se.fc1"]  # ReLU 输出
        elif name == "fc1":
            s, zp = act_q["tcn.2.conv2"]
        elif name == "fc2":
            s, zp = act_q["fc1"]
        lines.append(f"static const float v4se23_ch32_fullint8_s{seed}_si_{safe(name)}[1] = {{ {s:.6e}f }};")
        lines.append(f"static const int32_t v4se23_ch32_fullint8_s{seed}_zpi_{safe(name)}[1] = {{ {zp} }};")
    lines.append("")

    # 5) Per-tensor output scale + zp per layer (用于下一层 requant)
    lines.append(f"// ===== Per-tensor output quantization (用于下一层) =====")
    for name in layer_order:
        if name == "tcn.0.conv1":
            s, zp = act_q["tcn.0.conv1"]
        elif "conv1" in name and name != "tcn.0.conv1":
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.conv1"]
        elif "conv2" in name:
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.conv2"]
        elif "residual" in name:
            s, zp = act_q["tcn.0.conv1"]
        elif "se.fc1" in name:
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.se.fc1"]
        elif "se.fc2" in name:
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.se.fc2"]
        elif name == "fc1":
            s, zp = act_q["fc1"]
        elif name == "fc2":
            s, zp = act_q["fc2"]
        lines.append(f"static const float v4se23_ch32_fullint8_s{seed}_so_{safe(name)}[1] = {{ {s:.6e}f }};")
        lines.append(f"static const int32_t v4se23_ch32_fullint8_s{seed}_zpo_{safe(name)}[1] = {{ {zp} }};")
    lines.append("")

    # 6) Per-channel Requant M/S
    lines.append(f"// ===== Per-channel requant multiplier + shift =====")
    for name in layer_order:
        if name in {"fc1", "fc2"}:
            mult, shift = state["requant_fc"][f"{name}.weight"]
        else:
            mult, shift = state["requant"][f"{name}.weight"]
        lines.append(f"/* {name} requant M/S  numel={mult.size} */")
        lines.append(gen_int32_array_literal(f"v4se23_ch32_fullint8_s{seed}_mult_{safe(name)}", mult))
        lines.append(gen_int32_array_literal(f"v4se23_ch32_fullint8_s{seed}_shift_{safe(name)}", shift))
        lines.append("")

    lines.append(f"#ifdef __cplusplus")
    lines.append(f"}}")
    lines.append(f"#endif")
    lines.append(f"#endif")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return total_params


# ═════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(" Full INT8 PTQ (per-tensor act + per-ch weight + int32 bias + requant M/S)")
    print("=" * 70)

    os.makedirs(HEADER_DIR, exist_ok=True)
    t_start = time.time()

    summaries = {}
    for seed in SEEDS:
        t0 = time.time()
        state, summary = full_int8_quantize(seed)
        header_path = os.path.join(HEADER_DIR, f"model_v4_se_23dim_ch32_fullint8_s{seed}.h")
        n = write_full_int8_header(state, seed, header_path)
        size_kb = os.path.getsize(header_path) / 1024
        print(f"\n  -> Wrote {header_path}")
        print(f"     {n} params, {size_kb:.1f} KB")
        summaries[seed] = summary
        print(f"  seed={seed} took {time.time() - t0:.1f}s")

        # Save PTQ log
        log_path = os.path.join(LOG_DIR, f"fullint8_ptq_seed{seed}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  -> Wrote {log_path}")

    # Combined log
    combined_path = os.path.join(LOG_DIR, "fullint8_ptq_all_seeds.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n-> Wrote {combined_path}")

    print(f"\n[OK] Full INT8 PTQ done for {len(SEEDS)} seeds in {time.time() - t_start:.1f}s")
    print(f"   Output: {HEADER_DIR}\\model_v4_se_23dim_ch32_fullint8_s{{seed}}.h × 5")
    print(f"   Logs:   {LOG_DIR}\\fullint8_ptq_seed{{seed}}.json × 5 + combined")


if __name__ == "__main__":
    main()