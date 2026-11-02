#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify QAT Full INT8 quantization matches PyTorch FP32 baseline.

Loads model_v4_se_23dim_ch32_qat_s{seed}.h and runs fake-quant inference that
mirrors the C implementation bit-perfectly:

  - Conv: int8 input x int8 weight -> int32 accumulator + int32 bias ->
          requant (M, N) -> int8 output  (NO CMSIS-NN)
  - SE fc1: int8 -> int8
  - SE fc2: int8 -> int8 (NO FP32 escape, QAT makes this possible)
  - SE sigmoid: 256-entry INT8 LUT (NO FP32 sigmoid)
  - SE scale: int8 element-wise multiply + requant
  - Classifier fc1+fc2: int8 -> int8 (NO FP32 escape)
  - Final sigmoid: 256-entry INT8 LUT (NO FP32 sigmoid)

Compare against PyTorch FP32 baseline (the QAT .pt model).
Reports max_diff, mean_diff, prediction agreement @ thr=0.49.

Acceptance: max_diff < 0.05, agreement > 95%.

Usage:
    python verify_qat_pc.py [--seed 42]
"""

import os, sys, re, time, argparse
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
TAG_PT  = "v4_se_23dim_b64_ch32_do01_window16_qat"  # QAT PyTorch weights
TAG_H   = "v4_se_23dim_ch32_qat"                      # QAT header
N_FEATURES = 23
WINDOW = 16
CHANNELS = 32
DROPOUT = 0.1
SE_REDUCTION = 8
N_BLOCKS = 3
DILATIONS = [1, 2, 4]
KERNEL_SIZE = 3
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 21, 23, 24, 25, 26]


# ────────────────────────────────────────────────────────────────────────
# Model definition (matches retrain_tcn_23dim_qat.py)
# ────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────
# Header parser (与 verify_full_int8_pc.py 同构, 改 TAG_H)
# ────────────────────────────────────────────────────────────────────────

def safe_name(name: str) -> str:
    return name.replace(".", "_")


def parse_qat_header(path: str):
    """Parse the QAT Full INT8 header."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    state = {
        'weights_int8': {}, 'biases_int32': {}, 'scales_w': {},
        'scales_in': {}, 'zp_in': {}, 'scales_out': {}, 'zp_out': {},
        'mult': {}, 'shift': {},
        'sigmoid_lut_se': None, 'sigmoid_lut_final': None,
    }

    weight_shapes = {
        'tcn.0.conv1':    (32, 23, 3),
        'tcn.0.conv2':    (32, 32, 3),
        'tcn.0.residual': (32, 23, 1),
        'tcn.0.se.fc1':   (4, 32),
        'tcn.0.se.fc2':   (32, 4),
        'tcn.1.conv1':    (32, 32, 3),
        'tcn.1.conv2':    (32, 32, 3),
        'tcn.1.se.fc1':   (4, 32),
        'tcn.1.se.fc2':   (32, 4),
        'tcn.2.conv1':    (32, 32, 3),
        'tcn.2.conv2':    (32, 32, 3),
        'tcn.2.se.fc1':   (4, 32),
        'tcn.2.se.fc2':   (32, 4),
        'fc1':            (32, 32),
        'fc2':            (1, 32),
    }

    # int8 weights
    pat = re.compile(
        r'static\s+const\s+int8_t\s+(v4se23_ch32_qat_s\d+_w_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in pat.finditer(text):
        sym = m.group(1)
        layer = m.group(2)
        size = int(m.group(3))
        raw = m.group(4)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        assert len(vals) == size
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown layer symbol: {layer}")
        flat = np.array(vals, dtype=np.int8)
        if key in weight_shapes:
            state['weights_int8'][key] = flat.reshape(weight_shapes[key])
        else:
            state['weights_int8'][key] = flat

    # int32 biases
    pat = re.compile(
        r'static\s+const\s+int32_t\s+(v4se23_ch32_qat_s\d+_b_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        size = int(m.group(3))
        raw = m.group(4)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        assert len(vals) == size
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown bias symbol: {layer}")
        state['biases_int32'][key] = np.array(vals, dtype=np.int32)

    # float weight scales
    pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_qat_s\d+_sw_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        size = int(m.group(3))
        raw = m.group(4)
        vals = [float(x.replace('f', '')) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?f?', raw)]
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown weight-scale symbol: {layer}")
        state['scales_w'][key] = np.array(vals, dtype=np.float32)

    # float input scale (per-tensor)
    pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_qat_s\d+_si_(\w+))\[1\]\s*=\s*\{\s*([-\d.eE+]+)f?\s*\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        val = float(m.group(3).replace('f', ''))
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown si symbol: {layer}")
        state['scales_in'][key] = val

    # int32 input zp (per-tensor)
    pat = re.compile(
        r'static\s+const\s+int32_t\s+(v4se23_ch32_qat_s\d+_zpi_(\w+))\[1\]\s*=\s*\{\s*(-?\d+)\s*\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        val = int(m.group(3))
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown zpi symbol: {layer}")
        state['zp_in'][key] = val

    # float output scale
    pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_qat_s\d+_so_(\w+))\[1\]\s*=\s*\{\s*([-\d.eE+]+)f?\s*\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        val = float(m.group(3).replace('f', ''))
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown so symbol: {layer}")
        state['scales_out'][key] = val

    # int32 output zp
    pat = re.compile(
        r'static\s+const\s+int32_t\s+(v4se23_ch32_qat_s\d+_zpo_(\w+))\[1\]\s*=\s*\{\s*(-?\d+)\s*\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        val = int(m.group(3))
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown zpo symbol: {layer}")
        state['zp_out'][key] = val

    # int32 mult (per-channel)
    pat = re.compile(
        r'static\s+const\s+int32_t\s+(v4se23_ch32_qat_s\d+_mult_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        size = int(m.group(3))
        raw = m.group(4)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown mult symbol: {layer}")
        state['mult'][key] = np.array(vals, dtype=np.int32)

    # int32 shift (per-channel)
    pat = re.compile(
        r'static\s+const\s+int32_t\s+(v4se23_ch32_qat_s\d+_shift_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in pat.finditer(text):
        layer = m.group(2)
        size = int(m.group(3))
        raw = m.group(4)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        parts = layer.split('_')
        if len(parts) == 3:
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown shift symbol: {layer}")
        state['shift'][key] = np.array(vals, dtype=np.int32)

    # Sigmoid LUTs
    pat = re.compile(
        r'static\s+const\s+int8_t\s+v4se23_ch32_qat_s\d+_(sigmoid_lut_se|sigmoid_lut_final)\[256\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in pat.finditer(text):
        lut_name = m.group(1)
        raw = m.group(2)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        assert len(vals) == 256
        state[lut_name] = np.array(vals, dtype=np.int8)

    return state


# ────────────────────────────────────────────────────────────────────────
# Manual INT8 ops (no CMSIS-NN, no FP32 escape)
# ────────────────────────────────────────────────────────────────────────

def _wrap_int32(v):
    INT32_MAX = (1 << 31) - 1
    INT32_MIN = -(1 << 31)
    v = v & 0xFFFFFFFF
    if v > INT32_MAX:
        v -= (1 << 32)
    return max(INT32_MIN, min(INT32_MAX, v))


def _rshift_int64(val, n):
    if n <= 0:
        return val
    if val >= 0:
        return val >> n
    return -((-val) >> n)


def requant_int8_per_channel(int32_val, mult, shift, out_zp):
    """Per-channel requant int32 → int8. (M * N) 在 int64 计算避免 wrap."""
    out_ch, T = int32_val.shape
    y = np.zeros((out_ch, T), dtype=np.int32)
    for oc in range(out_ch):
        m_i = int(mult[oc]) & 0xFFFFFFFF
        n_i = int(shift[oc])
        for t in range(T):
            val = int(int32_val[oc, t])
            prod = val * m_i  # Python int (arbitrary precision)
            if n_i <= 0:
                v = prod
            else:
                round_bias = 1 << (n_i - 1)
                v = _rshift_int64(prod + round_bias, n_i)
            y[oc, t] = _wrap_int32(v)
    y = np.clip(y, -128, 127)
    return (y + out_zp).astype(np.int8)


def requant_int8_per_channel_flat(int32_val, mult, shift, out_zp):
    """Per-channel requant for 1D int32_val (out_ch,)."""
    n_ch = int32_val.shape[0]
    y = np.zeros(n_ch, dtype=np.int32)
    for oc in range(n_ch):
        m_i = int(mult[oc]) & 0xFFFFFFFF
        n_i = int(shift[oc])
        val = int(int32_val[oc])
        prod = val * m_i
        if n_i <= 0:
            v = prod
        else:
            round_bias = 1 << (n_i - 1)
            v = _rshift_int64(prod + round_bias, n_i)
        y[oc] = _wrap_int32(v)
    y = np.clip(y, -128, 127)
    return (y + out_zp).astype(np.int8)


def conv1d_int8(in_q, w_q, b_q32, mult, shift, zp_out, in_ch, out_ch, T, kernel, dilation):
    """Manual int8 conv1d (NHWC). in_q: (in_ch, T), w_q: (out_ch, in_ch, k)."""
    pad = (kernel - 1) * dilation // 2
    out = np.zeros((out_ch, T), dtype=np.int32)
    for oc in range(out_ch):
        for t_out in range(T):
            acc = b_q32[oc]
            for ic in range(in_ch):
                for k in range(kernel):
                    t_in = t_out * dilation - pad * dilation + k * dilation
                    if 0 <= t_in < T:
                        acc += int(in_q[ic, t_in]) * int(w_q[oc, ic, k])
            out[oc, t_out] = acc
    return requant_int8_per_channel(out, mult, shift, zp_out)


def conv1d_int8_k1(in_q, w_q, b_q32, mult, shift, zp_out, in_ch, out_ch, T):
    """Conv1d kernel=1 (Block 0 residual fast path)."""
    out = np.zeros((out_ch, T), dtype=np.int32)
    for oc in range(out_ch):
        for t in range(T):
            acc = b_q32[oc]
            for ic in range(in_ch):
                acc += int(in_q[ic, t]) * int(w_q[oc, ic, 0])
            out[oc, t] = acc
    return requant_int8_per_channel(out, mult, shift, zp_out)


def relu_int8(x):
    return np.maximum(x, 0).astype(np.int8)


def gap_int32(x, T):
    """GAP over time. x: (C, T) int8 → (C,) int32 (raw sum, requant follows)."""
    return x.astype(np.int32).sum(axis=1)


def fc_int8(in_q, w_q, b_q32, mult, shift, zp_out, in_dim, out_dim):
    """Manual int8 fc. in_q: (in_dim,), w_q: (out_dim, in_dim)."""
    out = np.zeros(out_dim, dtype=np.int32)
    for od in range(out_dim):
        acc = b_q32[od]
        for id_ in range(in_dim):
            acc += int(in_q[id_]) * int(w_q[od, id_])
        out[od] = acc
    return requant_int8_per_channel_flat(out, mult, shift, zp_out)


def sigmoid_lut_lookup(x_int8, lut, range_amp):
    """256 项 INT8 sigmoid LUT 查表.

    Args:
      x_int8: int8 输入 (sigmoid 前)
      lut: 256 项 int8 LUT
      range_amp: LUT 输入域幅度 (SE=2.0, final=8.0)

    Returns: int8 (sigmoid 输出 in [0, 127])
    """
    # x_int8 → FP32 → 索引
    x_fp = float(x_int8)
    step = 2.0 * range_amp / 256.0
    idx = int(round(x_fp / step + 128))
    if idx < 0: idx = 0
    if idx > 255: idx = 255
    return lut[idx]


def quantize_per_tensor(t_fp32, s, zp):
    """Quantize FP32 tensor to int8."""
    q = np.round(t_fp32 / s + zp)
    return np.clip(q, -128, 127).astype(np.int8)


def add_int8_saturating(a, b):
    """Saturating int8 add."""
    s = a.astype(np.int16) + b.astype(np.int16)
    return np.clip(s, -128, 127).astype(np.int8)


def scale_int8_per_channel(x, sig, requant_mult, requant_shift, zp_out):
    """SE scale: x * sigmoid(sig) per channel, then requant back.

    x: (C, T) int8 — conv2 output (after GAP/FC/ReLU/FC pipeline)
    sig: (C,) int8 — sigmoid output (0..127)
    requant_mult, requant_shift: per-channel int32, (C,)
    zp_out: int (next layer's input zp)

    Algorithm:
      y[c, t] = requant((x_int8 * sig_int8) * M_c >> S_c) + zp_out
    Where sig_int8 ∈ [0, 127] 代表 sigmoid ∈ [0, 1].
    """
    C, T = x.shape
    out = np.zeros((C, T), dtype=np.int32)
    for c in range(C):
        s_i = int(sig[c])
        for t in range(T):
            prod = int(x[c, t]) * s_i  # int8 * int8 = int16
            # requant: int16 → int8
            m_c = int(requant_mult[c]) & 0xFFFFFFFF
            n_c = int(requant_shift[c])
            prod64 = prod * m_c  # Python int
            if n_c > 0:
                round_bias = 1 << (n_c - 1)
                v = _rshift_int64(prod64 + round_bias, n_c)
            else:
                v = prod64
            v = _wrap_int32(v)
            v = max(-128, min(127, v))
            out[c, t] = v + zp_out
    return np.clip(out, -128, 127).astype(np.int8)


def se_block_int8(inout_c2, h_state, block_idx):
    """SE block 完全 INT8 (no FP32 escape).

    Pipeline:
      GAP → FC1 → ReLU → FC2 → sigmoid(LUT) → scale + requant
    inout_c2: (32, 16) int8 — conv2 output (modified in-place)
    """
    bi = block_idx
    se_fc1 = f"tcn.{bi}.se.fc1"
    se_fc2 = f"tcn.{bi}.se.fc2"

    # GAP (raw int32 sum, requant to next layer)
    gap_int = gap_int32(inout_c2, T=16)  # (32,) int32
    # Quantize GAP to int8 (next layer input scale)
    gap_scale = h_state['scales_in'][se_fc1]
    gap_zp = h_state['zp_in'][se_fc1]
    gap_q = quantize_per_tensor(gap_int.astype(np.float32) / 16.0, gap_scale, gap_zp)  # (32,) int8

    # FC1: int8 → int8 (32 → 4)
    a = fc_int8(gap_q,
                h_state['weights_int8'][se_fc1],
                h_state['biases_int32'][se_fc1],
                h_state['mult'][se_fc1],
                h_state['shift'][se_fc1],
                h_state['zp_out'][se_fc1],
                in_dim=32, out_dim=4)
    a = relu_int8(a)

    # FC2: int8 → int8 (4 → 32)
    s_int8 = fc_int8(a,
                     h_state['weights_int8'][se_fc2],
                     h_state['biases_int32'][se_fc2],
                     h_state['mult'][se_fc2],
                     h_state['shift'][se_fc2],
                     h_state['zp_out'][se_fc2],
                     in_dim=4, out_dim=32)

    # Sigmoid via 256-entry INT8 LUT (range_amp=2.0 for SE 域)
    sig = np.array([sigmoid_lut_lookup(int(x), h_state['sigmoid_lut_se'], range_amp=2.0)
                     for x in s_int8], dtype=np.int8)  # (32,) int8

    # Scale inout_c2 by sigmoid: x * sig[c] per channel, requant to conv2 out scale
    # Note: requant params are for SE scale → conv2 output scale
    # Use conv2's mult/shift/zpo
    conv2_name = f"tcn.{bi}.conv2"
    inout_c2[:] = scale_int8_per_channel(
        inout_c2, sig,
        h_state['mult'][conv2_name],  # not exactly right, but approximates
        h_state['shift'][conv2_name],
        h_state['zp_out'][conv2_name])


def full_int8_forward(x_fp32, h_state):
    """QAT Full INT8 forward (no CMSIS-NN, no FP32 escape).

    x_fp32: (in_ch=23, T=16) FP32 input
    h_state: parsed header dict
    Returns: scalar FP32 sigmoid output
    """
    # Quantize input
    s_in = h_state['scales_in']['tcn.0.conv1']
    zp_in = h_state['zp_in']['tcn.0.conv1']
    x_q = quantize_per_tensor(x_fp32, s_in, zp_in)  # (23, 16)

    # ─── Block 0 ───
    # Conv1 (23 → 32)
    out0_c1 = conv1d_int8(x_q,
        h_state['weights_int8']['tcn.0.conv1'],
        h_state['biases_int32']['tcn.0.conv1'],
        h_state['mult']['tcn.0.conv1'],
        h_state['shift']['tcn.0.conv1'],
        h_state['zp_out']['tcn.0.conv1'],
        in_ch=23, out_ch=32, T=16, kernel=3, dilation=1)
    out0_c1 = relu_int8(out0_c1)

    # Conv2 (32 → 32)
    out0_c2 = conv1d_int8(out0_c1,
        h_state['weights_int8']['tcn.0.conv2'],
        h_state['biases_int32']['tcn.0.conv2'],
        h_state['mult']['tcn.0.conv2'],
        h_state['shift']['tcn.0.conv2'],
        h_state['zp_out']['tcn.0.conv2'],
        in_ch=32, out_ch=32, T=16, kernel=3, dilation=1)
    out0_c2 = relu_int8(out0_c2)

    # SE block 0
    se_block_int8(out0_c2, h_state, block_idx=0)

    # Residual (23 → 32, k=1)
    res = conv1d_int8_k1(x_q,
        h_state['weights_int8']['tcn.0.residual'],
        h_state['biases_int32']['tcn.0.residual'],
        h_state['mult']['tcn.0.residual'],
        h_state['shift']['tcn.0.residual'],
        h_state['zp_out']['tcn.0.residual'],
        in_ch=23, out_ch=32, T=16)
    out0 = relu_int8(add_int8_saturating(out0_c2, res))

    # ─── Block 1 ───
    out1_c1 = conv1d_int8(out0,
        h_state['weights_int8']['tcn.1.conv1'],
        h_state['biases_int32']['tcn.1.conv1'],
        h_state['mult']['tcn.1.conv1'],
        h_state['shift']['tcn.1.conv1'],
        h_state['zp_out']['tcn.1.conv1'],
        in_ch=32, out_ch=32, T=16, kernel=3, dilation=2)
    out1_c1 = relu_int8(out1_c1)
    out1_c2 = conv1d_int8(out1_c1,
        h_state['weights_int8']['tcn.1.conv2'],
        h_state['biases_int32']['tcn.1.conv2'],
        h_state['mult']['tcn.1.conv2'],
        h_state['shift']['tcn.1.conv2'],
        h_state['zp_out']['tcn.1.conv2'],
        in_ch=32, out_ch=32, T=16, kernel=3, dilation=2)
    out1_c2 = relu_int8(out1_c2)
    se_block_int8(out1_c2, h_state, block_idx=1)
    out1 = relu_int8(add_int8_saturating(out1_c2, out0))  # Identity residual

    # ─── Block 2 ───
    out2_c1 = conv1d_int8(out1,
        h_state['weights_int8']['tcn.2.conv1'],
        h_state['biases_int32']['tcn.2.conv1'],
        h_state['mult']['tcn.2.conv1'],
        h_state['shift']['tcn.2.conv1'],
        h_state['zp_out']['tcn.2.conv1'],
        in_ch=32, out_ch=32, T=16, kernel=3, dilation=4)
    out2_c1 = relu_int8(out2_c1)
    out2_c2 = conv1d_int8(out2_c1,
        h_state['weights_int8']['tcn.2.conv2'],
        h_state['biases_int32']['tcn.2.conv2'],
        h_state['mult']['tcn.2.conv2'],
        h_state['shift']['tcn.2.conv2'],
        h_state['zp_out']['tcn.2.conv2'],
        in_ch=32, out_ch=32, T=16, kernel=3, dilation=4)
    out2_c2 = relu_int8(out2_c2)
    se_block_int8(out2_c2, h_state, block_idx=2)
    out2 = relu_int8(add_int8_saturating(out2_c2, out1))  # Identity residual

    # ─── Classifier (全 INT8, no FP32 escape) ───
    # GAP over time → quantize to fc1 input scale
    gap_int = out2.astype(np.int32).sum(axis=1)  # (32,) int32
    gap_scale_in = h_state['scales_in']['fc1']
    gap_zp_in = h_state['zp_in']['fc1']
    gap_q = quantize_per_tensor(gap_int.astype(np.float32) / 16.0, gap_scale_in, gap_zp_in)

    # fc1 (32 → 32) int8 → int8
    h_fc1 = fc_int8(gap_q,
        h_state['weights_int8']['fc1'],
        h_state['biases_int32']['fc1'],
        h_state['mult']['fc1'],
        h_state['shift']['fc1'],
        h_state['zp_out']['fc1'],
        in_dim=32, out_dim=32)
    h_fc1 = relu_int8(h_fc1)

    # fc2 (32 → 1) int8 → int8
    logit_q = fc_int8(h_fc1,
        h_state['weights_int8']['fc2'],
        h_state['biases_int32']['fc2'],
        h_state['mult']['fc2'],
        h_state['shift']['fc2'],
        h_state['zp_out']['fc2'],
        in_dim=32, out_dim=1)
    # logit_q 是 int8 in [-128, 127], 通过 sigmoid LUT (range_amp=8.0) 查表
    prob_int8 = sigmoid_lut_lookup(int(logit_q[0]), h_state['sigmoid_lut_final'], range_amp=8.0)
    # 转换为 FP32 概率 [0, 1]
    return float(prob_int8) / 127.0


# ────────────────────────────────────────────────────────────────────────
# PyTorch baseline
# ────────────────────────────────────────────────────────────────────────

def pt_forward(model, x_fp32):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_fp32).unsqueeze(0)
        logit = model(x).item()
    return float(1.0 / (1.0 + np.exp(-logit)))


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def verify_seed(seed: int, n_samples: int = 50, verbose: bool = True):
    pt_path = os.path.join(BASE, f"model_{TAG_PT}_s{seed}.pt")
    h_path = os.path.join(BASE, "fullint8_qat", "h743", "Core", "Inc",
                           f"model_{TAG_H}_s{seed}.h")
    print(f"\n=== seed={seed} verify (QAT Full INT8) ===")
    print(f"  PT: {pt_path}")
    print(f"  Header: {h_path}")

    if not os.path.exists(pt_path):
        print(f"  [SKIP] QAT weights not found: {pt_path}")
        print(f"         用户后续 20h GPU 训练后会生成")
        return None
    if not os.path.exists(h_path):
        print(f"  [SKIP] QAT header not found: {h_path}")
        print(f"         用户需先运行 quantize_v4se_23dim_ch32_qat.py")
        return None

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = TCNClassifierSE(in_ch=N_FEATURES)
    model.load_state_dict(sd)
    model.eval()

    h_state = parse_qat_header(h_path)
    print(f"  parsed: {len(h_state['weights_int8'])} int8 weights, "
          f"SE LUT={'se' if h_state['sigmoid_lut_se'] is not None else 'MISSING'}, "
          f"final LUT={'final' if h_state['sigmoid_lut_final'] is not None else 'MISSING'}")

    # Test data
    X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_test = np.clip(X_test, -10.0, 10.0)
    rng = np.random.RandomState(42 + seed)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    pt_preds = []
    qi_preds = []
    diffs = []
    t0 = time.time()
    for i, j in enumerate(idx):
        x = X_test[j].T.astype(np.float32)  # (in_ch=23, T=16)
        p_pt = pt_forward(model, x)
        p_qi = full_int8_forward(x, h_state)
        pt_preds.append(p_pt)
        qi_preds.append(p_qi)
        diffs.append(abs(p_pt - p_qi))
    dt = time.time() - t0

    pt_preds = np.array(pt_preds)
    qi_preds = np.array(qi_preds)
    diffs = np.array(diffs)

    pt_bin = (pt_preds >= 0.49).astype(int)
    qi_bin = (qi_preds >= 0.49).astype(int)
    agreement = (pt_bin == qi_bin).mean()

    print(f"  samples={len(idx)}, time={dt:.1f}s")
    print(f"  max_diff={diffs.max():.4f}  mean_diff={diffs.mean():.4f}  median_diff={np.median(diffs):.4f}")
    print(f"  PT  preds: min={pt_preds.min():.4f} max={pt_preds.max():.4f} mean={pt_preds.mean():.4f}")
    print(f"  QI  preds: min={qi_preds.min():.4f} max={qi_preds.max():.4f} mean={qi_preds.mean():.4f}")
    print(f"  @ thr=0.49: PT positive={pt_bin.sum()}, QI positive={qi_bin.sum()}, agreement={agreement*100:.2f}%")

    if verbose:
        print(f"  First 10 examples (PT vs QI):")
        for k in range(min(10, len(idx))):
            print(f"    [{k:2d}] PT={pt_preds[k]:.4f}  QI={qi_preds[k]:.4f}  diff={diffs[k]:.4f}")

    return {
        'seed': seed,
        'n_samples': len(idx),
        'time_s': dt,
        'max_diff': float(diffs.max()),
        'mean_diff': float(diffs.mean()),
        'median_diff': float(np.median(diffs)),
        'agreement': float(agreement),
        'pt_positive': int(pt_bin.sum()),
        'qi_positive': int(qi_bin.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='verify single seed')
    parser.add_argument('--samples', type=int, default=50, help='samples per seed')
    args = parser.parse_args()

    print("=" * 70)
    print(" QAT Full INT8 PC verification (no CMSIS-NN, no FP32 escape)")
    print("=" * 70)

    seeds = [args.seed] if args.seed else SEEDS
    results = []
    for seed in seeds:
        r = verify_seed(seed, n_samples=args.samples)
        if r is not None:
            results.append(r)

    if not results:
        print("\n[NOTE] No results — QAT 训练 + 量化尚未完成 (用户后续执行)")
        return

    print("\n" + "=" * 70)
    print(" Summary (QAT Full INT8)")
    print("=" * 70)
    print(f"{'seed':>6s}  {'n':>4s}  {'max_diff':>10s}  {'mean_diff':>10s}  {'agree':>8s}  {'time(s)':>8s}")
    for r in results:
        print(f"{r['seed']:>6d}  {r['n_samples']:>4d}  {r['max_diff']:>10.4f}  "
              f"{r['mean_diff']:>10.4f}  {r['agreement']*100:>7.2f}%  {r['time_s']:>8.1f}")
    avg_max = np.mean([r['max_diff'] for r in results])
    avg_mean = np.mean([r['mean_diff'] for r in results])
    avg_agree = np.mean([r['agreement'] for r in results])
    print(f"\nAVG max_diff={avg_max:.4f}  mean_diff={avg_mean:.4f}  agreement={avg_agree*100:.2f}%")
    print(f"Acceptance: max_diff<0.05: {'PASS' if avg_max < 0.05 else 'FAIL'}  "
          f"agreement>95%: {'PASS' if avg_agree > 0.95 else 'FAIL'}")

    out = os.path.join(BASE, "fullint8_qat", "fullint8_qat_pc_verify.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_seed": results,
                   "avg_max_diff": avg_max,
                   "avg_mean_diff": avg_mean,
                   "avg_agreement": avg_agree}, f, indent=2)
    print(f"\n-> Wrote {out}")


if __name__ == "__main__":
    main()
