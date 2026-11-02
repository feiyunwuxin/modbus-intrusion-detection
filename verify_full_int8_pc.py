#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify Full INT8 quantization matches PyTorch FP32 baseline.

For each .pt model, parse the generated Full INT8 header (per-channel int8 weights,
int32 bias, per-tensor act scales/zp, per-channel requant M/S) and run a fake-quant
inference that mimics CMSIS-NN's arm_convolve_s8 path:
  - Conv: int8 input x int8 weight -> int32 accumulator + int32 bias ->
          requant (M, N) -> int8 output
  - SE fc1: int8 -> int8
  - SE fc2 / sigmoid / scale / Classifier fc1+fc2: FP32 escape

Compare against PyTorch FP32 baseline (the original .pt model).
Reports max_diff, mean_diff, prediction agreement @ thr=0.49.

Acceptance: max_diff < 0.05, agreement > 99%.

Usage:
    python verify_full_int8_pc.py [--seed 42]
"""

import os, sys, re, time, argparse
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
TAG_PT = "v4_se_23dim_b64_ch32_do01_window16"
TAG_H = "v4_se_23dim_ch32_fullint8"
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
# Model definition (matches retrain script)
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
# Header parser
# ────────────────────────────────────────────────────────────────────────

def safe_name(name: str) -> str:
    """tcn.0.conv1 -> tcn_0_conv1, fc1 -> fc1"""
    return name.replace(".", "_")


def parse_full_int8_header(path: str):
    """Parse the Full INT8 header.

    Returns dict:
      weights_int8: {layer: np.int8 array (already reshaped to logical shape)}
      biases_int32: {layer: np.int32 array}
      scales_w: {layer: np.float32 array}  (per-channel)
      scales_in: {layer: float}  (per-tensor)
      zp_in: {layer: int}        (per-tensor)
      scales_out: {layer: float} (per-tensor)
      zp_out: {layer: int}       (per-tensor)
      mult: {layer: np.int32 array} (per-channel)
      shift: {layer: np.int32 array} (per-channel)
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    state = {
        'weights_int8': {}, 'biases_int32': {}, 'scales_w': {},
        'scales_in': {}, 'zp_in': {}, 'scales_out': {}, 'zp_out': {},
        'mult': {}, 'shift': {},
    }

    # Logical shape per layer (matches the model's PyTorch parameter shapes)
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
        r'static\s+const\s+int8_t\s+(v4se23_ch32_fullint8_s\d+_w_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in pat.finditer(text):
        sym = m.group(1)
        layer = m.group(2)  # e.g. tcn_0_conv1
        size = int(m.group(3))
        raw = m.group(4)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        assert len(vals) == size, f"{sym}: header says {size} but parsed {len(vals)}"
        # Convert back to dotted key
        key = layer.replace('tcn_', 'tcn.').replace('_se_', '.se.').replace('_', '.', 1) if False else None
        # Easier: tcn_0_conv1 -> tcn.0.conv1, tcn_0_se_fc1 -> tcn.0.se.fc1
        parts = layer.split('_')
        if len(parts) == 3:  # tcn_0_conv1
            key = f"tcn.{parts[1]}.{parts[2]}"
        elif len(parts) == 4 and parts[2] == 'se':  # tcn_0_se_fc1
            key = f"tcn.{parts[1]}.se.{parts[3]}"
        elif parts[0] == 'fc1' or parts[0] == 'fc2':
            key = parts[0]
        else:
            raise ValueError(f"unknown layer symbol: {layer}")
        flat = np.array(vals, dtype=np.int8)
        # Reshape to logical shape if known
        if key in weight_shapes:
            state['weights_int8'][key] = flat.reshape(weight_shapes[key])
        else:
            state['weights_int8'][key] = flat

    # int32 biases
    pat = re.compile(
        r'static\s+const\s+int32_t\s+(v4se23_ch32_fullint8_s\d+_b_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
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
        r'static\s+const\s+float\s+(v4se23_ch32_fullint8_s\d+_sw_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
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
        r'static\s+const\s+float\s+(v4se23_ch32_fullint8_s\d+_si_(\w+))\[1\]\s*=\s*\{\s*([-\d.eE+]+)f?\s*\}',
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
        r'static\s+const\s+int32_t\s+(v4se23_ch32_fullint8_s\d+_zpi_(\w+))\[1\]\s*=\s*\{\s*(-?\d+)\s*\}',
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
        r'static\s+const\s+float\s+(v4se23_ch32_fullint8_s\d+_so_(\w+))\[1\]\s*=\s*\{\s*([-\d.eE+]+)f?\s*\}',
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
        r'static\s+const\s+int32_t\s+(v4se23_ch32_fullint8_s\d+_zpo_(\w+))\[1\]\s*=\s*\{\s*(-?\d+)\s*\}',
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
        r'static\s+const\s+int32_t\s+(v4se23_ch32_fullint8_s\d+_mult_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
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
        r'static\s+const\s+int32_t\s+(v4se23_ch32_fullint8_s\d+_shift_(\w+))\[(\d+)\]\s*=\s*\{([^}]+)\}',
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

    return state


# ────────────────────────────────────────────────────────────────────────
# Fake-quant inference (mimics CMSIS-NN arm_convolve_s8 path)
# ────────────────────────────────────────────────────────────────────────

def _wrap_int32(v):
    """Wrap arbitrary-precision int to int32 range [-2^31, 2^31-1] using two's complement."""
    INT32_MAX = (1 << 31) - 1
    INT32_MIN = -(1 << 31)
    v = v & 0xFFFFFFFF  # mask to 32 bits
    if v > INT32_MAX:
        v -= (1 << 32)
    return max(INT32_MIN, min(INT32_MAX, v))


def _rshift_int64(val, n):
    """Arithmetic right shift of Python int (works on negative numbers)."""
    if n <= 0:
        return val
    if val >= 0:
        return val >> n
    # For negative, arithmetic shift: floor division by 2^n
    return -((-val) >> n)


def requant_int8_per_channel(int32_val, mult, shift, out_zp):
    """CMSIS-NN style requant: int32 -> int8 with per-channel M/S + per-tensor out_zp.

    y = saturate(((val * M) + (1<<(N-1))) >> N) + out_zp

    CRITICAL: compute (val * M) in INT64 (no wrap), then >> N (which is at most 31),
    then saturate to int32, then add out_zp and saturate to int8.

    mult, shift shape (out_ch,), int32_val shape (out_ch, T).
    """
    out_ch, T = int32_val.shape
    y = np.zeros((out_ch, T), dtype=np.int32)
    for oc in range(out_ch):
        m_i = int(mult[oc]) & 0xFFFFFFFF
        n_i = int(shift[oc])
        for t in range(T):
            val = int(int32_val[oc, t])
            # val * M in int64 (Python int is arbitrary precision, then >> N is arithmetic)
            prod = val * m_i  # Python int, no wrap
            if n_i <= 0:
                v = prod
            else:
                round_bias = 1 << (n_i - 1)
                v = _rshift_int64(prod + round_bias, n_i)
            # Saturate to int32
            y[oc, t] = _wrap_int32(v)
    y = np.clip(y, -128, 127)
    return (y + out_zp).astype(np.int8)


def requant_int8_per_channel_flat(int32_val, mult, shift, out_zp):
    """Same but for 1D int32_val (out_ch,)."""
    n_ch = int32_val.shape[0]
    y = np.zeros(n_ch, dtype=np.int32)
    for oc in range(n_ch):
        m_i = int(mult[oc]) & 0xFFFFFFFF
        n_i = int(shift[oc])
        val = int(int32_val[oc])
        prod = val * m_i  # Python int, no wrap
        if n_i <= 0:
            v = prod
        else:
            round_bias = 1 << (n_i - 1)
            v = _rshift_int64(prod + round_bias, n_i)
        y[oc] = _wrap_int32(v)
    y = np.clip(y, -128, 127)
    return (y + out_zp).astype(np.int8)


def conv1d_int8(in_q, w_q, b_q32, mult, shift, zp_out, in_ch, out_ch, T, kernel, dilation):
    """Fake-quant Conv1d: int8 input (in_ch, T) @ int8 weight (out_ch, in_ch, k) + int32 bias.

    Returns int8 output (out_ch, T) after per-channel requant.
    """
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
    # Requant to int8
    return requant_int8_per_channel(out, mult, shift, zp_out)


def conv1d_int8_k1(in_q, w_q, b_q32, mult, shift, zp_out, in_ch, out_ch, T):
    """Conv1d with kernel=1, dilation=1 — used for Block 0 residual.

    Optimized: no padding needed.
    """
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
    """Per-channel mean over time dim. x: (C, T) int8 -> (C,) float (dequantized)."""
    return x.astype(np.int32).sum(axis=1).astype(np.float32) / float(T)


def fc_int8(in_q, w_q, b_q32, mult, shift, zp_out, in_dim, out_dim):
    """Fake-quant FC: int8 input (in_dim,) @ int8 weight (out_dim, in_dim) + int32 bias."""
    out = np.zeros(out_dim, dtype=np.int32)
    for od in range(out_dim):
        acc = b_q32[od]
        for id_ in range(in_dim):
            acc += int(in_q[id_]) * int(w_q[od, id_])
        out[od] = acc
    return requant_int8_per_channel_flat(out, mult, shift, zp_out)


def fc_fp32(in_q_int8, w_q_int8, s_w_per_ch, b_fp32, s_in, zp_in):
    """FC with dequant-to-FP32 escape. Used for SE fc2 (small) and Classifier.

    in_q_int8: (in_dim,) int8
    w_q_int8: (out_dim, in_dim) int8
    s_w_per_ch: (out_dim,) FP32 per-channel scale
    b_fp32: (out_dim,) FP32
    s_in, zp_in: per-tensor input scale + zp
    Returns: (out_dim,) FP32
    """
    in_fp = (in_q_int8.astype(np.float32) - zp_in) * s_in  # dequant
    # weight dequant: per-channel
    out = np.zeros(w_q_int8.shape[0], dtype=np.float32)
    for od in range(w_q_int8.shape[0]):
        s_w = s_w_per_ch[od]
        acc = 0.0
        for id_ in range(w_q_int8.shape[1]):
            acc += float(w_q_int8[od, id_]) * s_w * in_fp[id_]
        out[od] = acc + b_fp32[od]
    return out


def quantize_per_tensor(t_fp32, s, zp):
    """Quantize FP32 tensor to int8."""
    q = np.round(t_fp32 / s + zp)
    return np.clip(q, -128, 127).astype(np.int8)


def full_int8_forward(x_fp32, h_state, fp32_biases):
    """Run fake-quant forward through the model.

    x_fp32: (in_ch=23, T=16) FP32 input
    h_state: parsed header dict
    fp32_biases: {'fc1.bias': np.float32 array (32,), 'fc2.bias': np.float32 array (1,)}

    Returns: scalar FP32 sigmoid output
    """
    # Quantize input
    s_in = h_state['scales_in']['tcn.0.conv1']
    zp_in = h_state['zp_in']['tcn.0.conv1']
    x_q = quantize_per_tensor(x_fp32, s_in, zp_in)  # (23, 16)

    # Block 0
    # Conv1 (23 -> 32)
    w = h_state['weights_int8']['tcn.0.conv1']
    b = h_state['biases_int32']['tcn.0.conv1']
    m = h_state['mult']['tcn.0.conv1']
    sh = h_state['shift']['tcn.0.conv1']
    zp_out = h_state['zp_out']['tcn.0.conv1']
    out0_c1 = conv1d_int8(x_q, w, b, m, sh, zp_out,
                          in_ch=23, out_ch=32, T=16, kernel=3, dilation=1)
    out0_c1 = relu_int8(out0_c1)

    # Conv2 (32 -> 32)
    # Feed raw int8 output of conv1 (already requantized) directly.
    w = h_state['weights_int8']['tcn.0.conv2']
    b = h_state['biases_int32']['tcn.0.conv2']
    m = h_state['mult']['tcn.0.conv2']
    sh = h_state['shift']['tcn.0.conv2']
    zp_out = h_state['zp_out']['tcn.0.conv2']
    out0_c2 = conv1d_int8(out0_c1, w, b, m, sh, zp_out,
                          in_ch=32, out_ch=32, T=16, kernel=3, dilation=1)
    out0_c2 = relu_int8(out0_c2)

    # SE block 0 — GAP -> FC1 -> ReLU -> FC2 -> sigmoid -> scale
    z = gap_int32(out0_c2, T=16)  # FP32 (dequantized)
    # Need to quantize z to int8 for FC1
    s_in_se = h_state['scales_in']['tcn.0.se.fc1']
    zp_in_se = h_state['zp_in']['tcn.0.se.fc1']
    z_q = quantize_per_tensor(z, s_in_se, zp_in_se)
    w = h_state['weights_int8']['tcn.0.se.fc1']
    b = h_state['biases_int32']['tcn.0.se.fc1']
    m = h_state['mult']['tcn.0.se.fc1']
    sh = h_state['shift']['tcn.0.se.fc1']
    zp_out = h_state['zp_out']['tcn.0.se.fc1']
    a = fc_int8(z_q, w, b, m, sh, zp_out, in_dim=32, out_dim=4)
    a = relu_int8(a)
    # FC2 — FP32 escape
    s_in_se2 = h_state['scales_in']['tcn.0.se.fc2']
    zp_in_se2 = h_state['zp_in']['tcn.0.se.fc2']
    w_se2 = h_state['weights_int8']['tcn.0.se.fc2']
    s_w_se2 = h_state['scales_w']['tcn.0.se.fc2']
    # We don't have fp32 bias for SE fc2 in the parsed header; recompute by dequant of int32 bias.
    # Actually int32 bias is b_q32[i] = round(b_fp32[i] / (s_in * s_w[i]))
    # So b_fp32[i] ≈ b_q32[i] * (s_in * s_w[i])
    b_q32_se2 = h_state['biases_int32']['tcn.0.se.fc2']
    b_fp32_se2 = b_q32_se2.astype(np.float32) * (s_in_se2 * s_w_se2)
    s = fc_fp32(a, w_se2, s_w_se2, b_fp32_se2, s_in_se2, zp_in_se2)
    s_sig = 1.0 / (1.0 + np.exp(-s))  # (32,)
    # Scale and requantize back
    out0_scaled = out0_c2.astype(np.float32) * s_sig[:, None]  # (32, 16)
    s_in_res = h_state['scales_in']['tcn.0.residual']
    zp_in_res = h_state['zp_in']['tcn.0.residual']
    out0_scaled_q = quantize_per_tensor(out0_scaled, h_state['scales_out']['tcn.0.conv2'], h_state['zp_out']['tcn.0.conv2'])

    # Residual (Block 0): conv 1x1 (23 -> 32)
    w = h_state['weights_int8']['tcn.0.residual']
    b = h_state['biases_int32']['tcn.0.residual']
    m = h_state['mult']['tcn.0.residual']
    sh = h_state['shift']['tcn.0.residual']
    zp_out = h_state['zp_out']['tcn.0.residual']
    res = conv1d_int8_k1(x_q, w, b, m, sh, zp_out, in_ch=23, out_ch=32, T=16)
    out0 = relu_int8(out0_scaled_q + res)

    # Block 1
    w = h_state['weights_int8']['tcn.1.conv1']
    b = h_state['biases_int32']['tcn.1.conv1']
    m = h_state['mult']['tcn.1.conv1']
    sh = h_state['shift']['tcn.1.conv1']
    zp_out = h_state['zp_out']['tcn.1.conv1']
    out1_c1 = conv1d_int8(out0, w, b, m, sh, zp_out,
                          in_ch=32, out_ch=32, T=16, kernel=3, dilation=2)
    out1_c1 = relu_int8(out1_c1)
    w = h_state['weights_int8']['tcn.1.conv2']
    b = h_state['biases_int32']['tcn.1.conv2']
    m = h_state['mult']['tcn.1.conv2']
    sh = h_state['shift']['tcn.1.conv2']
    zp_out = h_state['zp_out']['tcn.1.conv2']
    out1_c2 = conv1d_int8(out1_c1, w, b, m, sh, zp_out,
                          in_ch=32, out_ch=32, T=16, kernel=3, dilation=2)
    out1_c2 = relu_int8(out1_c2)

    z = gap_int32(out1_c2, T=16)
    s_in_se = h_state['scales_in']['tcn.1.se.fc1']
    zp_in_se = h_state['zp_in']['tcn.1.se.fc1']
    z_q = quantize_per_tensor(z, s_in_se, zp_in_se)
    w = h_state['weights_int8']['tcn.1.se.fc1']
    b = h_state['biases_int32']['tcn.1.se.fc1']
    m = h_state['mult']['tcn.1.se.fc1']
    sh = h_state['shift']['tcn.1.se.fc1']
    zp_out = h_state['zp_out']['tcn.1.se.fc1']
    a = fc_int8(z_q, w, b, m, sh, zp_out, in_dim=32, out_dim=4)
    a = relu_int8(a)
    s_in_se2 = h_state['scales_in']['tcn.1.se.fc2']
    zp_in_se2 = h_state['zp_in']['tcn.1.se.fc2']
    w_se2 = h_state['weights_int8']['tcn.1.se.fc2']
    s_w_se2 = h_state['scales_w']['tcn.1.se.fc2']
    b_q32_se2 = h_state['biases_int32']['tcn.1.se.fc2']
    b_fp32_se2 = b_q32_se2.astype(np.float32) * (s_in_se2 * s_w_se2)
    s = fc_fp32(a, w_se2, s_w_se2, b_fp32_se2, s_in_se2, zp_in_se2)
    s_sig = 1.0 / (1.0 + np.exp(-s))
    out1_scaled = out1_c2.astype(np.float32) * s_sig[:, None]
    out1_scaled_q = quantize_per_tensor(out1_scaled, h_state['scales_out']['tcn.1.conv2'], h_state['zp_out']['tcn.1.conv2'])
    out1 = relu_int8(out1_scaled_q + out0)  # Identity residual

    # Block 2
    w = h_state['weights_int8']['tcn.2.conv1']
    b = h_state['biases_int32']['tcn.2.conv1']
    m = h_state['mult']['tcn.2.conv1']
    sh = h_state['shift']['tcn.2.conv1']
    zp_out = h_state['zp_out']['tcn.2.conv1']
    out2_c1 = conv1d_int8(out1, w, b, m, sh, zp_out,
                          in_ch=32, out_ch=32, T=16, kernel=3, dilation=4)
    out2_c1 = relu_int8(out2_c1)
    w = h_state['weights_int8']['tcn.2.conv2']
    b = h_state['biases_int32']['tcn.2.conv2']
    m = h_state['mult']['tcn.2.conv2']
    sh = h_state['shift']['tcn.2.conv2']
    zp_out = h_state['zp_out']['tcn.2.conv2']
    out2_c2 = conv1d_int8(out2_c1, w, b, m, sh, zp_out,
                          in_ch=32, out_ch=32, T=16, kernel=3, dilation=4)
    out2_c2 = relu_int8(out2_c2)

    z = gap_int32(out2_c2, T=16)
    s_in_se = h_state['scales_in']['tcn.2.se.fc1']
    zp_in_se = h_state['zp_in']['tcn.2.se.fc1']
    z_q = quantize_per_tensor(z, s_in_se, zp_in_se)
    w = h_state['weights_int8']['tcn.2.se.fc1']
    b = h_state['biases_int32']['tcn.2.se.fc1']
    m = h_state['mult']['tcn.2.se.fc1']
    sh = h_state['shift']['tcn.2.se.fc1']
    zp_out = h_state['zp_out']['tcn.2.se.fc1']
    a = fc_int8(z_q, w, b, m, sh, zp_out, in_dim=32, out_dim=4)
    a = relu_int8(a)
    s_in_se2 = h_state['scales_in']['tcn.2.se.fc2']
    zp_in_se2 = h_state['zp_in']['tcn.2.se.fc2']
    w_se2 = h_state['weights_int8']['tcn.2.se.fc2']
    s_w_se2 = h_state['scales_w']['tcn.2.se.fc2']
    b_q32_se2 = h_state['biases_int32']['tcn.2.se.fc2']
    b_fp32_se2 = b_q32_se2.astype(np.float32) * (s_in_se2 * s_w_se2)
    s = fc_fp32(a, w_se2, s_w_se2, b_fp32_se2, s_in_se2, zp_in_se2)
    s_sig = 1.0 / (1.0 + np.exp(-s))
    out2_scaled = out2_c2.astype(np.float32) * s_sig[:, None]
    out2_scaled_q = quantize_per_tensor(out2_scaled, h_state['scales_out']['tcn.2.conv2'], h_state['zp_out']['tcn.2.conv2'])
    out2 = relu_int8(out2_scaled_q + out1)  # Identity residual

    # Classifier (full FP32 escape)
    # Dequantize GAP from int8 (tcn.2.conv2 output scale) to FP32
    s_out_tcn2 = h_state['scales_out']['tcn.2.conv2']
    zp_out_tcn2 = h_state['zp_out']['tcn.2.conv2']
    gap_fp = ((out2.astype(np.float32) - zp_out_tcn2) * s_out_tcn2).mean(axis=1)  # (32,) FP32
    # fc1 weight is int8 in this header — dequantize to FP32
    w_fc1 = h_state['weights_int8']['fc1']  # (32, 32) int8
    s_w_fc1 = h_state['scales_w']['fc1']    # (32,) FP32
    b_fp32_fc1 = fp32_biases['fc1.bias']
    h_fc1 = np.zeros(32, dtype=np.float32)
    for od in range(32):
        acc = 0.0
        for id_ in range(32):
            acc += float(w_fc1[od, id_]) * s_w_fc1[od] * gap_fp[id_]
        h_fc1[od] = acc + b_fp32_fc1[od]
    h_fc1 = np.maximum(h_fc1, 0)
    # fc2
    w_fc2 = h_state['weights_int8']['fc2']
    s_w_fc2 = h_state['scales_w']['fc2']
    b_fp32_fc2 = fp32_biases['fc2.bias']
    logit = 0.0
    for id_ in range(32):
        logit += float(w_fc2[0, id_]) * s_w_fc2[0] * h_fc1[id_]
    logit += b_fp32_fc2[0]
    return float(1.0 / (1.0 + np.exp(-logit)))


# ────────────────────────────────────────────────────────────────────────
# PyTorch baseline
# ────────────────────────────────────────────────────────────────────────

def pt_forward(model, x_fp32):
    """PyTorch FP32 baseline. x_fp32: (in_ch, T) -> scalar sigmoid output."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_fp32).unsqueeze(0)  # (1, C, T)
        logit = model(x).item()
    return float(1.0 / (1.0 + np.exp(-logit)))


# ────────────────────────────────────────────────────────────────────────
# Main: per-seed verification
# ────────────────────────────────────────────────────────────────────────

def verify_seed(seed: int, n_samples: int = 50, verbose: bool = True):
    pt_path = os.path.join(BASE, f"model_{TAG_PT}_s{seed}.pt")
    h_path = os.path.join(BASE, "h743", "Core", "Inc",
                           f"model_{TAG_H}_s{seed}.h")
    print(f"\n=== seed={seed} verify ===")
    print(f"  PT: {pt_path}")
    print(f"  Header: {h_path}")

    # Load PT model
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = TCNClassifierSE(in_ch=N_FEATURES)
    model.load_state_dict(sd)
    model.eval()

    # Parse header
    h_state = parse_full_int8_header(h_path)
    print(f"  parsed: {len(h_state['weights_int8'])} int8 weight tensors, "
          f"{len(h_state['biases_int32'])} int32 biases, "
          f"{len(h_state['mult'])} requant params")

    # Get FP32 biases for classifier (fc1.bias, fc2.bias) from PyTorch model
    fp32_biases = {
        'fc1.bias': model.fc1.bias.data.numpy().copy(),
        'fc2.bias': model.fc2.bias.data.numpy().copy(),
    }

    # Load test data + apply KEEP_23 + clip (matches retrain_tcn_23dim_b64_ch32_do01_savept.py)
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
        p_qi = full_int8_forward(x, h_state, fp32_biases)
        pt_preds.append(p_pt)
        qi_preds.append(p_qi)
        diffs.append(abs(p_pt - p_qi))
    dt = time.time() - t0

    pt_preds = np.array(pt_preds)
    qi_preds = np.array(qi_preds)
    diffs = np.array(diffs)

    # Metrics at thr=0.49
    pt_bin = (pt_preds >= 0.49).astype(int)
    qi_bin = (qi_preds >= 0.49).astype(int)
    agreement = (pt_bin == qi_bin).mean()

    print(f"  samples={len(idx)}, time={dt:.1f}s")
    print(f"  max_diff={diffs.max():.4f}  mean_diff={diffs.mean():.4f}  median_diff={np.median(diffs):.4f}")
    print(f"  PT  preds: min={pt_preds.min():.4f} max={pt_preds.max():.4f} mean={pt_preds.mean():.4f}")
    print(f"  QI  preds: min={qi_preds.min():.4f} max={qi_preds.max():.4f} mean={qi_preds.mean():.4f}")
    print(f"  @ thr=0.49: PT positive={pt_bin.sum()}, QI positive={qi_bin.sum()}, agreement={agreement*100:.2f}%")

    # Show first 10 examples
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
    parser.add_argument('--seed', type=int, default=None,
                       help='verify single seed (default: all 5)')
    parser.add_argument('--samples', type=int, default=50,
                       help='number of random samples to test per seed')
    args = parser.parse_args()

    print("=" * 70)
    print(" Full INT8 PC verification: PyTorch FP32 baseline vs fake-quant INT8")
    print("=" * 70)

    seeds = [args.seed] if args.seed else SEEDS
    results = []
    for seed in seeds:
        r = verify_seed(seed, n_samples=args.samples)
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(" Summary (5 seeds at thr=0.49)")
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
          f"agreement>99%: {'PASS' if avg_agree > 0.99 else 'FAIL'}")

    # Save results
    import json
    out = os.path.join(BASE, "fullint8_pc_verify.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_seed": results,
                   "avg_max_diff": avg_max,
                   "avg_mean_diff": avg_mean,
                   "avg_agreement": avg_agree}, f, indent=2)
    print(f"\n-> Wrote {out}")


if __name__ == "__main__":
    main()