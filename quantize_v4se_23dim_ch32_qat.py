#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT-aware Full INT8 quantization for STM32H743 deployment.

Loads QAT-fine-tuned weights (model_v4_se_23dim_b64_ch32_do01_window16_qat_s{seed}.pt),
applies BN-fold + per-channel symmetric INT8 weights + per-tensor symmetric INT8 activations
(zp=0), with 2 × 256-entry sigmoid LUTs (SE + final).

Forks quantize_v4se_23dim_ch32_full_int8.py (PTQ) with:
  - Input: _qat_ .pt files instead of FP32 .pt files
  - Output: model_v4_se_23dim_ch32_qat_s{seed}.h (命名不同, 区分 PTQ vs QAT)
  - Extra: 2 × 256-entry sigmoid LUTs embedded in header (无 INT8 sigmoid kernel)
  - Same 4 bug fixes: clip(±10), symmetric zp=0, int64 requant, GAP dequant

Usage:
    python quantize_v4se_23dim_ch32_qat.py [--seed 42]
    python quantize_v4se_23dim_ch32_qat.py  # all 5 seeds

输出:
    fullint8_qat/h743/Core/Inc/model_v4_se_23dim_ch32_qat_s{seed}.h × 5
    fullint8_qat_ptq_seed{seed}.json × 5  (注意: 文件名沿用 PTQ 模式, content 是 QAT)
"""

import os, json, time, argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
TAG_PT  = "v4_se_23dim_b64_ch32_do01_window16_qat"  # QAT weights (区别于 FP32 的 v4_se_23dim_..._window16)
TAG_H   = "v4_se_23dim_ch32_qat"                      # QAT header 命名
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
HEADER_DIR = os.path.join(BASE, "fullint8_qat", "h743", "Core", "Inc")
LOG_DIR = os.path.join(BASE, "fullint8_qat")


# ═════════════════════════════════════════════════════════════════════
# 模型定义 (与 retrain_tcn_23dim_qat.py 完全一致,确保 load_state_dict 兼容)
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
# 量化函数 (从 quantize_v4se_23dim_ch32_full_int8.py 复用)
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


def quantize_per_tensor_symmetric(min_val, max_val):
    """Per-tensor 对称 INT8 (TFLite default for activations)."""
    absmax = max(abs(min_val), abs(max_val))
    if absmax < 1e-12:
        return 1.0, 0
    scale = absmax / 127.0
    return float(scale), 0


def compute_int32_bias(fp32_bias, input_scale, weight_scale_per_channel):
    """Per-channel INT32 bias."""
    eff = input_scale * weight_scale_per_channel
    eff = np.maximum(np.abs(eff), 1e-12)
    b_q32 = np.round(fp32_bias / eff).astype(np.int32)
    return b_q32


def compute_requant_multiplier_shift(input_scale, weight_scale_per_channel, output_scale):
    """CMSIS-NN 风格 per-channel Requant M/N. Returns (M, N) per channel."""
    multiplier_fp = (input_scale * weight_scale_per_channel) / output_scale
    n_ch = multiplier_fp.size
    mult = np.zeros(n_ch, dtype=np.int32)
    shift = np.zeros(n_ch, dtype=np.int32)
    INT32_MAX = (1 << 31) - 1
    for i, m in enumerate(multiplier_fp):
        if m <= 0.0:
            mult[i] = 0; shift[i] = 0; continue
        if m >= 1.0:
            N = 0; M = INT32_MAX
        else:
            N_f = np.log2(INT32_MAX / m)
            N = int(np.floor(N_f))
            if N > 31: N = 31
            elif N < 0: N = 0
            M = int(round(m * (1 << N)))
            if M > INT32_MAX: M = INT32_MAX
        if M == 0:
            M = 1; N = 0
        mult[i] = np.int32(M)
        shift[i] = np.int32(N)
    return mult, shift


def fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """把 Conv1d + BN1d 折叠成一个等效 Conv1d (bias=True)."""
    w = conv.weight.data.numpy().copy()
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


def calibrate_activation_ranges(model, X_calib):
    """对模型跑 calibration, 收集每层激活的 (min, max)."""
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

    ranges['input'] = (float(X_calib.min()), float(X_calib.max()))

    for i, block in enumerate(model.tcn):
        handles.append(block.conv1.register_forward_hook(make_hook(f'tcn.{i}.conv1')))
        handles.append(block.conv2.register_forward_hook(make_hook(f'tcn.{i}.conv2')))
        if isinstance(block.residual, nn.Conv1d):
            handles.append(block.residual.register_forward_hook(make_hook(f'tcn.{i}.residual')))
        handles.append(block.se.fc1.register_forward_hook(make_hook(f'tcn.{i}.se.fc1')))
        handles.append(block.se.fc2.register_forward_hook(make_hook(f'tcn.{i}.se.fc2')))

    handles.append(model.fc1.register_forward_hook(make_hook('fc1')))
    handles.append(model.fc2.register_forward_hook(make_hook('fc2')))

    model.eval()
    with torch.no_grad():
        BATCH = 64
        for i in range(0, len(X_calib), BATCH):
            xb = torch.from_numpy(X_calib[i:i + BATCH])
            _ = model(xb)

    for h in handles:
        h.remove()
    return ranges


def load_calib_data(n_samples=500):
    """Load training windows + KEEP_23 + clip ±10 + transpose (QAT calib data).

    QAT calib 用 QAT 模型的 forward 收集激活范围 (与 PTQ 同方法).
    """
    import joblib
    KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
               16, 17, 18, 19, 21, 23, 24, 25, 26]
    X_train = np.load(X_TRAIN_NPY).astype(np.float32)[:, :, KEEP_23]
    CLIP_VAL = 10.0
    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL)  # ⭐ Bug fix #1: 必须 clip
    scaler = joblib.load(SCALER_PATH)
    X_calib = X_train[:n_samples].transpose(0, 2, 1).astype(np.float32)
    return X_calib, scaler


# ═════════════════════════════════════════════════════════════════════
# Sigmoid LUT 生成 (256 项 INT8)
# ═════════════════════════════════════════════════════════════════════

def gen_sigmoid_lut(range_amp=2.0, out_amp=1.0):
    """生成 256 项 sigmoid LUT, int8 范围 [-128, 127].

    LUT 输入域: [-range_amp, +range_amp] (FP32)
    LUT 索引: i = 0..255 对应 x = (i - 128) * (2 * range_amp / 256)
    LUT 输出: int8 = clip(round(sigmoid(x) * 127 / out_amp), -128, 127)

    Args:
      range_amp: 输入范围幅度 (e.g. SE 域 = 2.0, final 域 = 8.0)
      out_amp: 输出 sigmoid 范围 (默认 1.0 → [0, 1])

    Returns: list of 256 ints (int8 values)
    """
    entries = []
    step = 2.0 * range_amp / 256.0
    for i in range(256):
        x = (i - 128) * step + step / 2  # 中心采样
        y = 1.0 / (1.0 + math.exp(-x))   # sigmoid
        q = round(y * 127.0 / out_amp)
        entries.append(max(-128, min(127, int(q))))
    return entries


# ═════════════════════════════════════════════════════════════════════
# 完整 QAT 量化 (per seed)
# ═════════════════════════════════════════════════════════════════════

def qat_int8_quantize(seed: int):
    """对 1 个 _qat_ .pt 跑 Full INT8 量化.

    与 PTQ 路径 (full_int8_quantize) 同构, 但:
      - 加载 _qat_ .pt (已 QAT fine-tune)
      - SE fc2 现在也走 INT8 (no FP32 escape) - 因为 QAT 让 fc2 适应量化
      - Classifier fc1/fc2 走 INT8 (no FP32 escape)
      - 2 × sigmoid LUT 嵌入 header
    """
    pt_path = os.path.join(BASE, f"model_{TAG_PT}_s{seed}.pt")
    print(f"\n=== seed={seed} 加载 {pt_path} ===")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"QAT weights not found: {pt_path}\n"
            f"请先运行 retrain_tcn_23dim_qat.py 训练 (用户后续 GPU 执行).")
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
    print(f"  ranges:")
    for k, (mn, mx) in ranges.items():
        print(f"    {k:20s}  min={mn:+.4f}  max={mx:+.4f}  range={mx - mn:.2f}")

    # per-tensor SYMMETRIC activation quant (zp=0) - ⭐ Bug fix #2
    act_q = {}
    for k, (mn, mx) in ranges.items():
        act_q[k] = quantize_per_tensor_symmetric(mn, mx)
    print(f"\n  per-tensor SYMMETRIC activation quant (scale, zp):")
    for k, (s, z) in act_q.items():
        print(f"    {k:20s}  s={s:.6e}  zp={z:4d}")

    # 提取权重 + 折叠 bias
    state = {
        "weights_int8": {},
        "scales": {},
        "biases_fp32": {},
        "biases_int32": {},
        "activations": act_q,
        "requant": {},
        "weights_int8_fc": {},
        "biases_int32_fc": {},
        "scales_fc": {},
        "requant_fc": {},
    }

    layer_meta = [
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
        tail = name.split('.', 2)[2]
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

    fc_layers = [("fc1", model.fc1), ("fc2", model.fc2)]
    for name, fc in fc_layers:
        w_raw = fc.weight.data.numpy().copy()
        b_raw = fc.bias.data.numpy().copy()
        w_q, s_w = quantize_per_channel_symmetric(w_raw)
        state["weights_int8_fc"][f"{name}.weight"] = w_q
        state["scales_fc"][f"{name}.weight"] = s_w
        state["biases_fp32"][f"{name}.bias"] = b_raw.astype(np.float32)

    # ── bias int32 (per-channel) + requant mult/shift
    def input_scale_of(layer_name: str):
        if layer_name == "tcn.0.conv1":
            return act_q["input"][0]
        if "conv2" in layer_name:
            return act_q[f"tcn.{layer_name.split('.')[1]}.conv1"][0]
        if "conv1" in layer_name and layer_name != "tcn.0.conv1":
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i - 1}.conv2"][0]
        if "residual" in layer_name:
            return act_q["input"][0]
        if "se.fc1" in layer_name:
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i}.conv2"][0]
        if "se.fc2" in layer_name:
            return act_q[f"tcn.{layer_name.split('.')[1]}.se.fc1"][0]
        if layer_name == "fc1":
            return act_q["tcn.2.conv2"][0]
        if layer_name == "fc2":
            return act_q["fc1"][0]
        raise ValueError(f"unknown layer: {layer_name}")

    def output_scale_of(layer_name: str):
        if layer_name == "tcn.0.conv1":
            return act_q["tcn.0.conv1"][0]
        if "conv1" in layer_name and layer_name != "tcn.0.conv1":
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i}.conv1"][0]
        if "conv2" in layer_name:
            blk_i = int(layer_name.split('.')[1])
            return act_q[f"tcn.{blk_i}.conv2"][0]
        if "residual" in layer_name:
            return act_q["tcn.0.conv1"][0]
        if "se.fc1" in layer_name:
            return act_q[f"tcn.{layer_name.split('.')[1]}.se.fc1"][0]
        if "se.fc2" in layer_name:
            return act_q[f"tcn.{layer_name.split('.')[1]}.se.fc2"][0]
        if layer_name == "fc1":
            return act_q["fc1"][0]
        if layer_name == "fc2":
            return act_q["fc2"][0]
        raise ValueError(f"unknown layer: {layer_name}")

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
        "method": "QAT full INT8 (PTQ calib on QAT weights)",
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "act_q": {k: [float(v[0]), int(v[1])] for k, v in act_q.items()},
        "layer_sizes": {
            "weights_int8": {k: int(v.size) for k, v in state["weights_int8"].items()},
            "weights_int8_fc": {k: int(v.size) for k, v in state["weights_int8_fc"].items()},
        },
    }
    return state, summary


# ═════════════════════════════════════════════════════════════════════
# C header writer (含 sigmoid LUT)
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


def gen_sigmoid_lut_array_literal(name, lut):
    """256 项 INT8 sigmoid LUT, 16 项/行."""
    out = [f"static const int8_t {name}[256] = {{"]
    row = []
    for i, v in enumerate(lut):
        row.append(f"{int(v):4d}")
        if (i + 1) % 16 == 0 or i == 255:
            out.append("    " + ", ".join(row) + ("," if i != 255 else ""))
            row = []
    out.append("};")
    return "\n".join(out)


def write_qat_header(state, seed, output_path):
    """生成 QAT Full INT8 C header (含 sigmoid LUT)."""
    safe = lambda k: k.replace(".", "_").replace(",", "_")
    lines = []
    lines.append(f"/* Auto-generated by quantize_v4se_23dim_ch32_qat.py")
    lines.append(f" * TCN+SE ch=32 (23-dim SCADA, dropout=0.1) - QAT Full INT8")
    lines.append(f" * seed={seed}")
    lines.append(f" *")
    lines.append(f" * 量化方案 (与 PTQ 路径相同, 但因为 QAT fine-tune, 精度恢复):")
    lines.append(f" *   - 权重 INT8 per-channel symmetric (Flash)")
    lines.append(f" *   - 激活 INT8 per-tensor symmetric (zp=0, 运行期 int8 buffer)")
    lines.append(f" *   - Bias INT32 per-channel (BN-folded)")
    lines.append(f" *   - SE fc2 + sigmoid: 走 INT8 + 256 项 LUT (no FP32 escape!)")
    lines.append(f" *   - Classifier fc1+fc2+sigmoid: 走 INT8 + 256 项 LUT (no FP32 escape!)")
    lines.append(f" *")
    lines.append(f" * H743 C 代码路径: ch32_inference_full_int8_qat.c (手写 INT8, NO CMSIS-NN)")
    lines.append(f" */")
    lines.append(f"#ifndef TCN_V4_SE_23DIM_CH32_QAT_S{seed}_WEIGHTS_H")
    lines.append(f"#define TCN_V4_SE_23DIM_CH32_QAT_S{seed}_WEIGHTS_H")
    lines.append(f"#include <stdint.h>")
    lines.append(f"#ifdef __cplusplus")
    lines.append(f'extern "C" {{')
    lines.append(f"#endif")
    lines.append(f"")

    act_q = state["activations"]

    total_params = sum(v.size for v in state["weights_int8"].values()) + sum(v.size for v in state["weights_int8_fc"].values())
    total_bytes_int8 = total_params
    total_bytes_int32_bias = sum(v.size for v in state["biases_int32"].values()) + sum(v.size for v in state["biases_int32_fc"].values())
    total_bytes_int32_requant = sum(v[0].size + v[1].size for v in state["requant"].values()) + sum(v[0].size + v[1].size for v in state["requant_fc"].values())
    total_bytes_fp32_scales = sum(v.size for v in state["scales"].values()) + sum(v.size for v in state["scales_fc"].values()) * 4
    total_bytes_lut = 2 * 256  # sigmoid_lut_se + sigmoid_lut_final

    lines.append(f"#define TCN_V4_SE_23DIM_CH32_QAT_S{seed}_N_QUANT_TENSORS 15")
    lines.append(f"#define TCN_V4_SE_23DIM_CH32_QAT_S{seed}_TOTAL_PARAMS    {total_params}")
    lines.append(f"/* Flash ≈ {total_bytes_int8/1024:.1f}KB (int8 weights) + "
                 f"{total_bytes_int32_bias/1024:.1f}KB (int32 bias) + "
                 f"{total_bytes_int32_requant/1024:.1f}KB (int32 requant M/S) + "
                 f"{total_bytes_fp32_scales/1024:.1f}KB (FP32 weight scales) + "
                 f"{total_bytes_lut}B (2 × sigmoid LUT) */")
    lines.append(f"")

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
        lines.append(gen_int8_array_literal(f"v4se23_ch32_qat_s{seed}_w_{safe(name)}", w))
        lines.append("")

    # 2) 每层: per-channel int32 bias
    for name in layer_order:
        if name in {"fc1", "fc2"}:
            b = state["biases_int32_fc"][f"{name}.bias"]
        else:
            b = state["biases_int32"][f"{name}.bias"]
        lines.append(f"/* {name}.bias  numel={b.size}  int32 (per-channel) */")
        lines.append(gen_int32_array_literal(f"v4se23_ch32_qat_s{seed}_b_{safe(name)}", b))
        lines.append("")

    # 3) 每层: per-channel FP32 weight scale
    for name in layer_order:
        if name in {"fc1", "fc2"}:
            s_w = state["scales_fc"][f"{name}.weight"]
        else:
            s_w = state["scales"][f"{name}.weight"]
        lines.append(f"/* {name} weight scale  numel={s_w.size} */")
        lines.append(gen_float_array_literal(f"v4se23_ch32_qat_s{seed}_sw_{safe(name)}", s_w))
        lines.append("")

    # 4) Per-tensor activation scale + zp per layer
    lines.append(f"// ===== Per-tensor activation quantization (scale + zp) =====")
    for name in layer_order:
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
            s, zp = act_q[f"tcn.{blk}.conv2"]
        elif "se.fc2" in name:
            blk = int(name.split('.')[1])
            s, zp = act_q[f"tcn.{blk}.se.fc1"]
        elif name == "fc1":
            s, zp = act_q["tcn.2.conv2"]
        elif name == "fc2":
            s, zp = act_q["fc1"]
        lines.append(f"static const float v4se23_ch32_qat_s{seed}_si_{safe(name)}[1] = {{ {s:.6e}f }};")
        lines.append(f"static const int32_t v4se23_ch32_qat_s{seed}_zpi_{safe(name)}[1] = {{ {zp} }};")
    lines.append("")

    # 5) Per-tensor output scale + zp per layer
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
        lines.append(f"static const float v4se23_ch32_qat_s{seed}_so_{safe(name)}[1] = {{ {s:.6e}f }};")
        lines.append(f"static const int32_t v4se23_ch32_qat_s{seed}_zpo_{safe(name)}[1] = {{ {zp} }};")
    lines.append("")

    # 6) Per-channel Requant M/S
    lines.append(f"// ===== Per-channel requant multiplier + shift =====")
    for name in layer_order:
        if name in {"fc1", "fc2"}:
            mult, shift = state["requant_fc"][f"{name}.weight"]
        else:
            mult, shift = state["requant"][f"{name}.weight"]
        lines.append(f"/* {name} requant M/S  numel={mult.size} */")
        lines.append(gen_int32_array_literal(f"v4se23_ch32_qat_s{seed}_mult_{safe(name)}", mult))
        lines.append(gen_int32_array_literal(f"v4se23_ch32_qat_s{seed}_shift_{safe(name)}", shift))
        lines.append("")

    # 7) Sigmoid LUTs (256 项 INT8 each)
    # SE 域: range_amp=2.0 (sigmoid 输入大致在 ±2 内)
    # final 域: range_amp=8.0 (logit 输出大致在 ±8 内)
    sigmoid_lut_se     = gen_sigmoid_lut(range_amp=2.0, out_amp=1.0)
    sigmoid_lut_final  = gen_sigmoid_lut(range_amp=8.0, out_amp=1.0)
    lines.append(f"// ===== Sigmoid LUTs (256 项 INT8 each) =====")
    lines.append(f"/* SE 域 sigmoid LUT: 输入范围 [-2, +2], 输出 [0, 1] */")
    lines.append(gen_sigmoid_lut_array_literal(
        f"v4se23_ch32_qat_s{seed}_sigmoid_lut_se", sigmoid_lut_se))
    lines.append("")
    lines.append(f"/* Final sigmoid LUT: 输入范围 [-8, +8], 输出 [0, 1] */")
    lines.append(gen_sigmoid_lut_array_literal(
        f"v4se23_ch32_qat_s{seed}_sigmoid_lut_final", sigmoid_lut_final))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                       help='quantize single seed (default: all 5)')
    args = parser.parse_args()

    print("=" * 70)
    print(" QAT Full INT8 Quantization")
    print("   加载 QAT .pt → per-channel sym W + per-tensor sym A + sigmoid LUT")
    print("=" * 70)

    os.makedirs(HEADER_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    t_start = time.time()

    summaries = {}
    seeds = [args.seed] if args.seed else SEEDS
    for seed in seeds:
        t0 = time.time()
        state, summary = qat_int8_quantize(seed)
        header_path = os.path.join(HEADER_DIR, f"model_{TAG_H}_s{seed}.h")
        n = write_qat_header(state, seed, header_path)
        size_kb = os.path.getsize(header_path) / 1024
        print(f"\n  -> Wrote {header_path}")
        print(f"     {n} params, {size_kb:.1f} KB")
        summaries[seed] = summary
        print(f"  seed={seed} took {time.time() - t0:.1f}s")

        log_path = os.path.join(LOG_DIR, f"fullint8_qat_ptq_seed{seed}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  -> Wrote {log_path}")

    combined_path = os.path.join(LOG_DIR, "fullint8_qat_ptq_all_seeds.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n-> Wrote {combined_path}")

    print(f"\n[OK] QAT Full INT8 quantization done for {len(seeds)} seeds "
          f"in {time.time() - t_start:.1f}s")
    print(f"   Output: {HEADER_DIR}\\model_{TAG_H}_s{{seed}}.h × {len(seeds)}")
    print(f"   Logs:   {LOG_DIR}\\fullint8_qat_ptq_seed{{seed}}.json × {len(seeds)}")


if __name__ == "__main__":
    main()
