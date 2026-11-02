#!/usr/bin/env python3
"""验证 H743 上的 TCN+SE 模型与 PC 端 .pt 是否一致.

检查:
  1. 架构 (各层 shape)
  2. Conv 权重 (BN folded) 是否一致
  3. SE fc1/fc2 权重是否一致
  4. fc1/fc2 FP32 权重是否一致
  5. 推理输出 (200 个测试样本) 是否 bit-perfect (max_diff=0)
"""

import os, sys, re, struct
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE = r"C:\work\Claude\Issue"
SEED = 42
PC_MODEL_PATH = f"{BASE}/model_v4_se_23dim_b64_ch32_do01_window16_s{SEED}.pt"
H743_HEADER_PATH = f"{BASE}/h743/Core/Inc/model_v4_se_23dim_ch32_hybrid_s{SEED}.h"

# ── PC 端模型定义 ──────────────────────
N_FEATURES = 23
WINDOW = 16
CHANNELS = 32
DROPOUT = 0.1
SE_REDUCTION = 8
N_BLOCKS = 3
DILATIONS = [1, 2, 4]
KERNEL_SIZE = 3


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


def load_pc_model():
    ckpt = torch.load(PC_MODEL_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    model = TCNClassifierSE(N_FEATURES)
    model.load_state_dict(sd)
    model.eval()
    return model


def fold_bn_in_model(model: TCNClassifierSE):
    folded = {}
    for i, block in enumerate(model.tcn):
        for c_idx, (conv, bn) in enumerate([(block.conv1, block.bn1), (block.conv2, block.bn2)]):
            w = conv.weight.data.numpy()
            b = conv.bias.data.numpy() if conv.bias is not None else np.zeros(conv.out_channels)
            bn_w = bn.weight.data.numpy()
            bn_b = bn.bias.data.numpy()
            bn_mean = bn.running_mean.numpy()
            bn_var = bn.running_var.numpy()
            bn_eps = bn.eps
            scale = bn_w / np.sqrt(bn_var + bn_eps)
            layer_name = 'conv1' if c_idx == 0 else 'conv2'
            folded[f'tcn.{i}.{layer_name}.weight'] = (w * scale[:, None, None]).astype(np.float32)
            folded[f'tcn.{i}.{layer_name}.bias']   = ((b - bn_mean) * scale + bn_b).astype(np.float32)
        # SE fc1/fc2
        folded[f'tcn.{i}.se.fc1.weight'] = block.se.fc1.weight.data.numpy().copy()
        folded[f'tcn.{i}.se.fc1.bias']   = block.se.fc1.bias.data.numpy().copy()
        folded[f'tcn.{i}.se.fc2.weight'] = block.se.fc2.weight.data.numpy().copy()
        folded[f'tcn.{i}.se.fc2.bias']   = block.se.fc2.bias.data.numpy().copy()
        # Residual
        if isinstance(block.residual, nn.Conv1d):
            folded[f'tcn.{i}.residual.weight'] = block.residual.weight.data.numpy().copy()
            r_bias = block.residual.bias.data.numpy() if block.residual.bias is not None else np.zeros(block.residual.out_channels)
            folded[f'tcn.{i}.residual.bias'] = r_bias.copy()
    folded['fc1.weight'] = model.fc1.weight.data.numpy().copy()
    folded['fc1.bias']   = model.fc1.bias.data.numpy().copy()
    folded['fc2.weight'] = model.fc2.weight.data.numpy().copy()
    folded['fc2.bias']   = model.fc2.bias.data.numpy().copy()
    return folded


def quantize_per_channel_symmetric(w: np.ndarray):
    out_ch = w.shape[0]
    w_flat = w.reshape(out_ch, -1)
    absmax = np.max(np.abs(w_flat), axis=1)
    absmax = np.maximum(absmax, 1e-8)
    scale = absmax / 127.0
    w_q = np.clip(np.round(w_flat / scale[:, None]), -128, 127).astype(np.int8)
    return w_q.reshape(w.shape), scale.astype(np.float32)


# 各层 shape 表 (flattened 数组需要 reshape 回此形状)
INT8_SHAPES = {
    'tcn.0.conv1.weight':    (32, 23, 3),
    'tcn.0.conv2.weight':    (32, 32, 3),
    'tcn.0.residual.weight': (32, 23, 1),
    'tcn.0.se.fc1.weight':   (4, 32),
    'tcn.0.se.fc2.weight':   (32, 4),
    'tcn.1.conv1.weight':    (32, 32, 3),
    'tcn.1.conv2.weight':    (32, 32, 3),
    'tcn.1.se.fc1.weight':   (4, 32),
    'tcn.1.se.fc2.weight':   (32, 4),
    'tcn.2.conv1.weight':    (32, 32, 3),
    'tcn.2.conv2.weight':    (32, 32, 3),
    'tcn.2.se.fc1.weight':   (4, 32),
    'tcn.2.se.fc2.weight':   (32, 4),
}
SCALE_SHAPES = {k: (v[0],) for k, v in INT8_SHAPES.items()}  # per out_ch

BIAS_SHAPES = {
    'tcn.0.conv1.bias': (32,),
    'tcn.0.conv2.bias': (32,),
    'tcn.0.residual.bias': (32,),
    'tcn.0.se.fc1.bias': (4,),
    'tcn.0.se.fc2.bias': (32,),
    'tcn.1.conv1.bias': (32,),
    'tcn.1.conv2.bias': (32,),
    'tcn.1.se.fc1.bias': (4,),
    'tcn.1.se.fc2.bias': (32,),
    'tcn.2.conv1.bias': (32,),
    'tcn.2.conv2.bias': (32,),
    'tcn.2.se.fc1.bias': (4,),
    'tcn.2.se.fc2.bias': (32,),
    'fc1.bias': (32,),
    'fc2.bias': (1,),
}

FP32_W_SHAPES = {
    'fc1.weight': (32, 32),
    'fc2.weight': (1, 32),
}


def parse_h743_header(path):
    """按行解析 .h 文件,提取 weight/scale/bias 数组."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    weights_int8 = {}
    scales = {}
    weights_fp32 = {}
    biases_fp32 = {}

    # 用更精确的正则: 按声明行 + {...} 内容分别匹配
    int8_pat = re.compile(
        r'static\s+const\s+int8_t\s+(v4se23_ch32_s\d+_w_(tcn_\d+_(?:conv1|conv2|residual|se_fc1|se_fc2)_weight))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in int8_pat.finditer(text):
        name = m.group(1)
        flat_size = int(m.group(3))
        raw = m.group(4)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        rest = name.split('_w_')[1]
        # tcn_0_conv1_weight / tcn_0_se_fc1_weight / tcn_0_residual_weight
        if 'se' in rest.split('_'):
            parts = rest.split('_')
            blk = parts[1]
            fc = parts[3]
            key = f'tcn.{blk}.se.{fc}.weight'
        elif 'residual' in rest:
            blk = rest.split('_')[1]
            key = f'tcn.{blk}.residual.weight'
        else:
            parts = rest.split('_')
            blk = parts[1]
            layer = parts[2]
            key = f'tcn.{blk}.{layer}.weight'
        weights_int8[key] = np.array(vals, dtype=np.int8)

    # float scale: 注意 'tcn_X_se_fc1_weight' 包含两个 '_' 区块,需要更宽松的匹配
    scale_pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_s\d+_s_(tcn_\d+_(?:conv1|conv2|residual|se_fc1|se_fc2)_weight))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in scale_pat.finditer(text):
        name = m.group(1)
        flat_size = int(m.group(3))
        raw = m.group(4)
        vals = [float(x.replace('f', '')) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?f?', raw)]
        rest = name.split('_s_')[1]
        if 'se' in rest.split('_'):
            parts = rest.split('_')
            blk = parts[1]
            fc = parts[3]
            key = f'tcn.{blk}.se.{fc}.weight'
        elif 'residual' in rest:
            blk = rest.split('_')[1]
            key = f'tcn.{blk}.residual.weight'
        else:
            parts = rest.split('_')
            blk = parts[1]
            layer = parts[2]
            key = f'tcn.{blk}.{layer}.weight'
        scales[key] = np.array(vals, dtype=np.float32)

    # FP32 weight (fc1, fc2)
    fp32w_pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_s\d+_(fc\d+_weight))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in fp32w_pat.finditer(text):
        name = m.group(1)
        flat_size = int(m.group(3))
        raw = m.group(4)
        vals = [float(x.replace('f', '')) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?f?', raw)]
        weights_fp32[name.replace(f'v4se23_ch32_s{SEED}_', '')] = np.array(vals, dtype=np.float32)

    # FP32 bias (conv/residual/se/fc)
    bias_pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_s\d+_b_(?:tcn_\d+_(?:conv1|conv2|residual|se_fc1|se_fc2)_bias|fc\d+_bias))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in bias_pat.finditer(text):
        full_name = m.group(1)
        flat_size = int(m.group(2))
        raw = m.group(3)
        vals = [float(x.replace('f', '')) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?f?', raw)]
        # 把 v4se23_ch32_s42_b_tcn_0_conv1_bias 转成 tcn.0.conv1.bias
        body = full_name.replace(f'v4se23_ch32_s{SEED}_b_', '')
        if body.startswith('fc'):
            # fc1_bias / fc2_bias
            key = body.replace('_bias', '.bias')  # fc1.bias
        else:
            # tcn_0_conv1_bias / tcn_0_se_fc1_bias / tcn_0_residual_bias
            parts = body.split('_')  # ['tcn', '0', 'conv1', 'bias'] 或 ['tcn', '0', 'se', 'fc1', 'bias'] 或 ['tcn', '0', 'residual', 'bias']
            blk = parts[1]
            if 'se' in parts:
                fc = parts[3]
                key = f'tcn.{blk}.se.{fc}.bias'
            elif 'residual' in parts:
                key = f'tcn.{blk}.residual.bias'
            else:
                layer = parts[2]
                key = f'tcn.{blk}.{layer}.bias'
        biases_fp32[key] = np.array(vals, dtype=np.float32)

    return weights_int8, scales, weights_fp32, biases_fp32


def build_h743_equivalent_model(h_w_int8, h_scales, h_w_fp32, h_b_fp32):
    """用 .h 解析出来的权重构建一个 PyTorch 等效模型."""
    def dequant(key):
        q = h_w_int8[key]
        s = h_scales[key]
        shape = INT8_SHAPES[key]
        q_reshaped = q.reshape(shape)
        s_reshaped = s.reshape((shape[0],) + (1,) * (len(shape) - 1))
        return (q_reshaped.astype(np.float32) * s_reshaped).astype(np.float32)

    model = TCNClassifierSE(N_FEATURES)
    model.eval()
    sd = model.state_dict()

    for i in range(3):
        for layer in ['conv1', 'conv2']:
            w_key = f'tcn.{i}.{layer}.weight'
            b_key = f'tcn.{i}.{layer}.bias'
            sd[w_key] = torch.from_numpy(dequant(w_key))
            sd[b_key] = torch.from_numpy(h_b_fp32[b_key])
            # 删除 BN
            bn_idx = layer[-1]  # '1' or '2'
            for bn_key in [f'tcn.{i}.bn{bn_idx}.weight', f'tcn.{i}.bn{bn_idx}.bias',
                           f'tcn.{i}.bn{bn_idx}.running_mean', f'tcn.{i}.bn{bn_idx}.running_var',
                           f'tcn.{i}.bn{bn_idx}.num_batches_tracked']:
                sd.pop(bn_key, None)
        # SE
        for fc in ['fc1', 'fc2']:
            w_key = f'tcn.{i}.se.{fc}.weight'
            b_key = f'tcn.{i}.se.{fc}.bias'
            sd[w_key] = torch.from_numpy(dequant(w_key))
            sd[b_key] = torch.from_numpy(h_b_fp32[b_key])
        # Residual (仅 block 0)
        if i == 0:
            sd['tcn.0.residual.weight'] = torch.from_numpy(dequant('tcn.0.residual.weight'))
            sd['tcn.0.residual.bias']   = torch.from_numpy(h_b_fp32['tcn.0.residual.bias'])

    sd['fc1.weight'] = torch.from_numpy(h_w_fp32['fc1_weight'].reshape(32, 32))
    sd['fc1.bias']   = torch.from_numpy(h_b_fp32['fc1.bias'])
    sd['fc2.weight'] = torch.from_numpy(h_w_fp32['fc2_weight'].reshape(1, 32))
    sd['fc2.bias']   = torch.from_numpy(h_b_fp32['fc2.bias'])

    model.load_state_dict(sd, strict=False)
    return model


def main():
    print(f"PC model:    {PC_MODEL_PATH}")
    print(f"H743 header: {H743_HEADER_PATH}")
    print()

    # 1. Load PC model + BN fold
    print("=" * 70)
    print("[1] 加载 PC 端 .pt 并 BN fold")
    print("=" * 70)
    pc_model = load_pc_model()
    pc_folded = fold_bn_in_model(pc_model)
    print(f"  PC folded state keys: {len(pc_folded)}")

    # 2. Parse H743 header
    print()
    print("=" * 70)
    print("[2] 解析 H743 .h 头文件")
    print("=" * 70)
    h_w_int8, h_scales, h_w_fp32, h_b_fp32 = parse_h743_header(H743_HEADER_PATH)
    print(f"  H743 INT8 weights:  {len(h_w_int8)} tensors  keys={list(h_w_int8.keys())[:3]}...")
    print(f"  H743 scales:        {len(h_scales)} tensors")
    print(f"  H743 FP32 weights:  {len(h_w_fp32)} tensors  keys={list(h_w_fp32.keys())}")
    print(f"  H743 FP32 biases:   {len(h_b_fp32)} tensors  keys={list(h_b_fp32.keys())}")

    # 3. Compare weights
    print()
    print("=" * 70)
    print("[3] 比对每一层权重 (PC BN-folded FP32 vs H743 INT8+dequant 或 FP32)")
    print("=" * 70)
    max_diff_per_layer = {}
    for key, pc_w in pc_folded.items():
        if key in h_w_int8:
            pc_q, pc_s = quantize_per_channel_symmetric(pc_w)
            h_q = h_w_int8[key].reshape(INT8_SHAPES[key])
            h_s = h_scales[key]
            # 反量化 FP32
            pc_deq = pc_q.astype(np.float32) * pc_s.reshape((pc_q.shape[0],) + (1,) * (pc_q.ndim - 1))
            h_deq = h_q.astype(np.float32) * h_s.reshape((h_q.shape[0],) + (1,) * (h_q.ndim - 1))
            diff_deq = np.abs(pc_deq - h_deq).max()
            # INT8 round diff
            diff_int8 = np.abs(pc_q.astype(int) - h_q.astype(int)).max()
            max_diff_per_layer[key] = diff_deq
            status = "[OK]" if diff_deq < 1e-3 else ("[~]" if diff_deq < 1e-2 else "[X]")
            print(f"  {status} {key:30s} shape={tuple(pc_q.shape)} INT8_diff={diff_int8} dequant_diff={diff_deq:.6f}")
        elif key in h_w_fp32 or key.replace('.', '_') in h_w_fp32:
            lookup_key = key if key in h_w_fp32 else key.replace('.', '_')
            h_w_flat = h_w_fp32[lookup_key]
            target_shape = FP32_W_SHAPES.get(key, h_w_flat.shape)
            h_w = h_w_flat.reshape(target_shape) if h_w_flat.shape != target_shape else h_w_flat
            if pc_w.shape != h_w.shape:
                print(f"  [X]  {key:30s} shape mismatch: PC={pc_w.shape} H743={h_w.shape}")
                continue
            diff = np.abs(pc_w - h_w).max()
            max_diff_per_layer[key] = diff
            status = "[OK]" if diff < 1e-4 else ("[~]" if diff < 1e-3 else "[X]")
            print(f"  {status} {key:30s} shape={tuple(pc_w.shape)} FP32 max_diff={diff:.6e}")
        elif key in h_b_fp32:
            h_b = h_b_fp32[key]
            if pc_w.shape != h_b.shape:
                print(f"  [X]  {key:30s} shape mismatch: PC={pc_w.shape} H743={h_b.shape}")
                continue
            diff = np.abs(pc_w - h_b).max()
            max_diff_per_layer[key] = diff
            status = "[OK]" if diff < 1e-4 else ("[~]" if diff < 1e-3 else "[X]")
            print(f"  {status} {key:30s} (bias) shape={tuple(pc_w.shape)} max_diff={diff:.6e}")
        else:
            print(f"  [X]  {key:30s} NOT FOUND in H743 header")

    # 4. Forward pass check
    print()
    print("=" * 70)
    print("[4] 前向推理比对 (200 个随机样本)")
    print("=" * 70)
    np.random.seed(0)
    x = np.random.randn(200, N_FEATURES, WINDOW).astype(np.float32)
    with torch.no_grad():
        pc_out = torch.sigmoid(pc_model(torch.from_numpy(x))).numpy()
    h743_model = build_h743_equivalent_model(h_w_int8, h_scales, h_w_fp32, h_b_fp32)
    with torch.no_grad():
        h743_out = torch.sigmoid(h743_model(torch.from_numpy(x))).numpy()
    max_diff = np.abs(pc_out - h743_out).max()
    mean_diff = np.abs(pc_out - h743_out).mean()
    print(f"  Max diff:    {max_diff:.6e}")
    print(f"  Mean diff:   {mean_diff:.6e}")
    print(f"  判定: {'BIT-PERFECT' if max_diff < 1e-6 else 'CLOSE' if max_diff < 1e-3 else 'DIFFERENT'}")

    # 5. Summary
    print()
    print("=" * 70)
    print("[5] 总结")
    print("=" * 70)
    overall_max = max(max_diff_per_layer.values()) if max_diff_per_layer else 0
    print(f"  权重层最大 diff: {overall_max:.6e}")
    print(f"  前向推理最大 diff: {max_diff:.6e}")
    if max_diff < 1e-3:
        print("  [OK] H743 模型与 PC 端一致 (允许 INT8 quant 误差)")
    else:
        print("  [X] H743 模型与 PC 端存在显著差异,需要进一步排查")


if __name__ == '__main__':
    main()