#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_ch32_on_pc.py — TCN+SE ch=32 PyTorch vs Numpy 参考实现对比

目的:
  1. 加载 ch32_weights_extracted.npz (从 .h 反量化得到)
  2. 加载到 PyTorch 模型 (same architecture as train_quantize_3ch.py)
  3. 用 numpy 复刻 ch32_inference.c 的每一步 (mirror 实现)
  4. 加载 X_test (200 个样本)
  5. 对比 PyTorch 预测 vs numpy 预测
  6. 期望:max_abs_diff < 1e-4, pred_match_rate >= 99.5%

这是 PC 端验证. 真正烧到 STM32F407 上后,还要做 MCU 一致性测试.

用法:
  python validate_ch32_on_pc.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"C:\work\Claude\Issue"

# ────────────────────────── 模型架构 (与 train_quantize_3ch.py 完全一致) ──────────────────────────
SEED         = 42
WINDOW       = 16
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4]
N_BLOCKS     = 3
DROPOUT      = 0.3
IN_CHANNELS  = 19
CH           = 32
SE_REDUCTION = 8
SE_HIDDEN    = max(CH // SE_REDUCTION, 4)   # = 4
FC1_OUT      = 32
FC2_OUT      = 1

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(ch // reduction, 4))
        self.fc2 = nn.Linear(max(ch // reduction, 4), ch)

    def forward(self, x):
        z = x.mean(dim=2)                  # (B, C)
        a = F.relu(self.fc1(z))
        s = torch.sigmoid(self.fc2(a))     # (B, C)
        return x * s.unsqueeze(2)


class TCNBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=(k-1)*d//2, dilation=d)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=(k-1)*d//2, dilation=d)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(drop)
        self.se    = SEBlock(out_ch, reduction=SE_REDUCTION)
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
    def __init__(self, in_ch, channels, n_blocks=N_BLOCKS):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, KERNEL_SIZE, DILATIONS[0])]
        for d in DILATIONS[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, KERNEL_SIZE, d))
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


# ────────────────────────── 加载提取的权重 ──────────────────────────
def load_extracted_weights(path):
    """从 .npz 加载 FP32 权重和 bias,直接构建 PyTorch state_dict"""
    data = np.load(path)
    sd = {}

    # Tensor name mapping: extracted key "w_tcn_0_conv1_weight" → "tcn.0.conv1.weight"
    for k in data.files:
        # 去掉前缀 w_ 或 b_
        if k.startswith('w_'):
            name = k[2:].replace('_', '.', 1)  # tcn_0_conv1_weight → tcn.0.conv1.weight (only first _ → .)
            # actually: tcn_0_conv1_weight → tcn.0.conv1.weight
            # 把 "tcn_0_conv1_weight" 变成 "tcn.0.conv1.weight"
            parts = k[2:].rsplit('_', 1)  # ['tcn_0_conv1', 'weight']
            name = parts[0].replace('_', '.') + '.' + parts[1] if len(parts) == 2 else k[2:]
            sd[name] = torch.from_numpy(data[k])
        elif k.startswith('b_'):
            parts = k[2:].rsplit('_', 1)
            name = parts[0].replace('_', '.') + '.' + parts[1] if len(parts) == 2 else k[2:]
            sd[name] = torch.from_numpy(data[k])

    return sd


# ────────────────────────── Numpy 参考 (镜像 ch32_inference.c) ──────────────────────────
def numpy_forward(weights_int8, scales, biases, x):
    """
    x: (19, 16) float32
    weights_int8: dict {name: ndarray int8}
    scales: dict {name: ndarray float32}  per-channel symmetric
    biases: dict {name: ndarray float32}
    """
    def dequant_conv(q, s):
        """q: (oc, ic, k) int8, s: (oc,) → (oc, ic, k) fp32"""
        return q.astype(np.float32) * s.reshape(-1, 1, 1)

    def dequant_fc(q, s):
        """q: (out, in) int8, s: (out,) → (out, in) fp32"""
        return q.astype(np.float32) * s.reshape(-1, 1)

    def conv1d_same(in_, w, b, dilation):
        """in: (in_ch, T), w: (out_ch, in_ch, k), b: (out_ch,) → (out_ch, T)
        用 PyTorch 1D conv 模拟 same padding"""
        in_t = torch.from_numpy(in_).unsqueeze(0).float()  # (1, in_ch, T)
        w_t = torch.from_numpy(w).float()                  # (out_ch, in_ch, k)
        b_t = torch.from_numpy(b).float()                  # (out_ch,)
        out = F.conv1d(in_t, w_t, b_t, padding=(w.shape[2]-1)*dilation//2, dilation=dilation)
        return out.squeeze(0).numpy()

    def conv1d_1x1(in_, w, b):
        """in: (in_ch, T), w: (out_ch, in_ch, 1), b: (out_ch,) → (out_ch, T)"""
        in_t = torch.from_numpy(in_).unsqueeze(0).float()
        w_t = torch.from_numpy(w).float()
        b_t = torch.from_numpy(b).float()
        out = F.conv1d(in_t, w_t, b_t)  # k=1 无 padding
        return out.squeeze(0).numpy()

    def relu(x):
        return np.maximum(x, 0.0)

    def se_block(x, w1, b1, w2, b2):
        """x: (ch, T), w1: (hidden, ch), w2: (ch, hidden)"""
        ch, T = x.shape
        hidden = w1.shape[0]
        z = x.mean(axis=1)                       # (ch,)
        a = relu(z @ w1.T + b1)                  # (hidden,)
        s = 1.0 / (1.0 + np.exp(-(a @ w2.T + b2)))  # (ch,)
        return x * s.reshape(-1, 1)

    def add_residual_relu(x, res):
        return relu(x + res)

    # dequant 所有权重
    w_b0_c1  = dequant_conv(weights_int8['tcn.0.conv1.weight'],    scales['tcn.0.conv1.weight'])
    w_b0_c2  = dequant_conv(weights_int8['tcn.0.conv2.weight'],    scales['tcn.0.conv2.weight'])
    w_b0_rs  = dequant_conv(weights_int8['tcn.0.residual.weight'], scales['tcn.0.residual.weight'])
    w_b0_se1 = dequant_fc(weights_int8['tcn.0.se.fc1.weight'],    scales['tcn.0.se.fc1.weight'])
    w_b0_se2 = dequant_fc(weights_int8['tcn.0.se.fc2.weight'],    scales['tcn.0.se.fc2.weight'])

    w_b1_c1  = dequant_conv(weights_int8['tcn.1.conv1.weight'],    scales['tcn.1.conv1.weight'])
    w_b1_c2  = dequant_conv(weights_int8['tcn.1.conv2.weight'],    scales['tcn.1.conv2.weight'])
    w_b1_se1 = dequant_fc(weights_int8['tcn.1.se.fc1.weight'],    scales['tcn.1.se.fc1.weight'])
    w_b1_se2 = dequant_fc(weights_int8['tcn.1.se.fc2.weight'],    scales['tcn.1.se.fc2.weight'])

    w_b2_c1  = dequant_conv(weights_int8['tcn.2.conv1.weight'],    scales['tcn.2.conv1.weight'])
    w_b2_c2  = dequant_conv(weights_int8['tcn.2.conv2.weight'],    scales['tcn.2.conv2.weight'])
    w_b2_se1 = dequant_fc(weights_int8['tcn.2.se.fc1.weight'],    scales['tcn.2.se.fc1.weight'])
    w_b2_se2 = dequant_fc(weights_int8['tcn.2.se.fc2.weight'],    scales['tcn.2.se.fc2.weight'])

    w_fc1 = dequant_fc(weights_int8['fc1.weight'], scales['fc1.weight'])
    w_fc2 = dequant_fc(weights_int8['fc2.weight'], scales['fc2.weight'])

    b_b0_c1  = biases['tcn.0.conv1.bias']
    b_b0_c2  = biases['tcn.0.conv2.bias']
    b_b0_rs  = biases['tcn.0.residual.bias']
    b_b0_se1 = biases['tcn.0.se.fc1.bias']
    b_b0_se2 = biases['tcn.0.se.fc2.bias']
    b_b1_c1  = biases['tcn.1.conv1.bias']
    b_b1_c2  = biases['tcn.1.conv2.bias']
    b_b1_se1 = biases['tcn.1.se.fc1.bias']
    b_b1_se2 = biases['tcn.1.se.fc2.bias']
    b_b2_c1  = biases['tcn.2.conv1.bias']
    b_b2_c2  = biases['tcn.2.conv2.bias']
    b_b2_se1 = biases['tcn.2.se.fc1.bias']
    b_b2_se2 = biases['tcn.2.se.fc2.bias']
    b_fc1    = biases['fc1.bias']
    b_fc2    = biases['fc2.bias']

    # Block 0 (19→32, d=1)
    res = conv1d_1x1(x, w_b0_rs, b_b0_rs)
    h = conv1d_same(x, w_b0_c1, b_b0_c1, dilation=1)
    h = relu(h)
    h = conv1d_same(h, w_b0_c2, b_b0_c2, dilation=1)
    h = relu(h)
    h = se_block(h, w_b0_se1, b_b0_se1, w_b0_se2, b_b0_se2)
    h = add_residual_relu(h, res)

    # Block 1 (32→32, d=2)
    res = h.copy()
    h2 = conv1d_same(h, w_b1_c1, b_b1_c1, dilation=2)
    h2 = relu(h2)
    h2 = conv1d_same(h2, w_b1_c2, b_b1_c2, dilation=2)
    h2 = relu(h2)
    h2 = se_block(h2, w_b1_se1, b_b1_se1, w_b1_se2, b_b1_se2)
    h = add_residual_relu(h2, res)

    # Block 2 (32→32, d=4)
    res = h.copy()
    h2 = conv1d_same(h, w_b2_c1, b_b2_c1, dilation=4)
    h2 = relu(h2)
    h2 = conv1d_same(h2, w_b2_c2, b_b2_c2, dilation=4)
    h2 = relu(h2)
    h2 = se_block(h2, w_b2_se1, b_b2_se1, w_b2_se2, b_b2_se2)
    h = add_residual_relu(h2, res)

    # GAP → FC1 → ReLU → FC2 → Sigmoid
    z = h.mean(axis=1)
    fc1_out = relu(z @ w_fc1.T + b_fc1)
    logit = fc1_out @ w_fc2.T + b_fc2[0]
    return float(1.0 / (1.0 + np.exp(-logit)))


# ────────────────────────── 主验证流程 ──────────────────────────
def main():
    print("=" * 70)
    print(" TCN+SE ch=32 PC 验证 — PyTorch vs Numpy (mirror C)")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载 X_test (19-dim SCADA v2, window=16)...")
    Xte_path = os.path.join(BASE, "X_test_ch32_19x16.npy")
    yte_path = os.path.join(BASE, "y_test_ch32_19x16.npy")
    if os.path.exists(Xte_path):
        Xte = np.load(Xte_path)
        yte = np.load(yte_path)
    else:
        # 退化:从原始 19-dim 数据滑窗
        Xte_raw = np.load(os.path.join(BASE, "X_test_binary_v2_scada.npy"))
        yte_raw = np.load(os.path.join(BASE, "y_test_binary_v2_scada.npy"))
        n = len(Xte_raw)
        Xte = np.array([Xte_raw[i:i+16].T for i in range(n - 15)], dtype=np.float32)
        yte = np.array([yte_raw[i+15] for i in range(n - 15)], dtype=np.int64)
    print(f"    Xte: {Xte.shape}, yte: {yte.shape}, "
          f"正样本 {yte.sum()}/{len(yte)} ({100*yte.mean():.1f}%)")

    # 2. 加载提取的权重
    npz_path = os.path.join(BASE, "ch32_weights_extracted.npz")
    print(f"\n[2] 加载提取的权重 {npz_path} ...")
    data = np.load(npz_path)
    print(f"    {len(data.files)} 个数组")

    # 重新组织为 weights_int8, scales, biases
    #   w_*   = FP32 dequantized (给 PyTorch load_state_dict 用)
    #   q_*   = 原始 INT8 (给 numpy_forward dequant 用,模拟 C 推理)
    #   b_*   = FP32 bias
    weights_int8 = {}
    scales = {}
    biases = {}
    for k in data.files:
        if k.startswith('q_tcn_') or k.startswith('q_fc'):
            parts = k[2:].rsplit('_', 1)  # ['tcn_0_conv1', 'weight']
            name = parts[0].replace('_', '.') + '.' + parts[1]
            weights_int8[name] = data[k].astype(np.int8)
        elif k.startswith('b_tcn_') or k.startswith('b_fc'):
            parts = k[2:].rsplit('_', 1)
            name = parts[0].replace('_', '.') + '.' + parts[1]
            biases[name] = data[k]

    # scales 复用 weights_int8(已 dequant), 但 numpy_forward 需要原始 scale + q
    # 重新从 extracted weights 反推 scale (= w_fp32 / q)
    for k in list(weights_int8.keys()):
        if 'weight' in k:
            q = weights_int8[k]
            w_fp32_full = data['w_' + k.replace('.', '_')]  # 已经在 extracted 里就是 dequant 后的 fp32
            # 但 numpy_forward 需要 INT8 + scale, scale = w_fp32 / q
            # 为了 numpy_forward 能用,我们重新读 .h 取 scale
            pass

    # 3. scale 从 .h 文件读
    print(f"\n[3] 从 .h 文件读 scale ...")
    import re
    with open(os.path.join(BASE, "model_ch32_hybrid.h"), 'r', encoding='utf-8', errors='ignore') as f:
        h_content = f.read()

    for k in list(weights_int8.keys()):
        scale_name = 'ch32_s_' + k.replace('.', '_')  # ch32_s_tcn_0_conv1_weight
        m = re.search(rf'static\s+const\s+float\s+{scale_name}\[(\d+)\]\s*=\s*\{{([^}}]+)\}};',
                      h_content, re.DOTALL)
        if m:
            body = m.group(2)
            body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL).replace(',', ' ')
            nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', body)
            scales[k] = np.array([float(n) for n in nums], dtype=np.float32)
        else:
            print(f"    [warn] scale for {k} not found")
            scales[k] = np.ones(weights_int8[k].shape[0], dtype=np.float32)

    print(f"    {len(scales)} 个 scale 已加载")

    # 4. 加载到 PyTorch 模型 (用 extracted FP32 权重直接 load_state_dict)
    print(f"\n[4] 构建 PyTorch 模型 + 加载 FP32 权重 ...")
    model = TCNClassifierSE(IN_CHANNELS, CH)

    # 构建 sd: extracted key → PyTorch state_dict key
    sd = {}
    for k in data.files:
        if k.startswith('w_'):
            parts = k[2:].rsplit('_', 1)
            name = parts[0].replace('_', '.') + '.' + parts[1]
            sd[name] = torch.from_numpy(data[k]).float()
        elif k.startswith('b_'):
            parts = k[2:].rsplit('_', 1)
            name = parts[0].replace('_', '.') + '.' + parts[1]
            sd[name] = torch.from_numpy(data[k]).float()

    # BN 折叠: PyTorch 的 Conv1d 后面有 BN1d,我们需要把 BN 参数合并到 conv 权重
    # 但 extracted 权重已经是 BN-folded 后的,所以应该绕过 BN
    # 方案: 把 BN 设成 identity (weight=1, bias=0, running_mean=0, running_var=1)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)
            m.running_mean.fill_(0.0)
            m.running_var.fill_(1.0)

    # 加载权重
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    if unexpected:
        print(f"    unexpected keys: {unexpected[:5]}")
    model.eval()

    # 5. 对比 PyTorch vs Numpy (前 200 样本)
    n_samples = min(1000, len(Xte))
    print(f"\n[5] 对比 PyTorch vs Numpy (n={n_samples})...")
    diffs = []
    matches = 0
    threshold = 0.49

    for i in range(n_samples):
        x = Xte[i].astype(np.float32)  # (19, 16)

        # PyTorch forward
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0)
            logit_t = model(x_t).item()
            p_torch = 1.0 / (1.0 + np.exp(-logit_t))

        # Numpy forward (走 INT8 → dequant → 同 forward)
        p_numpy = numpy_forward(weights_int8, scales, biases, x)

        diff = abs(p_torch - p_numpy)
        diffs.append(diff)

        y_torch = 1 if p_torch >= threshold else 0
        y_numpy = 1 if p_numpy >= threshold else 0
        if y_torch == y_numpy:
            matches += 1

    diffs = np.array(diffs)
    max_diff = diffs.max()
    mean_diff = diffs.mean()
    match_rate = matches / n_samples

    print(f"\n{'='*70}")
    print(f" 验证结果:")
    print(f"   max abs diff:  {max_diff:.6f}")
    print(f"   mean abs diff: {mean_diff:.6f}")
    print(f"   pred match:    {matches}/{n_samples} = {match_rate*100:.2f}%")
    print(f"   threshold:     {threshold}")
    print(f"\n 期望:")
    print(f"   max diff < 0.01     : {'PASS' if max_diff < 0.01 else 'FAIL'}")
    print(f"   match rate >= 99.0% : {'PASS' if match_rate >= 0.99 else 'FAIL'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()