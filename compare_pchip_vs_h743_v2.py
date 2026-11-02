#!/usr/bin/env python3
"""compare_hybrid_int8_vs_h743_actual.py 的修补版.

修补点:
1. CSV 路径的 'label_binary' 改成 'binary result'
2. 17 列原始特征不够, 改用 .npy 训练数据 (27-dim) + KEEP_23 切片
3. 数据集用 X_test_binary_v2_scada_window16.npy (3432 windows) 而不是 17164

目的: 验证 H743 INT8 实现 vs PC Hybrid INT8 (dequant) 是否有差距.
"""
import io
import sys
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r"C:\work\Claude\Issue"
SEED = 42
N_FEATURES = 23
WINDOW = 16
CHANNELS = 32
DILATIONS = [1, 2, 4]

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]

# ─── PC 模型定义 (跟 H743 端 ch32_inference.c 等价) ───

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
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
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return F.relu(x + residual)


class TCNSE(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlockSE(in_ch, CHANNELS, 3, DILATIONS[0], 0.1),
            TCNBlockSE(CHANNELS, CHANNELS, 3, DILATIONS[1], 0.1),
            TCNBlockSE(CHANNELS, CHANNELS, 3, DILATIONS[2], 0.1),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(CHANNELS, 32)
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)

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


def parse_h743_header(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    weights_int8, scales, weights_fp32, biases_fp32 = {}, {}, {}, {}

    int8_pat = re.compile(
        r'static\s+const\s+int8_t\s+(v4se23_ch32_s\d+_w_(?:tcn_\d+_(?:conv1|conv2|residual|se_fc1|se_fc2)_weight))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in int8_pat.finditer(text):
        name = m.group(1)
        raw = m.group(3)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        rest = name.split('_w_')[1]
        if 'se' in rest.split('_'):
            parts = rest.split('_')
            blk, fc = parts[1], parts[3]
            key = f'tcn.{blk}.se.{fc}.weight'
        elif 'residual' in rest:
            blk = rest.split('_')[1]
            key = f'tcn.{blk}.residual.weight'
        else:
            parts = rest.split('_')
            blk, layer = parts[1], parts[2]
            key = f'tcn.{blk}.{layer}.weight'
        weights_int8[key] = np.array(vals, dtype=np.int8)

    scale_pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_s\d+_s_(?:tcn_\d+_(?:conv1|conv2|residual|se_fc1|se_fc2)_weight))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in scale_pat.finditer(text):
        name = m.group(1)
        raw = m.group(3)
        vals = [float(x.replace('f', '')) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?f?', raw)]
        rest = name.split('_s_')[1]
        if 'se' in rest.split('_'):
            parts = rest.split('_')
            blk, fc = parts[1], parts[3]
            key = f'tcn.{blk}.se.{fc}.weight'
        elif 'residual' in rest:
            blk = rest.split('_')[1]
            key = f'tcn.{blk}.residual.weight'
        else:
            parts = rest.split('_')
            blk, layer = parts[1], parts[2]
            key = f'tcn.{blk}.{layer}.weight'
        scales[key] = np.array(vals, dtype=np.float32)

    fp32w_pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_s\d+_(?:fc\d+_weight))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in fp32w_pat.finditer(text):
        name = m.group(1)
        raw = m.group(3)
        vals = [float(x.replace('f', '')) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?f?', raw)]
        weights_fp32[name.replace(f'v4se23_ch32_s{SEED}_', '')] = np.array(vals, dtype=np.float32)

    bias_pat = re.compile(
        r'static\s+const\s+float\s+(v4se23_ch32_s\d+_b_(?:tcn_\d+_(?:conv1|conv2|residual|se_fc1|se_fc2)_bias|fc\d+_bias))\[(\d+)\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    for m in bias_pat.finditer(text):
        full_name = m.group(1)
        raw = m.group(3)
        vals = [float(x.replace('f', '')) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?f?', raw)]
        body = full_name.replace(f'v4se23_ch32_s{SEED}_b_', '')
        if body.startswith('fc'):
            key = body.replace('_bias', '.bias')
        else:
            parts = body.split('_')
            blk = parts[1]
            if 'se' in parts:
                key = f'tcn.{blk}.se.{parts[3]}.bias'
            elif 'residual' in parts:
                key = f'tcn.{blk}.residual.bias'
            else:
                key = f'tcn.{blk}.{parts[2]}.bias'
        biases_fp32[key] = np.array(vals, dtype=np.float32)

    return weights_int8, scales, weights_fp32, biases_fp32


def build_hybrid_model(h_w_int8, h_scales, h_w_fp32, h_b_fp32):
    def dequant(key):
        q = h_w_int8[key]
        s = h_scales[key]
        shape = INT8_SHAPES[key]
        q_reshaped = q.reshape(shape)
        s_reshaped = s.reshape((shape[0],) + (1,) * (len(shape) - 1))
        return (q_reshaped.astype(np.float32) * s_reshaped).astype(np.float32)

    model = TCNSE(N_FEATURES)
    model.eval()
    sd = model.state_dict()

    for i in range(3):
        for layer in ['conv1', 'conv2']:
            w_key = f'tcn.{i}.{layer}.weight'
            b_key = f'tcn.{i}.{layer}.bias'
            sd[w_key] = torch.from_numpy(dequant(w_key))
            sd[b_key] = torch.from_numpy(h_b_fp32[b_key])
            bn_idx = layer[-1]
            for bn_key in [f'tcn.{i}.bn{bn_idx}.weight', f'tcn.{i}.bn{bn_idx}.bias',
                           f'tcn.{i}.bn{bn_idx}.running_mean', f'tcn.{i}.bn{bn_idx}.running_var',
                           f'tcn.{i}.bn{bn_idx}.num_batches_tracked']:
                sd.pop(bn_key, None)
        for fc in ['fc1', 'fc2']:
            w_key = f'tcn.{i}.se.{fc}.weight'
            b_key = f'tcn.{i}.se.{fc}.bias'
            sd[w_key] = torch.from_numpy(dequant(w_key))
            sd[b_key] = torch.from_numpy(h_b_fp32[b_key])
        if i == 0:
            sd['tcn.0.residual.weight'] = torch.from_numpy(dequant('tcn.0.residual.weight'))
            sd['tcn.0.residual.bias']   = torch.from_numpy(h_b_fp32['tcn.0.residual.bias'])

    sd['fc1.weight'] = torch.from_numpy(h_w_fp32['fc1_weight'].reshape(32, 32))
    sd['fc1.bias']   = torch.from_numpy(h_b_fp32['fc1.bias'])
    sd['fc2.weight'] = torch.from_numpy(h_w_fp32['fc2_weight'].reshape(1, 32))
    sd['fc2.bias']   = torch.from_numpy(h_b_fp32['fc2.bias'])
    model.load_state_dict(sd, strict=False)
    return model


def hybrid_forward_with_temperature(model, x, T=2.0):
    x = model.tcn(x)
    x = model.gap(x).squeeze(-1)
    x = F.relu(model.fc1(x))
    logit = model.fc2(x).squeeze(-1) / T
    return torch.sigmoid(logit)


def load_test_dataset():
    """Load test npy (3432 windows × 16 × 27), apply KEEP_23 slicing."""
    X = np.load(f'{BASE}/X_test_binary_v2_scada_window16.npy').astype(np.float32)
    y = np.load(f'{BASE}/y_test_binary_v2_scada_window16.npy').astype(np.int64)
    X = X[:, :, KEEP_23]  # → (3432, 16, 23)
    # transpose to (N, C, L) for Conv1d: (N, 23, 16)
    X = X.transpose(0, 2, 1).astype(np.float32)
    return X, y


def extract_h743_probs(csv_path):
    """从测试日志 CSV 提取每个 ALARM 的 frame= 和 prob=."""
    probs = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'ALARM' not in line:
                continue
            m = re.search(r'frame=(\d+)', line)
            p = re.search(r'prob=([\d.]+)', line)
            if m and p:
                probs[int(m.group(1))] = float(p.group(1))
    return probs


def main():
    print("=" * 70)
    print("[1] 加载 H743 Hybrid INT8 权重")
    print("=" * 70)
    h_w_int8, h_scales, h_w_fp32, h_b_fp32 = parse_h743_header(
        f'{BASE}/h743/Core/Inc/model_v4_se_23dim_ch32_hybrid_s{SEED}.h')
    print(f"  INT8 layers: {len(h_w_int8)}, FP32 fc1/fc2: {len(h_w_fp32)}")
    model = build_hybrid_model(h_w_int8, h_scales, h_w_fp32, h_b_fp32)

    print()
    print("=" * 70)
    print("[2] 加载测试集 (X_test_binary_v2_scada_window16.npy, 3432 windows)")
    print("=" * 70)
    Xw, yw = load_test_dataset()
    print(f"  Xw shape: {Xw.shape}, yw shape: {yw.shape}")
    print(f"  Attack windows: {yw.sum()}/{len(yw)} ({yw.mean()*100:.1f}%)")

    print()
    print("=" * 70)
    print("[3] PC Hybrid INT8 (dequant) + T=2.0 推理")
    print("=" * 70)
    with torch.no_grad():
        x_t = torch.from_numpy(Xw)
        probs_pc = np.zeros(len(yw), dtype=np.float32)
        for start in range(0, len(yw), 512):
            end = min(start + 512, len(yw))
            batch = x_t[start:end]
            p = hybrid_forward_with_temperature(model, batch, T=2.0)
            probs_pc[start:end] = p.numpy()
    print(f"  PC prob range: [{probs_pc.min():.4f}, {probs_pc.max():.4f}], mean={probs_pc.mean():.4f}")
    print(f"  PC T=2.0 thr=0.5: alarms={int((probs_pc >= 0.5).sum())}")
    tp_pc = int(((probs_pc >= 0.5) & (yw == 1)).sum())
    fp_pc = int(((probs_pc >= 0.5) & (yw == 0)).sum())
    fn_pc = int(((probs_pc < 0.5) & (yw == 1)).sum())
    tn_pc = int(((probs_pc < 0.5) & (yw == 0)).sum())
    print(f"  PC T=2.0 thr=0.5: TP={tp_pc} FP={fp_pc} FN={fn_pc} TN={tn_pc}")

    print()
    print("=" * 70)
    print("[4] 提取 H743 实测 prob (from CSV)")
    print("=" * 70)
    csv_path = f'{BASE}/logs/test_20261013-233901_auto-40ms.csv'
    h743_probs = extract_h743_probs(csv_path)
    print(f"  H743 ALARM frames extracted: {len(h743_probs)}")
    if h743_probs:
        print(f"  H743 prob range: [{min(h743_probs.values()):.4f}, {max(h743_probs.values()):.4f}]")
        print(f"  H743 mean prob: {np.mean(list(h743_probs.values())):.4f}")

    # Map H743 frame to window index (full 274k → 0-indexed window)
    # test npy covers 3432 windows, but they are random split, NOT last 3432
    # We can only compare PROB distributions on alarmed windows
    # Frame K → window (K-1)//16 in full data; for test npy, no direct mapping
    # BUT we can compare PC prob distribution vs H743 prob distribution

    print()
    print("=" * 70)
    print("[5] 概率分布对比 (H743 实测 alarm vs PC 全量 3432 windows)")
    print("=" * 70)
    def histo(arr, name):
        bins = [0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1.001]
        h, _ = np.histogram(arr, bins=bins)
        total = len(arr)
        print(f"  {name} (n={total}):")
        for i in range(len(h)):
            pct = h[i] / total * 100
            print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}): {h[i]:5d}  ({pct:5.1f}%)")
    histo(np.array(list(h743_probs.values())), "H743 实测 (all alarmed frames)")
    print()
    histo(probs_pc, "PC Hybrid + T=2.0 (all 3432 test windows)")

    print()
    print("=" * 70)
    print("[6] PC T=2.0 thr=0.5 KPI (test set 3432 windows)")
    print("=" * 70)
    pc_pred = (probs_pc >= 0.5).astype(int)
    f1_pc = 2*tp_pc / (2*tp_pc + fp_pc + fn_pc) if (2*tp_pc+fp_pc+fn_pc) > 0 else 0
    recall_pc = tp_pc / (tp_pc + fn_pc) if (tp_pc+fn_pc) > 0 else 0
    fpr_pc = fp_pc / (fp_pc + tn_pc) if (fp_pc+tn_pc) > 0 else 0
    prec_pc = tp_pc / (tp_pc + fp_pc) if (tp_pc+fp_pc) > 0 else 0
    print(f"  PC KPI (test, N=3432): TP={tp_pc} FP={fp_pc} TN={tn_pc} FN={fn_pc}")
    print(f"  F1={f1_pc*100:.2f}% Recall={recall_pc*100:.2f}% FPR={fpr_pc*100:.2f}% Precision={prec_pc*100:.2f}%")

    print()
    print("=" * 70)
    print("[7] PC vs H743 全量 (test set, for cross-validation)")
    print("=" * 70)
    print("H743 实测 274k 全量 per-window: TP=7508 FP=1415 FPR=16.96%")
    print(f"PC 预测 test set (3432 windows): FPR={fpr_pc*100:.2f}%")
    print("注意: test set 仅 3432 windows 不可直接对比 17164 windows 全量 FPR")
    print("     但 PC test FPR vs H743 全量 FPR 反映 PTQ 漂移量级")
    if fpr_pc < 0.05:
        print("  ✓ PC test FPR < 5% → 模型本身是好, gap 全是 PTQ 漂移")
    elif fpr_pc < 0.10:
        print("  ✓ PC test FPR < 10% → 漂移是主导因素")
    else:
        print("  ⚠ PC test FPR > 10% → 可能模型本身也有问题")


if __name__ == '__main__':
    main()
