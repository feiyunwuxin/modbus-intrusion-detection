#!/usr/bin/env python3
"""比较 H743 实测 prob vs PC 端 Hybrid INT8 (dequant) + T=2.0 模拟预测.

目的: 定位 H743 实际 F1=84.63% vs PC 预测 F1=88.14% 的差距根因.

如果 PC 模拟 ≈ H743 实测 → H743 实现正确, 差距是 INT8 量化的固有 trade-off
如果 PC 模拟 ≠ H743 实测 → H743 实现有 bug (SE 顺序, expf 精度, 等)
"""

import io
import sys
import re
import csv
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


# ─── 解析 H743 .h 头文件 ───

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
    """Build PyTorch model with H743's Hybrid INT8 weights (dequantized)."""
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
    """Hybrid forward + T scaling (matches H743 ch32_inference.c:301)."""
    # Manually replicate forward without fc_drop (H743 doesn't apply it at inference)
    x = model.tcn(x)
    x = model.gap(x).squeeze(-1)
    x = F.relu(model.fc1(x))
    logit = model.fc2(x).squeeze(-1) / T
    return torch.sigmoid(logit)


# ─── 加载数据 ───

def load_dataset():
    """Load IanArffDataset.csv and windowize."""
    df = pd.read_csv(f'{BASE}/IanArffDataset.csv')
    n_feat = 23
    cols = [c for c in df.columns if c not in ('label_binary', 'label_attack', 'label_specific', 't_event_ms')][:n_feat]
    X = df[cols].values.astype(np.float32)
    y = df['label_binary'].values.astype(np.int64)
    n = (len(X) // WINDOW) * WINDOW
    Xw = X[:n].reshape(n // WINDOW, WINDOW, n_feat).transpose(0, 2, 1).astype(np.float32)
    yw = y[:n].reshape(n // WINDOW, WINDOW).max(axis=1)
    return Xw, yw


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


# ─── 主函数 ───

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
    print("[2] 加载数据集 (IanArffDataset.csv 274k)")
    print("=" * 70)
    Xw, yw = load_dataset()
    print(f"  Xw shape: {Xw.shape}, yw shape: {yw.shape}")
    print(f"  Attack windows: {yw.sum()}/{len(yw)} ({yw.mean()*100:.1f}%)")

    print()
    print("=" * 70)
    print("[3] PC Hybrid INT8 (dequant) + T=2.0 全量推理")
    print("=" * 70)
    with torch.no_grad():
        x_t = torch.from_numpy(Xw)
        # 分批推理避免 OOM
        probs_pc = np.zeros(len(yw), dtype=np.float32)
        for start in range(0, len(yw), 512):
            end = min(start + 512, len(yw))
            batch = x_t[start:end]
            p = hybrid_forward_with_temperature(model, batch, T=2.0)
            probs_pc[start:end] = p.numpy()
    print(f"  PC prob range: [{probs_pc.min():.4f}, {probs_pc.max():.4f}], mean={probs_pc.mean():.4f}")
    print(f"  PC T=2.0 thr=0.5: alarms={int((probs_pc >= 0.5).sum())}")
    print(f"  PC T=2.0 thr=0.5: TP={int(((probs_pc >= 0.5) & (yw == 1)).sum())} FP={int(((probs_pc >= 0.5) & (yw == 0)).sum())}")

    # Save for later use
    np.savez('pc_hybrid_int8_t2_full274k.npz', probs=probs_pc, y=yw)

    print()
    print("=" * 70)
    print("[4] 提取 H743 实测 prob (from CSV)")
    print("=" * 70)
    h743_probs = extract_h743_probs(f'{BASE}/logs/test_20261013-233901_auto-40ms.csv')
    print(f"  H743 ALARM frames extracted: {len(h743_probs)}")
    print(f"  H743 prob range: [{min(h743_probs.values()):.4f}, {max(h743_probs.values()):.4f}]")
    print(f"  H743 mean prob: {np.mean(list(h743_probs.values())):.4f}")

    # Frame = 16 * (window_idx) for non-overlapping
    # H743 frame 字段是 1-indexed 帧号, 最后一个帧号 (frame % 16 == 0) -> window K = frame/16
    # 我们需要映射: h743 frame K -> 0-indexed window K-1 (因为 window 1 = 帧 1-16)
    h743_window_probs = {}
    for frame, prob in h743_probs.items():
        win_idx = (frame - 1) // 16  # 0-indexed window
        if 0 <= win_idx < len(yw):
            h743_window_probs[win_idx] = prob
    print(f"  H743 window-level probs: {len(h743_window_probs)}")

    # H743 only emits prob for alarmed windows, so we can only compare those
    # For these alarmed windows, compare H743 prob vs PC hybrid prob
    print()
    print("=" * 70)
    print("[5] H743 实测 vs PC Hybrid + T=2.0 比较 (仅 alarm 窗口)")
    print("=" * 70)
    common = sorted(h743_window_probs.keys())
    h_arr = np.array([h743_window_probs[w] for w in common])
    p_arr = probs_pc[common]
    diff = np.abs(h_arr - p_arr)
    print(f"  N common: {len(common)}")
    print(f"  Max |diff|:  {diff.max():.4f}")
    print(f"  Mean |diff|: {diff.mean():.4f}")
    print(f"  Median |diff|: {np.median(diff):.4f}")
    print(f"  P95 |diff|:  {np.percentile(diff, 95):.4f}")
    # 同号性 (sign of (prob - 0.5) 应该一致)
    h_alarm = h_arr >= 0.5
    p_alarm = p_arr >= 0.5
    agree = (h_alarm == p_alarm).mean()
    print(f"  thr=0.5 alarm agreement: {agree*100:.2f}%  ({agree*len(common):.0f}/{len(common)})")

    # 如果 H743 推 0.7 但 PC 推 0.95, 这就是 T scaling 之后还有的差距
    # 看一下 H743 是 alarm 但 PC 不是的情况
    h_alarm_p_not = sum(h_alarm & ~p_alarm)
    h_not_p_alarm = sum(~h_alarm & p_alarm)
    print(f"  H743 alarm but PC normal: {h_alarm_p_not}")
    print(f"  H743 normal but PC alarm: {h_not_p_alarm}")

    # Distribution comparison for ALARMED windows only
    print()
    print("=" * 70)
    print("[6] 概率分布对比 (H743 实测 alarmed 窗口 vs PC 全量 17164 窗口)")
    print("=" * 70)
    def histo(arr, name):
        bins = [0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1.001]
        h, _ = np.histogram(arr, bins=bins)
        total = len(arr)
        print(f"  {name} (n={total}):")
        for i in range(len(h)):
            pct = h[i] / total * 100
            print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}): {h[i]:5d}  ({pct:5.1f}%)")
    histo(h_arr, "H743 实测 (alarmed windows)")
    print()
    histo(probs_pc, "PC Hybrid + T=2.0 (all 17164 windows)")

    # Per-Window 4-metrics 对比
    print()
    print("=" * 70)
    print("[7] Per-Window 4 指标对比 (thr=0.5)")
    print("=" * 70)
    pc_pred = (probs_pc >= 0.5).astype(int)
    h_pred = np.zeros(len(yw), dtype=int)
    for w_idx, p in h743_window_probs.items():
        h_pred[w_idx] = int(p >= 0.5)

    # TP/FP/TN/FN
    def kpi(yt, yp):
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) > 0 else 0
        recall = tp / (tp + fn) if (tp+fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, f1=f1, recall=recall, fpr=fpr, prec=prec)

    kpi_pc = kpi(yw, pc_pred)
    print(f"  PC Hybrid + T=2.0:  {kpi_pc}")
    # H743 真实评测: TP=7508, FP=1415, TN=6928, FN=1313
    print(f"  H743 实测:          TP=7508 FP=1415 TN=6928 FN=1313  (from summary)")


if __name__ == '__main__':
    main()
