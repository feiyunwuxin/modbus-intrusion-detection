#!/usr/bin/env python3
"""
diag_pc_vs_h743_int8.py

PC Hybrid INT8 (dequant) forward vs H743 actual — 完整 274k 数据.

数据路径 (跟 diag_pc_vs_h743_full274k.py 一样, 用 19-dim preprocessed + aggregates):
  X_train/val/test = X_*_binary_v2_scada.npy  (19-dim, 已 scaled)
  + 8 per-window aggregates → 27-dim
  + KEEP_23 + clip[-10,10] + transpose(0,2,1)
  → (17164, 23, 16) 输入

模型:
  - Hybrid INT8 dequant weights (从 H743 model_v4_se_23dim_ch32_hybrid_s42.h 解析)
  - SE block / FC head FP32 (跟 H743 一样)
  - logit / T=2.0 (跟 H743 ch32_inference.c:301 一样)

目的:
  如果 PC_Hybrid_INT8_T2 ≈ H743 实测 FPR=16.96% → H743 实现正确, drift 来自 PTQ 固有
  如果 PC_Hybrid_INT8_T2 ≈ PC_FP32 FPR=2.27% → INT8 dequant 没漂移, H743 实现有 bug
"""
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\work\Claude\Issue"
SEED = 42
WINDOW = 16
N_FEATURES = 23
CHANNELS = 32
DILATIONS = [1, 2, 4]
T = 2.0

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]


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
            parts = rest.split('_'); blk, fc = parts[1], parts[3]
            key = f'tcn.{blk}.se.{fc}.weight'
        elif 'residual' in rest:
            blk = rest.split('_')[1]; key = f'tcn.{blk}.residual.weight'
        else:
            parts = rest.split('_'); blk, layer = parts[1], parts[2]
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
            parts = rest.split('_'); blk, fc = parts[1], parts[3]
            key = f'tcn.{blk}.se.{fc}.weight'
        elif 'residual' in rest:
            blk = rest.split('_')[1]; key = f'tcn.{blk}.residual.weight'
        else:
            parts = rest.split('_'); blk, layer = parts[1], parts[2]
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
            parts = body.split('_'); blk = parts[1]
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
        q = h_w_int8[key]; s = h_scales[key]; shape = INT8_SHAPES[key]
        qr = q.reshape(shape); sr = s.reshape((shape[0],) + (1,) * (len(shape) - 1))
        return (qr.astype(np.float32) * sr).astype(np.float32)
    model = TCNSE(N_FEATURES); model.eval(); sd = model.state_dict()
    for i in range(3):
        for layer in ['conv1', 'conv2']:
            w_key = f'tcn.{i}.{layer}.weight'; b_key = f'tcn.{i}.{layer}.bias'
            sd[w_key] = torch.from_numpy(dequant(w_key))
            sd[b_key] = torch.from_numpy(h_b_fp32[b_key])
            bn_idx = layer[-1]
            for bn_key in [f'tcn.{i}.bn{bn_idx}.weight', f'tcn.{i}.bn{bn_idx}.bias',
                           f'tcn.{i}.bn{bn_idx}.running_mean', f'tcn.{i}.bn{bn_idx}.running_var',
                           f'tcn.{i}.bn{bn_idx}.num_batches_tracked']:
                sd.pop(bn_key, None)
        for fc in ['fc1', 'fc2']:
            w_key = f'tcn.{i}.se.{fc}.weight'; b_key = f'tcn.{i}.se.{fc}.bias'
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


def load_full_dataset():
    """19-dim preprocessed + 8 aggregates → 23-dim windowed, full 274k."""
    X_train = np.load("X_train_binary_v2_scada.npy")
    X_val   = np.load("X_val_binary_v2_scada.npy")
    X_test  = np.load("X_test_binary_v2_scada.npy")
    X_full = np.concatenate([X_train, X_val, X_test], axis=0).astype(np.float32)
    y_train = np.load("y_train_binary_v2_scada.npy")
    y_val   = np.load("y_val_binary_v2_scada.npy")
    y_test  = np.load("y_test_binary_v2_scada.npy")
    y_full  = np.concatenate([y_train, y_val, y_test], axis=0).astype(np.int64)
    n = (len(X_full) // WINDOW) * WINDOW
    Xw = X_full[:n].reshape(n // WINDOW, WINDOW, -1)  # (17164, 16, 19)
    yw = y_full[:n].reshape(n // WINDOW, WINDOW).max(axis=1).astype(np.int64)

    IDX_PRESS, IDX_CRC, IDX_ISRESP, IDX_LENGTH, IDX_ISUNUSUAL = 13, 14, 18, 2, 17
    press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
    crc_mean_w   = Xw[:, :, IDX_CRC].mean(axis=1)
    crc_max_w    = Xw[:, :, IDX_CRC].max(axis=1)
    cmd_count_w  = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)
    resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
    balance_w    = (cmd_count_w - resp_count_w) / WINDOW
    length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(n // WINDOW)], dtype=np.float32)
    unusual_count_w  = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)
    aggs = np.column_stack([press_mean_w, crc_mean_w, crc_max_w, cmd_count_w, resp_count_w, balance_w, length_nunique_w, unusual_count_w]).astype(np.float32)
    aggs_tiled = np.repeat(aggs[:, np.newaxis, :], WINDOW, axis=1)
    Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)  # (17164, 16, 27)
    Xw_23 = np.clip(Xw_full[:, :, KEEP_23], -10.0, 10.0).transpose(0, 2, 1)  # (17164, 23, 16)
    return Xw_23, yw


def main():
    print("=" * 70)
    print("[1] 加载 Hybrid INT8 model + FP32 PC baseline 概率")
    print("=" * 70)
    h_w_int8, h_scales, h_w_fp32, h_b_fp32 = parse_h743_header(
        f'{BASE}/h743/Core/Inc/model_v4_se_23dim_ch32_hybrid_s{SEED}.h')
    model_hybrid = build_hybrid_model(h_w_int8, h_scales, h_w_fp32, h_b_fp32)

    # 也加载 FP32 baseline 概率 (从之前的 npz)
    pc_baseline = np.load("pc_full274k_probs.npz")
    yw_baseline = pc_baseline['y']
    print(f"  PC FP32 baseline (T=1): shape={pc_baseline['probs'].shape}")

    print()
    print("=" * 70)
    print("[2] 加载数据集 (X_*_binary_v2_scada 19-dim preprocessed + aggregates)")
    print("=" * 70)
    Xw, yw = load_full_dataset()
    assert (yw == yw_baseline).all(), "y label mismatch!"
    print(f"  Xw shape: {Xw.shape}, yw shape: {yw.shape}")
    print(f"  Attack windows: {yw.sum()}/{len(yw)} ({yw.mean()*100:.1f}%)")

    print()
    print("=" * 70)
    print(f"[3] Hybrid INT8 (dequant) + T={T} forward on FULL 274k")
    print("=" * 70)
    with torch.no_grad():
        x_t = torch.from_numpy(Xw)
        BATCH = 512
        probs_hybrid = np.zeros(len(yw), dtype=np.float32)
        for start in range(0, len(yw), BATCH):
            end = min(start + BATCH, len(yw))
            batch = x_t[start:end]
            p = hybrid_forward_with_temperature(model_hybrid, batch, T=T)
            probs_hybrid[start:end] = p.numpy()
    print(f"  Hybrid INT8 + T={T} probs range: [{probs_hybrid.min():.4f}, {probs_hybrid.max():.4f}]")
    print(f"  mean: {probs_hybrid.mean():.4f}, median: {np.median(probs_hybrid):.4f}")

    # KPI for hybrid
    tp_h = int(((probs_hybrid >= 0.5) & (yw == 1)).sum())
    fp_h = int(((probs_hybrid >= 0.5) & (yw == 0)).sum())
    fn_h = int(((probs_hybrid < 0.5) & (yw == 1)).sum())
    tn_h = int(((probs_hybrid < 0.5) & (yw == 0)).sum())
    f1_h = 2*tp_h / (2*tp_h+fp_h+fn_h) if (2*tp_h+fp_h+fn_h) else 0
    fpr_h = fp_h / (fp_h+tn_h) if (fp_h+tn_h) else 0
    prec_h = tp_h / (tp_h+fp_h) if (tp_h+fp_h) else 0
    rec_h = tp_h / (tp_h+fn_h) if (tp_h+fn_h) else 0
    print(f"  Hybrid KPI thr=0.5: TP={tp_h} FP={fp_h} TN={tn_h} FN={fn_h}")
    print(f"    F1={f1_h*100:.2f}% Recall={rec_h*100:.2f}% FPR={fpr_h*100:.2f}% Prec={prec_h*100:.2f}%")

    print()
    print("=" * 70)
    print("[4] 对比汇总")
    print("=" * 70)
    print(f"  | Metric | PC_FP32 T=1.0 thr=0.5 | Hybrid INT8 T={T} thr=0.5 | H743 actual T={T} thr=0.5 |")
    print(f"  | F1     | 88.14%                | {f1_h*100:.2f}%                      | 84.63%                    |")
    print(f"  | Recall | 80.48%                | {rec_h*100:.2f}%                      | 85.12%                    |")
    print(f"  | FPR    | 2.27%                 | {fpr_h*100:.2f}%                     | 16.96%                    |")
    print(f"  | Prec   | 97.41%                | {prec_h*100:.2f}%                     | 84.14%                    |")

    fp_pc_baseline = int(((pc_baseline['probs'] >= 0.5) & (yw == 0)).sum())
    tp_pc_baseline = int(((pc_baseline['probs'] >= 0.5) & (yw == 1)).sum())
    print(f"  | PC FP32 baseline thr=0.5: TP={tp_pc_baseline} FP={fp_pc_baseline}")

    # 关键判断
    print()
    print("=" * 70)
    print("[5] 诊断结论")
    print("=" * 70)
    if abs(fpr_h*100 - 2.27) < 1:
        print(f"  ✓ Hybrid INT8 sim FPR={fpr_h*100:.2f}% ≈ PC FP32 FPR=2.27%")
        print(f"  → H743 实际 FPR=16.96% 是 H743 代码侧 bug, 不是 INT8 量化本身")
        print(f"  → 修复方向: 检查 H743 inference 代码 (BN fold / SE / cmsis-nn)")
    elif abs(fpr_h*100 - 16.96) < 2:
        print(f"  ✓ Hybrid INT8 sim FPR={fpr_h*100:.2f}% ≈ H743 实际 FPR=16.96%")
        print(f"  → INT8 dequant 漂移就是 gap 来源, 量化本身要把所有 probs 推向 extreme")
        print(f"  → 修复方向: 重新做 PTQ 校准或 QAT 重训, 用 per-channel + EMA 校准")
    else:
        print(f"  ⚠ Hybrid INT8 sim FPR={fpr_h*100:.2f}% 在 PC 与 H743 之间")
        print(f"  → 部分是 INT8 drift, 部分是 H743 代码 bug, 需进一步定位")


if __name__ == '__main__':
    main()
