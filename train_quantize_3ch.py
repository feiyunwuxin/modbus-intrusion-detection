#!/usr/bin/env python3
"""
TCN+SE 19-dim SCADA — Hybrid 量化 3 个 MCU 重点候选 (ch=8/16/32, LR=2e-3, b=3)

每个模型: 训练 → 评估 FP32 → 折叠 BN → INT8 per-channel 量化 → 评估 Hybrid
输出: FP32 vs Hybrid 对比表 + .bin/.h/.json 三件套 × 3

覆盖 MCU 部署三层:
  - ch=8  (2.3K params)  → Arduino Uno 极致小
  - ch=16 (6.4K params)  → STM32F4 / ESP32 平衡
  - ch=32 (20.4K params) → RP2040 / 边缘服务器 生产通用
"""

import os, json, time, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score
import tracemalloc

BASE = r"C:\work\Claude\Issue"

# 配置 (与 tcn_v4_se_ch_sweep_lr2e3.py 完全一致)
SEED         = 42
WINDOW       = 16
BATCH_TRAIN  = 128
BATCH_EVAL   = 512
EPOCHS       = 40
LR           = 2e-3
WD           = 1e-5
PATIENCE     = 10
GRAD_CLIP    = 0.5
DROPOUT      = 0.3
CLIP_VAL     = 10.0
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4]
SE_REDUCTION = 8
N_BLOCKS     = 3

CHANNELS_LIST = [8, 16, 32]   # 三个 MCU 重点候选


# ────────── 架构 (与 channel sweep 完全一致) ──────────
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
    def __init__(self, in_ch, channels, n_blocks=N_BLOCKS, kernel_size=KERNEL_SIZE,
                 dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
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


# ────────── 数据加载 (19-dim SCADA v2) ──────────
def load_data():
    Xtr = np.load(os.path.join(BASE, "X_train_binary_v2_scada.npy")).astype(np.float32)
    Xva = np.load(os.path.join(BASE, "X_val_binary_v2_scada.npy")).astype(np.float32)
    Xte = np.load(os.path.join(BASE, "X_test_binary_v2_scada.npy")).astype(np.float32)
    ytr = np.load(os.path.join(BASE, "y_train_binary_v2_scada.npy")).astype(np.int64)
    yva = np.load(os.path.join(BASE, "y_val_binary_v2_scada.npy")).astype(np.int64)
    yte = np.load(os.path.join(BASE, "y_test_binary_v2_scada.npy")).astype(np.int64)

    def make_windows(X, y, w=WINDOW):
        n = (len(X) // w) * w
        Xw = X[:n].reshape(n // w, w, -1)
        yw = (y[:n].reshape(n // w, w).max(axis=1)).astype(np.int64)
        return Xw, yw

    Xtrw, ytrw = make_windows(Xtr, ytr)
    Xvaw, yvaw = make_windows(Xva, yva)
    Xtew, ytew = make_windows(Xte, yte)

    N_FEATURES = Xtrw.shape[-1]
    Xtrw = np.clip(Xtrw, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    Xvaw = np.clip(Xvaw, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    Xtew = np.clip(Xtew, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)

    return (N_FEATURES,
            DataLoader(TensorDataset(torch.from_numpy(Xtrw), torch.from_numpy(ytrw)),
                       batch_size=BATCH_TRAIN, shuffle=True, num_workers=0),
            DataLoader(TensorDataset(torch.from_numpy(Xvaw), torch.from_numpy(yvaw)),
                       batch_size=BATCH_EVAL, shuffle=False, num_workers=0),
            DataLoader(TensorDataset(torch.from_numpy(Xtew), torch.from_numpy(ytew)),
                       batch_size=BATCH_EVAL, shuffle=False, num_workers=0),
            ytew)


# ────────── 量化核心 (移植自 hybrid_quantize_v19.py) ──────────
def fold_bn(conv, bn):
    w = conv.weight.data.clone()
    b = conv.bias.data.clone() if conv.bias is not None else torch.zeros(conv.out_channels)
    γ, β = bn.weight.data.clone(), bn.bias.data.clone()
    μ, σ = bn.running_mean.data.clone(), bn.running_var.data.clone().sqrt() + bn.eps
    scale = (γ / σ).reshape(-1, 1, 1)
    return w * scale, (b - μ) * (γ / σ) + β


def quantize_int8_per_channel(w):
    out_ch = w.shape[0]
    w_flat = w.reshape(out_ch, -1).contiguous()
    amax = w_flat.abs().max(dim=1).values
    scale = torch.clamp(amax / 127.0, min=1e-8)
    q = torch.round(w_flat / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q.reshape(w.shape), scale


def dequantize_int8_per_channel(q, scale):
    out_ch = q.shape[0]
    return q.float() * scale.reshape(out_ch, *([1] * (q.ndim - 1)))


def collect_quant_state(model):
    """返回 (quant_tensors, bias_tensors) for 整个模型, BN 已折叠进 Conv"""
    quant_tensors = []
    bias_tensors = []
    snr_per_layer = {}

    for bi, blk in enumerate(model.tcn):
        for cname, bname in [("conv1", "bn1"), ("conv2", "bn2")]:
            w_fold, b_fold = fold_bn(getattr(blk, cname), getattr(blk, bname))
            q, s = quantize_int8_per_channel(w_fold)
            dq = dequantize_int8_per_channel(q, s)
            snr = 10 * torch.log10((w_fold**2).mean() / ((w_fold - dq)**2).mean() + 1e-12).item()
            quant_tensors.append((f"tcn.{bi}.{cname}.weight", q, s, w_fold.shape))
            bias_tensors.append((f"tcn.{bi}.{cname}.bias", b_fold))
            snr_per_layer[f"tcn.{bi}.{cname}"] = snr
        if isinstance(blk.residual, nn.Conv1d):
            w = blk.residual.weight.data.clone()
            q, s = quantize_int8_per_channel(w)
            dq = dequantize_int8_per_channel(q, s)
            snr = 10 * torch.log10((w**2).mean() / ((w - dq)**2).mean() + 1e-12).item()
            quant_tensors.append((f"tcn.{bi}.residual.weight", q, s, w.shape))
            bias_tensors.append((f"tcn.{bi}.residual.bias", blk.residual.bias.data.clone()))
            snr_per_layer[f"tcn.{bi}.residual"] = snr
        for sename in ["fc1", "fc2"]:
            w = getattr(getattr(blk.se, sename), "weight").data.clone()
            q, s = quantize_int8_per_channel(w)
            dq = dequantize_int8_per_channel(q, s)
            snr = 10 * torch.log10((w**2).mean() / ((w - dq)**2).mean() + 1e-12).item()
            quant_tensors.append((f"tcn.{bi}.se.{sename}.weight", q, s, w.shape))
            bias_tensors.append((f"tcn.{bi}.se.{sename}.bias",
                                 getattr(getattr(blk.se, sename), "bias").data.clone()))
            snr_per_layer[f"tcn.{bi}.se.{sename}"] = snr

    for fcname in ["fc1", "fc2"]:
        fc = getattr(model, fcname)
        w = fc.weight.data.clone()
        q, s = quantize_int8_per_channel(w)
        dq = dequantize_int8_per_channel(q, s)
        snr = 10 * torch.log10((w**2).mean() / ((w - dq)**2).mean() + 1e-12).item()
        quant_tensors.append((f"{fcname}.weight", q, s, w.shape))
        bias_tensors.append((f"{fcname}.bias", fc.bias.data.clone()))
        snr_per_layer[fcname] = snr

    return quant_tensors, bias_tensors, snr_per_layer


def save_hybrid_files(channels, quant_tensors, bias_tensors, snr):
    """保存 .bin / .h / .json 三件套"""
    base_name = f"model_ch{channels}_hybrid"
    bin_path = os.path.join(BASE, f"{base_name}.bin")
    h_path   = os.path.join(BASE, f"{base_name}.h")
    meta_path = os.path.join(BASE, f"{base_name}_meta.json")

    with open(bin_path, "wb") as f:
        f.write(struct.pack("<4sHI", b"TCNH", 1, len(quant_tensors)))
        for name, q, s, shape in quant_tensors:
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb))); f.write(nb)
            f.write(struct.pack("<B", len(shape)))
            for d in shape: f.write(struct.pack("<I", d))
            f.write(q.contiguous().numpy().tobytes())
            f.write(struct.pack("<I", s.numel()))
            f.write(s.float().numpy().astype(np.float32).tobytes())
        for name, b in bias_tensors:
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb))); f.write(nb)
            f.write(b.float().numpy().astype(np.float32).tobytes())
    bin_size = os.path.getsize(bin_path)

    with open(h_path, "w") as f:
        f.write(f"""/* Auto-generated by train_quantize_3ch.py
 * TCN+SE ch={channels} (19-dim SCADA) — Hybrid INT8-weights / FP32-activations
 * Total Flash: {bin_size:,} bytes ({bin_size/1024:.2f} KB)
 * Strategy: 权重 INT8 + scale FP32, 激活 FP32, BN 已折叠进 Conv
 */
#ifndef TCN_CH{channels}_HYBRID_WEIGHTS_H
#define TCN_CH{channels}_HYBRID_WEIGHTS_H
#include <stdint.h>
#define TCN_CH{channels}_N_QUANT_TENSORS {len(quant_tensors)}
#define TCN_CH{channels}_TOTAL_PARAMS   {sum(q.numel() for _, q, _, _ in quant_tensors) + sum(b.numel() for _, b in bias_tensors)}
#ifdef __cplusplus
extern "C" {{
#endif
""")
        for name, q, s, shape in quant_tensors:
            cname = f"ch{channels}_w_" + name.replace(".", "_")
            sname = f"ch{channels}_s_" + name.replace(".", "_")
            f.write(f"/* {name}  shape={tuple(shape)} */\n")
            f.write(f"static const int8_t {cname}[{q.numel()}] = {{\n    ")
            arr = q.numpy().flatten().tolist()
            for i, v in enumerate(arr):
                f.write(f"{v:4d}")
                if i < len(arr) - 1:
                    f.write(",")
                    if (i + 1) % 16 == 0: f.write("\n    ")
                    else: f.write(" ")
            f.write("\n};\n")
            f.write(f"static const float {sname}[{s.numel()}] = {{\n    ")
            arr = s.numpy().tolist()
            for i, v in enumerate(arr):
                f.write(f"{v:.6e}f")
                if i < len(arr) - 1:
                    f.write(",")
                    if (i + 1) % 8 == 0: f.write("\n    ")
                    else: f.write(" ")
            f.write("\n};\n\n")
        for name, b in bias_tensors:
            cname = f"ch{channels}_b_" + name.replace(".", "_")
            f.write(f"/* {name}  numel={b.numel()} */\n")
            f.write(f"static const float {cname}[{b.numel()}] = {{\n    ")
            arr = b.numpy().tolist()
            for i, v in enumerate(arr):
                f.write(f"{v:.6e}f")
                if i < len(arr) - 1:
                    f.write(",")
                    if (i + 1) % 8 == 0: f.write("\n    ")
                    else: f.write(" ")
            f.write("\n};\n\n")
        f.write("#ifdef __cplusplus\n}\n#endif\n#endif\n")
    h_size = os.path.getsize(h_path)

    meta = {
        "model": f"TCN+SE ch={channels} (19-dim SCADA, b=3, d=[1,2,4], SE_r=8, LR=2e-3, seed=42)",
        "strategy": "weights INT8 per-channel symmetric, activations FP32, BN folded into Conv",
        "channels": channels,
        "n_quant_tensors": len(quant_tensors),
        "n_bias_tensors": len(bias_tensors),
        "bin_size_bytes": bin_size,
        "h_size_bytes": h_size,
        "snr_db": {k: float(v) for k, v in snr.items()},
        "tensors": [
            {"name": n, "shape": list(sh), "dtype": "int8", "scale_count": s.numel()}
            for n, _, s, sh in quant_tensors
        ] + [
            {"name": n, "numel": b.numel(), "dtype": "float32"} for n, b in bias_tensors
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return bin_size, h_size, meta


# ────────── Hybrid 重建 + 推理类 ──────────
def build_hybrid_model(channels, n_features, quant_tensors, bias_tensors):
    """返回重建后的 Hybrid 模型 (BN→Identity, 权重已 dequant 回来)"""
    model = TCNClassifierSE(in_ch=n_features, channels=channels)
    # 把 BN 替换成 Identity
    for bi in range(N_BLOCKS):
        blk = model.tcn[bi]
        blk.bn1 = nn.Identity()
        blk.bn2 = nn.Identity()
    # 写回 dequant 后的权重
    with torch.no_grad():
        for name, q, s, shape in quant_tensors:
            w_dq = dequantize_int8_per_channel(q, s)
            parts = name.split(".")
            target = model
            for p in parts[:-1]:
                target = target[int(p)] if p.isdigit() else getattr(target, p)
            getattr(target, parts[-1]).data.copy_(w_dq)
        for name, b in bias_tensors:
            parts = name.split(".")
            target = model
            for p in parts[:-1]:
                target = target[int(p)] if p.isdigit() else getattr(target, p)
            getattr(target, parts[-1]).data.copy_(b)
    model.eval()
    return model


# ────────── 训练 + 量化主流程 ──────────
def train_one(channels, n_features, train_loader, val_loader, test_loader, y_test):
    torch.manual_seed(SEED); np.random.seed(SEED)
    g = torch.Generator(); g.manual_seed(SEED)
    # 用 generator 复现 sweep 的 train loader 顺序
    train_loader = DataLoader(train_loader.dataset, batch_size=BATCH_TRAIN, shuffle=True,
                              num_workers=0, generator=g)

    model = TCNClassifierSE(in_ch=n_features, channels=channels)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((train_loader.dataset.tensors[1] == 1).sum())
    n_neg = int((train_loader.dataset.tensors[1] == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    @torch.no_grad()
    def predict_probs(loader, m):
        m.eval()
        ps = []
        for xb, _ in loader:
            ps.append(torch.sigmoid(m(xb)).numpy())
        return np.concatenate(ps)

    best_f1m, best_state, best_epoch = -1, None, -1
    t_c = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        val_prob = predict_probs(val_loader, model)
        val_f1m = f1_score(val_loader.dataset.tensors[1], (val_prob >= 0.5).astype(int), average="macro")
        scheduler.step(val_f1m)
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= PATIENCE:
            break

    train_time = time.time() - t_c
    model.load_state_dict(best_state)
    return model, n_params, best_epoch, train_time


@torch.no_grad()
def predict(loader, m):
    m.eval()
    ps, ls = [], []
    for xb, yb in loader:
        ps.append(torch.sigmoid(m(xb)).numpy())
        ls.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ls)


def latency_bench(m, n_features, n_runs=2000, warmup=20):
    x = torch.randn(1, n_features, WINDOW)
    with torch.no_grad():
        for _ in range(warmup): _ = m(x)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs): _ = m(x)
    return (time.time() - t0) / n_runs * 1000


def ram_bench(m):
    tracemalloc.start()
    with torch.no_grad():
        _ = m(torch.randn(1, m.tcn[0].conv1.in_channels, WINDOW))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    weight_mem = sum(p.numel() * p.element_size() for p in m.parameters())
    return weight_mem, peak


# ────────── 主流程 ──────────
def main():
    t_start = time.time()
    n_features, train_loader, val_loader, test_loader, y_test = load_data()

    print("=" * 90)
    print(" TCN+SE 19-dim — 3 个 MCU 重点候选 (ch=8/16/32) Hybrid 量化")
    print(" 配置: LR=2e-3, B=128, b=3, d=[1,2,4], seed=42")
    print("=" * 90)
    print(f" 数据: {n_features} 维 features, W={WINDOW}, N_train/val/test windows = "
          f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    all_results = []

    for ch in CHANNELS_LIST:
        print(f"\n{'─' * 90}")
        print(f"▶ ch = {ch}  (目标: {['Arduino Uno', 'STM32F4 / ESP32', 'RP2040 / 边缘服务器'][CHANNELS_LIST.index(ch)]})")
        print(f"{'─' * 90}")

        # 1. 训练
        print(f"\n[1] 训练 ch={ch} ...")
        model, n_params, best_epoch, train_time = train_one(
            ch, n_features, train_loader, val_loader, test_loader, y_test
        )
        print(f"    Params: {n_params:,}  best epoch: {best_epoch}  train: {train_time:.1f}s")

        # 2. FP32 评估
        print(f"\n[2] FP32 评估 ...")
        p_fp32, _ = predict(test_loader, model)
        f1m_fp32 = f1_score(y_test, (p_fp32 >= 0.5).astype(int), average="macro")
        pr_fp32  = average_precision_score(y_test, p_fp32)
        bin_fp32 = f1_score(y_test, (p_fp32 >= 0.5).astype(int), average="binary")
        acc_fp32 = float(((p_fp32 >= 0.5).astype(int) == y_test).mean())
        lat_fp32 = latency_bench(model, n_features)
        wmem_fp32, act_fp32 = ram_bench(model)
        print(f"    F1m={f1m_fp32:.4f}  PR-AUC={pr_fp32:.4f}  Bin-F1={bin_fp32:.4f}  Acc={acc_fp32:.4f}")
        print(f"    Latency: {lat_fp32:.3f} ms  RAM: w={wmem_fp32/1024:.2f}KB act={act_fp32/1024:.2f}KB")

        # 3. 量化
        print(f"\n[3] Hybrid 量化 (INT8-w + FP32-a + BN 折叠) ...")
        quant_tensors, bias_tensors, snr = collect_quant_state(model)
        min_snr = min(snr.values())
        avg_snr = float(np.mean(list(snr.values())))
        print(f"    {len(quant_tensors)} weight tensors quantized, {len(bias_tensors)} FP32 biases")
        print(f"    Per-layer SNR: min={min_snr:.2f} dB  avg={avg_snr:.2f} dB")
        if min_snr < 20:
            print(f"    ⚠️  WARN: 某层 SNR < 20 dB, 可能有精度损失")

        # 4. 保存 .bin/.h/.json
        print(f"\n[4] 保存 .bin/.h/.json ...")
        bin_size, h_size, meta = save_hybrid_files(ch, quant_tensors, bias_tensors, snr)
        print(f"    .bin = {bin_size:,} B ({bin_size/1024:.2f} KB)")
        print(f"    .h   = {h_size:,} B ({h_size/1024:.2f} KB) [编译后约 {bin_size/1024:.1f} KB]")

        # 5. Hybrid 重建 + 评估
        print(f"\n[5] Hybrid 模型重建 + 评估 ...")
        hybrid = build_hybrid_model(ch, n_features, quant_tensors, bias_tensors)
        p_hyb, _ = predict(test_loader, hybrid)
        f1m_hyb = f1_score(y_test, (p_hyb >= 0.5).astype(int), average="macro")
        pr_hyb  = average_precision_score(y_test, p_hyb)
        bin_hyb = f1_score(y_test, (p_hyb >= 0.5).astype(int), average="binary")
        acc_hyb = float(((p_hyb >= 0.5).astype(int) == y_test).mean())
        lat_hyb = latency_bench(hybrid, n_features)
        wmem_hyb, act_hyb = ram_bench(hybrid)
        diff = np.abs(p_fp32 - p_hyb)
        agree = float(((p_fp32 >= 0.5) == (p_hyb >= 0.5)).mean())
        print(f"    F1m={f1m_hyb:.4f}  PR-AUC={pr_hyb:.4f}  Bin-F1={bin_hyb:.4f}  Acc={acc_hyb:.4f}")
        print(f"    Δ vs FP32: F1m={f1m_hyb-f1m_fp32:+.4f}  PR-AUC={pr_hyb-pr_fp32:+.4f}  Bin-F1={bin_hyb-bin_fp32:+.4f}")
        print(f"    Latency: {lat_hyb:.3f} ms  RAM: w={wmem_hyb/1024:.2f}KB act={act_hyb/1024:.2f}KB")
        print(f"    Pred agreement: {agree*100:.2f}%  max|Δp|={diff.max():.4f}  mean|Δp|={diff.mean():.5f}")

        all_results.append({
            "channels": ch, "n_params": n_params, "best_epoch": best_epoch, "train_time_s": train_time,
            "fp32": {"f1m": f1m_fp32, "pr_auc": pr_fp32, "bin_f1": bin_fp32, "acc": acc_fp32,
                     "latency_ms": lat_fp32, "weight_kb": wmem_fp32/1024, "act_peak_kb": act_fp32/1024},
            "hybrid": {"f1m": f1m_hyb, "pr_auc": pr_hyb, "bin_f1": bin_hyb, "acc": acc_hyb,
                       "latency_ms": lat_hyb, "weight_kb": wmem_hyb/1024, "act_peak_kb": act_hyb/1024,
                       "bin_size_b": bin_size, "h_size_b": h_size,
                       "min_snr_db": min_snr, "avg_snr_db": avg_snr,
                       "pred_agreement": agree, "max_diff": float(diff.max()), "mean_diff": float(diff.mean())},
        })

    # ────────── 最终汇总 ──────────
    total = time.time() - t_start
    print(f"\n{'=' * 90}")
    print(f" FINAL SUMMARY — 3 ch × FP32 vs Hybrid (总耗时 {total:.1f}s)")
    print(f"{'=' * 90}")
    print(f"{'ch':>4} {'Params':>7} │ {'F1m FP32→Hyb':>15} {'Δ':>7} {'PR-AUC Δ':>9} │ "
          f"{'.bin KB':>8} {'Lat FP32→Hyb':>15} │ {'Agreement':>10} {'SNR min':>8}")
    print("─" * 90)
    for r in all_results:
        df1 = r["hybrid"]["f1m"] - r["fp32"]["f1m"]
        dpr = r["hybrid"]["pr_auc"] - r["fp32"]["pr_auc"]
        print(f"{r['channels']:>4} {r['n_params']:>7,} │ "
              f"{r['fp32']['f1m']:.4f}→{r['hybrid']['f1m']:.4f} {df1:>+7.4f} {dpr:>+9.4f} │ "
              f"{r['hybrid']['bin_size_b']/1024:>7.2f} {r['fp32']['latency_ms']:.3f}→{r['hybrid']['latency_ms']:.3f} ms │ "
              f"{r['hybrid']['pred_agreement']*100:>9.2f}% {r['hybrid']['min_snr_db']:>7.2f} dB")
    print("─" * 90)

    # 部署推荐
    print(f"\n🎯 部署推荐 (基于 3-ch Hybrid):")
    print(f"   - 极致小 (Arduino Uno 32K Flash / 2K RAM)    → ch=8   "
          f"({all_results[0]['n_params']:,} params, .bin={all_results[0]['hybrid']['bin_size_b']/1024:.1f}KB, F1m={all_results[0]['hybrid']['f1m']:.4f})")
    print(f"   - 平衡 (STM32F4 / ESP32)                       → ch=16  "
          f"({all_results[1]['n_params']:,} params, .bin={all_results[1]['hybrid']['bin_size_b']/1024:.1f}KB, F1m={all_results[1]['hybrid']['f1m']:.4f})")
    print(f"   - 生产通用 (RP2040 / 边缘服务器)               → ch=32  "
          f"({all_results[2]['n_params']:,} params, .bin={all_results[2]['hybrid']['bin_size_b']/1024:.1f}KB, F1m={all_results[2]['hybrid']['f1m']:.4f})")

    # 保存完整 JSON
    with open(os.path.join(BASE, "hybrid_quant_3ch_results.json"), "w") as f:
        json.dump({"config": {"lr": LR, "batch": BATCH_TRAIN, "epochs": EPOCHS, "seed": SEED,
                              "window": WINDOW, "n_features": n_features, "n_blocks": N_BLOCKS,
                              "dilations": DILATIONS, "se_reduction": SE_REDUCTION},
                   "results": all_results, "total_time_s": total}, f, indent=2)
    print(f"\n完整结果已保存: hybrid_quant_3ch_results.json")


if __name__ == "__main__":
    main()
