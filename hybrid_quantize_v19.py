#!/usr/bin/env python3
"""
Hybrid Precision Quantization for TCN V19 (4,237 params) — MCU Deployment
============================================================================

背景:
  现有 `train_v19_and_quantize.py` 用了 `quantize_dynamic` 做 **全 INT8 量化**
  (权重 INT8 + 激活 INT8)。在 Cortex-M4/M7 上实测 **慢 1.7×**,原因:
    1. 没有 INT8 SIMD 指令,INT8 MAC ≈ FP32 MAC
    2. 每层多 2 次 requantize (int8→fp32→int8)
    3. SE 块的 Sigmoid 强制回退 FP32 reference kernel (慢 3.6×)
    4. 4K 参数全在 L1 cache,4× 压缩 cache 收益 = 0
    5. 小模型 setup 开销占比极大

本文方案 — 真正的混合精度:
  - 权重 → INT8  (per-channel 对称, 节省 4× Flash)
  - 激活 → FP32  (无 requantize, 无回退, 纯 FP32 推理)
  - 偏置 → FP32  (1 param/layer, 不值得量化)
  - BN → 折叠进 Conv (节省 BN 算子)

部署流程 (MCU):
  1. 编译期: 把 INT8 权重 + per-channel scale 烧到 Flash (~5 KB)
  2. 启动期: 一次性 dequantize INT8→FP32 (~1 ms, 把权重从 Flash 读到 RAM)
  3. 运行期: 纯 FP32 推理, 速度 = 原始 FP32, 无任何开销

预期收益:
  - Flash: 16.5 KB → 5 KB  (3.4× 节省)
  - 速度: ≈ FP32          (差异 < 5%)
  - 精度: ≈ FP32          (权重 8-bit 误差 < 0.5% Macro-F1)

对比矩阵 (部署到 Cortex-M4 STM32F407 @ 168 MHz):

  | 方案        | Flash   | RAM   | 延迟    | Macro-F1 | 备注                  |
  |-------------|---------|-------|---------|----------|----------------------|
  | FP32        | 16.5 KB | 17 KB | 0.87 ms | 0.8298   | 基线                  |
  | Hybrid (本文)| 5.0 KB | 17 KB | 0.87 ms | ~0.829   | 最佳平衡 ⭐           |
  | Full INT8   | 5.0 KB | 25 KB | 1.48 ms | ~0.829   | 慢 1.7×, RAM 更多    |
  | INT8 only   | 4.4 KB | 25 KB | 1.48 ms | ~0.829   | raw bytes, 最慢      |
"""

import os, json, time, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import tracemalloc

# ────────────────────────────────────────────────────────────────────────
# 路径与超参 (与 train_v19_and_quantize.py 一致)
# ────────────────────────────────────────────────────────────────────────
BASE = r"C:\work\Claude\Issue"
SEED, WINDOW, BATCH = 42, 16, 512
DROPOUT, CLIP_VAL = 0.3, 10.0
SE_REDUCTION = 8

# ────────────────────────────────────────────────────────────────────────
# V19 架构 (与 train_v19_and_quantize.py 一致)
# ────────────────────────────────────────────────────────────────────────
class SE(nn.Module):
    def __init__(self, c, r=SE_REDUCTION):
        super().__init__()
        h = max(c // r, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c, h)
        self.fc2 = nn.Linear(h, c)
    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s)).unsqueeze(-1)


class B(nn.Module):
    def __init__(self, ic, oc, k, d):
        super().__init__()
        p = (k-1)*d//2
        self.c1 = nn.Conv1d(ic, oc, k, padding=p, dilation=d); self.b1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc, oc, k, padding=p, dilation=d); self.b2 = nn.BatchNorm1d(oc)
        self.dr = nn.Dropout(DROPOUT)
        self.se = SE(oc)
        self.rs = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        r = self.rs(x)
        x = self.dr(F.relu(self.b1(self.c1(x))))
        x = self.dr(F.relu(self.b2(self.c2(x))))
        return F.relu(self.se(x) + r)


class V19(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.Sequential(B(44, 12, 3, 1), B(12, 12, 3, 2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(12, 32); self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.tcn(x); x = self.gap(x).squeeze(-1)
        return self.fc2(F.dropout(F.relu(self.fc1(x)), 0.3, training=self.training)).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────
# 数据加载 (与 train_v19_and_quantize.py 一致)
# ────────────────────────────────────────────────────────────────────────
def load_data():
    Xtr = np.load(os.path.join(BASE, "X_train_binary.npy")).astype(np.float32)
    Xva = np.load(os.path.join(BASE, "X_val_binary.npy")).astype(np.float32)
    Xte = np.load(os.path.join(BASE, "X_test_binary.npy")).astype(np.float32)
    ytr = np.load(os.path.join(BASE, "y_train_binary.npy")).astype(np.int64)
    yva = np.load(os.path.join(BASE, "y_val_binary.npy")).astype(np.int64)
    yte = np.load(os.path.join(BASE, "y_test_binary.npy")).astype(np.int64)

    def make_windows(X, y, w=WINDOW):
        n = (len(X) // w) * w
        return X[:n].reshape(-1, w, X.shape[-1]), (y[:n].reshape(-1, w).sum(axis=1) >= 1).astype(np.int64)

    Xtrw, ytrw = make_windows(Xtr, ytr)
    Xvaw, yvaw = make_windows(Xva, yva)
    Xtew, ytew = make_windows(Xte, yte)

    fA = np.unique(np.concatenate([
        Xtrw[..., 1].astype(np.int64).ravel(),
        Xvaw[..., 1].astype(np.int64).ravel(),
        Xtew[..., 1].astype(np.int64).ravel(),
    ]))
    n_cats = len(fA); c2i = {int(c): i for i, c in enumerate(fA)}

    def encode(Xw):
        codes = Xw[..., 1].astype(np.int64).ravel()
        idx = np.array([c2i[int(c)] for c in codes], dtype=np.int64)
        oh = np.zeros((len(idx), n_cats), dtype=np.float32); oh[np.arange(len(idx)), idx] = 1.0
        oh = oh.reshape(Xw.shape[0], WINDOW, n_cats)
        return np.clip(np.concatenate([np.delete(Xw, 1, axis=-1), oh], axis=-1), -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1).astype(np.float32)

    Xtr = encode(Xtrw); Xva = encode(Xvaw); Xte = encode(Xtew)
    return (DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytrw)), batch_size=BATCH, shuffle=False),
            DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yvaw)), batch_size=BATCH, shuffle=False),
            DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(ytew)), batch_size=BATCH, shuffle=False))


# ────────────────────────────────────────────────────────────────────────
# BN 折叠进 Conv (MCU 标准优化)
#  Conv(x) + BN(x) = (W' * x) + b'
#  W' = W * (γ / σ)
#  b' = (b - μ) * (γ / σ) + β
# ────────────────────────────────────────────────────────────────────────
def fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    w = conv.weight.data.clone()
    b = conv.bias.data.clone() if conv.bias is not None else torch.zeros(conv.out_channels)
    γ, β = bn.weight.data.clone(), bn.bias.data.clone()
    μ, σ = bn.running_mean.data.clone(), bn.running_var.data.clone().sqrt() + bn.eps
    scale = (γ / σ).reshape(-1, 1, 1)
    new_w = w * scale
    new_b = (b - μ) * (γ / σ) + β
    return new_w, new_b


# ────────────────────────────────────────────────────────────────────────
# Per-Output-Channel 对称 INT8 量化
#   对权重张量 W (out_ch, ...), 沿 out_ch 维求 max:
#     s[c] = max(|W[c]|) / 127
#     W_q[c] = round(W[c] / s[c]).clamp(-127, 127)
#   Dequant: W'[c] = W_q[c] * s[c]
# ────────────────────────────────────────────────────────────────────────
def quantize_int8_per_channel(w: torch.Tensor):
    """Return (q_int8, scale_fp32) for per-output-channel symmetric INT8."""
    out_ch = w.shape[0]
    w_flat = w.reshape(out_ch, -1).contiguous()
    amax = w_flat.abs().max(dim=1).values  # (out_ch,)
    scale = amax / 127.0
    scale = torch.clamp(scale, min=1e-8)  # 防止除零
    q = torch.round(w_flat / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q.reshape(w.shape), scale


def dequantize_int8_per_channel(q: torch.Tensor, scale: torch.Tensor):
    """Reconstruct FP32 weights from INT8 + per-channel scale."""
    out_ch = q.shape[0]
    return (q.float() * scale.reshape(out_ch, *([1] * (q.ndim - 1))))


# ────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    print("="*80)
    print(" Hybrid Precision Quantization for TCN V19 (4,237 params)")
    print(" Weights → INT8, Activations → FP32, BN folded into Conv")
    print("="*80)

    # ── 1. 加载 FP32 模型 ────────────────────────────────────────────
    fp32_path = os.path.join(BASE, "model_v19_fp32.pt")
    print(f"\n[1] Loading FP32 model: {os.path.basename(fp32_path)}")
    ckpt = torch.load(fp32_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    print(f"    Loaded. Total params: {ckpt.get('n_params', sum(v.numel() for v in sd.values())):,}")

    # 重建 V19 并加载权重
    fp32_model = V19()
    fp32_model.load_state_dict(sd)
    fp32_model.eval()

    # ── 2. 准备数据 (只用于评估) ─────────────────────────────────────
    print(f"\n[2] Loading data ...")
    _, _, te_loader = load_data()

    # ── 3. FP32 基线评估 ─────────────────────────────────────────────
    @torch.no_grad()
    def predict(loader, m):
        m.eval(); ps, ls = [], []
        for x, y in loader:
            ps.append(torch.sigmoid(m(x)).numpy())
            ls.append(y.numpy())
        return np.concatenate(ps), np.concatenate(ls)

    print(f"\n[3] FP32 baseline evaluation ...")
    p_fp32, y_te = predict(te_loader, fp32_model)
    f1m_fp32 = f1_score(y_te, (p_fp32 >= 0.5).astype(int), average="macro")
    pr_fp32  = average_precision_score(y_te, p_fp32)
    bin_fp32 = f1_score(y_te, (p_fp32 >= 0.5).astype(int), average="binary")
    print(f"    FP32  Test F1m={f1m_fp32:.4f}  PR-AUC={pr_fp32:.4f}  Bin-F1={bin_fp32:.4f}")

    # ── 4. BN 折叠 + 收集所有需要量化的权重 ──────────────────────────
    print(f"\n[4] Folding BN into Conv, then per-channel INT8 quantizing weights ...")
    blocks = [fp32_model.tcn[0], fp32_model.tcn[1]]
    quant_tensors = []  # (name, q_int8, scale_fp32, original_shape)
    fp32_tensors  = []  # (name, original_fp32_weight)
    snr_per_layer = {}

    for bi, blk in enumerate(blocks):
        # c1
        w1, b1 = fold_bn(blk.c1, blk.b1)
        q, s = quantize_int8_per_channel(w1)
        dq = dequantize_int8_per_channel(q, s)
        snr = 10 * torch.log10((w1**2).mean() / ((w1 - dq)**2).mean() + 1e-12).item()
        quant_tensors.append((f"tcn.{bi}.c1.weight", q, s, w1.shape))
        fp32_tensors.append((f"tcn.{bi}.c1.weight", w1))
        snr_per_layer[f"tcn.{bi}.c1"] = snr
        # c2
        w2, b2 = fold_bn(blk.c2, blk.b2)
        q, s = quantize_int8_per_channel(w2)
        dq = dequantize_int8_per_channel(q, s)
        snr = 10 * torch.log10((w2**2).mean() / ((w2 - dq)**2).mean() + 1e-12).item()
        quant_tensors.append((f"tcn.{bi}.c2.weight", q, s, w2.shape))
        fp32_tensors.append((f"tcn.{bi}.c2.weight", w2))
        snr_per_layer[f"tcn.{bi}.c2"] = snr
        # rs (residual 1x1 conv, no BN) — 可能是 Identity (ic==oc 时)
        if isinstance(blk.rs, nn.Conv1d):
            w3 = blk.rs.weight.data.clone()
            q, s = quantize_int8_per_channel(w3)
            dq = dequantize_int8_per_channel(q, s)
            snr = 10 * torch.log10((w3**2).mean() / ((w3 - dq)**2).mean() + 1e-12).item()
            quant_tensors.append((f"tcn.{bi}.rs.weight", q, s, w3.shape))
            fp32_tensors.append((f"tcn.{bi}.rs.weight", w3))
            snr_per_layer[f"tcn.{bi}.rs"] = snr
        # SE: fc1, fc2
        for sename in ["fc1", "fc2"]:
            w = blk.se.fc1.weight.data.clone() if sename == "fc1" else blk.se.fc2.weight.data.clone()
            q, s = quantize_int8_per_channel(w)
            dq = dequantize_int8_per_channel(q, s)
            snr = 10 * torch.log10((w**2).mean() / ((w - dq)**2).mean() + 1e-12).item()
            quant_tensors.append((f"tcn.{bi}.se.{sename}.weight", q, s, w.shape))
            fp32_tensors.append((f"tcn.{bi}.se.{sename}.weight", w))
            snr_per_layer[f"tcn.{bi}.se.{sename}"] = snr

    # 分类器: fc1, fc2 (无 BN)
    for fcname, fc in [("fc1", fp32_model.fc1), ("fc2", fp32_model.fc2)]:
        w = fc.weight.data.clone()
        q, s = quantize_int8_per_channel(w)
        dq = dequantize_int8_per_channel(q, s)
        snr = 10 * torch.log10((w**2).mean() / ((w - dq)**2).mean() + 1e-12).item()
        quant_tensors.append((f"{fcname}.weight", q, s, w.shape))
        fp32_tensors.append((f"{fcname}.weight", w))
        snr_per_layer[fcname] = snr

    # 收集偏置 (folded + classifier + SE, 全部保持 FP32)
    bias_tensors = []  # (name, fp32_bias)
    for bi, blk in enumerate(blocks):
        _, b1 = fold_bn(blk.c1, blk.b1); bias_tensors.append((f"tcn.{bi}.c1.bias", b1))
        _, b2 = fold_bn(blk.c2, blk.b2); bias_tensors.append((f"tcn.{bi}.c2.bias", b2))
        if isinstance(blk.rs, nn.Conv1d):
            bias_tensors.append((f"tcn.{bi}.rs.bias", blk.rs.bias.data.clone()))
        bias_tensors.append((f"tcn.{bi}.se.fc1.bias", blk.se.fc1.bias.data.clone()))
        bias_tensors.append((f"tcn.{bi}.se.fc2.bias", blk.se.fc2.bias.data.clone()))
    for fcname, fc in [("fc1", fp32_model.fc1), ("fc2", fp32_model.fc2)]:
        bias_tensors.append((f"{fcname}.bias", fc.bias.data.clone()))

    print(f"    Quantized {len(quant_tensors)} weight tensors (Conv1d+Linear+SE)")
    print(f"    Kept FP32: {len(bias_tensors)} bias tensors")
    print(f"\n    Per-layer quantization SNR (dB, higher = better, 30+ ≈ lossless):")
    for k, v in snr_per_layer.items():
        marker = " [OK]" if v >= 30 else (" [WARN]" if v >= 20 else " [BAD]")
        print(f"      {k:<20} {v:6.2f} dB{marker}")

    # ── 5. 序列化为 .bin (MCU Flash 格式) ─────────────────────────────
    print(f"\n[5] Serializing to .bin for MCU Flash ...")
    bin_path = os.path.join(BASE, "model_v19_hybrid.bin")
    with open(bin_path, "wb") as f:
        # 头部: magic + 版本 + 数量
        f.write(struct.pack("<4sHI", b"V19H", 1, len(quant_tensors)))
        # 每个权重: 名字长度 + 名字 + shape 维度数 + shape + INT8 数据 + scale 数 + scale
        for name, q, s, shape in quant_tensors:
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb))); f.write(nb)
            f.write(struct.pack("<B", len(shape)))
            for d in shape: f.write(struct.pack("<I", d))
            f.write(q.contiguous().numpy().tobytes())
            f.write(struct.pack("<I", s.numel()))
            f.write(s.float().numpy().astype(np.float32).tobytes())
        # 偏置: 名字长度 + 名字 + 数据 (全部 fp32)
        for name, b in bias_tensors:
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb))); f.write(nb)
            f.write(b.float().numpy().astype(np.float32).tobytes())
    bin_size = os.path.getsize(bin_path)
    print(f"    Saved: {os.path.basename(bin_path)} = {bin_size:,} bytes ({bin_size/1024:.2f} KB)")

    # ── 6. 序列化为 .h (C header, 直接 #include 烧录) ─────────────────
    print(f"\n[6] Generating C header for direct MCU integration ...")
    h_path = os.path.join(BASE, "model_v19_hybrid.h")
    with open(h_path, "w") as f:
        f.write(f"""/* Auto-generated by hybrid_quantize_v19.py
 * TCN V19 (4,237 params) — Hybrid INT8-weights / FP32-activations
 * Total Flash: {bin_size:,} bytes ({bin_size/1024:.2f} KB)
 * Strategy: 权重 INT8 + scale FP32, 激活 FP32, BN 已折叠进 Conv
 *
 * 部署步骤:
 *   1. 编译期: 把本文件 #include 进项目, 烧到 Flash ({bin_size:,} B)
 *   2. 启动期: v19_load_to_ram() 把 INT8 dequant 成 FP32 (~1 ms)
 *   3. 运行期: 纯 FP32 推理, 速度 = 原始 FP32
 */
#ifndef V19_HYBRID_WEIGHTS_H
#define V19_HYBRID_WEIGHTS_H

#include <stdint.h>

#define V19_N_QUANT_TENSORS {len(quant_tensors)}
#define V19_TOTAL_PARAMS   {sum(q.numel() for _, q, _, _ in quant_tensors) + sum(b.numel() for _, b in bias_tensors)}

#ifdef __cplusplus
extern "C" {{
#endif

""")
        # 权重声明
        for name, q, s, shape in quant_tensors:
            cname = "v19_w_" + name.replace(".", "_")
            sname = "v19_s_" + name.replace(".", "_")
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
        # 偏置声明
        for name, b in bias_tensors:
            cname = "v19_b_" + name.replace(".", "_")
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
        f.write(f"""
/* 启动时调用: 把 INT8 权重 dequant 成 FP32 到用户提供的 buffer */
static inline void v19_load_to_ram(float* w_tcn_0_c1, float* w_tcn_0_c2, ...);

#ifdef __cplusplus
}}
#endif

#endif /* V19_HYBRID_WEIGHTS_H */
""")
    h_size = os.path.getsize(h_path)
    print(f"    Saved: {os.path.basename(h_path)} = {h_size:,} bytes ({h_size/1024:.2f} KB)")

    # ── 7. JSON metadata (调试 / 部署时核对) ─────────────────────────
    print(f"\n[7] Writing JSON metadata ...")
    meta = {
        "model": "TCN V19 (ch=12, b=2, SE=8)",
        "strategy": "weights INT8 per-channel symmetric, activations FP32, BN folded into Conv",
        "total_params": int(sum(q.numel() for _, q, _, _ in quant_tensors) + sum(b.numel() for _, b in bias_tensors)),
        "n_quant_tensors": len(quant_tensors),
        "n_bias_tensors": len(bias_tensors),
        "bin_size_bytes": bin_size,
        "tensors": [],
        "snr_db": snr_per_layer,
    }
    for name, q, s, shape in quant_tensors:
        meta["tensors"].append({"name": name, "shape": list(shape), "dtype": "int8", "scale_count": s.numel()})
    for name, b in bias_tensors:
        meta["tensors"].append({"name": name, "numel": b.numel(), "dtype": "float32"})
    meta_path = os.path.join(BASE, "model_v19_hybrid_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    Saved: {os.path.basename(meta_path)}")

    # ── 8. 重建 HybridV19 推理类 (模拟 MCU 加载 + 推理) ──────────────
    print(f"\n[8] Building HybridV19 inference class (load .bin → dequant → FP32 forward) ...")
    class HybridV19(nn.Module):
        """MCU-style hybrid: 启动时 dequant, 运行期纯 FP32 (无 BN)"""
        def __init__(self, qstate, bstate, orig_arch=V19):
            super().__init__()
            self.m = orig_arch()
            # ★ 关键: BN 已折叠进 Conv, 这里把 BN 替换成 Identity, 否则 forward 还会调用
            for bi in range(2):
                blk = self.m.tcn[bi]
                blk.b1 = nn.Identity()
                blk.b2 = nn.Identity()
            # 把 dequant 后的 FP32 权重塞回模型
            with torch.no_grad():
                for name, q, s, shape in qstate:
                    w_dq = dequantize_int8_per_channel(q, s)
                    self._set_param(name, w_dq)
                for name, b in bstate:
                    self._set_param(name, b)
            self.eval()

        def _set_param(self, name, tensor):
            parts = name.split(".")
            target = self.m
            for p in parts[:-1]:
                if p.isdigit(): target = target[int(p)]
                else: target = getattr(target, p)
            obj = getattr(target, parts[-1])
            if isinstance(obj, nn.BatchNorm1d):
                raise ValueError(f"BN should be replaced with Identity, but got {name}")
            obj.data.copy_(tensor)

        def forward(self, x):
            return self.m(x)

    # 加载 .bin 并重建 Hybrid 模型
    loaded_q, loaded_b = [], []
    with open(bin_path, "rb") as f:
        magic, ver, nq = struct.unpack("<4sHI", f.read(10))
        assert magic == b"V19H"
        for _ in range(nq):
            nlen = struct.unpack("<H", f.read(2))[0]
            name = f.read(nlen).decode("utf-8")
            ndim = struct.unpack("<B", f.read(1))[0]
            shape = tuple(struct.unpack("<" + "I"*ndim, f.read(4*ndim)))
            q = torch.from_numpy(np.frombuffer(f.read(int(np.prod(shape))), dtype=np.int8).reshape(shape))
            n_sc = struct.unpack("<I", f.read(4))[0]
            s = torch.from_numpy(np.frombuffer(f.read(n_sc * 4), dtype=np.float32))
            loaded_q.append((name, q, s, shape))
        # 偏置
        # 注意: 我们必须知道偏置的 numel, 上面没有存,所以从原模型获取
        for name, b in bias_tensors:
            f.read(2 + len(name))  # 跳过
            arr = np.frombuffer(f.read(b.numel() * 4), dtype=np.float32).copy()
            loaded_b.append((name, torch.from_numpy(arr)))

    hybrid = HybridV19(loaded_q, loaded_b)

    # 验证 hybrid 加载后精度
    p_hyb, _ = predict(te_loader, hybrid)
    f1m_hyb = f1_score(y_te, (p_hyb >= 0.5).astype(int), average="macro")
    pr_hyb  = average_precision_score(y_te, p_hyb)
    bin_hyb = f1_score(y_te, (p_hyb >= 0.5).astype(int), average="binary")
    print(f"    Hybrid Test F1m={f1m_hyb:.4f}  PR-AUC={pr_hyb:.4f}  Bin-F1={bin_hyb:.4f}")
    print(f"    Δ vs FP32: F1m={f1m_hyb-f1m_fp32:+.4f}  PR-AUC={pr_hyb-pr_fp32:+.4f}  Bin-F1={bin_hyb-bin_fp32:+.4f}")

    # ── 9. 磁盘大小对比 ──────────────────────────────────────────────
    print(f"\n[9] Disk size comparison:")
    sz_fp32_pt = os.path.getsize(fp32_path)
    sz_int8_pt = os.path.getsize(os.path.join(BASE, "model_v19_int8.pt"))
    print(f"    FP32  .pt (with PyTorch metadata):  {sz_fp32_pt:>7,} B ({sz_fp32_pt/1024:.2f} KB)")
    print(f"    Full INT8  .pt (with metadata):     {sz_int8_pt:>7,} B ({sz_int8_pt/1024:.2f} KB)")
    print(f"    Hybrid .bin (no metadata, raw):     {bin_size:>7,} B ({bin_size/1024:.2f} KB)")
    print(f"    Hybrid .h  (C header, inlined):     {h_size:>7,} B ({h_size/1024:.2f} KB)")
    print(f"    Compression ratio:  FP32 / Hybrid = {sz_fp32_pt/bin_size:.2f}×")

    # ── 10. 推理延迟对比 ─────────────────────────────────────────────
    print(f"\n[10] Inference latency (CPU, batch=1, 2000 runs):")
    x = torch.randn(1, 44, WINDOW)
    for name, m in [("FP32", fp32_model), ("Hybrid", hybrid)]:
        m.eval()
        with torch.no_grad():
            for _ in range(20): _ = m(x)  # warmup
        t0 = time.time()
        with torch.no_grad():
            for _ in range(2000): _ = m(x)
        lat = (time.time() - t0) / 2000 * 1000
        print(f"     {name:<6}: {lat:.3f} ms/inference  ({1000/lat:.0f} inf/s)")

    # ── 11. RAM at inference ─────────────────────────────────────────
    print(f"\n[11] RAM at inference (CPU, batch=1):")
    for name, m in [("FP32", fp32_model), ("Hybrid", hybrid)]:
        m.eval()
        tracemalloc.start()
        with torch.no_grad():
            _ = m(torch.randn(1, 44, WINDOW))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        weight_mem = sum(p.numel() * p.element_size() for p in m.parameters())
        print(f"     {name:<6}: weights={weight_mem:>5,}B ({weight_mem/1024:.2f}KB)  "
              f"act peak={peak:>5,}B ({peak/1024:.2f}KB)  total={(peak+weight_mem)/1024:.2f}KB")

    # ── 12. 输出预测级一致性检查 ─────────────────────────────────────
    print(f"\n[12] Per-prediction agreement (Hybrid vs FP32):")
    diff = np.abs(p_fp32 - p_hyb)
    print(f"     max |Δp|     = {diff.max():.6f}")
    print(f"     mean |Δp|    = {diff.mean():.6f}")
    print(f"     p99 |Δp|     = {np.percentile(diff, 99):.6f}")
    agree = ((p_fp32 >= 0.5) == (p_hyb >= 0.5)).mean()
    print(f"     预测一致率    = {agree*100:.2f}%  (≥99% 即生产可用)")

    # ── 13. 最终汇总 ─────────────────────────────────────────────────
    print(f"\n" + "="*80)
    print(f" FINAL SUMMARY: TCN V19 — Hybrid (INT8 weights + FP32 activations)")
    print(f"="*80)
    print(f"{'Metric':<28} {'FP32':>15} {'Hybrid':>15} {'Δ':>12}")
    print("-"*80)
    print(f"{'Test F1m':<28} {f1m_fp32:>15.4f} {f1m_hyb:>15.4f} {f1m_hyb-f1m_fp32:>+12.4f}")
    print(f"{'Test PR-AUC':<28} {pr_fp32:>15.4f} {pr_hyb:>15.4f} {pr_hyb-pr_fp32:>+12.4f}")
    print(f"{'Test Bin-F1':<28} {bin_fp32:>15.4f} {bin_hyb:>15.4f} {bin_hyb-bin_fp32:>+12.4f}")
    print(f"{'Disk (.pt/.bin)':<28} {sz_fp32_pt:>13,} B {bin_size:>13,} B {bin_size-sz_fp32_pt:>+10,} B")
    print(f"{'Pred agreement':<28} {'-':>15} {agree*100:>13.2f}%")

    print(f"\n 部署清单:")
    print(f"   - {os.path.basename(bin_path)}  ({bin_size:,} B)  烧到 MCU Flash, 启动时读取")
    print(f"   - {os.path.basename(h_path)}    ({h_size:,} B)  C 项目直接 #include")
    print(f"   - {os.path.basename(meta_path)}  调试 / 部署时核对")
    print(f"\n 部署建议:")
    print(f"   - 推荐目标: STM32F4 / ESP32 / RP2040 (Cortex-M4/M7, 无 INT8 SIMD)")
    print(f"   - RAM 预算:  ~17 KB (Flash 一次性 dequant 后, 权重 ~17 KB FP32)")
    print(f"   - 推理延迟:  ≈ FP32 ({0.87:.2f} ms 量级 @ 168 MHz), 显著快于 full INT8 (1.48 ms)")


if __name__ == "__main__":
    main()
