#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""23 维 CNN1D (hidden=64) -> STM32H743 C header

流程:
  1. 加载 model_cnn1d_23dim_ch64_s{seed}.pt
  2. BN fold 进 Conv (推理时 C 端只需 Conv+ReLU)
  3. 权重 FP32 导出 (与 H743 ch32_inference.c 的 Hybrid 策略一致:
     启动时一次性拷贝到 .data/.bss, 运行期纯 FP32)
  4. 生成 model_cnn_23dim_ch64_s{seed}.h

架构 (与 train_cnn_23dim.py 完全一致):
  Conv1d(23->64,k=3,pad=1)+BN -> ReLU
  Conv1d(64->64,k=3,pad=1)+BN -> ReLU
  Conv1d(64->64,k=3,pad=1)+BN -> ReLU
  AdaptiveAvgPool1d(1)
  Linear(64->32) -> ReLU
  Linear(32->1)   -> Sigmoid

用法:
    python export_cnn_23dim_to_h.py [--seed 42]
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn

BASE = r"C:\work\Claude\Issue"
H743_INC = os.path.join(BASE, "CNNH743", "h743", "Core", "Inc")

N_FEATURES = 23
WINDOW     = 16
HIDDEN     = 64
KERNEL     = 3
PAD        = 1
FC1_OUT    = 32
FC2_OUT    = 1


def fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """Conv1d + BN1d -> 等效 Conv1d (bias=True)"""
    w = conv.weight.data.numpy().astype(np.float64)
    b = (conv.bias.data.numpy().astype(np.float64)
         if conv.bias is not None else np.zeros(conv.out_channels))
    bn_w = bn.weight.data.numpy().astype(np.float64)
    bn_b = bn.bias.data.numpy().astype(np.float64)
    bn_mean = bn.running_mean.numpy().astype(np.float64)
    bn_var = bn.running_var.numpy().astype(np.float64)
    scale = bn_w / np.sqrt(bn_var + bn.eps)
    folded_w = w * scale[:, None, None]
    folded_b = (b - bn_mean) * scale + bn_b
    return folded_w.astype(np.float32), folded_b.astype(np.float32)


def gen_float_array(name: str, arr: np.ndarray, per_line: int = 8) -> str:
    flat = arr.flatten()
    n = flat.size
    lines = [f"static const float {name}[{n}] = {{"]
    row = []
    for i, v in enumerate(flat):
        row.append(f"{float(v):.9e}f")
        if (i + 1) % per_line == 0 or i == n - 1:
            lines.append("    " + ", ".join(row) + ("," if i != n - 1 else ""))
            row = []
    lines.append("};")
    return "\n".join(lines)


def export(seed: int):
    ckpt_path = os.path.join(BASE, f"model_cnn1d_23dim_ch64_s{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    conv1w, conv1b = fold_bn(
        nn.Conv1d(N_FEATURES, HIDDEN, KERNEL, padding=PAD), nn.BatchNorm1d(HIDDEN))
    # 直接用 state_dict 构造等价模块再 fold, 避免加载整个模型
    def _fold(prefix):
        conv = nn.Conv1d(sd[f"{prefix}.weight"].shape[1], sd[f"{prefix}.weight"].shape[0],
                         sd[f"{prefix}.weight"].shape[2], padding=PAD)
        with torch.no_grad():
            conv.weight.copy_(sd[f"{prefix}.weight"])
            conv.bias.copy_(sd[f"{prefix}.bias"])
        bn = nn.BatchNorm1d(sd[f"{prefix}.weight"].shape[0])
        with torch.no_grad():
            bn.weight.copy_(sd[f"{prefix.replace('conv', 'bn')}.weight"])
            bn.bias.copy_(sd[f"{prefix.replace('conv', 'bn')}.bias"])
            bn.running_mean.copy_(sd[f"{prefix.replace('conv', 'bn')}.running_mean"])
            bn.running_var.copy_(sd[f"{prefix.replace('conv', 'bn')}.running_var"])
        return fold_bn(conv, bn)

    w1, b1 = _fold("conv1")
    w2, b2 = _fold("conv2")
    w3, b3 = _fold("conv3")

    wfc1 = sd["fc1.weight"].numpy().astype(np.float32)   # (32, 64)
    bfc1 = sd["fc1.bias"].numpy().astype(np.float32)     # (32,)
    wfc2 = sd["fc2.weight"].numpy().astype(np.float32)   # (1, 32)
    bfc2 = sd["fc2.bias"].numpy().astype(np.float32)     # (1,)

    total_params = int(sum(t.numel() for t in sd.values() if torch.is_tensor(t)))
    n_bytes = (w1.size + b1.size + w2.size + b2.size + w3.size + b3.size
               + wfc1.size + bfc1.size + wfc2.size + bfc2.size) * 4

    thr = ckpt["best_threshold"]
    p = f"cnn23_ch64_s{seed}"
    out = os.path.join(H743_INC, f"model_cnn_23dim_ch64_s{seed}.h")

    L = []
    L.append("/* ==========================================================================")
    L.append(f" * model_cnn_23dim_ch64_s{seed}.h  --  AUTO-GENERATED, DO NOT EDIT")
    L.append(" *")
    L.append(f" * 1D-CNN (23-dim SCADA, window=16, hidden=64)")
    L.append(f" * Source : model_cnn1d_23dim_ch64_s{seed}.pt  (seed={seed})")
    L.append(" *")
    L.append(" *   x (23, 16)")
    L.append(" *     -> Conv1d(23->64,k=3,pad=1)+ReLU")
    L.append(" *     -> Conv1d(64->64,k=3,pad=1)+ReLU")
    L.append(" *     -> Conv1d(64->64,k=3,pad=1)+ReLU")
    L.append(" *     -> GAP -> Linear(64->32)+ReLU -> Linear(32->1) -> Sigmoid")
    L.append(" *")
    L.append(" * BN 已 fold 进 Conv 权重 (见 export_cnn_23dim_to_h.py)")
    L.append(f" * 部署阈值: {thr}")
    L.append(f" * 权重总量: {n_bytes} B ({n_bytes/1024:.2f} KB, FP32)")
    L.append(" *")
    L.append(" * 数据布局 (channel-major, 与 H743 sliding_window.c 一致):")
    L.append(" *   x[c * 16 + t] = 第 c 个特征在 t 时刻的值")
    L.append(" * ==========================================================================")
    L.append(f"#ifndef MODEL_CNN_23DIM_CH64_S{seed}_H")
    L.append(f"#define MODEL_CNN_23DIM_CH64_S{seed}_H")
    L.append("")
    L.append("#include <stdint.h>")
    L.append("")
    L.append(f'#define CNN23_IN_CHANNELS  {N_FEATURES}')
    L.append(f'#define CNN23_WINDOW       {WINDOW}')
    L.append(f'#define CNN23_HIDDEN       {HIDDEN}')
    L.append(f'#define CNN23_KERNEL       {KERNEL}')
    L.append(f'#define CNN23_PAD          {PAD}')
    L.append(f'#define CNN23_FC1_OUT      {FC1_OUT}')
    L.append(f'#define CNN23_FC2_OUT      {FC2_OUT}')
    L.append(f'#define CNN23_IN_SIZE      ({N_FEATURES} * {WINDOW})')
    L.append(f'#define CNN23_BEST_THRESHOLD  {thr:.4f}f')
    L.append(f'#define CNN23_TOTAL_PARAMS {total_params}')
    L.append(f'#define CNN23_WEIGHT_BYTES {n_bytes}')
    L.append("")
    L.append("#ifdef __cplusplus")
    L.append('extern "C" {')
    L.append("#endif")
    L.append("")
    for nm, arr, shape in [
        (f"{p}_conv1_w", w1, f"({HIDDEN}, {N_FEATURES}, {KERNEL})"),
        (f"{p}_conv1_b", b1, f"({HIDDEN},)"),
        (f"{p}_conv2_w", w2, f"({HIDDEN}, {HIDDEN}, {KERNEL})"),
        (f"{p}_conv2_b", b2, f"({HIDDEN},)"),
        (f"{p}_conv3_w", w3, f"({HIDDEN}, {HIDDEN}, {KERNEL})"),
        (f"{p}_conv3_b", b3, f"({HIDDEN},)"),
        (f"{p}_fc1_w", wfc1, f"({FC1_OUT}, {HIDDEN})"),
        (f"{p}_fc1_b", bfc1, f"({FC1_OUT},)"),
        (f"{p}_fc2_w", wfc2, f"({FC2_OUT}, {FC1_OUT})"),
        (f"{p}_fc2_b", bfc2, f"({FC2_OUT},)"),
    ]:
        L.append(f"/* {nm}  shape={shape}  numel={arr.size} */")
        L.append(gen_float_array(nm, arr))
        L.append("")
    L.append("#ifdef __cplusplus")
    L.append("}")
    L.append("#endif")
    L.append("#endif /* MODEL_CNN_23DIM_CH64_S%d_H */" % seed)

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"  写入 {out}")
    print(f"    权重 {n_bytes} B ({n_bytes/1024:.2f} KB), 阈值 {thr:.4f}")
    return out, thr, n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print(f"=== 23-dim CNN1D -> C header (seed={args.seed}) ===")
    print("=" * 70)
    out, thr, nbytes = export(args.seed)

    meta = json.load(open(os.path.join(BASE, "processed_meta_cnn1d_23dim_ch64.json"),
                          encoding="utf-8"))
    print(f"\n测试指标 (seed={args.seed}):")
    t = meta["best"]["test_at_thr"]
    t5 = meta["best"]["test_at_0.5"]
    print(f"  @thr={thr:.2f}:  P={t['precision']:.4f} R={t['recall']:.4f} "
          f"F1={t['binary_f1']:.4f} Acc={t['accuracy']:.4f}")
    print(f"  @0.50:  F1={t5['binary_f1']:.4f}  PR-AUC={t5['pr_auc']:.4f} ROC-AUC={t5['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
