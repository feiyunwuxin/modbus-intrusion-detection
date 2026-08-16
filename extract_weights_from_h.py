#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_weights_from_h.py — 从 model_ch32_hybrid.h 反向提取权重

解析 .h 文件中的 static const 数组,把 INT8 权重 + FP32 scale + FP32 bias
保存为 .npz 格式,供 validate_ch32_on_pc.py 加载使用.

这避免了重新训练 ch=32 模型 (实测要 ~1 分钟).
"""

import re
import os
import sys
import json
import numpy as np

# Windows console UTF-8 fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"C:\work\Claude\Issue"
H_FILE = os.path.join(BASE, "model_ch32_hybrid.h")
META_FILE = os.path.join(BASE, "model_ch32_hybrid_meta.json")
OUT_FILE = os.path.join(BASE, "ch32_weights_extracted.npz")


def parse_h_file(path):
    """解析 .h 文件,提取所有 static const 数组"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 匹配模式: static const {dtype} {name}[{size}] = { {values}, ... };
    # 支持 int8_t 和 float 两种 dtype
    pattern = re.compile(
        r'static\s+const\s+(int8_t|float)\s+(\w+)\[(\d+)\]\s*=\s*\{([^}]+)\};',
        re.DOTALL
    )

    arrays = {}
    for m in pattern.finditer(content):
        dtype = m.group(1)
        name = m.group(2)
        size = int(m.group(3))
        body = m.group(4)

        # 解析数组元素
        # 去掉注释 (/* ... */)
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
        # 去掉逗号
        body = body.replace(',', ' ')
        # 提取所有数字 (支持科学计数法 e-03)
        if dtype == 'int8_t':
            nums = re.findall(r'-?\d+', body)
        else:
            nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', body)
        if len(nums) != size:
            print(f"  [warn] {name}: expected {size} elements, got {len(nums)}")

        if dtype == 'int8_t':
            arr = np.array([int(n) for n in nums[:size]], dtype=np.int8)
        else:
            arr = np.array([float(n) for n in nums[:size]], dtype=np.float32)
        arrays[name] = arr

    return arrays


def main():
    print("=" * 70)
    print(" 从 model_ch32_hybrid.h 提取权重")
    print("=" * 70)

    # 1. 解析 .h
    print(f"\n[1] 解析 {H_FILE} ...")
    arrays = parse_h_file(H_FILE)
    print(f"    提取了 {len(arrays)} 个数组:")
    for k in sorted(arrays.keys()):
        print(f"      {k}: shape={arrays[k].shape}, dtype={arrays[k].dtype}")

    # 2. 加载 meta (确认 tensor 列表)
    print(f"\n[2] 加载 {META_FILE} ...")
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print(f"    {len(meta['tensors'])} 个 tensor 定义")

    # 3. 映射 .h symbol → tensor name
    # .h 文件用 ch32_w_tcn_0_conv1_weight 表示 tcn.0.conv1.weight
    print("\n[3] 映射 symbol → tensor name ...")
    mapping = {}
    for t in meta['tensors']:
        name = t['name']   # e.g. "tcn.0.conv1.weight"
        # .h 里的 symbol: weight 加 _w_ 前缀,bias 加 _b_ 前缀
        if 'weight' in name:
            sym = 'ch32_w_' + name.replace('.', '_')
        else:
            sym = 'ch32_b_' + name.replace('.', '_')
        if sym in arrays:
            mapping[name] = sym
        else:
            print(f"    [warn] {sym} not found")

    print(f"    成功映射 {len(mapping)}/{len(meta['tensors'])} tensors")

    # 4. 重建 FP32 权重 (dequant INT8 → FP32)
    # 对 weight: out[name] = q[name] * scale[name].reshape(-1, 1, 1)  (conv) 或 1, 1 (fc)
    # 对 bias: 直接用 FP32
    print("\n[4] 反量化 INT8 → FP32 ...")
    weights_fp32 = {}
    weights_int8_raw = {}  # 原始 INT8 权重 (供 numpy_forward dequant 使用)
    scales = {}
    biases = {}
    for t in meta['tensors']:
        name = t['name']
        if name not in mapping:
            continue
        sym = mapping[name]
        arr = arrays[sym]
        if 'weight' in name:
            q = arr
            weights_int8_raw[name] = q.reshape(t['shape']).astype(np.int8)  # 保留原始 INT8
            # scale 数组: 纯 symmetric INT8,只有 scale,没有 zero_point
            scale_sym = 'ch32_s_' + name.replace('.', '_')
            if scale_sym not in arrays:
                print(f"    [warn] scale {scale_sym} not found")
                continue
            s = arrays[scale_sym]  # (out_ch,) 都是正值

            # dequant: fp32 = int8 * scale (对称量化,无 zero_point)
            if len(t['shape']) == 3:  # (out_ch, in_ch, k)
                w_fp32 = q.reshape(t['shape']).astype(np.float32) * s.reshape(-1, 1, 1)
            elif len(t['shape']) == 2:  # (out, in)
                w_fp32 = q.reshape(t['shape']).astype(np.float32) * s.reshape(-1, 1)
            else:
                continue

            weights_fp32[name] = w_fp32
            scales[name] = s.astype(np.float32)
        elif 'bias' in name:
            biases[name] = arr.astype(np.float32)

    print(f"    重建了 {len(weights_fp32)} 个 FP32 权重, {len(biases)} 个 bias")

    # 5. sanity check: 权重分布
    print("\n[5] 权重分布 (前 5 个 conv):")
    for k in list(weights_fp32.keys())[:5]:
        w = weights_fp32[k]
        print(f"    {k}: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}, "
              f"mean={w.mean():.4f}, std={w.std():.4f}")

    # 6. 保存为 .npz
    print(f"\n[6] 保存到 {OUT_FILE} ...")
    save_dict = {}
    for k, v in weights_fp32.items():
        save_dict[f'w_{k.replace(".", "_")}'] = v
    for k, v in biases.items():
        save_dict[f'b_{k.replace(".", "_")}'] = v
    for k, v in weights_int8_raw.items():
        save_dict[f'q_{k.replace(".", "_")}'] = v  # INT8 raw (供 numpy_forward dequant 使用)
    np.savez(OUT_FILE, **save_dict)
    print(f"    [OK] 已保存 {len(save_dict)} 个数组 (FP32={len(weights_fp32)}, bias={len(biases)}, INT8={len(weights_int8_raw)})")

    print("\n" + "=" * 70)
    print(" 提取完成. 现在可以跑 validate_ch32_on_pc.py 做对比验证")
    print("=" * 70)


if __name__ == "__main__":
    main()