#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mcu_pc_parity_audit.py  --  PC 端 .tflite 参考推理 (保存为 JSON 给 parity_diff.py)

⏵ 2026-10-15 TFLite Micro H743 部署配套脚本
   加载 model_v4_se_23dim_ch32_qat_s{seed}_int8.tflite, 用与 MCU 完全一样的输入
   (来自 IanArffDatasetF32.csv 的 200 个 16-帧滑动窗口), 跑 tflite.Interpreter
   得到每个窗口的 raw logit (int8 dequant), 保存为 pc_refs.json.

   MCU 端 (H743) 用完全相同的 .tflite + 完全相同的 int8 kernel 实现, 应当
   给出 bit-perfect 一致的结果. parity_diff.py 对比两者:
     - max_diff 期望 0.0 (浮点表示)
     - 实际可能 < 1e-6 (sigmoid/exp 在 AC6 vs numpy 的最后一位舍入差异)

依赖:
   pip install tensorflow  (或 pip install tflite-runtime --no-deps)
   pip install numpy pandas
"""
import argparse
import json
import os
import sys
import numpy as np

# 尝试多种 tflite 加载方式
try:
    import tensorflow as tf
    _INTERP_FN = lambda path: tf.lite.Interpreter(model_path=path)
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        _INTERP_FN = lambda path: tflite.Interpreter(model_path=path)
    except ImportError:
        sys.stderr.write(
            "[ERROR] 需要 tensorflow 或 tflite-runtime:\n"
            "    pip install tensorflow  (大, ~400MB)\n"
            "    pip install tflite-runtime  (小, ~10MB)\n")
        sys.exit(1)

try:
    import pandas as pd
except ImportError:
    sys.stderr.write("[ERROR] 需要 pandas: pip install pandas\n")
    sys.exit(1)

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# .tflite 推理 (跟 verify_tflite_pc.py 同样的 int8 量化公式)
# ---------------------------------------------------------------------------
def tflite_predict_one(interp, x_fp32: np.ndarray) -> float:
    """1 个 (23, 16) 窗口 → 1 个 FP32 概率.

    量化:  clip(round(x/s) + zp, -128, 127)
    反量化: (q - out_zp) * out_scale
    sigmoid: 1 / (1 + exp(-logit))
    """
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_s  = in_d['quantization_parameters']['scales'][0]
    in_zp = in_d['quantization_parameters']['zero_points'][0]
    out_s = out_d['quantization_parameters']['scales'][0]
    out_zp= out_d['quantization_parameters']['zero_points'][0]

    x = x_fp32.reshape(1, *x_fp32.shape).astype(np.float32)
    x_q = np.clip(np.round(x / in_s + in_zp), -128, 127).astype(np.int8)
    interp.set_tensor(in_d['index'], x_q)
    interp.invoke()
    y_q = interp.get_tensor(out_d['index'])[0]
    logit = (float(y_q) - out_zp) * out_s
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# 从 IanArffDatasetF32.csv 抽 N 个滑动窗口
# ---------------------------------------------------------------------------
def sample_windows_from_csv(csv_path: str, n_samples: int, seed: int = 0,
                            window: int = 16, in_channels: int = 23):
    """从 csv 取 N 个连续的 16-帧滑动窗口, 每个 (23, 16) layout.

    csv 每行 1 帧, 23 列 (KEEP_23 选择后). 跟 H743 sliding_window_get() 输出一致.
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < in_channels:
        raise ValueError(f"csv 只有 {df.shape[1]} 列, 期望 >= {in_channels}")
    arr = df.iloc[:, :in_channels].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    starts = rng.integers(low=0, high=len(arr) - window, size=n_samples)
    windows = np.stack([arr[s:s+window].T for s in starts], axis=0)  # (N, 23, 16)
    return windows, starts


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42,
                    help=f'模型 seed (默认 42; 可选 {SEEDS})')
    ap.add_argument('--csv',  default=os.path.join(BASE, 'IanArffDatasetF32.csv'),
                    help='f32 CSV 输入路径')
    ap.add_argument('--n_samples', type=int, default=200)
    ap.add_argument('--out',  default=os.path.join(BASE, 'pc_refs.json'))
    ap.add_argument('--all_seeds', action='store_true',
                    help='对 5 个 seed 全跑, 输出到 pc_refs_5seed.json')
    args = ap.parse_args()

    seeds = SEEDS if args.all_seeds else [args.seed]
    all_refs = {}

    for sd in seeds:
        tflite_path = os.path.join(BASE, f"model_v4_se_23dim_ch32_qat_s{sd}_int8.tflite")
        if not os.path.exists(tflite_path):
            print(f"[skip] {tflite_path} 不存在", file=sys.stderr)
            continue
        print(f"[run] seed={sd} 加载 {os.path.basename(tflite_path)}")
        interp = _INTERP_FN(tflite_path)
        interp.allocate_tensors()

        # 跟 verify_tflite_pc.py 用同一个 seed 抽样本 (保证 PC/MCU 同输入)
        X, starts = sample_windows_from_csv(args.csv, args.n_samples, seed=sd)
        probs = []
        for i in range(len(X)):
            p = tflite_predict_one(interp, X[i])
            probs.append(float(p))

        all_refs[sd] = {
            'tflite': os.path.basename(tflite_path),
            'n_samples': len(probs),
            'window_starts': [int(s) for s in starts],
            'probs': probs,
        }
        print(f"  → {len(probs)} 个窗口, prob range = [{min(probs):.6f}, {max(probs):.6f}]")

    out_path = args.out if not args.all_seeds else \
               os.path.join(os.path.dirname(args.out) or '.', 'pc_refs_5seed.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_refs, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] 写入 {out_path}")
    print(f"     seeds: {list(all_refs.keys())}")


if __name__ == "__main__":
    main()
