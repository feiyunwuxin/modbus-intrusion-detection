"""
ptqat_to_onnx.py  --  PyTorch QAT .pt → ONNX 导出
                       (下一步: onnx2tf → saved_model → .tflite)

用法 (PC):
    python ptqat_to_onnx.py --seed 42
    python ptqat_to_onnx.py --all

输入:
    model_v4_se_23dim_b64_ch32_do01_window16_qat_s{seed}.pt

输出:
    model_v4_se_23dim_ch32_qat_s{seed}.onnx  (FP32 graph, fake-quant 已 baked)

依赖:
    仅 torch + onnx (不需要 TF / TFMOT)
    推理验证: numpy
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import onnx

# === import TCN+SE 架构 (跟训练同源) ===
sys.path.insert(0, os.path.dirname(__file__))
from tcn_se_keras_model_bridge import TCNClassifierSE_Torch  # 桥接文件见下


def load_qat_state(pt_path: str):
    """Load .pt QAT checkpoint, return (state_dict, config)."""
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    return ckpt


def export_one(seed: int, out_dir: str, verbose: bool = True) -> str:
    """Export one PyTorch QAT .pt to ONNX."""
    pt_path = f'model_v4_se_23dim_b64_ch32_do01_window16_qat_s{seed}.pt'
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f'{pt_path} not found')

    ckpt = load_qat_state(pt_path)
    # 配置在 top-level (n_features/window/n_blocks/channels/kernel_size/dilations/se_reduction)
    sd = ckpt['state_dict']

    # 1. 重建模型 (QAT 模式=FakeQuant inserted, 但我们只 export 计算图, 不跑 fake-quant)
    model = TCNClassifierSE_Torch(
        in_features=ckpt['n_features'],         # 23
        window=ckpt['window'],                  # 16
        n_blocks=ckpt['n_blocks'],              # 3
        channels=ckpt['channels'],              # 32
        kernel_size=ckpt['kernel_size'],        # 3
        dilations=ckpt['dilations'],            # [1,2,4]
        se_reduction=ckpt['se_reduction'],      # 8
        dropout=0.0,  # export 模式关闭 dropout
    )
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 2. dummy input (B=1, T=16, C=23)
    dummy = torch.randn(1, ckpt['window'], ckpt['n_features'])

    # 3. ONNX export
    onnx_path = os.path.join(out_dir, f'model_v4_se_23dim_ch32_qat_s{seed}.onnx')
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['logit'],
        dynamic_axes={'input': {0: 'batch'}, 'logit': {0: 'batch'}},
        opset_version=13,
        do_constant_folding=True,
    )

    # 4. onnx checker
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    if verbose:
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f'  [seed={seed}] ONNX OK  {onnx_path}  ({size_kb:.1f} KB)')

    return onnx_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=None, help='single seed export')
    p.add_argument('--all',  action='store_true', help='export all 5 seeds (42,123,456,789,1024)')
    p.add_argument('--out',  default='.', help='output directory')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.all:
        seeds = [42, 123, 456, 789, 1024]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        p.error('--seed N 或 --all 必填一个')

    for s in seeds:
        export_one(s, args.out)
    print(f'  done: {len(seeds)} seeds exported to {args.out}/')


if __name__ == '__main__':
    main()
