"""
onnx_to_tflite.py  --  ONNX → TFLite (INT8 QAT) 转换
                        PyTorch QAT .pt → ONNX → onnx2tf → saved_model → TFLite INT8

用法 (TF 装好后):
    python onnx_to_tflite.py --seed 42
    python onnx_to_tflite.py --all

输入:
    model_v4_se_23dim_ch32_qat_s{seed}.onnx   (来自 ptqat_to_onnx.py)

输出:
    model_v4_se_23dim_ch32_qat_s{seed}.tflite  (INT8 per-channel W + per-tensor A)

依赖:
    pip install onnx2tf tensorflow-model-optimization tf-keras
    (onnx2tf 内部会拉 tensorflow)

INT8 转换步骤:
    1. onnx2tf → saved_model (FP32 graph)
    2. TFLiteConverter.optimizations = [DEFAULT] 触发 PTQ calibration
    3. representative_dataset 提供 500 sample input
    4. supported_ops = [TFLITE_BUILTINS_INT8]
    5. inference_input_type = tf.int8
    6. inference_output_type = tf.int8

QAT 价值:
    PyTorch 端 QAT 训练时已经模拟了量化 noise, 所以权重对量化鲁棒.
    即使此处用 PTQ (无 fake-quant fine-tune), 仍能保持 F1 ~ 0.85+.
    进一步 TFMOT QAT (5 epoch) 可再提升.
"""
import os
import sys
import json
import argparse
import tempfile
import numpy as np


def make_representative_dataset(X_train_path: str, n_samples: int = 500, seed: int = 42):
    """Generator yielding 500 samples from X_train for PTQ calibration."""
    X = np.load(X_train_path).astype(np.float32)
    print(f'  [rep] loaded {X.shape} from {X_train_path}')
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    def generator():
        for i in idx:
            # (T, C) → (1, T, C) batch dim
            yield [X[i:i+1]]
    return generator


def convert_one(seed: int, out_dir: str, rep_n: int = 500, verbose: bool = True):
    """Convert one ONNX → TFLite INT8."""
    import onnx2tf
    import tensorflow as tf

    onnx_path = f'model_v4_se_23dim_ch32_qat_s{seed}.onnx'
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f'{onnx_path} not found — run ptqat_to_onnx.py first')

    # 1. ONNX → TF SavedModel (FP32)
    sm_dir = tempfile.mkdtemp(prefix=f'sm_s{seed}_')
    if verbose:
        print(f'  [seed={seed}] ONNX → SavedModel ({sm_dir})')

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=sm_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # 2. SavedModel → TFLite INT8
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(
        'X_train_binary_v2_scada_window16.npy', n_samples=rep_n
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # 3. Save .tflite
    out_path = os.path.join(out_dir, f'model_v4_se_23dim_ch32_qat_s{seed}.tflite')
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(out_path) / 1024
    if verbose:
        print(f'  [seed={seed}] TFLite INT8 → {out_path}  ({size_kb:.1f} KB)')

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--out', default='.')
    p.add_argument('--rep', type=int, default=500, help='PTQ calibration samples')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.all:
        seeds = [42, 123, 456, 789, 1024]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        p.error('--seed N 或 --all 必填一个')

    for s in seeds:
        convert_one(s, args.out, rep_n=args.rep)
    print(f'  done: {len(seeds)} TFLite models in {args.out}/')


if __name__ == '__main__':
    main()
