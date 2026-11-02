"""
convert_5seed_fp32.py  --  批量 5-seed QAT ONNX → FP32 TFLite + ensemble F1

⭐ 2026-10-17 新增: 抛弃 INT8, 直接转 FP32 .tflite
   原因: TFLM runtime INT8 算子 (CMSIS-NN / ref) 跟 PC TFLite XNNPACK delegate
         数值不一致, 5-seed INT8 ensemble F1=0.8646 (PC) → 0.07 (H743) FPR=92.86%.
   解法: 模型全 FP32, input/output 都是 fp32, TFLM ref 算子跟 PC TFLite fp32
         数值 bit-perfect 一致. F1 跟训练 baseline 0.8646 一致.

输入:  model_v4_se_23dim_ch32_qat_s{seed}.onnx  (5 个 PyTorch QAT ONNX)
输出:  model_v4_se_23dim_ch32_qat_s{seed}_fp32.tflite  (FP32, ~120 KB each)
       ensemble_fp32_5seed_results.json  (5-seed 集成 F1 + 阈值扫描)
       model_data_fp32_s{seed}.h  (C 数组, 跟 INT8 同结构)

关键参数:
  - 不量化! converter.optimizations = []
  - input/output fp32
  - 输入 layout (B, C, T) — 跟 PyTorch Conv1d 一致
  - 5-seed ensemble 概率均值
"""
import os
import sys
import json
import argparse
import tempfile
import shutil
import numpy as np
import onnx2tf
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEEDS = [42, 123, 456, 789, 1024]


def onnx_to_saved_model(onnx_path: str, out_dir: str):
    """onnx2tf → TF SavedModel (FP32, 不量化)."""
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )


def saved_model_to_tflite_fp32(sm_dir: str):
    """SavedModel → TFLite FP32 (不量化, input/output 都是 fp32)."""
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.optimizations = []   # ⭐ 不量化!
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    # 不指定 supported_ops, 默认 TFLITE_BUILTINS (含 FP32)
    return converter.convert()


def tflite_predict_probs_fp32(tflite_bytes: bytes, X: np.ndarray) -> np.ndarray:
    """Run FP32 TFLite inference, return probs (sigmoid(logit))."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    probs = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        # TFLite expects (B, C, T) layout matching PyTorch Conv1d
        x_t = np.transpose(X[i:i+1], (0, 2, 1)).astype(np.float32)  # (1, T, C) → (1, C, T)
        interp.set_tensor(inp_d['index'], x_t)
        interp.invoke()
        logit = float(interp.get_tensor(out_d['index'])[0])
        probs[i] = 1.0 / (1.0 + np.exp(-logit))
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='.', help='output dir for .tflite files')
    p.add_argument('--header-out', default='MCU/Core/Inc', help='output dir for .h C arrays')
    p.add_argument('--seeds', default='42,123,456,789,1024', help='comma-separated seeds')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.header_out, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(',')]

    # 加载数据 (跟训练一致: clip ±10)
    X_train = np.load('X_train_binary_v2_scada_window16.npy').astype(np.float32)[:, :, KEEP_23]
    X_train = np.clip(X_train, -10, 10)
    X_test = np.load('X_test_binary_v2_scada_window16.npy').astype(np.float32)[:, :, KEEP_23]
    X_test = np.clip(X_test, -10, 10)
    y_test = np.load('y_test_binary_v2_scada_window16.npy').astype(np.int64)
    print(f'X_train: {X_train.shape}  X_test: {X_test.shape}  y_test: {y_test.shape}')

    results = {'per_seed': {}, 'ensemble': {}}
    all_probs = []

    for seed in seeds:
        onnx_path = f'model_v4_se_23dim_ch32_qat_s{seed}.onnx'
        if not os.path.exists(onnx_path):
            print(f'  [seed={seed}] SKIP — {onnx_path} not found')
            continue

        # 1. ONNX → SavedModel (FP32, 不量化)
        sm_dir = tempfile.mkdtemp(prefix=f'sm_fp32_s{seed}_')
        try:
            onnx_to_saved_model(onnx_path, sm_dir)

            # 2. SavedModel → TFLite FP32
            tflite_bytes = saved_model_to_tflite_fp32(sm_dir)
            out_path = os.path.join(args.out, f'model_v4_se_23dim_ch32_qat_s{seed}_fp32.tflite')
            with open(out_path, 'wb') as f:
                f.write(tflite_bytes)

            # 3. 推理 + F1
            probs = tflite_predict_probs_fp32(tflite_bytes, X_test)
            all_probs.append(probs)

            for thr in [0.40, 0.45, 0.49, 0.50, 0.55, 0.60]:
                y_pred = (probs >= thr).astype(int)
                results['per_seed'].setdefault(str(seed), {})[f'{thr:.2f}'] = {
                    'F1': float(f1_score(y_test, y_pred)),
                    'Acc': float(accuracy_score(y_test, y_pred)),
                    'P': float(precision_score(y_test, y_pred)),
                    'R': float(recall_score(y_test, y_pred)),
                }
            results['per_seed'][str(seed)]['AUC'] = float(roc_auc_score(y_test, probs))
            results['per_seed'][str(seed)]['tflite_bytes'] = len(tflite_bytes)
            print(f'  [seed={seed}] TFLite FP32 {len(tflite_bytes)/1024:.1f} KB  '
                  f'F1@0.5={results["per_seed"][str(seed)]["0.50"]["F1"]:.4f}  '
                  f'AUC={results["per_seed"][str(seed)]["AUC"]:.4f}')

        finally:
            shutil.rmtree(sm_dir, ignore_errors=True)

    # 4. 5-seed ensemble
    if len(all_probs) == len(seeds):
        all_probs = np.stack(all_probs, axis=0)
        ens_probs = all_probs.mean(axis=0)
        print(f'\n=== 5-seed ensemble (FP32, prob mean) ===')
        for thr in [0.40, 0.45, 0.49, 0.50, 0.55, 0.60]:
            y_pred = (ens_probs >= thr).astype(int)
            results['ensemble'][f'{thr:.2f}'] = {
                'F1': float(f1_score(y_test, y_pred)),
                'Acc': float(accuracy_score(y_test, y_pred)),
                'P': float(precision_score(y_test, y_pred)),
                'R': float(recall_score(y_test, y_pred)),
            }
            print(f'  thr={thr:.2f}  F1={results["ensemble"][f"{thr:.2f}"]["F1"]:.4f}  '
                  f'P={results["ensemble"][f"{thr:.2f}"]["P"]:.4f}  '
                  f'R={results["ensemble"][f"{thr:.2f}"]["R"]:.4f}')
        results['ensemble']['AUC'] = float(roc_auc_score(y_test, ens_probs))
        print(f'  AUC: {results["ensemble"]["AUC"]:.4f}')

        np.savez('ensemble_fp32_5seed_probs.npz', probs=ens_probs, y_test=y_test)
    else:
        print(f'\n[warn] only {len(all_probs)}/{len(seeds)} seeds converted, skip ensemble')

    # 5. 写结果
    out_json = os.path.join(args.out, 'ensemble_fp32_5seed_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  results → {out_json}')

    # 6. 生成 C 头文件
    print(f'\n=== Generate C headers ===')
    import subprocess
    for seed in seeds:
        tflite_path = os.path.join(args.out, f'model_v4_se_23dim_ch32_qat_s{seed}_fp32.tflite')
        if not os.path.exists(tflite_path):
            continue
        header_path = os.path.join(args.header_out, f'model_data_fp32_s{seed}.h')
        cmd = ['python', 'MCU/scripts/tflite_to_c_array.py',
               '--tflite', tflite_path, '--out', header_path,
               '--name', f'v4_se_23dim_ch32_qat_s{seed}_fp32']
        subprocess.run(cmd, check=True)
    print(f'\n[done] all fp32 .tflite + .h generated')


if __name__ == '__main__':
    main()
