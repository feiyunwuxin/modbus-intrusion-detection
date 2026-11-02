"""
convert_5seed_int8.py  --  批量 5-seed QAT ONNX → INT8 TFLite + ensemble F1

输入:  model_v4_se_23dim_ch32_qat_s{seed}.onnx  (5 个 .pt QAT 模型的 ONNX 导出)
输出:  model_v4_se_23dim_ch32_qat_s{seed}_int8.tflite  (INT8, ~49 KB each)
       ensemble_int8_results.json  (5-seed 集成 F1 + 阈值扫描)

PTQ 关键参数:
  - input/output int8, per-channel sym weight, per-tensor sym activation
  - 500 samples representative dataset (X_train 二值化后 clip ±10)
  - 输入 layout (B, C, T) — 跟 PyTorch Conv1d 一致
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
from ai_edge_litert.interpreter import Interpreter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEEDS = [42, 123, 456, 789, 1024]


def onnx_to_saved_model(onnx_path: str, out_dir: str):
    """onnx2tf → TF SavedModel (FP32)."""
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )


def saved_model_to_tflite_int8(sm_dir: str, rep_data: np.ndarray, n_samples: int = 500):
    """SavedModel → TFLite INT8 (per-channel W, per-tensor A, int8 IO)."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(rep_data), min(n_samples, len(rep_data)), replace=False)

    def rep():
        for i in idx:
            # TFLite expects (B, C, T) layout matching PyTorch Conv1d
            yield [np.transpose(rep_data[i:i+1], (0, 2, 1)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def tflite_predict_probs(tflite_bytes: bytes, X: np.ndarray) -> np.ndarray:
    """Run INT8 TFLite inference, return probs."""
    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']

    probs = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        x_t = np.transpose(X[i:i+1], (0, 2, 1))  # (1, T, C) → (1, C, T)
        x_q = np.clip(np.round(x_t / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_d['index'], x_q)
        interp.invoke()
        y_q = interp.get_tensor(out_d['index'])
        logit = (float(y_q[0]) - out_zp) * out_scale
        probs[i] = 1.0 / (1.0 + np.exp(-logit))
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='.', help='output dir for .tflite files')
    p.add_argument('--rep', type=int, default=500, help='PTQ calibration samples')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 加载数据 (clip ±10 跟训练一致)
    X_train = np.load('X_train_binary_v2_scada_window16.npy').astype(np.float32)[:, :, KEEP_23]
    X_train = np.clip(X_train, -10, 10)
    X_test = np.load('X_test_binary_v2_scada_window16.npy').astype(np.float32)[:, :, KEEP_23]
    X_test = np.clip(X_test, -10, 10)
    y_test = np.load('y_test_binary_v2_scada_window16.npy').astype(np.int64)
    print(f'X_train: {X_train.shape}  X_test: {X_test.shape}  y_test: {y_test.shape}')

    results = {'per_seed': {}, 'ensemble': {}}
    all_probs = []

    for seed in SEEDS:
        onnx_path = f'model_v4_se_23dim_ch32_qat_s{seed}.onnx'
        if not os.path.exists(onnx_path):
            print(f'  [seed={seed}] SKIP — {onnx_path} not found')
            continue

        # 1. ONNX → SavedModel
        sm_dir = tempfile.mkdtemp(prefix=f'sm_s{seed}_')
        try:
            onnx_to_saved_model(onnx_path, sm_dir)

            # 2. SavedModel → TFLite INT8
            tflite_bytes = saved_model_to_tflite_int8(sm_dir, X_train, n_samples=args.rep)
            out_path = os.path.join(args.out, f'model_v4_se_23dim_ch32_qat_s{seed}_int8.tflite')
            with open(out_path, 'wb') as f:
                f.write(tflite_bytes)

            # 3. 推理 + F1
            probs = tflite_predict_probs(tflite_bytes, X_test)
            all_probs.append(probs)

            for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                y_pred = (probs >= thr).astype(int)
                results['per_seed'].setdefault(str(seed), {})[f'{thr:.2f}'] = {
                    'F1': float(f1_score(y_test, y_pred)),
                    'Acc': float(accuracy_score(y_test, y_pred)),
                    'P': float(precision_score(y_test, y_pred)),
                    'R': float(recall_score(y_test, y_pred)),
                }
            results['per_seed'][str(seed)]['AUC'] = float(roc_auc_score(y_test, probs))
            results['per_seed'][str(seed)]['tflite_bytes'] = len(tflite_bytes)
            print(f'  [seed={seed}] TFLite {len(tflite_bytes)} B  F1@0.5={results["per_seed"][str(seed)]["0.50"]["F1"]:.4f}  AUC={results["per_seed"][str(seed)]["AUC"]:.4f}')

        finally:
            shutil.rmtree(sm_dir, ignore_errors=True)

    # 4. 5-seed ensemble
    if len(all_probs) == 5:
        all_probs = np.stack(all_probs, axis=0)  # (5, N)
        ens_probs = all_probs.mean(axis=0)
        print(f'\n=== 5-seed ensemble (prob mean) ===')
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

        np.savez('ensemble_int8_5seed_probs.npz', probs=ens_probs, y_test=y_test)

    # 5. 写结果
    out_json = os.path.join(args.out, 'ensemble_int8_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  results → {out_json}')


if __name__ == '__main__':
    main()
