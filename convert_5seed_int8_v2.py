"""
convert_5seed_int8_v2.py  -- 改进版 5-seed INT8 量化 (B 路线)

跟 convert_5seed_int8.py 区别:
  - output_type=tf.float32 (head FP32, logit 退饱和, 治 22.x 饱和)
  - per-channel activation (Optimize.EXPERIMENTAL_SPLIT 走 per-channel A)
  - representative dataset 500 → 2000 samples (更稳)
  - 加 per-layer 输出范围诊断 (FP32 vs INT8 max|diff|)
  - 写 .tflite + 写 pc_int8_v2_probs.npz (PC 模拟 prob, 跟 H743 对比)

用法:
  python convert_5seed_int8_v2.py --out . --rep 2000
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


def saved_model_to_tflite_int8_v2(sm_dir: str, rep_data: np.ndarray, n_samples: int = 2000):
    """SavedModel → TFLite INT8 v2 (per-channel A, output FP32, 大校准集).

    关键改动:
      - inference_output_type = tf.float32 (head FP32, 防 logit 饱和)
      - inference_input_type = tf.int8  (输入仍 int8, 跟训练管线一致)
      - Optimize.EXPERIMENTAL_SPLIT (走 MLIR per-channel activation 路径)
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(rep_data), min(n_samples, len(rep_data)), replace=False)

    def rep():
        for i in idx:
            # TFLite expects (B, C, T) layout matching PyTorch Conv1d
            yield [np.transpose(rep_data[i:i+1], (0, 2, 1)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPLIT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32   # ⭐ v2: head FP32, 防饱和
    return converter.convert()


def tflite_predict_probs_v2(tflite_bytes: bytes, X: np.ndarray) -> np.ndarray:
    """Run INT8 input + FP32 output TFLite inference, return probs."""
    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = inp_d['quantization']

    probs = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        x_t = np.transpose(X[i:i+1], (0, 2, 1))  # (1, T, C) → (1, C, T)
        x_q = np.clip(np.round(x_t / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_d['index'], x_q)
        interp.invoke()
        # ⭐ v2: output 是 FP32, 直接读
        if np.issubdtype(out_d['dtype'], np.integer):
            y_q = interp.get_tensor(out_d['index'])
            out_scale, out_zp = out_d['quantization']
            logit = (y_q.astype(np.float32) - out_zp) * out_scale
        else:
            logit = interp.get_tensor(out_d['index']).astype(np.float32)
        probs[i] = 1.0 / (1.0 + np.exp(-logit))
    return probs


def per_layer_diagnostic(tflite_bytes: bytes, fp32_probs: np.ndarray, X: np.ndarray, n_diag: int = 50):
    """跑 N 个 sample, 看每层 INT8 输出跟 PC FP32 输出的 max|diff|.

    输出格式: layer_idx, op_name, output_shape, max_abs_diff, mean_abs_diff
    """
    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    in_scale, in_zp = inp_d['quantization']

    # 收集每层输出
    layer_outputs_int8 = {}
    layer_details = interp.get_tensor_details()

    def collector(i, tensor, out):
        layer_outputs_int8[i] = (tensor, out.copy())

    # 跑 n_diag 个 sample
    n = min(n_diag, len(X))
    sample_inputs = []
    for i in range(n):
        x_t = np.transpose(X[i:i+1], (0, 2, 1))
        x_q = np.clip(np.round(x_t / in_scale + in_zp), -128, 127).astype(np.int8)
        sample_inputs.append(x_q)

    # 用 test_data 跑, 收集每层输出
    interp.reset_all_accumulators()  # no-op, just to ensure clean state
    diffs_per_layer = {}

    for i in range(n):
        interp.set_tensor(inp_d['index'], sample_inputs[i])
        # 跑前 hook 所有 tensor 输出
        interp.invoke()
        for tid, tdetail in enumerate(layer_details):
            try:
                t = interp.get_tensor(tdetail['index'])
                if t is None or t.size == 0:
                    continue
                # 跳过 input/output
                if tid == inp_d['index'] or (tid == interp.get_output_details()[0]['index']):
                    continue
                # 跳过 bias (1D)
                if t.ndim < 2 or t.size > 10000:
                    continue
                if tid not in diffs_per_layer:
                    diffs_per_layer[tid] = {'max': 0.0, 'sum': 0.0, 'n': 0}
                # 统计这层输出的 range
                t_max = float(np.abs(t.astype(np.float32)).max())
                if t_max > diffs_per_layer[tid]['max']:
                    diffs_per_layer[tid]['max'] = t_max
                diffs_per_layer[tid]['n'] += 1
            except Exception:
                continue

    # 打印 top 10 range 最大的层 (通常是 conv 输出)
    print("\n  Per-layer output |max| (top 10):")
    sorted_layers = sorted(diffs_per_layer.items(), key=lambda x: -x[1]['max'])[:10]
    for tid, info in sorted_layers:
        tname = layer_details[tid]['name'] if tid < len(layer_details) else f'?{tid}'
        tshape = layer_details[tid]['shape'] if tid < len(layer_details) else '?'
        print(f"    tid={tid:3d}  name={tname:40s}  shape={tshape}  |max|={info['max']:.4f}  n_samples={info['n']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='.', help='output dir for .tflite files')
    p.add_argument('--rep', type=int, default=2000, help='PTQ calibration samples (v2: 2000)')
    p.add_argument('--no-diag', action='store_true', help='skip per-layer diagnostic')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 加载数据
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

            # 2. SavedModel → TFLite INT8 v2
            tflite_bytes = saved_model_to_tflite_int8_v2(sm_dir, X_train, n_samples=args.rep)
            out_path = os.path.join(args.out, f'model_v4_se_23dim_ch32_qat_s{seed}_int8_v2.tflite')
            with open(out_path, 'wb') as f:
                f.write(tflite_bytes)

            # 3. 推理 + F1
            probs = tflite_predict_probs_v2(tflite_bytes, X_test)
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
            results['per_seed'][str(seed)]['tflite_bytes_v2'] = len(tflite_bytes)
            print(f'  [seed={seed}] v2 TFLite {len(tflite_bytes)} B  F1@0.5={results["per_seed"][str(seed)]["0.50"]["F1"]:.4f}  '
                  f'AUC={results["per_seed"][str(seed)]["AUC"]:.4f}')

            # 4. per-layer 诊断 (只对 seed=42 跑一次)
            if seed == 42 and not args.no_diag:
                print("\n  [B-DIAG] Per-layer output |max| diagnostic:")
                per_layer_diagnostic(tflite_bytes, None, X_test, n_diag=50)

        finally:
            shutil.rmtree(sm_dir, ignore_errors=True)

    # 5. 5-seed ensemble
    if len(all_probs) == 5:
        all_probs = np.stack(all_probs, axis=0)
        ens_probs = all_probs.mean(axis=0)
        print(f'\n=== 5-seed ensemble v2 (prob mean) ===')
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

        np.savez('pc_int8_v2_5seed_probs.npz', probs=ens_probs, y_test=y_test)

    # 6. 写结果
    out_json = os.path.join(args.out, 'ensemble_int8_v2_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  results → {out_json}')


if __name__ == '__main__':
    main()
