"""
convert_5seed_fp32_v2.py  --  自研 ONNX -> TFLite FP32 转换器 (绕过 onnx2tf + ai_edge_litert)

⭐ 2026-10-18: 因 onnx2tf 2.6.9 跟现有包不兼容, ai_edge_litert 安装慢, 改用 onnx 直接
   解析 + tf.keras 自建模型. 模型结构固定 (3-block TCN+SE), 写死 op 映射.

输入:  model_v4_se_23dim_ch32_qat_s{seed}.onnx  (5 个 PyTorch 导出 ONNX, BN 已 fold)
输出:  model_v4_se_23dim_ch32_qat_s{seed}_fp32.tflite  (FP32)
       + model_data_fp32_s{seed}.h  (C 数组)
       + ensemble_fp32_5seed_results.json

ONNX op 映射 (TFLite 全部支持 fp32):
  Transpose(0,1,2 -> 1,0,2)  |  调整 (T,C) -> (C,T) (PyTorch Conv1d 输入)
  Conv1d (kernel=[1,3], stride=[1,1], valid pad)
    - TFLite: tf.keras.layers.Conv1D  (fused BN)
  Relu
  Unsqueeze/Squeeze/ReduceMean  |  TFLite 算子都支持
  Gemm  (FC)                    |  tf.keras.layers.Dense
  Sigmoid, Mul, Add             |  TFLite 算子都支持
"""
import os
import sys
import json
import argparse
import numpy as np
import onnx
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEEDS = [42, 123, 456, 789, 1024]


def get_init_np(m, name):
    """从 ONNX initializer 取 numpy array (FP32)."""
    for init in m.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init).astype(np.float32)
    raise KeyError(f"initializer {name} not found")


def build_tcn_se_keras(num_blocks=3, in_channels=23, channels=32, se_reduction=8, window=16):
    """Build a tf.keras functional TCN+SE model matching the PyTorch arch.

    Input shape: (None, window, in_channels)  -- Keras Conv1D 期望 (B, T, C)

    PyTorch 训练时用 Conv1d(k=3, padding=d, dilation=d) 保持 T=16 不变:
      - block 0: d=1, padding=1
      - block 1: d=2, padding=2
      - block 2: d=4, padding=4
    """
    inputs = tf.keras.Input(shape=(window, in_channels), name='input')

    dilations = [1, 2, 4]  # per block

    x = inputs
    for b in range(num_blocks):
        d = dilations[b]
        # Conv1D kernel 3, dilation d, 'same' padding (= pad = d for k=3)
        h = tf.keras.layers.Conv1D(channels, 3, padding='same', dilation_rate=d,
                                     name=f'block{b}_conv1', use_bias=True)(x)
        h = tf.keras.layers.ReLU(name=f'block{b}_relu1')(h)
        h = tf.keras.layers.Conv1D(channels, 3, padding='same', dilation_rate=d,
                                     name=f'block{b}_conv2', use_bias=True)(h)
        h = tf.keras.layers.ReLU(name=f'block{b}_relu2')(h)

        # SE
        se = tf.keras.layers.GlobalAveragePooling1D(name=f'block{b}_se_gap')(h)
        se = tf.keras.layers.Dense(channels // se_reduction, name=f'block{b}_se_fc1',
                                    use_bias=True)(se)
        se = tf.keras.layers.ReLU(name=f'block{b}_se_relu')(se)
        se = tf.keras.layers.Dense(channels, name=f'block{b}_se_fc2',
                                    use_bias=True)(se)
        se = tf.keras.layers.Activation('sigmoid', name=f'block{b}_se_sig')(se)
        se = tf.keras.layers.Reshape((1, channels), name=f'block{b}_se_reshape')(se)
        h = tf.keras.layers.Multiply(name=f'block{b}_se_mul')([h, se])

        # residual: first block 1x1 conv 23->32; later blocks identity
        if b == 0:
            res = tf.keras.layers.Conv1D(channels, 1, padding='same', name=f'block{b}_res',
                                           use_bias=True)(x)
        else:
            res = x

        x = tf.keras.layers.Add(name=f'block{b}_add')([h, res])
        x = tf.keras.layers.ReLU(name=f'block{b}_relu3')(x)

    # head
    x = tf.keras.layers.GlobalAveragePooling1D(name='head_gap')(x)
    x = tf.keras.layers.Dense(channels, name='fc1', use_bias=True)(x)
    logits = tf.keras.layers.Dense(1, name='fc2', use_bias=True)(x)

    return tf.keras.Model(inputs=inputs, outputs=logits, name='tcn_se')


def load_onnx_weights_to_keras(keras_model, onnx_path, num_blocks=3, channels=32):
    """从 ONNX initializers 读 weight, 写到 keras model.

    ONNX Conv1d weight layout: (out_C, in_C, k)   ← PyTorch style
    Keras Conv1D weight layout: (k, in_C, out_C)
    → 需要转置 (0, 2, 1)
    """
    m = onnx.load(onnx_path)
    init_map = {init.name: onnx.numpy_helper.to_array(init).astype(np.float32)
                for init in m.graph.initializer}

    # Get layer dict by name
    layer_dict = {l.name: l for l in keras_model.layers if l.weights}

    for b in range(num_blocks):
        # Conv1D weight: ONNX (out_C, in_C, k) → Keras (k, in_C, out_C)
        for sub in ['conv1', 'conv2']:
            w_name = f'tcn.{b}.{sub}.weight'
            b_name = f'tcn.{b}.{sub}.bias'
            w = init_map[w_name]  # (out_C, in_C, k)
            k_w = w.shape[2]  # kernel size
            # PyTorch conv1d kernel is (out_C, in_C, k)
            w_keras = np.transpose(w, (2, 1, 0))  # (k, in_C, out_C)
            layer_name = f'block{b}_{sub}'
            layer_dict[layer_name].set_weights([w_keras, init_map[b_name]])

        # SE
        for sub, expected_keras_shape in [('fc1', (channels, channels // 8)),
                                           ('fc2', (channels // 8, channels))]:
            w_name = f'tcn.{b}.se.{sub}.weight'
            b_name = f'tcn.{b}.se.{sub}.bias'
            w = init_map[w_name]
            # ONNX Gemm with transB=1 stores weight as (in, out); PyTorch has (out, in)
            # → always transpose to (out, in) for Keras Dense
            w_keras = w.T
            layer_name = f'block{b}_se_{sub}'
            layer_dict[layer_name].set_weights([w_keras, init_map[b_name]])

        # Residual (only block 0)
        if b == 0:
            w_name = 'tcn.0.residual.weight'
            b_name = 'tcn.0.residual.bias'
            w = init_map[w_name]
            w_keras = np.transpose(w, (2, 1, 0))  # (k=1, in_C=23, out_C=32)
            layer_dict[f'block{b}_res'].set_weights([w_keras, init_map[b_name]])

    # FC head
    for sub in ['fc1', 'fc2']:
        w_name = sub + '.weight'
        b_name = sub + '.bias'
        w = init_map[w_name]
        # PyTorch nn.Linear weight is (out, in), ONNX export uses transB=1 → ONNX stored as (in, out)
        # → always transpose to (out, in) for Keras Dense
        w_keras = w.T
        layer_dict[sub].set_weights([w_keras, init_map[b_name]])


def keras_to_tflite_fp32(keras_model, out_path):
    """Keras -> SavedModel -> TFLite FP32."""
    sm_dir = out_path + '_sm'
    if os.path.exists(sm_dir):
        import shutil
        shutil.rmtree(sm_dir)

    keras_model.export(sm_dir)  # tf 2.16+ uses export, not save
    print(f'  saved SavedModel → {sm_dir}')

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.optimizations = []  # ⭐ 不量化!
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_bytes = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_bytes)

    import shutil
    shutil.rmtree(sm_dir, ignore_errors=True)
    return tflite_bytes


def tflite_predict_probs_fp32(tflite_bytes, X):
    """X shape (N, 16, 23) — Keras layout (B, T, C). TFLite expected same."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f'  input:  {inp_d["shape"]} {inp_d["dtype"]}')
    print(f'  output: {out_d["shape"]} {out_d["dtype"]}')

    probs = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        interp.set_tensor(inp_d['index'], X[i:i+1].astype(np.float32))
        interp.invoke()
        out = interp.get_tensor(out_d['index'])
        logit = float(out.flatten()[0])
        probs[i] = 1.0 / (1.0 + np.exp(-logit))
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='.', help='output dir for .tflite')
    p.add_argument('--header-out', default='MCU/Core/Inc', help='output dir for .h C arrays')
    p.add_argument('--seeds', default='42,123,456,789,1024', help='comma-separated seeds')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.header_out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]

    # Load data
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

        print(f'\n[seed={seed}] building keras model...')
        keras_model = build_tcn_se_keras(num_blocks=3, in_channels=23, channels=32,
                                          se_reduction=8, window=16)
        # Don't print full summary (too long)
        if seed == 42:
            keras_model.summary()

        # Load weights from ONNX
        load_onnx_weights_to_keras(keras_model, onnx_path)
        print(f'  weights loaded from {onnx_path}')

        # Convert to TFLite FP32
        out_path = os.path.join(args.out, f'model_v4_se_23dim_ch32_qat_s{seed}_fp32.tflite')
        tflite_bytes = keras_to_tflite_fp32(keras_model, out_path)
        print(f'  → TFLite FP32 {len(tflite_bytes)/1024:.1f} KB')

        # Sanity: predict on small subset
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
        print(f'  F1@0.5={results["per_seed"][str(seed)]["0.50"]["F1"]:.4f}  '
              f'AUC={results["per_seed"][str(seed)]["AUC"]:.4f}')

    # 5-seed ensemble
    if len(all_probs) == len(seeds):
        all_probs = np.stack(all_probs, axis=0)
        ens_probs = all_probs.mean(axis=0)
        print(f'\n=== 5-seed ensemble (FP32) ===')
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

    # Write results
    out_json = os.path.join(args.out, 'ensemble_fp32_5seed_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  results → {out_json}')

    # Generate C headers
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
    print(f'\n[done]')


if __name__ == '__main__':
    main()
