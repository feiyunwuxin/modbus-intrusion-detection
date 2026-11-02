#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TFMOT QAT fine-tune on Keras TCN+SE 23-dim ch=32, then export INT8 .tflite.

Stage 1 → Stage 2 transition: QAT bridge from FP32 baseline (.h5) → INT8 .tflite
that is expected to match FP32 within ~5%.

Pipeline:
  1. Load FP32 baseline (`model_v4_se_keras_s{seed}.h5`)
  2. Apply TFMOT quantize_model() — inserts FakeQuant layers
  3. Fine-tune 5 epochs with Adam + low LR (2e-4)
  4. Convert QAT model to TFLite with full INT8 quantization:
       - per-channel symmetric weight (default INT8)
       - per-tensor symmetric activation (zp=0)
       - INT8 input, INT8 output
  5. Save `model_v4_se_23dim_ch32_qat_s{seed}.tflite`

Usage:
    .venv311/Scripts/python qat_tcn_se_keras.py --seed 42
    .venv311/Scripts/python qat_tcn_se_keras.py --all
"""

import argparse
import os
import time
import numpy as np
import tensorflow as tf

# TFMOT (TensorFlow Model Optimization Toolkit)
import tensorflow_model_optimization as tfmot

from tcn_se_keras_model import WINDOW, N_FEATURES
from train_tcn_se_keras import load_data, TAG, DEFAULT_BATCH, DEFAULT_WD, PATIENCE, KEEP_23, CLIP_VAL

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
TAG_H5 = "v4_se_23dim_b64_ch32_do01_window16_keras"

# QAT fine-tune hyperparameters (vs PyTorch retrain_tcn_23dim_qat.py)
QAT_EPOCHS = 5
QAT_BATCH  = 64
QAT_LR     = 2e-4       # lower than FP32 (4e-3) to avoid disrupting conv kernels
QAT_WD     = 1e-5


# ===========================================================================
# Calibration data generator
# ===========================================================================
def rep_dataset_gen(num_samples=500, seed=42):
    """500 representative samples for TFLite PTQ calibration."""
    X_train, _, _, _, _, _ = load_data()
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(X_train), size=num_samples, replace=False)
    for i in idx:
        yield [X_train[i:i+1].astype(np.float32)]


# ===========================================================================
# QAT fine-tune
# ===========================================================================
def qat_finetune_one_seed(seed):
    """QAT fine-tune one seed end-to-end. Returns tflite_path."""
    print(f"\n========== QAT SEED {seed} ==========")
    tf.random.set_seed(int(seed))
    np.random.seed(int(seed))

    # Step 1: Load FP32 baseline
    h5_path = os.path.join(BASE, f"model_{TAG_H5}_s{seed}.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"FP32 baseline missing: {h5_path}. Run train_tcn_se_keras.py first.")
    print(f"  Loading FP32 baseline: {h5_path}")
    keras_fp32 = tf.keras.models.load_model(h5_path, compile=False)
    print(f"    params: {keras_fp32.count_params():,}")

    # Step 2: Apply TFMOT quantize_model() — inserts FakeQuant layers
    qat_model = tfmot.quantize_model(keras_fp32)
    print(f"  TFMOT quantize_model inserted FakeQuant layers")

    # Step 3: Compile + fine-tune
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LR, weight_decay=QAT_WD),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='acc', threshold=0.0)],
    )

    # pos_weight for imbalance
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))
    sw = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)

    cb = [tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', mode='max', patience=2, restore_best_weights=True, verbose=1)]

    print(f"  Fine-tune {QAT_EPOCHS} epochs (lr={QAT_LR}, batch={QAT_BATCH})...")
    t0 = time.time()
    qat_model.fit(
        X_train, y_train,
        sample_weight=sw,
        validation_data=(X_val, y_val),
        batch_size=QAT_BATCH, epochs=QAT_EPOCHS,
        callbacks=cb, verbose=2,
    )
    train_t = time.time() - t0
    print(f"  Fine-tune done in {train_t:.1f}s")

    # Step 4: TFLite conversion with INT8
    print(f"  Converting QAT → TFLite INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_dataset_gen(num_samples=500, seed=int(seed))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_qat = converter.convert()

    out_path = os.path.join(BASE, f"model_v4_se_23dim_ch32_qat_s{seed}.tflite")
    with open(out_path, 'wb') as f:
        f.write(tflite_qat)
    print(f"  Saved: {out_path} ({len(tflite_qat):,} bytes)")

    # Quick test eval
    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale = in_det['quantization_parameters']['scales'][0]
    in_zp    = in_det['quantization_parameters']['zero_points'][0]
    out_scale = out_det['quantization_parameters']['scales'][0]
    out_zp    = out_det['quantization_parameters']['zero_points'][0]

    correct = 0
    n_test = min(2000, len(X_test))
    for i in range(n_test):
        x_q = np.clip(np.round(X_test[i:i+1] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det['index'], x_q)
        interp.invoke()
        y_q = interp.get_tensor(out_det['index'])[0]
        # Dequantize to logit, then sigmoid
        logit = (y_q.astype(np.float32) - out_zp) * out_scale
        prob = 1.0 / (1.0 + np.exp(-logit))
        pred = 1 if prob >= 0.5 else 0
        if pred == int(y_test[i]):
            correct += 1
    acc = correct / n_test
    print(f"  Sanity test (n={n_test}): accuracy={acc:.4f}")

    # Cleanup
    del keras_fp32, qat_model, interp
    return {
        'seed': int(seed),
        'tflite_path': out_path,
        'tflite_bytes': len(tflite_qat),
        'qat_train_time_s': train_t,
        'qat_test_acc': acc,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    seeds = SEEDS if args.all else ([args.seed] if args.seed is not None else [42])

    print(f"=== QAT fine-tune TCN+SE 23-dim ch=32 × {len(seeds)} seed(s) ===")
    all_results = [qat_finetune_one_seed(s) for s in seeds]

    # Summary
    print("\n=== QAT Summary ===")
    for r in all_results:
        print(f"  seed={r['seed']}  size={r['tflite_bytes']:,} B  "
              f"train_t={r['qat_train_time_s']:.1f}s  acc={r['qat_test_acc']:.4f}")


if __name__ == "__main__":
    main()
