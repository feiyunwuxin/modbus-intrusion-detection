#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Keras TCN+SE 23-dim ch=32 model to TFLite (FP32 or INT8).

Two flavors:
  --mode fp32     : full FP32 .tflite (no quantization) — for sanity check
  --mode int8_ptq : post-training quantization (PTQ) INT8 — uses
                    representative dataset of 500 samples for calibration

Both outputs are .tflite files consumed by:
  - PC  : tf.lite.Interpreter  (Python runtime, for verification)
  - MCU : TFLite Micro on STM32H743 (Stage 3 of the plan)

Usage:
    .venv311/Scripts/python convert_tcn_se_to_tflite.py --seed 42 --mode fp32
    .venv311/Scripts/python convert_tcn_se_to_tflite.py --seed 42 --mode int8_ptq
    .venv311/Scripts/python convert_tcn_se_to_tflite.py --all --mode int8_ptq
"""

import argparse
import os
import numpy as np
import tensorflow as tf

# Local
from tcn_se_keras_model import WINDOW, N_FEATURES
from train_tcn_se_keras import load_data, KEEP_23, CLIP_VAL

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
TAG_H5 = "v4_se_23dim_b64_ch32_do01_window16_keras"


# ===========================================================================
# Representative dataset generator for INT8 PTQ calibration
# ===========================================================================
def rep_dataset_gen(num_samples=500, seed=42):
    """Yield calibration samples (X, y_dummy) for PTQ."""
    X_train, _, _, _, _, _ = load_data()
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(X_train), size=num_samples, replace=False)
    for i in idx:
        x = X_train[i:i+1].astype(np.float32)  # (1, 16, 23)
        yield [x]


# ===========================================================================
# Conversion
# ===========================================================================
def convert_fp32(h5_path, tflite_path):
    """FP32 .tflite conversion (no quantization)."""
    model = tf.keras.models.load_model(h5_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []   # no quantization
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    del model
    return tflite_model


def convert_int8_ptq(h5_path, tflite_path, num_calib=500):
    """INT8 PTQ .tflite conversion.

    Uses per-channel symmetric weight + per-tensor symmetric activation (zp=0)
    schema (default TFLite behavior). Output is fully INT8.

    Calibration dataset: 500 samples from X_train.
    """
    model = tf.keras.models.load_model(h5_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_dataset_gen(num_calib)
    # Force fully INT8 (input + output both int8)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    del model
    return tflite_model


# ===========================================================================
# Driver
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--mode', choices=['fp32', 'int8_ptq'], default='int8_ptq')
    ap.add_argument('--num_calib', type=int, default=500)
    args = ap.parse_args()

    seeds = SEEDS if args.all else ([args.seed] if args.seed is not None else [42])

    mode_str = 'FP32' if args.mode == 'fp32' else 'INT8 PTQ'
    print(f"=== Converting Keras TCN+SE → TFLite ({mode_str}) × {len(seeds)} seed(s) ===")

    for s in seeds:
        h5_path = os.path.join(BASE, f"model_{TAG_H5}_s{s}.h5")
        if not os.path.exists(h5_path):
            print(f"  [SKIP] seed={s}: missing .h5 ({h5_path})")
            continue

        suffix = '' if args.mode == 'fp32' else '_ptq'
        tflite_path = os.path.join(BASE, f"model_v4_se_23dim_ch32{suffix}_s{s}.tflite")
        print(f"  seed={s}  {h5_path} → {tflite_path}")

        if args.mode == 'fp32':
            data = convert_fp32(h5_path, tflite_path)
        else:
            data = convert_int8_ptq(h5_path, tflite_path, num_calib=args.num_calib)
        print(f"    size = {len(data):,} bytes")


if __name__ == "__main__":
    main()
