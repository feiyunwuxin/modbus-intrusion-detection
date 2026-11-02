#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify TFLite TCN+SE 23-dim ch=32 model matches Keras FP32 baseline.

Procedure:
  1. Load Keras FP32 baseline (.h5)
  2. Load .tflite (FP32 or INT8)
  3. On N=200 test samples, predict with both and compare:
     - max_diff    (max abs difference of logits / probabilities)
     - mean_diff
     - agreement   (binary agreement @ thr=0.5)

Acceptance (INT8 PTQ): max_diff < 0.10, agreement > 0.95
Acceptance (INT8 QAT): max_diff < 0.05, agreement > 0.97

Usage:
    .venv311/Scripts/python verify_tflite_pc.py --seed 42 --tflite ...ptq.tflite
    .venv311/Scripts/python verify_tflite_pc.py --seed 42 --tflite ...qat.tflite
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from train_tcn_se_keras import load_data

BASE = r"C:\work\Claude\Issue"
TAG_H5 = "v4_se_23dim_b64_ch32_do01_window16_keras"


# ===========================================================================
# Keras predictor
# ===========================================================================
def keras_predict(model, X):
    """Return sigmoid probabilities."""
    logits = model.predict(X, verbose=0)
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))


# ===========================================================================
# TFLite predictor
# ===========================================================================
def tflite_predict(interpreter, X):
    """Return (logits or probabilities) per sample.

    For FP32 .tflite: output is float, raw logit → apply sigmoid here.
    For INT8 .tflite: output is int8 with (scale, zero_point). Dequantize
        to FP32 raw logit, then apply sigmoid.
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_dtype = input_details['dtype']
    out_dtype = output_details['dtype']
    out_scale = output_details['quantization_parameters']['scales'][0]
    out_zp    = output_details['quantization_parameters']['zero_points'][0]
    in_scale  = input_details['quantization_parameters']['scales'][0]
    in_zp     = input_details['quantization_parameters']['zero_points'][0]

    out = []
    for i in range(len(X)):
        x = X[i:i+1].astype(np.float32)
        if in_dtype == np.int8:
            x_q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details['index'], x_q)
        elif in_dtype == np.uint8:
            x_q = np.clip(np.round(x / in_scale + in_zp), 0, 255).astype(np.uint8)
            interpreter.set_tensor(input_details['index'], x_q)
        else:  # float32
            interpreter.set_tensor(input_details['index'], x.astype(np.float32))

        interpreter.invoke()

        if out_dtype == np.int8 or out_dtype == np.uint8:
            y_q = interpreter.get_tensor(output_details['index'])[0]
            # Dequantize to FP32 (raw logit)
            y_fp = (y_q.astype(np.float32) - out_zp) * out_scale
        else:
            y_fp = interpreter.get_tensor(output_details['index'])[0].astype(np.float32)
            if y_fp.ndim == 0:
                y_fp = np.array([y_fp])
        out.append(y_fp[0])
    return np.array(out, dtype=np.float32)


# ===========================================================================
# Comparison
# ===========================================================================
def compare(probs_keras, probs_tflite, threshold=0.5):
    """Compute (max_diff, mean_diff, agreement). All on raw logit space."""
    logit_keras = np.log(probs_keras / (1.0 - np.clip(probs_keras, 1e-7, 1.0 - 1e-7)))
    logit_tflite = np.log(probs_tflite / (1.0 - np.clip(probs_tflite, 1e-7, 1.0 - 1e-7)))
    diff = np.abs(logit_keras - logit_tflite)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    pred_keras = (logit_keras >= np.log(threshold / (1.0 - threshold))).astype(np.int64)
    pred_tflite = (logit_tflite >= np.log(threshold / (1.0 - threshold))).astype(np.int64)
    agreement = float((pred_keras == pred_tflite).mean())

    return {
        'max_diff_logit': max_diff,
        'mean_diff_logit': mean_diff,
        'agreement_at_05': agreement,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tflite', type=str, required=True,
                    help='Path to .tflite file to verify')
    ap.add_argument('--n_samples', type=int, default=200)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    # Load data
    _, _, _, _, X_test, y_test = load_data()
    rng = np.random.default_rng(seed=args.seed)
    idx = rng.choice(len(X_test), size=args.n_samples, replace=False)
    X = X_test[idx]
    y = y_test[idx]

    # Load Keras baseline
    h5_path = os.path.join(BASE, f"model_{TAG_H5}_s{args.seed}.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Keras baseline missing: {h5_path}")
    print(f"Loading Keras FP32 baseline: {h5_path}")
    keras_model = tf.keras.models.load_model(h5_path, compile=False)
    probs_keras = keras_predict(keras_model, X)

    # Load TFLite
    print(f"Loading TFLite model: {args.tflite}")
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    probs_tflite = tflite_predict(interpreter, X)

    # Compare
    res = compare(probs_keras, probs_tflite, args.threshold)
    print(f"\n=== Verification: Keras FP32 ↔ {os.path.basename(args.tflite)} ===")
    print(f"  Samples:          {args.n_samples}")
    print(f"  Threshold:        {args.threshold}")
    print(f"  max_diff (logit): {res['max_diff_logit']:.6f}")
    print(f"  mean_diff (logit):{res['mean_diff_logit']:.6f}")
    print(f"  Agreement @ 0.5:  {res['agreement_at_05']*100:.2f}%")

    # Acceptance
    if 'qat' in args.tflite.lower():
        print(f"  Acceptance (QAT): max_diff < 0.05, agreement > 0.97  "
              f"→ {'PASS' if (res['max_diff_logit']<0.05 and res['agreement_at_05']>0.97) else '❌ FAIL'}")
    else:
        print(f"  Acceptance (PTQ): max_diff < 0.10, agreement > 0.95  "
              f"→ {'PASS' if (res['max_diff_logit']<0.10 and res['agreement_at_05']>0.95) else '❌ FAIL'}")


if __name__ == "__main__":
    main()
