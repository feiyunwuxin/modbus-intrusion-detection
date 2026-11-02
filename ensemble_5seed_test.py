#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5-seed ensemble TFLite evaluation for TCN+SE 23-dim ch=32.

Loads 5 .tflite (FP32 or INT8 QAT) and evaluates:
  - Per-seed metrics (Acc, P, R, Bin-F1, Macro-F1, ROC-AUC, PR-AUC)
  - 5-seed ensemble (probability mean)
  - Threshold sweep @ 0.40 / 0.45 / 0.49 / 0.50 / 0.55

Acceptance (INT8 QAT):
  - Single seed: F1m ≥ 0.85
  - 5-seed ensemble: F1m ≥ 0.87

Acceptance (FP32):
  - Single seed: F1m ≥ 0.87
  - 5-seed ensemble: F1m ≥ 0.88 (target)

Usage:
    .venv311/Scripts/python ensemble_5seed_test.py --quant qat
    .venv311/Scripts/python ensemble_5seed_test.py --quant fp32
    .venv311/Scripts/python ensemble_5seed_test.py --quant ptq
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

from train_tcn_se_keras import load_data

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]


def tflite_predict_proba(interpreter, X):
    """Return sigmoid probabilities for each sample in X."""
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale = in_det['quantization_parameters']['scales'][0]
    in_zp    = in_det['quantization_parameters']['zero_points'][0]
    out_scale = out_det['quantization_parameters']['scales'][0]
    out_zp    = out_det['quantization_parameters']['zero_points'][0]
    in_dtype = in_det['dtype']

    out = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        x = X[i:i+1]
        if in_dtype == np.int8:
            x_q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        elif in_dtype == np.uint8:
            x_q = np.clip(np.round(x / in_scale + in_zp), 0, 255).astype(np.uint8)
        else:
            x_q = x.astype(np.float32)
        interpreter.set_tensor(in_det['index'], x_q)
        interpreter.invoke()
        y_q = interpreter.get_tensor(out_det['index'])[0]
        if y_q.ndim == 0:
            y_q = np.array([y_q])
        logit = (y_q.astype(np.float32) - out_zp) * out_scale
        out[i] = 1.0 / (1.0 + np.exp(-logit[0]))
    return out


def find_tflite(seed, quant):
    """Path to .tflite for given seed + quant mode."""
    if quant == 'fp32':
        return os.path.join(BASE, f"model_v4_se_23dim_ch32_s{seed}.tflite")
    elif quant == 'ptq':
        return os.path.join(BASE, f"model_v4_se_23dim_ch32_ptq_s{seed}.tflite")
    elif quant == 'qat':
        return os.path.join(BASE, f"model_v4_se_23dim_ch32_qat_s{seed}.tflite")
    raise ValueError(f"unknown quant: {quant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quant', choices=['fp32', 'ptq', 'qat'], default='qat')
    ap.add_argument('--seeds', type=str, default='42,123,456,789,1024')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    # Load test data
    _, _, _, _, X_test, y_test = load_data()
    print(f"=== 5-seed ensemble: {args.quant.upper()} × {len(seeds)} seed(s) on {len(X_test)} windows ===")
    print(f"  y_test dist: 0={(y_test==0).sum()} 1={(y_test==1).sum()} (Attack%={100*(y_test==1).mean():.1f}%)")

    per_seed_metrics = []
    all_probs = []
    for seed in seeds:
        path = find_tflite(seed, args.quant)
        if not os.path.exists(path):
            print(f"  [SKIP] seed={seed}: missing {path}")
            continue
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        probs = tflite_predict_proba(interpreter, X_test)
        all_probs.append(probs)
        interpreter.close()

        # Per-seed metrics @ thr=0.5
        pred = (probs >= 0.5).astype(np.int64)
        m = {
            'seed': seed,
            'path': path,
            'size_bytes': os.path.getsize(path),
            'accuracy':  float(accuracy_score(y_test, pred)),
            'precision': float(precision_score(y_test, pred, zero_division=0)),
            'recall':    float(recall_score(y_test, pred, zero_division=0)),
            'f1_binary': float(f1_score(y_test, pred, average='binary', zero_division=0)),
            'f1_macro':  float(f1_score(y_test, pred, average='macro')),
            'roc_auc':   float(roc_auc_score(y_test, probs)),
            'pr_auc':    float(average_precision_score(y_test, probs)),
        }
        per_seed_metrics.append(m)
        print(f"  seed={seed:4d}  size={m['size_bytes']:>7,}  "
              f"Acc={m['accuracy']:.4f}  F1m={m['f1_macro']:.4f}  BinF1={m['f1_binary']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    if not all_probs:
        print("No model successfully loaded.")
        return

    # Ensemble
    ens_probs = np.mean(all_probs, axis=0)
    print(f"\n=== 5-seed Ensemble ({args.quant.upper()}, prob mean) ===")

    # Threshold sweep
    print(f"\n  Threshold sweep:")
    print(f"  {'thr':>6} | {'Acc':>6} | {'F1b':>6} | {'F1m':>6} | {'P':>6} | {'R':>6} | {'FPR':>6}")
    print(f"  {'-'*6}+{'-'*6}+{'-'*6}+{'-'*6}+{'-'*6}+{'-'*6}+{'-'*6}")
    sweep = []
    for thr in [0.40, 0.45, 0.49, 0.50, 0.55, 0.60]:
        pred = (ens_probs >= thr).astype(np.int64)
        acc = float(accuracy_score(y_test, pred))
        f1b = float(f1_score(y_test, pred, average='binary', zero_division=0))
        f1m = float(f1_score(y_test, pred, average='macro'))
        prec = float(precision_score(y_test, pred, zero_division=0))
        rec = float(recall_score(y_test, pred, zero_division=0))
        fpr = float(((pred == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1))
        sweep.append({'threshold': thr, 'accuracy': acc, 'f1_binary': f1b, 'f1_macro': f1m,
                      'precision': prec, 'recall': rec, 'fpr': fpr})
        print(f"  {thr:>6.2f} | {acc:>6.4f} | {f1b:>6.4f} | {f1m:>6.4f} | "
              f"{prec:>6.4f} | {rec:>6.4f} | {fpr:>6.4f}")
    roc_auc_ens = float(roc_auc_score(y_test, ens_probs))
    pr_auc_ens  = float(average_precision_score(y_test, ens_probs))
    print(f"\n  Ensemble ROC-AUC = {roc_auc_ens:.4f}  PR-AUC = {pr_auc_ens:.4f}")

    # Save
    out = {
        'quant': args.quant,
        'n_seeds': len(per_seed_metrics),
        'seeds': seeds,
        'per_seed': per_seed_metrics,
        'ensemble': {
            'roc_auc': roc_auc_ens,
            'pr_auc': pr_auc_ens,
            'threshold_sweep': sweep,
        },
        'n_test': int(len(X_test)),
        'y_test_attack_ratio': float((y_test == 1).mean()),
    }
    out_path = os.path.join(BASE, f"ensemble_{args.quant}_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
