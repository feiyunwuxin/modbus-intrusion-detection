#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train TCN+SE 23-dim ch=32 in Keras FP32 (baseline before TFLite conversion).

Mirrors `retrain_tcn_23dim_b64_ch32_do01_savept.py` (PyTorch) to give the same
training data, same feature indices, same hyperparameters — output is a
Keras-native FP32 .h5 model that will be the **starting point** for the
TFLite converter (PTQ or QAT).

Pipeline:
  1. Load window-level data:
       X_train_binary_v2_scada_window16.npy  → (N_train, 16, 27)
       X_val_binary_v2_scada_window16.npy    → (N_val,   16, 27)
       X_test_binary_v2_scada_window16.npy   → (N_test,  16, 27)
       y_{train,val,test}_binary_v2_scada_window16.npy
  2. Slice KEEP_23 (drop indices {2, 3, 20, 22} of 27-dim)
  3. Clip ±10.0 (max abs value)
  4. Train 5 seeds × 20 epoch (Adam + BCEWithLogitsLoss)
  5. Save to `model_v4_se_keras_s{seed}.h5`

Usage:
    .venv311/Scripts/python train_tcn_se_keras.py --seed 42 [--epochs 20]
    .venv311/Scripts/python train_tcn_se_keras.py --all
"""

import argparse
import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Local import
from tcn_se_keras_model import (
    build_model,
    data_to_keras_input,
    WINDOW, N_FEATURES, CHANNELS, N_BLOCKS, KERNEL_SIZE, DILATIONS,
    SE_REDUCTION, DROPOUT, FC_DROPOUT,
)


# === Constants =============================================================
BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
TAG = "v4_se_23dim_b64_ch32_do01_window16_keras"
CLIP_VAL = 10.0
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 64
DEFAULT_LR = 4e-3
DEFAULT_WD = 1e-5
PATIENCE = 5

# KEEP_23: drop {2 (length), 3 (setpoint), 20 (crc_mean_w), 22 (cmd_count_w)} from 27-dim
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 21, 23, 24, 25, 26]
assert len(KEEP_23) == 23, f"KEEP_23 must have 23 indices, got {len(KEEP_23)}"


# === Helpers ===============================================================

def load_data():
    """Load (X_train, y_train), (X_val, y_val), (X_test, y_test).

    Returns all as float32 (X) and int64 (y), sliced to KEEP_23 + clipped ±10.
    """
    paths = {
        'X_train': os.path.join(BASE, 'X_train_binary_v2_scada_window16.npy'),
        'y_train': os.path.join(BASE, 'y_train_binary_v2_scada_window16.npy'),
        'X_val':   os.path.join(BASE, 'X_val_binary_v2_scada_window16.npy'),
        'y_val':   os.path.join(BASE, 'y_val_binary_v2_scada_window16.npy'),
        'X_test':  os.path.join(BASE, 'X_test_binary_v2_scada_window16.npy'),
        'y_test':  os.path.join(BASE, 'y_test_binary_v2_scada_window16.npy'),
    }
    out = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing: {p}")
        if k.startswith('X'):
            arr = np.load(p).astype(np.float32)[:, :, KEEP_23]
            arr = np.clip(arr, -CLIP_VAL, CLIP_VAL)
        else:
            arr = np.load(p).astype(np.int64)
        out[k] = arr
    return out['X_train'], out['y_train'], out['X_val'], out['y_val'], out['X_test'], out['y_test']


def make_pos_weight(y_train):
    """BCEWithLogitsLoss pos_weight = n_neg / n_pos."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return float(n_neg / max(n_pos, 1))


def compile_model(model, pos_weight_value, lr=DEFAULT_LR, wd=DEFAULT_WD):
    """Compile Keras model with Adam + BCE."""
    # BCEWithLogits equivalent: use 'binary_crossentropy' with from_logits=False,
    # but output is a logit (no sigmoid). Apply sigmoid manually after loss.
    # Simpler: use binary_crossentropy and apply sigmoid inside the model.
    # For QAT-friendly model, output raw logit and use tf.keras.losses.BinaryCrossentropy(from_logits=True).
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, weight_decay=wd),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name='acc', threshold=0.0)],
    )
    return model


def train_one_seed(seed, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH, verbose=2):
    """Train one seed end-to-end. Returns dict of metrics."""
    print(f"\n========== SEED {seed} ==========")
    tf.random.set_seed(int(seed))
    np.random.seed(int(seed))

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"  X_train: {X_train.shape}  y_train dist: 0={(y_train==0).sum()} 1={(y_train==1).sum()}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # pos_weight
    pos_weight = make_pos_weight(y_train)
    print(f"  pos_weight (n_neg/n_pos): {pos_weight:.4f}")

    # Build model
    model = build_model(seed=int(seed))
    n_params = model.count_params()
    print(f"  Total params: {n_params:,}")

    # Sample weight for class imbalance (mirrors PyTorch pos_weight)
    sw = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)

    # Compile
    model = compile_model(model, pos_weight)

    # Callbacks
    cb = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_acc', mode='max', factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_acc', mode='max', patience=PATIENCE, restore_best_weights=True, verbose=1),
    ]

    # Train
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        sample_weight=sw,
        validation_data=(X_val, y_val),
        batch_size=batch_size, epochs=epochs,
        callbacks=cb, verbose=verbose,
    )
    train_t = time.time() - t0
    print(f"  Trained in {train_t:.1f}s, best val acc = {max(history.history['val_acc']):.4f}")

    # Save .h5
    h5_path = os.path.join(BASE, f"model_{TAG}_s{seed}.h5")
    model.save(h5_path)
    print(f"  Saved: {h5_path} ({os.path.getsize(h5_path):,} bytes)")

    # Test metrics
    test_logits = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_prob = 1.0 / (1.0 + np.exp(-test_logits))   # sigmoid
    test_pred = (test_prob >= 0.5).astype(np.int64)

    from sklearn.metrics import (f1_score, accuracy_score,
                                 roc_auc_score, average_precision_score)
    f1m = float(f1_score(y_test, test_pred, average='macro'))
    f1b = float(f1_score(y_test, test_pred, average='binary'))
    acc = float(accuracy_score(y_test, test_pred))
    roc_auc = float(roc_auc_score(y_test, test_prob))
    pr_auc = float(average_precision_score(y_test, test_prob))

    print(f"  Test  Acc={acc:.4f}  F1m={f1m:.4f}  BinF1={f1b:.4f}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    return {
        'seed': int(seed),
        'n_params': n_params,
        'train_time_s': train_t,
        'best_epoch': len(history.history['loss']),
        'test_metrics': {
            'accuracy': acc,
            'macro_f1': f1m,
            'binary_f1': f1b,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
        },
        'h5_path': h5_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None, help='Single seed.')
    ap.add_argument('--all', action='store_true', help='All 5 seeds.')
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--batch',  type=int, default=DEFAULT_BATCH)
    ap.add_argument('--verbose', type=int, default=2)
    args = ap.parse_args()

    seeds = SEEDS if args.all else ([args.seed] if args.seed is not None else [42])
    print(f"=== Training Keras TCN+SE 23-dim ch=32 × {len(seeds)} seed(s) ===")

    all_results = []
    for s in seeds:
        r = train_one_seed(s, epochs=args.epochs, batch_size=args.batch, verbose=args.verbose)
        all_results.append(r)

    # Summary
    if len(all_results) > 1:
        f1m_list = [r['test_metrics']['macro_f1'] for r in all_results]
        pr_list = [r['test_metrics']['pr_auc'] for r in all_results]
        print(f"\n=== Ensemble baseline (5-seed prob mean) ===")
        # Run ensemble
        _, _, _, _, X_test, y_test = load_data()
        all_probs = []
        for r in all_results:
            m = keras.models.load_model(r['h5_path'], compile=False)
            logits = m.predict(X_test, batch_size=DEFAULT_BATCH, verbose=0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_probs.append(probs)
        ens_prob = np.mean(all_probs, axis=0)
        ens_pred = (ens_prob >= 0.5).astype(np.int64)
        from sklearn.metrics import f1_score, average_precision_score
        ens_f1m = float(f1_score(y_test, ens_pred, average='macro'))
        ens_f1b = float(f1_score(y_test, ens_pred, average='binary'))
        ens_pr = float(average_precision_score(y_test, ens_prob))
        print(f"  Per-seed F1m mean = {np.mean(f1m_list):.4f} ± {np.std(f1m_list):.4f}")
        print(f"  5-seed ENSEMBLE F1m = {ens_f1m:.4f}  PR-AUC = {ens_pr:.4f}  BinF1 = {ens_f1b:.4f}")

    # Save meta
    meta = {
        'model': 'TCN+SE (Keras)',
        'tag': TAG,
        'seeds': seeds,
        'n_seeds': len(seeds),
        'per_seed': all_results,
    }
    meta_path = os.path.join(BASE, f'processed_meta_{TAG}.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[saved meta] {meta_path}")


if __name__ == "__main__":
    main()
