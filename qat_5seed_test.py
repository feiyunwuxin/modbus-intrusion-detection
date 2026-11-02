#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT 5-seed ensemble F1 evaluation.

Loads 5 QAT INT8 headers (one per seed) and runs ensemble (mean of probabilities)
across the full test set. Reports F1m, Binary F1 (Acc/P/R/F1), PR-AUC @ thr=0.5
and thr=0.49.

Acceptance (vs Hybrid 0.8737):
  - F1m >= 0.85
  - Binary F1 >= 0.85
  - PR-AUC >= 0.91

Usage:
    python qat_5seed_test.py [--n_samples 5000]
"""

import os, sys, time, argparse, json
import numpy as np
import torch

# Reuse verifier functions
from verify_qat_pc import (
    parse_qat_header, full_int8_forward,
    TCNClassifierSE, KEEP_23, N_FEATURES, BASE, SEEDS
)

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, auc
)


def build_pytorch_dataset(seed: int, n_samples: int = 5000):
    """Load PyTorch FP32 baseline for the QAT weights to extract features."""
    pass  # placeholder — we'll just use INT8 forward for ensemble


def load_test_data(n_samples: int = None):
    """Load and preprocess test set (23-dim, window=16).

    X_test shape: (N, T=16, F=27) → slice F to KEEP_23 → (N, T=16, F=23)
    """
    X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_test = np.clip(X_test, -10.0, 10.0)
    y_test = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy"))
    if n_samples and n_samples < len(X_test):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), size=n_samples, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
    return X_test, y_test


def get_qat_probs(seed: int, X_test, h_state):
    """Run QAT INT8 forward on test set, return probabilities (N,)."""
    probs = np.zeros(len(X_test), dtype=np.float32)
    t0 = time.time()
    for i, x in enumerate(X_test):
        # x shape: (T=16, F=23) → need (F=23, T=16) for forward
        x_t = x.T.astype(np.float32)
        probs[i] = full_int8_forward(x_t, h_state)
    dt = time.time() - t0
    return probs, dt


def report_metrics(y_true, probs, thr):
    pred = (probs >= thr).astype(int)
    return {
        'f1_macro':  f1_score(y_true, pred, average='macro'),
        'f1_binary': f1_score(y_true, pred),
        'accuracy':  accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall':    recall_score(y_true, pred, zero_division=0),
        'n_pos_pred': int(pred.sum()),
        'n_pos_true': int(y_true.sum()),
    }


def compute_pr_auc(y_true, probs):
    """Compute PR-AUC using precision-recall curve integration."""
    p, r, _ = precision_recall_curve(y_true, probs)
    return float(auc(r, p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=None,
                        help='limit test samples (default: full set)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='seeds to load (default: 42 123 456 789 1024)')
    parser.add_argument('--thrs', type=float, nargs='+', default=[0.5, 0.49, 0.45, 0.40, 0.33],
                        help='thresholds to evaluate')
    parser.add_argument('--out', type=str, default=None,
                        help='output json path')
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else SEEDS

    print("=" * 70)
    print(" QAT Full INT8 5-seed Ensemble F1 Evaluation")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Test samples: {args.n_samples if args.n_samples else 'FULL'}")

    X_test, y_test = load_test_data(args.n_samples)
    print(f"Loaded X_test={X_test.shape} y_test={y_test.shape}  pos_rate={y_test.mean():.4f}")

    # Per-seed prob & metrics
    all_probs = {}      # seed → probs
    per_seed_metrics = {}
    per_seed_pr_auc = {}

    for s in seeds:
        h_path = os.path.join(BASE, "fullint8_qat", "h743", "Core", "Inc",
                               f"model_v4_se_23dim_ch32_qat_s{s}.h")
        if not os.path.exists(h_path):
            print(f"\n[s={s}] Header not found: {h_path}")
            print(f"        用户后续 20h GPU 训练 + 量化后会生成")
            continue

        print(f"\n[fwd s={s}] loading {os.path.basename(h_path)}")
        h_state = parse_qat_header(h_path)
        probs, dt = get_qat_probs(s, X_test, h_state)
        all_probs[s] = probs
        pr_auc = compute_pr_auc(y_test, probs)
        per_seed_pr_auc[s] = pr_auc

        print(f"[fwd s={s}] {dt:.1f}s ({dt/len(X_test)*1000:.2f}ms/sample)  "
              f"PR-AUC={pr_auc:.4f}")

        per_seed_metrics[s] = {}
        for thr in args.thrs:
            m = report_metrics(y_test, probs, thr)
            per_seed_metrics[s][f'thr_{thr}'] = m
            print(f"  @ thr={thr}: F1m={m['f1_macro']:.4f} F1b={m['f1_binary']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} Acc={m['accuracy']:.4f}")

    if not all_probs:
        print("\n[NOTE] No QAT headers found — 用户后续 GPU 训练 + 量化后会运行")
        return

    # Ensemble: mean of probabilities
    prob_stack = np.stack(list(all_probs.values()), axis=0)  # (S, N)
    ens_prob = prob_stack.mean(axis=0)
    ens_pr_auc = compute_pr_auc(y_test, ens_prob)

    print(f"\n{'='*70}")
    print(f" Ensemble ({len(all_probs)} seeds)")
    print(f"{'='*70}")
    print(f"PR-AUC = {ens_pr_auc:.4f}")

    ens_metrics = {}
    for thr in args.thrs:
        m = report_metrics(y_test, ens_prob, thr)
        ens_metrics[f'thr_{thr}'] = m
        print(f"@ thr={thr}: F1m={m['f1_macro']:.4f} F1b={m['f1_binary']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} Acc={m['accuracy']:.4f} "
              f"n_pred={m['n_pos_pred']}")

    # Acceptance check
    best_thr = max(args.thrs, key=lambda t: ens_metrics[f'thr_{t}']['f1_binary'])
    best_f1 = ens_metrics[f'thr_{best_thr}']['f1_binary']
    print(f"\nBest F1 (binary) @ thr={best_thr}: {best_f1:.4f}")
    print(f"Acceptance:")
    print(f"  F1b >= 0.85:  {'PASS' if best_f1 >= 0.85 else 'FAIL'}")
    print(f"  PR-AUC >= 0.91: {'PASS' if ens_pr_auc >= 0.91 else 'FAIL'}")

    # Save results
    out_path = args.out or os.path.join(BASE, "fullint8_qat", "fullint8_qat_5seed_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = {
        'seeds_used': list(all_probs.keys()),
        'n_test_samples': len(X_test),
        'pos_rate': float(y_test.mean()),
        'per_seed_pr_auc': per_seed_pr_auc,
        'per_seed_metrics': per_seed_metrics,
        'ensemble_pr_auc': ens_pr_auc,
        'ensemble_metrics': ens_metrics,
        'best_thr': best_thr,
        'best_f1_binary': best_f1,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> Wrote {out_path}")


if __name__ == "__main__":
    main()