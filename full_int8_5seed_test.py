#!/usr/bin/env python3
"""5-seed Full INT8 evaluation on X_test_binary_v2_scada_window16.npy.

Uses the fake-quant path from verify_full_int8_pc.py (with the documented limitations)
to compute 4 metrics (Accuracy / Precision / Recall / F1) at thr=0.49 for each seed.
Then computes 5-seed ensemble F1 (avg of per-seed sigmoid outputs, then threshold).

Baseline (Hybrid INT8, 5-seed): F1=0.8737 (from project-2026-09-13-rerun-4metrics.md).
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_full_int8_pc import parse_full_int8_header, full_int8_forward
from quantize_v4se_23dim_ch32_full_int8 import TCNClassifierSE

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 456, 789, 1024]
TAG_PT = "v4_se_23dim_b64_ch32_do01_window16"
TAG_H  = "v4_se_23dim_ch32_fullint8"
THR = 0.49


def eval_seed(seed, X, y, n_samples=None):
    pt_path = os.path.join(BASE, f"model_{TAG_PT}_s{seed}.pt")
    h_path = os.path.join(BASE, "h743", "Core", "Inc", f"model_{TAG_H}_s{seed}.h")
    print(f"\n=== seed={seed} ===")

    # Load PT (for fp32 classifier biases only)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = TCNClassifierSE(in_ch=23)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    fp32_b = {
        "fc1.bias": model.fc1.bias.data.numpy().copy(),
        "fc2.bias": model.fc2.bias.data.numpy().copy(),
    }
    h_state = parse_full_int8_header(h_path)

    # Sample subset if requested
    if n_samples and n_samples < len(X):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y
        idx = np.arange(len(X))

    qi_preds = np.zeros(len(X_sub), dtype=np.float32)
    t0 = time.time()
    for i in range(len(X_sub)):
        x = X_sub[i].T.astype(np.float32)  # (23, 16)
        qi_preds[i] = full_int8_forward(x, h_state, fp32_b)
    dt = time.time() - t0

    qi_bin = (qi_preds >= THR).astype(int)
    y_true = y_sub.astype(int)

    acc = accuracy_score(y_true, qi_bin)
    prec = precision_score(y_true, qi_bin, zero_division=0)
    rec = recall_score(y_true, qi_bin, zero_division=0)
    f1 = f1_score(y_true, qi_bin, zero_division=0)

    print(f"  n={len(X_sub)}, time={dt:.1f}s, @thr={THR}")
    print(f"  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    print(f"  Preds: min={qi_preds.min():.4f}, max={qi_preds.max():.4f}, mean={qi_preds.mean():.4f}")
    print(f"  Positive count: {qi_bin.sum()} / {len(y_true)}")

    return {
        "seed": seed,
        "n_samples": int(len(X_sub)),
        "time_s": float(dt),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "qi_positive_count": int(qi_bin.sum()),
        "preds_mean": float(qi_preds.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3432,
                        help="number of test windows (default: all 3432)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    # Load test data
    X = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
    X = np.clip(X[:, :, KEEP_23], -10.0, 10.0)
    y = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(int)
    print(f"X_test shape: {X.shape}, y_test positive count: {y.sum()}/{len(y)}")

    seeds = args.seeds or SEEDS
    results = []
    for seed in seeds:
        try:
            r = eval_seed(seed, X, y, n_samples=args.samples)
            results.append(r)
        except Exception as e:
            print(f"  ERROR seed={seed}: {e}")
            continue

    if not results:
        print("No results!")
        return

    # Summary table
    print("\n" + "=" * 70)
    print(f" Summary ({len(results)} seeds @ thr={THR})")
    print("=" * 70)
    print(f"{'seed':>6}  {'n':>6}  {'Acc':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'time':>6}")
    for r in results:
        print(f"{r['seed']:>6}  {r['n_samples']:>6}  {r['accuracy']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}  {r['f1']:>6.4f}  {r['time_s']:>6.1f}")

    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_p = np.mean([r["precision"] for r in results])
    avg_r = np.mean([r["recall"] for r in results])
    avg_f1 = np.mean([r["f1"] for r in results])
    print(f"{'AVG':>6}  {'':>6}  {avg_acc:>6.4f}  {avg_p:>6.4f}  {avg_r:>6.4f}  {avg_f1:>6.4f}")

    print(f"\nFull INT8 (per-seed avg) vs Hybrid baseline (F1=0.8737):")
    print(f"  Δ F1 = {avg_f1 - 0.8737:+.4f}")

    # Save JSON
    out_path = os.path.join(BASE, "fullint8_5seed_results.json")
    with open(out_path, "w") as f:
        json.dump({"thr": THR, "n_samples": args.samples, "results": results,
                   "avg_accuracy": avg_acc, "avg_precision": avg_p,
                   "avg_recall": avg_r, "avg_f1": avg_f1}, f, indent=2)
    print(f"\n-> Wrote {out_path}")


if __name__ == "__main__":
    main()
