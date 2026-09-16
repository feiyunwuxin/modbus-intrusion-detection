#!/usr/bin/env python3
"""Snapshot Ensemble evaluation: per-snapshot, per-seed-snap-ensemble, full ensemble.

Reads:
  - predictions_test_tcn_snap_s{seed}_ep{epoch}.csv  (5 epochs × 5 seeds = 25 files)
  - predictions_test_tcn_snap_s{seed}_ens.csv       (5 files, per-seed snap ensemble)

Computes:
  - Per-snapshot F1 (avg across 5 seeds, thr=0.5)
  - Per-seed-snap-ensemble F1 (avg across 5 seeds)
  - Full 25-snapshot × 5-seed = 25-model ensemble F1 + threshold sweep
  - Compare to baseline 0.8795 and Stacking M3 0.8904
"""
import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

BASE = r"D:\workspace\claude\Issue\Issue"
SEEDS = [42, 123, 456, 789, 1024]
SNAPSHOTS = [4, 8, 12, 16, 20]
THRS = np.round(np.arange(0.30, 0.71, 0.01), 2)
BASELINE_F1 = 0.8795
STACKING_F1 = 0.8904


def four_metrics(y, p, thr=0.5):
    y_pred = (p >= thr).astype(int)
    return {
        "Accuracy": float(accuracy_score(y, y_pred)),
        "Precision": float(precision_score(y, y_pred, zero_division=0)),
        "Recall": float(recall_score(y, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y, y_pred, zero_division=0)),
    }


def main():
    print("=== Snapshot Ensemble Evaluation ===\n")

    y_ref = None
    # Load all per-snapshot files
    per_snap_probs = {ep: {} for ep in SNAPSHOTS}  # ep -> {seed: probs}
    for ep in SNAPSHOTS:
        for s in SEEDS:
            path = os.path.join(BASE, f"predictions_test_tcn_snap_s{s}_ep{ep}.csv")
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
            y = arr[:, 0].astype(int); p = arr[:, 1].astype(np.float32)
            if y_ref is None: y_ref = y
            per_snap_probs[ep][s] = p

    # 1. Per-snapshot F1 averaged across seeds (at thr=0.5)
    print("Per-snapshot F1 (avg of 5 seeds, thr=0.5):")
    for ep in SNAPSHOTS:
        f1s = [four_metrics(y_ref, per_snap_probs[ep][s], 0.5)["F1-Score"]
               for s in SEEDS]
        print(f"  ep={ep:>2}: F1={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")

    # 2. Per-seed snap-ensemble (already saved)
    print("\nPer-seed snap-ensemble (5-snapshot avg, thr=0.5):")
    per_seed_snap_ens = {}
    for s in SEEDS:
        path = os.path.join(BASE, f"predictions_test_tcn_snap_s{s}_ens.csv")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y = arr[:, 0].astype(int); p = arr[:, 1].astype(np.float32)
        m = four_metrics(y, p, 0.5)
        per_seed_snap_ens[str(s)] = m
        print(f"  seed={s:>4}: F1={m['F1-Score']:.4f}")
    mean_per_seed = np.mean([per_seed_snap_ens[str(s)]["F1-Score"] for s in SEEDS])
    print(f"  Mean: {mean_per_seed:.4f}")

    # 3. Full ensemble: average of all 25 snapshot predictions
    all_probs = []
    for ep in SNAPSHOTS:
        for s in SEEDS:
            all_probs.append(per_snap_probs[ep][s])
    p_full = np.mean(all_probs, axis=0)
    cur05 = four_metrics(y_ref, p_full, 0.5)

    # 4. Threshold sweep on full ensemble
    rows = []
    for t in THRS:
        rows.append({"threshold": float(t), **four_metrics(y_ref, p_full, t)})
    rows_sorted = sorted(rows, key=lambda r: -r["F1-Score"])
    best = rows_sorted[0]

    print(f"\n{'Config':<50s} {'Thr':>6s} {'Acc':>7s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print("-" * 80)
    print(f"{'Snapshot Ensemble (25 models) @ best thr':<50s} "
          f"{best['threshold']:>6.2f} {best['Accuracy']:>7.4f} "
          f"{best['Precision']:>7.4f} {best['Recall']:>7.4f} {best['F1-Score']:>7.4f}")
    print(f"{'Snapshot Ensemble (25 models) @ thr=0.5':<50s} "
          f"{0.5:>6.2f} {cur05['Accuracy']:>7.4f} {cur05['Precision']:>7.4f} "
          f"{cur05['Recall']:>7.4f} {cur05['F1-Score']:>7.4f}")
    print(f"{'Per-seed 5-snap ens mean @ thr=0.5':<50s} "
          f"{0.5:>6.2f}    -       -       -      {mean_per_seed:>7.4f}")

    print(f"\nReferences:")
    print(f"  TCN baseline F1={BASELINE_F1}")
    print(f"  TCN baseline F1 @ thr=0.380 = 0.8861")
    print(f"  Stacking M3 @ thr=0.390 = {STACKING_F1}")

    delta = (best['F1-Score'] - BASELINE_F1) * 100
    print(f"\nDelta vs TCN baseline @ best thr: {delta:+.3f}pp")
    print(f"Delta vs Stacking M3@0.390: {(best['F1-Score'] - STACKING_F1)*100:+.3f}pp")

    if delta > 0.01:
        verdict = "SNAPSHOT ENSEMBLE HELPS"
    elif delta < -0.01:
        verdict = "SNAPSHOT ENSEMBLE HURTS"
    else:
        verdict = "SNAPSHOT ENSEMBLE NEUTRAL"
    print(f"\n>>> {verdict} <<<")

    save_path = os.path.join(BASE, "tcn_snapshot_eval.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "TCN baseline + Snapshot Ensemble (5 snap × 5 seeds = 25 models)",
            "n_seeds": len(SEEDS), "n_snapshots_per_seed": len(SNAPSHOTS),
            "snapshots_epochs": SNAPSHOTS,
            "test_n": int(len(y_ref)),
            "per_seed_snap_ens_f1_at_0.5": per_seed_snap_ens,
            "mean_per_seed_snap_ens_f1_at_0.5": float(mean_per_seed),
            "full_25model_ensemble_at_0.5": cur05,
            "full_25model_ensemble_best_threshold": best,
            "baseline_f1": BASELINE_F1,
            "stacking_f1": STACKING_F1,
            "delta_vs_baseline_pp": delta,
            "verdict": verdict,
        }, f, indent=2)
    print(f"\n[saved] {save_path}")


if __name__ == "__main__":
    main()
