#!/usr/bin/env python3
"""5-seed prob_mean ensemble + threshold sweep for TCN pos_weight=2.0.

Reads predictions_test_tcn_pw2_s{seed}.csv (5 files).
Outputs F1 at thr=0.5 and best-F1 from threshold sweep in [0.30, 0.70].
"""
import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

BASE = r"D:\workspace\claude\Issue\Issue"
SEEDS = [42, 123, 456, 789, 1024]
THRS = np.round(np.arange(0.30, 0.71, 0.01), 2)
BASELINE_F1 = 0.8795
STACKING_F1 = 0.8904
TAG = "tcn_pw2"


def four_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():
    print(f"=== TCN pos_weight=2.0 — 5-seed ensemble + threshold sweep ===\n")

    y_ref = None
    probs_dict = {}
    per_seed_at_05 = {}
    for s in SEEDS:
        path = os.path.join(BASE, f"predictions_test_{TAG}_s{s}.csv")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y = arr[:, 0].astype(int); p = arr[:, 1].astype(np.float32)
        if y_ref is None: y_ref = y
        probs_dict[s] = p
        per_seed_at_05[str(s)] = {**four_metrics(y, (p >= 0.5).astype(int))}

    seeds_ok = sorted(probs_dict.keys())
    p_ens = np.mean([probs_dict[s] for s in seeds_ok], axis=0)

    # per-seed F1 mean at thr=0.5
    mean_f1_05 = float(np.mean([per_seed_at_05[str(s)]["F1-Score"] for s in seeds_ok]))

    # threshold sweep on ensemble
    rows = []
    for t in THRS:
        y_pred = (p_ens >= t).astype(int)
        rows.append({"threshold": float(t), **four_metrics(y_ref, y_pred)})
    rows_sorted = sorted(rows, key=lambda r: -r["F1-Score"])
    best = rows_sorted[0]
    cur05 = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9)

    print(f"{'Config':<45s} {'Thr':>6s} {'Acc':>7s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print("-" * 80)
    print(f"{'TCN+pw2 ensemble @ best thr':<45s} {best['threshold']:>6.2f} "
          f"{best['Accuracy']:>7.4f} {best['Precision']:>7.4f} "
          f"{best['Recall']:>7.4f} {best['F1-Score']:>7.4f}")
    print(f"{'TCN+pw2 ensemble @ thr=0.5':<45s} {0.5:>6.2f} "
          f"{cur05['Accuracy']:>7.4f} {cur05['Precision']:>7.4f} "
          f"{cur05['Recall']:>7.4f} {cur05['F1-Score']:>7.4f}")
    print(f"{'TCN+pw2 mean-of-per-seed F1 @ thr=0.5':<45s} {0.5:>6.2f}  -      -      -      {mean_f1_05:>7.4f}")

    print(f"\nReferences:")
    print(f"  TCN baseline F1={BASELINE_F1}")
    print(f"  TCN baseline F1 @ thr=0.380 = 0.8861")
    print(f"  Stacking M3 @ thr=0.390 = {STACKING_F1}")

    delta_base = (best['F1-Score'] - BASELINE_F1) * 100
    delta_stack = (best['F1-Score'] - STACKING_F1) * 100
    print(f"\nDelta vs TCN baseline @ thr=0.5: {(mean_f1_05 - BASELINE_F1)*100:+.3f}pp")
    print(f"Delta vs TCN baseline @ best thr: {delta_base:+.3f}pp")
    print(f"Delta vs Stacking M3@0.390: {delta_stack:+.3f}pp")

    if delta_base > 0.01:
        verdict = "POS_WEIGHT HELPS"
    elif delta_base < -0.01:
        verdict = "POS_WEIGHT HURTS"
    else:
        verdict = "POS_WEIGHT NEUTRAL"
    print(f"\n>>> {verdict} TCN baseline <<<")

    save_path = os.path.join(BASE, "tcn_posweight2_eval.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "TCN baseline + pos_weight=2.0, 5-seed ensemble + threshold sweep",
            "n_seeds": len(seeds_ok), "test_n": int(len(y_ref)),
            "per_seed_at_0.5": per_seed_at_05,
            "mean_per_seed_f1_at_0.5": mean_f1_05,
            "ensemble_at_0.5": cur05,
            "ensemble_best_threshold": best,
            "baseline_f1": BASELINE_F1,
            "stacking_f1": STACKING_F1,
            "delta_vs_baseline_pp": delta_base,
            "verdict": verdict,
        }, f, indent=2)
    print(f"\n[saved] {save_path}")


if __name__ == "__main__":
    main()
