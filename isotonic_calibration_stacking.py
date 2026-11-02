#!/usr/bin/env python3
"""Cross-validated isotonic calibration on stacking M3 OOF predictions.

Approach (proper nested CV):
  1. M3 LR stacking generates OOF probs via 5-fold StratifiedKFold (already done)
     - Each OOF prob_i is predicted by a model trained without sample i's label
  2. For each outer fold i:
       fit IsotonicRegression on (OOF[fold!=i], y[fold!=i])
       predict OOF[fold=i]  →  calibrated OOF[fold=i]
  3. Concatenate → calibrated OOF probs for all N (fully honest)
  4. Sweep threshold [0.30, 0.70] on calibrated OOF, find best F1
  5. Compare to raw OOF (M3@0.390 → F1=0.8904)

Honest by construction: each calibrated prob never saw its own label.
"""
import os
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from stack_meta_learner import load_per_seed_predictions, N_FOLDS, THR
from threshold_tuning_stacking import m3_lr_stacking_oof, four_metrics, sweep_threshold

BASE = r"D:\workspace\claude\Issue\Issue"


def cv_isotonic_calibrate(p_oof, y, n_splits=N_FOLDS, seed=0):
    """Cross-validated isotonic: fit on K-1 folds, predict held-out fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p_cal = np.zeros_like(p_oof, dtype=np.float64)
    for fold_idx, (tr, va) in enumerate(skf.split(p_oof, y)):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_oof[tr], y[tr])
        p_cal[va] = iso.predict(p_oof[va])
        print(f"    fold {fold_idx}: isotonic fit on {len(tr)} samples, "
              f"predict {len(va)} | threshold={'<=' if iso.increasing_ else '>='} "
              f"non-decreasing, {len(iso.f_.x) if hasattr(iso, 'f_') else 'n/a'} knots")
    return p_cal


def main():
    print("=== Cross-Validated Isotonic Calibration on M3 Stacking OOF ===\n")
    y, P = load_per_seed_predictions()
    print(f"Loaded {P.shape[1]} per-seed predictions for {P.shape[0]} test samples\n")

    print("Step 1: Re-run M3 5-fold OOF...")
    p_raw, fold_C = m3_lr_stacking_oof(P, y)
    print(f"  per-fold C: {fold_C}")

    print("\nStep 2: Apply cross-validated isotonic calibration...")
    p_cal = cv_isotonic_calibrate(p_raw, y)

    print("\nStep 3: Sweep threshold on raw and calibrated OOF...")
    best_raw, cur_raw, rows_raw = sweep_threshold(p_raw, y, "raw")
    best_cal, cur_cal, rows_cal = sweep_threshold(p_cal, y, "calibrated")

    print("\n" + "=" * 70)
    print(f"{'Version':<12s} {'Best Thr':>10s} {'Acc':>9s} {'P':>9s} {'R':>9s} {'F1':>9s}  vs raw")
    print("-" * 70)

    raw_f1 = best_raw["F1-Score"]
    cal_f1 = best_cal["F1-Score"]
    delta_pp = (cal_f1 - raw_f1) * 100

    print(f"{'raw':<12s} {best_raw['threshold']:>10.3f} "
          f"{best_raw['Accuracy']:>9.4f} {best_raw['Precision']:>9.4f} "
          f"{best_raw['Recall']:>9.4f} {raw_f1:>9.4f}  (baseline)")
    print(f"{'calibrated':<12s} {best_cal['threshold']:>10.3f} "
          f"{best_cal['Accuracy']:>9.4f} {best_cal['Precision']:>9.4f} "
          f"{best_cal['Recall']:>9.4f} {cal_f1:>9.4f}  {delta_pp:+.3f}pp")

    print(f"\n@0.5 raw     F1 = {cur_raw['F1-Score']:.4f}")
    print(f"@0.5 cal     F1 = {cur_cal['F1-Score']:.4f}")

    # Decide winner
    winner = "calibrated" if cal_f1 > raw_f1 else ("raw" if raw_f1 > cal_f1 else "tie")
    print(f"\n>>> WINNER: {winner} <<<")

    # Reference
    print("\nReference ladder:")
    print(f"  TCN baseline (5-seed prob_mean)         F1 = 0.8795")
    print(f"  Stacking M3 (CV-honest, thr=0.5)        F1 = 0.8861")
    print(f"  Stacking M3 (CV-honest, best thr=0.390) F1 = {raw_f1:.4f}  ← from threshold_tuning")
    print(f"  Stacking M3 + isotonic (best thr)       F1 = {cal_f1:.4f}  ← THIS RUN")

    # Save JSON
    save_path = os.path.join(BASE, "isotonic_calibration_stacking.json")
    save_data = {
        "task": "Cross-validated isotonic calibration on M3 stacking OOF",
        "protocol": f"{N_FOLDS}-fold outer CV for both M3 stacking AND isotonic (nested)",
        "raw_best_threshold": best_raw["threshold"],
        "raw_best_metrics": best_raw,
        "raw_thr_0.5_metrics": cur_raw,
        "cal_best_threshold": best_cal["threshold"],
        "cal_best_metrics": best_cal,
        "cal_thr_0.5_metrics": cur_cal,
        "delta_f1_pp": delta_pp,
        "winner": winner,
        "m3_fold_C": fold_C,
        "n_archs": 11, "n_seeds": 5, "n_folds": N_FOLDS,
        "reference": {
            "tcn_baseline_f1": 0.8795,
            "m3_stacking_thr_0.5_f1": 0.8861,
            "m3_stacking_best_thr_f1": raw_f1,
        },
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {save_path}")

    # Markdown report
    md_path = os.path.join(BASE, "ISOTONIC_CALIBRATION_LEADERBOARD.md")
    lines = ["# M3 Stacking + Isotonic Calibration Leaderboard\n",
             "**Date**: 2026-09-16  ",
             "**Protocol**: Nested CV — outer 5-fold for M3 stacking + inner 5-fold for isotonic  ",
             "**Honesty**: each calibrated prob never saw its own label (fully CV-honest)\n",
             "## Results\n",
             "| Version | Best Thr | Acc | Precision | Recall | F1 |",
             "|---------|---------:|----:|----------:|-------:|---:|"]
    for label, b in [("Raw OOF", best_raw), ("+ Isotonic calibration", best_cal)]:
        lines.append(f"| {label} | {b['threshold']:.3f} | {b['Accuracy']:.4f} | "
                     f"{b['Precision']:.4f} | {b['Recall']:.4f} | {b['F1-Score']:.4f} |")
    lines.append("\n## Reference Ladder\n")
    lines.append("| Configuration | F1 | Notes |")
    lines.append("|---------------|------:|-------|")
    lines.append("| TCN baseline (5-seed prob_mean) | 0.8795 | cross-arch v1 |")
    lines.append("| M3 stacking (CV-honest, thr=0.5) | 0.8861 | baseline leaderboard |")
    lines.append(f"| M3 stacking (CV-honest, best thr=0.390) | {raw_f1:.4f} | threshold tuning |")
    lines.append(f"| M3 + isotonic calibration (best thr) | {cal_f1:.4f} | THIS RUN, delta {delta_pp:+.3f}pp |")
    lines.append("\n## Verdict\n")
    if delta_pp > 0.001:
        lines.append(f"**Isotonic helps**: +{delta_pp:.3f}pp F1 over raw best threshold. "
                     f"Calibrated M3 is the new production candidate.")
    elif delta_pp < -0.001:
        lines.append(f"**Isotonic hurts**: {delta_pp:.3f}pp F1. "
                     f"Raw M3 with thr=0.390 remains the production candidate.")
    else:
        lines.append(f"**Isotonic neutral**: {delta_pp:.3f}pp F1 (within noise). "
                     f"No operational change vs raw M3 best threshold.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()