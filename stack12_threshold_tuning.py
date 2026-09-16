#!/usr/bin/env python3
"""Stack12: add XGBoost as 12th base learner + threshold tuning.

Same protocol as stack_meta_learner.py (5-fold CV honest stacking) but with
XGBoost added → 12 archs × 5 seeds = 60 per-seed columns.

Workflow:
  1. Load 60 per-seed test predictions (11 originals + XGBoost)
  2. Re-run M3 LR stacking (5-fold CV)
  3. Sweep threshold [0.30, 0.70] on OOF
  4. Report best F1 + delta vs 11-arch baseline (F1=0.8904)
"""
import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import stack_meta_learner as sml
from threshold_tuning_stacking import four_metrics, THR_RANGE

BASE = r"D:\workspace\claude\Issue\Issue"


# 12 base learners (added XGBoost at end)
ARCHS_12 = sml.ARCHS + [
    ("xgb",        "XGBoost",                  "xgb_23dim_w16"),
]
SEEDS = sml.SEEDS
N_FOLDS = sml.N_FOLDS


def load_per_seed_predictions_12():
    """Load 60 per-seed test predictions (11 originals + XGBoost)."""
    rows = []
    y_true_ref = None
    for s in SEEDS:
        for short, disp, tag in ARCHS_12:
            path = os.path.join(BASE, f"predictions_test_{tag}_s{s}.csv")
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
            if y_true_ref is None:
                y_true_ref = arr[:, 0].astype(int)
            else:
                assert np.array_equal(y_true_ref, arr[:, 0].astype(int)), \
                    f"seed {s} {tag} y_true mismatch"
            rows.append(arr[:, 1].astype(np.float32))
    return y_true_ref, np.stack(rows, axis=1)


def m3_lr_stacking_oof(P, y):
    """Same as threshold_tuning_stacking.m3_lr_stacking_oof (re-imported for isolation)."""
    from sklearn.linear_model import LogisticRegression
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    n = len(y)
    p_oof = np.zeros(n, dtype=np.float64)
    fold_C = []
    for fold_idx, (tr, va) in enumerate(skf.split(P, y)):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_idx)
        Cs = [0.001, 0.01, 0.1, 1.0, 10.0]
        best_C, best_inner_f1 = None, -np.inf
        for C in Cs:
            cv_f1s = []
            for itr, iva in inner.split(P[tr], y[tr]):
                m = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                       max_iter=2000, random_state=0)
                m.fit(P[tr][itr], y[tr][itr])
                p_in = m.predict_proba(P[tr][iva])[:, 1]
                cv_f1s.append(f1_score(y[tr][iva], (p_in >= 0.5).astype(int),
                                       zero_division=0))
            mean_f1 = np.mean(cv_f1s)
            if mean_f1 > best_inner_f1:
                best_inner_f1, best_C = mean_f1, C
        fold_C.append(float(best_C))
        m = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs",
                               max_iter=2000, random_state=0)
        m.fit(P[tr], y[tr])
        p_oof[va] = m.predict_proba(P[va])[:, 1]
    return p_oof, fold_C


def sweep_threshold(p_oof, y):
    rows = []
    for t in THR_RANGE:
        y_pred = (p_oof >= t).astype(int)
        rows.append({"threshold": float(t), **four_metrics(y, y_pred)})
    rows_sorted = sorted(rows, key=lambda r: -r["F1-Score"])
    return rows_sorted[0], next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9), rows


def main():
    print("=== Stack12 + Threshold Tuning (XGBoost added as 12th base learner) ===\n")
    y, P = load_per_seed_predictions_12()
    print(f"Loaded {P.shape[1]} per-seed predictions for {P.shape[0]} test samples  "
          f"(12 archs × 5 seeds = 60 cols)\n")

    # Run M3 5-fold OOF
    print("Running M3 LR stacking (5-fold CV)...")
    p_oof, fold_C = m3_lr_stacking_oof(P, y)
    print(f"  per-fold C: {fold_C}")

    # Sweep threshold
    best, cur, _ = sweep_threshold(p_oof, y)

    print("\n" + "=" * 70)
    print(f"{'Config':<35s} {'Thr':>8s} {'Acc':>9s} {'P':>9s} {'R':>9s} {'F1':>9s}")
    print("-" * 70)
    print(f"{'Stack12 M3 best threshold':<35s} {best['threshold']:>8.3f} "
          f"{best['Accuracy']:>9.4f} {best['Precision']:>9.4f} "
          f"{best['Recall']:>9.4f} {best['F1-Score']:>9.4f}")
    print(f"{'Stack12 M3 @0.5':<35s} {0.500:>8.3f} "
          f"{cur['Accuracy']:>9.4f} {cur['Precision']:>9.4f} "
          f"{cur['Recall']:>9.4f} {cur['F1-Score']:>9.4f}")
    print()
    print(f"Reference: Stack11 M3 @ thr=0.390: F1=0.8904  (previous best)")
    print(f"           Stack11 M3 @ thr=0.5:   F1=0.8861")
    print(f"           Stack12 M3 @ thr={best['threshold']:.3f}: F1={best['F1-Score']:.4f}  (THIS RUN)")

    delta_pp = (best["F1-Score"] - 0.8904) * 100
    print(f"\nDelta vs Stack11 M3@0.390: {delta_pp:+.3f}pp")

    if delta_pp > 0.005:
        print(">>> XGBoost helps; new production candidate <<<")
    elif delta_pp < -0.005:
        print(">>> XGBoost hurts; Stack11 M3@0.390 remains production <<<")
    else:
        print(">>> XGBoost neutral (within noise) <<<")

    # Save JSON
    save_path = os.path.join(BASE, "stack12_threshold_tuning.json")
    save_data = {
        "task": "Stack12 (11 original + XGBoost) M3 + threshold tuning",
        "eval_protocol": f"{N_FOLDS}-fold StratifiedKFold (CV-honest)",
        "n_archs": 12, "n_seeds": 5, "n_folds": N_FOLDS,
        "m3_fold_C": fold_C,
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "thr_0.5_metrics": cur,
        "delta_vs_stack11_pp": delta_pp,
        "reference": {
            "stack11_m3_thr_0.5_f1": 0.8861,
            "stack11_m3_thr_0.390_f1": 0.8904,
            "stack12_m3_best_f1": best["F1-Score"],
        },
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {save_path}")

    # Markdown report
    md_path = os.path.join(BASE, "STACK12_LEADERBOARD.md")
    lines = ["# Stack12 (XGBoost-added) M3 Threshold Tuning\n",
             "**Date**: 2026-09-16  ",
             "**Protocol**: 5-fold StratifiedKFold (CV-honest)  ",
             "**Base learners**: 12 (11 original + XGBoost) → 60 per-seed columns  \n",
             "## Results\n",
             "| Config | Threshold | Acc | P | R | F1 |",
             "|--------|----------:|----:|--:|--:|--:|"]
    lines.append(f"| Stack12 M3 (best thr) | {best['threshold']:.3f} | {best['Accuracy']:.4f} | "
                 f"{best['Precision']:.4f} | {best['Recall']:.4f} | **{best['F1-Score']:.4f}** |")
    lines.append(f"| Stack12 M3 (thr=0.5) | 0.500 | {cur['Accuracy']:.4f} | "
                 f"{cur['Precision']:.4f} | {cur['Recall']:.4f} | {cur['F1-Score']:.4f} |")
    lines.append("\n## Reference Ladder\n")
    lines.append("| Configuration | F1 | Notes |")
    lines.append("|---------------|----:|-------|")
    lines.append("| TCN baseline (5-seed prob_mean) | 0.8795 | cross-arch v1 |")
    lines.append("| Stack11 M3 (CV-honest, thr=0.5) | 0.8861 | baseline leaderboard |")
    lines.append("| Stack11 M3 (CV-honest, thr=0.390) | 0.8904 | threshold tuning Phase 5 |")
    lines.append(f"| **Stack12 M3 (best thr={best['threshold']:.3f})** | **{best['F1-Score']:.4f}** | "
                 f"XGBoost added, delta {delta_pp:+.3f}pp |")
    lines.append("\n## Verdict\n")
    if delta_pp > 0.005:
        lines.append(f"**XGBoost improves stacking by {delta_pp:.3f}pp.** "
                     f"New production baseline: Stack12 M3 @ thr={best['threshold']:.3f}, F1={best['F1-Score']:.4f}.")
    elif delta_pp < -0.005:
        lines.append(f"**XGBoost hurts stacking by {delta_pp:.3f}pp.** "
                     f"Stack11 M3 @ thr=0.390 (F1=0.8904) remains production.")
    else:
        lines.append(f"**XGBoost neutral ({delta_pp:+.3f}pp).** No change to production candidate.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()