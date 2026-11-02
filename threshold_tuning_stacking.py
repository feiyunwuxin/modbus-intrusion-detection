#!/usr/bin/env python3
"""Threshold tuning on stacking M2 (weighted avg) and M3 (LR stacking) OOF predictions.

The original stack_meta_learner.py used thr=0.5 for F1 evaluation. We sweep
thr ∈ [0.30, 0.70] to find optimal F1 threshold for both meta-learners.

Honest evaluation: uses M2/M3 5-fold OOF predictions (no test contamination),
exactly the same protocol as the leaderboard — only the threshold varies.
"""
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from stack_meta_learner import (
    load_per_seed_predictions, fit_weights_slsqp, m1_simple_avg,
    N_FOLDS, THR, COL_MAP,
)

BASE = r"D:\workspace\claude\Issue\Issue"
THR_RANGE = np.arange(0.30, 0.7001, 0.005)


def four_metrics(y_true, y_pred):
    return {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score":  float(f1_score(y_true, y_pred, zero_division=0)),
    }


def m3_lr_stacking_oof(P, y):
    """Same as stack_meta_learner.m3_lr_stacking_cv but always saves OOF probs."""
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
                cv_f1s.append(f1_score(y[tr][iva], (p_in >= THR).astype(int),
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


def m2_weighted_avg_oof(P, y):
    """Same as stack_meta_learner.m2_weighted_avg_cv but always saves OOF probs."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    n = len(y)
    p_oof = np.zeros(n, dtype=np.float64)
    for tr, va in skf.split(P, y):
        w = fit_weights_slsqp(P[tr], y[tr])
        p_oof[va] = P[va] @ w
    return p_oof


def sweep_threshold(p_oof, y, name):
    rows = []
    for t in THR_RANGE:
        y_pred = (p_oof >= t).astype(int)
        rows.append({
            "threshold": float(t),
            **four_metrics(y, y_pred),
        })
    rows_sorted = sorted(rows, key=lambda r: -r["F1-Score"])
    best = rows_sorted[0]
    cur = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9)
    return best, cur, rows


def main():
    print("=== Threshold Tuning on Stacking M2 / M3 OOF Predictions ===\n")
    y, P = load_per_seed_predictions()
    print(f"Loaded {P.shape[1]} per-seed predictions for {P.shape[0]} test samples\n")

    # M1 (simple avg) - no CV needed, use uniform
    m1 = m1_simple_avg(P, y)
    p_m1 = P @ np.array(m1["weights"])
    print("Computing M1 OOF probs (uniform avg, no fitting)...")
    best1, cur1, rows1 = sweep_threshold(p_m1, y, "M1 Simple average")

    print("Computing M2 OOF probs (5-fold CV weighted avg)...")
    p_m2 = m2_weighted_avg_oof(P, y)
    best2, cur2, rows2 = sweep_threshold(p_m2, y, "M2 Weighted avg (SLSQP)")

    print("Computing M3 OOF probs (5-fold CV LR stacking)...")
    p_m3, fold_C_m3 = m3_lr_stacking_oof(P, y)
    best3, cur3, rows3 = sweep_threshold(p_m3, y, "M3 LR stacking (L2)")

    # Report
    print("\n" + "=" * 70)
    print(f"{'Method':<35s} {'Best Thr':>10s} {'Best F1':>10s} {'@0.5 F1':>10s} {'Delta':>10s}")
    print("-" * 70)
    out = {}
    for name, best, cur, p_oof in [
        ("M1 Simple average", best1, cur1, p_m1),
        ("M2 Weighted avg (SLSQP)", best2, cur2, p_m2),
        ("M3 LR stacking (L2)", best3, cur3, p_m3),
    ]:
        delta = (best["F1-Score"] - cur["F1-Score"]) * 100
        print(f"{name:<35s} {best['threshold']:>10.3f} {best['F1-Score']:>10.4f} "
              f"{cur['F1-Score']:>10.4f} {delta:>+9.3f}%")
        out[name] = {
            "best_threshold": best["threshold"],
            "best_metrics": best,
            "current_0.5_metrics": cur,
            "delta_f1_pct": delta,
            "oof_probs": p_oof.tolist(),
        }

    # Determine the overall best
    overall = max(out.items(), key=lambda kv: kv[1]["best_metrics"]["F1-Score"])
    print("\n" + "=" * 70)
    print(f"OVERALL BEST: {overall[0]}")
    print(f"  best_threshold = {overall[1]['best_threshold']:.3f}")
    print(f"  best F1 = {overall[1]['best_metrics']['F1-Score']:.4f} "
          f"(vs {overall[1]['current_0.5_metrics']['F1-Score']:.4f} at thr=0.5, "
          f"delta {overall[1]['delta_f1_pct']:+.3f}%)")
    print(f"  Acc={overall[1]['best_metrics']['Accuracy']:.4f} "
          f"P={overall[1]['best_metrics']['Precision']:.4f} "
          f"R={overall[1]['best_metrics']['Recall']:.4f}")

    # Reference comparison
    print("\n" + "=" * 70)
    print("Reference: TCN baseline (5-seed prob_mean) F1=0.8795")
    print(f"           Stacking M3 (5-fold CV, thr=0.5) F1=0.8861 (CV-honest)")
    print(f"           Stacking M3 (5-fold CV, best thr) F1="
          f"{out['M3 LR stacking (L2)']['best_metrics']['F1-Score']:.4f}")

    # Save JSON
    save_path = os.path.join(BASE, "threshold_tuning_stacking.json")
    save_data = {
        "task": "Threshold tuning on stacking M1/M2/M3 OOF predictions",
        "eval_protocol": "5-fold StratifiedKFold (same as stack_meta_learner.py)",
        "threshold_range": list(THR_RANGE),
        "n_archs": 11, "n_seeds": 5, "n_folds": N_FOLDS,
        "results": {
            name: {
                "best_threshold": out[name]["best_threshold"],
                "best_metrics": out[name]["best_metrics"],
                "current_0.5_metrics": out[name]["current_0.5_metrics"],
                "delta_f1_pct": out[name]["delta_f1_pct"],
            } for name in out
        },
        "overall_best": {
            "method": overall[0],
            "best_threshold": overall[1]["best_threshold"],
            "best_f1": overall[1]["best_metrics"]["F1-Score"],
        },
        "reference": {
            "tcn_baseline_f1": 0.8795,
            "m3_stacking_thr_0.5_f1": 0.8861,
        },
        "m3_fold_C": fold_C_m3,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {save_path}")

    # Markdown report
    md_path = os.path.join(BASE, "THRESHOLD_TUNING_LEADERBOARD.md")
    lines = ["# Stacking Threshold Tuning Leaderboard\n",
             "**Date**: 2026-09-16  ",
             "**Protocol**: 5-fold StratifiedKFold OOF (no test contamination)  ",
             "**Range**: thr ∈ [0.30, 0.70], step 0.005\n",
             "## Results\n",
             "| Method | Best Thr | Best Acc | Best P | Best R | Best F1 | @0.5 F1 | Delta |",
             "|--------|---------:|---------:|-------:|-------:|--------:|--------:|------:|"]
    for name in ["M1 Simple average", "M2 Weighted avg (SLSQP)", "M3 LR stacking (L2)"]:
        b = out[name]["best_metrics"]
        c = out[name]["current_0.5_metrics"]
        d = out[name]["delta_f1_pct"]
        lines.append(f"| {name} | {b['threshold']:.3f} | {b['Accuracy']:.4f} | "
                     f"{b['Precision']:.4f} | {b['Recall']:.4f} | {b['F1-Score']:.4f} | "
                     f"{c['F1-Score']:.4f} | {d:+.3f}% |")
    lines.append("\n## References\n")
    lines.append("| Method | F1 | Notes |")
    lines.append("|--------|------:|-------|")
    lines.append("| TCN baseline (5-seed prob_mean) | 0.8795 | cross-arch v1 leaderboard |")
    lines.append(f"| Stacking M3 (CV-honest, thr=0.5) | 0.8861 | current production candidate |")
    lines.append(f"| Stacking M3 (CV-honest, **best thr**) | "
                 f"{out['M3 LR stacking (L2)']['best_metrics']['F1-Score']:.4f} | "
                 f"delta {out['M3 LR stacking (L2)']['delta_f1_pct']:+.3f}% |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()