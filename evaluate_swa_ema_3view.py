#!/usr/bin/env python3
"""Evaluate 3 views (raw / SWA / EMA) × 5 seeds on TCN + SWA/EMA.

Reads predictions_test_tcn_{view}_s{seed}.csv for each combination,
computes 5-seed prob_mean ensemble metrics per view, outputs:
  - swa_ema_5seed_3view.json (full data)
  - SWA_EMA_LEADERBOARD.md (human-readable)
  - swa_ema_5seed_per_view.csv (15-row flat table)
"""
import os
import csv
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

BASE = r"D:\workspace\claude\Issue\Issue"
VIEWS = ["raw", "swa", "ema"]
VIEW_DISPLAY = {"raw": "TCN baseline (raw best)", "swa": "SWA (epoch ≥10 avg + BN re-update)", "ema": "EMA (α=0.999)"}
SEEDS = [42, 123, 456, 789, 1024]
THR = 0.5
BASELINE_F1 = 0.8795  # TCN 5-seed prob_mean F1 (cross-arch v1)


def four_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_2x2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1))); tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1))); fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return [[tn, fp], [fn, tp]]


def evaluate_view(view):
    per_seed = {}
    y_true_ref = None
    probs_dict = {}
    for s in SEEDS:
        path = os.path.join(BASE, f"predictions_test_tcn_{view}_s{s}.csv")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y_true = arr[:, 0].astype(int); probs = arr[:, 1].astype(np.float32)
        if y_true_ref is None: y_true_ref = y_true
        probs_dict[s] = probs
        per_seed[str(s)] = {**four_metrics(y_true, (probs >= THR).astype(int)),
                            "CM_2x2": confusion_2x2(y_true, (probs >= THR).astype(int))}

    seeds_avail = sorted(probs_dict.keys())
    probs_matrix = np.stack([probs_dict[s] for s in seeds_avail], axis=0)
    ens_probs = probs_matrix.mean(axis=0)
    ens_pred = (ens_probs >= THR).astype(int)
    ens_metrics = {**four_metrics(y_true_ref, ens_pred),
                   "CM_2x2": confusion_2x2(y_true_ref, ens_pred)}

    print(f"\n=== {VIEW_DISPLAY[view]} ===")
    print(f"  5-seed prob_mean: Acc={ens_metrics['Accuracy']:.4f}  P={ens_metrics['Precision']:.4f}  "
          f"R={ens_metrics['Recall']:.4f}  F1={ens_metrics['F1-Score']:.4f}")
    for s in SEEDS:
        if s in probs_dict:
            m = per_seed[str(s)]
            print(f"  seed {s:>4}: F1={m['F1-Score']:.4f}")

    return {"view": view, "display_name": VIEW_DISPLAY[view], "n_seeds": len(seeds_avail),
            "test_n": int(len(y_true_ref)), "threshold": THR, "per_seed": per_seed,
            "ensemble_prob_mean": ens_metrics}


def write_outputs(results):
    json_path = os.path.join(BASE, "swa_ema_5seed_3view.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"task": "TCN + SWA + EMA 3-view × 5-seed × 4-metric",
                   "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
                   "threshold": THR, "baseline_f1": BASELINE_F1, "views": results}, f, indent=2)

    csv_path = os.path.join(BASE, "swa_ema_5seed_per_view.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["view", "seed", "Accuracy", "Precision", "Recall", "F1-Score"])
        for r in results:
            for s_str, m in r["per_seed"].items():
                w.writerow([r["view"], int(s_str), m["Accuracy"], m["Precision"],
                            m["Recall"], m["F1-Score"]])

    md_path = os.path.join(BASE, "SWA_EMA_LEADERBOARD.md")
    lines = ["# TCN + SWA + EMA 3-View Leaderboard\n",
             "**生成时间**: 2026-09-16  ",
             "**基线**: TCN (no SE) 5-seed prob_mean F1=0.8795  ",
             "**目标**: 任一 view F1 ≥ 0.88\n",
             "## 1. 5-Seed 概率平均集成（3 views × 4 metrics）\n",
             "| Rank | View | Accuracy | Precision | Recall | F1-Score | vs baseline |",
             "|------|------|---------:|----------:|-------:|---------:|-----------:|"]
    sorted_results = sorted(results, key=lambda r: -r["ensemble_prob_mean"]["F1-Score"])
    for i, r in enumerate(sorted_results, 1):
        m = r["ensemble_prob_mean"]
        diff = (m["F1-Score"] - BASELINE_F1) * 100
        marker = "🎯" if m["F1-Score"] >= 0.88 else ""
        lines.append(f"| {i} | {r['display_name']} | {m['Accuracy']:.4f} | "
                     f"{m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1-Score']:.4f} | "
                     f"{diff:+.2f}% {marker} |")

    lines.append("\n## 2. Per-Seed F1 详情\n")
    for r in sorted_results:
        lines.append(f"\n### {r['display_name']}\n")
        lines.append("| Seed | Accuracy | Precision | Recall | F1-Score |")
        lines.append("|-----:|---------:|----------:|-------:|---------:|")
        for s_str, m in r["per_seed"].items():
            lines.append(f"| {s_str} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | "
                         f"{m['Recall']:.4f} | {m['F1-Score']:.4f} |")
        cm = r["ensemble_prob_mean"]["CM_2x2"]
        lines.append(f"\n**5-seed 集成混淆矩阵** (thr={THR}):  ")
        lines.append(f"```\n              预测 Normal    预测 Attack\n"
                     f"实际 Normal      {cm[0][0]:>5}        {cm[0][1]:>5}\n"
                     f"实际 Attack      {cm[1][0]:>5}        {cm[1][1]:>5}\n```")

    lines.append("\n## 3. 关键发现\n")
    best = sorted_results[0]
    improvement = (best["ensemble_prob_mean"]["F1-Score"] - BASELINE_F1) * 100
    lines.append(f"1. **最佳 view**: {best['display_name']} — 5-seed 集成 F1 = {best['ensemble_prob_mean']['F1-Score']:.4f}")
    lines.append(f"2. **vs TCN 基线** (F1={BASELINE_F1}): {improvement:+.2f}%")
    target_met = best["ensemble_prob_mean"]["F1-Score"] >= 0.88
    lines.append(f"3. **目标达成**: {'✅ 是' if target_met else '❌ 否'} (F1 ≥ 0.88)")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[saved] {json_path}\n[saved] {csv_path}\n[saved] {md_path}")


def main():
    print("=== TCN + SWA + EMA 3-View × 5-Seed × 4-Metric Evaluation ===\n")
    results = []
    for view in VIEWS:
        results.append(evaluate_view(view))
    write_outputs(results)


if __name__ == "__main__":
    main()