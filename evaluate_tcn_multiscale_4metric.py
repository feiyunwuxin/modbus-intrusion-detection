#!/usr/bin/env python3
"""Multi-Scale TCN 5-seed ensemble evaluation: 4 metrics + leaderboard.

Reads predictions_test_tcn_ms_s{seed}.csv (5 files), computes 5-seed prob_mean,
outputs:
  - multiscale_5seed_4metric.json (full data)
  - MULTISCALE_LEADERBOARD.md (human-readable)
  - multiscale_5seed_per_seed.csv (5 rows: 5 seeds × 4 metrics)
"""
import os
import csv
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

BASE = r"D:\workspace\claude\Issue\Issue"
SEEDS = [42, 123, 456, 789, 1024]
THR = 0.5
BASELINE_F1 = 0.8795
STACKING_F1 = 0.8904


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


def main():
    print("=== Multi-Scale TCN 5-seed × 4-metric Evaluation ===\n")
    per_seed = {}
    y_true_ref = None
    probs_dict = {}
    for s in SEEDS:
        path = os.path.join(BASE, f"predictions_test_tcn_ms_s{s}.csv")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y_true = arr[:, 0].astype(int); probs = arr[:, 1].astype(np.float32)
        if y_true_ref is None: y_true_ref = y_true
        probs_dict[s] = probs
        per_seed[str(s)] = {**four_metrics(y_true, (probs >= THR).astype(int)),
                            "CM_2x2": confusion_2x2(y_true, (probs >= THR).astype(int))}

    probs_matrix = np.stack([probs_dict[s] for s in sorted(probs_dict.keys())], axis=0)
    ens_probs = probs_matrix.mean(axis=0)
    ens_pred = (ens_probs >= THR).astype(int)
    ens_metrics = {**four_metrics(y_true_ref, ens_pred),
                   "CM_2x2": confusion_2x2(y_true_ref, ens_pred)}

    print(f"5-seed prob_mean ensemble:")
    print(f"  Acc={ens_metrics['Accuracy']:.4f}  P={ens_metrics['Precision']:.4f}  "
          f"R={ens_metrics['Recall']:.4f}  F1={ens_metrics['F1-Score']:.4f}")
    print(f"\nPer-seed F1:")
    for s in SEEDS:
        if s in probs_dict:
            print(f"  seed {s:>4}: F1={per_seed[str(s)]['F1-Score']:.4f}")

    diff_base = (ens_metrics['F1-Score'] - BASELINE_F1) * 100
    diff_stack = (ens_metrics['F1-Score'] - STACKING_F1) * 100
    target_met = ens_metrics['F1-Score'] >= 0.88
    print(f"\nvs TCN baseline (F1={BASELINE_F1}): {diff_base:+.2f}%")
    print(f"vs Stacking M3@0.390 (F1={STACKING_F1}): {diff_stack:+.2f}%")
    print(f"Target F1 >= 0.88: {'MET' if target_met else 'MISS'}")

    json_path = os.path.join(BASE, "multiscale_5seed_4metric.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"task": "Multi-Scale TCN 5-seed × 4-metric evaluation",
                   "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
                   "threshold": THR, "baseline_f1": BASELINE_F1,
                   "stacking_f1": STACKING_F1, "n_seeds": len(probs_dict),
                   "test_n": int(len(y_true_ref)), "per_seed": per_seed,
                   "ensemble_prob_mean": ens_metrics}, f, indent=2)

    csv_path = os.path.join(BASE, "multiscale_5seed_per_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "Accuracy", "Precision", "Recall", "F1-Score"])
        for s_str, m in per_seed.items():
            w.writerow([int(s_str), m["Accuracy"], m["Precision"],
                        m["Recall"], m["F1-Score"]])

    md_path = os.path.join(BASE, "MULTISCALE_LEADERBOARD.md")
    cm = ens_metrics["CM_2x2"]
    lines = [
        "# Multi-Scale TCN 5-Seed Leaderboard\n",
        "**Date**: 2026-09-16  ",
        "**Architecture**: 3 MultiScaleTCNBlocks × parallel kernels=[1,3,5,7] × dilations=[1,2,4]  ",
        "**Fusion**: Concat → 1x1 conv (ch=128→32) + residual  ",
        "**Reference**: TCN baseline F1=0.8795; Stacking M3@0.390 F1=0.8904\n",
        "## 1. 5-Seed 概率平均集成\n",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Accuracy | {ens_metrics['Accuracy']:.4f} |",
        f"| Precision | {ens_metrics['Precision']:.4f} |",
        f"| Recall | {ens_metrics['Recall']:.4f} |",
        f"| **F1-Score** | **{ens_metrics['F1-Score']:.4f}** |",
        "\n## 2. 关键对比\n",
        "| Comparison | Delta |",
        "|------------|------:|",
        f"| vs TCN baseline (F1={BASELINE_F1}) | {diff_base:+.2f}% |",
        f"| vs Stacking M3@0.390 (F1={STACKING_F1}) | {diff_stack:+.2f}% |",
        f"| Target F1 >= 0.88 | {'MET' if target_met else 'MISS'} |",
        "\n## 3. Per-Seed F1 详情\n",
        "| Seed | Accuracy | Precision | Recall | F1-Score |",
        "|-----:|---------:|----------:|-------:|---------:|",
    ]
    for s_str, m in per_seed.items():
        lines.append(f"| {s_str} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | "
                     f"{m['Recall']:.4f} | {m['F1-Score']:.4f} |")
    lines.append("\n## 4. 5-Seed 集成混淆矩阵 (thr=0.5)\n")
    lines.append(f"```\n              预测 Normal    预测 Attack\n"
                 f"实际 Normal      {cm[0][0]:>5}        {cm[0][1]:>5}\n"
                 f"实际 Attack      {cm[1][0]:>5}        {cm[1][1]:>5}\n```")
    lines.append("\n## 5. 关键发现\n")
    if target_met:
        lines.append(f"1. **目标达成**: ✅ F1={ens_metrics['F1-Score']:.4f} >= 0.88")
        lines.append(f"2. **vs TCN baseline**: {diff_base:+.2f}%")
        if ens_metrics['F1-Score'] >= STACKING_F1:
            lines.append(f"3. **vs Stacking M3**: {diff_stack:+.2f}% — **multi-scale 超越 stacking!**")
        else:
            lines.append(f"3. **vs Stacking M3**: {diff_stack:+.2f}% — multi-scale 略低但可独立部署")
    else:
        lines.append(f"1. **目标失败**: ❌ F1={ens_metrics['F1-Score']:.4f} < 0.88")
        lines.append(f"2. **vs TCN baseline**: {diff_base:+.2f}%")
        lines.append(f"3. **TCN 单架构 tuning 已穷尽**: SWA/EMA, augmentation, multi-scale 三轮失败")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[saved] {json_path}\n[saved] {csv_path}\n[saved] {md_path}")


if __name__ == "__main__":
    main()