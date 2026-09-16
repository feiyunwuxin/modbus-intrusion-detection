#!/usr/bin/env python3
"""Evaluate 5 architectures × 5 seeds on 4 metrics (Acc / Precision / Recall / F1).

Reads predictions_test_<arch_tag>_s{seed}.csv for each combination,
computes 4 metrics under three views (single seed, 5-seed prob mean ensemble,
5-seed majority vote), and outputs:
  - cross_arch_5seed_4metrics.json (full data)
  - CROSS_ARCH_LEADERBOARD_4METRICS.md (human report)
  - cross_arch_5seed_4metrics_per_seed.csv (25-row flat table)
"""
import os
import csv
import json
import numpy as np

BASE = r"D:\workspace\claude\Issue\Issue"
ARCHS = [
    # Original 5-arch cross-arch (2026-09-16)
    ("tcn_se",  "TCN+SE", "v4_se_23dim_b64_ch32_do01_window16"),
    ("tcn",     "TCN (no SE)", "tcn_23dim_w16"),
    ("lstm",    "BiLSTM",  "lstm_23dim_w16"),
    ("gru",     "BiGRU",   "gru_23dim_w16"),
    ("cnnlstm", "CNN-LSTM","cnnlstm_23dim_w16"),
    # v2 expansion (2026-09-16 + later): pure CNN + scaled hidden/channels
    ("cnn",     "Pure CNN",       "cnn_23dim_w16"),
    ("lstm_v2", "BiLSTM (h128)",  "lstm_23dim_w16_h128"),
    ("gru_v2",  "BiGRU (h128)",   "gru_23dim_w16_h128"),
    ("cnnlstm_v2", "CNN-LSTM (ch256,h128)", "cnnlstm_23dim_w16_ch256"),
]
SEEDS = [42, 123, 456, 789, 1024]
THR = 0.5


def confusion_2x2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return [[tn, fp], [fn, tp]]


def four_metrics(y_true, y_pred):
    """4 metrics under Binary / Positive-class (Attack=positive) convention."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1),
        "CM_2x2": [[tn, fp], [fn, tp]],
    }


def load_arch_predictions(arch_tag: str):
    """Return (y_true_array, {seed: probs_array})."""
    probs_dict = {}
    y_true_ref = None
    for s in SEEDS:
        path = os.path.join(BASE, f"predictions_test_{arch_tag}_s{s}.csv")
        if not os.path.exists(path):
            print(f"  [WARN] missing {path}")
            continue
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y_true = arr[:, 0].astype(int)
        probs = arr[:, 1].astype(np.float32)
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.array_equal(y_true_ref, y_true), f"seed {s} y_true mismatch"
        probs_dict[s] = probs
    return y_true_ref, probs_dict


def evaluate_one_arch(short_name: str, display_name: str, arch_tag: str):
    print(f"\n=== {display_name} ({arch_tag}) ===")
    y_true, probs_dict = load_arch_predictions(arch_tag)
    if y_true is None or len(probs_dict) == 0:
        return None

    per_seed = {}
    for s in SEEDS:
        if s not in probs_dict:
            continue
        y_pred = (probs_dict[s] >= THR).astype(int)
        per_seed[str(s)] = four_metrics(y_true, y_pred)
        m = per_seed[str(s)]
        print(f"  seed {s:>4}: Acc={m['Accuracy']:.4f}  P={m['Precision']:.4f}  "
              f"R={m['Recall']:.4f}  F1={m['F1-Score']:.4f}")

    # 5-seed probability mean ensemble
    seeds_avail = sorted(probs_dict.keys())
    probs_matrix = np.stack([probs_dict[s] for s in seeds_avail], axis=0)
    ens_probs = probs_matrix.mean(axis=0)
    ens_pred = (ens_probs >= THR).astype(int)
    ens_metrics = four_metrics(y_true, ens_pred)
    print(f"  5-seed prob_mean: Acc={ens_metrics['Accuracy']:.4f}  P={ens_metrics['Precision']:.4f}  "
          f"R={ens_metrics['Recall']:.4f}  F1={ens_metrics['F1-Score']:.4f}")

    # 5-seed majority vote
    votes = (probs_matrix >= THR).astype(int)
    maj_pred = (votes.sum(axis=0) >= (len(seeds_avail) / 2 + 0.5)).astype(int)
    maj_metrics = four_metrics(y_true, maj_pred)
    print(f"  5-seed majority : Acc={maj_metrics['Accuracy']:.4f}  P={maj_metrics['Precision']:.4f}  "
          f"R={maj_metrics['Recall']:.4f}  F1={maj_metrics['F1-Score']:.4f}")

    return {
        "short_name": short_name,
        "display_name": display_name,
        "tag": arch_tag,
        "n_seeds": len(seeds_avail),
        "test_n": int(len(y_true)),
        "threshold": THR,
        "per_seed": per_seed,
        "ensemble_prob_mean": ens_metrics,
        "ensemble_majority_vote": maj_metrics,
    }


def write_json_report(results: list, path: str):
    out = {
        "task": "Cross-architecture 5-arch × 5-seed × 4-metric evaluation",
        "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "convention": "Binary / Positive-class (Attack=positive)",
        "threshold": THR,
        "archs": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def write_per_seed_csv(results: list, path: str):
    """25-row flat table: arch × seed × 4 metrics."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "display_name", "seed", "Accuracy", "Precision", "Recall", "F1-Score"])
        for r in results:
            for s_str, m in r["per_seed"].items():
                w.writerow([r["short_name"], r["display_name"], int(s_str),
                            m["Accuracy"], m["Precision"], m["Recall"], m["F1-Score"]])


def write_markdown_report(results: list, path: str):
    """Human-readable leaderboard with 5-seed ensemble + per-seed mean."""
    lines = []
    lines.append("# Cross-Architecture 5-Seed × 4-Metric Leaderboard\n")
    lines.append("**生成时间**: 2026-09-15  ")
    lines.append("**数据集**: 23-dim × window=16 (IanArffDataset v2)  ")
    lines.append("**测试集大小**: N = 3432 (Normal = 1627, Attack = 1805)  ")
    lines.append("**阈值**: 0.5  ")
    lines.append("**指标口径**: Binary / Positive-class (Attack = positive)\n")

    lines.append("## 1. 5-Seed 概率平均集成（推荐）\n")
    lines.append("| Rank | Architecture | Accuracy | Precision | Recall | F1-Score |")
    lines.append("|------|--------------|---------:|----------:|-------:|---------:|")
    sorted_results = sorted(results, key=lambda r: -r["ensemble_prob_mean"]["F1-Score"])
    for i, r in enumerate(sorted_results, 1):
        m = r["ensemble_prob_mean"]
        lines.append(f"| {i} | {r['display_name']} | {m['Accuracy']:.4f} | "
                     f"{m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1-Score']:.4f} |")
    lines.append("")

    lines.append("## 2. 5-Seed 多数投票集成\n")
    lines.append("| Architecture | Accuracy | Precision | Recall | F1-Score |")
    lines.append("|--------------|---------:|----------:|-------:|---------:|")
    for r in sorted_results:
        m = r["ensemble_majority_vote"]
        lines.append(f"| {r['display_name']} | {m['Accuracy']:.4f} | "
                     f"{m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1-Score']:.4f} |")
    lines.append("")

    lines.append("## 3. 单 Seed 平均（Mean ± Std over 5 seeds）\n")
    lines.append("| Architecture | Acc (mean±std) | P (mean±std) | R (mean±std) | F1 (mean±std) |")
    lines.append("|--------------|----------------|--------------|--------------|---------------|")
    for r in sorted_results:
        accs = [m["Accuracy"] for m in r["per_seed"].values()]
        ps = [m["Precision"] for m in r["per_seed"].values()]
        rs = [m["Recall"] for m in r["per_seed"].values()]
        f1s = [m["F1-Score"] for m in r["per_seed"].values()]
        lines.append(f"| {r['display_name']} | "
                     f"{np.mean(accs):.4f}±{np.std(accs, ddof=0):.4f} | "
                     f"{np.mean(ps):.4f}±{np.std(ps, ddof=0):.4f} | "
                     f"{np.mean(rs):.4f}±{np.std(rs, ddof=0):.4f} | "
                     f"{np.mean(f1s):.4f}±{np.std(f1s, ddof=0):.4f} |")
    lines.append("")

    lines.append("## 4. Per-Seed 详情\n")
    for r in sorted_results:
        lines.append(f"\n### {r['display_name']}\n")
        lines.append("| Seed | Accuracy | Precision | Recall | F1-Score |")
        lines.append("|-----:|---------:|----------:|-------:|---------:|")
        for s_str, m in r["per_seed"].items():
            lines.append(f"| {s_str} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | "
                         f"{m['Recall']:.4f} | {m['F1-Score']:.4f} |")
        lines.append("")

        # Confusion matrix for ensemble
        cm = r["ensemble_prob_mean"]["CM_2x2"]
        lines.append(f"**5-seed 集成混淆矩阵** (thr=0.5):  ")
        lines.append(f"```\n              预测 Normal    预测 Attack\n"
                     f"实际 Normal      {cm[0][0]:>5}        {cm[0][1]:>5}\n"
                     f"实际 Attack      {cm[1][0]:>5}        {cm[1][1]:>5}\n```")

    lines.append("\n## 5. 关键发现\n")
    best = sorted_results[0]
    lines.append(f"1. **最佳架构**: {best['display_name']} — 5-seed 集成 F1 = {best['ensemble_prob_mean']['F1-Score']:.4f}")
    lines.append(f"2. **TCN+SE vs TCN baseline**: 通过对比有/无 SE 模块的 F1 差值，量化 channel attention 的边际收益")
    lines.append(f"3. **概率集成 vs 多数投票**: 见第 1、2 节对比")
    lines.append(f"4. **单 seed 不稳定性**: 见第 3 节 mean ± std")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=== Cross-Architecture 5-Arch × 5-Seed × 4-Metric Evaluation ===\n")
    results = []
    for short, display, tag in ARCHS:
        r = evaluate_one_arch(short, display, tag)
        if r is not None:
            results.append(r)

    if not results:
        print("\n[FATAL] No architectures evaluated — check predictions CSVs exist")
        return

    json_path = os.path.join(BASE, "cross_arch_5seed_4metrics.json")
    md_path = os.path.join(BASE, "CROSS_ARCH_LEADERBOARD_4METRICS.md")
    csv_path = os.path.join(BASE, "cross_arch_5seed_4metrics_per_seed.csv")

    write_json_report(results, json_path)
    write_per_seed_csv(results, csv_path)
    write_markdown_report(results, md_path)

    print(f"\n[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()