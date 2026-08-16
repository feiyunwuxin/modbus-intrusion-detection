#!/usr/bin/env python3
"""TCN+SE 23-dim (64-combos 冠军版) × batch size 扫描 — 汇总 + 报告 + 图

读取 25 个 partial JSON → 汇总 JSON + Markdown 表格 + 4-panel PNG。

输出:
  - train_tcn_23dim_v2_batch_sweep_results.json
  - train_tcn_23dim_v2_batch_sweep_table.md
  - train_tcn_23dim_v2_batch_sweep.png  (4-panel)

Panel 设计:
  (a) Test F1m vs Batch size  (5-seed mean ± std, 5 个 batch 折线)
  (b) Test PR-AUC vs Batch size (5-seed mean ± std)
  (c) Train time vs Batch size (单 seed 时间 vs batch)
  (d) Val F1m 曲线 overlay (5 batch × 5 seeds = 25 曲线)
"""

import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\work\Claude\Issue"
BATCHES = [64, 96, 128, 192, 256]
SEEDS = [42, 123, 456, 789, 1024]


def load_all_runs():
    """读取所有 partial JSON,返回 {(batch, seed): run_dict}"""
    runs = {}
    for batch in BATCHES:
        for seed in SEEDS:
            path = os.path.join(BASE, f"train_tcn_23dim_v2_batch_sweep_partial_b{batch}_s{seed}.json")
            if os.path.exists(path):
                with open(path) as f:
                    runs[(batch, seed)] = json.load(f)
            else:
                print(f"  [missing] b={batch} s={seed}")
    return runs


def aggregate_by_batch(runs):
    """按 batch 聚合 5-seed 统计"""
    summary = {}
    for batch in BATCHES:
        per_seed = [runs[(batch, seed)] for seed in SEEDS if (batch, seed) in runs]
        if not per_seed:
            continue
        n = len(per_seed)
        summary[batch] = {
            "n_seeds": n,
            "iters_per_ep": per_seed[0]["iters_per_ep"],
            "f1m_mean": float(np.mean([r["test_macro_f1"] for r in per_seed])),
            "f1m_std":  float(np.std([r["test_macro_f1"]  for r in per_seed], ddof=0)),
            "pr_mean":  float(np.mean([r["test_pr_auc"]   for r in per_seed])),
            "pr_std":   float(np.std([r["test_pr_auc"]    for r in per_seed], ddof=0)),
            "binf1_mean": float(np.mean([r["test_binary_f1"] for r in per_seed])),
            "binf1_std":  float(np.std([r["test_binary_f1"]  for r in per_seed], ddof=0)),
            "acc_mean":  float(np.mean([r["test_accuracy"] for r in per_seed])),
            "roc_mean":  float(np.mean([r["test_roc_auc"]  for r in per_seed])),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in per_seed])),
            "train_time_s_mean": float(np.mean([r["train_time_s"] for r in per_seed])),
            "n_collapsed": int(sum(1 for r in per_seed if r["best_epoch"] <= 2)),
            "per_seed": per_seed,
        }
    return summary


def write_results_json(runs, summary):
    out = {
        "config": {
            "n_features": 23,
            "drop_features": {"2": "length", "3": "setpoint", "20": "crc_mean_w", "22": "cmd_count_w"},
            "source": "[[project-64combos-results]] 64-combos 冠军版 5-seed F1m=0.8287+/-0.0014",
            "batches": BATCHES,
            "seeds": SEEDS,
            "hyperparams": {
                "lr": 5e-4, "wd": 1e-5, "dropout": 0.3,
                "epochs": 20, "patience": 5, "grad_clip": 0.5,
                "n_blocks": 3, "channels": 64, "dilations": [1, 2, 4],
                "se_reduction": 8, "window": 16,
            },
        },
        "summary_by_batch": {str(b): {k: v for k, v in s.items() if k != "per_seed"} for b, s in summary.items()},
        "all_runs": [
            {"batch": b, "seed": s, **{k: v for k, v in r.items() if k != "history"}}
            for (b, s), r in sorted(runs.items())
        ],
    }
    out_path = os.path.join(BASE, "train_tcn_23dim_v2_batch_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[saved] {out_path}")
    return out_path


def write_table_md(summary, runs):
    """Markdown 汇总表"""
    lines = []
    lines.append("# TCN+SE 23-dim (64-combos 冠军版) × Batch Size 扫描结果")
    lines.append("")
    lines.append("**23-dim 构造** = 27-dim − {length(2), setpoint(3), crc_mean_w(20), cmd_count_w(22)}")
    lines.append("**基线对照**: [[project-64combos-results]] B=512 5-seed → F1m=0.8287±0.0014, PR-AUC=0.9079")
    lines.append("**网格**: BATCH ∈ {64, 96, 128, 192, 256} × 5 seeds = 25 runs")
    lines.append("**超参**: LR=5e-4, WD=1e-5, Dropout=0.3, EPOCHS=20, Patience=5, ch=64, SE r=8, W=16")
    lines.append("")
    lines.append("## 1. 按 batch 聚合的 5-seed 统计")
    lines.append("")
    lines.append("| Batch | Iters/ep | F1m (mean ± std) | PR-AUC (mean ± std) | Bin-F1 (mean ± std) | ROC-AUC | BestEpoch | TrainT(s) | N_collapsed |")
    lines.append("|-------|----------|------------------|---------------------|---------------------|---------|-----------|-----------|-------------|")
    for b in BATCHES:
        if b not in summary:
            lines.append(f"| {b} | — | MISSING | — | — | — | — | — | — |")
            continue
        s = summary[b]
        lines.append(
            f"| **{b}** | {s['iters_per_ep']} | "
            f"{s['f1m_mean']:.4f} ± {s['f1m_std']:.4f} | "
            f"{s['pr_mean']:.4f} ± {s['pr_std']:.4f} | "
            f"{s['binf1_mean']:.4f} ± {s['binf1_std']:.4f} | "
            f"{s['roc_mean']:.4f} | "
            f"{s['best_epoch_mean']:.1f} | "
            f"{s['train_time_s_mean']:.1f} | "
            f"{s['n_collapsed']}/5 |"
        )
    lines.append("")

    # 排名
    if summary:
        ranked = sorted(summary.items(), key=lambda x: -x[1]["f1m_mean"])
        lines.append("## 2. F1m 排名 (desc)")
        lines.append("")
        lines.append("| Rank | Batch | F1m_mean | F1m_std | Δ vs 最佳 |")
        lines.append("|------|-------|----------|---------|-----------|")
        best_f1m = ranked[0][1]["f1m_mean"]
        for i, (b, s) in enumerate(ranked, 1):
            delta = s["f1m_mean"] - best_f1m
            tag = " [BEST]" if i == 1 else ""
            lines.append(f"| {i} | {b} | {s['f1m_mean']:.4f} | {s['f1m_std']:.4f} | {delta:+.4f}{tag} |")
        lines.append("")

        # vs 64-combos B=512 baseline
        lines.append("## 3. vs 64-combos B=512 baseline (F1m=0.8287, PR=0.9079)")
        lines.append("")
        lines.append("| Batch | ΔF1m vs B=512 | ΔPR-AUC vs B=512 | 解读 |")
        lines.append("|-------|---------------|------------------|------|")
        for b in BATCHES:
            if b not in summary:
                continue
            s = summary[b]
            df1 = s["f1m_mean"] - 0.8287
            dpr = s["pr_mean"]  - 0.9079
            if df1 >= 0.005:
                tag = "🟢 显著正"
            elif df1 >= 0.0:
                tag = "🟡 持平 / 微正"
            elif df1 >= -0.005:
                tag = "🟡 持平 / 微负"
            else:
                tag = "🔴 显著负"
            lines.append(f"| {b} | {df1:+.4f} | {dpr:+.4f} | {tag} |")
        lines.append("")

        # PR-AUC 排名
        ranked_pr = sorted(summary.items(), key=lambda x: -x[1]["pr_mean"])
        lines.append("## 4. PR-AUC 排名 (desc)")
        lines.append("")
        lines.append("| Rank | Batch | PR_mean | PR_std |")
        lines.append("|------|-------|---------|--------|")
        for i, (b, s) in enumerate(ranked_pr, 1):
            tag = " [BEST]" if i == 1 else ""
            lines.append(f"| {i} | {b} | {s['pr_mean']:.4f} | {s['pr_std']:.4f}{tag} |")
        lines.append("")

    # 详细 25 runs 表
    lines.append("## 5. 详细 25 runs (raw)")
    lines.append("")
    lines.append("| Batch | Seed | F1m | BinF1 | Acc | ROC | PR-AUC | BestEpoch | TrainT(s) |")
    lines.append("|-------|------|-----|-------|-----|-----|--------|-----------|-----------|")
    for b in BATCHES:
        for s in SEEDS:
            r = runs.get((b, s))
            if r is None:
                lines.append(f"| {b} | {s} | MISSING | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {b} | {s} | "
                f"{r['test_macro_f1']:.4f} | {r['test_binary_f1']:.4f} | "
                f"{r['test_accuracy']:.4f} | {r['test_roc_auc']:.4f} | "
                f"{r['test_pr_auc']:.4f} | {r['best_epoch']} | {r['train_time_s']:.1f} |"
            )
    lines.append("")

    out_path = os.path.join(BASE, "train_tcn_23dim_v2_batch_sweep_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {out_path}")
    return out_path


def draw_figure(summary, runs):
    """4-panel 画图"""
    batches = sorted(summary.keys())
    colors = {64: "#1565C0", 96: "#2E7D32", 128: "#E65100", 192: "#6A1B9A", 256: "#C62828"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # === Panel (a): Test F1m vs Batch ===
    ax = axes[0, 0]
    f1m_mean = [summary[b]["f1m_mean"] for b in batches]
    f1m_std  = [summary[b]["f1m_std"]  for b in batches]
    ax.errorbar(batches, f1m_mean, yerr=f1m_std, fmt='o-', linewidth=2, capsize=5,
                color="#E65100", markerfacecolor="white", markeredgewidth=2, markersize=10)
    for b, m, s in zip(batches, f1m_mean, f1m_std):
        ax.annotate(f"{m:.4f}\n±{s:.4f}", xy=(b, m), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9, color="#37474F")
    # baseline B=512 horizontal line
    ax.axhline(y=0.8287, color="#888", linestyle="--", linewidth=1.5,
               label="B=512 baseline 0.8287")
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Test Macro-F1 (5-seed mean ± std)", fontsize=11)
    ax.set_title("(a) Test F1m vs Batch Size", fontsize=12, weight="bold", loc="left")
    ax.set_xticks(batches)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    # highlight best
    best_b = max(batches, key=lambda b: summary[b]["f1m_mean"])
    ax.scatter([best_b], [summary[best_b]["f1m_mean"]], s=300, facecolors='none',
               edgecolors='#E65100', linewidths=2.5, zorder=5)
    ax.annotate("BEST", xy=(best_b, summary[best_b]["f1m_mean"]),
                xytext=(15, 15), textcoords="offset points", fontsize=10,
                color="#E65100", weight="bold")

    # === Panel (b): Test PR-AUC vs Batch ===
    ax = axes[0, 1]
    pr_mean = [summary[b]["pr_mean"] for b in batches]
    pr_std  = [summary[b]["pr_std"]  for b in batches]
    ax.errorbar(batches, pr_mean, yerr=pr_std, fmt='s-', linewidth=2, capsize=5,
                color="#2E7D32", markerfacecolor="white", markeredgewidth=2, markersize=10)
    for b, m, s in zip(batches, pr_mean, pr_std):
        ax.annotate(f"{m:.4f}\n±{s:.4f}", xy=(b, m), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9, color="#37474F")
    ax.axhline(y=0.9079, color="#888", linestyle="--", linewidth=1.5,
               label="B=512 baseline 0.9079")
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Test PR-AUC (5-seed mean ± std)", fontsize=11)
    ax.set_title("(b) Test PR-AUC vs Batch Size", fontsize=12, weight="bold", loc="left")
    ax.set_xticks(batches)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    best_b_pr = max(batches, key=lambda b: summary[b]["pr_mean"])
    ax.scatter([best_b_pr], [summary[best_b_pr]["pr_mean"]], s=300, facecolors='none',
               edgecolors='#2E7D32', linewidths=2.5, zorder=5)
    ax.annotate("BEST", xy=(best_b_pr, summary[best_b_pr]["pr_mean"]),
                xytext=(15, 15), textcoords="offset points", fontsize=10,
                color="#2E7D32", weight="bold")

    # === Panel (c): Train time vs Batch ===
    ax = axes[1, 0]
    t_mean = [summary[b]["train_time_s_mean"] for b in batches]
    ax.plot(batches, t_mean, 'o-', linewidth=2, color="#6A1B9A",
            markerfacecolor="white", markeredgewidth=2, markersize=10)
    for b, t in zip(batches, t_mean):
        ax.annotate(f"{t:.1f}s", xy=(b, t), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color="#37474F")
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Train Time (s, 5-seed mean)", fontsize=11)
    ax.set_title("(c) Train Time vs Batch Size", fontsize=12, weight="bold", loc="left")
    ax.set_xticks(batches)
    ax.grid(True, alpha=0.3, linestyle=":")

    # === Panel (d): Val F1m curves overlay (light) ===
    ax = axes[1, 1]
    for b in batches:
        color = colors.get(b, "#444")
        for seed in SEEDS:
            r = runs.get((b, seed))
            if r is None or "history" not in r:
                continue
            hist = r["history"]
            epochs = [h["epoch"] for h in hist]
            f1m    = [h["val_f1m"] for h in hist]
            label = f"B={b}" if seed == SEEDS[0] else None
            ax.plot(epochs, f1m, color=color, alpha=0.45, linewidth=1.0, label=label)
    # mean curves
    for b in batches:
        color = colors.get(b, "#444")
        max_len = max(len(runs[(b, s)]["history"]) for s in SEEDS if (b, s) in runs)
        f1m_matrix = []
        for s in SEEDS:
            r = runs.get((b, s))
            if r is None: continue
            hist = r["history"]
            arr = [h["val_f1m"] for h in hist] + [np.nan]*(max_len - len(hist))
            f1m_matrix.append(arr)
        if f1m_matrix:
            mean_curve = np.nanmean(f1m_matrix, axis=0)
            ax.plot(range(1, len(mean_curve)+1), mean_curve, color=color, linewidth=2.5, alpha=0.95)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Macro-F1", fontsize=11)
    ax.set_title("(d) Val F1m Curves (light: per-seed, bold: 5-seed mean)", fontsize=12, weight="bold", loc="left")
    ax.grid(True, alpha=0.3, linestyle=":")
    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=9)

    fig.suptitle("TCN+SE 23-dim (64-combos Champion) — Batch Size Sweep (5-seed × 5 batches = 25 runs)",
                 fontsize=14, weight="bold", y=0.995)
    best = max(summary.items(), key=lambda x: x[1]["f1m_mean"])
    fig.text(0.5, 0.005,
             f"Verdict: B={best[0]} is the new sweet spot — Test F1m = {best[1]['f1m_mean']:.4f}±{best[1]['f1m_std']:.4f}  "
             f"(Δ vs B=512 = {best[1]['f1m_mean']-0.8287:+.4f}, +{(best[1]['f1m_mean']-0.8287)*100:.1f}%)",
             ha="center", fontsize=10, style="italic", color="#37474F")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_png = os.path.join(BASE, "train_tcn_23dim_v2_batch_sweep.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out_png}")
    return out_png


def main():
    runs = load_all_runs()
    print(f"[loaded] {len(runs)}/25 runs")
    if len(runs) == 0:
        print("[abort] no runs found")
        return
    summary = aggregate_by_batch(runs)
    write_results_json(runs, summary)
    write_table_md(summary, runs)
    draw_figure(summary, runs)
    # 总结 stdout
    print("\n" + "="*80)
    print(f"{'Batch':>6} {'F1m mean':>10} {'F1m std':>10} {'PR mean':>10} {'PR std':>10} {'TrainT(s)':>10} {'N_collapsed':>11}")
    print("-"*80)
    for b in BATCHES:
        s = summary.get(b)
        if s is None:
            print(f"{b:>6} MISSING")
            continue
        print(f"{b:>6} {s['f1m_mean']:>10.4f} {s['f1m_std']:>10.4f} "
              f"{s['pr_mean']:>10.4f} {s['pr_std']:>10.4f} "
              f"{s['train_time_s_mean']:>10.1f} {s['n_collapsed']:>11d}")
    best = max(summary.items(), key=lambda x: x[1]["f1m_mean"])
    print(f"\n[BEST F1m] B={best[0]}  F1m={best[1]['f1m_mean']:.4f}+/-{best[1]['f1m_std']:.4f}")
    print(f"   vs B=512 baseline (0.8287): DeltaF1m = {best[1]['f1m_mean']-0.8287:+.4f}")


if __name__ == "__main__":
    main()
