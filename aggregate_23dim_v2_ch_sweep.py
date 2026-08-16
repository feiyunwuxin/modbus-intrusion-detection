#!/usr/bin/env python3
"""TCN+SE 23-dim + B=64 + LR=4e-3 + ep=20 × Channels Sweep — 汇总 + 报告 + 图

读取 25 个 partial JSON (5 channels × 5 seed) → 汇总 JSON + Markdown + 4-panel PNG。
对照基线: 23-dim + B=64 + LR=4e-3 + ep=20 + ch=64 → F1m=0.8705±0.0040, 73,945 params
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\work\Claude\Issue"
CHANNELS_GRID = [16, 32, 48, 64, 96]
SEEDS = [42, 123, 456, 789, 1024]


def load_all_runs():
    runs = {}
    for ch in CHANNELS_GRID:
        for seed in SEEDS:
            path = os.path.join(BASE, f"train_tcn_23dim_v2_ch_sweep_partial_ch{ch}_s{seed}.json")
            if os.path.exists(path):
                with open(path) as f:
                    runs[(ch, seed)] = json.load(f)
            else:
                print(f"  [missing] ch={ch} s={seed}")
    return runs


def aggregate_by_ch(runs):
    summary = {}
    for ch in CHANNELS_GRID:
        per_seed = [runs[(ch, seed)] for seed in SEEDS if (ch, seed) in runs]
        if not per_seed:
            continue
        n = len(per_seed)
        params = per_seed[0]["n_params"]
        summary[ch] = {
            "n_seeds": n,
            "n_params": params,
            "f1m_mean": float(np.mean([r["test_macro_f1"] for r in per_seed])),
            "f1m_std":  float(np.std([r["test_macro_f1"]  for r in per_seed], ddof=0)),
            "pr_mean":  float(np.mean([r["test_pr_auc"]   for r in per_seed])),
            "pr_std":   float(np.std([r["test_pr_auc"]    for r in per_seed], ddof=0)),
            "binf1_mean": float(np.mean([r["test_binary_f1"] for r in per_seed])),
            "binf1_std":  float(np.std([r["test_binary_f1"]  for r in per_seed], ddof=0)),
            "acc_mean":  float(np.mean([r["test_accuracy"] for r in per_seed])),
            "roc_mean":  float(np.mean([r["test_roc_auc"]  for r in per_seed])),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in per_seed])),
            "n_early_stopped": int(sum(1 for r in per_seed if r.get("early_stopped"))),
            "train_time_s_mean": float(np.mean([r["train_time_s"] for r in per_seed])),
            "f1m_per_kp": float(np.mean([r["test_macro_f1"] for r in per_seed])) / (params / 1000),
            "prauc_per_kp": float(np.mean([r["test_pr_auc"] for r in per_seed])) / (params / 1000),
            "per_seed": per_seed,
        }
    return summary


def write_results_json(runs, summary):
    out = {
        "config": {
            "n_features": 23,
            "drop_features": {"2": "length", "3": "setpoint", "20": "crc_mean_w", "22": "cmd_count_w"},
            "source": "TCN+SE 23-dim + B=64 + LR=4e-3 + ep=20 (4 轮 sweep 累积最佳)",
            "channels_grid": CHANNELS_GRID,
            "seeds": SEEDS,
            "fixed_hyperparams": {
                "batch": 64, "lr": 4e-3, "epochs": 20, "wd": 1e-5, "dropout": 0.3,
                "patience": 5, "grad_clip": 0.5,
                "n_blocks": 3, "kernel_size": 3, "dilations": [1, 2, 4],
                "se_reduction": 8, "window": 16,
            },
        },
        "summary_by_channels": {str(ch): {k: v for k, v in s.items() if k != "per_seed"}
                                for ch, s in summary.items()},
        "all_runs": [
            {"channels": ch, "seed": s, **{k: v for k, v in r.items() if k != "history"}}
            for (ch, s), r in sorted(runs.items())
        ],
    }
    out_path = os.path.join(BASE, "train_tcn_23dim_v2_ch_sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[saved] {out_path}")
    return out_path


def write_table_md(summary, runs):
    lines = []
    lines.append("# TCN+SE 23-dim + B=64 + LR=4e-3 + ep=20 (4 轮累积最佳) × Channels 扫描结果")
    lines.append("")
    lines.append("**23-dim** = 27-dim - {length(2), setpoint(3), crc_mean_w(20), cmd_count_w(22)}")
    lines.append("**固定超参**: B=64, LR=4e-3, ep=20, WD=1e-5, Dropout=0.3, ch=variable, SE r=8, W=16, patience=5")
    lines.append("**基线对照 (本工作)**: ch=64 → F1m=0.8705±0.0040, 73,945 params, PR-AUC=0.9154±0.0109")
    lines.append("**历史对照**: [[project-tcn-v4-ch-sweep-full]] 19-dim 扫 ch=8-128, ch=48 PR-AUC 0.9329 冠军, ch=128 F1m 0.8729 冠军")
    lines.append("**网格**: channels ∈ {16, 32, 48, 64, 96} × 5 seeds = 25 runs")
    lines.append("")
    lines.append("## 1. 按 channels 聚合的 5-seed 统计")
    lines.append("")
    lines.append("| Channels | Params | F1m (mean ± std) | PR-AUC (mean ± std) | Bin-F1 (mean ± std) | ROC-AUC | BestEpoch | EarlyStop | TrainT(s) | F1m/Kparam |")
    lines.append("|----------|--------|------------------|---------------------|---------------------|---------|-----------|-----------|-----------|------------|")
    for ch in CHANNELS_GRID:
        if ch not in summary:
            lines.append(f"| {ch} | MISSING | — | — | — | — | — | — | — | — |")
            continue
        s = summary[ch]
        lines.append(
            f"| **{ch}** | {s['n_params']:,} | "
            f"{s['f1m_mean']:.4f} ± {s['f1m_std']:.4f} | "
            f"{s['pr_mean']:.4f} ± {s['pr_std']:.4f} | "
            f"{s['binf1_mean']:.4f} ± {s['binf1_std']:.4f} | "
            f"{s['roc_mean']:.4f} | "
            f"{s['best_epoch_mean']:.1f} | "
            f"{s['n_early_stopped']}/5 | "
            f"{s['train_time_s_mean']:.1f} | "
            f"{s['f1m_per_kp']:.4f} |"
        )
    lines.append("")

    if summary:
        ranked = sorted(summary.items(), key=lambda x: -x[1]["f1m_mean"])
        lines.append("## 2. F1m 排名 (desc)")
        lines.append("")
        lines.append("| Rank | Channels | Params | F1m_mean | F1m_std | Δ vs 最佳 |")
        lines.append("|------|----------|--------|----------|---------|-----------|")
        best_f1m = ranked[0][1]["f1m_mean"]
        for i, (ch, s) in enumerate(ranked, 1):
            delta = s["f1m_mean"] - best_f1m
            tag = " [BEST]" if i == 1 else ""
            lines.append(f"| {i} | {ch} | {s['n_params']:,} | {s['f1m_mean']:.4f} | {s['f1m_std']:.4f} | {delta:+.4f}{tag} |")
        lines.append("")

        lines.append("## 3. vs 本工作 baseline (ch=64 → F1m=0.8705, PR=0.9154)")
        lines.append("")
        lines.append("| Channels | ΔF1m vs ch=64 | ΔPR-AUC vs ch=64 | 解读 |")
        lines.append("|----------|---------------|------------------|------|")
        for ch in CHANNELS_GRID:
            if ch not in summary:
                continue
            s = summary[ch]
            df1 = s["f1m_mean"] - 0.8705
            dpr = s["pr_mean"]  - 0.9154
            if df1 >= 0.005:
                tag = "🟢 显著正"
            elif df1 >= 0.0:
                tag = "🟡 持平 / 微正"
            elif df1 >= -0.005:
                tag = "🟡 持平 / 微负"
            else:
                tag = "🔴 显著负"
            lines.append(f"| {ch} | {df1:+.4f} | {dpr:+.4f} | {tag} |")
        lines.append("")

        ranked_pr = sorted(summary.items(), key=lambda x: -x[1]["pr_mean"])
        lines.append("## 4. PR-AUC 排名 (desc)")
        lines.append("")
        lines.append("| Rank | Channels | Params | PR_mean | PR_std |")
        lines.append("|------|----------|--------|---------|--------|")
        for i, (ch, s) in enumerate(ranked_pr, 1):
            tag = " [BEST]" if i == 1 else ""
            lines.append(f"| {i} | {ch} | {s['n_params']:,} | {s['pr_mean']:.4f} | {s['pr_std']:.4f}{tag} |")
        lines.append("")

    lines.append("## 5. 详细 25 runs (raw)")
    lines.append("")
    lines.append("| Channels | Seed | F1m | BinF1 | Acc | ROC | PR-AUC | BestEpoch | ActualEp | Params | TrainT(s) |")
    lines.append("|----------|------|-----|-------|-----|-----|--------|-----------|----------|--------|-----------|")
    for ch in CHANNELS_GRID:
        for s in SEEDS:
            r = runs.get((ch, s))
            if r is None:
                lines.append(f"| {ch} | {s} | MISSING | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {ch} | {s} | "
                f"{r['test_macro_f1']:.4f} | {r['test_binary_f1']:.4f} | "
                f"{r['test_accuracy']:.4f} | {r['test_roc_auc']:.4f} | "
                f"{r['test_pr_auc']:.4f} | {r['best_epoch']} | {r['actual_epochs_run']} | "
                f"{r['n_params']:,} | {r['train_time_s']:.1f} |"
            )
    lines.append("")

    out_path = os.path.join(BASE, "train_tcn_23dim_v2_ch_sweep_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {out_path}")
    return out_path


def draw_figure(summary, runs):
    chs = sorted(summary.keys())
    colors = {16: "#1565C0", 32: "#2E7D32", 48: "#E65100", 64: "#6A1B9A", 96: "#C62828"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # === Panel (a): Test F1m vs Channels ===
    ax = axes[0, 0]
    f1m_mean = [summary[ch]["f1m_mean"] for ch in chs]
    f1m_std  = [summary[ch]["f1m_std"]  for ch in chs]
    params   = [summary[ch]["n_params"] for ch in chs]
    ax.errorbar(chs, f1m_mean, yerr=f1m_std, fmt='o-', linewidth=2, capsize=5,
                color="#E65100", markerfacecolor="white", markeredgewidth=2, markersize=10)
    for ch, m, s, p in zip(chs, f1m_mean, f1m_std, params):
        ax.annotate(f"{m:.4f}\n±{s:.4f}\n({p//1000}K params)", xy=(ch, m), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=8.5, color="#37474F")
    ax.axhline(y=0.8705, color="#888", linestyle="--", linewidth=1.5,
               label="ch=64 baseline F1m=0.8705")
    ax.set_xlabel("Channels", fontsize=11)
    ax.set_ylabel("Test Macro-F1 (5-seed mean ± std)", fontsize=11)
    ax.set_title("(a) Test F1m vs Channels", fontsize=12, weight="bold", loc="left")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    best_ch = max(chs, key=lambda ch: summary[ch]["f1m_mean"])
    ax.scatter([best_ch], [summary[best_ch]["f1m_mean"]], s=300, facecolors='none',
               edgecolors='#E65100', linewidths=2.5, zorder=5)
    ax.annotate("BEST", xy=(best_ch, summary[best_ch]["f1m_mean"]),
                xytext=(15, 15), textcoords="offset points", fontsize=10,
                color="#E65100", weight="bold")

    # === Panel (b): Test PR-AUC vs Channels ===
    ax = axes[0, 1]
    pr_mean = [summary[ch]["pr_mean"] for ch in chs]
    pr_std  = [summary[ch]["pr_std"]  for ch in chs]
    ax.errorbar(chs, pr_mean, yerr=pr_std, fmt='s-', linewidth=2, capsize=5,
                color="#2E7D32", markerfacecolor="white", markeredgewidth=2, markersize=10)
    for ch, m, s in zip(chs, pr_mean, pr_std):
        ax.annotate(f"{m:.4f}\n±{s:.4f}", xy=(ch, m), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=8.5, color="#37474F")
    ax.axhline(y=0.9154, color="#888", linestyle="--", linewidth=1.5,
               label="ch=64 baseline PR-AUC=0.9154")
    ax.set_xlabel("Channels", fontsize=11)
    ax.set_ylabel("Test PR-AUC (5-seed mean ± std)", fontsize=11)
    ax.set_title("(b) Test PR-AUC vs Channels", fontsize=12, weight="bold", loc="left")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    best_ch_pr = max(chs, key=lambda ch: summary[ch]["pr_mean"])
    ax.scatter([best_ch_pr], [summary[best_ch_pr]["pr_mean"]], s=300, facecolors='none',
               edgecolors='#2E7D32', linewidths=2.5, zorder=5)
    ax.annotate("BEST", xy=(best_ch_pr, summary[best_ch_pr]["pr_mean"]),
                xytext=(15, 15), textcoords="offset points", fontsize=10,
                color="#2E7D32", weight="bold")

    # === Panel (c): Params vs Channels + F1m/Params ratio (Pareto) ===
    ax = axes[1, 0]
    f1m_per_kp = [summary[ch]["f1m_per_kp"] for ch in chs]
    ax.plot(chs, f1m_per_kp, 'o-', linewidth=2, color="#6A1B9A",
            markerfacecolor="white", markeredgewidth=2, markersize=10)
    for ch, r in zip(chs, f1m_per_kp):
        ax.annotate(f"{r:.3f}", xy=(ch, r), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color="#37474F")
    ax.set_xlabel("Channels", fontsize=11)
    ax.set_ylabel("F1m / Kparams (efficiency ratio)", fontsize=11)
    ax.set_title("(c) Param Efficiency: F1m per 1K params", fontsize=12, weight="bold", loc="left")
    ax.grid(True, alpha=0.3, linestyle=":")

    # === Panel (d): Val F1m curves overlay ===
    ax = axes[1, 1]
    for ch in chs:
        color = colors.get(ch, "#444")
        for seed in SEEDS:
            r = runs.get((ch, seed))
            if r is None or "history" not in r:
                continue
            hist = r["history"]
            epochs = [h["epoch"] for h in hist]
            f1m    = [h["val_f1m"] for h in hist]
            label = f"ch={ch}" if seed == SEEDS[0] else None
            ax.plot(epochs, f1m, color=color, alpha=0.4, linewidth=0.9, label=label)
    for ch in chs:
        color = colors.get(ch, "#444")
        max_len = max(len(runs[(ch, s)]["history"]) for s in SEEDS if (ch, s) in runs)
        f1m_matrix = []
        for s in SEEDS:
            r = runs.get((ch, s))
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
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=9)

    fig.suptitle("TCN+SE 23-dim + B=64 + LR=4e-3 (3-sweep champion) — Channels Sweep (5-seed × 5 ch = 25 runs)",
                 fontsize=14, weight="bold", y=0.995)
    best = max(summary.items(), key=lambda x: x[1]["f1m_mean"])
    fig.text(0.5, 0.005,
             f"Verdict: ch={best[0]} is the new sweet spot — Test F1m = {best[1]['f1m_mean']:.4f}+/-{best[1]['f1m_std']:.4f}  "
             f"({best[1]['n_params']:,} params, Delta vs ch=64 = {best[1]['f1m_mean']-0.8705:+.4f})",
             ha="center", fontsize=10, style="italic", color="#37474F")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_png = os.path.join(BASE, "train_tcn_23dim_v2_ch_sweep.png")
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
    summary = aggregate_by_ch(runs)
    write_results_json(runs, summary)
    write_table_md(summary, runs)
    draw_figure(summary, runs)
    print("\n" + "="*110)
    print(f"{'CH':>4} {'Params':>8} {'F1m mean':>10} {'F1m std':>10} {'PR mean':>10} {'PR std':>10} {'TrainT(s)':>10} {'F1m/Kp':>8} {'EarlyStop':>10}")
    print("-"*110)
    for ch in CHANNELS_GRID:
        s = summary.get(ch)
        if s is None:
            print(f"{ch:>4} MISSING")
            continue
        print(f"{ch:>4} {s['n_params']:>8,} {s['f1m_mean']:>10.4f} {s['f1m_std']:>10.4f} "
              f"{s['pr_mean']:>10.4f} {s['pr_std']:>10.4f} "
              f"{s['train_time_s_mean']:>10.1f} {s['f1m_per_kp']:>8.4f} {s['n_early_stopped']:>10d}/5")
    best = max(summary.items(), key=lambda x: x[1]["f1m_mean"])
    print(f"\n[BEST F1m] ch={best[0]}  F1m={best[1]['f1m_mean']:.4f}+/-{best[1]['f1m_std']:.4f}  ({best[1]['n_params']:,} params)")
    print(f"   vs ch=64 baseline (0.8705): DeltaF1m = {best[1]['f1m_mean']-0.8705:+.4f}")


if __name__ == "__main__":
    main()
