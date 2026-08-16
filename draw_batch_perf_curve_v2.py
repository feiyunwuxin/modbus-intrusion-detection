#!/usr/bin/env python3
"""TCN+SE 23-dim batch sweep — F1m 和 PR-AUC 并排两个子图

1x2 subplots,各自独立 Y 轴,共享 X 轴 (Batch Size)
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\work\Claude\Issue"
BATCHES = [64, 96, 128, 192, 256]
SEEDS = [42, 123, 456, 789, 1024]

# 加载 25 runs
runs = {}
for b in BATCHES:
    for s in SEEDS:
        path = os.path.join(BASE, f"train_tcn_23dim_v2_batch_sweep_partial_b{b}_s{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                runs[(b, s)] = json.load(f)

# 按 batch 聚合
f1m_means, f1m_stds = [], []
pr_means, pr_stds = [], []
for b in BATCHES:
    per = [runs[(b, s)] for s in SEEDS if (b, s) in runs]
    f1m_means.append(np.mean([r["test_macro_f1"] for r in per]))
    f1m_stds.append(np.std([r["test_macro_f1"] for r in per], ddof=0))
    pr_means.append(np.mean([r["test_pr_auc"] for r in per]))
    pr_stds.append(np.std([r["test_pr_auc"] for r in per], ddof=0))

# 5 seed 颜色
seed_color = {42: "#90CAF9", 123: "#A5D6A7", 456: "#FFCC80", 789: "#CE93D8", 1024: "#EF9A9A"}

# 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# === Panel 1: F1m ===
color_f1 = "#E65100"
ax1.set_xlabel("Batch Size", fontsize=13, weight="bold")
ax1.set_ylabel("Test Macro-F1 (5-seed mean ± std)", fontsize=13, weight="bold", color=color_f1)
line1 = ax1.errorbar(BATCHES, f1m_means, yerr=f1m_stds, fmt='o-', linewidth=2.8,
                     capsize=6, capthick=2, markersize=12,
                     color=color_f1, markerfacecolor="white", markeredgewidth=2.5,
                     label="F1m (5-seed mean ± std)", zorder=5)
for b, m, s in zip(BATCHES, f1m_means, f1m_stds):
    ax1.annotate(f"{m:.4f}\n±{s:.4f}", xy=(b, m), xytext=(0, 50),
                 textcoords="offset points", ha="center", fontsize=10,
                 color=color_f1, weight="bold")
# (per-seed 散点已移除)
# baseline
ax1.axhline(y=0.8287, color="#888", linestyle="--", linewidth=2, alpha=0.85,
            label="B=512 baseline (0.8287)", zorder=2)
# B=64 best 标记
best_b = BATCHES[np.argmax(f1m_means)]
ax1.scatter([best_b], [max(f1m_means)], s=600, facecolors='none',
            edgecolors=color_f1, linewidths=3.5, zorder=6)
# BEST 标注: 放在子图正上方居中 (x=160 中点, y=0.8665),无连线
ax1.annotate("BEST\nB=64", xy=(160, 0.8665),
             ha="center", fontsize=13, color=color_f1, weight="bold")

ax1.set_xticks(BATCHES)
ax1.tick_params(axis="y", labelcolor=color_f1)
ax1.set_ylim(0.820, 0.872)
ax1.grid(True, alpha=0.25, linestyle=":")
ax1.set_axisbelow(True)
ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax1.set_title("(a) Test Macro-F1 vs Batch Size", fontsize=13, weight="bold", loc="left", pad=8)

# === Panel 2: PR-AUC ===
color_pr = "#2E7D32"
ax2.set_xlabel("Batch Size", fontsize=13, weight="bold")
ax2.set_ylabel("Test PR-AUC (5-seed mean ± std)", fontsize=13, weight="bold", color=color_pr)
line2 = ax2.errorbar(BATCHES, pr_means, yerr=pr_stds, fmt='s-', linewidth=2.8,
                     capsize=6, capthick=2, markersize=12,
                     color=color_pr, markerfacecolor="white", markeredgewidth=2.5,
                     label="PR-AUC (5-seed mean ± std)", zorder=5)
for b, m, s in zip(BATCHES, pr_means, pr_stds):
    ax2.annotate(f"{m:.4f}\n±{s:.4f}", xy=(b, m), xytext=(0, 50),
                 textcoords="offset points", ha="center", fontsize=10,
                 color=color_pr, weight="bold")
# (per-seed 散点已移除)
# baseline
ax2.axhline(y=0.9079, color="#888", linestyle="--", linewidth=2, alpha=0.85,
            label="B=512 baseline (0.9079)", zorder=2)
# B=64 best
best_b_pr = BATCHES[np.argmax(pr_means)]
ax2.scatter([best_b_pr], [max(pr_means)], s=600, facecolors='none',
            edgecolors=color_pr, linewidths=3.5, zorder=6)
# BEST 标注: 放在子图正上方居中 (x=160 中点, y=0.9335),无连线
ax2.annotate("BEST\nB=64", xy=(160, 0.9335),
             ha="center", fontsize=13, color=color_pr, weight="bold")

ax2.set_xticks(BATCHES)
ax2.tick_params(axis="y", labelcolor=color_pr)
ax2.set_ylim(0.890, 0.938)
ax2.grid(True, alpha=0.25, linestyle=":")
ax2.set_axisbelow(True)
ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax2.set_title("(b) Test PR-AUC vs Batch Size", fontsize=13, weight="bold", loc="left", pad=8)

# 总标题
fig.suptitle("TCN+SE 23-dim (64-combos Champion) — Batch Size Performance",
             fontsize=15, weight="bold", y=0.995)
# 副标题
delta_f1 = max(f1m_means) - 0.8287
delta_pr = max(pr_means) - 0.9079
fig.text(0.5, 0.94,
         f"5 seeds × 5 batches = 25 runs | B=64 wins both: F1m={max(f1m_means):.4f} (Δ {delta_f1:+.4f}), "
         f"PR-AUC={max(pr_means):.4f} (Δ {delta_pr:+.4f})",
         ha="center", fontsize=10.5, style="italic", color="#37474F")

plt.tight_layout(rect=[0, 0, 1, 0.92])
out_png = os.path.join(BASE, "batch_perf_curve_v2.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {out_png}")
print(f"\nF1m BEST  : B={BATCHES[np.argmax(f1m_means)]}  {max(f1m_means):.4f}+/-{f1m_stds[np.argmax(f1m_means)]:.4f}  (Δ vs B=512 = {delta_f1:+.4f})")
print(f"PR-AUC BEST: B={BATCHES[np.argmax(pr_means)]}  {max(pr_means):.4f}+/-{pr_stds[np.argmax(pr_means)]:.4f}  (Δ vs B=512 = {delta_pr:+.4f})")
