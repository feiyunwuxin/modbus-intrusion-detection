#!/usr/bin/env python3
"""TCN+SE 23-dim + B=64 × LR Sweep — F1m 和 PR-AUC 并排 2 子图

复用 batch sweep v2 的布局,数据标签上移 + Y 轴扩边 + BEST 圆圈 + 居中 BEST 文字
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\work\Claude\Issue"
LR_GRID = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3]
SEEDS = [42, 123, 456, 789, 1024]


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


# Load 35 runs
runs = {}
for lr in LR_GRID:
    tag = lr_tag(lr)
    for seed in SEEDS:
        path = os.path.join(BASE, f"train_tcn_23dim_v2_lr_sweep_partial_lr{tag}_s{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                runs[(lr, seed)] = json.load(f)

# Aggregate
f1m_means, f1m_stds = [], []
pr_means, pr_stds = [], []
for lr in LR_GRID:
    per = [runs[(lr, s)] for s in SEEDS if (lr, s) in runs]
    f1m_means.append(np.mean([r["test_macro_f1"] for r in per]))
    f1m_stds.append(np.std([r["test_macro_f1"] for r in per], ddof=0))
    pr_means.append(np.mean([r["test_pr_auc"] for r in per]))
    pr_stds.append(np.std([r["test_pr_auc"] for r in per], ddof=0))

# 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# === Panel 1: F1m ===
color_f1 = "#E65100"
ax1.set_xlabel("Learning Rate", fontsize=13, weight="bold")
ax1.set_ylabel("Test Macro-F1 (5-seed mean ± std)", fontsize=13, weight="bold", color=color_f1)
line1 = ax1.errorbar(LR_GRID, f1m_means, yerr=f1m_stds, fmt='o-', linewidth=2.5,
                     capsize=5, capthick=2, markersize=11,
                     color=color_f1, markerfacecolor="white", markeredgewidth=2.5,
                     label="F1m (5-seed mean ± std)", zorder=5)
for lr, m, s in zip(LR_GRID, f1m_means, f1m_stds):
    ax1.annotate(f"{m:.4f}\n±{s:.4f}", xy=(lr, m), xytext=(0, 22),
                 textcoords="offset points", ha="center", fontsize=9.5,
                 color=color_f1, weight="bold")
# baseline B=64 + LR=5e-4
ax1.axhline(y=0.8454, color="#888", linestyle="--", linewidth=2, alpha=0.85,
            label="LR=5e-4 baseline F1m=0.8454", zorder=2)
# B=64 best
best_lr = LR_GRID[int(np.argmax(f1m_means))]
ax1.scatter([best_lr], [max(f1m_means)], s=550, facecolors='none',
            edgecolors=color_f1, linewidths=3.5, zorder=6)
# BEST 标注: 子图正上方居中 (X = LR=4e-3, Y=顶部 0.886)
ax1.annotate("BEST\nLR=4e-3", xy=(4e-3, 0.886),
             ha="center", fontsize=12, color=color_f1, weight="bold")
ax1.set_xscale("log")
ax1.tick_params(axis="y", labelcolor=color_f1)
ax1.set_ylim(0.815, 0.892)
ax1.grid(True, alpha=0.25, linestyle=":")
ax1.set_axisbelow(True)
ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax1.set_title("(a) Test Macro-F1 vs LR", fontsize=13, weight="bold", loc="left", pad=8)

# === Panel 2: PR-AUC ===
color_pr = "#2E7D32"
ax2.set_xlabel("Learning Rate", fontsize=13, weight="bold")
ax2.set_ylabel("Test PR-AUC (5-seed mean ± std)", fontsize=13, weight="bold", color=color_pr)
line2 = ax2.errorbar(LR_GRID, pr_means, yerr=pr_stds, fmt='s-', linewidth=2.5,
                     capsize=5, capthick=2, markersize=11,
                     color=color_pr, markerfacecolor="white", markeredgewidth=2.5,
                     label="PR-AUC (5-seed mean ± std)", zorder=5)
for lr, m, s in zip(LR_GRID, pr_means, pr_stds):
    ax2.annotate(f"{m:.4f}\n±{s:.4f}", xy=(lr, m), xytext=(0, 22),
                 textcoords="offset points", ha="center", fontsize=9.5,
                 color=color_pr, weight="bold")
ax2.axhline(y=0.9143, color="#888", linestyle="--", linewidth=2, alpha=0.85,
            label="LR=5e-4 baseline PR=0.9143", zorder=2)
best_lr_pr = LR_GRID[int(np.argmax(pr_means))]
ax2.scatter([best_lr_pr], [max(pr_means)], s=550, facecolors='none',
            edgecolors=color_pr, linewidths=3.5, zorder=6)
# BEST 标注: 子图正上方居中 (X = LR=1e-3, Y=顶部 0.932)
ax2.annotate("BEST\nLR=1e-3", xy=(1e-3, 0.932),
             ha="center", fontsize=12, color=color_pr, weight="bold")
ax2.set_xscale("log")
ax2.tick_params(axis="y", labelcolor=color_pr)
ax2.set_ylim(0.895, 0.937)
ax2.grid(True, alpha=0.25, linestyle=":")
ax2.set_axisbelow(True)
ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax2.set_title("(b) Test PR-AUC vs LR", fontsize=13, weight="bold", loc="left", pad=8)

# 总标题 + 副标题
fig.suptitle("TCN+SE 23-dim + B=64 (batch sweep champion) — LR Sweep Performance",
             fontsize=15, weight="bold", y=0.995)
delta_f1 = max(f1m_means) - 0.8454
fig.text(0.5, 0.94,
         f"5 seeds × 7 LR = 35 runs | LR=4e-3 wins F1m ({max(f1m_means):.4f}, Delta={delta_f1:+.4f}), "
         f"LR=1e-3 wins PR-AUC ({max(pr_means):.4f})",
         ha="center", fontsize=10.5, style="italic", color="#37474F")

plt.tight_layout(rect=[0, 0, 1, 0.92])
out_png = os.path.join(BASE, "lr_sweep_curve_v2.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {out_png}")
print(f"\nF1m   BEST: LR={LR_GRID[int(np.argmax(f1m_means))]:.0e}  F1m={max(f1m_means):.4f}+/-{f1m_stds[int(np.argmax(f1m_means))]:.4f}  (Delta vs LR=5e-4 = {delta_f1:+.4f})")
print(f"PR-AUC BEST: LR={LR_GRID[int(np.argmax(pr_means))]:.0e}  PR={max(pr_means):.4f}+/-{pr_stds[int(np.argmax(pr_means))]:.4f}")
