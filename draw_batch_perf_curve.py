#!/usr/bin/env python3
"""TCN+SE 23-dim batch sweep — clean performance curve

单图 + 多 seed 散点 + F1m/PR-AUC 双轴 + 64-combos B=512 baseline 参考线
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
f1m_means = []; f1m_stds = []
pr_means  = []; pr_stds  = []
for b in BATCHES:
    per = [runs[(b, s)] for s in SEEDS if (b, s) in runs]
    f1m_means.append(np.mean([r["test_macro_f1"] for r in per]))
    f1m_stds.append(np.std([r["test_macro_f1"] for r in per], ddof=0))
    pr_means.append(np.mean([r["test_pr_auc"] for r in per]))
    pr_stds.append(np.std([r["test_pr_auc"] for r in per], ddof=0))

# 画图
fig, ax1 = plt.subplots(figsize=(11, 6.5))

# === 主轴: F1m (左) ===
color_f1 = "#E65100"
ax1.set_xlabel("Batch Size", fontsize=13, weight="bold")
ax1.set_ylabel("Test Macro-F1 (5-seed mean ± std)", color=color_f1, fontsize=13, weight="bold")
line1 = ax1.errorbar(BATCHES, f1m_means, yerr=f1m_stds, fmt='o-', linewidth=2.8,
                     capsize=6, capthick=2, markersize=12,
                     color=color_f1, markerfacecolor="white", markeredgewidth=2.5,
                     label="Test F1m (5-seed mean ± std)", zorder=5)
# 标 F1m 数值
for b, m, s in zip(BATCHES, f1m_means, f1m_stds):
    ax1.annotate(f"{m:.4f}\n±{s:.4f}", xy=(b, m), xytext=(0, 14),
                 textcoords="offset points", ha="center", fontsize=10,
                 color=color_f1, weight="bold")

# 画每个 seed 的散点
for s in SEEDS:
    seed_f1 = [runs[(b, s)]["test_macro_f1"] for b in BATCHES if (b, s) in runs]
    seed_b  = [b for b in BATCHES if (b, s) in runs]
    seed_color = {42:"#90CAF9", 123:"#A5D6A7", 456:"#FFCC80", 789:"#CE93D8", 1024:"#EF9A9A"}[s]
    ax1.scatter(seed_b, seed_f1, s=70, color=seed_color, edgecolors="black",
                linewidths=0.8, alpha=0.7, zorder=4,
                label=f"seed={s} per-run" if s == 42 else None)

# 64-combos B=512 baseline 参考线
ax1.axhline(y=0.8287, color="#888", linestyle="--", linewidth=2, alpha=0.85,
            label="64-combos B=512 baseline F1m=0.8287", zorder=2)

# 高亮 B=64 best
best_b = BATCHES[np.argmax(f1m_means)]
ax1.scatter([best_b], [max(f1m_means)], s=600, facecolors='none',
            edgecolors=color_f1, linewidths=3.5, zorder=6)
ax1.annotate("BEST\nB=64", xy=(best_b, max(f1m_means)),
             xytext=(best_b+15, max(f1m_means)+0.005),
             fontsize=12, color=color_f1, weight="bold")

ax1.set_xticks(BATCHES)
ax1.tick_params(axis="y", labelcolor=color_f1)
ax1.set_ylim(0.820, 0.860)
ax1.grid(True, alpha=0.25, linestyle=":")
ax1.set_axisbelow(True)

# === 次轴: PR-AUC (右) ===
color_pr = "#2E7D32"
ax2 = ax1.twinx()
ax2.set_ylabel("Test PR-AUC (5-seed mean ± std)", color=color_pr, fontsize=13, weight="bold")
line2 = ax2.errorbar(BATCHES, pr_means, yerr=pr_stds, fmt='s--', linewidth=2.2,
                     capsize=5, capthick=1.5, markersize=10,
                     color=color_pr, markerfacecolor="white", markeredgewidth=2,
                     label="Test PR-AUC (5-seed mean ± std)", zorder=3, alpha=0.9)
for b, m, s in zip(BATCHES, pr_means, pr_stds):
    ax2.annotate(f"{m:.4f}\n±{s:.4f}", xy=(b, m), xytext=(0, -22),
                 textcoords="offset points", ha="center", fontsize=9,
                 color=color_pr, weight="bold")
# 64-combos B=512 PR-AUC baseline
ax2.axhline(y=0.9079, color="#888", linestyle=":", linewidth=1.8, alpha=0.7,
            label="64-combos B=512 PR-AUC=0.9079", zorder=2)
ax2.set_ylim(0.890, 0.930)
ax2.tick_params(axis="y", labelcolor=color_pr)

# 合并 legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# 简化 legend: 取 F1m 主 + baseline 参考 + PR-AUC 主
final_lines = [line1, lines1[1] if len(lines1)>1 else None, line2, lines2[1] if len(lines2)>1 else None]
final_labels = [labels1[0], labels1[1] if len(labels1)>1 else "",
                labels2[0], labels2[1] if len(labels2)>1 else ""]
final_lines = [l for l in final_lines if l is not None]
final_labels = [l for l in final_labels if l]
ax1.legend(final_lines, final_labels, loc="lower right", fontsize=9.5,
           framealpha=0.95, ncol=1)

# 标题 + 副标题
fig.suptitle("TCN+SE 23-dim (64-combos Champion) — Batch Size Performance Curve",
             fontsize=15, weight="bold", y=0.97)
delta = max(f1m_means) - 0.8287
ax1.set_title(f"5 seeds × 5 batches = 25 runs | B=64 BEST (F1m={max(f1m_means):.4f}±{f1m_stds[np.argmax(f1m_means)]:.4f}, "
             f"Δ vs B=512 = {delta:+.4f}, +{delta*100:.1f}%)",
             fontsize=11, style="italic", color="#37474F", pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_png = os.path.join(BASE, "batch_perf_curve.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {out_png}")

# Summary table
print("\n" + "="*100)
print(f"{'Batch':>6}  {'F1m_mean':>10}  {'F1m_std':>9}  {'PR_mean':>9}  {'PR_std':>8}  {'ΔF1m vs B=512':>14}")
print("-"*100)
for i, b in enumerate(BATCHES):
    df1 = f1m_means[i] - 0.8287
    print(f"{b:>6}  {f1m_means[i]:>10.4f}  {f1m_stds[i]:>9.4f}  {pr_means[i]:>9.4f}  {pr_stds[i]:>8.4f}  {df1:>+14.4f}")
print("="*100)
print(f"\n[BEST] B=64 -> F1m={f1m_means[0]:.4f}+/-{f1m_stds[0]:.4f}  (DeltaF1m vs B=512 = {f1m_means[0]-0.8287:+.4f})")
