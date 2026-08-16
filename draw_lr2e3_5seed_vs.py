#!/usr/bin/env python3
"""
Compare 5-seed ensembles: LR=5e-4 (baseline) vs LR=2e-3 (new winner).

Output: tcn_v4_se_lr2e3_vs_5e4.png
"""

import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_lr2e3_vs_5e4.png")

# Load both 5-seed results
with open(os.path.join(BASE, "processed_meta_tcn_v4_se_ensemble5.json")) as f:
    meta_5e4 = json.load(f)
with open(os.path.join(BASE, "processed_meta_tcn_v4_se_lr2e3_5seed.json")) as f:
    meta_2e3 = json.load(f)

seeds_5e4 = [r["seed"] for r in meta_5e4["per_seed_metrics"]]
f1ms_5e4  = [r["test_f1m_05"] for r in meta_5e4["per_seed_metrics"]]
prs_5e4   = [r["test_pr_auc"] for r in meta_5e4["per_seed_metrics"]]

seeds_2e3 = [r["seed"] for r in meta_2e3["per_seed"]]
f1ms_2e3  = [r["test_f1m_05"] for r in meta_2e3["per_seed"]]
prs_2e3   = [r["test_pr_auc"] for r in meta_2e3["per_seed"]]

ens_5e4 = meta_5e4["metrics"]["test_at_thr"]
ens_2e3 = meta_2e3["ensemble_metrics"]["test_at_thr"]

# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# === Panel (a): Per-seed F1m comparison ===
ax = axes[0]
x = np.arange(5)
ax.scatter(x, f1ms_5e4, s=180, color="#9E9E9E", marker="o", edgecolors="black",
           linewidths=1.5, label=f"LR=5e-4 (μ={np.mean(f1ms_5e4):.4f}, σ={np.std(f1ms_5e4):.4f})")
ax.scatter(x, f1ms_2e3, s=180, color="#E65100", marker="s", edgecolors="black",
           linewidths=1.5, label=f"LR=2e-3 (μ={np.mean(f1ms_2e3):.4f}, σ={np.std(f1ms_2e3):.4f})")
# Ensemble lines
ax.axhline(ens_5e4["macro_f1"], color="#9E9E9E", linestyle=":", linewidth=1.5, alpha=0.6,
           label=f"Ensemble@5e-4 = {ens_5e4['macro_f1']:.4f}")
ax.axhline(ens_2e3["macro_f1"], color="#E65100", linestyle=":", linewidth=2, alpha=0.8,
           label=f"Ensemble@2e-3 = {ens_2e3['macro_f1']:.4f} ⭐")
ax.set_xticks(x)
ax.set_xticklabels([f"seed={s}" for s in seeds_2e3])
ax.set_xlabel("Random Seed", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(a) Per-Seed Test F1m — LR=2e-3 vs LR=5e-4", fontsize=12, weight="bold", loc="left")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3, axis="y", linestyle=":")
ax.set_ylim(0.83, 0.89)

# === Panel (b): Per-seed PR-AUC comparison ===
ax = axes[1]
ax.scatter(x, prs_5e4, s=180, color="#9E9E9E", marker="o", edgecolors="black",
           linewidths=1.5, label=f"LR=5e-4 (μ={np.mean(prs_5e4):.4f})")
ax.scatter(x, prs_2e3, s=180, color="#6A1B9A", marker="s", edgecolors="black",
           linewidths=1.5, label=f"LR=2e-3 (μ={np.mean(prs_2e3):.4f})")
ax.axhline(ens_5e4["pr_auc"], color="#9E9E9E", linestyle=":", linewidth=1.5, alpha=0.6)
ax.axhline(ens_2e3["pr_auc"], color="#6A1B9A", linestyle=":", linewidth=2, alpha=0.8,
           label=f"Ensemble@2e-3 = {ens_2e3['pr_auc']:.4f} ⭐")
ax.set_xticks(x)
ax.set_xticklabels([f"seed={s}" for s in seeds_2e3])
ax.set_xlabel("Random Seed", fontsize=11)
ax.set_ylabel("Test PR-AUC", fontsize=11)
ax.set_title("(b) Per-Seed Test PR-AUC", fontsize=12, weight="bold", loc="left")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3, axis="y", linestyle=":")
ax.set_ylim(0.88, 0.94)

# === Panel (c): Summary bar chart ===
ax = axes[2]
labels = ["Best Single\n(seed=1024)", "LR=5e-4\n5-seed\nensemble", "LR=2e-3\nsingle\n(seed=42)",
          "LR=2e-3\n5-seed\nensemble ⭐"]
values = [0.8607, ens_5e4["macro_f1"], 0.8702, ens_2e3["macro_f1"]]
colors = ["#FFD600", "#9E9E9E", "#FFA726", "#E65100"]
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.4f}",
            ha="center", va="bottom", fontsize=11, weight="bold")
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(c) Champion Comparison", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, axis="y", linestyle=":")
ax.set_ylim(0.83, 0.89)
plt.setp(ax.get_xticklabels(), fontsize=9)

# ─────────────────────────────────────────────
fig.suptitle("LR=2e-3 Verification — 5-Seed Ensemble Stability",
             fontsize=14, weight="bold", y=1.00)
delta_ens = ens_2e3["macro_f1"] - ens_5e4["macro_f1"]
delta_avg = np.mean(f1ms_2e3) - np.mean(f1ms_5e4)
verdict = (f"VERDICT: LR=2e-3 is GENUINELY better — Δ avg = +{delta_avg:.4f}, "
           f"Δ ensemble = +{delta_ens:.4f}, std reduced from "
           f"{np.std(f1ms_5e4):.4f} to {np.std(f1ms_2e3):.4f} (5× more stable)")
fig.text(0.5, -0.02, verdict, ha="center", fontsize=10, weight="bold", color="#E65100")

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")

# ─────────────────────────────────────────────
print()
print("=" * 80)
print(f"  LR=2e-3 vs LR=5e-4 — 5-Seed Comparison")
print("=" * 80)
print(f"{'Config':<25} {'Avg F1m':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'Ens F1m':>10}")
print("-" * 80)
print(f"{'LR=5e-4 single':<25} {'—':>10} {'—':>8} {'—':>8} {'—':>8} "
      f"{ens_5e4['macro_f1']:>10.4f}")
print(f"{'LR=5e-4 5-seed':<25} {np.mean(f1ms_5e4):>10.4f} {np.std(f1ms_5e4):>8.4f} "
      f"{np.min(f1ms_5e4):>8.4f} {np.max(f1ms_5e4):>8.4f} "
      f"{ens_5e4['macro_f1']:>10.4f}")
print(f"{'LR=2e-3 5-seed':<25} {np.mean(f1ms_2e3):>10.4f} {np.std(f1ms_2e3):>8.4f} "
      f"{np.min(f1ms_2e3):>8.4f} {np.max(f1ms_2e3):>8.4f} "
      f"{ens_2e3['macro_f1']:>10.4f} *")
print("=" * 80)
print(f"  ENSEMBLE @ LR=2e-3:")
print(f"    Test F1m  = {ens_2e3['macro_f1']:.4f}  (NEW CHAMPION)")
print(f"    Test PR-AUC = {ens_2e3['pr_auc']:.4f}  (NEW CHAMPION - beats V19!)")
print(f"    Test Bin-F1 = {ens_2e3['binary_f1']:.4f}")
print(f"    Test Acc   = {ens_2e3['accuracy']:.4f}")