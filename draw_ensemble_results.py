#!/usr/bin/env python3
"""
5-seed ensemble analysis: per-seed breakdown + ensemble comparison.

Outputs:
  - tcn_v4_se_ensemble_analysis.png  (4-panel: per-seed F1m/PR-AUC, PR curves, scatter)
  - tcn_v4_se_ensemble_summary.csv
"""

import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_ensemble_analysis.png")
OUT_CSV = os.path.join(BASE, "tcn_v4_se_ensemble_summary.csv")

# Load ensemble meta
with open(os.path.join(BASE, "processed_meta_tcn_v4_se_ensemble5.json")) as f:
    meta = json.load(f)

per_seed = meta["per_seed_metrics"]
ensemble_test = meta["metrics"]["test_at_thr"]
ensemble_test_05 = meta["metrics"]["test_at_0.5"]

# Load test predictions for all 5 seeds + ensemble
preds = {}
for seed_info in per_seed:
    s = seed_info["seed"]
    # Re-train was done in the same script; we only have ensemble predictions saved
    pass
ens_test_pred = pd.read_csv(os.path.join(BASE, "predictions_test_tcn_v4_se_ensemble5.csv"))
preds["Ensemble"] = (ens_test_pred["y_true"].values, ens_test_pred["prob_attack"].values)

# Load single-seed predictions from earlier runs
single_files = {
    "B=128 seed=42":  "predictions_test_tcn_v4_se_b128_e40.csv",
    "B=512 seed=42 (orig)": "predictions_test_tcn_v3_v4se_window16.csv",
    "B=64 seed=42":   "predictions_test_tcn_v4_se_b64_e40.csv",
}
for name, fn in single_files.items():
    p = os.path.join(BASE, fn)
    if os.path.exists(p):
        d = pd.read_csv(p)
        preds[name] = (d["y_true"].values, d["prob_attack"].values)

# ─────────────────────────────────────────────
# Compute per-seed metrics from history (re-train summary)
# ─────────────────────────────────────────────
seeds = [r["seed"] for r in per_seed]
test_f1ms = [r["test_f1m_05"] for r in per_seed]
test_praucs = [r["test_pr_auc"] for r in per_seed]
val_f1ms = [r["best_val_f1m"] for r in per_seed]
train_times = [r["train_time"] for r in per_seed]

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# === Panel (a): Per-seed bar chart ===
ax = axes[0, 0]
x = np.arange(len(seeds))
width = 0.35
bars1 = ax.bar(x - width/2, test_f1ms, width, color="#1565C0", alpha=0.8, label="Test F1m")
bars2 = ax.bar(x + width/2, test_praucs, width, color="#E65100", alpha=0.8, label="Test PR-AUC")
# Ensemble reference
ax.axhline(ensemble_test["macro_f1"], color="#1565C0", linestyle="--", linewidth=1.5, alpha=0.6,
           label=f"Ensemble F1m = {ensemble_test['macro_f1']:.4f}")
ax.axhline(ensemble_test["pr_auc"], color="#E65100", linestyle="--", linewidth=1.5, alpha=0.6,
           label=f"Ensemble PR-AUC = {ensemble_test['pr_auc']:.4f}")
# Annotate best seed
best_seed_idx = int(np.argmax(test_f1ms))
ax.bar([best_seed_idx - width/2], [test_f1ms[best_seed_idx]], width, color="#FFD600", edgecolor="black", linewidth=1.5)
ax.annotate(f"best seed", xy=(best_seed_idx - width/2, test_f1ms[best_seed_idx]),
            xytext=(0, 10), textcoords="offset points", fontsize=9, ha="center",
            weight="bold")
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seeds])
ax.set_xlabel("Random Seed", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("(a) Per-Seed Test Performance", fontsize=12, weight="bold", loc="left")
ax.set_ylim(0.83, 0.93)
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax.grid(True, alpha=0.3, axis="y", linestyle=":")

# === Panel (b): Ensemble vs single best scatter ===
ax = axes[0, 1]
# Per-seed points
for s, f, p in zip(seeds, test_f1ms, test_praucs):
    ax.scatter(p, f, s=120, color="#9E9E9E", marker="o", alpha=0.6, edgecolors="black", linewidths=1)
    ax.annotate(f"{s}", xy=(p, f), xytext=(5, 5), textcoords="offset points",
                fontsize=9, color="#616161")
# Ensemble
ax.scatter(ensemble_test["pr_auc"], ensemble_test["macro_f1"], s=300, color="#E65100",
           marker="*", edgecolors="black", linewidths=2, zorder=5,
           label=f"Ensemble")
# Original single B=128
b128_prauc = 0.9122; b128_f1m = 0.8655
ax.scatter(b128_prauc, b128_f1m, s=200, color="#FFD600", marker="D",
           edgecolors="black", linewidths=1.5, label="Single B=128 (best)")
ax.set_xlabel("Test PR-AUC", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(b) Single Seeds vs Ensemble", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

# === Panel (c): PR curves comparison ===
ax = axes[1, 0]
colors_seed = ["#9E9E9E", "#9E9E9E", "#9E9E9E", "#9E9E9E", "#9E9E9E"]
for i, r in enumerate(per_seed):
    s = r["seed"]
    # We don't have per-seed predictions saved, so skip individual PR curves
    pass
# Just plot the 3 saved configurations
configs = [
    ("B=512 seed=42 (orig)",  preds.get("B=512 seed=42 (orig)"),  "#9E9E9E", ":"),
    ("B=64 seed=42",          preds.get("B=64 seed=42"),          "#6A1B9A", "--"),
    ("B=128 seed=42",         preds.get("B=128 seed=42"),         "#FFD600", "-."),
    ("5-Seed Ensemble",       preds.get("Ensemble"),              "#E65100", "-"),
]
for name, (y, p), color, ls in configs:
    if y is None: continue
    pr, rc, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    ax.plot(rc, pr, color=color, linestyle=ls, linewidth=2.2,
            label=f"{name}  (AP = {ap:.4f})")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("(c) PR Curves — Ensemble vs Singles", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.4, 1.02])

# === Panel (d): Per-seed val F1m history ===
ax = axes[1, 1]
for r in per_seed:
    ep = [h["epoch"] for h in r["history"]]
    f1 = [h["val_f1m"] for h in r["history"]]
    ax.plot(ep, f1, "-", linewidth=1.5, alpha=0.7, label=f"seed={r['seed']}")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Val Macro-F1", fontsize=11)
ax.set_title("(d) Val F1m Trajectory per Seed", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

# ─────────────────────────────────────────────
fig.suptitle("5-Seed Ensemble — TCN+SE (B=128, ep=40) on 19-dim SCADA",
             fontsize=14, weight="bold", y=0.995)

# Footer verdict
best_seed_f1m = max(test_f1ms)
f1_ensemble = ensemble_test["macro_f1"]
delta_ens_vs_best = f1_ensemble - best_seed_f1m
delta_ens_vs_b128 = f1_ensemble - b128_f1m
verdict = (f"Ensemble F1m = {f1_ensemble:.4f} | best single seed = {best_seed_f1m:.4f} "
           f"(Δ = {delta_ens_vs_best:+.4f}) | "
           f"vs B=128 saved best: Δ = {delta_ens_vs_b128:+.4f}")
fig.text(0.5, 0.005, verdict, ha="center", fontsize=10, style="italic", color="#37474F")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")

# Summary CSV
summary = pd.DataFrame({
    "config": ["5-Seed Ensemble"] + [f"seed={r['seed']}" for r in per_seed],
    "type": ["ensemble"] + ["single"] * 5,
    "test_f1m": [f1_ensemble] + test_f1ms,
    "test_pr_auc": [ensemble_test["pr_auc"]] + test_praucs,
    "val_f1m": [meta["metrics"]["val_at_thr"]["macro_f1"]] + val_f1ms,
    "best_val_epoch": [None] + [r["best_epoch"] for r in per_seed],
    "train_time_s": [meta["total_train_time_seconds"] / 5] + train_times,
})
summary.to_csv(OUT_CSV, index=False)
print(f"[saved] {OUT_CSV}")
print()
print("=" * 80)
print(summary.to_string(index=False))
print("=" * 80)