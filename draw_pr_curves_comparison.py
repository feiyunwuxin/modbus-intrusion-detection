#!/usr/bin/env python3
"""
Plot actual Precision-Recall curves for the 3 TCN+SE configurations.

Unlike the horizontal lines in panel (c) of the previous comparison,
PR curves show the full trade-off between precision and recall at
every possible threshold — making model differences much clearer.

Output: tcn_v4_se_pr_curves.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_pr_curves.png")

# Load predictions
configs = [
    {
        "name": "Original (B=512, ep=35)",
        "color": "#1565C0",
        "ls": "-",
        "csv": os.path.join(BASE, "predictions_test_tcn_v3_v4se_window16.csv"),
        "meta": os.path.join(BASE, "processed_meta_tcn_v3_v4se_window16.json"),
    },
    {
        "name": "Cosine 50 (B=512, ep=50)",
        "color": "#6A1B9A",
        "ls": "--",
        "csv": os.path.join(BASE, "predictions_test_tcn_v4_se_cosine50.csv"),
        "meta": os.path.join(BASE, "processed_meta_tcn_v4_se_cosine50.json"),
    },
    {
        "name": "Small Batch (B=128, ep=40)",
        "color": "#E65100",
        "ls": "-",
        "csv": os.path.join(BASE, "predictions_test_tcn_v4_se_b128_e40.csv"),
        "meta": os.path.join(BASE, "processed_meta_tcn_v4_se_b128_e40.json"),
    },
]

# Load each
data = []
for c in configs:
    df = pd.read_csv(c["csv"])
    with open(c["meta"]) as f:
        meta = json.load(f)
    c["y_true"] = df["y_true"].values
    c["prob"]   = df["prob_attack"].values
    c["meta"]   = meta
    data.append(c)

# ─────────────────────────────────────────────
# Plot — 1 figure with 3 subplots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# === Left panel: PR curves ===
ax = axes[0]
for c in data:
    p, r, _ = precision_recall_curve(c["y_true"], c["prob"])
    ap = average_precision_score(c["y_true"], c["prob"])
    ax.plot(r, p, color=c["color"], linestyle=c["ls"], linewidth=2.2,
            label=f"{c['name']}  (AP = {ap:.4f})")
    # Mark best threshold
    best_thr = c["meta"]["best_threshold"]
    # Find the point closest to this threshold
    df = pd.read_csv(c["csv"])
    mask = (df["prob_attack"] >= best_thr).astype(int)
    from sklearn.metrics import precision_score, recall_score
    prec_at = precision_score(c["y_true"], mask, zero_division=0)
    rec_at = recall_score(c["y_true"], mask)
    ax.scatter([rec_at], [prec_at], s=120, color=c["color"], marker="*",
               edgecolors="black", linewidths=1.2, zorder=5)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("(a) Precision-Recall Curves (test set)", fontsize=13, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.4, 1.02])

# Annotation: explain star markers
ax.text(0.98, 0.05,
        "* = operating point at\nbest val threshold",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, style="italic", color="#37474F",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#BDBDBD", linewidth=1))

# === Right panel: zoomed-in top region (high precision) ===
ax = axes[1]
for c in data:
    p, r, _ = precision_recall_curve(c["y_true"], c["prob"])
    ap = average_precision_score(c["y_true"], c["prob"])
    # Sort by descending precision
    order = np.argsort(-p)
    p_sorted = p[order]
    r_sorted = r[order]
    ax.plot(r_sorted, p_sorted, color=c["color"], linestyle=c["ls"], linewidth=2.2,
            label=f"{c['name']}  (AP = {ap:.4f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("(b) PR Curves (top region: Precision ≥ 0.7)", fontsize=13, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
ax.set_xlim([0.0, 0.6])
ax.set_ylim([0.7, 1.0])
# Add a horizontal line at precision=0.85
ax.axhline(0.85, color="grey", linestyle=":", linewidth=1.0, alpha=0.5)
ax.text(0.55, 0.855, "Precision = 0.85", fontsize=8, color="grey", style="italic")

# ─────────────────────────────────────────────
fig.suptitle("TCN+SE 19-dim SCADA — Precision-Recall Curves",
             fontsize=15, weight="bold", y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")

# Print summary table
print("\n" + "=" * 80)
print(f"{'Config':<35} {'AP':>8} {'Best Thr':>10} {'Prec@best':>10} {'Rec@best':>10}")
print("-" * 80)
for c in data:
    ap = average_precision_score(c["y_true"], c["prob"])
    df = pd.read_csv(c["csv"])
    mask = (df["prob_attack"] >= c["meta"]["best_threshold"]).astype(int)
    from sklearn.metrics import precision_score, recall_score
    prec_at = precision_score(c["y_true"], mask, zero_division=0)
    rec_at = recall_score(c["y_true"], mask)
    print(f"{c['name']:<35} {ap:>8.4f} {c['meta']['best_threshold']:>10.2f} {prec_at:>10.4f} {rec_at:>10.4f}")
print("=" * 80)