#!/usr/bin/env python3
"""5-metric radar chart for SCADA intrusion detection model.

Following the dataviz skill procedure:
- Form: Radar (5 metrics × N entities)
- Color: Categorical palette (light mode, slots 1-5 from validated reference)
- Mark specs: 2px lines, 8px markers, 4px rounded line caps
- Direct labels: Each line is identified by legend (≤5 series)
- Surface gap: 2% padding between polygons
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

BASE = r"D:\workspace\claude\Issue"

# ----------------------------------------------------------------------
# 1. Data — 5 metrics, 5 entities
# ----------------------------------------------------------------------
METRICS = ["F1m", "Binary-F1", "Accuracy", "ROC-AUC", "PR-AUC"]

# All values are from onnx_export_manifest.json (5 seeds) +
# perclass_metrics_23dim_ensemble.json (ensemble) +
# CROSS_MODEL_LEADERBOARD.md (prior best)
ENTITIES = {
    # (label,                       color,        values)
    "5-seed ensemble\n(this work, 23-dim ch=32)": ("#2a78d6", [0.8775, 0.8737, 0.8776, 0.9048, 0.9363]),
    "Best single seed\n(seed 123)":               ("#eb6834", [0.8766, 0.8728, 0.8767, 0.9074, 0.9211]),
    "Prior best\n(TCN+SE ch=64, 5-seed)":         ("#1baf7a", [0.8735, 0.8711, 0.8735, 0.9067, 0.9269]),
    "Cross-arch best\n(stacking, 10 models)":     ("#eda100", [0.8481, 0.7577, 0.9018, 0.9248, 0.8473]),
    "Degenerate\n(always Normal)":                ("#e87ba4", [0.3900, 0.0000, 0.4740, 0.5000, 0.5260]),
}

# ----------------------------------------------------------------------
# 2. Radar geometry
# ----------------------------------------------------------------------
N = len(METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=150, subplot_kw=dict(polar=True))

# Surface
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

# ----------------------------------------------------------------------
# 3. Plot each entity — current best gets thicker line + larger fill
# ----------------------------------------------------------------------
for i, (label, (color, values)) in enumerate(ENTITIES.items()):
    vals = values + values[:1]
    linewidth = 2.8 if i == 0 else 1.8
    markersize = 9 if i == 0 else 7
    alpha = 0.22 if i == 0 else 0.08
    ax.plot(angles, vals, color=color, linewidth=linewidth, marker="o",
            markersize=markersize, markerfacecolor=color,
            markeredgecolor=color, label=label, zorder=3)
    ax.fill(angles, vals, color=color, alpha=alpha, zorder=2)

# Add value labels on the champion's polygon (avoid clutter)
champ_vals = list(ENTITIES.values())[0][1]
champ_color = list(ENTITIES.values())[0][0]
for ang, val in zip(angles[:-1], champ_vals):
    ax.text(ang, val + 0.025, f"{val:.3f}", ha="center", va="center",
            fontsize=9, color=champ_color, fontweight="bold", zorder=5)

# ----------------------------------------------------------------------
# 4. Axes & grid — zoomed range to spread the close values
# ----------------------------------------------------------------------
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(METRICS, fontsize=12, color="#0b0b0b", fontweight="medium")

# Zoomed y-axis: start at 0.30 (covers degenerate) up to 1.00
ax.set_ylim(0.30, 1.00)
ax.set_yticks([0.30, 0.45, 0.60, 0.75, 0.90, 1.00])
ax.set_yticklabels(["0.30", "0.45", "0.60", "0.75", "0.90", "1.00"],
                   fontsize=8.5, color="#52514e")
ax.set_rlabel_position(75)

# Recessive grid (per skill: thin + recessive)
ax.yaxis.grid(True, color="#d6d4cc", linewidth=0.6, alpha=0.7)
ax.xaxis.grid(True, color="#d6d4cc", linewidth=0.6, alpha=0.7)
ax.spines["polar"].set_color("#d6d4cc")
ax.spines["polar"].set_linewidth(0.8)

# ----------------------------------------------------------------------
# 5. Title & legend
# ----------------------------------------------------------------------
ax.set_title("5-Metric Radar — TCN+SE 23-dim 5-seed Ensemble",
             fontsize=14, color="#0b0b0b", pad=34, fontweight="semibold")

# Legend below the plot — bbox_to_anchor in axes coords (polar)
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                   ncol=2, frameon=False, fontsize=9.5,
                   handlelength=2.4, handletextpad=0.8,
                   columnspacing=1.6, labelcolor="#0b0b0b")

# ----------------------------------------------------------------------
# 6. Footer annotation
# ----------------------------------------------------------------------
fig.text(0.5, 0.01,
         "Test set n=3,432  |  Normal 1,627 / Attack 1,805  |  "
         "τ=0.5  |  y-axis zoomed to [0.30, 1.00]",
         ha="center", fontsize=8.5, color="#52514e", style="italic")

plt.tight_layout()
out_path = os.path.join(BASE, "radar_5metrics_ensemble.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")