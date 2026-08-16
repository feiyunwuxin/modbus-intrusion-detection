#!/usr/bin/env python3
"""
Focused diagram: Position 1 — SE (Squeeze-and-Excitation) channel attention
inside a TCN residual block.

Layout:
  - Top half: the TCN residual block (Conv1d → BN → ReLU → Drop ×2 + SE + Add + ReLU)
  - Right half: SE block expanded into 5 ops (GAP → FC → ReLU → FC → Sigmoid → Scale)
  - Bottom: data shape flow with tensor dimensions

Output: tcn_se_position1.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BASE = r"C:\work\Claude\Issue"
OUT = os.path.join(BASE, "tcn_se_position1.png")

C = {
    "input":    "#E3F2FD",
    "conv":     "#1565C0",
    "bn":       "#42A5F5",
    "relu":     "#90CAF9",
    "drop":     "#FFCC80",
    "se":       "#C62828",     # red highlight for SE
    "add":      "#6A1B9A",
    "shortcut": "#9E9E9E",
    "arrow":    "#212121",
    "scale":    "#AD1457",
    "label":    "#37474F",
    "title":    "#1A237E",
}


def box(ax, x, y, w, h, text, fc, fontsize=9, tc="white", weight="bold"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.05",
                       facecolor=fc, edgecolor="black", linewidth=1.0)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.6, style="-|>", label=None, label_pos=0.5):
    if color is None:
        color = C["arrow"]
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=14,
                        color=color, linewidth=lw)
    ax.add_patch(a)
    if label:
        mx, my = x1 + (x2-x1)*label_pos, y1 + (y2-y1)*label_pos
        ax.text(mx + 0.08, my + 0.08, label, fontsize=7, color=color, style="italic")


# ═════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(8, 9.5, "Position 1: SE (Squeeze-and-Excitation) inside TCN Residual Block",
        ha="center", va="center", fontsize=15, weight="bold", color=C["title"])
ax.text(8, 9.05, "Channel attention — learns per-channel importance dynamically",
        ha="center", va="center", fontsize=10, style="italic", color="#555")

# ═════════════════════════════════════════════════════════════
# LEFT: TCN residual block (compressed view)
# ═════════════════════════════════════════════════════════════
# main trunk (top) — goes through conv1+conv2
# shortcut (bottom) — identity
# join at right (Add ⊕)

y_trunk = 6.3
y_shortcut = 4.6
y_join = 5.5

# Input
box(ax, 0.3, y_join - 0.3, 1.7, 0.7, "Input\n(B, C, T)\nC = 64", C["input"], 9, "black")

# TRUNK: Conv1 → BN → ReLU → Drop → Conv2 → BN → ReLU → Drop → SE
# (compress into one row of labeled boxes for clarity)
y = y_trunk
box(ax, 2.3, y, 1.4, 0.7, "Conv1d\nk=3, d=d", C["conv"], 8)
box(ax, 3.85, y, 0.7, 0.7, "BN", C["bn"], 9)
box(ax, 4.7, y, 0.8, 0.7, "ReLU", C["relu"], 8, "black")
box(ax, 5.65, y, 0.95, 0.7, "Drop 0.3", C["drop"], 8, "black")
box(ax, 6.75, y, 1.4, 0.7, "Conv1d\nk=3, d=d", C["conv"], 8)
box(ax, 8.3, y, 0.7, 0.7, "BN", C["bn"], 9)
box(ax, 9.15, y, 0.8, 0.7, "ReLU", C["relu"], 8, "black")
box(ax, 10.1, y, 0.95, 0.7, "Drop 0.3", C["drop"], 8, "black")

# SE (highlighted!)
box(ax, 11.2, y - 0.1, 1.5, 0.9, "SE Block", C["se"], 10)

# Add ⊕
box(ax, 12.85, y_join - 0.35, 1.0, 0.8, "Add ⊕", C["add"], 10)

# ReLU (post-add)
box(ax, 14.0, y_join - 0.35, 0.8, 0.8, "ReLU", C["relu"], 9, "black")

# Output
box(ax, 14.95, y_join - 0.35, 1.0, 0.8, "Output\n(B, C, T)", C["input"], 9, "black")

# Trunk arrows
arrow(ax, 2.0, y_join, 2.3, y + 0.35)
arrow(ax, 3.7, y + 0.35, 3.85, y + 0.35)
arrow(ax, 4.55, y + 0.35, 4.7, y + 0.35)
arrow(ax, 5.5, y + 0.35, 5.65, y + 0.35)
arrow(ax, 6.6, y + 0.35, 6.75, y + 0.35)
arrow(ax, 8.15, y + 0.35, 8.3, y + 0.35)
arrow(ax, 9.0, y + 0.35, 9.15, y + 0.35)
arrow(ax, 9.95, y + 0.35, 10.1, y + 0.35)
arrow(ax, 11.05, y + 0.35, 11.2, y + 0.4)
arrow(ax, 12.7, y + 0.35, 12.85, y_join)
arrow(ax, 13.85, y_join + 0.05, 14.0, y_join + 0.05)
arrow(ax, 14.8, y_join + 0.05, 14.95, y_join + 0.05)

# Shortcut path (bottom)
box(ax, 2.3, y_shortcut, 1.4, 0.7, "Identity", C["shortcut"], 9, "black")
arrow(ax, 2.0, y_join - 0.3, 2.3, y_shortcut + 0.35, color=C["shortcut"], lw=1.2)
arrow(ax, 3.7, y_shortcut + 0.35, 13.05, y_shortcut + 0.35, color=C["shortcut"], lw=1.2)
arrow(ax, 13.05, y_shortcut + 0.35, 13.0, y_join - 0.1, color=C["shortcut"], lw=1.2)
ax.text(7.5, y_shortcut + 0.85, "shortcut (identity or 1×1 Conv if C_in ≠ C_out)",
        fontsize=8, color=C["shortcut"], style="italic")

# Label "Main trunk" / "Shortcut"
ax.text(0.0, y + 0.5, "Main\ntrunk", fontsize=9, weight="bold", color=C["conv"],
        ha="left", va="center")
ax.text(0.0, y_shortcut + 0.5, "Shortcut", fontsize=9, weight="bold",
        color=C["shortcut"], ha="left", va="center")

# ═════════════════════════════════════════════════════════════
# BOTTOM: SE Block expanded — 6 steps
# ═════════════════════════════════════════════════════════════
se_y = 2.8
se_h = 0.55
se_x = 0.5
se_w = 2.2
se_gap = 0.05

# Header
ax.text(8.0, 3.7, "SE Block expanded — channel attention mechanism",
        ha="center", va="center", fontsize=11, weight="bold", color=C["se"])

# Box 1: Input (B, C, T)
box(ax, se_x + 0*(se_w+se_gap), se_y, se_w, se_h, "Input\n(B, C=64, T=16)", C["input"], 8, "black")

# Box 2: GAP → squeeze
box(ax, se_x + 1*(se_w+se_gap), se_y, se_w, se_h, "Squeeze:\nGAP → (B, C)", C["se"], 8)

# Box 3: FC C → C/r
box(ax, se_x + 2*(se_w+se_gap), se_y, se_w, se_h, "Excitation:\nFC(C → C/r=8)", C["se"], 8)

# Box 4: ReLU
box(ax, se_x + 3*(se_w+se_gap), se_y, se_w, se_h, "ReLU", C["relu"], 8, "black")

# Box 5: FC C/r → C
box(ax, se_x + 4*(se_w+se_gap), se_y, se_w, se_h, "FC(C/r → C)", C["se"], 8)

# Box 6: Sigmoid → weights (B, C)
box(ax, se_x + 5*(se_w+se_gap), se_y, se_w, se_h, "Sigmoid →\n(B, C)", C["se"], 8)

# Arrows between SE sub-steps
for i in range(5):
    x1 = se_x + (i+1)*se_w + i*se_gap
    x2 = se_x + (i+1)*(se_w + se_gap)
    arrow(ax, x1, se_y + se_h/2, x2, se_y + se_h/2)

# Box 7: Scale (multiply input by weights) — bring back to (B, C, T)
box(ax, se_x + 6*(se_w+se_gap), se_y, se_w, se_h, "Scale ⊗\nInput × weights\n→ (B, C, T)", C["scale"], 8)

# Curved arrow back to scale
arrow(ax, se_x + 6*se_w + 5*se_gap, se_y + se_h + 0.05,
      se_x + 6*se_w + 6*se_gap + 0.5, se_y + se_h + 0.05,
      color=C["scale"], lw=1.4)
ax.text(se_x + 6*se_w + 5.5*se_gap, se_y + se_h + 0.3,
        "↑ weights broadcast along T",
        fontsize=7, color=C["scale"], style="italic", ha="center")

# Label Squeeze vs Excitation zones
ax.annotate("", xy=(se_x + 1*(se_w+se_gap) - 0.1, se_y - 0.3),
            xytext=(se_x + 0.0, se_y - 0.3),
            arrowprops=dict(arrowstyle="->", color=C["se"], lw=1.0))
ax.text(se_x + 0.55, se_y - 0.55, "Squeeze", fontsize=9, weight="bold", color=C["se"], ha="center")

ax.annotate("", xy=(se_x + 5*(se_w+se_gap) - 0.1, se_y - 0.3),
            xytext=(se_x + 2*(se_w+se_gap), se_y - 0.3),
            arrowprops=dict(arrowstyle="->", color=C["se"], lw=1.0))
ax.text(se_x + 3.5*(se_w+se_gap), se_y - 0.55, "Excitation",
        fontsize=9, weight="bold", color=C["se"], ha="center")

# ═════════════════════════════════════════════════════════════
# Parameter count annotation (top-right)
# ═════════════════════════════════════════════════════════════
ax.text(15.5, 8.7, "Param cost per SE:\nC × (C/r) + (C/r) × C\n= 64×8 + 8×64\n= 1,024 params",
        ha="center", va="center", fontsize=8, color=C["se"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C["se"], linewidth=1.2))

ax.text(15.5, 7.8, "Total in model:\n3 blocks × 1,024\n= 3,072 params",
        ha="center", va="center", fontsize=8, color=C["add"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C["add"], linewidth=1.2))

# ═════════════════════════════════════════════════════════════
# Bottom key insight
# ═════════════════════════════════════════════════════════════
ax.text(8, 1.4, "Why here?  conv has already extracted 'what signals matter' — SE has meaning to learn",
        ha="center", va="center", fontsize=10, weight="bold", color=C["title"])
ax.text(8, 1.0,
        "After ReLU  ->  negative signal clipped  ->  SE can't recover  |  After residual add  ->  asymmetric weighting  ->  messy gradients",
        ha="center", va="center", fontsize=8, color="#555", style="italic")
ax.text(8, 0.55,
        "Channel attention dynamically weights SCADA signals per sample — focus on setpoint / command_response channels",
        ha="center", va="center", fontsize=9, color=C["se"])

# ═════════════════════════════════════════════════════════════
# Legend
# ═════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(facecolor=C["conv"],     label="Conv1d"),
    mpatches.Patch(facecolor=C["bn"],       label="BatchNorm"),
    mpatches.Patch(facecolor=C["relu"],     label="ReLU"),
    mpatches.Patch(facecolor=C["drop"],     label="Dropout"),
    mpatches.Patch(facecolor=C["se"],       label="SE Operations"),
    mpatches.Patch(facecolor=C["scale"],    label="Channel-wise Scale ⊗"),
    mpatches.Patch(facecolor=C["add"],      label="Add ⊕ (residual)"),
    mpatches.Patch(facecolor=C["shortcut"], label="Shortcut"),
    mpatches.Patch(facecolor=C["input"],    label="Tensor"),
]
ax.legend(handles=legend_handles, loc="lower left",
          bbox_to_anchor=(0.0, -0.02), fontsize=9, ncol=3,
          frameon=True, edgecolor="black")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT}")