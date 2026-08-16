#!/usr/bin/env python3
"""
Draw TCN+SE architecture diagram using matplotlib.
Outputs two PNGs:
  - tcn_se_architecture.png          : full model (overview)
  - tcn_se_architecture_detailed.png : zoomed into one TCN+SE block + SE block

Designed for paper/report insertion (16:9 ratio, high DPI).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

BASE = r"C:\work\Claude\Issue"
OUT_FULL    = os.path.join(BASE, "tcn_se_architecture.png")
OUT_DETAIL  = os.path.join(BASE, "tcn_se_architecture_detailed.png")

# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────
COLORS = {
    "input":    "#E3F2FD",  # light blue
    "conv":     "#1565C0",  # blue
    "bn":       "#42A5F5",  # light blue
    "relu":     "#90CAF9",  # lighter blue
    "drop":     "#FFCC80",  # orange
    "se":       "#C62828",  # red (highlight)
    "add":      "#6A1B9A",  # purple
    "gap":      "#2E7D32",  # green
    "fc":       "#558B2F",  # dark green
    "output":   "#FFF59D",  # yellow
    "shortcut": "#9E9E9E",  # grey
    "arrow":    "#212121",
}


def draw_box(ax, x, y, w, h, text, color, fontsize=9, textcolor="white", weight="bold"):
    """Draw a rounded box with text inside."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor="black", linewidth=1.0)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, weight=weight)


def draw_arrow(ax, x1, y1, x2, y2, color=None, lw=1.5, style="-|>"):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS["arrow"]
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=15,
                          color=color, linewidth=lw)
    ax.add_patch(arr)


# ═════════════════════════════════════════════════════════════
# FIGURE 1: Full model overview
# ═════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

ax.text(8, 11.6, "TCN+SE Architecture  (Macro-F1 Champion 0.8444 @ 19-dim SCADA)",
        ha="center", va="center", fontsize=16, weight="bold")
ax.text(8, 11.1, "Window=16 timesteps  |  19 input features  |  Output: P(attack)",
        ha="center", va="center", fontsize=11, style="italic", color="#555")

# ── Input box ──
draw_box(ax, 0.3, 9.0, 2.4, 1.4, "Input\n(B, 19, 16)\n[19 features ×\n16 timesteps]",
         COLORS["input"], fontsize=10, textcolor="black")

# ── TCN Block 1 ──
block_y = 7.5
draw_box(ax, 4.0, block_y, 9.0, 1.7,
         "TCN Block 1   dilation=1  (local patterns, RF +4)",
         COLORS["conv"], fontsize=11)
# inner sub-blocks
draw_box(ax, 4.2, block_y + 0.05, 2.5, 0.4, "Conv1d(19→64, k=3)", COLORS["conv"], fontsize=7, textcolor="white")
draw_box(ax, 6.85, block_y + 0.05, 0.6, 0.4, "BN", COLORS["bn"], fontsize=7)
draw_box(ax, 7.55, block_y + 0.05, 0.55, 0.4, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")
draw_box(ax, 8.20, block_y + 0.05, 0.7, 0.4, "Drop 0.3", COLORS["drop"], fontsize=7, textcolor="black")
draw_box(ax, 4.2, block_y + 0.55, 2.5, 0.4, "Conv1d(64→64, k=3)", COLORS["conv"], fontsize=7, textcolor="white")
draw_box(ax, 6.85, block_y + 0.55, 0.6, 0.4, "BN", COLORS["bn"], fontsize=7)
draw_box(ax, 7.55, block_y + 0.55, 0.55, 0.4, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")
draw_box(ax, 8.20, block_y + 0.55, 0.7, 0.4, "Drop 0.3", COLORS["drop"], fontsize=7, textcolor="black")
draw_box(ax, 9.0, block_y + 0.05, 1.7, 0.9, "SE Block\n(r=8)", COLORS["se"], fontsize=8)
draw_box(ax, 10.85, block_y + 0.05, 1.4, 0.9, "Add +\nResidual", COLORS["add"], fontsize=8)
draw_box(ax, 12.35, block_y + 0.05, 0.5, 0.9, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")

# shortcut for block 1
draw_arrow(ax, 2.7, 9.0, 11.2, 8.45, color=COLORS["shortcut"], lw=1.0, style="-|>")
ax.text(6.0, 8.85, "shortcut: Conv1d(19→64, k=1)", fontsize=7, color=COLORS["shortcut"], style="italic")

# ── TCN Block 2 ──
block_y = 5.5
draw_box(ax, 4.0, block_y, 9.0, 1.7,
         "TCN Block 2   dilation=2  (medium patterns, RF +8)",
         COLORS["conv"], fontsize=11)
draw_box(ax, 4.2, block_y + 0.05, 2.5, 0.4, "Conv1d(64→64, k=3, d=2)", COLORS["conv"], fontsize=7, textcolor="white")
draw_box(ax, 6.85, block_y + 0.05, 0.6, 0.4, "BN", COLORS["bn"], fontsize=7)
draw_box(ax, 7.55, block_y + 0.05, 0.55, 0.4, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")
draw_box(ax, 8.20, block_y + 0.05, 0.7, 0.4, "Drop 0.3", COLORS["drop"], fontsize=7, textcolor="black")
draw_box(ax, 4.2, block_y + 0.55, 2.5, 0.4, "Conv1d(64→64, k=3, d=2)", COLORS["conv"], fontsize=7, textcolor="white")
draw_box(ax, 6.85, block_y + 0.55, 0.6, 0.4, "BN", COLORS["bn"], fontsize=7)
draw_box(ax, 7.55, block_y + 0.55, 0.55, 0.4, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")
draw_box(ax, 8.20, block_y + 0.55, 0.7, 0.4, "Drop 0.3", COLORS["drop"], fontsize=7, textcolor="black")
draw_box(ax, 9.0, block_y + 0.05, 1.7, 0.9, "SE Block\n(r=8)", COLORS["se"], fontsize=8)
draw_box(ax, 10.85, block_y + 0.05, 1.4, 0.9, "Add +\nIdentity", COLORS["add"], fontsize=8)
draw_box(ax, 12.35, block_y + 0.05, 0.5, 0.9, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")

# shortcut for block 2
draw_arrow(ax, 13.5, 7.5, 12.25, 6.45, color=COLORS["shortcut"], lw=1.0, style="-|>")
ax.text(13.7, 6.95, "identity", fontsize=7, color=COLORS["shortcut"], style="italic")

# ── TCN Block 3 ──
block_y = 3.5
draw_box(ax, 4.0, block_y, 9.0, 1.7,
         "TCN Block 3   dilation=4  (global patterns, RF +16)",
         COLORS["conv"], fontsize=11)
draw_box(ax, 4.2, block_y + 0.05, 2.5, 0.4, "Conv1d(64→64, k=3, d=4)", COLORS["conv"], fontsize=7, textcolor="white")
draw_box(ax, 6.85, block_y + 0.05, 0.6, 0.4, "BN", COLORS["bn"], fontsize=7)
draw_box(ax, 7.55, block_y + 0.05, 0.55, 0.4, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")
draw_box(ax, 8.20, block_y + 0.05, 0.7, 0.4, "Drop 0.3", COLORS["drop"], fontsize=7, textcolor="black")
draw_box(ax, 4.2, block_y + 0.55, 2.5, 0.4, "Conv1d(64→64, k=3, d=4)", COLORS["conv"], fontsize=7, textcolor="white")
draw_box(ax, 6.85, block_y + 0.55, 0.6, 0.4, "BN", COLORS["bn"], fontsize=7)
draw_box(ax, 7.55, block_y + 0.55, 0.55, 0.4, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")
draw_box(ax, 8.20, block_y + 0.55, 0.7, 0.4, "Drop 0.3", COLORS["drop"], fontsize=7, textcolor="black")
draw_box(ax, 9.0, block_y + 0.05, 1.7, 0.9, "SE Block\n(r=8)", COLORS["se"], fontsize=8)
draw_box(ax, 10.85, block_y + 0.05, 1.4, 0.9, "Add +\nIdentity", COLORS["add"], fontsize=8)
draw_box(ax, 12.35, block_y + 0.05, 0.5, 0.9, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")

# shortcut for block 3
draw_arrow(ax, 13.5, 5.5, 12.25, 4.45, color=COLORS["shortcut"], lw=1.0, style="-|>")
ax.text(13.7, 4.95, "identity", fontsize=7, color=COLORS["shortcut"], style="italic")

# ── GAP + FC head ──
draw_box(ax, 4.0, 2.0, 2.5, 0.7, "GAP  (B,64,16)→(B,64)", COLORS["gap"], fontsize=10)
draw_box(ax, 7.0, 2.0, 2.0, 0.7, "FC1: 64→32", COLORS["fc"], fontsize=10)
draw_box(ax, 9.3, 2.0, 1.5, 0.7, "ReLU + Drop", COLORS["relu"], fontsize=9, textcolor="black")
draw_box(ax, 11.1, 2.0, 1.9, 0.7, "FC2: 32→1", COLORS["fc"], fontsize=10)

# ── Output ──
draw_box(ax, 7.5, 0.5, 3.0, 0.9, "Sigmoid → P(attack)\nthreshold tuning",
         COLORS["output"], fontsize=10, textcolor="black")

# ── Vertical arrows ──
draw_arrow(ax, 1.5, 9.0, 1.5, 9.2)             # input to block 1
draw_arrow(ax, 1.5, 9.2, 4.0, 8.45)
draw_arrow(ax, 13.0, 8.45, 13.0, 7.6)          # block 1 to block 2
draw_arrow(ax, 13.0, 7.6, 4.0, 6.45)
draw_arrow(ax, 13.0, 6.45, 13.0, 5.6)          # block 2 to block 3
draw_arrow(ax, 13.0, 5.6, 4.0, 4.45)
draw_arrow(ax, 13.0, 4.45, 13.0, 3.6)          # block 3 to GAP
draw_arrow(ax, 13.0, 3.6, 5.25, 2.7)
draw_arrow(ax, 6.5, 2.35, 7.0, 2.35)           # GAP to FC1
draw_arrow(ax, 9.0, 2.35, 9.3, 2.35)           # FC1 to ReLU
draw_arrow(ax, 10.8, 2.35, 11.1, 2.35)         # ReLU to FC2
draw_arrow(ax, 12.05, 2.0, 9.0, 1.4)           # FC2 to output
draw_arrow(ax, 1.5, 9.0, 1.5, 9.2)             # input (duplicate, harmless)

# ── Side annotation ──
ax.text(15.3, 8.45, "Receptive\nField\n29 ≥ 16", ha="center", va="center",
        fontsize=9, color=COLORS["se"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["se"]))
ax.annotate("", xy=(13.5, 8.45), xytext=(14.8, 8.45),
            arrowprops=dict(arrowstyle="->", color=COLORS["se"], lw=1.5))

ax.text(15.3, 4.45, "Total\nParams\n72,921", ha="center", va="center",
        fontsize=9, color=COLORS["add"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["add"]))
ax.annotate("", xy=(13.5, 4.45), xytext=(14.8, 4.45),
            arrowprops=dict(arrowstyle="->", color=COLORS["add"], lw=1.5))

# ── Legend ──
legend_handles = [
    mpatches.Patch(facecolor=COLORS["conv"], label="Conv1d"),
    mpatches.Patch(facecolor=COLORS["bn"],   label="BatchNorm"),
    mpatches.Patch(facecolor=COLORS["relu"], label="ReLU"),
    mpatches.Patch(facecolor=COLORS["drop"], label="Dropout"),
    mpatches.Patch(facecolor=COLORS["se"],   label="SE Attention"),
    mpatches.Patch(facecolor=COLORS["add"],  label="Residual Add"),
    mpatches.Patch(facecolor=COLORS["gap"],  label="Global Avg Pool"),
    mpatches.Patch(facecolor=COLORS["fc"],   label="Fully Connected"),
    mpatches.Patch(facecolor=COLORS["output"], label="Output (sigmoid)"),
    mpatches.Patch(facecolor=COLORS["shortcut"], label="Shortcut"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=8, ncol=2,
          frameon=True, edgecolor="black")

plt.tight_layout()
plt.savefig(OUT_FULL, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_FULL}")


# ═════════════════════════════════════════════════════════════
# FIGURE 2: Detailed view of one TCN+SE block + SE block
# ═════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(8, 8.5, "TCN+SE Block — Detailed View",
        ha="center", va="center", fontsize=15, weight="bold")
ax.text(8, 8.05, "Input (B, C_in, T) → Output (B, C_out, T)  with same temporal resolution",
        ha="center", va="center", fontsize=10, style="italic", color="#555")

# ── Main flow (left side) ──
x_start = 0.5
y_main = 4.5
draw_box(ax, x_start, y_main - 0.4, 1.5, 0.8, "Input\n(B, C_in, T)", COLORS["input"], fontsize=9, textcolor="black")

# conv1
y1 = 4.7
draw_box(ax, 2.5, y1, 1.8, 0.7, "Conv1d\nk=3, dilation=d", COLORS["conv"], fontsize=8)
# bn1
draw_box(ax, 4.5, y1, 0.9, 0.7, "BN", COLORS["bn"], fontsize=9)
# relu1
draw_box(ax, 5.55, y1, 0.85, 0.7, "ReLU", COLORS["relu"], fontsize=8, textcolor="black")
# drop1
draw_box(ax, 6.55, y1, 1.0, 0.7, "Drop 0.3", COLORS["drop"], fontsize=8, textcolor="black")

# conv2
y2 = 3.7
draw_box(ax, 2.5, y2, 1.8, 0.7, "Conv1d\nk=3, dilation=d", COLORS["conv"], fontsize=8)
draw_box(ax, 4.5, y2, 0.9, 0.7, "BN", COLORS["bn"], fontsize=9)
draw_box(ax, 5.55, y2, 0.85, 0.7, "ReLU", COLORS["relu"], fontsize=8, textcolor="black")
draw_box(ax, 6.55, y2, 1.0, 0.7, "Drop 0.3", COLORS["drop"], fontsize=8, textcolor="black")

# SE block (the centerpiece)
draw_box(ax, 7.8, y2 - 0.2, 2.5, 1.6, "SE Block\n(Squeeze-and-\nExcitation)",
         COLORS["se"], fontsize=10)

# Add (residual)
draw_box(ax, 10.5, y2 - 0.2, 1.3, 1.6, "Add\n⊕\n(Residual)", COLORS["add"], fontsize=10)
draw_box(ax, 12.0, y2 - 0.2, 1.0, 1.6, "ReLU", COLORS["relu"], fontsize=10, textcolor="black")

# Output
draw_box(ax, 13.2, y2 - 0.2, 2.2, 1.6,
         "Output\n(B, C_out, T)\nC_out = 64", COLORS["output"], fontsize=9, textcolor="black")

# ── Main flow arrows ──
draw_arrow(ax, 2.0, 5.05, 2.5, 5.05)   # input → conv1
draw_arrow(ax, 4.3, 5.05, 4.5, 5.05)   # conv1 → bn1
draw_arrow(ax, 5.4, 5.05, 5.55, 5.05)  # bn1 → relu1
draw_arrow(ax, 6.4, 5.05, 6.55, 5.05)  # relu1 → drop1
draw_arrow(ax, 7.05, 4.7, 7.05, 4.05)   # drop1 down to conv2
draw_arrow(ax, 7.05, 4.05, 2.5, 4.05)  # conv2 entry (curved would be better, but ok)
# Better: arrow from drop1 down to conv2 directly
# Actually let me redo with cleaner vertical connections

# ── SE block expanded (right side) ──
se_x = 8.0
draw_box(ax, se_x + 0.3, y2 - 0.5, 2.0, 0.4, "Input (B, C, T)", COLORS["input"], fontsize=7, textcolor="black")
# Squeeze
draw_box(ax, se_x + 0.3, y2 - 1.0, 2.0, 0.4, "GAP  → (B, C)", COLORS["se"], fontsize=7)
# Excitation part 1
draw_box(ax, se_x + 0.3, y2 - 1.5, 2.0, 0.4, "FC: C → C/r", COLORS["se"], fontsize=7)
# Excitation ReLU
draw_box(ax, se_x + 0.3, y2 - 2.0, 2.0, 0.4, "ReLU", COLORS["relu"], fontsize=7, textcolor="black")
# Excitation part 2
draw_box(ax, se_x + 0.3, y2 - 2.5, 2.0, 0.4, "FC: C/r → C", COLORS["se"], fontsize=7)
# Sigmoid
draw_box(ax, se_x + 0.3, y2 - 3.0, 2.0, 0.4, "Sigmoid  → (B, C)", COLORS["se"], fontsize=7)
# Scale
draw_box(ax, se_x + 0.3, y2 - 3.5, 2.0, 0.4, "Scale  ⊗  (B, C, T)", COLORS["add"], fontsize=7)

# arrows inside SE
for ya, yb in [(y2-0.5, y2-1.0), (y2-1.0, y2-1.5), (y2-1.5, y2-2.0),
                (y2-2.0, y2-2.5), (y2-2.5, y2-3.0), (y2-3.0, y2-3.5)]:
    draw_arrow(ax, se_x + 1.3, ya - 0.0, se_x + 1.3, yb + 0.4)

# ── Receptive field annotation ──
ax.text(0.5, 1.2, "Receptive Field Calculation", fontsize=11, weight="bold", color=COLORS["se"])
ax.text(0.5, 0.85, "RF = 1 + Σ(2 × d_i × (k-1))", fontsize=9, family="monospace")
ax.text(0.5, 0.5, "  = 1 + 2·1·2 + 2·2·2 + 2·4·2", fontsize=9, family="monospace")
ax.text(0.5, 0.15, "  = 29  ≥  Window=16  ✓ sees entire window",
        fontsize=10, family="monospace", color=COLORS["add"], weight="bold")

# Shortcut path
draw_arrow(ax, 1.25, y_main - 0.4, 1.25, y2 + 0.0, color=COLORS["shortcut"], lw=1.2)
draw_arrow(ax, 1.25, y2 + 0.0, 10.5, y2 + 0.4, color=COLORS["shortcut"], lw=1.2)
ax.text(5.5, y2 + 0.5, "shortcut (identity if C_in=C_out, else 1×1 Conv)",
        fontsize=7, color=COLORS["shortcut"], style="italic")

# Bottom annotation
ax.text(13.0, 1.4, "SE block expanded →",
        ha="right", fontsize=9, color=COLORS["se"], style="italic")

# Legend
legend_handles = [
    mpatches.Patch(facecolor=COLORS["conv"], label="Conv1d"),
    mpatches.Patch(facecolor=COLORS["bn"],   label="BatchNorm"),
    mpatches.Patch(facecolor=COLORS["relu"], label="ReLU"),
    mpatches.Patch(facecolor=COLORS["drop"], label="Dropout"),
    mpatches.Patch(facecolor=COLORS["se"],   label="SE Operations"),
    mpatches.Patch(facecolor=COLORS["add"],  label="Add / Scale (⊗)"),
    mpatches.Patch(facecolor=COLORS["input"], label="Tensor"),
    mpatches.Patch(facecolor=COLORS["output"], label="Output Tensor"),
    mpatches.Patch(facecolor=COLORS["shortcut"], label="Shortcut"),
]
ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.02),
          fontsize=8, ncol=3, frameon=True, edgecolor="black")

plt.tight_layout()
plt.savefig(OUT_DETAIL, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_DETAIL}")