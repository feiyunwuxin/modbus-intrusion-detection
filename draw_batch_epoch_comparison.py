#!/usr/bin/env python3
"""
3-way TCN+SE comparison:
  - Original:        B=512, ep=35, ReduceLROnPlateau  (project champion @ 0.8444)
  - Cosine 50:       B=512, ep=50, CosineAnnealing    (F1m 0.8422, PR-AUC 0.8926)
  - Small batch:     B=128, ep=40, ReduceLROnPlateau  (NEW CHAMPION @ 0.8655 ⭐)

Outputs:
  - tcn_v4_se_3way_comparison.png  (4-panel: Loss, F1m, AUC, Updates)
  - tcn_v4_se_3way_summary.csv
"""

import os, re, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_3way_comparison.png")
OUT_CSV = os.path.join(BASE, "tcn_v4_se_3way_summary.csv")


def parse_log(path):
    epochs, losses, f1ms, aucs, lrs = [], [], [], [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            m = re.match(
                r"\[\s*\d+\.\d+s?\s*\]\s+epoch\s+(\d+)\s+loss=([\d.]+)\s+val Macro-F1=([\d.]+)\s+val AUC=([\d.]+)\s+lr=([\d.e+-]+)",
                line,
            )
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                f1ms.append(float(m.group(3)))
                aucs.append(float(m.group(4)))
                lrs.append(float(m.group(5)))
    if not epochs:
        return None
    return {"epochs": epochs, "loss": losses, "f1m": f1ms, "auc": aucs, "lr": lrs}


def parse_meta(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# Load 3 configurations
configs = []

# Config 1: Original (B=512, ep=35, ReduceLROnPlateau)
hist_orig = parse_log(os.path.join(BASE, "tcn_v4_train_log.txt"))
meta_orig = parse_meta(os.path.join(BASE, "processed_meta_tcn_v3_v4se_window16.json"))
configs.append({
    "name": "Original\nB=512, ep=35\nReduceLROnPlateau",
    "short": "Original",
    "color": "#1565C0",   # blue
    "marker": "o",
    "epochs": hist_orig["epochs"],
    "loss": hist_orig["loss"],
    "f1m": hist_orig["f1m"],
    "auc": hist_orig["auc"],
    "lr": hist_orig["lr"],
    "meta": meta_orig,
})

# Config 2: Cosine 50
meta_cos = parse_meta(os.path.join(BASE, "processed_meta_tcn_v4_se_cosine50.json"))
hist_cos = meta_cos["history"]
configs.append({
    "name": "Cosine 50\nB=512, ep=50\nCosineAnnealing",
    "short": "Cosine 50",
    "color": "#6A1B9A",   # purple
    "marker": "D",
    "epochs": [h["epoch"] for h in hist_cos],
    "loss":   [h["loss"]   for h in hist_cos],
    "f1m":    [h["val_f1m"] for h in hist_cos],
    "auc":    [h["val_auc"] for h in hist_cos],
    "lr":     [h["lr"]     for h in hist_cos],
    "meta": meta_cos,
})

# Config 3: B=128 + ep=40 ⭐ NEW CHAMPION
meta_b128 = parse_meta(os.path.join(BASE, "processed_meta_tcn_v4_se_b128_e40.json"))
hist_b128 = meta_b128["history"]
configs.append({
    "name": "Small Batch ⭐\nB=128, ep=40\nReduceLROnPlateau",
    "short": "B=128 (NEW)",
    "color": "#E65100",   # orange (champion)
    "marker": "s",
    "epochs": [h["epoch"] for h in hist_b128],
    "loss":   [h["loss"]   for h in hist_b128],
    "f1m":    [h["val_f1m"] for h in hist_b128],
    "auc":    [h["val_auc"] for h in hist_b128],
    "lr":     [h["lr"]     for h in hist_b128],
    "meta": meta_b128,
})

# ─────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────
summary_rows = []
for c in configs:
    summary_rows.append({
        "config": c["short"],
        "full_name": c["name"].replace("\n", " "),
        "batch": c["meta"].get("batch", 512),
        "epochs": max(c["epochs"]),
        "scheduler": "ReduceLROnPlateau" if "Reduce" in c["name"] else "CosineAnnealingWarmRestarts",
        "best_val_epoch": c["meta"]["best_epoch"],
        "best_val_f1m": max(c["f1m"]),
        "test_macro_f1": c["meta"]["metrics"]["test_at_thr"]["macro_f1"],
        "test_pr_auc": c["meta"]["metrics"]["test_at_thr"]["pr_auc"],
        "test_binary_f1": c["meta"]["metrics"]["test_at_thr"]["binary_f1"],
        "test_accuracy": c["meta"]["metrics"]["test_at_thr"]["accuracy"],
        "test_roc_auc": c["meta"]["metrics"]["test_at_thr"]["roc_auc"],
        "train_time_s": c["meta"]["train_time_seconds"],
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_CSV, index=False)
print(f"[saved] {OUT_CSV}")
print()
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(summary.to_string(index=False))

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

def plot_metric(ax, key, title, ylabel):
    for c in configs:
        ax.plot(c["epochs"], c[key], f'{c["marker"]}-', color=c["color"],
                linewidth=2.0, markersize=5, markerfacecolor="white",
                markeredgewidth=1.5, label=c["short"])
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, weight="bold", loc="left")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

# Panel (a): Training Loss
plot_metric(axes[0, 0], "loss", "(a) Training Loss", "Train Loss")
# Annotate end values
for c in configs:
    y = c["loss"][-1]
    ax = axes[0, 0]
    ax.annotate(f"{y:.3f}", xy=(c["epochs"][-1], y),
                xytext=(-30, 8), textcoords="offset points",
                fontsize=9, color=c["color"], weight="bold")

# Panel (b): Val Macro-F1
plot_metric(axes[0, 1], "f1m", "(b) Validation Macro-F1", "Val Macro-F1")
# Annotate peaks
for c in configs:
    peak_idx = int(np.argmax(c["f1m"]))
    peak_val = c["f1m"][peak_idx]
    axes[0, 1].annotate(f"{peak_val:.4f}", xy=(c["epochs"][peak_idx], peak_val),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, color=c["color"], weight="bold")

# Panel (c): Val AUC
plot_metric(axes[1, 0], "auc", "(c) Validation AUC", "Val AUC")
# Test PR-AUC lines
for c in configs:
    test_pr = c["meta"]["metrics"]["test_at_thr"]["pr_auc"]
    axes[1, 0].axhline(test_pr, color=c["color"], linestyle=":", linewidth=1.3, alpha=0.6,
                        label=f"Test PR-AUC ({c['short']}) = {test_pr:.4f}")
axes[1, 0].legend(loc="lower right", fontsize=8, framealpha=0.95)

# Panel (d): Test Macro-F1 vs Train Time (scatter)
ax = axes[1, 1]
for c in configs:
    f1 = c["meta"]["metrics"]["test_at_thr"]["macro_f1"]
    t = c["meta"]["train_time_seconds"]
    ax.scatter(t, f1, s=200, color=c["color"], marker=c["marker"],
               edgecolors="black", linewidths=1.5, label=c["short"], zorder=3)
    ax.annotate(f"{f1:.4f}", xy=(t, f1), xytext=(10, 5),
                textcoords="offset points", fontsize=10, color=c["color"], weight="bold")
ax.set_xlabel("Training Time (s)", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(d) Test F1m vs Train Time", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

# ─── Figure title and footer ───
fig.suptitle("TCN+SE 19-dim SCADA — Batch Size & Epoch Ablation Study",
             fontsize=15, weight="bold", y=0.995)

# Bottom verdict
winner = max(configs, key=lambda c: c["meta"]["metrics"]["test_at_thr"]["macro_f1"])
f1_winner = winner["meta"]["metrics"]["test_at_thr"]["macro_f1"]
f1_orig = configs[0]["meta"]["metrics"]["test_at_thr"]["macro_f1"]
delta = f1_winner - f1_orig
verdict = f"🏆 NEW CHAMPION: {winner['short']} — Test Macro-F1 = {f1_winner:.4f} (Δ vs Original = {delta:+.4f}, +{delta*100:.1f}%)"
fig.text(0.5, 0.005, verdict, ha="center", fontsize=11, weight="bold",
         color=winner["color"])

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[saved] {OUT_PNG}")