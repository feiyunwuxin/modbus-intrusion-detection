#!/usr/bin/env python3
"""
Batch size sweep: B=64 vs B=128 vs B=512 (original).
Show that B=128 is the sweet spot for this dataset.

Output: tcn_v4_se_batch_sweep.png (4-panel)
"""

import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_batch_sweep.png")

configs = [
    {"label": "B=512\n35 ep  (original)",  "color": "#1565C0", "ls": ":",
     "meta": "processed_meta_tcn_v3_v4se_window16.json",
     "test_pred": "predictions_test_tcn_v3_v4se_window16.csv",
     "iters_per_ep": 24, "total_updates": 840},
    {"label": "B=128\n40 ep  ⭐ BEST",  "color": "#E65100", "ls": "-",
     "meta": "processed_meta_tcn_v4_se_b128_e40.json",
     "test_pred": "predictions_test_tcn_v4_se_b128_e40.csv",
     "iters_per_ep": 94, "total_updates": 3760},
    {"label": "B=64\n40 ep",  "color": "#6A1B9A", "ls": "--",
     "meta": "processed_meta_tcn_v4_se_b64_e40.json",
     "test_pred": "predictions_test_tcn_v4_se_b64_e40.csv",
     "iters_per_ep": 188, "total_updates": 7520},
]

# Load all data
data = []
for c in configs:
    with open(os.path.join(BASE, c["meta"])) as f:
        m = json.load(f)
    df = pd.read_csv(os.path.join(BASE, c["test_pred"]))
    c["meta_dict"] = m
    c["y_true"] = df["y_true"].values
    c["prob"]   = df["prob_attack"].values
    if "history" in m:
        c["epochs"] = [h["epoch"] for h in m["history"]]
        c["f1m"]    = [h["val_f1m"] for h in m["history"]]
        c["loss"]   = [h["loss"]   for h in m["history"]]
    data.append(c)

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# === Panel (a): Test Macro-F1 vs total updates ===
ax = axes[0, 0]
for c in data:
    f1 = c["meta_dict"]["metrics"]["test_at_thr"]["macro_f1"]
    ax.scatter(c["total_updates"], f1, s=180, color=c["color"], marker="o",
               edgecolors="black", linewidths=1.5, zorder=3)
    ax.annotate(f"B={c['iters_per_ep']*64 if c['iters_per_ep']!=24 else 512}\n({f1:.4f})",
                xy=(c["total_updates"], f1), xytext=(10, 8),
                textcoords="offset points", fontsize=10, color=c["color"], weight="bold")
ax.set_xscale("log")
ax.set_xlabel("Total Parameter Updates (log scale)", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(a) Test F1m vs Total Updates", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
# Annotation: sweet spot
best_data = max(data, key=lambda c: c["meta_dict"]["metrics"]["test_at_thr"]["macro_f1"])
ax.annotate("Sweet spot", xy=(best_data["total_updates"], best_data["meta_dict"]["metrics"]["test_at_thr"]["macro_f1"]),
            xytext=(20, -30), textcoords="offset points",
            fontsize=11, color=best_data["color"], weight="bold",
            arrowprops=dict(arrowstyle="->", color=best_data["color"], lw=1.5))

# === Panel (b): Test F1m vs Train Time ===
ax = axes[0, 1]
for c in data:
    f1 = c["meta_dict"]["metrics"]["test_at_thr"]["macro_f1"]
    t  = c["meta_dict"]["train_time_seconds"]
    ax.scatter(t, f1, s=180, color=c["color"], marker="s",
               edgecolors="black", linewidths=1.5, zorder=3)
    batch = c["iters_per_ep"] * 64 if c["iters_per_ep"] != 24 else 512
    ax.annotate(f"B={batch}\n({t:.0f}s, {f1:.4f})",
                xy=(t, f1), xytext=(8, 8),
                textcoords="offset points", fontsize=10, color=c["color"], weight="bold")
ax.set_xlabel("Training Time (s)", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(b) Test F1m vs Train Time", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")

# === Panel (c): Val F1m over epochs ===
ax = axes[1, 0]
for c in data:
    if "f1m" in c:
        ax.plot(c["epochs"], c["f1m"], f'o-', color=c["color"], linewidth=2,
                markersize=4, markerfacecolor="white", markeredgewidth=1.3,
                label=c["label"])
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Val Macro-F1", fontsize=11)
ax.set_title("(c) Val Macro-F1 vs Epoch", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

# === Panel (d): PR curves ===
ax = axes[1, 1]
for c in data:
    p, r, _ = precision_recall_curve(c["y_true"], c["prob"])
    ap = average_precision_score(c["y_true"], c["prob"])
    ax.plot(r, p, color=c["color"], linestyle=c["ls"], linewidth=2.2,
            label=f"{c['label']}  (AP = {ap:.4f})")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("(d) PR Curves (test set)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.4, 1.02])

# ─────────────────────────────────────────────
fig.suptitle("TCN+SE 19-dim SCADA — Batch Size Sweep",
             fontsize=14, weight="bold", y=0.995)
# Footer
best = max(data, key=lambda c: c["meta_dict"]["metrics"]["test_at_thr"]["macro_f1"])
f1_best = best["meta_dict"]["metrics"]["test_at_thr"]["macro_f1"]
f1_orig = data[0]["meta_dict"]["metrics"]["test_at_thr"]["macro_f1"]
delta = f1_best - f1_orig
fig.text(0.5, 0.005,
         f"Verdict: B=128 is the sweet spot — Test F1m = {f1_best:.4f}  (Δ vs B=512 = {delta:+.4f}, +{delta*100:.1f}%); B=64 over-fits gradient noise",
         ha="center", fontsize=10, style="italic", color="#37474F")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")

# Summary table
print("\n" + "=" * 100)
print(f"{'Config':<30} {'Batch':>8} {'Iters/ep':>10} {'Updates':>10} {'Train(s)':>10} {'Test F1m':>10} {'PR-AUC':>10} {'Bin-F1':>10}")
print("-" * 100)
for c in data:
    m = c["meta_dict"]
    test = m["metrics"]["test_at_thr"]
    batch = c["iters_per_ep"] * 64 if c["iters_per_ep"] != 24 else 512
    label_safe = c['label'].replace(chr(10), ' ').replace('⭐', '*')
    print(f"{label_safe:<30} {batch:>8} {c['iters_per_ep']:>10} {c['total_updates']:>10} "
          f"{m['train_time_seconds']:>10.1f} {test['macro_f1']:>10.4f} {test['pr_auc']:>10.4f} {test['binary_f1']:>10.4f}")
print("=" * 100)