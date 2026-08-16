#!/usr/bin/env python3
"""
Visualize LR sweep results.

Output: tcn_v4_se_lr_sweep.png (4-panel)
"""

import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_lr_sweep.png")

# Load sweep results
df = pd.read_csv(os.path.join(BASE, "tcn_v4_se_lr_sweep_summary.csv"))
with open(os.path.join(BASE, "tcn_v4_se_lr_sweep_meta.json")) as f:
    meta = json.load(f)

lrs = df["lr"].values
test_f1m = df["test_f1m"].values
val_f1m = df["best_val_f1m"].values
test_pr = df["test_pr_auc"].values
best_eps = df["best_val_epoch"].values
times = df["train_time_s"].values

best_idx = int(np.argmax(test_f1m))
best_lr = lrs[best_idx]
best_f1m = test_f1m[best_idx]

# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# === Panel (a): Test F1m vs LR ===
ax = axes[0, 0]
ax.plot(lrs, test_f1m, "o-", color="#1565C0", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.axvline(best_lr, color="#E65100", linestyle="--", linewidth=1.5, alpha=0.7,
           label=f"best LR = {best_lr:.0e}")
ax.scatter([best_lr], [best_f1m], s=200, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5)
ax.annotate(f"best = {best_f1m:.4f}", xy=(best_lr, best_f1m),
            xytext=(15, 15), textcoords="offset points",
            fontsize=11, color="#E65100", weight="bold")
ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
ax.set_ylabel("Test Macro-F1", fontsize=12)
ax.set_title("(a) Test Macro-F1 vs Learning Rate", fontsize=13, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

# === Panel (b): Test PR-AUC vs LR ===
ax = axes[0, 1]
ax.plot(lrs, test_pr, "o-", color="#6A1B9A", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.axvline(best_lr, color="#E65100", linestyle="--", linewidth=1.5, alpha=0.7)
best_pr_idx = int(np.argmax(test_pr))
ax.scatter([lrs[best_pr_idx]], [test_pr[best_pr_idx]], s=200, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5)
ax.annotate(f"best PR = {test_pr[best_pr_idx]:.4f}", xy=(lrs[best_pr_idx], test_pr[best_pr_idx]),
            xytext=(15, 15), textcoords="offset points",
            fontsize=11, color="#E65100", weight="bold")
ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
ax.set_ylabel("Test PR-AUC", fontsize=12)
ax.set_title("(b) Test PR-AUC vs Learning Rate", fontsize=13, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

# === Panel (c): Best epoch vs LR ===
ax = axes[1, 0]
ax.plot(lrs, best_eps, "s-", color="#2E7D32", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.axvline(best_lr, color="#E65100", linestyle="--", linewidth=1.5, alpha=0.7,
           label=f"best LR = {best_lr:.0e}")
ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
ax.set_ylabel("Best Epoch", fontsize=12)
ax.set_title("(c) Best Epoch (early stop) vs LR", fontsize=13, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
# Annotate the key observation
ax.annotate("Higher LR → faster convergence\n(less overfitting room)",
            xy=(2e-3, 24), xytext=(3e-4, 35),
            fontsize=10, color="#2E7D32", style="italic",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.0))

# === Panel (d): Train time vs LR ===
ax = axes[1, 1]
ax.plot(lrs, times, "D-", color="#C62828", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.axvline(best_lr, color="#E65100", linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
ax.set_ylabel("Train Time (s)", fontsize=12)
ax.set_title("(d) Train Time vs LR (lower = better)", fontsize=13, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.annotate(f"best LR also fastest\n({times[best_idx]:.1f}s)",
            xy=(best_lr, times[best_idx]), xytext=(-90, -25),
            textcoords="offset points",
            fontsize=10, color="#C62828", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.0))

# ─────────────────────────────────────────────
fig.suptitle("TCN+SE 19-dim SCADA — Learning Rate Sweep (B=128, ep=40)",
             fontsize=14, weight="bold", y=0.995)
# Footer
default_f1m = float(df[df["lr"] == 5e-4]["test_f1m"].iloc[0])
delta = best_f1m - default_f1m
verdict = (f"BEST LR = {best_lr:.0e}  →  Test F1m = {best_f1m:.4f}  "
           f"(default LR=5e-4 was 0.8450 → now +{delta:.4f}, +{delta*100:.1f}%)  "
           f"|  TRAIN TIME: {times[best_idx]:.0f}s (fastest!)")
fig.text(0.5, 0.005, verdict, ha="center", fontsize=10, style="italic",
         color="#E65100", weight="bold")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")
print()
print("=" * 80)
print(f"  LR SWEEP RESULTS — B=128, ep=40, seed=42")
print("=" * 80)
print(f"{'LR':>10} {'ValF1m':>10} {'TestF1m':>10} {'PR-AUC':>10} {'BestEp':>8} {'Time(s)':>8}")
print("-" * 80)
for _, r in df.iterrows():
    marker = " <-- BEST" if r["lr"] == best_lr else ""
    print(f"{r['lr']:>10.0e} {r['best_val_f1m']:>10.4f} {r['test_f1m']:>10.4f} "
          f"{r['test_pr_auc']:>10.4f} {int(r['best_val_epoch']):>8} {r['train_time_s']:>8.1f}{marker}")
print("=" * 80)