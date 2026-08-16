#!/usr/bin/env python3
"""
Plot the full 200-epoch training curve for TCN+SE (B=128).
Show the early-stop plateau + LR decay schedule + final test point.

Output: tcn_v4_se_b128_e200_curve.png
"""

import os, json
import matplotlib.pyplot as plt
import numpy as np

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_b128_e200_curve.png")

# Load 200-epoch history
with open(os.path.join(BASE, "processed_meta_tcn_v4_se_b128_e200.json")) as f:
    meta = json.load(f)
hist = meta["history"]
test_m = meta["metrics"]["test_at_thr"]
best_ep = meta["best_epoch"]

eps  = [h["epoch"] for h in hist]
loss = [h["loss"]   for h in hist]
f1m  = [h["val_f1m"] for h in hist]
auc  = [h["val_auc"] for h in hist]
lrs  = [h["lr"]     for h in hist]

# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# === Panel (a): Train Loss ===
ax = axes[0, 0]
ax.plot(eps, loss, "-", color="#1565C0", linewidth=1.8)
ax.fill_between(eps, loss, alpha=0.15, color="#1565C0")
ax.axvline(best_ep, color="#E65100", linestyle="--", linewidth=1.5, alpha=0.7,
           label=f"best epoch = {best_ep}")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Train Loss (BCE w/ pos_weight)", fontsize=11)
ax.set_title("(a) Training Loss (200 epochs)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

# === Panel (b): Val Macro-F1 ===
ax = axes[0, 1]
ax.plot(eps, f1m, "o-", color="#E65100", linewidth=1.8, markersize=3,
        markerfacecolor="white", markeredgewidth=1.0, label="Val Macro-F1")
ax.axhline(test_m["macro_f1"], color="#E65100", linestyle=":", linewidth=1.3, alpha=0.6,
           label=f"Test F1m = {test_m['macro_f1']:.4f}")
ax.axvline(best_ep, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
           label=f"best epoch = {best_ep}")
peak_idx = int(np.argmax(f1m))
ax.annotate(f"peak val = {f1m[peak_idx]:.4f}",
            xy=(eps[peak_idx], f1m[peak_idx]), xytext=(10, 10),
            textcoords="offset points", fontsize=10, color="#E65100", weight="bold")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Val Macro-F1", fontsize=11)
ax.set_title("(b) Validation Macro-F1", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

# === Panel (c): LR Schedule ===
ax = axes[1, 0]
ax.plot(eps, lrs, "o-", color="#6A1B9A", linewidth=1.8, markersize=3,
        markerfacecolor="white", markeredgewidth=1.0)
ax.set_yscale("log")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Learning Rate (log scale)", fontsize=11)
ax.set_title("(c) LR Schedule (ReduceLROnPlateau)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
# Mark decay points
prev_lr = lrs[0]
for i, lr in enumerate(lrs):
    if lr < prev_lr * 0.6:
        ax.axvline(eps[i], color="#C62828", linestyle=":", linewidth=0.8, alpha=0.5)
        prev_lr = lr

# === Panel (d): Val AUC vs F1m scatter (over time) ===
ax = axes[1, 1]
sc = ax.scatter(f1m, auc, c=eps, cmap="viridis", s=20, alpha=0.7, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Epoch")
ax.scatter(f1m[peak_idx], auc[peak_idx], s=200, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5,
           label=f"best (ep {best_ep})")
ax.set_xlabel("Val Macro-F1", fontsize=11)
ax.set_ylabel("Val AUC", fontsize=11)
ax.set_title("(d) Val AUC vs Val F1m (color = epoch)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

# ─────────────────────────────────────────────
fig.suptitle("TCN+SE (B=128) — 200 Epoch Training Curve",
             fontsize=14, weight="bold", y=0.995)
# Footer
total = meta["total_updates"]
verdict = (f"Verdict: 200 epochs gave NO test gain — Test F1m = {test_m['macro_f1']:.4f} "
           f"(essentially identical to 40-epoch: 0.8655). Best epoch = {best_ep}, "
           f"early-stopped at ep 79, total updates = {total}.")
fig.text(0.5, 0.005, verdict, ha="center", fontsize=10, style="italic", color="#37474F")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")
print()
print(f"Total epochs logged: {len(eps)}")
print(f"Best epoch:          {best_ep}")
print(f"Best val F1m:        {max(f1m):.4f}")
print(f"Test F1m @ best:     {test_m['macro_f1']:.4f}")
print(f"Test PR-AUC:         {test_m['pr_auc']:.4f}")
print(f"Total updates:       {total}")
print(f"Train time:          {meta['train_time_seconds']:.1f}s")
print()
print("=" * 72)
print("Comparison:")
print(f"  B=128 ep=40  → Test F1m 0.8655  PR-AUC 0.9122")
print(f"  B=128 ep=200 → Test F1m {test_m['macro_f1']:.4f}  PR-AUC {test_m['pr_auc']:.4f}")
print(f"  Δ Macro-F1  = {test_m['macro_f1'] - 0.8655:+.4f}")
print(f"  Δ PR-AUC    = {test_m['pr_auc'] - 0.9122:+.4f}")
print("=" * 72)