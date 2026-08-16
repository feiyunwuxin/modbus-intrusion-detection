#!/usr/bin/env python3
"""
Compare training curves: TCN+SE 35 epochs + ReduceLROnPlateau vs
                        TCN+SE 50 epochs + CosineAnnealingWarmRestarts

Outputs:
  - tcn_v4_se_epoch_comparison.png (3-panel: Loss, Val F1m, Val AUC)
  - tcn_v4_se_epoch_summary.csv    (numerical comparison)
"""

import os, re, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_PNG  = os.path.join(BASE, "tcn_v4_se_epoch_comparison.png")
OUT_CSV  = os.path.join(BASE, "tcn_v4_se_epoch_summary.csv")


def parse_log(path):
    """Parse training log file into (epochs, losses, f1ms, aucs, lrs)."""
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


# Load both runs
old = parse_log(os.path.join(BASE, "tcn_v4_train_log.txt"))
print(f"[old] 35 epochs: parsed {len(old['epochs']) if old else 0} epochs")

# New run: read from meta JSON (history embedded)
with open(os.path.join(BASE, "processed_meta_tcn_v4_se_cosine50.json")) as f:
    new_meta = json.load(f)
hist = new_meta["history"]
new = {
    "epochs": [h["epoch"] for h in hist],
    "loss":   [h["loss"]   for h in hist],
    "f1m":    [h["val_f1m"] for h in hist],
    "auc":    [h["val_auc"] for h in hist],
    "lr":     [h["lr"]     for h in hist],
}
print(f"[new] 50 epochs: parsed {len(new['epochs'])} epochs")

# Load meta for both
with open(os.path.join(BASE, "processed_meta_tcn_v3_v4se_window16.json")) as f:
    old_meta = json.load(f)

# ─────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────
summary = pd.DataFrame([
    {
        "config": "Original (35ep + ReduceLROnPlateau)",
        "epochs": 35,
        "scheduler": "ReduceLROnPlateau",
        "best_val_epoch": old_meta["best_epoch"],
        "best_val_f1m": max(old["f1m"]),
        "test_macro_f1": old_meta["metrics"]["test_at_thr"]["macro_f1"],
        "test_pr_auc": old_meta["metrics"]["test_at_thr"]["pr_auc"],
        "test_binary_f1": old_meta["metrics"]["test_at_thr"]["binary_f1"],
        "train_time_s": old_meta["train_time_seconds"],
    },
    {
        "config": "New (50ep + CosineAnnealingWarmRestarts)",
        "epochs": 50,
        "scheduler": "CosineAnnealingWarmRestarts",
        "best_val_epoch": new_meta["best_epoch"],
        "best_val_f1m": max(new["f1m"]),
        "test_macro_f1": new_meta["metrics"]["test_at_thr"]["macro_f1"],
        "test_pr_auc": new_meta["metrics"]["test_at_thr"]["pr_auc"],
        "test_binary_f1": new_meta["metrics"]["test_at_thr"]["binary_f1"],
        "train_time_s": new_meta["train_time_seconds"],
    },
])
summary.to_csv(OUT_CSV, index=False)
print(f"[saved] {OUT_CSV}")
print()
print(summary.to_string(index=False))

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Color scheme
c_old = "#1565C0"   # blue
c_new = "#E65100"   # orange

# === Panel 1: Train Loss ===
ax = axes[0]
ax.plot(old["epochs"], old["loss"], "o-", color=c_old, linewidth=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, label="Original (35ep + ReduceLROnPlateau)")
ax.plot(new["epochs"], new["loss"], "s-", color=c_new, linewidth=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, label="New (50ep + CosineAnnealing)")
ax.axvline(old_meta["best_epoch"], color=c_old, linestyle="--", linewidth=1.2, alpha=0.5)
ax.axvline(new_meta["best_epoch"], color=c_new, linestyle="--", linewidth=1.2, alpha=0.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Train Loss", fontsize=12)
ax.set_title("(a) Training Loss", fontsize=13, weight="bold", loc="left")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=":")
# Annotate end
ax.annotate(f"{new['loss'][-1]:.4f}", xy=(new["epochs"][-1], new["loss"][-1]),
            xytext=(-30, 8), textcoords="offset points", fontsize=9, color=c_new, weight="bold")
ax.annotate(f"{old['loss'][-1]:.4f}", xy=(old["epochs"][-1], old["loss"][-1]),
            xytext=(-45, 8), textcoords="offset points", fontsize=9, color=c_old, weight="bold")

# === Panel 2: Val Macro-F1 ===
ax = axes[1]
ax.plot(old["epochs"], old["f1m"], "o-", color=c_old, linewidth=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, label="Original")
ax.plot(new["epochs"], new["f1m"], "s-", color=c_new, linewidth=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, label="New")
ax.axvline(old_meta["best_epoch"], color=c_old, linestyle="--", linewidth=1.2, alpha=0.5,
           label=f"best ep (orig)={old_meta['best_epoch']}")
ax.axvline(new_meta["best_epoch"], color=c_new, linestyle="--", linewidth=1.2, alpha=0.5,
           label=f"best ep (new)={new_meta['best_epoch']}")
# Test F1m lines
ax.axhline(old_meta["metrics"]["test_at_thr"]["macro_f1"], color=c_old,
           linestyle=":", linewidth=1.3, alpha=0.7,
           label=f"Test F1m (orig) = {old_meta['metrics']['test_at_thr']['macro_f1']:.4f}")
ax.axhline(new_meta["metrics"]["test_at_thr"]["macro_f1"], color=c_new,
           linestyle=":", linewidth=1.3, alpha=0.7,
           label=f"Test F1m (new) = {new_meta['metrics']['test_at_thr']['macro_f1']:.4f}")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val Macro-F1", fontsize=12)
ax.set_title("(b) Validation Macro-F1", fontsize=13, weight="bold", loc="left")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=":")
# Annotate peak
peak_old = max(old["f1m"])
peak_new = max(new["f1m"])
ax.annotate(f"peak val = {peak_old:.4f}", xy=(old_meta["best_epoch"], peak_old),
            xytext=(-65, 12), textcoords="offset points", fontsize=9, color=c_old, weight="bold",
            arrowprops=dict(arrowstyle="->", color=c_old, lw=0.8))
ax.annotate(f"peak val = {peak_new:.4f}", xy=(new_meta["best_epoch"], peak_new),
            xytext=(-65, -18), textcoords="offset points", fontsize=9, color=c_new, weight="bold",
            arrowprops=dict(arrowstyle="->", color=c_new, lw=0.8))

# === Panel 3: Learning Rate Schedule ===
ax = axes[2]
ax.plot(old["epochs"], old["lr"], "o-", color=c_old, linewidth=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, label="Original (step decay)")
ax.plot(new["epochs"], new["lr"], "s-", color=c_new, linewidth=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, label="New (cosine + warm restarts)")
ax.set_yscale("log")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Learning Rate (log scale)", fontsize=12)
ax.set_title("(c) LR Schedule", fontsize=13, weight="bold", loc="left")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=":")

# === Figure title ===
fig.suptitle("TCN+SE 19-dim SCADA — Epoch / LR Schedule Ablation",
             fontsize=15, weight="bold", y=1.02)

# Footer with verdict
delta_test = new_meta["metrics"]["test_at_thr"]["macro_f1"] - old_meta["metrics"]["test_at_thr"]["macro_f1"]
delta_val  = max(new["f1m"]) - max(old["f1m"])
delta_time = new_meta["train_time_seconds"] - old_meta["train_time_seconds"]
verdict = ("✓ +50% epochs helps val" if delta_val > 0 else "✗ -epochs hurts val") + \
          f" (Δval={delta_val:+.4f}),  " + \
          ("✓ Test improved" if delta_test > 0 else "✗ Test slightly worse") + \
          f" (Δtest={delta_test:+.4f}),  " + \
          f"Time +{delta_time:.0f}s"
fig.text(0.5, -0.02, verdict, ha="center", fontsize=10, style="italic", color="#37474F")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")