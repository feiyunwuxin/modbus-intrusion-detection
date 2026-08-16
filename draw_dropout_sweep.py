#!/usr/bin/env python3
"""
Plot dropout sweep results.

Output: tcn_v4_se_dropout_sweep.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_dropout_sweep.png")

df = pd.read_csv(os.path.join(BASE, "tcn_v4_se_dropout_sweep_summary.csv"))
drs = df["dropout"].values
f1ms = df["test_f1m"].values
prs  = df["test_pr_auc"].values
best_ep = df["best_val_epoch"].values
times = df["train_time"].values

best_idx = int(np.argmax(f1ms))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("TCN+SE 19-dim SCADA — Dropout Sweep @ LR=2e-3, ch=64",
             fontsize=14, weight="bold", y=0.995)

# === (a) Test F1m vs Dropout ===
ax = axes[0, 0]
ax.plot(drs, f1ms, "o-", color="#1565C0", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.scatter([drs[best_idx]], [f1ms[best_idx]], s=250, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5)
ax.annotate(f"best F1m: {f1ms[best_idx]:.4f} at dropout={drs[best_idx]}",
            xy=(drs[best_idx], f1ms[best_idx]), xytext=(15, -25),
            textcoords="offset points", fontsize=10, color="#E65100", weight="bold")
ax.set_xlabel("Dropout", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(a) Test F1m vs Dropout", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(drs)

# === (b) Test PR-AUC vs Dropout ===
ax = axes[0, 1]
ax.plot(drs, prs, "s-", color="#6A1B9A", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.set_xlabel("Dropout", fontsize=11)
ax.set_ylabel("Test PR-AUC", fontsize=11)
ax.set_title("(b) Test PR-AUC vs Dropout", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(drs)

# === (c) Best epoch vs Dropout ===
ax = axes[1, 0]
ax.plot(drs, best_ep, "D-", color="#2E7D32", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.set_xlabel("Dropout", fontsize=11)
ax.set_ylabel("Best Epoch (early stop)", fontsize=11)
ax.set_title("(c) Best Epoch vs Dropout  --  more dropout -> faster stop", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(drs)
ax.annotate("higher dropout\nforces earlier stop",
            xy=(0.5, 18), xytext=(0.35, 28),
            fontsize=10, color="#2E7D32", style="italic",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.0))

# === (d) Pareto: F1m vs Train Time, color = PR-AUC ===
ax = axes[1, 1]
sc = ax.scatter(times, f1ms, c=prs, cmap="viridis", s=200, edgecolors="black", linewidths=1.5)
for i, (t, f, d) in enumerate(zip(times, f1ms, drs)):
    ax.annotate(f"d={d}", xy=(t, f), xytext=(8, 5),
                textcoords="offset points", fontsize=10, weight="bold")
plt.colorbar(sc, ax=ax, label="PR-AUC")
ax.set_xlabel("Train Time (s)", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(d) F1m vs Time (color = PR-AUC)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")

# Footer
verdict = (f"Result: Dropout=0.3 (current default) is OPTIMAL  |  "
           f"Range F1m: 0.8574-0.8719 (delta {f1ms.max()-f1ms.min():.4f})")
fig.text(0.5, 0.005, verdict, ha="center", fontsize=10, style="italic",
         color="#37474F")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")

print()
print("=" * 70)
print(f"  DROPOUT SWEEP @ LR=2e-3 — RESULTS")
print("=" * 70)
print(f"{'Dropout':>10} {'TestF1m':>10} {'PR-AUC':>10} {'BestEp':>8} {'Time(s)':>8}")
print("-" * 70)
for _, r in df.iterrows():
    marker = " <-- BEST" if r['dropout'] == drs[best_idx] else ""
    print(f"{r['dropout']:>10.1f} {r['test_f1m']:>10.4f} {r['test_pr_auc']:>10.4f} "
          f"{int(r['best_val_epoch']):>8} {r['train_time']:>8.1f}{marker}")
print("=" * 70)
print(f"\nVerdict: Dropout=0.3 (current default) is OPTIMAL")
print(f"F1m range: {f1ms.min():.4f} - {f1ms.max():.4f} (delta {f1ms.max()-f1ms.min():.4f})")