#!/usr/bin/env python3
"""
Channels sweep visualization — channels vs F1m/PR-AUC/params.

Output: tcn_v4_se_ch_sweep.png (4-panel)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_ch_sweep.png")

df = pd.read_csv(os.path.join(BASE, "tcn_v4_se_ch_sweep_lr2e3_summary.csv"))
chs = df["channels"].values
params = df["n_params"].values
f1ms = df["test_f1m"].values
prs = df["test_pr_auc"].values
times = df["train_time"].values

best_f1m_idx = int(np.argmax(f1ms))
best_pr_idx  = int(np.argmax(prs))

# Use single column, 4 rows layout (avoid colorbar rendering issue)
fig, axes = plt.subplots(4, 1, figsize=(10, 16))

# === Panel (a): Test F1m vs Channels ===
ax = axes[0]
ax.plot(chs, f1ms, "o-", color="#1565C0", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.scatter([chs[best_f1m_idx]], [f1ms[best_f1m_idx]], s=250, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5)
ax.annotate(f"best F1m: ch={chs[best_f1m_idx]}, {f1ms[best_f1m_idx]:.4f}",
            xy=(chs[best_f1m_idx], f1ms[best_f1m_idx]), xytext=(20, 5),
            textcoords="offset points", fontsize=10, color="#E65100", weight="bold")
ax.set_xlabel("Channels", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(a) Test F1m vs Channels  --  F1m is flat (0.869-0.873)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(chs)

# === Panel (b): Test PR-AUC vs Channels ===
ax = axes[1]
ax.plot(chs, prs, "s-", color="#6A1B9A", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.scatter([chs[best_pr_idx]], [prs[best_pr_idx]], s=250, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5)
ax.annotate(f"best PR-AUC: ch={chs[best_pr_idx]}, {prs[best_pr_idx]:.4f}",
            xy=(chs[best_pr_idx], prs[best_pr_idx]), xytext=(-100, -10),
            textcoords="offset points", fontsize=10, color="#E65100", weight="bold")
ax.set_xlabel("Channels", fontsize=11)
ax.set_ylabel("Test PR-AUC", fontsize=11)
ax.set_title("(b) Test PR-AUC vs Channels  --  U-shape with dip at ch=64, peak at ch=48", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(chs)

# === Panel (c): Params vs Channels ===
ax = axes[2]
ax.plot(chs, params, "D-", color="#2E7D32", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.set_xlabel("Channels", fontsize=11)
ax.set_ylabel("Total Params", fontsize=11)
ax.set_title("(c) Model Size vs Channels  --  Quadratic growth (~ch^2)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(chs)
for x, y in zip(chs, params):
    ax.annotate(f"{y:,}", xy=(x, y), xytext=(5, 5),
                textcoords="offset points", fontsize=9, color="#2E7D32")

# === Panel (d): Pareto — F1m vs PR-AUC, size = params ===
ax = axes[3]
sizes = params / params.max() * 800 + 100  # scale for visibility
ax.scatter(prs, f1ms, s=sizes, c=chs, cmap="viridis", alpha=0.7, edgecolors="black", linewidths=1.5)
for i, (p, f, c) in enumerate(zip(prs, f1ms, chs)):
    ax.annotate(f"ch={c}", xy=(p, f), xytext=(8, 8),
                textcoords="offset points", fontsize=10, weight="bold")
ax.set_xlabel("Test PR-AUC", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(d) Pareto: F1m vs PR-AUC (size = params, color = channels)", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
# Annotation
ax.annotate("bigger bubble = more params",
            xy=(0.92, 0.866), xytext=(0.92, 0.866), fontsize=9, color="gray", style="italic")

fig.suptitle("TCN+SE 19-dim SCADA — Channels Sweep @ LR=2e-3",
             fontsize=14, weight="bold", y=0.995)

verdict = (f"ch=128 wins F1m ({f1ms[best_f1m_idx]:.4f}, 274K params) | "
           f"ch=48 wins PR-AUC ({prs[best_pr_idx]:.4f}, 42K params) | "
           f"ch=32 Pareto-sweet (F1m {f1ms[0]:.4f}, PR-AUC {prs[0]:.4f}, only 20K params)")
fig.text(0.5, 0.005, verdict, ha="center", fontsize=10, style="italic",
         color="#37474F")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")

print()
print("=" * 80)
print(f"  CHANNELS SWEEP @ LR=2e-3")
print("=" * 80)
print(f"{'ch':>6} {'Params':>10} {'F1m':>8} {'PR-AUC':>8} {'Time(s)':>8}")
print("-" * 80)
for _, r in df.iterrows():
    print(f"{int(r['channels']):>6} {int(r['n_params']):>10,} {r['test_f1m']:>8.4f} "
          f"{r['test_pr_auc']:>8.4f} {r['train_time']:>8.1f}")
print("=" * 80)
print(f"\nF1m range:  {f1ms.min():.4f} - {f1ms.max():.4f} (Δ {f1ms.max()-f1ms.min():.4f})")
print(f"PR-AUC range: {prs.min():.4f} - {prs.max():.4f} (Δ {prs.max()-prs.min():.4f})")