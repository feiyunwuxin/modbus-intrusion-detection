#!/usr/bin/env python3
"""
Plot WD sweep results.

Output: tcn_v4_se_wd_sweep.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_PNG = os.path.join(BASE, "tcn_v4_se_wd_sweep.png")

df = pd.read_csv(os.path.join(BASE, "tcn_v4_se_wd_sweep_summary.csv"))
wds = df["weight_decay"].values
f1ms = df["test_f1m"].values
prs = df["test_pr_auc"].values
best_ep = df["best_val_epoch"].values
times = df["train_time"].values

best_f1m_idx = int(np.argmax(f1ms))
best_pr_idx = int(np.argmax(prs))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("TCN+SE 19-dim SCADA — WD Sweep @ LR=2e-3, Dropout=0.3, ch=64",
             fontsize=14, weight="bold", y=0.995)

# === (a) Test F1m vs WD (log x) ===
ax = axes[0, 0]
ax.plot(wds, f1ms, "o-", color="#1565C0", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.scatter([wds[best_f1m_idx]], [f1ms[best_f1m_idx]], s=250, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5)
ax.annotate(f"best F1m: WD={wds[best_f1m_idx]:.0e}\nF1m = {f1ms[best_f1m_idx]:.4f}",
            xy=(wds[best_f1m_idx], f1ms[best_f1m_idx]), xytext=(15, 5),
            textcoords="offset points", fontsize=10, color="#E65100", weight="bold")
ax.set_xscale("log")
ax.set_xlabel("Weight Decay (log scale)", fontsize=11)
ax.set_ylabel("Test Macro-F1", fontsize=11)
ax.set_title("(a) Test F1m vs WD  --  WD=1e-5 still wins", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(wds)
ax.set_xticklabels([f"{w:.0e}" for w in wds])

# === (b) Test PR-AUC vs WD (log x) ===
ax = axes[0, 1]
ax.plot(wds, prs, "s-", color="#6A1B9A", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.scatter([wds[best_pr_idx]], [prs[best_pr_idx]], s=250, color="#E65100", marker="*",
           edgecolors="black", linewidths=1.5, zorder=5)
ax.annotate(f"PR-AUC JUMP!\nWD=1e-3 -> {prs[best_pr_idx]:.4f}\n(+{prs[best_pr_idx]-prs[0]:.4f})",
            xy=(wds[best_pr_idx], prs[best_pr_idx]), xytext=(-90, -5),
            textcoords="offset points", fontsize=10, color="#E65100", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                      edgecolor="#E65100"))
ax.set_xscale("log")
ax.set_xlabel("Weight Decay (log scale)", fontsize=11)
ax.set_ylabel("Test PR-AUC", fontsize=11)
ax.set_title("(b) Test PR-AUC vs WD  --  WD=1e-3 dominates", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(wds)
ax.set_xticklabels([f"{w:.0e}" for w in wds])

# === (c) Best Epoch vs WD ===
ax = axes[1, 0]
ax.plot(wds, best_ep, "D-", color="#2E7D32", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.set_xscale("log")
ax.set_xlabel("Weight Decay (log scale)", fontsize=11)
ax.set_ylabel("Best Epoch (early stop)", fontsize=11)
ax.set_title("(c) Best Epoch vs WD  --  larger WD needs more epochs", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(wds)
ax.set_xticklabels([f"{w:.0e}" for w in wds])

# === (d) Train Time vs WD ===
ax = axes[1, 1]
ax.plot(wds, times, "v-", color="#C62828", linewidth=2.5, markersize=10,
        markerfacecolor="white", markeredgewidth=2)
ax.set_xscale("log")
ax.set_xlabel("Weight Decay (log scale)", fontsize=11)
ax.set_ylabel("Train Time (s)", fontsize=11)
ax.set_title("(d) Train Time vs WD  --  WD=1e-3 is 2x slower", fontsize=12, weight="bold", loc="left")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_xticks(wds)
ax.set_xticklabels([f"{w:.0e}" for w in wds])

# Footer
verdict = (f"Result: WD=1e-5 best for F1m ({f1ms[best_f1m_idx]:.4f}) | "
           f"WD=1e-3 best for PR-AUC ({prs[best_pr_idx]:.4f}, +{prs[best_pr_idx]-prs[0]:.4f}) | "
           f"F1m range: {f1ms.min():.4f}-{f1ms.max():.4f}, PR-AUC range: {prs.min():.4f}-{prs.max():.4f}")
fig.text(0.5, 0.005, verdict, ha="center", fontsize=10, style="italic",
         color="#37474F")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")

print()
print("=" * 70)
print(f"  WD SWEEP @ LR=2e-3, Dropout=0.3 — RESULTS")
print("=" * 70)
print(f"{'WD':>10} {'TestF1m':>10} {'PR-AUC':>10} {'BestEp':>8} {'Time(s)':>8}")
print("-" * 70)
for _, r in df.iterrows():
    flag = ""
    if r['test_f1m'] == f1ms.max(): flag += " F1m-best"
    if r['test_pr_auc'] == prs.max(): flag += " PR-AUC-best"
    print(f"{r['weight_decay']:>10.0e} {r['test_f1m']:>10.4f} {r['test_pr_auc']:>10.4f} "
          f"{int(r['best_val_epoch']):>8} {r['train_time']:>8.1f}  {flag}")
print("=" * 70)