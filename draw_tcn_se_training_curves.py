#!/usr/bin/env python3
"""
Plot comprehensive training curves for all 4 TCN+SE configurations:
  - 17 维 v1  (with one-hot, 44 Conv1D dims)
  - 17 维 v1b (no one-hot, 17 Conv1D dims)
  - 19 维 v3  (SCADA row-level, 19 Conv1D dims)
  - 27 维 v2  (SCADA row+window-agg, 27 Conv1D dims)

Parses training logs to extract epoch-by-epoch metrics and produces:
  - 4-panel comparison plot (Loss + Val Macro-F1 + Val AUC + best epoch markers)
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\work\Claude\Issue"

CONFIGS = {
    "17-dim v1  (one-hot, 44 dims)":  ("retrain_run.log",                "#1565C0", "o"),
    "17-dim v1b (no one-hot, 17 dims)":("v1b_v4se_window16_run.log",       "#FF6F00", "s"),
    "19-dim v3  (SCADA row, 19 dims)": ("v3_v4se_window16_run.log",        "#C62828", "^"),
    "27-dim v2  (SCADA + window-agg, 27 dims)": ("v2_v4se_window16_rerun.log", "#2E7D32", "D"),
}

# Final test metrics
FINAL_METRICS = {
    "17-dim v1  (one-hot, 44 dims)":   {"f1m": 0.8340, "prauc": 0.9025, "params": 79321, "thr": 0.48},
    "17-dim v1b (no one-hot, 17 dims)": {"f1m": 0.8002, "prauc": 0.8968, "params": 72409, "thr": 0.51},
    "19-dim v3  (SCADA row, 19 dims)":  {"f1m": 0.8444, "prauc": 0.8892, "params": 72921, "thr": 0.39},
    "27-dim v2  (SCADA + window-agg, 27 dims)": {"f1m": 0.8367, "prauc": 0.8990, "params": 74969, "thr": 0.50},
}


def parse_log(log_path):
    """Parse TCN+SE training log for epoch, loss, val_f1m, val_auc, lr."""
    pattern = re.compile(
        r"epoch\s+(\d+)\s+loss=([\d.]+)\s+val Macro-F1=([\d.]+)\s+val AUC=([\d.]+)\s+lr=([\d.e\-]+)"
    )
    rows = []
    if not os.path.exists(log_path):
        return pd.DataFrame()
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "epoch": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "val_f1m": float(m.group(3)),
                    "val_auc": float(m.group(4)),
                    "lr": float(m.group(5)),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Parse all logs
# ─────────────────────────────────────────────
histories = {}
for label, (log_file, color, marker) in CONFIGS.items():
    df = parse_log(os.path.join(BASE, log_file))
    if df.empty:
        print(f"[WARN] no data from {log_file}")
    histories[label] = df
    print(f"[OK] {label}: {len(df)} epochs (final f1m={df['val_f1m'].iloc[-1] if not df.empty else 'N/A'})")

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("TCN+SE Training Curves — 4 Configurations Compared\n"
             "(same architecture & hyperparameters, different input dimensions)",
             fontsize=14, weight="bold", y=0.995)

# Panel 1: Train Loss
ax = axes[0, 0]
for label, df in histories.items():
    if df.empty: continue
    color = CONFIGS[label][1]
    marker = CONFIGS[label][2]
    ax.plot(df["epoch"], df["loss"], color=color, marker=marker, markevery=3,
            markersize=6, linewidth=1.5, label=label, alpha=0.85)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Train Loss (BCEWithLogits)", fontsize=11)
ax.set_title("Training Loss Convergence", fontsize=12, weight="bold")
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 36)

# Panel 2: Val Macro-F1
ax = axes[0, 1]
for label, df in histories.items():
    if df.empty: continue
    color = CONFIGS[label][1]
    marker = CONFIGS[label][2]
    ax.plot(df["epoch"], df["val_f1m"], color=color, marker=marker, markevery=3,
            markersize=6, linewidth=1.5, label=label, alpha=0.85)
    # mark best epoch
    best_idx = df["val_f1m"].idxmax()
    best_ep = df.loc[best_idx, "epoch"]
    best_f1m = df.loc[best_idx, "val_f1m"]
    ax.scatter([best_ep], [best_f1m], s=180, facecolors='none', edgecolors=color, linewidths=2.5,
               zorder=5)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Validation Macro-F1", fontsize=11)
ax.set_title("Validation Macro-F1 (best epoch marked with circle)", fontsize=12, weight="bold")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 36)
ax.set_ylim(0.50, 0.90)

# Panel 3: Val AUC
ax = axes[1, 0]
for label, df in histories.items():
    if df.empty: continue
    color = CONFIGS[label][1]
    marker = CONFIGS[label][2]
    ax.plot(df["epoch"], df["val_auc"], color=color, marker=marker, markevery=3,
            markersize=6, linewidth=1.5, label=label, alpha=0.85)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Validation ROC-AUC", fontsize=11)
ax.set_title("Validation ROC-AUC", fontsize=12, weight="bold")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 36)
ax.set_ylim(0.60, 0.92)

# Panel 4: Test Macro-F1 bar chart with final results
ax = axes[1, 1]
labels = list(FINAL_METRICS.keys())
f1ms = [FINAL_METRICS[k]["f1m"] for k in labels]
colors = [CONFIGS[k][1] for k in labels]
bars = ax.barh(range(len(labels)), f1ms, color=colors, alpha=0.85, edgecolor="black")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)
ax.set_xlabel("Test Macro-F1", fontsize=11)
ax.set_title("Final Test Macro-F1 (Test @ best_thr)", fontsize=12, weight="bold")
ax.invert_yaxis()
ax.set_xlim(0.75, 0.86)
ax.grid(True, alpha=0.3, axis='x')
for i, (bar, f1m, m) in enumerate(zip(bars, f1ms, [FINAL_METRICS[k] for k in labels])):
    ax.text(f1m + 0.002, bar.get_y() + bar.get_height()/2,
            f"{f1m:.4f}  (PR-AUC={m['prauc']:.4f}, {m['params']:,}p)",
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "tcn_se_4configs_training_curves.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[saved] tcn_se_4configs_training_curves.png")

# ─────────────────────────────────────────────
# Also save a summary table
# ─────────────────────────────────────────────
summary = []
for label, df in histories.items():
    if df.empty: continue
    best_idx = df["val_f1m"].idxmax()
    m = FINAL_METRICS[label]
    summary.append({
        "Config": label,
        "Epochs": len(df),
        "Best val ep": int(df.loc[best_idx, "epoch"]),
        "Best val f1m": float(df.loc[best_idx, "val_f1m"]),
        "Final val f1m": float(df["val_f1m"].iloc[-1]),
        "Final val AUC": float(df["val_auc"].iloc[-1]),
        "Final loss": float(df["loss"].iloc[-1]),
        "Test f1m": m["f1m"],
        "Test PR-AUC": m["prauc"],
        "Params": m["params"],
        "Best thr": m["thr"],
    })
df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(BASE, "tcn_se_4configs_training_summary.csv"), index=False)
print(df_summary.to_string(index=False))
print(f"\n[saved] tcn_se_4configs_training_summary.csv")