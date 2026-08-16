#!/usr/bin/env python3
"""
Generate 14-model 19-dim cross-comparison report.
Outputs: 14_models_19dim_comparison.txt + 14_models_19dim_comparison.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\work\Claude\Issue"
OUT_TXT = os.path.join(BASE, "14_models_19dim_comparison.txt")
OUT_PNG = os.path.join(BASE, "14_models_19dim_comparison.png")

# ─────────────────────────────────────────────
# Collect all 14 model results
# ─────────────────────────────────────────────
def load_meta(name):
    # Try both naming conventions
    for path in [
        os.path.join(BASE, f"processed_meta_{name}_v3_19dim.json"),
        os.path.join(BASE, f"processed_meta_{name}.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None

# Manual collection (some models have different meta key names)
results = []

# 1. RF
m = load_meta("random_forest_binary_v3_19dim")
if m:
    t = m['metrics']['test_at_thr']
    results.append({"model": "Random Forest", "params": "500 trees", "params_num": 0,
                    "f1m": t['macro_f1'], "prauc": t['pr_auc'], "bin_f1": t['binary_f1'],
                    "acc": t['accuracy'], "train_time": 11.0, "family": "Tree"})

# 2. LGB
m = load_meta("lgb_binary_v3_19dim")
if m:
    t = m['metrics']['test_at_thr']
    results.append({"model": "LightGBM", "params": "~400 iters", "params_num": 0,
                    "f1m": t['macro_f1'], "prauc": t['pr_auc'], "bin_f1": t['binary_f1'],
                    "acc": t['accuracy'], "train_time": 3.5, "family": "Tree"})

# 3. SVM
m = load_meta("svm_binary_v3_19dim")
if m:
    t = m['metrics']['test_at_thr']
    results.append({"model": "LinearSVM", "params": "linear", "params_num": 0,
                    "f1m": t['macro_f1'], "prauc": t['pr_auc'], "bin_f1": t['binary_f1'],
                    "acc": t['accuracy'], "train_time": 7.4, "family": "Classical"})

# 4-8. Classic DL
classic = json.load(open(os.path.join(BASE, "classic_dl_19dim_comparison.json")))
for r in classic:
    family = "Lightweight" if r['name'] == 'cnn1d_19dim' else ("Recurrent" if r['name'] in ['rnn_19dim','lstm_19dim','bilstm_19dim'] else "Hybrid")
    if r['name'] == 'cnn1d_19dim': display = "CNN 1D"
    elif r['name'] == 'rnn_19dim': display = "Vanilla RNN"
    elif r['name'] == 'lstm_19dim': display = "LSTM"
    elif r['name'] == 'bilstm_19dim': display = "BiLSTM"
    elif r['name'] == 'cnn_lstm_19dim': display = "CNN-LSTM"
    else: display = r['name']
    results.append({"model": display, "params": f"{r['n_params']:,}", "params_num": r['n_params'],
                    "f1m": r['test_f1m'], "prauc": r['test_prauc'], "bin_f1": r['test_bin_f1'],
                    "acc": r['test_acc'], "train_time": r['train_time'], "family": family})

# 9. OCC-eSNN
m = load_meta("occ_esnn_v3_19dim")
if m:
    t = m['metrics']['test_at_thr']
    results.append({"model": "OCC-eSNN", "params": f"{m['n_neurons']} neurons", "params_num": m['n_neurons'],
                    "f1m": t['macro_f1'], "prauc": t['pr_auc'], "bin_f1": t['binary_f1'],
                    "acc": t['accuracy'], "train_time": 0.9, "family": "SNN"})

# 10. MobileViT
m = load_meta("mobilevit_v3_19dim")
if m:
    t = m['metrics']['test_at_thr']
    results.append({"model": "MobileViT 1D", "params": f"{m['n_params']:,}", "params_num": m['n_params'],
                    "f1m": t['macro_f1'], "prauc": t['pr_auc'], "bin_f1": t['binary_f1'],
                    "acc": t['accuracy'], "train_time": m['train_time_seconds'], "family": "Hybrid"})

# 11-14. Lightweight CNN
light = json.load(open(os.path.join(BASE, "lightweight_cnn_19dim_comparison.json")))
for r in light:
    if r['name'] == 'squeezenet1d_19dim': display = "SqueezeNet 1D"
    elif r['name'] == 'mobilenet1d_19dim': display = "MobileNet 1D"
    elif r['name'] == 'shufflenet1d_19dim': display = "ShuffleNet 1D"
    elif r['name'] == 'ghostnet1d_19dim': display = "GhostNet 1D"
    else: display = r['name']
    results.append({"model": display, "params": f"{r['n_params']:,}", "params_num": r['n_params'],
                    "f1m": r['test_f1m'], "prauc": r['test_prauc'], "bin_f1": r['test_bin_f1'],
                    "acc": r['test_acc'], "train_time": r['train_time'], "family": "Mobile"})

# TCN+SE 19-dim v3 (current champion, add for reference)
results.append({"model": "TCN+SE 19-dim (champion)", "params": "72,921", "params_num": 72921,
                "f1m": 0.8444, "prauc": 0.8892, "bin_f1": 0.8347, "acc": 0.8450,
                "train_time": 59.2, "family": "TCN"})

# Sort by Macro-F1 desc
results.sort(key=lambda x: x['f1m'], reverse=True)

# ─────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────
print("\n" + "="*110)
print(" 14-MODEL CROSS-COMPARISON ON 19-DIM SCADA FEATURES (Macro-F1 ranked)")
print("="*110)
print(f"{'#':>3} {'Model':<26} {'Family':<10} {'Params':<14} {'F1m':>7} {'PR-AUC':>7} {'Bin-F1':>7} {'Acc':>7} {'Train(s)':>9}")
print("-"*110)
for i, r in enumerate(results, 1):
    star = " [CHAMPION]" if r['model'] == "TCN+SE 19-dim (champion)" else ""
    print(f"{i:>3} {r['model']:<26} {r['family']:<10} {r['params']:<14} {r['f1m']:>7.4f} "
          f"{r['prauc']:>7.4f} {r['bin_f1']:>7.4f} {r['acc']:>7.4f} {r['train_time']:>9.1f}{star}")
print("="*110)

# Save text
with open(OUT_TXT, "w") as f:
    f.write("="*110 + "\n")
    f.write(" 14-MODEL CROSS-COMPARISON ON 19-DIM SCADA FEATURES (Macro-F1 ranked)\n")
    f.write("="*110 + "\n")
    f.write(f"{'#':>3} {'Model':<26} {'Family':<10} {'Params':<14} {'F1m':>7} {'PR-AUC':>7} {'Bin-F1':>7} {'Acc':>7} {'Train(s)':>9}\n")
    f.write("-"*110 + "\n")
    for i, r in enumerate(results, 1):
        star = " [CHAMPION]" if r['model'] == "TCN+SE 19-dim (champion)" else ""
        f.write(f"{i:>3} {r['model']:<26} {r['family']:<10} {r['params']:<14} {r['f1m']:>7.4f} "
                f"{r['prauc']:>7.4f} {r['bin_f1']:>7.4f} {r['acc']:>7.4f} {r['train_time']:>9.1f}{star}\n")
    f.write("="*110 + "\n")
    f.write("\nFamily-level summary:\n")
    families = {}
    for r in results:
        families.setdefault(r['family'], []).append(r)
    for fam, rs in families.items():
        f1ms = [r['f1m'] for r in rs]
        f.write(f"  {fam:<10}: n={len(rs)}, avg F1m={np.mean(f1ms):.4f}, max F1m={max(f1ms):.4f}, min F1m={min(f1ms):.4f}\n")

print(f"\n[saved] {OUT_TXT}")

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle("14-Model Cross-Comparison on 19-Dim SCADA Features", fontsize=14, weight="bold")

# Plot 1: Macro-F1 vs Params (log scale)
df = pd.DataFrame(results)
colors = {"Tree": "#2E7D32", "Classical": "#558B2F", "Recurrent": "#1565C0",
          "Hybrid": "#C62828", "Lightweight": "#FF6F00", "Mobile": "#6A1B9A",
          "SNN": "#795548", "TCN": "#FFD600"}
for fam in df['family'].unique():
    sub = df[df['family'] == fam]
    ax1.scatter(sub['params_num'].clip(lower=1), sub['f1m'],
                s=120, c=colors.get(fam, "#999"), alpha=0.7,
                edgecolors="black", linewidth=1, label=fam)
    for _, r in sub.iterrows():
        offset_y = 0.003 if r['f1m'] > 0.80 else -0.005
        ax1.annotate(r['model'], (max(r['params_num'], 1), r['f1m']),
                     fontsize=7, alpha=0.85,
                     xytext=(5, 5 if offset_y > 0 else -10), textcoords="offset points")
ax1.set_xscale("symlog", linthresh=100)
ax1.set_xlabel("Params (log scale)", fontsize=11)
ax1.set_ylabel("Test Macro-F1", fontsize=11)
ax1.set_title("Macro-F1 vs Params (color = family)", fontsize=12)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.50, 0.88)

# Plot 2: Macro-F1 vs PR-AUC scatter
for fam in df['family'].unique():
    sub = df[df['family'] == fam]
    ax2.scatter(sub['prauc'], sub['f1m'],
                s=120, c=colors.get(fam, "#999"), alpha=0.7,
                edgecolors="black", linewidth=1, label=fam)
    for _, r in sub.iterrows():
        ax2.annotate(r['model'], (r['prauc'], r['f1m']),
                     fontsize=7, alpha=0.85,
                     xytext=(5, 3), textcoords="offset points")
ax2.set_xlabel("Test PR-AUC", fontsize=11)
ax2.set_ylabel("Test Macro-F1", fontsize=11)
ax2.set_title("Macro-F1 vs PR-AUC (Pareto frontier)", fontsize=12)
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, alpha=0.3)

# Add Pareto front line
sorted_by_prauc = df.sort_values('prauc')
pareto_f1m = []
max_f1m = -1
for _, r in sorted_by_prauc.iterrows():
    if r['f1m'] > max_f1m:
        pareto_f1m.append((r['prauc'], r['f1m']))
        max_f1m = r['f1m']
if pareto_f1m:
    px, py = zip(*pareto_f1m)
    ax2.plot(px, py, "r--", alpha=0.5, linewidth=1, label="Pareto front")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {OUT_PNG}")