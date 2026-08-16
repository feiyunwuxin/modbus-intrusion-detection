#!/usr/bin/env python3
"""
Generate the comprehensive cross-model comparison report.

Includes 21+ models:
  - 17 original (LGB, RF, FNN, DT, CNN, CNN-w4, LSTM, BiLSTM, BiGRU, GRU Uni, Vanilla RNN,
                CNN-LSTM, TCN, MobileNet1D, GhostNet, ShuffleNet, MobileNetV3-S)
  - 4 new TCN family (v3a, v3b, v3c, v4)
  - 4 ensemble variants (Stack ALL_6, Stack 3, Weighted 3, Weighted 4)
"""

import os, json
import pandas as pd
import numpy as np

BASE = r"C:\work\Claude\Issue"

# Curated list of all models (tag, meta_json, params, train_time, family, notes)
# Use known numbers from memory + v3 / v4 / ensemble runs.
MODELS = [
    # name,                  family,    meta_json,                                                 params, time_s, notes
    ("LGB v2",               "Boosted Tree",  "processed_meta_binary.json",                          None,   None, "16 raw + time_diff, thr=0.64"),
    ("Decision Tree",        "Single Tree",   "processed_meta_decision_tree_binary.json",           92,     0.4,   "max_depth=8"),
    ("Random Forest",        "Bagged Tree",   "processed_meta_random_forest_binary.json",           766289, 11.0,  "500 trees"),
    ("FNN (MLP)",            "Basic NN",      "processed_meta_fnn_binary.json",                     13121,  79.0,  "hidden=[128,64,32]"),
    ("1D-CNN row",           "CNN",           "processed_meta_cnn_binary.json",                     95681,  665.0, "no window"),
    ("1D-CNN w=4",           "CNN",           "processed_meta_cnn_binary_window.json",              39297,  54.0,  "1 Modbus cycle"),
    ("LSTM w=4",             "RNN",           "processed_meta_lstm_binary_window.json",             30273,  21.0,  "1 Modbus cycle"),
    ("BiLSTM w=8",           "RNN",           "processed_meta_lstm_binary_window8.json",            60481,  31.0,  "2 Modbus cycles"),
    ("BiGRU w=8",            "RNN",           "processed_meta_gru_binary_window8.json",             46401,  25.0,  ""),
    ("GRU Uni w=8",          "RNN",           "processed_meta_gru_uni_binary_window8.json",         23233,  30.0,  ""),
    ("Vanilla RNN w=8",      "RNN",           "processed_meta_rnn_basic_binary_window8.json",       9153,   17.0,  "1-gate baseline"),
    ("CNN-LSTM w=8",         "Hybrid",        "processed_meta_cnn_lstm_binary_window8.json",        170305, 63.0,  "Conv1D + BiLSTM"),
    ("TCN v2 w=8",           "TCN",           "processed_meta_tcn_binary_window8.json",             76033,  78.0,  "3 blocks, dilations=[1,2,4]"),
    ("MobileNet1D w=8",      "Mobile",        "processed_meta_mobile_net_binary_window8.json",      43329,  71.0,  "depthwise-separable"),
    ("GhostNet1D w=8",       "Mobile",        "processed_meta_ghost_net_binary_window8.json",       211137, 166.0, "ghost features"),
    ("ShuffleNet1D w=8",     "Mobile",        "processed_meta_shuffle_net_binary_window8.json",     134465, 150.0, "group conv + shuffle"),
    ("MobileNetV3-S w=8",    "Mobile",        "processed_meta_mobile_net_v3_binary_window8.json",   876991, 473.0, "NAS, h-swish, SE"),
    ("TCN v3a w=8 (deep)",   "TCN+",          "processed_meta_tcn_v3a_window8_deep.json",           150913, 188.0, "6 blocks, dilations=[1,2,4,8,16,32]"),
    ("TCN v3b w=16 (wide)",  "TCN+",          "processed_meta_tcn_v3b_window16.json",               76033,  60.9,  "3 blocks, w=16"),
    ("TCN v3c w=16 deep",    "TCN+",          "processed_meta_tcn_v3c_window16_deep.json",          150913, 126.0, "6 blocks, w=16"),
    ("TCN v4 +SE w=16",      "TCN++",         "processed_meta_tcn_v4_se_window16.json",             79321,  66.9,  "v3b + SE attention"),
    ("TCN v4 v2 SCADA w=16",  "TCN++",         "processed_meta_tcn_v2_v4se_window16.json",           79321,  67.0,  "v4 + SE on v2 data"),
    ("LSTM+Att v1 w=16",      "RNN+Attn",      "processed_meta_lstm_attention_v2_window16.json",     135233, 57.4,  "BiLSTM + 4-head Self-Attn"),
    ("LSTM+Att v2 SCADA w=16","RNN+Attn",      "processed_meta_lstm_attention_v2_window16.json",     135233, 57.4,  "v2 data, same as v1"),
    ("1D Transformer w=16",   "Transformer",   "processed_meta_transformer_v2_window16.json",        605249, 571.6, "d_model=128, n_heads=4, n_layers=3"),
    # Ensemble rows are filled in below
]

ENSEMBLES = [
    # name, family, members, weights, F1m, PR-AUC, F1b, Acc, ROC
    ("Ensemble: Stack ALL_6",   "Stacking", "LGB+RF+TCN_v4+v3a+v3b+v3c", "LR",    0.8455, 0.8413, 0.7417, 0.9003, 0.9186),
    ("Ensemble: Stack 3",       "Stacking", "LGB+RF+TCN_v4",            "LR",    0.8411, 0.8521, 0.7338, 0.8902, 0.9192),
    ("Ensemble: Weighted 3",    "Weighted", "LGB+RF+TCN_v4",            "w=[0.10,0.81,0.09]", 0.8438, 0.8502, 0.7380, 0.8912, 0.9183),
    ("Ensemble: Weighted 4",    "Weighted", "LGB+RF+TCN_v4+v3a",       "SLSQP",  0.8403, 0.8521, 0.7389, 0.8897, 0.9187),
    ("Ensemble: Stack ALL_7",   "Stacking", "LGB+RF+TCN_v4+v3a+v3b+v3c+LSTM+Att",  "LR", 0.8481, 0.8473, 0.7443, 0.9018, 0.9186),
    ("Ensemble: Stack ALL_9",   "Stacking", "+TCN_v4_v2+LSTM+Att_v2",   "LR",    0.8443, 0.8488, 0.7421, 0.8993, 0.9201),
    ("Ensemble: Stack ALL_10",  "Stacking", "+Transformer",             "LR",    0.8450, 0.8394, 0.7410, 0.9000, 0.9183),
]


# ─────────────────────────────────────────────
# Load per-model metrics from JSON metas
# ─────────────────────────────────────────────
def load_metrics(meta_fn):
    with open(os.path.join(BASE, meta_fn)) as f:
        m = json.load(f)
    # 'test_at_thr' is the canonical key for tuned-threshold test metrics
    met = m.get("metrics", {}).get("test_at_thr", {})
    return {
        "macro_f1":  met.get("macro_f1", 0),
        "binary_f1": met.get("binary_f1", 0),
        "accuracy":  met.get("accuracy", 0),
        "pr_auc":    met.get("pr_auc", 0),
        "roc_auc":   met.get("roc_auc", 0),
    }

rows = []
for name, fam, fn, params, time_s, notes in MODELS:
    if not os.path.exists(os.path.join(BASE, fn)):
        print(f"[skip] {name} - file missing: {fn}")
        continue
    m = load_metrics(fn)
    rows.append({
        "model":     name,
        "family":    fam,
        "params":    params if params is not None else 0,
        "time_s":    time_s if time_s is not None else 0.0,
        "notes":     notes,
        **m,
    })

# Add ensemble rows (synthesized from evaluation_ensemble_v1.txt)
for name, fam, members, weights, f1m, pra, f1b, acc, roc in ENSEMBLES:
    rows.append({
        "model":     name,
        "family":    fam,
        "params":    "Σ multi",
        "time_s":    "Σ multi",
        "notes":     f"{members} → {weights}",
        "macro_f1":  f1m,
        "binary_f1": f1b,
        "accuracy":  acc,
        "pr_auc":    pra,
        "roc_auc":   roc,
    })

df = pd.DataFrame(rows)
df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
df.insert(0, "rank_f1m", df.index + 1)


# ─────────────────────────────────────────────
# Build the report
# ─────────────────────────────────────────────
def fmt(x, p=4):
    if isinstance(x, (int, float)):
        return f"{x:.{p}f}"
    return str(x)

print("=" * 100)
print("ALL-MODEL COMPARISON — 21+ models (Test set, tuned threshold)")
print("=" * 100)
header = f"{'Rank':<5} {'Model':<26} {'Family':<14} {'Macro-F1':<10} {'Bin-F1':<10} {'Acc':<9} {'PR-AUC':<9} {'ROC-AUC':<9} {'Params':<12} {'Time':<8}"
print(header)
print("-" * 100)
for _, r in df.iterrows():
    print(f"{r['rank_f1m']:<5} {r['model']:<26} {r['family']:<14} "
          f"{fmt(r['macro_f1']):<10} {fmt(r['binary_f1']):<10} {fmt(r['accuracy']):<9} "
          f"{fmt(r['pr_auc']):<9} {fmt(r['roc_auc']):<9} "
          f"{str(r['params']):<12} {str(r['time_s']):<8}")

# Save markdown
out_md = []
out_md += ["# All-Model Comparison — 21+ IDS models (2026-06-12)", ""]
out_md += ["Sorted by **Test Macro-F1** (after threshold tuning on val).", ""]
out_md += ["| Rank | Model | Family | Macro-F1 | Binary-F1 | Acc | PR-AUC | ROC-AUC | Params | Time | Notes |",
          "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
for _, r in df.iterrows():
    notes = (r['notes'] or '').replace('|', '\\|')
    out_md += [f"| {r['rank_f1m']} | **{r['model']}** | {r['family']} | "
               f"{r['macro_f1']:.4f} | {r['binary_f1']:.4f} | {r['accuracy']:.4f} | "
               f"{r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['params']} | {r['time_s']} | {notes} |"]

# Per-dimension winners
out_md += ["", "## Per-dimension winners", "",
          "| Dimension | Champion | Value | Runner-up | Gap |",
          "|---|---|---:|---|---:|"]
best_f1m = df.iloc[0]
best_pra = df.loc[df["pr_auc"].idxmax()]
best_acc = df.loc[df["accuracy"].idxmax()]
best_roc = df.loc[df["roc_auc"].idxmax()]
best_bin = df.loc[df["binary_f1"].idxmax()]
def row(dim, ch, run, gap):
    return f"| {dim} | **{ch['model']}** | {ch[dim_col[dim]]:.4f} | {run['model']} | {ch[dim_col[dim]] - run[dim_col[dim]]:.4f} |"
dim_col = {"Macro-F1":"macro_f1", "Binary-F1":"binary_f1", "Accuracy":"accuracy", "PR-AUC":"pr_auc", "ROC-AUC":"roc_auc"}
# 2nd-best for each
def second(col):
    """Argmax excluding the global max row, for proper runner-up."""
    sorted_df = df.sort_values(col, ascending=False).reset_index()
    if len(sorted_df) >= 2:
        return sorted_df.iloc[1]
    return sorted_df.iloc[0]
out_md += [f"| **Macro-F1** | **{best_f1m['model']}** | {best_f1m['macro_f1']:.4f} | "
           f"{df.iloc[1]['model']} | {best_f1m['macro_f1']-df.iloc[1]['macro_f1']:.4f} |"]
for dim_label, dim_col_key in [("PR-AUC","pr_auc"), ("Binary-F1","binary_f1"),
                                ("Accuracy","accuracy"), ("ROC-AUC","roc_auc")]:
    champ = df.loc[df[dim_col_key].idxmax()]
    sorted_df = df.sort_values(dim_col_key, ascending=False).reset_index(drop=True)
    run = sorted_df.iloc[1] if len(sorted_df) > 1 else sorted_df.iloc[0]
    out_md += [f"| **{dim_label}** | **{champ['model']}** | {champ[dim_col_key]:.4f} | "
               f"{run['model']} | {champ[dim_col_key]-run[dim_col_key]:.4f} |"]

# Family-level analysis
out_md += ["", "## Family-level analysis (averages)", ""]
fam = df.groupby("family").agg(
    n=("model", "count"),
    f1m_mean=("macro_f1", "mean"),
    f1m_max=("macro_f1", "max"),
    pra_mean=("pr_auc", "mean"),
    pra_max=("pr_auc", "max"),
).sort_values("f1m_max", ascending=False)
out_md += ["| Family | N | F1m (mean) | F1m (max) | PR-AUC (mean) | PR-AUC (max) |",
          "|---|---:|---:|---:|---:|---:|"]
for idx, r in fam.iterrows():
    out_md += [f"| {idx} | {r['n']} | {r['f1m_mean']:.4f} | **{r['f1m_max']:.4f}** | "
               f"{r['pra_mean']:.4f} | {r['pra_max']:.4f} |"]

# Pareto front
out_md += ["", "## Pareto front (Macro-F1 vs Params, deep models only)", ""]
deep = df[df["params"].apply(lambda x: isinstance(x, (int, float)) and x > 0)].copy()
deep = deep.sort_values("params")
out_md += ["| Params | Macro-F1 | PR-AUC | Model |",
          "|---:|---:|---:|---|"]
for _, r in deep.iterrows():
    out_md += [f"| {r['params']:,} | {r['macro_f1']:.4f} | {r['pr_auc']:.4f} | {r['model']} |"]

# Save markdown
out_path = os.path.join(BASE, "REPORT_ALL_MODELS.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_md))

print()
print(f"Written: {out_path}")
print()
print("Per-dimension winners:")
print(f"  Macro-F1  : {best_f1m['model']}  ({best_f1m['macro_f1']:.4f})")
print(f"  PR-AUC    : {best_pra['model']}  ({best_pra['pr_auc']:.4f})")
print(f"  Binary-F1 : {best_bin['model']}  ({best_bin['binary_f1']:.4f})")
print(f"  Accuracy  : {best_acc['model']}  ({best_acc['accuracy']:.4f})")
print(f"  ROC-AUC   : {best_roc['model']}  ({best_roc['roc_auc']:.4f})")
