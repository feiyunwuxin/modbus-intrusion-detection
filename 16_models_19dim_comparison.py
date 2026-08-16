#!/usr/bin/env python3
"""
Generate updated 16-model cross-comparison (14 yesterday + 2 new MobileNet variants).
Reads meta JSON from each model and produces a Markdown report + CSV.
"""

import os, json, glob
import pandas as pd

BASE = r"C:\work\Claude\Issue"
OUT_MD  = os.path.join(BASE, "16_models_19dim_comparison.md")
OUT_CSV = os.path.join(BASE, "16_models_19dim_comparison.csv")

# (file_path, model_name, family)
SOURCES = [
    # Yesterday's 14 models
    ("processed_meta_tcn_v3_v4se_window16.json",         "TCN+SE 19-dim (champion)", "TCN"),
    ("processed_meta_lgb_binary_v3_19dim.json",          "LightGBM",                 "Tree"),
    ("processed_meta_cnn_lstm_19dim_v3_19dim.json",      "CNN-LSTM",                 "Hybrid"),
    ("processed_meta_cnn1d_19dim_v3_19dim.json",         "CNN 1D",                   "Lightweight"),
    ("processed_meta_random_forest_binary_v3_19dim.json","Random Forest",            "Tree"),
    ("processed_meta_shufflenet1d_19dim_v3_19dim.json",   "ShuffleNet 1D",            "Mobile"),
    ("processed_meta_mobilenet1d_19dim_v3_19dim.json",   "MobileNet V1",             "Mobile"),
    ("processed_meta_bilstm_19dim_v3_19dim.json",        "BiLSTM",                   "Recurrent"),
    ("processed_meta_squeezenet1d_19dim_v3_19dim.json",  "SqueezeNet 1D",            "Mobile"),
    ("processed_meta_lstm_19dim_v3_19dim.json",          "LSTM",                     "Recurrent"),
    ("processed_meta_ghostnet1d_19dim_v3_19dim.json",    "GhostNet 1D",              "Mobile"),
    ("processed_meta_rnn_19dim_v3_19dim.json",           "Vanilla RNN",              "Recurrent"),
    ("processed_meta_mobilevit_v3_19dim.json",           "MobileViT 1D",             "Hybrid"),
    ("processed_meta_svm_binary_v3_19dim.json",          "LinearSVM",                "Classical"),
    ("processed_meta_occ_esnn_v3_19dim.json",            "OCC-eSNN",                 "SNN"),
    # Today's 2 new models
    ("processed_meta_mobilenetv2_19dim_window16.json",   "MobileNetV2",              "Mobile"),
    ("processed_meta_mobilenetv3_small_19dim_window16.json", "MobileNetV3-Small",    "Mobile"),
]

rows = []
for fn, name, family in SOURCES:
    path = os.path.join(BASE, fn)
    if not os.path.exists(path):
        print(f"[skip] missing {fn}")
        continue
    with open(path) as f:
        d = json.load(f)
    m = d.get("metrics", {}).get("test_at_thr", {})
    n_params = d.get("n_params", "N/A")
    train_time = d.get("train_time_seconds", None)
    rows.append({
        "model": name,
        "family": family,
        "params": n_params,
        "f1m": m.get("macro_f1", 0),
        "pr_auc": m.get("pr_auc", 0),
        "bin_f1": m.get("binary_f1", 0),
        "acc": m.get("accuracy", 0),
        "roc_auc": m.get("roc_auc", 0),
        "train_time_s": round(train_time, 1) if train_time else None,
        "best_epoch": d.get("best_epoch", "N/A"),
        "best_threshold": d.get("best_threshold", "N/A"),
    })

df = pd.DataFrame(rows).sort_values("f1m", ascending=False).reset_index(drop=True)
df.insert(0, "rank", df.index + 1)

# Save CSV
df.to_csv(OUT_CSV, index=False)
print(f"[saved] {OUT_CSV}")

# Generate Markdown
md = []
md.append("# 16-Model Cross-Comparison on 19-Dim SCADA (Macro-F1 ranked)\n")
md.append("Yesterday's 14 models + today's 2 new MobileNet variants.\n\n")
md.append("| # | Model | Family | Params | F1m | PR-AUC | Bin-F1 | Acc | ROC-AUC | Train(s) |\n")
md.append("|---|-------|--------|--------|-----|--------|--------|-----|---------|----------|\n")
for _, r in df.iterrows():
    params_str = f"{r['params']:,}" if isinstance(r['params'], int) else str(r['params'])
    time_str = f"{r['train_time_s']:.1f}" if r['train_time_s'] else "—"
    md.append(f"| {r['rank']} | **{r['model']}** | {r['family']} | {params_str} | "
              f"{r['f1m']:.4f} | {r['pr_auc']:.4f} | {r['bin_f1']:.4f} | "
              f"{r['acc']:.4f} | {r['roc_auc']:.4f} | {time_str} |\n")

# Family summary
md.append("\n## Family Summary\n\n")
md.append("| Family | Count | Avg F1m | Max F1m | Min F1m |\n")
md.append("|--------|-------|---------|---------|---------|\n")
fam_summary = df.groupby("family").agg(
    n=("model", "count"),
    avg=("f1m", "mean"),
    mx=("f1m", "max"),
    mn=("f1m", "min"),
).sort_values("avg", ascending=False)
for fam, row in fam_summary.iterrows():
    md.append(f"| {fam} | {int(row['n'])} | {row['avg']:.4f} | {row['mx']:.4f} | {row['mn']:.4f} |\n")

# Highlights
md.append("\n## Highlights\n\n")
md.append(f"- **Macro-F1 Champion**: {df.iloc[0]['model']} ({df.iloc[0]['f1m']:.4f})\n")
pr_auc_top = df.sort_values("pr_auc", ascending=False).iloc[0]
md.append(f"- **PR-AUC Champion**: {pr_auc_top['model']} ({pr_auc_top['pr_auc']:.4f})\n")
acc_top = df.sort_values("acc", ascending=False).iloc[0]
md.append(f"- **Accuracy Champion**: {acc_top['model']} ({acc_top['acc']:.4f})\n")
# Param efficiency (F1m / log(Params)) — heuristic
df_eff = df[df['params'].apply(lambda x: isinstance(x, int))].copy()
df_eff["eff"] = df_eff["f1m"] / (df_eff["params"].apply(lambda x: max(x, 1)) ** 0.5) * 1000
eff_top = df_eff.sort_values("eff", ascending=False).iloc[0]
md.append(f"- **Param Efficiency Champion**: {eff_top['model']} (F1m={eff_top['f1m']:.4f}, {int(eff_top['params']):,} params)\n")

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.writelines(md)
print(f"[saved] {OUT_MD}")
print()
print("=" * 80)
print(md[1].strip())
print("=" * 80)
for line in md[3:]:
    print(line, end="")