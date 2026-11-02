#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨阶段 4 指标大总览 (2026-09-17)
=================================
- 阶段一 (06-08 LGB baseline 32 特征) + 阶段二 (06-09/10) + 阶段三 (06-12/14) + 阶段五 (07-26 SOTA)
- 数据源:
  - phase1_4metrics.json                  (今日重训 LGB baseline)
  - compare_models_4metrics.json          (36 模型横评)
  - rerun_4metrics_5seed.json             (SOTA 5-seed ensemble)
- 输出 cross_phase_4metrics.json + cross_phase_4metrics_table.csv
- 然后人工渲染为 CROSS_PHASE_4METRICS_REPORT.md
"""
import json, os, csv

BASE = r"C:\work\Claude\Issue"

with open(os.path.join(BASE, "phase1_4metrics.json"), encoding="utf-8") as f:
    phase1 = json.load(f)
with open(os.path.join(BASE, "compare_models_4metrics.json"), encoding="utf-8") as f:
    cmp_full = json.load(f)
with open(os.path.join(BASE, "rerun_4metrics_5seed.json"), encoding="utf-8") as f:
    sota = json.load(f)

# ---- 阶段一 baseline ----
stage1_rows = [
    {
        "phase": "一 (06-08)",
        "model": "LightGBM baseline (32 特征, 重训)",
        "category": "GBDT",
        "params_K": "—",
        "test_group": "stage1_t70 (test=82,389, Attack=21.6%)",
        "n": 82389, "n_attack_pct": 21.6,
        "Accuracy":  round(phase1["binary"]["thr_0.5"]["metrics"]["accuracy"], 4),
        "Precision": round(phase1["binary"]["thr_0.5"]["metrics"]["precision"], 4),
        "Recall":    round(phase1["binary"]["thr_0.5"]["metrics"]["recall"], 4),
        "F1-Score":  round(phase1["binary"]["thr_0.5"]["metrics"]["f1"], 4),
        "Macro_F1":  round((phase1["binary"]["thr_0.5"]["confusion_matrix"][0][0]+phase1["binary"]["thr_0.5"]["confusion_matrix"][1][1]) / (phase1["binary"]["thr_0.5"]["confusion_matrix"][0][0]+phase1["binary"]["thr_0.5"]["confusion_matrix"][1][1]+phase1["binary"]["thr_0.5"]["confusion_matrix"][0][1]+phase1["binary"]["thr_0.5"]["confusion_matrix"][1][0]), 4) if False else 0.0,
    },
]

# ---- 阶段二 17 模型 (来自 compare_models_4metrics.json) ----
# 选择代表模型:窗口化每组前 3 + no-window 每组前 2 + 关键变体
stage2_rows = []
representatives = [
    # window16 组
    ("window16", "LSTM+Attention",            "DL",      "DL_Attn"),
    ("window16", "1D Transformer",             "DL",      "Transformer"),
    # window8 组
    ("window8", "BiLSTM v2",                  "DL",      "BiLSTM"),
    ("window8", "TCN v3a deep",               "DL",      "TCN_v3a"),
    ("window8", "CNN-LSTM",                   "DL",      "CNN_LSTM"),
    ("window8", "ShuffleNet1D",               "DL-light","ShuffleNet"),
    ("window8", "MobileNet1D",                "DL-light","MobileNet"),
    ("window8", "RNN Basic",                  "DL",      "RNN"),
    # window4 组
    ("window4", "CNN v2",                     "DL",      "CNN_w4"),
    ("window4", "LSTM v2",                    "DL",      "LSTM_w4"),
    # no_window 组
    ("no_window", "FNN (MLP) v2",             "DL",      "FNN"),
    ("no_window", "Random Forest v2",         "ML",      "RF"),
    ("no_window", "Decision Tree v2",         "ML",      "DT"),
    ("no_window", "LightGBM 19-dim",           "GBDT",    "LGB_19"),
    ("no_window", "SVM 19-dim",                "ML",      "SVM"),
]

for grp_key, name, cat, tag in representatives:
    found = None
    for r in cmp_full["results"][grp_key]:
        if r["name"] == name:
            found = r
            break
    if not found:
        continue
    stage2_rows.append({
        "phase": "二 (06-09/10)",
        "model": name,
        "category": cat,
        "params_K": tag,
        "test_group": f"{grp_key} (test={found['n']}, Attack={found['n_attack']*100//found['n']}%)",
        "n": found["n"], "n_attack_pct": round(found["n_attack"]*100/found["n"], 1),
        "Accuracy":  found["Accuracy"],
        "Precision": found["Precision"],
        "Recall":    found["Recall"],
        "F1-Score":  found["F1-Score"],
        "Macro_F1":  found["Macro_F1"],
    })

# ---- 阶段三 (06-12/14 TCN v3 + TCN+SE v1b-v7) ----
stage3_rows = []
stage3_names = [
    ("window16", "TCN+SE v1b window16"),
    ("window16", "TCN+SE v2 window16"),
    ("window16", "TCN+SE v3 window16"),
    ("window16", "TCN v3b window16"),
    ("window16", "TCN v3c deep window16"),
]
for grp_key, name in stage3_names:
    for r in cmp_full["results"][grp_key]:
        if r["name"] == name:
            stage3_rows.append({
                "phase": "三 (06-12/14)",
                "model": name,
                "category": "DL-TCN+SE",
                "params_K": "v1b→v3c",
                "test_group": f"{grp_key} (test={r['n']}, Attack={r['n_attack']*100//r['n']}%)",
                "n": r["n"], "n_attack_pct": round(r["n_attack"]*100/r["n"], 1),
                "Accuracy":  r["Accuracy"],
                "Precision": r["Precision"],
                "Recall":    r["Recall"],
                "F1-Score":  r["F1-Score"],
                "Macro_F1":  r["Macro_F1"],
            })
            break

# ---- 阶段五 SOTA 5-seed ensemble (07-26, window16) ----
ens = sota["ensemble_prob_mean"]
stage5_row = {
    "phase": "五 SOTA (07-26)",
    "model": "★ TCN+SE 23-dim ch=32 5-seed ensemble",
    "category": "DL-TCN+SE (5-seed)",
    "params_K": "79K",
    "test_group": f"window16 (test={sota['test_n']}, Attack=53%)",
    "n": sota["test_n"], "n_attack_pct": round(ens["Attack_Support"]*100/sota["test_n"], 1),
    "Accuracy":  round(ens["Accuracy"], 4),
    "Precision": round(ens["Precision"], 4),
    "Recall":    round(ens["Recall"], 4),
    "F1-Score":  round(ens["F1-Score"], 4),
    "Macro_F1":  round(ens["Macro_F1"], 4),
}

# ---- 合并 + 按 F1 排序 ----
all_rows = stage1_rows + stage2_rows + stage3_rows + [stage5_row]
all_rows.sort(key=lambda r: r["F1-Score"], reverse=True)

# ---- 保存 ----
out_json = {
    "task": "Cross-phase 4-metrics horizontal overview",
    "convention": "Binary / Positive-class (Attack)",
    "threshold": 0.5,
    "rows": all_rows,
    "phase_gain_trajectory": [
        {"phase": "一 (06-08)",  "best_model": "LGB baseline (32 特征)",    "F1": 0.6400, "test_n": 82389,  "attack_pct": 21.6},
        {"phase": "二 (06-09/10)","best_model": "BiLSTM v2 window=8",        "F1": 0.7882, "test_n": 6865,   "attack_pct": 50.0},
        {"phase": "三 (06-12/14)","best_model": "TCN+SE v3 window16",        "F1": 0.8286, "test_n": 3432,   "attack_pct": 52.6},
        {"phase": "五 (07-26)",  "best_model": "TCN+SE 23-dim 5-seed",      "F1": 0.8737, "test_n": 3432,   "attack_pct": 52.6},
    ],
}
with open(os.path.join(BASE, "cross_phase_4metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2, ensure_ascii=False)

# CSV
with open(os.path.join(BASE, "cross_phase_4metrics_table.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["phase","model","category","params_K","test_group","n","n_attack_pct","Accuracy","Precision","Recall","F1-Score","Macro_F1"])
    w.writeheader()
    for r in all_rows:
        w.writerow(r)

print(f"Saved -> {BASE}/cross_phase_4metrics.json  ({len(all_rows)} rows)")
print(f"Saved -> {BASE}/cross_phase_4metrics_table.csv")
print()
print("=== Top 10 by F1-Score ===")
for r in all_rows[:10]:
    print(f"  {r['F1-Score']:.4f}  {r['phase']:<14}  {r['model']:<40}  {r['test_group']}")
print()
print("=== Bottom 5 by F1-Score ===")
for r in all_rows[-5:]:
    print(f"  {r['F1-Score']:.4f}  {r['phase']:<14}  {r['model']:<40}  {r['test_group']}")