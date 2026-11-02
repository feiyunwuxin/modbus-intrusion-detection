#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨模型横向对比 - Accuracy / Precision / Recall / F1-Score (Binary convention)

策略:
  - 按测试集分组 (window16 / window8 / window4 / no-window)
  - 每组选代表模型,排除重复变体
  - Binary convention: P/R/F1 = Attack 类
  - 同时记录 Normal_P/R/F1 + Macro-F1 作附录
"""
import os, json
import numpy as np

BASE = r"C:\work\Claude\Issue"
THR = 0.5

# ---- 工具函数 (与 rerun_4metrics_5seed.py 完全一致) ----
def per_class_prf(y_true, y_pred, cls):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == cls) & (y_pred == cls)))
    fp = int(np.sum((y_true != cls) & (y_pred == cls)))
    fn = int(np.sum((y_true == cls) & (y_pred != cls)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, int(np.sum(y_true == cls))

def all_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    p_n, r_n, f1_n, sup_n = per_class_prf(y_true, y_pred, 0)
    p_a, r_a, f1_a, sup_a = per_class_prf(y_true, y_pred, 1)
    return {
        "Accuracy": acc,
        "Precision": p_a,        # Attack P
        "Recall": r_a,           # Attack R
        "F1-Score": f1_a,        # Attack F1
        "Macro_F1": (f1_n + f1_a) / 2,
        "Normal_Precision": p_n,
        "Normal_Recall": r_n,
        "Normal_F1": f1_n,
        "Attack_Precision": p_a,
        "Attack_Recall": r_a,
        "Attack_F1": f1_a,
        "Attack_Support": sup_a,
        "Normal_Support": sup_n,
    }

# ---- 模型清单 (按测试集分组,每组选代表) ----
MODELS = {
    # ===== Group A: window16 测试集 (N=3432, Normal=1627, Attack=1805) =====
    # 注意:历史 ensemble5.csv 是旧 5-seed 实验,与当前 23-dim ch=32 5-seed 不一致,已排除。
    # 当前 SOTA = 23-dim ch=32 B=64 do=0.1 5-seed self-ensemble (seeds 42/123/456/789/1024)
    "window16": [
        # [name, csv_file, params_or_note]
        ("TCN+SE 23-dim ch=32 single (seed 123)", "predictions_test_v4_se_23dim_b64_ch32_do01_window16_s123.csv",
         "单 seed 冠军"),
        ("TCN+SE v1b window16", "predictions_test_tcn_v1b_v4se_window16.csv", "v1 早版"),
        ("TCN+SE v2 window16", "predictions_test_tcn_v2_v4se_window16.csv", "v2 +length"),
        ("TCN+SE v3 window16", "predictions_test_tcn_v3_v4se_window16.csv", "v3 +lcc/lcs"),
        ("TCN v3b window16", "predictions_test_tcn_v3b_window16.csv", "v3b 27-dim"),
        ("TCN v3c deep window16", "predictions_test_tcn_v3c_window16_deep.csv", "v3c 深宽"),
        ("TCN+SE v5 +temporal", "predictions_test_tcn_v5_se_temporal_window16.csv", "v5 时序池化"),
        ("TCN+SE v6 dual-SE", "predictions_test_tcn_v6_dual_se_window16.csv", "v6 双 SE"),
        ("TCN+SE v7 multiscale+focal", "predictions_test_tcn_v7_multiscale_focal_window16.csv", "v7 多尺度"),
        ("1D Transformer", "predictions_test_transformer_v2_window16.csv", "window16"),
        ("LSTM+Attention", "predictions_test_lstm_attention_v2_window16.csv", "window16"),
        ("MobileNetV2 19-dim", "predictions_test_mobilenetv2_19dim_window16.csv", "window16"),
        ("MobileNetV3-S 19-dim", "predictions_test_mobilenetv3_small_19dim_window16.csv", "NAS"),
        ("MobileViT v3 19-dim", "predictions_test_mobilevit_v3_19dim.csv", "ViT"),
        ("OCC ESNN v3 19-dim", "predictions_test_occ_esnn_v3_19dim.csv", "脉冲网络"),
    ],
    # ===== Group B: window8 测试集 (N=6865, Normal=3434, Attack=3431) =====
    "window8": [
        ("TCN v2", "predictions_test_tcn_binary_v2_window8.csv", "v2 窗口=8"),
        ("TCN v3a deep", "predictions_test_tcn_v3a_window8_deep.csv", "v3a 深宽"),
        ("LSTM", "predictions_test_lstm_binary_v2_window8.csv", "v2"),
        ("GRU", "predictions_test_gru_binary_v2_window8.csv", "v2"),
        ("GRU Uni", "predictions_test_gru_uni_binary_v2_window8.csv", "单向"),
        ("CNN-LSTM", "predictions_test_cnn_lstm_binary_v2_window8.csv", "混合"),
        ("MobileNet1D", "predictions_test_mobile_net_binary_v2_window8.csv", "深度可分离"),
        ("MobileNetV3 1D", "predictions_test_mobile_net_v3_binary_v2_window8.csv", "NAS"),
        ("ShuffleNet1D", "predictions_test_shuffle_net_binary_v2_window8.csv", "通道 shuffle"),
        ("GhostNet1D", "predictions_test_ghost_net_binary_v2_window8.csv", "Ghost 模块"),
        ("RNN Basic", "predictions_test_rnn_basic_binary_v2_window8.csv", "基线 RNN"),
    ],
    # ===== Group C: window4 测试集 (N=13731) =====
    "window4": [
        ("CNN v2", "predictions_test_cnn_binary_v2_window.csv", "窗口=4"),
        ("LSTM v2", "predictions_test_lstm_binary_v2_window.csv", "v2"),
    ],
    # ===== Group D: no-window 测试集 (N=54927, Attack=12009 21.9%) =====
    "no_window": [
        ("LightGBM 19-dim", "predictions_test_lgb_binary_v3_19dim.csv", "梯度提升"),
        ("Random Forest 19-dim", "predictions_test_random_forest_binary_v3_19dim.csv", "RF"),
        ("Random Forest v2", "predictions_test_random_forest_binary_v2.csv", "v2 baseline"),
        ("FNN (MLP) v2", "predictions_test_fnn_binary_v2.csv", "全连接 NN"),
        ("Decision Tree v2", "predictions_test_decision_tree_binary_v2.csv", "单树"),
        ("SVM 19-dim", "predictions_test_svm_binary_v3_19dim.csv", "线性 SVM"),
        ("CNN v2 (no-window)", "predictions_test_cnn_binary_v2.csv", "原始 CNN"),
    ],
}

# ---- 计算所有模型 4 指标 ----
results = {}
for group, models in MODELS.items():
    results[group] = []
    for name, csv_file, note in models:
        path = os.path.join(BASE, csv_file)
        if not os.path.exists(path):
            print(f"  MISSING: {csv_file}")
            continue
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y_true = arr[:, 0].astype(int)
        y_pred = arr[:, 2].astype(int) if arr.shape[1] >= 3 else (arr[:, 1] >= THR).astype(int)  # pred_05 if available
        m = all_metrics(y_true, y_pred)
        results[group].append({
            "name": name,
            "file": csv_file,
            "note": note,
            "n": int(len(y_true)),
            "n_normal": m["Normal_Support"],
            "n_attack": m["Attack_Support"],
            "Accuracy": round(m["Accuracy"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1-Score": round(m["F1-Score"], 4),
            "Macro_F1": round(m["Macro_F1"], 4),
            "Normal_Precision": round(m["Normal_Precision"], 4),
            "Normal_Recall": round(m["Normal_Recall"], 4),
            "Normal_F1": round(m["Normal_F1"], 4),
        })

# ---- 在 window16 组顶部插入当前 SOTA 5-seed ensemble (从 rerun_4metrics_5seed.json 读) ----
sota_path = os.path.join(BASE, "rerun_4metrics_5seed.json")
if os.path.exists(sota_path):
    with open(sota_path, encoding="utf-8") as f:
        sota = json.load(f)
    ens = sota["ensemble_prob_mean"]
    sota_row = {
        "name": "★ TCN+SE 23-dim ch=32 5-seed ensemble (current SOTA)",
        "file": "predictions_test_v4_se_23dim_..._s{42,123,456,789,1024}.csv (prob mean)",
        "note": "概率平均集成 5 seed",
        "n": sota["test_n"],
        "n_normal": ens["Normal_Support"],
        "n_attack": ens["Attack_Support"],
        "Accuracy": round(ens["Accuracy"], 4),
        "Precision": round(ens["Precision"], 4),
        "Recall": round(ens["Recall"], 4),
        "F1-Score": round(ens["F1-Score"], 4),
        "Macro_F1": round(ens["Macro_F1"], 4),
        "Normal_Precision": round(ens["Normal_Precision"], 4),
        "Normal_Recall": round(ens["Normal_Recall"], 4),
        "Normal_F1": round(ens["Normal_F1"], 4),
    }
    results["window16"].insert(0, sota_row)

# ---- 打印对比表 ----
def print_group(grp_name, rows):
    if not rows:
        return
    print(f"\n{'='*96}")
    print(f"  测试集组: {grp_name}    N={rows[0]['n']}  (Normal={rows[0]['n_normal']}, Attack={rows[0]['n_attack']})")
    print(f"{'='*96}")
    print(f"{'模型':<38} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9} {'Macro-F1':>9}   {'备注'}")
    print(f"{'-'*38} {'-'*9} {'-'*10} {'-'*8} {'-'*9} {'-'*9}   {'-'*20}")
    # 按 F1 降序
    rows_sorted = sorted(rows, key=lambda r: r["F1-Score"], reverse=True)
    for r in rows_sorted:
        print(f"{r['name']:<38} {r['Accuracy']:>9.4f} {r['Precision']:>10.4f} {r['Recall']:>8.4f} {r['F1-Score']:>9.4f} {r['Macro_F1']:>9.4f}   {r['note']}")
    return rows_sorted

print("\n" + "="*96)
print(" " * 30 + "跨模型 4 指标横向对比 (Binary Convention)")
print(" " * 30 + "Accuracy / Precision / Recall / F1-Score — Attack 类")
print("="*96)

all_sorted = {}
for grp in ["window16", "window8", "window4", "no_window"]:
    sorted_rows = print_group(grp, results[grp])
    if sorted_rows:
        all_sorted[grp] = sorted_rows[0]  # best per group

# ---- 全局冠军 (按 Macro-F1 找最佳 deep learning) ----
print(f"\n{'='*96}")
print("  各测试集 F1 冠军")
print(f"{'='*96}")
for grp, r in all_sorted.items():
    print(f"  {grp:>10}: {r['name']:<40} F1={r['F1-Score']:.4f}  Acc={r['Accuracy']:.4f}  P={r['Precision']:.4f}  R={r['Recall']:.4f}")

# ---- 保存 JSON ----
out_path = os.path.join(BASE, "compare_models_4metrics.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "task": "Cross-model horizontal comparison with Accuracy/Precision/Recall/F1-Score",
        "convention": "Binary / Positive-class (Attack)",
        "threshold": THR,
        "results": {k: v for k, v in results.items()},
        "best_per_group": all_sorted,
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")