#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重跑 23-dim ch=32 5-seed 模型并输出 Accuracy / Precision / Recall / F1
依赖：仅 numpy（已在系统） + 已有的 5 个 predictions_test_*.csv
"""
import os, json
import numpy as np

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]
THR = 0.5

# ---------- 工具：纯 numpy 计算 4 指标 ----------
def confusion(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]]"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return [[tn, fp], [fn, tp]]

def per_class_prf(y_true, y_pred, cls):
    """Precision/Recall/F1 for a specific class (0=normal, 1=attack)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == cls) & (y_pred == cls)))
    fp = int(np.sum((y_true != cls) & (y_pred == cls)))
    fn = int(np.sum((y_true == cls) & (y_pred != cls)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    support = int(np.sum(y_true == cls))
    return p, r, f1, support

def all_metrics(y_true, y_pred):
    """返回 4 指标的所有视图。

    论文惯例 (Binary / IDS convention):
        Precision  = Attack Precision  (pos_label=1)
        Recall     = Attack Recall
        F1-Score   = Binary F1 on attack
        Accuracy   = 全局 accuracy

    另保留 Macro / Weighted / Normal / Attack 全套指标供比较。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))

    p_n, r_n, f1_n, sup_n = per_class_prf(y_true, y_pred, 0)
    p_a, r_a, f1_a, sup_a = per_class_prf(y_true, y_pred, 1)

    # 论文标准 (Binary / positive-class convention)
    binary_p   = p_a
    binary_r   = r_a
    binary_f1  = f1_a

    # Macro (与 Macro-F1 同源)
    macro_p = (p_n + p_a) / 2
    macro_r = (r_n + r_a) / 2
    macro_f1 = (f1_n + f1_a) / 2

    # Weighted
    weighted_p = (p_n * sup_n + p_a * sup_a) / n
    weighted_r = (r_n * sup_n + r_a * sup_a) / n
    weighted_f1 = (f1_n * sup_n + f1_a * sup_a) / n

    return {
        # === 论文标准 4 指标 (Binary / Positive-class convention) ===
        "Accuracy":         acc,
        "Precision":        binary_p,    # = Attack Precision
        "Recall":           binary_r,    # = Attack Recall
        "F1-Score":         binary_f1,   # = Binary F1 on attack
        # === Macro 平均 (与 Macro-F1 对齐) ===
        "Macro_Precision":  macro_p,
        "Macro_Recall":     macro_r,
        "Macro_F1":         macro_f1,
        # === Weighted 平均 ===
        "Weighted_Precision": weighted_p,
        "Weighted_Recall":    weighted_r,
        "Weighted_F1":        weighted_f1,
        # === Per-class ===
        "Normal_Precision": p_n,
        "Normal_Recall":    r_n,
        "Normal_F1":        f1_n,
        "Attack_Precision": p_a,
        "Attack_Recall":    r_a,
        "Attack_F1":        f1_a,
        # === 支持数 ===
        "Normal_Support":   sup_n,
        "Attack_Support":   sup_a,
        # === 混淆矩阵 ===
        "CM_2x2": confusion(y_true, y_pred),
    }


# ---------- 加载 5 个 seed 的概率 ----------
probs_dict = {}
y_true_ref = None
for s in SEEDS:
    path = os.path.join(BASE, f"predictions_test_v4_se_23dim_b64_ch32_do01_window16_s{s}.csv")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)  # columns: y_true, prob_attack
    y_true = arr[:, 0].astype(int)
    probs = arr[:, 1].astype(np.float32)
    if y_true_ref is None:
        y_true_ref = y_true
    else:
        assert np.array_equal(y_true_ref, y_true), f"seed {s} 与 seed 42 真实标签不一致"
    probs_dict[s] = probs

n = len(y_true_ref)
y_true_arr = y_true_ref
print(f"Test set: N = {n}  Normal = {int(np.sum(y_true_arr==0))}  Attack = {int(np.sum(y_true_arr==1))}\n")

# ---------- 单 seed 4 指标 ----------
per_seed = {}
for s in SEEDS:
    y_pred = (probs_dict[s] >= THR).astype(int)
    per_seed[s] = all_metrics(y_true_arr, y_pred)
    m = per_seed[s]
    # 论文标准 4 指标输出
    print(f"Seed {s:>4}: Acc={m['Accuracy']:.4f}  Precision={m['Precision']:.4f}  "
          f"Recall={m['Recall']:.4f}  F1={m['F1-Score']:.4f}  | "
          f"(macro: P={m['Macro_Precision']:.4f} R={m['Macro_Recall']:.4f} F1={m['Macro_F1']:.4f})")

# ---------- 5-seed probability-average ensemble ----------
probs_matrix = np.stack([probs_dict[s] for s in SEEDS], axis=0)  # (5, N)
ens_probs = probs_matrix.mean(axis=0)
ens_pred = (ens_probs >= THR).astype(int)
ens_metrics = all_metrics(y_true_arr, ens_pred)
m = ens_metrics
print(f"\n5-seed ensemble: Acc={m['Accuracy']:.4f}  Precision={m['Precision']:.4f}  "
      f"Recall={m['Recall']:.4f}  F1={m['F1-Score']:.4f}  | "
      f"(macro: P={m['Macro_Precision']:.4f} R={m['Macro_Recall']:.4f} F1={m['Macro_F1']:.4f})")

# ---------- 5-seed 多数投票 ensemble（额外） ----------
votes = (probs_matrix >= THR).astype(int)
maj_pred = (votes.sum(axis=0) >= 3).astype(int)
maj_metrics = all_metrics(y_true_arr, maj_pred)
m = maj_metrics
print(f"5-seed majority vote: Acc={m['Accuracy']:.4f}  Precision={m['Precision']:.4f}  "
      f"Recall={m['Recall']:.4f}  F1={m['F1-Score']:.4f}")

# ---------- 保存 JSON ----------
out = {
    "model": "TCN+SE 23-dim ch=32 5-seed self-ensemble",
    "task": "Re-run with Accuracy / Precision / Recall / F1-Score",
    "test_n": int(n),
    "threshold": THR,
    "per_seed": {str(s): per_seed[s] for s in SEEDS},
    "ensemble_prob_mean": ens_metrics,
    "ensemble_majority_vote": maj_metrics,
}
out_path = os.path.join(BASE, "rerun_4metrics_5seed.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")
