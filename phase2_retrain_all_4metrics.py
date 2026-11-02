#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二 17 个二分类模型重训 + 4 指标重跑 (2026-09-17)
=====================================================
策略:
  - 顺序调用 16 个训练脚本 (subprocess),每个跑完后从 predictions_test_*.csv 算 4 指标
  - 全部用 Binary convention (pos_label=1, Attack)
  - 失败/超时跳过,记录错误,继续下一个
  - 输出 phase2_retrain_4metrics.json
  - 复现指标:Accuracy / Precision / Recall / F1-Score

⚠️ 与 `phase1_lgb_4metrics.py` (阶段一 baseline) 和 `PHASE2_4METRICS_REPORT.md`
   (加载已有 predictions,09-17 算指标) 的区别:这里**真的重训模型**,不是仅算指标.
"""
import os, sys, time, json, subprocess, traceback
import numpy as np

BASE = r"C:\work\Claude\Issue"
PYTHON = sys.executable
THR = 0.5

# ---- 模型清单 (脚本路径 + 输出预测文件 + 测试集组) ----
# 顺序:window4 → window8 → no_window (按测试集大小归类)
MODELS = [
    # window=4 测试集
    ("CNN v2 window=4",      "cnn_binary_v2_window.py",        "predictions_test_cnn_binary_v2_window.csv",        "window4"),
    ("LSTM v2 window=4",     "lstm_binary_v2_window.py",       "predictions_test_lstm_binary_v2_window.csv",       "window4"),
    # window=8 测试集 (BiLSTM = lstm_binary_v2_window8 bidirectional=True)
    ("BiLSTM v2 window=8",   "lstm_binary_v2_window8.py",      "predictions_test_lstm_binary_v2_window8.csv",      "window8"),
    ("CNN-LSTM v2 window=8", "cnn_lstm_binary_v2_window8.py",  "predictions_test_cnn_lstm_binary_v2_window8.csv",  "window8"),
    ("TCN v2 window=8",      "tcn_binary_v2_window8.py",       "predictions_test_tcn_binary_v2_window8.csv",       "window8"),
    ("BiGRU v2 window=8",    "gru_binary_v2_window8.py",       "predictions_test_gru_binary_v2_window8.csv",       "window8"),
    ("GRU Uni v2 window=8",  "gru_uni_binary_v2_window8.py",   "predictions_test_gru_uni_binary_v2_window8.csv",   "window8"),
    ("MobileNet1D v2 w=8",   "mobile_net_binary_v2_window8.py","predictions_test_mobile_net_binary_v2_window8.csv","window8"),
    ("MobileNetV3-S v2 w=8", "mobile_net_v3_binary_v2_window8.py","predictions_test_mobile_net_v3_binary_v2_window8.csv","window8"),
    ("ShuffleNet1D w=8",     "shuffle_net_binary_v2_window8.py","predictions_test_shuffle_net_binary_v2_window8.csv","window8"),
    ("GhostNet1D w=8",       "ghost_net_binary_v2_window8.py", "predictions_test_ghost_net_binary_v2_window8.csv", "window8"),
    ("RNN Basic w=8",        "rnn_basic_binary_v2_window8.py", "predictions_test_rnn_basic_binary_v2_window8.csv", "window8"),
    # no-window 测试集
    ("CNN v2 (no-window)",   "cnn_binary_v2.py",               "predictions_test_cnn_binary_v2.csv",               "no_window"),
    ("FNN (MLP) v2",         "fnn_binary_v2.py",               "predictions_test_fnn_binary_v2.csv",               "no_window"),
    ("Decision Tree v2",     "decision_tree_binary_v2.py",     "predictions_test_decision_tree_binary_v2.csv",     "no_window"),
    ("Random Forest v2",     "random_forest_binary_v2.py",     "predictions_test_random_forest_binary_v2.csv",     "no_window"),
]

# Per-script timeout (seconds). Larger for slow DL.
TIMEOUT = {
    "window4": 300,
    "window8": 600,
    "no_window": 600,
}

# ---- 4 指标计算 (与 compare_models_4metrics.py 一致口径) ----
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
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    p_n, r_n, f1_n, sup_n = per_class_prf(y_true, y_pred, 0)
    p_a, r_a, f1_a, sup_a = per_class_prf(y_true, y_pred, 1)
    return {
        "Accuracy":  round(acc, 4),
        "Precision": round(p_a, 4),  # Attack
        "Recall":    round(r_a, 4),  # Attack
        "F1-Score":  round(f1_a, 4), # Attack
        "Macro_F1":  round((f1_n + f1_a) / 2, 4),
        "Normal_Precision": round(p_n, 4),
        "Normal_Recall":    round(r_n, 4),
        "Normal_F1":        round(f1_n, 4),
        "Normal_Support":   sup_n,
        "Attack_Support":   sup_a,
        "N":                int(len(y_true)),
    }

def calc_4metrics_from_csv(csv_path, threshold=0.5):
    """从 predictions_test_*.csv 读 (y_true, prob_attack, [pred_05]) 算 4 指标"""
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    y_true = arr[:, 0].astype(int)
    if arr.shape[1] >= 3:
        y_pred = arr[:, 2].astype(int)
    else:
        y_pred = (arr[:, 1] >= threshold).astype(int)
    return all_metrics(y_true, y_pred)

# ---- 主调度 ----
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

results = []
for model_name, script, csv_file, group in MODELS:
    log(f"{'='*70}")
    log(f"START: {model_name}  (script={script})")
    script_path = os.path.join(BASE, script)
    if not os.path.exists(script_path):
        log(f"  ! MISSING script: {script_path}")
        results.append({"model": model_name, "script": script, "group": group, "status": "MISSING"})
        continue

    t_script = time.time()
    log(f"  -> launching {script} (timeout={TIMEOUT[group]}s)")
    proc = subprocess.run(
        [PYTHON, script_path],
        cwd=BASE,
        timeout=TIMEOUT[group],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t_script
    log(f"  -> script exited rc={proc.returncode} in {elapsed:.1f}s")

    # 收集脚本最后几行输出
    script_tail = (proc.stdout or "")[-500:].strip()
    if proc.returncode != 0:
        log(f"  ! script failed (rc={proc.returncode})")
        results.append({
            "model": model_name, "script": script, "group": group,
            "status": "FAILED", "rc": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-500:].strip(),
        })
        continue

    # 检查 predictions CSV 是否生成/存在
    csv_path = os.path.join(BASE, csv_file)
    if not os.path.exists(csv_path):
        log(f"  ! predictions CSV not generated: {csv_file}")
        results.append({
            "model": model_name, "script": script, "group": group,
            "status": "NO_OUTPUT_CSV",
        })
        continue

    # 计算 4 指标
    try:
        m = calc_4metrics_from_csv(csv_path, threshold=THR)
        log(f"  -> F1={m['F1-Score']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}  Acc={m['Accuracy']:.4f}")
        results.append({
            "model": model_name, "script": script, "group": group,
            "status": "OK",
            "elapsed_s": round(elapsed, 1),
            "csv": csv_file,
            "metrics": m,
        })
    except Exception as e:
        log(f"  ! metric calc failed: {e}")
        results.append({
            "model": model_name, "script": script, "group": group,
            "status": "METRIC_FAIL", "error": str(e),
        })

log(f"{'='*70}")
log(f"DONE. {sum(1 for r in results if r['status']=='OK')}/{len(results)} OK")

# 保存 JSON
out_json = {
    "task": "Phase 2 retrain + 4-metrics (Binary convention, Accuracy/Precision/Recall/F1-Score)",
    "date": "2026-09-17",
    "threshold": THR,
    "n_models": len(results),
    "n_ok": sum(1 for r in results if r["status"] == "OK"),
    "results": results,
}
out_path = os.path.join(BASE, "phase2_retrain_4metrics.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2, ensure_ascii=False)
log(f"Saved -> {out_path}")

# 打印汇总表 (按 F1 降序)
log(f"\n{'='*100}")
log(" SUMMARY (4 metrics, Binary convention, F1 descending)")
log(f"{'='*100}")
log(f"{'模型':<28} {'组':<11} {'N':>6} {'Acc%':>7} {'P%':>7} {'R%':>7} {'F1%':>7} {'Macro-F1%':>10}  {'状态':<10} {'耗时':>6}")
log(f"{'-'*28} {'-'*11} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*10}  {'-'*10} {'-'*6}")
ok_rows = [r for r in results if r["status"] == "OK"]
ok_rows.sort(key=lambda r: r["metrics"]["F1-Score"], reverse=True)
for r in ok_rows:
    m = r["metrics"]
    log(f"{r['model']:<28} {r['group']:<11} {m['N']:>6} {m['Accuracy']*100:>6.2f}% {m['Precision']*100:>6.2f}% {m['Recall']*100:>6.2f}% {m['F1-Score']*100:>6.2f}% {m['Macro_F1']*100:>9.2f}%  {r['status']:<10} {r.get('elapsed_s',0):>5.1f}s")