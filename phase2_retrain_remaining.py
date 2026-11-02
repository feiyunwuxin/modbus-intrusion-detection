#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二剩余 8 个模型重训 (MobileNetV3-S + 后续 7 个)
- MobileNetV3-S timeout 1500s (重训慢)
- try/except 包装,失败/超时跳过,不影响后续
- 完成后合并到 phase2_retrain_4metrics.json
"""
import os, sys, time, json, subprocess
import numpy as np

BASE = r"C:\work\Claude\Issue"
PYTHON = sys.executable
THR = 0.5

# 剩余模型 (按原调度顺序)
MODELS = [
    ("MobileNetV3-S v2 w=8", "mobile_net_v3_binary_v2_window8.py","predictions_test_mobile_net_v3_binary_v2_window8.csv","window8", 1500),
    ("ShuffleNet1D w=8",     "shuffle_net_binary_v2_window8.py",  "predictions_test_shuffle_net_binary_v2_window8.csv","window8", 900),
    ("GhostNet1D w=8",       "ghost_net_binary_v2_window8.py",   "predictions_test_ghost_net_binary_v2_window8.csv", "window8", 900),
    ("RNN Basic w=8",        "rnn_basic_binary_v2_window8.py",   "predictions_test_rnn_basic_binary_v2_window8.csv", "window8", 600),
    ("CNN v2 (no-window)",   "cnn_binary_v2.py",                 "predictions_test_cnn_binary_v2.csv",               "no_window", 600),
    ("FNN (MLP) v2",         "fnn_binary_v2.py",                 "predictions_test_fnn_binary_v2.csv",               "no_window", 600),
    ("Decision Tree v2",     "decision_tree_binary_v2.py",       "predictions_test_decision_tree_binary_v2.csv",     "no_window", 300),
    ("Random Forest v2",     "random_forest_binary_v2.py",       "predictions_test_random_forest_binary_v2.csv",     "no_window", 300),
]

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
        "Accuracy": round(acc, 4), "Precision": round(p_a, 4), "Recall": round(r_a, 4),
        "F1-Score": round(f1_a, 4), "Macro_F1": round((f1_n + f1_a) / 2, 4),
        "Normal_Precision": round(p_n, 4), "Normal_Recall": round(r_n, 4), "Normal_F1": round(f1_n, 4),
        "Normal_Support": sup_n, "Attack_Support": sup_a, "N": int(len(y_true)),
    }

def calc_4metrics_from_csv(csv_path, threshold=0.5):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    y_true = arr[:, 0].astype(int)
    if arr.shape[1] >= 3:
        y_pred = arr[:, 2].astype(int)
    else:
        y_pred = (arr[:, 1] >= threshold).astype(int)
    return all_metrics(y_true, y_pred)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# 加载已完成的结果 (前 8 个, hardcoded 以免依赖未生成的 JSON)
EARLIER_RESULTS = [
    {"model": "CNN v2 window=4",  "script": "cnn_binary_v2_window.py",         "group": "window4",  "status": "OK", "csv": "predictions_test_cnn_binary_v2_window.csv",     "elapsed_s": 96.4,  "metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_cnn_binary_v2_window.csv"))},
    {"model": "LSTM v2 window=4", "script": "lstm_binary_v2_window.py",        "group": "window4",  "status": "OK", "csv": "predictions_test_lstm_binary_v2_window.csv",    "elapsed_s": 26.9,  "metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_lstm_binary_v2_window.csv"))},
    {"model": "BiLSTM v2 window=8","script":"lstm_binary_v2_window8.py",        "group": "window8",  "status": "OK", "csv": "predictions_test_lstm_binary_v2_window8.csv",   "elapsed_s": 54.0,  "metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_lstm_binary_v2_window8.csv"))},
    {"model": "CNN-LSTM v2 w=8",  "script": "cnn_lstm_binary_v2_window8.py",   "group": "window8",  "status": "OK", "csv": "predictions_test_cnn_lstm_binary_v2_window8.csv","elapsed_s": 192.4, "metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_cnn_lstm_binary_v2_window8.csv"))},
    {"model": "TCN v2 window=8",  "script": "tcn_binary_v2_window8.py",        "group": "window8",  "status": "OK", "csv": "predictions_test_tcn_binary_v2_window8.csv",    "elapsed_s": 204.8, "metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_tcn_binary_v2_window8.csv"))},
    {"model": "BiGRU v2 window=8","script": "gru_binary_v2_window8.py",        "group": "window8",  "status": "OK", "csv": "predictions_test_gru_binary_v2_window8.csv",    "elapsed_s": 67.2,  "metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_gru_binary_v2_window8.csv"))},
    {"model": "GRU Uni v2 w=8",   "script": "gru_uni_binary_v2_window8.py",    "group": "window8",  "status": "OK", "csv": "predictions_test_gru_uni_binary_v2_window8.csv","elapsed_s": 71.6,  "metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_gru_uni_binary_v2_window8.csv"))},
    {"model": "MobileNet1D v2 w=8","script":"mobile_net_binary_v2_window8.py", "group": "window8",  "status": "OK", "csv": "predictions_test_mobile_net_binary_v2_window8.csv","elapsed_s": 222.1,"metrics": calc_4metrics_from_csv(os.path.join(BASE, "predictions_test_mobile_net_binary_v2_window8.csv"))},
]

results = list(EARLIER_RESULTS)

# 重跑剩余
for model_name, script, csv_file, group, timeout in MODELS:
    log(f"{'='*70}")
    log(f"START: {model_name}  (timeout={timeout}s)")
    script_path = os.path.join(BASE, script)
    t_script = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, script_path],
            cwd=BASE,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t_script
        if proc.returncode != 0:
            log(f"  ! script rc={proc.returncode}")
            results.append({"model": model_name, "script": script, "group": group, "status": "FAILED", "rc": proc.returncode,
                            "stderr_tail": (proc.stderr or "")[-500:].strip()})
            continue
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_script
        log(f"  ! TIMEOUT after {timeout}s (elapsed={elapsed:.1f}s)")
        results.append({"model": model_name, "script": script, "group": group, "status": "TIMEOUT", "timeout": timeout, "elapsed_s": round(elapsed, 1)})
        continue
    except Exception as e:
        log(f"  ! EXCEPTION: {e}")
        results.append({"model": model_name, "script": script, "group": group, "status": "EXCEPTION", "error": str(e)})
        continue

    csv_path = os.path.join(BASE, csv_file)
    if not os.path.exists(csv_path):
        log(f"  ! NO OUTPUT CSV: {csv_file}")
        results.append({"model": model_name, "script": script, "group": group, "status": "NO_OUTPUT_CSV"})
        continue

    try:
        m = calc_4metrics_from_csv(csv_path, threshold=THR)
        log(f"  -> F1={m['F1-Score']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}  Acc={m['Accuracy']:.4f}  ({elapsed:.1f}s)")
        results.append({"model": model_name, "script": script, "group": group, "status": "OK",
                        "elapsed_s": round(elapsed, 1), "csv": csv_file, "metrics": m})
    except Exception as e:
        log(f"  ! metric calc failed: {e}")
        results.append({"model": model_name, "script": script, "group": group, "status": "METRIC_FAIL", "error": str(e)})

log(f"{'='*70}")
log(f"DONE. {sum(1 for r in results if r['status']=='OK')}/{len(results)} OK")

# 保存完整 JSON
out_json = {
    "task": "Phase 2 retrain + 4-metrics (Binary convention)",
    "date": "2026-09-17/18",
    "threshold": THR,
    "n_models": len(results),
    "n_ok": sum(1 for r in results if r["status"] == "OK"),
    "results": results,
}
out_path = os.path.join(BASE, "phase2_retrain_4metrics.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2, ensure_ascii=False)
log(f"Saved -> {out_path}")

# 打印汇总
log(f"\n{'='*100}")
log(" SUMMARY (F1 descending)")
log(f"{'='*100}")
log(f"{'模型':<28} {'组':<11} {'N':>6} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'M-F1':>6}  {'状态':<10} {'耗时':>6}")
log(f"{'-'*28} {'-'*11} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*10} {'-'*6}")
ok_rows = sorted([r for r in results if r["status"] == "OK"], key=lambda r: r["metrics"]["F1-Score"], reverse=True)
for r in ok_rows:
    m = r["metrics"]
    log(f"{r['model']:<28} {r['group']:<11} {m['N']:>6} {m['Accuracy']:>6.4f} {m['Precision']:>6.4f} {m['Recall']:>6.4f} {m['F1-Score']:>6.4f} {m['Macro_F1']:>6.4f}  {r['status']:<10} {r.get('elapsed_s',0):>5.1f}s")
# 失败/超时
for r in results:
    if r["status"] != "OK":
        log(f"  [non-OK] {r['model']}: {r['status']}")