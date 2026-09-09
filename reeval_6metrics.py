#!/usr/bin/env python3
"""6-指标综合评测: Accuracy, Precision, Recall, F1-Score, Detection Time, Model Size

对 5 个 seed 单模 + 5-seed 集成 + 量化版做完整评测。
"""
import os, sys, time, json
import numpy as np
import torch
import onnxruntime as ort
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

BASE = r"D:\workspace\claude\Issue"
sys.path.insert(0, BASE)

# ---------- 数据 ----------
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
y_test = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
X_test_t = np.clip(X_test, -10.0, 10.0).transpose(0, 2, 1)  # (N, F=23, T=16)

print(f"Test set: N={len(y_test)}, Normal={np.sum(y_test==0)}, Attack={np.sum(y_test==1)}")
print(f"X shape: {X_test_t.shape}\n")

SEEDS = [42, 123, 456, 789, 1024]
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE

# ---------- 工具函数：6 指标 ----------
def six(y_true, probs, y_pred=None):
    if y_pred is None:
        y_pred = (probs >= 0.5).astype(int)
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

def fmt(d):
    return "  ".join(f"{k}={v:.4f}" for k, v in d.items())

# ---------- 推理时间测试 ----------
def time_inference_onnx(onnx_path, X, n_warmup=10, n_runs=200):
    """ONNX Runtime 推理时间（CPU）"""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Warm-up
    for _ in range(n_warmup):
        sess.run(None, {"input": X[:1]})
    # 单样本推理
    times_single = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": X[i:i+1]})
        times_single.append((time.perf_counter() - t0) * 1000)  # ms
    # 批量推理（batch=32）
    times_batch = []
    for i in range(0, len(X) - 32, 32):
        t0 = time.perf_counter()
        sess.run(None, {"input": X[i:i+32]})
        times_batch.append((time.perf_counter() - t0) * 1000)
    return {
        "single_ms_mean": float(np.mean(times_single)),
        "single_ms_median": float(np.median(times_single)),
        "single_ms_p95": float(np.percentile(times_single, 95)),
        "batch32_ms_mean": float(np.mean(times_batch)),
        "batch32_ms_per_sample": float(np.mean(times_batch) / 32),
        "throughput_samples_per_s": float(1000.0 / np.mean(times_single)),
    }

def time_inference_torch(model, X, n_warmup=10, n_runs=200):
    """PyTorch 推理时间（CPU）"""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(torch.from_numpy(X[:1]))
        times_single = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            model(torch.from_numpy(X[i:i+1]))
            times_single.append((time.perf_counter() - t0) * 1000)
    return {
        "single_ms_mean": float(np.mean(times_single)),
        "single_ms_median": float(np.median(times_single)),
        "single_ms_p95": float(np.percentile(times_single, 95)),
    }

# ---------- 模型文件大小 ----------
def file_size(path):
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0

def pt_size_inference(pt_path):
    """PyTorch .pt 实际权重 tensor 大小（不含 pickle 开销）"""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    total_bytes = sum(t.numel() * t.element_size() for t in ckpt["state_dict"].values())
    n_params = sum(t.numel() for t in ckpt["state_dict"].values())
    return total_bytes, n_params

def hybrid_h_data_size(h_path):
    """解析 .h 文件中实际权重大小（INT8 weights + FP32 scales + biases）"""
    if not os.path.exists(h_path):
        return 0, 0
    text = open(h_path, "r", encoding="utf-8", errors="ignore").read()
    import re
    int8_total = 0
    fp32_total = 0
    # 匹配所有 static const int8_t xxx[N] = {...};
    for m in re.finditer(r"static const int8_t \w+\[(\d+)\]", text):
        int8_total += int(m.group(1))
    for m in re.finditer(r"static const float \w+\[(\d+)\]", text):
        fp32_total += int(m.group(1))
    n_params = int8_total + fp32_total
    bytes_total = int8_total * 1 + fp32_total * 4  # INT8 + FP32
    return bytes_total, n_params

# ============================================================
# 主流程
# ============================================================

results = {"per_seed": {}, "ensemble": {}, "model_files": {}}

# ---------- 1. Per-seed 评测 ----------
print("="*72)
print("  Per-seed 5 模型评测 (PyTorch + ONNX)")
print("="*72)

all_probs = []
all_sizes = []
for seed in SEEDS:
    pt_path = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt")
    onnx_path = os.path.join(BASE, f"model_v4_se_23dim_ch32_s{seed}.onnx")

    # PyTorch 加载 + 推理时间
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = TCNClassifierSE(in_ch=23)
    model.load_state_dict(ckpt["state_dict"])

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_t)).numpy().squeeze()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    all_probs.append(probs)

    # 6 指标
    metrics = six(y_test, probs)
    # PyTorch 推理时间
    pt_time = time_inference_torch(model, X_test_t)
    # ONNX 推理时间
    onnx_time = time_inference_onnx(onnx_path, X_test_t)

    # 文件大小
    pt_bytes, n_params = pt_size_inference(pt_path)
    onnx_bytes = file_size(onnx_path)
    # 量化版
    h_path = os.path.join(BASE, f"model_v4_se_23dim_ch32_hybrid_s{seed}.h")
    quant_data_bytes, quant_n_params = hybrid_h_data_size(h_path)
    quant_file_bytes = file_size(h_path)

    results["per_seed"][seed] = {
        "metrics": metrics,
        "pt_file_size_bytes": pt_bytes,
        "pt_file_size_kb": round(pt_bytes / 1024, 2),
        "onnx_file_size_bytes": onnx_bytes,
        "onnx_file_size_kb": round(onnx_bytes / 1024, 2),
        "quant_data_size_bytes": quant_data_bytes,
        "quant_data_size_kb": round(quant_data_bytes / 1024, 2),
        "quant_file_size_kb": round(quant_file_bytes / 1024, 2),
        "n_params": n_params,
        "quant_n_params": quant_n_params,
        "infer_time_pt_ms": pt_time["single_ms_mean"],
        "infer_time_onnx_ms": onnx_time["single_ms_mean"],
        "infer_time_onnx_p95_ms": onnx_time["single_ms_p95"],
        "throughput_per_s": onnx_time["throughput_samples_per_s"],
    }
    all_sizes.append(onnx_bytes)
    print(f"\nSeed {seed}:")
    print(f"  {fmt(metrics)}")
    print(f"  Params={n_params} | .pt={pt_bytes/1024:.1f}KB | .onnx={onnx_bytes/1024:.1f}KB | quant_data={quant_data_bytes/1024:.1f}KB")
    print(f"  Infer: PT={pt_time['single_ms_mean']:.3f}ms | ONNX={onnx_time['single_ms_mean']:.3f}ms (p95={onnx_time['single_ms_p95']:.3f}ms) | "
          f"Throughput={onnx_time['throughput_samples_per_s']:.0f} samples/s")

# ---------- 2. 5-Seed Ensemble ----------
print("\n" + "="*72)
print("  5-Seed Ensemble (Probability Averaging)")
print("="*72)

ensemble_probs = np.mean(all_probs, axis=0)
metrics_ens = six(y_test, ensemble_probs)
metrics_ens_t47 = six(y_test, ensemble_probs, (ensemble_probs >= 0.47).astype(int))

# 集成推理时间 = 5 个 ONNX 串行
print("\nEnsemble inference (5 ONNX models, sequential):")
ensemble_times = []
for seed in SEEDS:
    onnx_path = os.path.join(BASE, f"model_v4_se_23dim_ch32_s{seed}.onnx")
    t = time_inference_onnx(onnx_path, X_test_t, n_runs=100)
    ensemble_times.append(t["single_ms_mean"])
ensemble_infer_ms = sum(ensemble_times)
print(f"  5 × single = {ensemble_infer_ms:.3f} ms/sample")

# 累计 ONNX 大小
total_onnx_kb = sum(all_sizes) / 1024
print(f"  Total ONNX size: {total_onnx_kb:.1f} KB (5 models)")

results["ensemble"] = {
    "metrics_t05": metrics_ens,
    "metrics_t047": metrics_ens_t47,
    "infer_time_5models_ms": float(ensemble_infer_ms),
    "infer_time_per_model_ms": float(np.mean(ensemble_times)),
    "total_onnx_size_kb": round(total_onnx_kb, 2),
    "avg_single_onnx_size_kb": round(np.mean(all_sizes) / 1024, 2),
}

# ---------- 3. 量化模型大小（Hybrid INT8）----------
print("\n" + "="*72)
print("  Quantized Hybrid INT8 Models (real weight data only)")
print("="*72)
quant_data_sizes = []
quant_n_params_list = []
for seed in SEEDS:
    h_path = os.path.join(BASE, f"model_v4_se_23dim_ch32_hybrid_s{seed}.h")
    data_bytes, n_params = hybrid_h_data_size(h_path)
    quant_data_sizes.append(data_bytes)
    quant_n_params_list.append(n_params)
    file_kb = file_size(h_path) / 1024
    print(f"  Seed {seed}: data={data_bytes/1024:.1f}KB ({n_params} params) | .h source={file_kb:.1f}KB")

print(f"\n  Avg data per model: {np.mean(quant_data_sizes)/1024:.1f} KB ({np.mean(quant_n_params_list):.0f} params)")
print(f"  Total 5-model ensemble (quant data): {sum(quant_data_sizes)/1024:.1f} KB")
print(f"  Compression vs FP32 ONNX: {np.mean(all_sizes)/np.mean(quant_data_sizes):.2f}x")

results["quantized"] = {
    "avg_data_size_kb": round(float(np.mean(quant_data_sizes)) / 1024, 2),
    "total_5model_data_kb": round(float(sum(quant_data_sizes)) / 1024, 2),
    "avg_n_params": float(np.mean(quant_n_params_list)),
    "compression_ratio_vs_fp32_onnx": round(float(np.mean(all_sizes)) / float(np.mean(quant_data_sizes)), 2),
}

# ---------- 保存 ----------
out_path = os.path.join(BASE, "eval_6metrics_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[OK] Saved: {out_path}")

# ---------- 最终汇总 ----------
print("\n" + "="*72)
print("  ★ 最终汇总 — 6 指标 (5-Seed Ensemble, τ=0.5)")
print("="*72)
m = results["ensemble"]["metrics_t05"]
print(f"  Accuracy : {m['Accuracy']:.4f}")
print(f"  Precision: {m['Precision']:.4f}")
print(f"  Recall   : {m['Recall']:.4f}")
print(f"  F1-Score : {m['F1-Score']:.4f}")
print(f"  Det. Time: {results['ensemble']['infer_time_5models_ms']:.3f} ms/sample (5 models sequential)")
print(f"  Model Size: {results['quantized']['avg_data_size_kb']:.1f} KB (INT8 quant data) / "
      f"{results['ensemble']['avg_single_onnx_size_kb']:.1f} KB (FP32 ONNX)")