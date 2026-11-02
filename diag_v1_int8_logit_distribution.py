"""
diag_v1_int8_logit_distribution.py  -- 量化诊断 v1: 跑现有 INT8 .tflite 看 logit 分布

目的: 验证 B 路线 head FP32 假设 — 当前 v1 (head INT8) 把 logit 推到饱和区

输出:
  - v1 raw logit 分布 (反量化后, T=1.0)
  - v1 prob 分布 (T=1.0 / T=2.0 / T=5.0 三档)
  - 假设 head FP32 (logit 无量化误差) 的近似分布 (用 dequant 后的 '近似 raw logit')
  - 4 指标 F1/Recall/FPR/Precision (T=1.0/2.0/5.0 + thr=0.5/0.7/0.9)

不依赖 ai_edge_litert, 只用 tensorflow.lite.Interpreter
"""
import sys
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from sklearn.metrics import f1_score, precision_score, recall_score


KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEED = 42


def load_data():
    X = np.load('X_test_binary_v2_scada_window16.npy').astype(np.float32)[:, :, KEEP_23]
    X = np.clip(X, -10, 10)
    y = np.load('y_test_binary_v2_scada_window16.npy').astype(np.int64)
    return X, y


def run_int8_tflite_get_logits(tflite_path, X):
    """跑 INT8 .tflite, 返回 raw logit 数组 (反量化后, T=1.0).

    输入 int8, 输出 int8 → logit = (q - zp) * scale
    """
    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']
    print(f"  input:  dtype={inp_d['dtype']}  scale={in_scale:.6f}  zp={in_zp}")
    print(f"  output: dtype={out_d['dtype']}  scale={out_scale:.6f}  zp={out_zp}")

    logits = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        x_t = np.transpose(X[i:i+1], (0, 2, 1))  # (1, T, C) → (1, C, T)
        x_q = np.clip(np.round(x_t / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_d['index'], x_q)
        interp.invoke()
        y_q = interp.get_tensor(out_d['index'])
        logit = (y_q.astype(np.float32) - out_zp) * out_scale
        logits[i] = logit[0] if logit.ndim > 0 else logit
    return logits


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))   # 防 overflow


def kpi(yt, yp, name):
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    print(f"  {name:32s}  TP={tp:5d} FP={fp:5d} TN={tn:5d} FN={fn:5d}  "
          f"F1={f1:.4f}  R={recall:.4f}  FPR={fpr:.4f}  P={prec:.4f}")
    return dict(tp=tp, fp=fp, f1=f1, recall=recall, fpr=fpr, prec=prec)


def histo(arr, name, bins=None):
    if bins is None:
        bins = [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 50]
    h, _ = np.histogram(arr, bins=bins)
    total = len(arr)
    print(f"  {name} (n={total}):")
    for i in range(len(h)):
        pct = h[i] / total * 100
        print(f"    [{bins[i]:6.1f}, {bins[i+1]:6.1f}): {h[i]:5d}  ({pct:5.1f}%)")


def main():
    tflite_path = f'model_v4_se_23dim_ch32_qat_s{SEED}_int8.tflite'
    print(f"加载 v1 INT8 .tflite: {tflite_path}")
    X, y = load_data()
    print(f"X: {X.shape}  y: {y.shape}  attack%={y.mean()*100:.1f}\n")

    print("=" * 70)
    print("[1] 跑 v1 INT8 .tflite, 收集 raw logit")
    print("=" * 70)
    logits_v1 = run_int8_tflite_get_logits(tflite_path, X)

    print(f"\n  v1 raw logit (post dequant, T=1.0):")
    print(f"    min={logits_v1.min():.3f}  max={logits_v1.max():.3f}  "
          f"mean={logits_v1.mean():.3f}  std={logits_v1.std():.3f}")
    print(f"    P5={np.percentile(logits_v1, 5):.3f}  P50={np.percentile(logits_v1, 50):.3f}  "
          f"P95={np.percentile(logits_v1, 95):.3f}")

    print(f"\n  Attack windows logit:")
    print(f"    min={logits_v1[y==1].min():.3f}  max={logits_v1[y==1].max():.3f}  "
          f"mean={logits_v1[y==1].mean():.3f}  std={logits_v1[y==1].std():.3f}")
    print(f"  Normal windows logit:")
    print(f"    min={logits_v1[y==0].min():.3f}  max={logits_v1[y==0].max():.3f}  "
          f"mean={logits_v1[y==0].mean():.3f}  std={logits_v1[y==0].std():.3f}")

    print()
    print("=" * 70)
    print("[2] Logit 直方图")
    print("=" * 70)
    histo(logits_v1[y == 0], "Normal windows logit")
    print()
    histo(logits_v1[y == 1], "Attack windows logit")

    print()
    print("=" * 70)
    print("[3] 4 指标扫描 (T ∈ {1.0, 2.0, 5.0, 10.0}, thr ∈ {0.5, 0.7, 0.9})")
    print("=" * 70)
    for T in [1.0, 2.0, 5.0, 10.0]:
        probs = sigmoid(logits_v1 / T)
        print(f"\n  T={T}: prob range [{probs.min():.4f}, {probs.max():.4f}]  mean={probs.mean():.4f}")
        for thr in [0.5, 0.7, 0.9]:
            yp = (probs >= thr).astype(int)
            kpi(y, yp, f"  T={T} thr={thr}")

    print()
    print("=" * 70)
    print("[4] 假设 head FP32 (logit 退饱和估算)")
    print("=" * 70)
    print("  如果 head FP32, logit 不会被量化到 int8 边界.")
    print("  估算: 当前 v1 raw logit 已经是 dequant 后的 logit (跟 FP32 模型预测的差异).")
    print("  假设 v2 (FP32 head) 把 logit 范围压缩到 [-15, +15] (去除量化误差 ~30%).")
    print("  模拟: logits_v2 = logits_v1 * 0.7 (粗略估算 30% 量化噪声去除)")
    print()
    logits_v2_sim = logits_v1 * 0.7
    print(f"  v2 模拟 raw logit:")
    print(f"    min={logits_v2_sim.min():.3f}  max={logits_v2_sim.max():.3f}  "
          f"mean={logits_v2_sim.mean():.3f}  std={logits_v2_sim.std():.3f}")
    print()
    for T in [1.0, 2.0, 3.0]:
        probs = sigmoid(logits_v2_sim / T)
        print(f"\n  v2 模拟 T={T}: prob range [{probs.min():.4f}, {probs.max():.4f}]  mean={probs.mean():.4f}")
        for thr in [0.5, 0.7, 0.9]:
            yp = (probs >= thr).astype(int)
            kpi(y, yp, f"  v2 sim T={T} thr={thr}")


if __name__ == '__main__':
    main()
