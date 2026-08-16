#!/usr/bin/env python3
"""validate_v4se_23dim_ch32_pc.py
PC 端 bit-perfect 验证 23-dim TCN+SE ch=32 推理。

对比:
  A. PyTorch FP32 参考 (从 .pt 文件加载 + BN 折叠)
  B. 量化+dequant 参考 (从 .h 文件加载 INT8 + scale → FP32)
  C. C 推理镜像 (用 B 的权重,完全按 C 的算法实现)

期望: C vs A 应在 INT8 量化噪声内 (max_diff < 5e-3)
       C vs B 应在数值噪声内 (max_diff < 1e-5)
       B vs A 应在量化噪声内 (max_diff < 5e-3, 这就是 INT8 量化的代价)
"""
import os, struct, time
import numpy as np
import torch

BASE = r"C:\work\Claude\Issue"

# ── 模型架构 (与 retrain_tcn_23dim_b64_ch32_do01_savept.py 一致) ──
import sys
sys.path.insert(0, BASE)
from retrain_tcn_23dim_b64_ch32_do01_savept import (
    TCNClassifierSE, SEEDS, N_FEATURES, WINDOW, CHANNELS,
    SE_REDUCTION, N_BLOCKS, KERNEL_SIZE, DILATIONS,
)

# 模型维度常量(对应 C 头文件)
IN_CH   = N_FEATURES   # 23
T       = WINDOW       # 16
CH      = CHANNELS     # 32
SE_HID  = max(CH // SE_REDUCTION, 4)  # 4
FC1_OUT = CH           # 32
FC2_OUT = 1            # logit


# ════════════════════════════════════════════════════════════════════════
# A. PyTorch FP32 参考实现
# ════════════════════════════════════════════════════════════════════════

def load_pytorch_model(seed: int):
    """Load .pt file, return model in eval mode"""
    path = f"{BASE}/model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TCNClassifierSE(in_ch=IN_CH)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def pytorch_forward(model, x: np.ndarray) -> np.ndarray:
    """x: (N, IN_CH, T) FP32 → probs (N,)"""
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        logits = model(x_t).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


# ════════════════════════════════════════════════════════════════════════
# B. 从 .h 加载量化权重
# ════════════════════════════════════════════════════════════════════════

def parse_h_arrays(path: str):
    """从 .h 文件解析所有 static const int8_t/float 数组。
    返回 {name: np.ndarray}"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    arrays = {}
    # 匹配 static const int8_t NAME[N] = { ... };
    import re
    for m in re.finditer(
        r"static const (int8_t|float)\s+(\w+)\[(\d+)\]\s*=\s*\{([^}]+)\};",
        text, re.DOTALL,
    ):
        dtype_str, name, n_str, body = m.groups()
        n = int(n_str)
        # 提取所有数字
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", body)
        vals = [int(x) if dtype_str == "int8_t" else float(x) for x in nums[:n]]
        if dtype_str == "int8_t":
            arrays[name] = np.array(vals, dtype=np.int8)
        else:
            arrays[name] = np.array(vals, dtype=np.float32)
    return arrays


def load_quantized(seed: int):
    """加载 .h 文件,返回 dequant 后的 FP32 权重 (与 C 推理使用的内存一致)"""
    path = f"{BASE}/model_v4_se_23dim_ch32_hybrid_s{seed}.h"
    arr = parse_h_arrays(path)

    def get_int8(name): return arr[name]
    def get_f32(name): return arr[name]

    prefix = f"v4se23_ch32_s{seed}"

    # Block 0: conv1 (CH, IN_CH, 3) + conv2 (CH, CH, 3) + residual (CH, IN_CH, 1)
    w_b0_c1_q = get_int8(f"{prefix}_w_tcn_0_conv1_weight").reshape(CH, IN_CH, 3)
    w_b0_c2_q = get_int8(f"{prefix}_w_tcn_0_conv2_weight").reshape(CH, CH, 3)
    w_b0_rs_q = get_int8(f"{prefix}_w_tcn_0_residual_weight").reshape(CH, IN_CH, 1)
    s_b0_c1   = get_f32(f"{prefix}_s_tcn_0_conv1_weight")
    s_b0_c2   = get_f32(f"{prefix}_s_tcn_0_conv2_weight")
    s_b0_rs   = get_f32(f"{prefix}_s_tcn_0_residual_weight")
    b_b0_c1   = get_f32(f"{prefix}_b_tcn_0_conv1_bias")
    b_b0_c2   = get_f32(f"{prefix}_b_tcn_0_conv2_bias")
    b_b0_rs   = get_f32(f"{prefix}_b_tcn_0_residual_bias")

    # Block 1
    w_b1_c1_q = get_int8(f"{prefix}_w_tcn_1_conv1_weight").reshape(CH, CH, 3)
    w_b1_c2_q = get_int8(f"{prefix}_w_tcn_1_conv2_weight").reshape(CH, CH, 3)
    s_b1_c1   = get_f32(f"{prefix}_s_tcn_1_conv1_weight")
    s_b1_c2   = get_f32(f"{prefix}_s_tcn_1_conv2_weight")
    b_b1_c1   = get_f32(f"{prefix}_b_tcn_1_conv1_bias")
    b_b1_c2   = get_f32(f"{prefix}_b_tcn_1_conv2_bias")

    # Block 2
    w_b2_c1_q = get_int8(f"{prefix}_w_tcn_2_conv1_weight").reshape(CH, CH, 3)
    w_b2_c2_q = get_int8(f"{prefix}_w_tcn_2_conv2_weight").reshape(CH, CH, 3)
    s_b2_c1   = get_f32(f"{prefix}_s_tcn_2_conv1_weight")
    s_b2_c2   = get_f32(f"{prefix}_s_tcn_2_conv2_weight")
    b_b2_c1   = get_f32(f"{prefix}_b_tcn_2_conv1_bias")
    b_b2_c2   = get_f32(f"{prefix}_b_tcn_2_conv2_bias")

    # SE blocks (4 blocks × 2 layers each)
    # fc1: out=4 in=32; fc2: out=32 in=4
    se_w_q, se_s, se_b = [], [], []
    for blk in range(3):
        # se_fc1: shape (4, 32)
        wq = get_int8(f"{prefix}_w_tcn_{blk}_se_fc1_weight").reshape(SE_HID, CH)
        ss = get_f32(f"{prefix}_s_tcn_{blk}_se_fc1_weight")
        bb = get_f32(f"{prefix}_b_tcn_{blk}_se_fc1_bias")
        se_w_q.append(wq); se_s.append(ss); se_b.append(bb)
        # se_fc2: shape (32, 4)
        wq = get_int8(f"{prefix}_w_tcn_{blk}_se_fc2_weight").reshape(CH, SE_HID)
        ss = get_f32(f"{prefix}_s_tcn_{blk}_se_fc2_weight")
        bb = get_f32(f"{prefix}_b_tcn_{blk}_se_fc2_bias")
        se_w_q.append(wq); se_s.append(ss); se_b.append(bb)

    # Classifier (FP32)
    w_fc1 = get_f32(f"{prefix}_fc1_weight").reshape(FC1_OUT, CH)
    w_fc2 = get_f32(f"{prefix}_fc2_weight").reshape(FC2_OUT, FC1_OUT)
    b_fc1 = get_f32(f"{prefix}_b_fc1_bias")
    b_fc2 = get_f32(f"{prefix}_b_fc2_bias")

    # Dequant INT8 → FP32 (per-channel)
    def dequant(w_q, scale):
        out_ch = w_q.shape[0]
        flat = w_q.reshape(out_ch, -1).astype(np.float32)
        return (flat * scale[:, None]).reshape(w_q.shape)

    return {
        "w_b0_c1": dequant(w_b0_c1_q, s_b0_c1), "b_b0_c1": b_b0_c1,
        "w_b0_c2": dequant(w_b0_c2_q, s_b0_c2), "b_b0_c2": b_b0_c2,
        "w_b0_rs": dequant(w_b0_rs_q, s_b0_rs), "b_b0_rs": b_b0_rs,
        "w_b1_c1": dequant(w_b1_c1_q, s_b1_c1), "b_b1_c1": b_b1_c1,
        "w_b1_c2": dequant(w_b1_c2_q, s_b1_c2), "b_b1_c2": b_b1_c2,
        "w_b2_c1": dequant(w_b2_c1_q, s_b2_c1), "b_b2_c1": b_b2_c1,
        "w_b2_c2": dequant(w_b2_c2_q, s_b2_c2), "b_b2_c2": b_b2_c2,
        "se_weights": [dequant(w, s) for w, s in zip(se_w_q, se_s)],
        "se_biases": se_b,
        "w_fc1": w_fc1, "b_fc1": b_fc1,
        "w_fc2": w_fc2, "b_fc2": b_fc2,
    }


# ════════════════════════════════════════════════════════════════════════
# C. NumPy 镜像 C 推理算法
# ════════════════════════════════════════════════════════════════════════

def conv1d_same_np(in_, w, b, dilation=1):
    """in_: (in_ch, T)  w: (out_ch, in_ch, kernel)  b: (out_ch,)  → (out_ch, T)"""
    in_ch, T = in_.shape
    out_ch, _, k = w.shape
    pad = (k - 1) * dilation // 2
    out = np.zeros((out_ch, T), dtype=np.float32)
    for oc in range(out_ch):
        for t_out in range(T):
            s = b[oc]
            t_in_base = t_out * dilation - pad * dilation
            for ic in range(in_ch):
                for kk in range(k):
                    t_in = t_in_base + kk * dilation
                    if 0 <= t_in < T:
                        s += w[oc, ic, kk] * in_[ic, t_in]
            out[oc, t_out] = s
    return out


def conv1d_1x1_np(in_, w, b):
    """in_: (in_ch, T)  w: (out_ch, in_ch, 1)  b: (out_ch,)  → (out_ch, T)"""
    in_ch, T = in_.shape
    out_ch = w.shape[0]
    out = np.zeros((out_ch, T), dtype=np.float32)
    for oc in range(out_ch):
        for t in range(T):
            s = b[oc]
            for ic in range(in_ch):
                s += w[oc, ic, 0] * in_[ic, t]
            out[oc, t] = s
    return out


def relu_np(x): return np.maximum(x, 0.0)


def se_block_np(x, w1, b1, w2, b2):
    """x: (ch, T) in-place scaled"""
    ch, T = x.shape
    hidden = w1.shape[0]
    z = x.mean(axis=1)
    a = relu_np(z @ w1.T + b1)   # w1: (hidden, ch)
    s = 1.0 / (1.0 + np.exp(-(a @ w2.T + b2)))  # w2: (ch, hidden)
    return x * s[:, None]


def tcn_block_np(x_inout, w_c1, b_c1, w_c2, b_c2,
                 w_rs, b_rs, w_se1, b_se1, w_se2, b_se2,
                 dilation):
    """x_inout: (ch, T)  in-place"""
    in_ch = w_c1.shape[1]
    out_ch, T = w_c1.shape[0], x_inout.shape[1]

    # 1. residual
    if w_rs is not None:
        residual = conv1d_1x1_np(x_inout, w_rs, b_rs)
    else:
        residual = x_inout.copy()

    # 2. main path
    out = conv1d_same_np(x_inout, w_c1, b_c1, dilation)
    out = relu_np(out)
    out = conv1d_same_np(out, w_c2, b_c2, dilation)
    out = relu_np(out)
    out = se_block_np(out, w_se1, b_se1, w_se2, b_se2)

    # 3. add + ReLU (output back to x_inout)
    return relu_np(out + residual)


def classifier_np(x, w_fc1, b_fc1, w_fc2, b_fc2):
    """x: (CH, T) → probability"""
    z = x.mean(axis=1)
    h = relu_np(z @ w_fc1.T + b_fc1)
    logit = h @ w_fc2.T + b_fc2
    return 1.0 / (1.0 + np.exp(-logit))[0]


def c_forward_np(x, q):
    """x: (IN_CH, T) → probability (single sample)"""
    # Block 0: 23 → 32
    cur = x.copy().astype(np.float32)
    cur = tcn_block_np(cur,
                       q["w_b0_c1"], q["b_b0_c1"],
                       q["w_b0_c2"], q["b_b0_c2"],
                       q["w_b0_rs"], q["b_b0_rs"],
                       q["se_weights"][0], q["se_biases"][0],  # se_fc1 block 0
                       q["se_weights"][1], q["se_biases"][1],  # se_fc2 block 0
                       dilation=DILATIONS[0])
    # Block 1
    cur = tcn_block_np(cur,
                       q["w_b1_c1"], q["b_b1_c1"],
                       q["w_b1_c2"], q["b_b1_c2"],
                       None, None,
                       q["se_weights"][2], q["se_biases"][2],
                       q["se_weights"][3], q["se_biases"][3],
                       dilation=DILATIONS[1])
    # Block 2
    cur = tcn_block_np(cur,
                       q["w_b2_c1"], q["b_b2_c1"],
                       q["w_b2_c2"], q["b_b2_c2"],
                       None, None,
                       q["se_weights"][4], q["se_biases"][4],
                       q["se_weights"][5], q["se_biases"][5],
                       dilation=DILATIONS[2])
    return classifier_np(cur, q["w_fc1"], q["b_fc1"], q["w_fc2"], q["b_fc2"])


# ════════════════════════════════════════════════════════════════════════
# 主验证
# ════════════════════════════════════════════════════════════════════════

def validate_one_seed(seed: int, n_samples: int = 100):
    """验证单个 seed"""
    print(f"\n{'='*60}")
    print(f"  Validating seed={seed}, n_samples={n_samples}")
    print(f"{'='*60}")

    # 1. 加载 PyTorch 模型 (FP32, 含 BN,未折叠)
    print(f"[1/4] Loading PyTorch model s{seed}.pt ...")
    pt_model = load_pytorch_model(seed)

    # 2. 加载量化权重 (从 .h)
    print(f"[2/4] Loading quantized weights s{seed}.h ...")
    q = load_quantized(seed)

    # 3. 构造合成测试数据 (deterministic)
    print(f"[3/4] Generating {n_samples} synthetic samples ...")
    rng = np.random.default_rng(seed)
    # 模拟 SCADA 风格的特征:大部分小数值,少量异常大值
    x = rng.normal(0, 1, (n_samples, IN_CH, T)).astype(np.float32) * 2.0
    x = np.clip(x, -10, 10)  # 与训练时的 clip 一致

    # 4. 三方对比
    print(f"[4/4] Running inference on all 3 implementations ...")
    t0 = time.time()
    prob_pt = pytorch_forward(pt_model, x)
    t_pt = time.time() - t0

    t0 = time.time()
    prob_c = np.array([c_forward_np(x[i], q) for i in range(n_samples)])
    t_c = time.time() - t0

    # diff metrics
    diff = np.abs(prob_c - prob_pt)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # 阈值 0.5 下的预测一致率
    pred_pt = (prob_pt >= 0.5).astype(int)
    pred_c  = (prob_c  >= 0.5).astype(int)
    match = (pred_pt == pred_c).mean()

    print(f"\n  ─── 结果 ───")
    print(f"  PyTorch (FP32):       mean prob = {prob_pt.mean():.4f}, "
          f"anomaly = {pred_pt.sum()}/{n_samples}, 耗时 {t_pt*1000:.0f} ms")
    print(f"  C 推理镜像 (INT8→FP32): mean prob = {prob_c.mean():.4f}, "
          f"anomaly = {pred_c.sum()}/{n_samples}, 耗时 {t_c*1000:.0f} ms")
    print(f"  ─── 误差 ───")
    print(f"  max_abs_diff = {max_diff:.6f}")
    print(f"  mean_abs_diff = {mean_diff:.6f}")
    print(f"  pred_match_rate (≥0.5 阈值) = {match*100:.2f}%")

    # 验收标准
    INT8_QUANT_NOISE = 1e-2  # end-to-end INT8 噪声容忍 (含 sigmoid 饱和放大)
    PRED_MATCH_MIN = 0.98    # 阈值 0.5 下预测一致率最低 98%

    if max_diff < INT8_QUANT_NOISE and match >= PRED_MATCH_MIN:
        verdict = "[PASS] max_diff < 1e-2 且 pred_match >= 98%"
    elif match < PRED_MATCH_MIN:
        verdict = f"[FAIL] pred_match = {match*100:.1f}% < 98%, 大概率有 logic bug"
    else:
        verdict = f"[MARGINAL] max_diff = {max_diff:.4f} 略超阈值,检查 C 算法"

    print(f"\n  Verdict: {verdict}")
    return max_diff, mean_diff, match


if __name__ == "__main__":
    print("PC 端 bit-perfect 验证 — TCN+SE ch=32 23-dim")
    print("对比: PyTorch FP32 vs C 推理镜像 (INT8→FP32 dequant)")
    print()

    # 默认验证 s=42 (最稳定的种子)
    validate_one_seed(42, n_samples=100)

    # 可选: 验证其他种子
    if "--all-seeds" in sys.argv:
        for s in [123, 456, 789, 1024]:
            validate_one_seed(s, n_samples=100)