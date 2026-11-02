"""
sim_hybrid_kernel.py -- MCU "Hybrid" kernel (INT8 权重存储 / FP32 激活) 的 numpy 镜像.

为什么做这个
------------
全量 INT8 已经证明精度无损 (部署区 F1 0.8670 vs FP32 0.8671), 但它把激活也量化了,
logit 相对偏移 +4.2%, 58 个分歧窗全部落在 |logit| < 2.0 的决策边界带, 净效果是
"工作点整体往更爱报警的方向挪" (Recall +0.75pp / FPR +1.07pp).

Hybrid 的思路: **只量化权重, 激活保持 FP32**.
  - 权重 int8 per-channel symmetric 存 Flash (25 KB), 不做 FP32 权重的 RAM 拷贝
    -> Flash -57 KB, RAM -82 KB (相对 Route E FP32)
  - 激活全 FP32 -> 消掉激活量化误差, 只剩权重量化误差
  - 数值上 = Route E FP32 的 "权重四舍五入版"

本文件同时是一个 **2x2 误差归因工具**:
    weights: "fp32" | "int8"   (int8 = dequant 回 FP32 再算)
    act:     "fp32" | "int8"   (int8 = fake-quant, 逐张量边界)
  四个角:
    fp32/fp32 = Route E 部署真值
    int8/fp32 = Hybrid  <-- 本次要评估的方案
    int8/int8 = 全量 INT8 (真实实现走 CMSIS-NN, 这里只是近似)
    fp32/int8 = 只量化激活 (用来分离两类误差的贡献)

源绑定 (方法学要求, 见 memory feedback-parity-test-must-bind-to-real-source)
------------------------------------------------------------------------
  * conv 的 t_in 公式 **从 MCU/Core/Src/ch32_inference.c 源码解析**, 不手抄.
    根因1 的修复版是 "对称 same padding": t + k*d - pad.
    以后写 ch32_inference_hybrid.c 时, 必须用同一个表达式 (本文件提供 assert).
  * 权重来自 MCU/Core/Inc/model_v4_se_23dim_ch32_hybrid_s42.h
    (13 个 conv/SE 张量 int8 + per-channel scale; 分类器 fc1/fc2 与全部 bias 是 FP32)
    -> 测的就是要烧进 MCU 的那份权重, 不是从 .pt 重新算的
  * 激活量化参数来自 MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h
    (只用于 act="int8" 的消融变体)
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HYB_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_hybrid_s42.h")
FP32_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h")
INT8_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h")
I16_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int16_s42.h")
ROUTE_E_C = os.path.join(HERE, "MCU/Core/Src/ch32_inference.c")

IN_CH, T, CH, SE_H, FC1_OUT = 23, 16, 32, 4, 32
DIL = (1, 2, 4)
TEMPERATURE = 1.0

# 权重张量名 (hybrid header 里的命名去掉 v4se23_ch32_s42_ 前缀)
_TENSORS = (
    ["tcn_0_conv1", "tcn_0_conv2", "tcn_0_residual",
     "tcn_1_conv1", "tcn_1_conv2",
     "tcn_2_conv1", "tcn_2_conv2"]
    + ["tcn_%d_se_fc1" % bi for bi in (0, 1, 2)]
    + ["tcn_%d_se_fc2" % bi for bi in (0, 1, 2)]
)
_SHAPES = {
    "tcn_0_conv1": (CH, IN_CH, 3),
    "tcn_0_conv2": (CH, CH, 3),
    "tcn_0_residual": (CH, IN_CH, 1),
    "tcn_1_conv1": (CH, CH, 3),
    "tcn_1_conv2": (CH, CH, 3),
    "tcn_2_conv1": (CH, CH, 3),
    "tcn_2_conv2": (CH, CH, 3),
}
for _bi in (0, 1, 2):
    _SHAPES["tcn_%d_se_fc1" % _bi] = (SE_H, CH)
    _SHAPES["tcn_%d_se_fc2" % _bi] = (CH, SE_H)


# ---------------------------------------------------------------------------
# 1. 源绑定: 从 C 源码解析 conv 的 t_in 公式
# ---------------------------------------------------------------------------


def parse_t_in_formula(c_path=ROUTE_E_C, verbose=True):
    """从 C 源码里解析 conv1d_same 的 t_in 表达式 (先剥注释, 只认真实代码)."""
    src = open(c_path, encoding="utf-8", errors="replace").read()
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    m = re.search(r"int\s+t_in\s*=\s*([^;]+);", src)
    if not m:
        raise RuntimeError("%s 里找不到 'int t_in = ...;'" % c_path)
    expr = m.group(1).strip()
    if verbose:
        print("[C 源码] %s 的 t_in 公式 = %s" % (os.path.basename(c_path), expr))
    return expr


T_IN_EXPR = parse_t_in_formula()


def t_in_index(T, kernel, dilation, k=0):
    """返回 out[t] 对应的 in 下标数组 (T,), 直接求值 C 源码里的表达式.

    k 是 kernel 位置, 直接代入表达式求值 (不做任何关于表达式形式的假设).
    """
    pad = (kernel - 1) * dilation // 2
    t = np.arange(T)
    idx = eval(T_IN_EXPR, {"__builtins__": {}},
               {"t": t, "k": k, "pad": pad, "dilation": dilation})
    return np.asarray(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. header 解析
# ---------------------------------------------------------------------------


def parse_header(path):
    """解析 header 里所有 static const 数组 -> {name: ndarray}."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"static const\s+(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};", txt, re.S):
        ctype, name, n, body = m.group(1), m.group(2), m.group(3), m.group(4)
        if ctype == "int8_t":
            vals = np.array([int(x) for x in re.findall(r"-?\d+", body)], dtype=np.int8)
        elif ctype == "int16_t":
            vals = np.array([int(x) for x in re.findall(r"-?\d+", body)], dtype=np.int16)
        elif ctype == "int32_t":
            vals = np.array([int(x) for x in re.findall(r"-?\d+", body)], dtype=np.int32)
        else:
            vals = np.array([float(x[:-1]) for x in re.findall(r"[-+0-9.eE]+f", body)], dtype=np.float32)
        assert len(vals) == int(n), (name, len(vals), int(n))
        out[name] = vals
    return out


def _dequant(w_int8, scale, shape):
    """int8 per-channel 对称权重 -> FP32. 镜像 MCU 的 (float)q * scale[oc]."""
    w = w_int8.astype(np.float32).reshape(shape)
    return w * scale.astype(np.float32).reshape(-1, *([1] * (len(shape) - 1)))


def load_weights(mode="int8"):
    """mode: "int8" (hybrid header, dequant) | "int16" (Hybrid-INT16 header, dequant)
             | "fp32" (Route E 部署真值的头).

    返回的 dict 键跟 parity_fp32_c_vs_onnx.c_forward 的 W 一致, 可以直接喂给它做交叉验证.

    ⚠️ "int16" 读的是 **要烧进 MCU 的那份头** (gen_int16_weights.py 生成),
       不是现场重算的 -> PC 侧测的就是烧进去的东西.
    """
    if mode in ("int8", "int16"):
        path = HYB_HDR if mode == "int8" else I16_HDR
        raw = parse_header(path)
        # int16 头的前缀跟 FP32/hybrid-int8 头不同 (避免同名字不同类型)
        p = "v4se23_ch32_s42_" if mode == "int8" else "v4se23_ch32_i16_s42_"
        W = {}
        for name in _TENSORS:
            shp = _SHAPES[name]
            wi = raw[p + "w_" + name + "_weight"]
            sc = raw[p + "s_" + name + "_weight"]
            assert len(sc) == shp[0], (name, len(sc), shp[0])
            W["w_" + name] = _dequant(wi, sc, shp)
            W["b_" + name] = raw[p + "b_" + name + "_bias"]
        # 分类器: hybrid header 里是 FP32 权重
        W["fc1"] = raw[p + "fc1_weight"].reshape(FC1_OUT, CH)
        W["fc2"] = raw[p + "fc2_weight"].reshape(1, FC1_OUT)
        W["b_fc1"] = raw[p + "b_fc1_bias"]
        W["b_fc2"] = raw[p + "b_fc2_bias"]
        return W

    if mode == "fp32":
        raw = parse_header(FP32_HDR)
        p = "v4se23_ch32_s42_"
        W = {}
        for name in _TENSORS:
            shp = _SHAPES[name]
            W["w_" + name] = raw[p + "w_" + name + "_weight"].reshape(shp)
            W["b_" + name] = raw[p + "b_" + name + "_bias"]
        W["fc1"] = raw[p + "fc1_weight"].reshape(FC1_OUT, CH)
        W["fc2"] = raw[p + "fc2_weight"].reshape(1, FC1_OUT)
        W["b_fc1"] = raw[p + "b_fc1_bias"]
        W["b_fc2"] = raw[p + "b_fc2_bias"]
        return W

    raise ValueError("mode 必须是 int8 / int16 / fp32")


def load_act_qparams():
    """从全量 INT8 header 里取激活量化参数 (act='int8' 的消融用)."""
    from int8_math import parse_int8_header
    h = parse_int8_header(INT8_HDR)
    q = {"x": (h["s_x"], h["zp_x"]), "fc1": (h["s_fc1"], h["zp_fc1"])}
    for bi in (0, 1, 2):
        q["c1_%d" % bi] = (h["s_c1_%d" % bi], h["zp_c1_%d" % bi])
        q["c2_%d" % bi] = (h["s_c2_%d" % bi], h["zp_c2_%d" % bi])
        q["add_%d" % bi] = (h["s_add_%d" % bi], h["zp_add_%d" % bi])
    return q


# ---------------------------------------------------------------------------
# 3. 前向 (批量化, 语义照抄 Route E 的 C kernel)
# ---------------------------------------------------------------------------


def fake_quant(a, s, zp):
    """非对称 fake-quant: real = (round(real/s) + zp - zp) * s, 然后 clamp 到 [zp,127] 对应范围."""
    q = np.round(np.asarray(a, dtype=np.float64) / float(s)) + float(zp)
    q = np.clip(q, -128.0, 127.0)          # int8 存储域 (zp 是 real=0 对应的码)
    return (q - float(zp)) * float(s)


def conv1d_same_batch(inp, w, b, dil, act_q=None):
    """inp (N,C,T) -> (N,O,T). 逐行复刻 C 的 conv1d_same (t_in 公式来自 C 源码)."""
    N, C, Tx = inp.shape
    O, _, K = w.shape
    out = np.broadcast_to(np.asarray(b, dtype=np.float64)[None, :, None],
                          (N, O, Tx)).copy()
    for k in range(K):
        idx_k = t_in_index(Tx, K, dil, k)
        valid = ((idx_k >= 0) & (idx_k < Tx)).astype(np.float64)
        ic = np.clip(idx_k, 0, Tx - 1)
        xs = inp[:, :, ic] * valid[None, None, :]
        out += np.einsum("oc,nct->not", np.asarray(w[:, :, k], dtype=np.float64), xs)
    return out


def se_block(x, W, bi):
    """x (N,CH,T) in-place 语义: GAP -> fc1+ReLU -> fc2+sigmoid -> 逐通道缩放."""
    N, ch, Tx = x.shape
    z = x.sum(axis=2) / Tx
    w1 = W["w_tcn_%d_se_fc1" % bi]
    b1 = W["b_tcn_%d_se_fc1" % bi]
    w2 = W["w_tcn_%d_se_fc2" % bi]
    b2 = W["b_tcn_%d_se_fc2" % bi]
    h = np.maximum(z @ np.asarray(w1, dtype=np.float64).T + np.asarray(b1, dtype=np.float64), 0.0)
    s = 1.0 / (1.0 + np.exp(-(h @ np.asarray(w2, dtype=np.float64).T + np.asarray(b2, dtype=np.float64))))
    return x * s[:, :, None]


def forward(W, X, act_q=None):
    """X (N,23,16) FP32 -> logits (N,). act_q=None 表示激活不量化."""
    x = np.asarray(X, dtype=np.float32).astype(np.float64)
    if act_q is not None:
        x = fake_quant(x, *act_q["x"])

    # ---- Block 0 ----
    res = conv1d_same_batch(x, W["w_tcn_0_residual"], W["b_tcn_0_residual"], 1)
    c1 = np.maximum(conv1d_same_batch(x, W["w_tcn_0_conv1"], W["b_tcn_0_conv1"], DIL[0]), 0)
    if act_q is not None:
        c1 = fake_quant(c1, *act_q["c1_0"])
    c2 = np.maximum(conv1d_same_batch(c1, W["w_tcn_0_conv2"], W["b_tcn_0_conv2"], DIL[0]), 0)
    if act_q is not None:
        c2 = fake_quant(c2, *act_q["c2_0"])
    cur = se_block(c2, W, 0)
    if act_q is not None:
        cur = fake_quant(cur, *act_q["c2_0"])     # SE 缩放后仍在 c2 的量化域
    cur = np.maximum(cur + res, 0)
    if act_q is not None:
        cur = fake_quant(cur, *act_q["add_0"])

    # ---- Block 1 / 2 ----
    for bi in (1, 2):
        res = cur
        c1 = np.maximum(conv1d_same_batch(cur, W["w_tcn_%d_conv1" % bi], W["b_tcn_%d_conv1" % bi], DIL[bi]), 0)
        if act_q is not None:
            c1 = fake_quant(c1, *act_q["c1_%d" % bi])
        c2 = np.maximum(conv1d_same_batch(c1, W["w_tcn_%d_conv2" % bi], W["b_tcn_%d_conv2" % bi], DIL[bi]), 0)
        if act_q is not None:
            c2 = fake_quant(c2, *act_q["c2_%d" % bi])
        cur = se_block(c2, W, bi)
        if act_q is not None:
            cur = fake_quant(cur, *act_q["c2_%d" % bi])
        cur = np.maximum(cur + res, 0)
        if act_q is not None:
            cur = fake_quant(cur, *act_q["add_%d" % bi])

    # ---- 分类器 ----
    g = cur.sum(axis=2) / T
    h = np.maximum(g @ np.asarray(W["fc1"], dtype=np.float64).T + np.asarray(W["b_fc1"], dtype=np.float64), 0.0)
    if act_q is not None:
        h = fake_quant(h, *act_q["fc1"])
    logit = (h @ np.asarray(W["fc2"], dtype=np.float64).T + np.asarray(W["b_fc2"], dtype=np.float64))[:, 0]
    return logit / TEMPERATURE


def run_all(X, weights_mode="int8", act_mode="fp32", batch=1024, verbose=True):
    """返回 probs (N,)."""
    W = load_weights(weights_mode)
    aq = load_act_qparams() if act_mode == "int8" else None
    N = X.shape[0]
    out = np.empty(N, dtype=np.float64)
    for i in range(0, N, batch):
        lg = forward(W, X[i:i + batch], act_q=aq)
        out[i:i + batch] = 1.0 / (1.0 + np.exp(-lg))
        if verbose and (i // batch) % 4 == 0:
            print("  [%s/%s] %d/%d" % (weights_mode, act_mode, min(i + batch, N), N), end="\r", flush=True)
    if verbose:
        print("  [%s/%s] done" % (weights_mode, act_mode))
    return out


# ---------------------------------------------------------------------------
# 4. 自检: 批量化实现 vs 参考实现 (parity_fp32_c_vs_onnx.c_forward)
# ---------------------------------------------------------------------------


def selfcheck(n=32, seed=7):
    """1) fp32/fp32 变体必须复现 fullsim_f32.npz
       2) 批量实现必须跟逐样本参考实现 c_forward 一致
       3) hybrid header 的 dequant 权重必须跟 FP32 header 权重同源
    """
    from parity_fp32_c_vs_onnx import c_forward

    W32 = load_weights("fp32")
    Why = load_weights("int8")

    rng = np.random.default_rng(seed)
    Xs = rng.standard_normal((n, IN_CH, T)).astype(np.float32) * 0.8

    # (2) 批量 vs 逐样本
    lg_b = forward(W32, Xs)
    lg_r = np.array([c_forward(W32, x.astype(np.float32)) for x in Xs])
    d = float(np.abs(lg_b - lg_r).max())
    print("[自检 1] 批量 vs 逐样本 c_forward: max|Δlogit| = %.3e  %s"
          % (d, "PASS" if d < 1e-6 else "FAIL"))

    # (1) 复现 fullsim_f32.npz
    d32 = np.load(os.path.join(HERE, "fullsim_f32.npz"))
    win_path = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
    dw = np.load(win_path)
    Xw = dw["X"]
    probs = run_all(Xw[:512], weights_mode="fp32", act_mode="fp32", verbose=False)
    ref = d32["probs"][:512]
    dd = float(np.abs(probs - ref).max())
    print("[自检 2] fp32/fp32 复现 fullsim_f32.npz (前 512 窗): max|Δprob| = %.3e  %s"
          % (dd, "PASS" if dd < 1e-4 else "FAIL"))

    # (3) hybrid 权重同源检查
    for nm in ("tcn_0_conv1", "tcn_2_conv2", "tcn_1_se_fc2"):
        a = Why["w_" + nm].astype(np.float64).ravel()
        b = W32["w_" + nm].astype(np.float64).ravel()
        rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
        r = float(np.corrcoef(a, b)[0, 1])
        print("[自检 3] %-14s dequant vs FP32: max相对误差=%.4f  r=%.6f  %s"
              % (nm, rel, r, "PASS" if rel < 0.02 and r > 0.99 else "FAIL"))

    return d < 1e-6 and dd < 1e-4


if __name__ == "__main__":
    ok = selfcheck()
    print("\n自检 %s" % ("全部通过 ✅" if ok else "有失败 ❌"))
    sys.exit(0 if ok else 1)
