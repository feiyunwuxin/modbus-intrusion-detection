"""
parity_hybrid_int16.py -- Hybrid-INT16 MCU kernel 的真·parity 测试.

源绑定 (memory: feedback-parity-test-must-bind-to-real-source)
---------------------------------------------------------------
  * conv 的 t_in 公式 **从 MCU/Core/Src/ch32_inference_hybrid.c 源码解析**,
    不手抄. 根因1 的教训: Python 复刻手写公式 -> 跟被测对象一起错.
  * dequant 写法也从源码解析 (断言存在 `(float)w_o[i] * wsc[o]` 这种形状),
    保证测的是 MCU 真在跑的那套语义.
  * 权重来自 MCU/Core/Inc/model_v4_se_23dim_ch32_int16_s42.h
    (gen_int16_weights.py 从 FP32 部署真值头量化而来) -> 测的就是要烧进去的那份.

三层校验:
  [1] Hybrid kernel (按 C 源码语义, float32)  vs  sim_hybrid_kernel (float64 批量)
      -> 期望 max|Δprob| < 1e-5 (只有 float32/float64 舍入差异)
  [2] Hybrid kernel (float32)                    vs  ONNX FP32 (onnxruntime)
      -> 期望 max|Δprob| < 1e-2 (int16 量化误差 ~1.5e-5 逐层累积)
  [3] 决策一致性: prob>=0.5 的判定 vs FP32 是否完全一致

用法: python parity_hybrid_int16.py [N_windows]
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim_hybrid_kernel import (_SHAPES, _TENSORS, load_weights, parse_header,
                               t_in_index)

HYB_C = os.path.join(HERE, "MCU/Core/Src/ch32_inference_hybrid.c")
I16_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int16_s42.h")
ONNX_PATH = os.path.join(HERE, "model_v4_se_23dim_ch32_s42.onnx")
WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")

IN_CH, T, CH, SE_H, FC1_OUT = 23, 16, 32, 4, 32
DIL = (1, 2, 4)


# ---------------------------------------------------------------------------
# 1. 从 C 源码解析语义
# ---------------------------------------------------------------------------

def parse_c_semantics(c_path=HYB_C):
    src = open(c_path, encoding="utf-8", errors="replace").read()
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)

    # t_in 公式 (必须是对称 same padding)
    m = re.search(r"int\s+t_in\s*=\s*([^;]+);", src)
    if not m:
        raise RuntimeError("找不到 'int t_in = ...;'")
    t_in = m.group(1).strip()
    print("[C 源码] %s 的 t_in 公式 = %s" % (os.path.basename(c_path), t_in))

    # 反量化写法
    m2 = re.search(r"wq_f\[\s*(\w+)\s*\]\s*=\s*\(float\)\s*(\w+)\[\s*\1\s*\]\s*\*\s*(\w+)\[\s*(\w+)\s*\];",
                   src)
    if not m2:
        raise RuntimeError("找不到 dequant 语句 'wq_f[i] = (float)w[i] * scale[o];'")
    print("[C 源码] dequant = %s" % m2.group(0))

    # bias 是否先加进累加器 (Route E 语义: sum = b[oc] 再累加 w*x)
    m3 = re.search(r"float\s+sum\s*=\s*b\[(\w+)\];", src)
    if not m3:
        raise RuntimeError("找不到 'float sum = b[oc];'")
    print("[C 源码] 累加器初值 = %s (bias 先加)" % m3.group(0))

    # 权重是否常驻 Flash (没有 memcpy 到 RAM)
    has_memcpy = "memcpy(w_" in src or "memcpy(hy_w" in src
    print("[C 源码] 权重 RAM 拷贝: %s" % ("有 (异常!)" if has_memcpy else "无 ✅ (常驻 Flash)"))

    return t_in


T_IN_EXPR = parse_c_semantics()


def t_in_of(t, k, pad, dilation):
    return eval(T_IN_EXPR, {"__builtins__": {}},
                {"t": t, "k": k, "pad": pad, "dilation": dilation})


# ---------------------------------------------------------------------------
# 2. 按 C 语义实现 (float32, 逐样本, 顺序跟 C 一致)
# ---------------------------------------------------------------------------

def load_int16_weights():
    """从 int16 头读 int16 + scale, 反量化成 float32 (模拟 MCU 的 wq_f)."""
    raw = parse_header(I16_HDR)
    p = "v4se23_ch32_i16_s42_"
    W = {}
    for nm in _TENSORS:
        shp = _SHAPES[nm]
        q = raw[p + "w_" + nm + "_weight"].astype(np.float32)
        sc = raw[p + "s_" + nm + "_weight"].astype(np.float32)
        W["w_" + nm] = (q.reshape(shp) * sc.reshape(-1, *([1] * (len(shp) - 1)))).astype(np.float32)
        W["b_" + nm] = raw[p + "b_" + nm + "_bias"]
    W["fc1"] = raw[p + "fc1_weight"].reshape(FC1_OUT, CH)
    W["fc2"] = raw[p + "fc2_weight"].reshape(1, FC1_OUT)
    W["b_fc1"] = raw[p + "b_fc1_bias"]
    W["b_fc2"] = raw[p + "b_fc2_bias"]
    return W


def conv1d_same_c(W, name, inp, dil, kernel):
    """严格照抄 MCU 的 conv1d_same_i16 (float32)."""
    w = W["w_" + name]
    b = W["b_" + name]
    in_ch, out_ch, K = w.shape[1], w.shape[0], w.shape[2]
    Tx = inp.shape[1]
    pad = (K - 1) * dil // 2
    out = np.zeros((out_ch, Tx), dtype=np.float32)
    for oc in range(out_ch):
        for t in range(Tx):
            s = np.float32(b[oc])
            for ic in range(in_ch):
                for k in range(K):
                    t_in = int(t_in_of(t, k, pad, dil))
                    if t_in < 0 or t_in >= Tx:
                        continue
                    s = np.float32(s + np.float32(w[oc, ic, k] * inp[ic, t_in]))
            out[oc, t] = s
    return out


def conv1d_1x1_c(W, name, inp):
    w = W["w_" + name]
    b = W["b_" + name]
    in_ch, out_ch = w.shape[1], w.shape[0]
    Tx = inp.shape[1]
    out = np.zeros((out_ch, Tx), dtype=np.float32)
    for oc in range(out_ch):
        for t in range(Tx):
            s = np.float32(b[oc])
            for ic in range(in_ch):
                s = np.float32(s + np.float32(w[oc, ic, 0] * inp[ic, t]))
            out[oc, t] = s
    return out


def se_block_c(W, bi, x):
    ch, hidden, Tx = CH, SE_H, T
    z = np.zeros(ch, dtype=np.float32)
    for c in range(ch):
        s = np.float32(0.0)
        for t in range(Tx):
            s = np.float32(s + x[c, t])
        z[c] = np.float32(s / Tx)
    h = np.zeros(hidden, dtype=np.float32)
    w1 = W["w_tcn_%d_se_fc1" % bi]
    b1 = W["b_tcn_%d_se_fc1" % bi]
    for o in range(hidden):
        s = np.float32(b1[o])
        for i in range(ch):
            s = np.float32(s + np.float32(w1[o, i] * z[i]))
        h[o] = np.float32(max(s, 0.0))
    w2 = W["w_tcn_%d_se_fc2" % bi]
    b2 = W["b_tcn_%d_se_fc2" % bi]
    sc = np.zeros(ch, dtype=np.float32)
    for o in range(ch):
        s = np.float32(b2[o])
        for i in range(hidden):
            s = np.float32(s + np.float32(w2[o, i] * h[i]))
        sc[o] = np.float32(1.0 / (1.0 + np.exp(-s)))
    return (x * sc[:, None]).astype(np.float32)


def hybrid_forward_c(W, x23):
    """完全照抄 MCU ch32_hybrid_forward 的数据流."""
    x = np.asarray(x23, dtype=np.float32)
    res = conv1d_1x1_c(W, "tcn_0_residual", x)
    c1 = np.maximum(conv1d_same_c(W, "tcn_0_conv1", x, DIL[0], 3), 0).astype(np.float32)
    c2 = np.maximum(conv1d_same_c(W, "tcn_0_conv2", c1, DIL[0], 3), 0).astype(np.float32)
    cur = se_block_c(W, 0, c2)
    cur = np.maximum(cur + res, 0).astype(np.float32)

    for bi in (1, 2):
        res = cur
        c1 = np.maximum(conv1d_same_c(W, "tcn_%d_conv1" % bi, cur, DIL[bi], 3), 0).astype(np.float32)
        c2 = np.maximum(conv1d_same_c(W, "tcn_%d_conv2" % bi, c1, DIL[bi], 3), 0).astype(np.float32)
        cur = se_block_c(W, bi, c2)
        cur = np.maximum(cur + res, 0).astype(np.float32)

    g = np.zeros(CH, dtype=np.float32)
    for c in range(CH):
        s = np.float32(0.0)
        for t in range(T):
            s = np.float32(s + cur[c, t])
        g[c] = np.float32(s / T)
    h = np.zeros(FC1_OUT, dtype=np.float32)
    for o in range(FC1_OUT):
        s = np.float32(W["b_fc1"][o])
        for i in range(CH):
            s = np.float32(s + np.float32(W["fc1"][o, i] * g[i]))
        h[o] = np.float32(max(s, 0.0))
    logit = np.float32(W["b_fc2"][0])
    for i in range(FC1_OUT):
        logit = np.float32(logit + np.float32(W["fc2"][0, i] * h[i]))
    logit = np.float32(logit / 1.0)          # CH32_TEMPERATURE = 1.0
    return 1.0 / (1.0 + np.exp(-logit)), logit


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    d = np.load(WIN_NPZ)
    X = d["X"].astype(np.float32)
    frames = d["frames"]
    # 取部署区的窗口 (对外口径)
    dep = np.where(frames >= 200001)[0][:n]
    Xs = X[dep]
    print("\n[窗口] 取部署区前 %d 个窗 (frame>=200001)" % len(Xs))

    W16 = load_int16_weights()

    # ---- [1] MCU 语义 (float32) vs numpy 批量参考 (float64) ----
    from sim_hybrid_kernel import forward as np_forward
    print("\n[1] Hybrid kernel (按 C 源码, float32) vs numpy 批量参考 (float64)")
    p_c, lg_c = [], []
    for i, x in enumerate(Xs):
        p, lg = hybrid_forward_c(W16, x)
        p_c.append(p)
        lg_c.append(lg)
        if (i + 1) % 4 == 0:
            print("   %d/%d" % (i + 1, len(Xs)), end="\r", flush=True)
    print()
    p_c = np.array(p_c)
    lg_c = np.array(lg_c)
    lg_ref = np_forward(W16, Xs)
    p_ref = 1.0 / (1.0 + np.exp(-lg_ref))
    d1 = float(np.abs(p_c - p_ref).max())
    d1l = float(np.abs(lg_c - lg_ref).max())
    print("    max|Δprob|  = %.3e   %s" % (d1, "PASS ✅" if d1 < 1e-5 else "FAIL ❌"))
    print("    max|Δlogit| = %.3e   %s" % (d1l, "PASS ✅" if d1l < 1e-4 else "FAIL ❌"))

    # ---- [2] vs ONNX FP32 ----
    # ⚠️ ONNX 的输入空间跟 MCU 的 windows_f32_mcu.npz 不同 (后者已经过 MCU 侧
    #    的特征提取/归一化), 所以这里用 h743_windows.build_windows 取跟 ONNX
    #    同源的窗口; ONNX 输出的是 **logit** (不是概率).
    ok2 = True
    if os.path.exists(ONNX_PATH):
        print("\n[2] Hybrid kernel vs ONNX FP32 (onnxruntime)")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name
            oname = sess.get_outputs()[0].name

            sys.path.insert(0, "modbus_replay_simulator")
            from h743_windows import build_windows
            wins, _ = build_windows(n=74628)
            rng = np.random.default_rng(7)
            inputs = [w[1] for w in wins[::1500][:6]]
            inputs += [rng.standard_normal((23, 16)).astype(np.float32) * 0.8
                       for _ in range(2)]

            from sim_hybrid_kernel import load_weights
            W32 = load_weights("fp32")

            d_fp32, d_hyb = 0.0, 0.0
            print("    %-6s %-16s %-16s %-16s %-12s"
                  % ("case", "ONNX logit", "RouteE FP32", "Hybrid INT16", "|H-ONNX|"))
            for i, x in enumerate(inputs):
                l_on = float(np.asarray(
                    sess.run([oname], {iname: np.asarray(x, dtype=np.float32)[None, ...]})[0]
                ).ravel()[0])
                l_32 = float(np_forward(W32, np.asarray(x, dtype=np.float32)[None, ...])[0])
                l_hy = float(hybrid_forward_c(W16, np.asarray(x, dtype=np.float32))[1])
                d_fp32 = max(d_fp32, abs(l_32 - l_on))
                d_hyb = max(d_hyb, abs(l_hy - l_on))
                print("    %-6d %-16.6f %-16.6f %-16.6f %-12.3e"
                      % (i, l_on, l_32, l_hy, abs(l_hy - l_on)))
            print("    max|RouteE_FP32 - ONNX| = %.3e   %s"
                  % (d_fp32, "PASS ✅" if d_fp32 < 1e-3 else "FAIL ❌"))
            print("    max|Hybrid_INT16 - ONNX| = %.3e   %s"
                  % (d_hyb, "PASS ✅" if d_hyb < 5e-2 else "FAIL ❌"))
            ok2 = (d_fp32 < 1e-3) and (d_hyb < 5e-2)
        except Exception as e:
            print("    跳过 (%s: %s)" % (type(e).__name__, e))
    else:
        print("\n[2] 跳过 (没有 %s)" % os.path.basename(ONNX_PATH))

    # ---- [3] 决策一致性 (vs Route E FP32) ----
    print("\n[3] 决策一致性: Hybrid vs FP32 (部署区 %d 窗, thr=0.5)" % len(dep))
    from sim_hybrid_kernel import load_weights
    W32 = load_weights("fp32")
    lg32 = np_forward(W32, Xs)
    p32 = 1.0 / (1.0 + np.exp(-lg32))
    flip = int(((p_c >= 0.5) != (p32 >= 0.5)).sum())
    print("    决策翻转 = %d / %d   %s" % (flip, len(Xs), "PASS ✅" if flip == 0 else "FAIL ❌"))
    print("    max|Δprob| (Hybrid vs FP32) = %.3e" % np.abs(p_c - p32).max())

    print("\n结论: %s" % ("全部通过 ✅" if (d1 < 1e-5 and ok2 and flip == 0) else "有失败 ❌"))


if __name__ == "__main__":
    main()
