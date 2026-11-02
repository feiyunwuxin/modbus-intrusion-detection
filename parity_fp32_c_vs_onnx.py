"""
parity_fp32_c_vs_onnx.py -- 部署版 (FP32 Route E) 的真·parity 测试.

与旧 parity_ch32_vs_onnx.py 的关键区别:
  ⭐ 旧脚本的 conv1d_same 是脚本自己手写的 Python 复刻, **从来没跑过真正的 C 代码**,
    所以 C 源码里的 dilated-conv 对齐 bug (t + (k-pad)*d  vs  t + k*d - pad) 一直漏网.
  ⭐ 本脚本直接从 MCU/Core/Src/ch32_inference.c 源码里 **解析出 t_in 公式** 再求值,
    C 源码一改, 测试立刻跟着变 —— 不会再出现"测试通过但固件跑错函数".

对比基准: model_v4_se_23dim_ch32_s42.onnx (onnxruntime) == 训练模型 (PyTorch+BN).
"""
import math
import re
import sys

import numpy as np
import torch

C_SRC = "MCU/Core/Src/ch32_inference.c"
H_PATH = "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h"
ONNX_PATH = "model_v4_se_23dim_ch32_s42.onnx"
PT_PATH = "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"
IN_CHANNELS, WINDOW, CH, SE_HIDDEN, FC1_OUT = 23, 16, 32, 4, 32
DILATIONS = (1, 2, 4)
TEMPERATURE = 1.0


def load_t_in_formula():
    """从 C 源码解析 conv1d_same 里的 t_in 表达式 (先剥掉注释, 只认真实代码)."""
    src = open(C_SRC, encoding="utf-8", errors="replace").read()
    # 剥掉块注释和行注释 —— 注释里常出现旧公式的说明文字, 不能当真代码
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    m = re.search(r"int\s+t_in\s*=\s*([^;]+);", src)
    if not m:
        raise RuntimeError("C 源码里找不到 'int t_in = ...;'")
    expr = m.group(1).strip()
    expr = re.sub(r"/\*.*?\*/", " ", expr, flags=re.S).strip()
    print("[C 源码] conv1d_same 的 t_in 公式 = %s" % expr)
    return expr


T_IN_EXPR = load_t_in_formula()


def t_in_of(t, k, pad, dilation):
    return eval(T_IN_EXPR, {"__builtins__": {}},
                {"t": t, "k": k, "pad": pad, "dilation": dilation})


def parse_h(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    A = {}
    for m in re.finditer(r"static const float\s+(\w+)\[(\d+)\]\s*=\s*\{(.*?)\};", txt, re.S):
        vals = [float(x[:-1]) for x in re.findall(r"[-+0-9.eE]+f", m.group(3))]
        assert len(vals) == int(m.group(2)), m.group(1)
        A[m.group(1)] = np.array(vals, dtype=np.float32)
    return A


def conv1d_same(inp, w, b, dil):
    """逐行复刻 C conv1d_same (t_in 公式来自 C 源码)."""
    O, C, K = w.shape
    T = inp.shape[1]
    pad = (K - 1) * dil // 2
    out = np.repeat(np.asarray(b, dtype=np.float64)[:, None], T, axis=1)
    for k in range(K):
        idx = np.array([t_in_of(t, k, pad, dil) for t in range(T)])
        valid = (idx >= 0) & (idx < T)
        ic = np.clip(idx, 0, T - 1)
        out += np.einsum('oc,ct->ot', np.asarray(w[:, :, k], dtype=np.float64),
                         np.asarray(inp[:, ic], dtype=np.float64) * valid[None, :])
    return out


def se_block(x, W, bi):
    T = x.shape[1]
    z = x.sum(axis=1) / T
    w1 = W["w_tcn_%d_se_fc1" % bi].reshape(SE_HIDDEN, CH)
    b1 = W["b_tcn_%d_se_fc1" % bi]
    w2 = W["w_tcn_%d_se_fc2" % bi].reshape(CH, SE_HIDDEN)
    b2 = W["b_tcn_%d_se_fc2" % bi]
    h = np.maximum(w1 @ z + b1, 0)
    s = 1.0 / (1.0 + np.exp(-(w2 @ h + b2)))
    return x * s[:, None]


def c_forward(W, x23):
    """复刻 MCU ch32_forward (顺序: conv1->relu->conv2->relu->SE->+res->relu)."""
    x = np.asarray(x23, dtype=np.float32)
    res = conv1d_same(x, W["w_tcn_0_residual"].reshape(CH, IN_CHANNELS, 1),
                      W["b_tcn_0_residual"], 1)
    tmp = np.maximum(conv1d_same(x, W["w_tcn_0_conv1"].reshape(CH, IN_CHANNELS, 3),
                                  W["b_tcn_0_conv1"], DILATIONS[0]), 0)
    cur = np.maximum(conv1d_same(tmp, W["w_tcn_0_conv2"].reshape(CH, CH, 3),
                                  W["b_tcn_0_conv2"], DILATIONS[0]), 0)
    cur = se_block(cur, W, 0)
    cur = np.maximum(cur + res, 0)
    for bi in (1, 2):
        dil = DILATIONS[bi]
        res = cur.copy()
        tmp = np.maximum(conv1d_same(cur, W["w_tcn_%d_conv1" % bi].reshape(CH, CH, 3),
                                      W["b_tcn_%d_conv1" % bi], dil), 0)
        cur = np.maximum(conv1d_same(tmp, W["w_tcn_%d_conv2" % bi].reshape(CH, CH, 3),
                                      W["b_tcn_%d_conv2" % bi], dil), 0)
        cur = se_block(cur, W, bi)
        cur = np.maximum(cur + res, 0)
    g = cur.sum(axis=1) / WINDOW
    h = np.maximum(W["fc1"].reshape(FC1_OUT, CH) @ g + W["b_fc1"], 0)
    logit = float((W["fc2"].reshape(1, FC1_OUT) @ h + W["b_fc2"])[0]) / TEMPERATURE
    return logit


def main():
    raw = parse_h(H_PATH)
    W = {
        "w_tcn_0_conv1": raw["v4se23_ch32_s42_w_tcn_0_conv1_weight"],
        "b_tcn_0_conv1": raw["v4se23_ch32_s42_b_tcn_0_conv1_bias"],
        "w_tcn_0_conv2": raw["v4se23_ch32_s42_w_tcn_0_conv2_weight"],
        "b_tcn_0_conv2": raw["v4se23_ch32_s42_b_tcn_0_conv2_bias"],
        "w_tcn_0_residual": raw["v4se23_ch32_s42_w_tcn_0_residual_weight"],
        "b_tcn_0_residual": raw["v4se23_ch32_s42_b_tcn_0_residual_bias"],
        "fc1": raw["v4se23_ch32_s42_fc1_weight"],
        "b_fc1": raw["v4se23_ch32_s42_b_fc1_bias"],
        "fc2": raw["v4se23_ch32_s42_fc2_weight"],
        "b_fc2": raw["v4se23_ch32_s42_b_fc2_bias"],
    }
    for bi in (0, 1, 2):
        W["w_tcn_%d_conv1" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_conv1_weight" % bi]
        W["b_tcn_%d_conv1" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_conv1_bias" % bi]
        W["w_tcn_%d_conv2" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_conv2_weight" % bi]
        W["b_tcn_%d_conv2" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_conv2_bias" % bi]
        W["w_tcn_%d_se_fc1" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_se_fc1_weight" % bi]
        W["b_tcn_%d_se_fc1" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_se_fc1_bias" % bi]
        W["w_tcn_%d_se_fc2" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_se_fc2_weight" % bi]
        W["b_tcn_%d_se_fc2" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_se_fc2_bias" % bi]

    sys.path.insert(0, "modbus_replay_simulator")
    from modbus_replay.ids_worker_torch import build_tcn_se_model
    ck = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    model = build_tcn_se_model(
        in_ch=ck["n_features"], n_blocks=ck["n_blocks"], channels=ck["channels"],
        kernel_size=ck["kernel_size"], dilations=tuple(ck["dilations"]),
        dropout=ck["dropout"], se_reduction=ck["se_reduction"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        print("onnxruntime 不可用: %s" % e)
        sess = None

    from h743_windows import build_windows
    wins, _ = build_windows(n=74628)
    rng = np.random.default_rng(7)
    inputs = [w[1] for w in wins[::1500][:8]]
    inputs += [rng.standard_normal((23, 16)).astype(np.float32) * 0.8 for _ in range(4)]

    print()
    print("%-8s %-16s %-16s %-16s %-12s" % ("case", "C(源码公式)", "ONNX", "PyTorch+BN", "|C-ONNX|"))
    worst = 0.0
    for i, x in enumerate(inputs):
        l_c = c_forward(W, x)
        l_on = float("nan")
        if sess is not None:
            out = sess.run(None, {sess.get_inputs()[0].name: x[None, ...]})[0]
            l_on = float(np.array(out).ravel()[0])
        with torch.no_grad():
            l_pt = float(model(torch.from_numpy(x).float().unsqueeze(0)).item())
        d = abs(l_c - l_on) if sess is not None else float("nan")
        worst = max(worst, d if d == d else 0.0)
        print("%-8d %-16.6f %-16.6f %-16.6f %-12.3e" % (i, l_c, l_on, l_pt, d))

    print()
    print("  max |C_kernel - ONNX| = %.3e" % worst)
    ok = worst < 1e-3
    print("  PARITY %s  (阈值 1e-3)" % ("PASS ✅" if ok else "FAIL ❌"))
    if not ok:
        print("  ⭐ 提示: 检查 C 源码 conv1d_same 的 t_in 公式是否等于 't + k * dilation - pad'")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
