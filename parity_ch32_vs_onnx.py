"""
parity_ch32_vs_onnx.py -- 路线 E (手写 C kernel) PC parity test

目的: 把 h743/Core/Src/ch32_inference.c 的 ch32_init + ch32_forward 1:1 移植到 Python
      (numpy 实现), 然后跟 model_v4_se_23dim_ch32_s42.onnx (FP32 ONNX) 比 logit.

如果 match (FP32 precision): C kernel 是 correct 的, 可以 build+烧录.
如果不 match: C kernel 某个 op (dequant / conv1d_same / dilation / se_block /
              add_residual_relu / classifier) 跟 ONNX 等价实现有 bug,
              需要进一步定位.

5 controlled input 跟 Round 19 controlled self-test 同步:
  ZEROS, ONES, 0.5, NEG1, TWOS  (shape = (23, 16), 全 23 channel 全 16 timestep 同值)
"""
import os
import re
import sys
import numpy as np
import onnxruntime as ort

# GBK console can't encode unicode => force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# 跟 ch32_inference.h 常量一致
IN_CHANNELS = 23
WINDOW = 16
CH = 32
N_BLOCKS = 3
KERNEL = 3
SE_HIDDEN = 4
FC1_OUT = 32
DILATIONS = (1, 2, 4)
TEMPERATURE = 2.0  # CH32_TEMPERATURE


H_PATH = "h743/Core/Inc/model_v4_se_23dim_ch32_hybrid_s42.h"
ONNX_PATH = "model_v4_se_23dim_ch32_s42.onnx"
SEED = "s42"


# ---------------------------------------------------------------------------
# 1. 解析 .h header 提取权重/尺度/偏置 (跟 ch32_init() 用的符号一致)
# ---------------------------------------------------------------------------

def parse_h_arrays(path):
    """抓所有 'static const {int8_t,float} NAME[N] = { v1, v2, ... };'"""
    # .h 是 GBK 编码 (跟 scaler_export.h 同坑,见 project-scaler-export-h-trap), 用 gb18030
    with open(path, "r", encoding="gb18030", errors="replace") as f:
        src = f.read()

    pat = re.compile(
        r"static\s+const\s+(int8_t|float)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}\s*;",
        re.DOTALL,
    )
    arrays = {}
    for m in pat.finditer(src):
        dtype, name, n_str, body = m.groups()
        n = int(n_str)
        # 数字 (支持科学计数法 + 后缀 f / F)
        nums = re.findall(
            r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[fF]?", body
        )
        vals = []
        for s in nums:
            s2 = s.rstrip("fF")
            if "." not in s2 and "e" not in s2.lower():
                vals.append(int(s2))
            else:
                vals.append(float(s2))
        if len(vals) != n:
            print(f"WARN: {name} 声明 N={n} 实际抓到 {len(vals)}", file=sys.stderr)
        if dtype == "int8_t":
            arrays[name] = np.asarray(vals, dtype=np.int8)
        else:
            arrays[name] = np.asarray(vals, dtype=np.float32)
        print(f"  {name:48s} dtype={dtype:8s} N={n:5d}  loaded")
    return arrays


# ---------------------------------------------------------------------------
# 2. Python 1:1 port of ch32_init()
#    weight layout (per ch32_init):
#       w_b0_c1 = dequant(w_tcn_0_conv1, s_tcn_0_conv1)  → (32, 23, 3)
#       w_b0_c2 = dequant(w_tcn_0_conv2, s_tcn_0_conv2)  → (32, 32, 3)
#       w_b0_rs = dequant(w_tcn_0_residual, s_tcn_0_residual) → (32, 23, 1)
#       w_b0_se1 = dequant(w_tcn_0_se_fc1, s_tcn_0_se_fc1)   → (4, 32)
#       w_b0_se2 = dequant(w_tcn_0_se_fc2, s_tcn_0_se_fc2)   → (32, 4)
#       (block 1, 2 同)
#       w_fc1 = fc1_weight (FP32, 32, 32)
#       w_fc2 = fc2_weight (FP32, 32)
# ---------------------------------------------------------------------------

def dequant_per_channel(q: np.ndarray, scale: np.ndarray, out_ch: int, in_ch: int, kernel: int) -> np.ndarray:
    """exact port of C dequant_per_channel:
        for oc in range(out_ch):
            for i in range(in_ch*kernel):
                out[oc][i] = q[oc][i] * scale[oc]
    """
    n_per_ch = in_ch * kernel
    out = np.zeros(out_ch * n_per_ch, dtype=np.float32)
    for oc in range(out_ch):
        s = scale[oc]
        q_row = q[oc * n_per_ch: (oc + 1) * n_per_ch]
        out[oc * n_per_ch: (oc + 1) * n_per_ch] = q_row.astype(np.float32) * s
    return out.reshape(out_ch, in_ch, kernel)


def load_ch32_weights(arr):
    """apply ch32_init() dequant, returns dict of all FP32 weights/biases"""
    W = {}
    # Block 0
    W["b0_c1"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_0_conv1_weight"],
        arr["v4se23_ch32_s42_s_tcn_0_conv1_weight"], CH, IN_CHANNELS, KERNEL)
    W["b0_c2"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_0_conv2_weight"],
        arr["v4se23_ch32_s42_s_tcn_0_conv2_weight"], CH, CH, KERNEL)
    W["b0_rs"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_0_residual_weight"],
        arr["v4se23_ch32_s42_s_tcn_0_residual_weight"], CH, IN_CHANNELS, 1)
    W["b0_se1"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_0_se_fc1_weight"],
        arr["v4se23_ch32_s42_s_tcn_0_se_fc1_weight"], SE_HIDDEN, CH, 1)
    W["b0_se2"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_0_se_fc2_weight"],
        arr["v4se23_ch32_s42_s_tcn_0_se_fc2_weight"], CH, SE_HIDDEN, 1)
    # Block 1
    W["b1_c1"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_1_conv1_weight"],
        arr["v4se23_ch32_s42_s_tcn_1_conv1_weight"], CH, CH, KERNEL)
    W["b1_c2"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_1_conv2_weight"],
        arr["v4se23_ch32_s42_s_tcn_1_conv2_weight"], CH, CH, KERNEL)
    W["b1_se1"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_1_se_fc1_weight"],
        arr["v4se23_ch32_s42_s_tcn_1_se_fc1_weight"], SE_HIDDEN, CH, 1)
    W["b1_se2"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_1_se_fc2_weight"],
        arr["v4se23_ch32_s42_s_tcn_1_se_fc2_weight"], CH, SE_HIDDEN, 1)
    # Block 2
    W["b2_c1"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_2_conv1_weight"],
        arr["v4se23_ch32_s42_s_tcn_2_conv1_weight"], CH, CH, KERNEL)
    W["b2_c2"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_2_conv2_weight"],
        arr["v4se23_ch32_s42_s_tcn_2_conv2_weight"], CH, CH, KERNEL)
    W["b2_se1"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_2_se_fc1_weight"],
        arr["v4se23_ch32_s42_s_tcn_2_se_fc1_weight"], SE_HIDDEN, CH, 1)
    W["b2_se2"] = dequant_per_channel(
        arr["v4se23_ch32_s42_w_tcn_2_se_fc2_weight"],
        arr["v4se23_ch32_s42_s_tcn_2_se_fc2_weight"], CH, SE_HIDDEN, 1)
    # Classifier (FP32, 直接 memcpy)
    W["fc1"] = arr["v4se23_ch32_s42_fc1_weight"].reshape(FC1_OUT, CH)
    W["fc2"] = arr["v4se23_ch32_s42_fc2_weight"].reshape(1, FC1_OUT)
    # Bias
    for prefix in ("b0_c1", "b0_c2", "b0_rs", "b0_se1", "b0_se2",
                   "b1_c1", "b1_c2", "b1_se1", "b1_se2",
                   "b2_c1", "b2_c2", "b2_se1", "b2_se2"):
        # symbol name: v4se23_ch32_s42_b_tcn_X_convY_bias
        #             v4se23_ch32_s42_b_tcn_X_residual_bias
        #             v4se23_ch32_s42_b_tcn_X_se_fcY_bias
        pass
    W["b_b0_c1"]  = arr["v4se23_ch32_s42_b_tcn_0_conv1_bias"]
    W["b_b0_c2"]  = arr["v4se23_ch32_s42_b_tcn_0_conv2_bias"]
    W["b_b0_rs"]  = arr["v4se23_ch32_s42_b_tcn_0_residual_bias"]
    W["b_b0_se1"] = arr["v4se23_ch32_s42_b_tcn_0_se_fc1_bias"]
    W["b_b0_se2"] = arr["v4se23_ch32_s42_b_tcn_0_se_fc2_bias"]
    W["b_b1_c1"]  = arr["v4se23_ch32_s42_b_tcn_1_conv1_bias"]
    W["b_b1_c2"]  = arr["v4se23_ch32_s42_b_tcn_1_conv2_bias"]
    W["b_b1_se1"] = arr["v4se23_ch32_s42_b_tcn_1_se_fc1_bias"]
    W["b_b1_se2"] = arr["v4se23_ch32_s42_b_tcn_1_se_fc2_bias"]
    W["b_b2_c1"]  = arr["v4se23_ch32_s42_b_tcn_2_conv1_bias"]
    W["b_b2_c2"]  = arr["v4se23_ch32_s42_b_tcn_2_conv2_bias"]
    W["b_b2_se1"] = arr["v4se23_ch32_s42_b_tcn_2_se_fc1_bias"]
    W["b_b2_se2"] = arr["v4se23_ch32_s42_b_tcn_2_se_fc2_bias"]
    W["b_fc1"]    = arr["v4se23_ch32_s42_b_fc1_bias"]
    W["b_fc2"]    = arr["v4se23_ch32_s42_b_fc2_bias"]
    return W


# ---------------------------------------------------------------------------
# 3. Python 1:1 port of ch32_forward()
#    layout: x [c, t] = x[c*WINDOW + t]
# ---------------------------------------------------------------------------

def conv1d_same(in_, w, b, dilation, T):
    """in: (in_ch, T)   w: (out_ch, in_ch, kernel)   b: (out_ch,)
       out: (out_ch, T)
    Port of:
       int pad = (kernel - 1) * dilation / 2;
       for oc in out_ch:
         for t_out in T:
           sum = b[oc]
           t_in_base = t_out - pad
           for ic in in_ch:
             w_ic = w[oc, ic, :]
             in_ic = in_[ic, :]
             for k in kernel:
               t_in = t_in_base + k * dilation
               if t_in >= 0 and t_in < T:
                 sum += w_ic[k] * in_ic[t_in]
    """
    in_ch, _ = in_.shape
    out_ch, _, kernel = w.shape
    pad = (kernel - 1) * dilation // 2
    out = np.zeros((out_ch, T), dtype=np.float32)
    for oc in range(out_ch):
        bias_val = float(b[oc])
        for t_out in range(T):
            s = bias_val
            t_in_base = t_out - pad
            for ic in range(in_ch):
                for k in range(kernel):
                    t_in = t_in_base + k * dilation
                    if 0 <= t_in < T:
                        s += w[oc, ic, k] * in_[ic, t_in]
            out[oc, t_out] = s
    return out


def conv1d_1x1(in_, w, b, T):
    """in: (in_ch, T)   w: (out_ch, in_ch, 1)   b: (out_ch,)"""
    in_ch, _ = in_.shape
    out_ch = w.shape[0]
    out = np.zeros((out_ch, T), dtype=np.float32)
    for oc in range(out_ch):
        bias_val = float(b[oc])
        for t in range(T):
            s = bias_val
            for ic in range(in_ch):
                s += w[oc, ic, 0] * in_[ic, t]
            out[oc, t] = s
    return out


def relu_(x):
    return np.maximum(x, 0.0)


def se_block(x, w1, b1, w2, b2):
    """x: (ch, T)   w1: (hidden, ch, 1)   b1: (hidden,)
       w2: (ch, hidden, 1)   b2: (ch,)"""
    ch, T = x.shape
    hidden = w1.shape[0]
    # 1. GAP
    z = x.mean(axis=1)  # (ch,)
    # 2. a = ReLU(z @ W1.T + b1)
    #    W1: (hidden, ch, 1) — collapse kernel: W1[h, c] = W1[h, c, 0]
    W1_2d = w1.reshape(hidden, ch)
    a = (W1_2d @ z + b1)
    a = relu_(a)
    # 3. s = sigmoid(a @ W2.T + b2)
    #    W2: (ch, hidden, 1)
    W2_2d = w2.reshape(ch, hidden)
    s_logits = (W2_2d @ a + b2)
    s = 1.0 / (1.0 + np.exp(-s_logits))
    # 4. scale in-place
    return x * s[:, None]  # (ch, T)


def add_residual_relu(y_, x_, res_):
    return relu_(x_ + res_)


def classifier(x_, w_fc1, b_fc1, w_fc2, b_fc2):
    """x: (ch, T=16)   w_fc1: (32, 32)   b_fc1: (32,)
       w_fc2: (1, 32)  b_fc2: (1, logit)
       返回 (raw_logit_no_T, post_T_logit, prob)
    """
    ch, T = x_.shape
    # GAP
    z = x_.mean(axis=1)  # (ch,)
    # FC1
    h = relu_(w_fc1 @ z + b_fc1)  # (32,)
    # FC2
    raw_logit = float(w_fc2 @ h + b_fc2)  # scalar, 不除 T
    post_T_logit = raw_logit / TEMPERATURE
    prob = 1.0 / (1.0 + np.exp(-post_T_logit))
    return raw_logit, post_T_logit, prob


def ch32_forward_py(x_arr_flat, W):
    """x_arr_flat: (IN_CHANNELS*WINDOW,)  flat row-major  → logit, prob"""
    # 重塑 (23, 16)
    x = x_arr_flat.reshape(IN_CHANNELS, WINDOW).astype(np.float32)
    cur = np.zeros((CH, WINDOW), dtype=np.float32)
    nxt = np.zeros((CH, WINDOW), dtype=np.float32)
    tmp = np.zeros((CH, WINDOW), dtype=np.float32)

    # ----- Block 0 -----
    # residual = 1x1 conv(23->32)
    nxt = conv1d_1x1(x, W["b0_rs"], W["b_b0_rs"], WINDOW)
    # main path
    tmp = conv1d_same(x, W["b0_c1"], W["b_b0_c1"], DILATIONS[0], WINDOW)
    tmp = relu_(tmp)
    cur = conv1d_same(tmp, W["b0_c2"], W["b_b0_c2"], DILATIONS[0], WINDOW)
    cur = relu_(cur)
    cur = se_block(cur, W["b0_se1"], W["b_b0_se1"], W["b0_se2"], W["b_b0_se2"])
    # add residual + ReLU
    cur = add_residual_relu(nxt, cur, nxt)  # cur = ReLU(cur + nxt)
    # NB: 实测 ch32_inference.c line 394:  add_residual_relu(cur, cur, nxt, ...)
    #     即 y = cur (写), x = cur (读), res = nxt.  → y[i] = ReLU(cur[i] + nxt[i])
    #     跟上面一样.

    # ----- Block 1 -----
    nxt = cur.copy()
    tmp = conv1d_same(cur, W["b1_c1"], W["b_b1_c1"], DILATIONS[1], WINDOW)
    tmp = relu_(tmp)
    cur = conv1d_same(tmp, W["b1_c2"], W["b_b1_c2"], DILATIONS[1], WINDOW)
    cur = relu_(cur)
    cur = se_block(cur, W["b1_se1"], W["b_b1_se1"], W["b1_se2"], W["b_b1_se2"])
    # ch32 line 406: add_residual_relu(tmp, cur, nxt, ...)
    # 即 tmp[i] = ReLU(cur[i] + nxt[i])
    tmp = add_residual_relu(tmp, cur, nxt)
    cur = tmp.copy()

    # ----- Block 2 -----
    nxt = cur.copy()
    tmp = conv1d_same(cur, W["b2_c1"], W["b_b2_c1"], DILATIONS[2], WINDOW)
    tmp = relu_(tmp)
    cur = conv1d_same(tmp, W["b2_c2"], W["b_b2_c2"], DILATIONS[2], WINDOW)
    cur = relu_(cur)
    cur = se_block(cur, W["b2_se1"], W["b_b2_se1"], W["b2_se2"], W["b_b2_se2"])
    # ch32 line 419: add_residual_relu(tmp, cur, nxt, ...)
    tmp = add_residual_relu(tmp, cur, nxt)
    cur = tmp.copy()

    # ----- Classifier -----
    logit, logit_T, prob = classifier(cur, W["fc1"], W["b_fc1"], W["fc2"], W["b_fc2"])
    return logit, logit_T, prob


# ---------------------------------------------------------------------------
# 4. ONNX reference (FP32 inference)
# ---------------------------------------------------------------------------

def make_onnx_session():
    """Load FP32 ONNX (no _qat_ suffix). NOTE: ONNX is QAT-aware-exported
       so we use it as FP32 reference. Input shape (1, 16, 23) → logit."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])




def load_onnx_fp32_weights(path=ONNX_PATH):
    """Bypass INT8 dequant: load ONNX FP32 weights directly. Tests forward ops
       only -- rules out quantization noise as the source of any drift."""
    import onnx
    m = onnx.load(path)
    return {t.name: onnx.numpy_helper.to_array(t).astype(np.float32).copy()
            for t in m.graph.initializer}


def build_W_from_onnx(onnx_w):
    """Convert ONNX initializer dict into the W dict format that ch32_forward_py
       consumes. Skips dequant_per_channel entirely."""
    return {
        "b0_c1":  onnx_w["onnx::Conv_125"],
        "b0_c2":  onnx_w["onnx::Conv_128"],
        "b0_rs":  onnx_w["tcn.0.residual.weight"],
        "b0_se1": onnx_w["tcn.0.se.fc1.weight"].reshape(4, 32, 1),
        "b0_se2": onnx_w["tcn.0.se.fc2.weight"].reshape(32, 4, 1),
        "b1_c1":  onnx_w["onnx::Conv_131"],
        "b1_c2":  onnx_w["onnx::Conv_134"],
        "b1_se1": onnx_w["tcn.1.se.fc1.weight"].reshape(4, 32, 1),
        "b1_se2": onnx_w["tcn.1.se.fc2.weight"].reshape(32, 4, 1),
        "b2_c1":  onnx_w["onnx::Conv_137"],
        "b2_c2":  onnx_w["onnx::Conv_140"],
        "b2_se1": onnx_w["tcn.2.se.fc1.weight"].reshape(4, 32, 1),
        "b2_se2": onnx_w["tcn.2.se.fc2.weight"].reshape(32, 4, 1),
        "fc1":    onnx_w["fc1.weight"],
        "fc2":    onnx_w["fc2.weight"],
        "b_b0_c1":  onnx_w["onnx::Conv_126"],
        "b_b0_c2":  onnx_w["onnx::Conv_129"],
        "b_b0_rs":  onnx_w["tcn.0.residual.bias"],
        "b_b0_se1": onnx_w["tcn.0.se.fc1.bias"],
        "b_b0_se2": onnx_w["tcn.0.se.fc2.bias"],
        "b_b1_c1":  onnx_w["onnx::Conv_132"],
        "b_b1_c2":  onnx_w["onnx::Conv_135"],
        "b_b1_se1": onnx_w["tcn.1.se.fc1.bias"],
        "b_b1_se2": onnx_w["tcn.1.se.fc2.bias"],
        "b_b2_c1":  onnx_w["onnx::Conv_138"],
        "b_b2_c2":  onnx_w["onnx::Conv_141"],
        "b_b2_se1": onnx_w["tcn.2.se.fc1.bias"],
        "b_b2_se2": onnx_w["tcn.2.se.fc2.bias"],
        "b_fc1":    onnx_w["fc1.bias"],
        "b_fc2":    onnx_w["fc2.bias"],
    }

def onnx_logit(sess, x_flat):
    """x_flat: (368,) — used as (1, 23, 16) ONNX input (跟 C kernel 同 layout)."""
    x = x_flat.reshape(1, IN_CHANNELS, WINDOW).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    logit_q = out[0]  # may be raw float or quantized
    # ONNX FP32 export should give raw logit directly. Check dtype.
    logit = float(logit_q.flatten()[0])
    prob = 1.0 / (1.0 + np.exp(-logit))
    return logit, prob


# ---------------------------------------------------------------------------
# 5. Main: 5 controlled input, compare
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-onnx-weights", action="store_true",
                        help="Skip INT8 dequant, use ONNX FP32 weights directly. "
                             "Tests forward ops only -- rules out quantization noise.")
    args = parser.parse_args()

    print("=" * 70)
    print("路线 E (手写 C kernel) PC parity test")
    print("=" * 70)
    print(f"[1] Parse .h weights: {H_PATH}")
    if not os.path.exists(H_PATH):
        print(f"ERROR: {H_PATH} 不存在")
        return
    arrs = parse_h_arrays(H_PATH)
    print(f"     loaded {len(arrs)} arrays\n")

    print(f"[2] Apply dequant_per_channel (mirror of ch32_init)")
    if args.use_onnx_weights:
        print(f"[2b] Build W from ONNX FP32 weights (skip dequant)")
        onnx_w = load_onnx_fp32_weights()
        W = build_W_from_onnx(onnx_w)
    else:
        W = load_ch32_weights(arrs)
    print(f"     b0_c1 shape: {W['b0_c1'].shape}  dtype={W['b0_c1'].dtype}")
    print(f"     fc1  shape: {W['fc1'].shape}    dtype={W['fc1'].dtype}\n")

    print(f"[3] Load FP32 ONNX: {ONNX_PATH}")
    if not os.path.exists(ONNX_PATH):
        print(f"ERROR: {ONNX_PATH} 不存在")
        return
    sess = make_onnx_session()
    inp_meta = sess.get_inputs()[0]
    print(f"     input: name={inp_meta.name}  shape={inp_meta.shape}  dtype={inp_meta.type}")
    print()

    # 5 controlled inputs — same as Round 19 controlled self-test
    patterns = [
        ("ZEROS", np.zeros(WINDOW * IN_CHANNELS, dtype=np.float32)),
        ("ONES",  np.ones(WINDOW * IN_CHANNELS, dtype=np.float32)),
        ("0.5",   np.full(WINDOW * IN_CHANNELS, 0.5, dtype=np.float32)),
        ("NEG1",  np.full(WINDOW * IN_CHANNELS, -1.0, dtype=np.float32)),
        ("TWOS",  np.full(WINDOW * IN_CHANNELS, 2.0, dtype=np.float32)),
    ]

    print("=" * 70)
    print(f"{'pattern':10s} {'logit_C':>12s} {'logit_ONNX':>12s} {'|diff|':>10s} "
          f"{'prob_C':>8s} {'prob_ONNX':>10s}  verdict")
    print("-" * 70)

    for name, x in patterns:
        logit_c_raw, logit_c_T, prob_c = ch32_forward_py(x, W)
        logit_o, prob_o = onnx_logit(sess, x)
        # raw C (no T-scaling) vs ONNX (no T-scaling): should match exactly
        diff_raw = abs(logit_c_raw - logit_o)
        diff_rel = diff_raw / (abs(logit_o) + 1e-9)
        if diff_rel < 1e-4:
            verdict = "OK bit-perfect (FP32 noise)"
        elif diff_rel < 1e-2:
            verdict = "ok close (accumul error)"
        else:
            verdict = "XX MISMATCH -- op or dequant wrong"
        print(f"{name:10s} rawC={logit_c_raw:+11.4f} ONNX={logit_o:+11.4f} "
              f"|raw_diff|={diff_raw:8.4f} rel={diff_rel:.2e}  postT_C={logit_c_T:+9.4f}  {verdict}")

    print()
    print("C logit 来自 ch32_forward_py (Python port of ch32_inference.c line 367-424)")
    print("ONNX logit 来自 ONNX Runtime FP32 inference (PC reference).")
    print()
    print("如何读 verdict:")
    print("  ✓ bit-perfect / close → C kernel 数学跟 ONNX 一致, build+烧录就能跑")
    print("  ✗ MISMATCH 且 |diff| 大:")
    print("    - 同一 pattern diff 一致 → 整个 forward 偏一个常量 (e.g., fc1/fc2 weight 装反)")
    print("    - 不同 pattern diff 大小不一 → 某层有 bug (e.g., kernel_size=3 dilation 算错,")
    print("      conv1d_same pad 算错, SE block GAP 漏 T, residual 顺序反, "
    "      temperature scaling logit/=T 写错)")


if __name__ == "__main__":
    main()
