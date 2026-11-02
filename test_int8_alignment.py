"""
test_int8_alignment.py -- INT8 kernel 的 **对齐语义** 验证 (根因 1 的回归测试).

背景:
  2026-10-20 定位的根因 1: MCU 的 dilated conv 用了因果对齐
      t_in = t + (k - pad) * dilation      (错)
  训练模型 (PyTorch nn.Conv1d) 用的是对称 same padding
      t_in = t + k * dilation - pad        (对)
  d=1 时两者巧合相同, d=2/d=4 全错.

INT8 路线换了实现 (CMSIS-NN), **必须重新验证对齐** —— CMSIS 的 padding 语义跟
"对称 same" 是否一致, 不能假设, 必须测.

本测试做两件事:
  A. 直接验证 CMSIS 语义的 padding: 用一个已知形状的输入, 人工推算 conv 结果,
     跟 sim_int8_kernel.cmsis_conv 对比.
  B. 端到端: INT8 每一层 conv 的输出 (反量化后) 跟 FP32 同层输出对比,
     算相关系数. 若对齐错了, d=2/d=4 的相关系数会掉到 0 附近.
  C. 顺便跟"因果对齐"版本对比, 证明这个测试确实能抓到根因 1 (阴性对照).

运行: python test_int8_alignment.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from int8_math import quantize_multiplier_vec
from sim_int8_kernel import Int8Model, cmsis_conv, gap_int8
from quantize_fp32_to_int8 import conv1d_same, load_fp32_model, DIL, CH, IN_CH, T

FAIL = []


def check(name, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAIL.append(name)


def test_cmsis_padding_semantics():
    """A. CMSIS-NN 语义镜像的 padding: 人工构造, 期望 t_in = t + k*d - pad."""
    print("\nA. CMSIS padding 语义 (对称 same) 人工验证")
    rng = np.random.RandomState(0)
    for dil in (1, 2, 4):
        K = 3
        Cin, Cout, Tx = 2, 3, 16
        pad = (K - 1) * dil // 2
        q = rng.randint(-100, 100, size=(1, Tx, Cin)).astype(np.int8)
        w = rng.randint(-100, 100, size=(Cout, K, Cin)).astype(np.int8)
        bq = np.zeros(Cout, dtype=np.int64)
        mult, shift = quantize_multiplier_vec(np.full(Cout, 1.0 / 255.0))
        out = cmsis_conv(q, w, bq, mult, shift, 0, 0, -128, 127, K, dil)
        # 人工: acc = sum_k sum_ic (q[t + k*d - pad] - 0) * w  (越界算 0)
        ref = np.zeros((1, Tx, Cout), dtype=np.int64)
        qc = q.astype(np.int64)
        for t in range(Tx):
            for k in range(K):
                t_in = t + k * dil - pad
                if 0 <= t_in < Tx:
                    for oc in range(Cout):
                        ref[0, t, oc] += int((qc[0, t_in] * w[oc, k]).sum())
        # requantize 之后再比 -> 这里只验证累加部分: 用 mult 使其近似恒等
        acc_ref = ref
        # cmsis_conv 已经做了 requant, 这里反推: 用同样的 requant 处理 ref
        from int8_math import requantize_np
        out_ref = np.clip(requantize_np(acc_ref, mult, shift) + 0, -128, 127)
        ok = np.array_equal(out.astype(np.int64), out_ref.astype(np.int64))
        check("dilation=%d 的 conv 累加对齐" % dil, ok,
              "" if ok else "max_diff=%d" % np.abs(out.astype(int) - out_ref.astype(int)).max())


def test_causal_negative_control():
    """C. 阴性对照: 故意用因果对齐, 证明本测试能抓到根因 1."""
    print("\nC. 阴性对照 (因果对齐 t+(k-pad)*d 必须被测出来)")
    rng = np.random.RandomState(1)
    K, dil, Cin, Cout, Tx = 3, 4, 2, 3, 16
    pad = (K - 1) * dil // 2
    q = rng.randint(-100, 100, size=(1, Tx, Cin)).astype(np.int8)
    w = rng.randint(-100, 100, size=(Cout, K, Cin)).astype(np.int8)
    bq = np.zeros(Cout, dtype=np.int64)
    mult, shift = quantize_multiplier_vec(np.full(Cout, 1.0 / 255.0))
    out = cmsis_conv(q, w, bq, mult, shift, 0, 0, -128, 127, K, dil)
    from int8_math import requantize_np
    causal = np.zeros((1, Tx, Cout), dtype=np.int64)
    qc = q.astype(np.int64)
    for t in range(Tx):
        for k in range(K):
            t_in = t + (k - pad) * dil
            if 0 <= t_in < Tx:
                for oc in range(Cout):
                    causal[0, t, oc] += int((qc[0, t_in] * w[oc, k]).sum())
    causal = np.clip(requantize_np(causal, mult, shift), -128, 127)
    differ = (out.astype(np.int64) != causal.astype(np.int64)).any()
    check("对称 same != 因果 (d=4)", bool(differ))


def test_layerwise_agreement():
    """B. 端到端逐层对比: INT8 反量化 vs FP32 同层."""
    print("\nB. 逐层对比 (INT8 反量化 vs FP32) -- 相关系数")
    os.environ["H743_CSV"] = "IanArffDatasetF32.csv"
    from h743_windows import build_windows
    from quantize_fp32_to_int8 import fp32_trace

    W32 = load_fp32_model()
    m = Int8Model()
    wins, labels = build_windows(start=0, n=4000, phase=0)
    X = np.stack([w[2] for w in wins[:64]], axis=0).astype(np.float32)

    tr = fp32_trace(W32, X)
    _, dbg = m.forward(X, ret_debug=True)
    h = m.h

    def nhwc(t):
        """FP32 trace 是 (N, C, T), INT8 镜像是 (N, T, C) -> 统一转 NHWC."""
        return np.asarray(t).transpose(0, 2, 1)

    def corr(a, b):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # c1/c2 各 block: INT8 镜像的中间量 (需要单独跑一遍拿到)
    xq = m.quant_input(X)
    c1_0 = cmsis_conv(xq, m.W["tcn_0_conv1"], h["b_tcn_0_conv1"], h["mult_tcn_0_conv1"],
                       h["shift_tcn_0_conv1"], h["zp_x"], h["zp_c1_0"], h["zp_c1_0"], 127, 3, DIL[0])
    dq0 = (c1_0.astype(np.float64) - h["zp_c1_0"]) * h["s_c1_0"]
    r = corr(dq0, nhwc(tr["c1_0"]))
    check("block0 conv1 (d=1) 相关", r > 0.99, "r=%.5f" % r)

    c2_0 = cmsis_conv(c1_0, m.W["tcn_0_conv2"], h["b_tcn_0_conv2"], h["mult_tcn_0_conv2"],
                       h["shift_tcn_0_conv2"], h["zp_c1_0"], h["zp_c2_0"], h["zp_c2_0"], 127, 3, DIL[0])
    dq0 = (c2_0.astype(np.float64) - h["zp_c2_0"]) * h["s_c2_0"]
    r = corr(dq0, nhwc(tr["c2_0"]))
    check("block0 conv2 (d=1) 相关", r > 0.99, "r=%.5f" % r)

    # block1: 输入是 add_0 的 FP32 (用 FP32 量化成 int8 再跑, 隔离 conv 本身的错)
    a0q = m.quant_with(tr["add_0"], h["s_add_0"], h["zp_add_0"])
    c1_1 = cmsis_conv(a0q, m.W["tcn_1_conv1"], h["b_tcn_1_conv1"], h["mult_tcn_1_conv1"],
                       h["shift_tcn_1_conv1"], h["zp_add_0"], h["zp_c1_1"], h["zp_c1_1"], 127, 3, DIL[1])
    dq0 = (c1_1.astype(np.float64) - h["zp_c1_1"]) * h["s_c1_1"]
    r = corr(dq0, nhwc(tr["c1_1"]))
    check("block1 conv1 (d=2) 相关", r > 0.99, "r=%.5f" % r)

    c2_1 = cmsis_conv(c1_1, m.W["tcn_1_conv2"], h["b_tcn_1_conv2"], h["mult_tcn_1_conv2"],
                       h["shift_tcn_1_conv2"], h["zp_c1_1"], h["zp_c2_1"], h["zp_c2_1"], 127, 3, DIL[1])
    dq0 = (c2_1.astype(np.float64) - h["zp_c2_1"]) * h["s_c2_1"]
    r = corr(dq0, nhwc(tr["c2_1"]))
    check("block1 conv2 (d=2) 相关", r > 0.99, "r=%.5f" % r)

    a1q = m.quant_with(tr["add_1"], h["s_add_1"], h["zp_add_1"])
    c1_2 = cmsis_conv(a1q, m.W["tcn_2_conv1"], h["b_tcn_2_conv1"], h["mult_tcn_2_conv1"],
                       h["shift_tcn_2_conv1"], h["zp_add_1"], h["zp_c1_2"], h["zp_c1_2"], 127, 3, DIL[2])
    dq0 = (c1_2.astype(np.float64) - h["zp_c1_2"]) * h["s_c1_2"]
    r = corr(dq0, nhwc(tr["c1_2"]))
    check("block2 conv1 (d=4) 相关", r > 0.99, "r=%.5f" % r)

    c2_2 = cmsis_conv(c1_2, m.W["tcn_2_conv2"], h["b_tcn_2_conv2"], h["mult_tcn_2_conv2"],
                       h["shift_tcn_2_conv2"], h["zp_c1_2"], h["zp_c2_2"], h["zp_c2_2"], 127, 3, DIL[2])
    dq0 = (c2_2.astype(np.float64) - h["zp_c2_2"]) * h["s_c2_2"]
    r = corr(dq0, nhwc(tr["c2_2"]))
    check("block2 conv2 (d=4) 相关", r > 0.99, "r=%.5f" % r)


def main():
    print("=" * 78)
    print("INT8 对齐语义验证 (根因 1 回归测试)")
    print("=" * 78)
    test_cmsis_padding_semantics()
    test_causal_negative_control()
    test_layerwise_agreement()
    print("\n" + "=" * 78)
    if FAIL:
        print("结果: %d 项 FAIL -> %s" % (len(FAIL), FAIL))
        sys.exit(1)
    print("结果: 全部 PASS")


if __name__ == "__main__":
    main()
