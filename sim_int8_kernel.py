"""
sim_int8_kernel.py -- MCU INT8 kernel 的 numpy 镜像 (bit-exact 目标).

用途:
  1. 跟 FP32 (fullsim_f32.npz) 逐窗对比 -> 量化掉点分析 (compare_fp32_vs_int8.py)
  2. 当作 MCU 侧 C 实现的 **参考实现**: 两边必须逐位一致 (L1 源绑定 + L6 上机验证)

镜像的定点语义严格照抄:
  - conv: CMSIS-NN arm_convolve_s8 (int32 精确累加, 无中途舍入)
      filter 布局 [C_OUT][HK][WK][C_IN], 激活布局 NHWC [T][C]
      acc = sum_ic,k (q_in + input_offset) * q_w + bias_q[oc]
      input_offset = -zp_in
      out = requantize(acc, mult[oc], shift[oc]) + zp_out, 再 clamp 到 [act_min, act_max]
  - FC / SE scale-back / identity-residual requant: 手写 int32 (跟 MCU 的 ch32_inference_int8.c 一致)
      acc = sum_i (q_in + in_off) * q_w + bias_q[oc]
  - requantize: arm_nn_requantize (CMSIS-NN double-rounding 分支) -> int8_math.requantize_np
  - GAP: sum -> +-T/2 -> 截断除法 (跟 C 的整数除法一致)
  - sigmoid: FP32 (sigmoid 本身无法定点化)

本文件只读 MCU 的 INT8 header (交付物本身), 不读 JSON -> 保证测的就是要烧进 MCU 的那份权重.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from int8_math import parse_int8_header, requantize_np

INT8_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h")

IN_CH, T, CH, SE_H, FC1_OUT = 23, 16, 32, 4, 32
DIL = (1, 2, 4)
TEMPERATURE = 1.0


# ---------------------------------------------------------------------------
# 定点原语
# ---------------------------------------------------------------------------


def q_round(v):
    """round half away from zero (C: (int32_t)(v>=0 ? v+0.5f : v-0.5f) 的等价)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)


def cmsis_conv(q_in, w, bq, mult, shift, in_zp, out_zp, act_min, act_max, K, dil):
    """CMSIS-NN arm_convolve_s8 的语义镜像.

    q_in : (N, T, C_in) int8   (NHWC)
    w    : (C_out, K, C_in) int8  (OHWI)
    返回  (N, T, C_out) int8
    """
    N, Tx, Cin = q_in.shape
    Cout = w.shape[0]
    pad = (K - 1) * dil // 2
    centered = q_in.astype(np.int64) - in_zp
    acc = np.zeros((N, Tx, Cout), dtype=np.int64)
    for k in range(K):
        idx = np.arange(Tx) + k * dil - pad
        valid = ((idx >= 0) & (idx < Tx)).astype(np.int64)
        ic = np.clip(idx, 0, Tx - 1)
        shifted = centered[:, ic, :] * valid[None, :, None]     # (N, T, C_in)
        acc += shifted @ np.asarray(w[:, k, :], dtype=np.int64).T  # (N, T, C_out)
    acc += np.asarray(bq, dtype=np.int64)[None, None, :]
    out = requantize_np(acc, mult, shift) + out_zp
    return np.clip(out, act_min, act_max).astype(np.int8)


def fc_int8(q_in, w, bq, mult, shift, in_zp, out_zp, act_min, act_max):
    """手写 FC (跟 MCU 的 ch32_fc_s8 一致): (N, C_in) -> (N, C_out)."""
    centered = np.asarray(q_in, dtype=np.int64) - in_zp
    acc = centered @ np.asarray(w, dtype=np.int64).T + np.asarray(bq, dtype=np.int64)[None, :]
    out = requantize_np(acc, mult, shift) + out_zp
    return np.clip(out, act_min, act_max).astype(np.int8)


def fc_logit(q_in, w, bq, in_zp, logit_scale):
    """输出保持 int32 累加器 -> FP32 logit (per-channel 反量化乘子)."""
    centered = np.asarray(q_in, dtype=np.int64) - in_zp
    acc = centered @ np.asarray(w, dtype=np.int64).T + np.asarray(bq, dtype=np.int64)[None, :]
    return acc.astype(np.float32) * np.asarray(logit_scale, dtype=np.float32)


def gap_int8(q):
    """GAP over T: (N, T, C) -> (N, C), int8, 保持 scale/zp.

    舍入跟 C 一致: sum>0 -> (sum+half)/T, sum<=0 -> (sum-half)/T, 整数除法截断向零.
    """
    s = np.asarray(q, dtype=np.int64).sum(axis=1)
    half = T // 2
    v = np.where(s > 0, s + half, s - half)
    # C 的整数除法截断向零
    g = np.where(v >= 0, v // T, -((-v) // T))
    return g.astype(np.int8)


# ---------------------------------------------------------------------------
# 整网前向
# ---------------------------------------------------------------------------


class Int8Model(object):
    def __init__(self, hdr_path=INT8_HDR):
        self.h = parse_int8_header(hdr_path)
        h = self.h
        # 形状还原
        self.W = {}
        self.W["tcn_0_conv1"] = h["w_tcn_0_conv1"].reshape(CH, 3, IN_CH)
        self.W["tcn_0_conv2"] = h["w_tcn_0_conv2"].reshape(CH, 3, CH)
        self.W["tcn_0_residual"] = h["w_tcn_0_residual"].reshape(CH, 1, IN_CH)
        for bi in (1, 2):
            self.W["tcn_%d_conv1" % bi] = h["w_tcn_%d_conv1" % bi].reshape(CH, 3, CH)
            self.W["tcn_%d_conv2" % bi] = h["w_tcn_%d_conv2" % bi].reshape(CH, 3, CH)
        for bi in (0, 1, 2):
            self.W["tcn_%d_se_fc1" % bi] = h["w_tcn_%d_se_fc1" % bi].reshape(SE_H, CH)
            self.W["tcn_%d_se_fc2" % bi] = h["w_tcn_%d_se_fc2" % bi].reshape(CH, SE_H)
        self.W["fc1"] = h["w_fc1"].reshape(FC1_OUT, CH)
        self.W["fc2"] = h["w_fc2"].reshape(1, FC1_OUT)
        # logit 反量化乘子 (per-channel):  s_in * s_w[oc]
        self.LSW = {}
        for bi in (0, 1, 2):
            self.LSW["tcn_%d_se_fc2" % bi] = h["s_seh_%d" % bi] * h["sw_tcn_%d_se_fc2" % bi]
        self.LSW["fc2"] = h["s_fc1"] * h["sw_fc2"]

    # -- 输入量化 --
    def quant_input(self, x):
        """x: (N, 23, 16) FP32 -> (N, 16, 23) int8 (NHWC)."""
        return self.quant_with(x, self.h["s_x"], self.h["zp_x"])

    @staticmethod
    def quant_with(x, s, zp):
        """按 (scale, zp) 量化并转成 NHWC. x: (N, C, T) FP32 -> (N, T, C) int8.

        非对称量化: real = (q - zp) * s  =>  q = round(real / s) + zp.
        返回的是**未中心化**的 q (跟 zp 同一坐标系), 中心化由 cmsis_conv 用 in_zp 做."""
        t = np.asarray(x, dtype=np.float32) / np.float32(s)
        q = q_round(t.astype(np.float64)).astype(np.int64) + int(zp)
        q = np.clip(q, -128, 127)
        return np.transpose(q, (0, 2, 1)).astype(np.int8)

    def forward(self, x, se_fp32=False, ret_debug=False):
        h = self.h
        N = x.shape[0]
        dbg = {}

        xq = self.quant_input(x)                       # (N,16,23)
        dbg["xq"] = xq

        # ---- Block 0 ----
        res = cmsis_conv(xq, self.W["tcn_0_residual"], h["b_tcn_0_residual"],
                         h["mult_tcn_0_residual"], h["shift_tcn_0_residual"],
                         h["zp_x"], h["zp_add_0"], -128, 127, 1, 1)
        c1 = cmsis_conv(xq, self.W["tcn_0_conv1"], h["b_tcn_0_conv1"],
                        h["mult_tcn_0_conv1"], h["shift_tcn_0_conv1"],
                        h["zp_x"], h["zp_c1_0"], h["zp_c1_0"], 127, 3, DIL[0])
        c2 = cmsis_conv(c1, self.W["tcn_0_conv2"], h["b_tcn_0_conv2"],
                        h["mult_tcn_0_conv2"], h["shift_tcn_0_conv2"],
                        h["zp_c1_0"], h["zp_c2_0"], h["zp_c2_0"], 127, 3, DIL[0])
        cur = self._se_block(c2, 0, se_fp32, dbg)
        # add: 中心化整数相加
        res_c = res.astype(np.int64) - h["zp_add_0"]
        cur_c = cur.astype(np.int64) - h["zp_add_0"]
        add0 = np.clip(cur_c + res_c + h["zp_add_0"], h["zp_add_0"], 127).astype(np.int8)
        dbg["add_0"] = add0

        # ---- Block 1 / 2 ----
        for bi in (1, 2):
            prev = add0 if bi == 1 else add1
            zp_prev = h["zp_add_%d" % (bi - 1)]
            # identity residual: requant 到本 block 的 add 域
            res_c = requantize_np(prev.astype(np.int64) - zp_prev,
                                  h["mult_res_%d" % bi], h["shift_res_%d" % bi])
            c1 = cmsis_conv(prev, self.W["tcn_%d_conv1" % bi], h["b_tcn_%d_conv1" % bi],
                            h["mult_tcn_%d_conv1" % bi], h["shift_tcn_%d_conv1" % bi],
                            zp_prev, h["zp_c1_%d" % bi], h["zp_c1_%d" % bi], 127, 3, DIL[bi])
            c2 = cmsis_conv(c1, self.W["tcn_%d_conv2" % bi], h["b_tcn_%d_conv2" % bi],
                            h["mult_tcn_%d_conv2" % bi], h["shift_tcn_%d_conv2" % bi],
                            h["zp_c1_%d" % bi], h["zp_c2_%d" % bi], h["zp_c2_%d" % bi], 127, 3, DIL[bi])
            cur = self._se_block(c2, bi, se_fp32, dbg)
            cur_c = cur.astype(np.int64) - h["zp_add_%d" % bi]
            add = np.clip(cur_c + res_c + h["zp_add_%d" % bi],
                          h["zp_add_%d" % bi], 127).astype(np.int8)
            if bi == 1:
                add1 = add
            else:
                add2 = add
            dbg["add_%d" % bi] = add

        # ---- Classifier ----
        g = gap_int8(add2)                            # (N,32)
        h1 = fc_int8(g, self.W["fc1"], h["b_fc1"], h["mult_fc1"], h["shift_fc1"],
                     h["zp_add_2"], h["zp_fc1"], h["zp_fc1"], 127)
        logit = fc_logit(h1, self.W["fc2"], h["b_fc2"], h["zp_fc1"], self.LSW["fc2"])[:, 0]
        logit = logit / np.float32(TEMPERATURE)
        prob = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logit.astype(np.float32)))).astype(np.float32)
        if ret_debug:
            dbg["logit"] = logit
            dbg["prob"] = prob
            return prob, dbg
        return prob

    def _se_block(self, c2, bi, se_fp32, dbg):
        """SE: GAP -> fc1(+ReLU) -> fc2 -> sigmoid -> 逐通道缩放 (定点)."""
        h = self.h
        g = gap_int8(c2)                              # (N,32), scale/zp 同 c2
        if se_fp32:
            # FP32 SE: h 保持 FP32 (只有权重是 int8)
            g32 = (g.astype(np.float32) - np.float32(h["zp_c2_%d" % bi])) * np.float32(h["s_c2_%d" % bi])
            w1 = self.W["tcn_%d_se_fc1" % bi].astype(np.float32) * h["sw_tcn_%d_se_fc1" % bi][:, None]
            b1 = h["b_tcn_%d_se_fc1" % bi].astype(np.float32) * (np.float32(h["s_c2_%d" % bi]) *
                                                                 h["sw_tcn_%d_se_fc1" % bi])
            hf = np.maximum(g32 @ w1.T + b1, 0.0).astype(np.float32)
            w2 = self.W["tcn_%d_se_fc2" % bi].astype(np.float32) * h["sw_tcn_%d_se_fc2" % bi][:, None]
            b2 = h["b_tcn_%d_se_fc2" % bi].astype(np.float32) * (np.float32(h["s_seh_%d" % bi]) *
                                                                 h["sw_tcn_%d_se_fc2" % bi])
            logit = (hf @ w2.T + b2).astype(np.float32)
            sig = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logit))).astype(np.float32)
            # 缩放 (FP32 -> 定点 add 域)
            v = (c2.astype(np.int64) - h["zp_c2_%d" % bi])
            qsig = q_round(sig.astype(np.float64) * h["sig_q"]).astype(np.int64)[:, None, :]
            acc = v * qsig
            out = requantize_np(acc, h["mult_se_%d" % bi], h["shift_se_%d" % bi])
            return np.clip(out + h["zp_add_%d" % bi], -128, 127).astype(np.int8)
        # int8 SE
        hq = fc_int8(g, self.W["tcn_%d_se_fc1" % bi], h["b_tcn_%d_se_fc1" % bi],
                     h["mult_tcn_%d_se_fc1" % bi], h["shift_tcn_%d_se_fc1" % bi],
                     h["zp_c2_%d" % bi], h["zp_seh_%d" % bi], h["zp_seh_%d" % bi], 127)
        logit = fc_logit(hq, self.W["tcn_%d_se_fc2" % bi], h["b_tcn_%d_se_fc2" % bi],
                         h["zp_seh_%d" % bi], self.LSW["tcn_%d_se_fc2" % bi])
        sig = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logit))).astype(np.float32)
        v = c2.astype(np.int64) - h["zp_c2_%d" % bi]
        qsig = q_round(sig.astype(np.float64) * h["sig_q"]).astype(np.int64)[:, None, :]
        acc = v * qsig
        out = requantize_np(acc, h["mult_se_%d" % bi], h["shift_se_%d" % bi])
        return np.clip(out + h["zp_add_%d" % bi], -128, 127).astype(np.int8)


def run_all(x, hdr_path=INT8_HDR, batch=512, se_fp32=False, verbose=True):
    """对一批窗口跑 INT8 前向 -> probs (N,)."""
    m = Int8Model(hdr_path)
    out = []
    for i in range(0, len(x), batch):
        p = m.forward(x[i:i + batch], se_fp32=se_fp32)
        out.append(p)
        if verbose:
            print("[sim] %d/%d" % (min(i + batch, len(x)), len(x)), end="\r", flush=True)
    if verbose:
        print()
    return np.concatenate(out)


if __name__ == "__main__":
    import time
    os.environ["H743_CSV"] = "IanArffDatasetF32.csv"
    from h743_windows import build_windows
    wins, labels = build_windows(start=0, n=274628, phase=0)
    X = np.stack([w[2] for w in wins], axis=0).astype(np.float32)
    print("windows=%d" % len(X))
    t0 = time.time()
    p8 = run_all(X)
    print("int8 sim 用时 %.1fs" % (time.time() - t0))
    np.savez(os.path.join(HERE, "fullsim_int8.npz"),
             probs=p8, atk=np.array([False] * len(p8)),
             frames=np.array([w[0] for w in wins]))
    print("prob 统计: min=%.4f max=%.4f mean=%.4f  >0.5: %d (%.2f%%)"
          % (p8.min(), p8.max(), p8.mean(), (p8 > 0.5).sum(), 100.0 * (p8 > 0.5).mean()))
