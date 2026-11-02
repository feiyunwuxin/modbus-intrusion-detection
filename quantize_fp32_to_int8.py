"""
quantize_fp32_to_int8.py -- 把部署版 FP32 模型 (Route E) 全量量化成 INT8.

背景 (2026-10-21):
  H743 现在跑的是 FP32 手写 C kernel (MCU/Core/Src/ch32_inference.c), 部署区可信基线
  F1=0.8671 / FPR=5.91%. 本脚本把同一份权重 (model_v4_se_23dim_ch32_fp32_s42.h,
  部署真值) 量化成 INT8, 供 MCU 的 CMSIS-NN kernel 消费, 并跟 FP32 做逐窗对比.

量化方案 (TFLite/CMSIS-NN 标准):
  - 权重: int8, conv 用 per-output-channel 对称 (scale=max|w[oc]|/127),
          FC 用 per-tensor 对称 (CMSIS-NN 的 FC 只有 per-tensor requant).
  - 偏置: int32 = round(b / (s_in * s_w)).
  - 激活: int8 per-tensor 非对称 (real=0 <-> q=zp, ReLU 语义成立),
          zp 通常 -128, 用满 256 级.
  - requant: per-channel 定点 (multiplier, shift), 严格照抄 CMSIS-NN
    arm_nn_requantize (double rounding 分支, 见 int8_math.requantize_np).
  - FP32 escape: 只有 sigmoid / 最终 logit 反量化 (TFLite full-int8 也这么做).

校准集: train 区 (frame < 200001) 的 H743-view 窗口 (h743_windows.build_windows).
         部署区 (frame >= 200001) **不参与**校准, 只用于评估 -> 无泄漏.

用法:
    python quantize_fp32_to_int8.py            # 校准 + 写 header + 写 JSON 日志
    python quantize_fp32_to_int8.py --ranges   # 只打印各层激活范围 (设计用)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from int8_math import (asymmetric_activation, quantize_bias, quantize_multiplier_vec,
                       quantize_weight_per_channel, quantize_weight_per_tensor,
                       quantize_multiplier, round_half_away, write_int8_header)

FP32_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h")
INT8_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h")
QLOG_JSON = os.path.join(HERE, "int8_qparams_s42.json")

IN_CH, T, CH, SE_H, FC1_OUT, FC2_OUT = 23, 16, 32, 4, 32, 1
DIL = (1, 2, 4)
TRAIN_END = 200000        # train 区: frame <= 200000 (部署区从 200001 开始)
N_CALIB = 4000            # 校准窗口数 (train 区 12500 窗里等间隔抽)
CLIP_VAL = 10.0

# ---------------------------------------------------------------------------
# 1. 读 FP32 header
# ---------------------------------------------------------------------------


def parse_fp32_header(path):
    import re
    txt = open(path, encoding="utf-8", errors="replace").read()
    A = {}
    for m in re.finditer(r"static const float\s+(\w+)\[(\d+)\]\s*=\s*\{(.*?)\};", txt, re.S):
        vals = [float(x[:-1]) for x in re.findall(r"[-+0-9.eE]+f", m.group(3))]
        assert len(vals) == int(m.group(2)), m.group(1)
        A[m.group(1)] = np.array(vals, dtype=np.float32)
    return A


P = "v4se23_ch32_s42_"


def load_fp32_model():
    raw = parse_fp32_header(FP32_HDR)
    W = {}
    W["w_tcn_0_conv1"] = raw[P + "w_tcn_0_conv1_weight"].reshape(CH, IN_CH, 3)
    W["b_tcn_0_conv1"] = raw[P + "b_tcn_0_conv1_bias"]
    W["w_tcn_0_conv2"] = raw[P + "w_tcn_0_conv2_weight"].reshape(CH, CH, 3)
    W["b_tcn_0_conv2"] = raw[P + "b_tcn_0_conv2_bias"]
    W["w_tcn_0_residual"] = raw[P + "w_tcn_0_residual_weight"].reshape(CH, IN_CH, 1)
    W["b_tcn_0_residual"] = raw[P + "b_tcn_0_residual_bias"]
    W["w_fc1"] = raw[P + "fc1_weight"].reshape(FC1_OUT, CH)
    W["b_fc1"] = raw[P + "b_fc1_bias"]
    W["w_fc2"] = raw[P + "fc2_weight"].reshape(FC2_OUT, FC1_OUT)
    W["b_fc2"] = raw[P + "b_fc2_bias"]
    for bi in (1, 2):
        W["w_tcn_%d_conv1" % bi] = raw[P + ("w_tcn_%d_conv1_weight" % bi)].reshape(CH, CH, 3)
        W["b_tcn_%d_conv1" % bi] = raw[P + ("b_tcn_%d_conv1_bias" % bi)]
        W["w_tcn_%d_conv2" % bi] = raw[P + ("w_tcn_%d_conv2_weight" % bi)].reshape(CH, CH, 3)
        W["b_tcn_%d_conv2" % bi] = raw[P + ("b_tcn_%d_conv2_bias" % bi)]
    for bi in (0, 1, 2):
        W["w_tcn_%d_se_fc1" % bi] = raw[P + ("w_tcn_%d_se_fc1_weight" % bi)].reshape(SE_H, CH)
        W["b_tcn_%d_se_fc1" % bi] = raw[P + ("b_tcn_%d_se_fc1_bias" % bi)]
        W["w_tcn_%d_se_fc2" % bi] = raw[P + ("w_tcn_%d_se_fc2_weight" % bi)].reshape(CH, SE_H)
        W["b_tcn_%d_se_fc2" % bi] = raw[P + ("b_tcn_%d_se_fc2_bias" % bi)]
    return W


# ---------------------------------------------------------------------------
# 2. FP32 traced forward (拿各中间张量的范围)
# ---------------------------------------------------------------------------


def conv1d_same(x, w, b, dil):
    """x (N,C,T) -> (N,OC,T). 对称 same padding, 跟 C 的 conv1d_same / PyTorch 一致."""
    N, C, Tx = x.shape
    OC, _, K = w.shape
    pad = (K - 1) * dil // 2
    out = np.zeros((N, OC, Tx), dtype=np.float64)
    out += np.asarray(b, dtype=np.float64)[None, :, None]
    for k in range(K):
        idx = np.arange(Tx) + k * dil - pad
        valid = ((idx >= 0) & (idx < Tx)).astype(np.float64)
        ic = np.clip(idx, 0, Tx - 1)
        out += np.einsum("oc,nct->not", np.asarray(w[:, :, k], dtype=np.float64),
                         np.asarray(x[:, :, ic], dtype=np.float64) * valid[None, None, :])
    return out


def fp32_trace(W, x):
    """返回各中间张量 (numpy, float64) + 最终 logit."""
    x = np.asarray(x, dtype=np.float64)
    tr = {}
    tr["x"] = x
    res = conv1d_same(x, W["w_tcn_0_residual"], W["b_tcn_0_residual"], 1)
    tr["res_0"] = res
    c1 = np.maximum(conv1d_same(x, W["w_tcn_0_conv1"], W["b_tcn_0_conv1"], DIL[0]), 0)
    tr["c1_0"] = c1
    c2 = np.maximum(conv1d_same(c1, W["w_tcn_0_conv2"], W["b_tcn_0_conv2"], DIL[0]), 0)
    tr["c2_0"] = c2
    se, h, s = se_fp32(c2, W, 0)
    tr["se_0"], tr["seh_0"], tr["ses_0"] = se, h, s
    add = np.maximum(se + res, 0)
    tr["add_0"] = add
    cur = add
    for bi in (1, 2):
        res = cur.copy()
        tr["res_%d" % bi] = res
        c1 = np.maximum(conv1d_same(cur, W["w_tcn_%d_conv1" % bi], W["b_tcn_%d_conv1" % bi], DIL[bi]), 0)
        tr["c1_%d" % bi] = c1
        c2 = np.maximum(conv1d_same(c1, W["w_tcn_%d_conv2" % bi], W["b_tcn_%d_conv2" % bi], DIL[bi]), 0)
        tr["c2_%d" % bi] = c2
        se, h, s = se_fp32(c2, W, bi)
        tr["se_%d" % bi], tr["seh_%d" % bi], tr["ses_%d" % bi] = se, h, s
        cur = np.maximum(se + res, 0)
        tr["add_%d" % bi] = cur
    g = cur.sum(axis=2) / T
    tr["gap"] = g
    h = np.maximum(g @ W["w_fc1"].T + W["b_fc1"], 0)
    tr["fc1"] = h
    logit = (h @ W["w_fc2"].T + W["b_fc2"])[:, 0]
    tr["logit"] = logit
    return tr


def se_fp32(x, W, bi):
    """SE block (FP32): GAP -> fc1+ReLU -> fc2+sigmoid -> 逐通道缩放.

    返回 (缩放后的张量, fc1 输出 h, sigmoid 系数 s)."""
    g = x.sum(axis=2) / T
    h = np.maximum(g @ W["w_tcn_%d_se_fc1" % bi].T + W["b_tcn_%d_se_fc1" % bi], 0)
    s = 1.0 / (1.0 + np.exp(-(h @ W["w_tcn_%d_se_fc2" % bi].T + W["b_tcn_%d_se_fc2" % bi])))
    return x * s[:, :, None], h, s


# ---------------------------------------------------------------------------
# 3. 校准集
# ---------------------------------------------------------------------------


def build_calib(n=N_CALIB):
    """train 区的 H743-view 窗口 (等间隔抽 n 个)."""
    os.environ["H743_CSV"] = "IanArffDatasetF32.csv"
    from h743_windows import build_windows
    t0 = time.time()
    wins, labels = build_windows(start=0, n=TRAIN_END, phase=0)
    print("[calib] train 区窗口 %d 个 (frame<=200000), 用时 %.1fs" % (len(wins), time.time() - t0))
    idx = np.linspace(0, len(wins) - 1, min(n, len(wins))).astype(int)
    X = np.stack([wins[i][2] for i in idx], axis=0).astype(np.float32)
    print("[calib] 采样 %d 窗, 形状 %s" % (len(idx), X.shape))
    return X, [wins[i][0] for i in idx]


PCT_KEYS = (0.01, 0.1, 1.0, 99.0, 99.9, 99.99)


def collect_ranges(W, X, batch=250):
    """在校准集上跑 FP32 traced forward, 收集每个张量的 min/max + 分位数.

    返回 (minmax dict, pct dict). pct dict: {key: {"99.0": v, "99.9": v, "99.99": v}}
    分位数用于"截断式"激活范围 (去掉离群点, 换取分辨率).
    """
    keys = ["x", "res_0", "c1_0", "c2_0", "se_0", "add_0", "seh_0", "ses_0",
            "res_1", "c1_1", "c2_1", "se_1", "add_1", "seh_1", "ses_1",
            "res_2", "c1_2", "c2_2", "se_2", "add_2", "seh_2", "ses_2",
            "gap", "fc1", "logit"]
    rng = {k: [np.inf, -np.inf] for k in keys}
    allv = {k: [] for k in keys}
    for i in range(0, len(X), batch):
        tr = fp32_trace(W, X[i:i + batch])
        for k in keys:
            v = tr[k]
            rng[k][0] = min(rng[k][0], float(np.min(v)))
            rng[k][1] = max(rng[k][1], float(np.max(v)))
            allv[k].append(np.asarray(v, dtype=np.float64).ravel())
        print("[calib] %d/%d" % (min(i + batch, len(X)), len(X)), end="\r", flush=True)
    print()
    pct = {}
    for k in keys:
        v = np.concatenate(allv[k])
        pct[k] = {str(p): float(np.percentile(v, p)) for p in PCT_KEYS}
    return {k: (v[0], v[1]) for k, v in rng.items()}, pct


# ---------------------------------------------------------------------------
# 4. 量化参数设计
# ---------------------------------------------------------------------------

# SE sigmoid 的定点表示: Q12 (0..4096).
#   - 精度: 相对误差 <= 1/8192 = 0.012%  (对 SE 门控完全可忽略)
#   - 为什么要定点: MCU 的 expf 跟 PC 端 numpy.exp 可能差 1 ULP (~1e-7 相对).
#     若直接把 FP32 sigmoid 乘进 SE scale-back, 这点差异会原样传到最终 prob.
#     走定点 Q12 后, 只有当 sig*4096 恰好落在 .5 边界 0.04% 附近才会翻转 -> 可接受.
SIG_Q = 4096


def design(W, rng, input_range, pct=None, pct_sel=99.9):
    """根据激活范围设计所有量化参数. 返回 (q, log).

    input_range = (s_x, zp_x, real_lo, real_hi)
    pct         = 分位数表 (可选), 用来做截断式激活范围
    pct_sel     = 用哪个分位数当激活上界 (99 / 99.9 / 99.99)"""
    q = {}
    log = {"ranges": {k: list(v) for k, v in rng.items()}}

    # ---- 模型输入: int8. 范围是 **可调旋钮**, 只能在 train 区选 (不能看部署区) ----
    #   默认: 对称 +-CLIP_VAL (a priori 已知: feature_extractor 最后 clip +-10)
    #   可选: 非对称 [ilo, ihi] (train 区实测分位数), 提高输入分辨率
    s_x, zp_x = float(input_range[0]), int(input_range[1])
    q["s_x"], q["zp_x"] = s_x, zp_x
    log["input"] = {"observed": list(rng["x"]), "scale": float(s_x), "zp": int(zp_x),
                    "real_lo": float(input_range[2]), "real_hi": float(input_range[3])}

    def hi_of(key):
        """激活上界: 默认真实 max; 若给了 pct 表则用分位数 (截断离群点换分辨率)."""
        if pct is None:
            return rng[key][1]
        return pct[key][str(pct_sel)]

    def lo_of(key):
        """激活下界: 默认真实 min; 分位数模式下用下尾分位数."""
        if pct is None:
            return rng[key][0]
        return pct[key][str(round(100.0 - pct_sel, 2))]

    for bi in (0, 1, 2):
        # conv1 / conv2 输出 (post-ReLU): 非对称, real 0 <-> q=zp
        for nm in ("c1", "c2"):
            s, zp = asymmetric_activation(0.0, hi_of("%s_%d" % (nm, bi)))
            q["s_%s_%d" % (nm, bi)] = s
            q["zp_%s_%d" % (nm, bi)] = zp
        # add 域: 同时容纳 SE 输出和 residual
        lo = min(0.0, lo_of("res_%d" % bi))
        hi = max(hi_of("se_%d" % bi), hi_of("res_%d" % bi))
        s_add, zp_add = asymmetric_activation(lo, hi)
        q["s_add_%d" % bi] = s_add
        q["zp_add_%d" % bi] = zp_add
        log["add_%d" % bi] = {"lo": lo, "hi": hi, "scale": float(s_add), "zp": zp_add}
        # SE fc1 输出 (post-ReLU)
        s, zp = asymmetric_activation(0.0, hi_of("seh_%d" % bi))
        q["s_seh_%d" % bi] = s
        q["zp_seh_%d" % bi] = zp
        # SE scale-back 定点乘子:  (q_c2 - zp_c2) * q_sig  ->  add 域
        #   q_sig = sigmoid(logit) * 32768  (Q15)
        #   目标整数 = (q_c2-zp_c2) * s_c2 * sig / s_add
        #            = (q_c2-zp_c2) * q_sig * (s_c2 / (32768 * s_add))
        f = float(q["s_c2_%d" % bi]) / (SIG_Q * float(s_add))
        m, sh = quantize_multiplier(f)
        q["mult_se_%d" % bi] = np.int32(m)
        q["shift_se_%d" % bi] = np.int32(sh)
        log["se_scaleback_%d" % bi] = {"factor": f, "mult": int(m), "shift": int(sh)}
        # identity residual 的 requant (block1/2): 上一个 block 的 add 输出 -> 本 block 的 add 域
        if bi > 0:
            f = float(q["s_add_%d" % (bi - 1)]) / float(s_add)
            m, sh = quantize_multiplier(f)
            q["mult_res_%d" % bi] = np.int32(m)
            q["shift_res_%d" % bi] = np.int32(sh)
            log["res_requant_%d" % bi] = {"factor": f, "mult": int(m), "shift": int(sh)}

    # ---- classifier ----
    s, zp = asymmetric_activation(0.0, hi_of("fc1"))
    q["s_fc1"], q["zp_fc1"] = s, zp
    return q, log


# ---------------------------------------------------------------------------
# 5. 权重 / 偏置 / requant 量化
# ---------------------------------------------------------------------------


def build_arrays(W, q, log):
    """生成 INT8 header 需要的全部数组 (int8 权重 + scale + int32 bias + requant)."""
    A = {}
    S = {}

    def add_conv(tag, w, b, s_in, s_out, zp_out, act_min, act_max):
        """w: (oc, ic, k) FP32 -> CMSIS 布局 (oc, k, ic)."""
        qw, sw = quantize_weight_per_channel(w)
        qw = np.transpose(qw, (0, 2, 1)).reshape(w.shape[0], w.shape[2], w.shape[1])  # (oc,k,ic)
        bq = quantize_bias(b, s_in, sw)
        mult, shift = quantize_multiplier_vec(float(s_in) * sw / float(s_out))
        A["w_%s" % tag] = ("i8", qw.reshape(-1))
        A["sw_%s" % tag] = ("f32", sw)
        A["b_%s" % tag] = ("i32", bq)
        A["mult_%s" % tag] = ("i32", mult)
        A["shift_%s" % tag] = ("i32", shift)
        log.setdefault("requant", {})[tag] = {
            "s_in": float(s_in), "s_out": float(s_out),
            "mult_min": int(mult.min()), "mult_max": int(mult.max()),
            "shift_min": int(shift.min()), "shift_max": int(shift.max()),
            "w_absmax": float(np.max(np.abs(w))),
        }
        return qw, sw, bq, mult, shift

    def add_fc(tag, w, b, s_in, s_out=None, zp_out=None, requant=True):
        """w: (oc, ic) FP32. requant=False -> 输出保持 int32 累加器 (logit 走 FP32)."""
        qw, sw = quantize_weight_per_channel(w)
        bq = quantize_bias(b, s_in, sw)
        A["w_%s" % tag] = ("i8", qw.reshape(-1))
        A["sw_%s" % tag] = ("f32", sw)
        A["b_%s" % tag] = ("i32", bq)
        if requant:
            mult, shift = quantize_multiplier_vec(float(s_in) * sw / float(s_out))
            A["mult_%s" % tag] = ("i32", mult)
            A["shift_%s" % tag] = ("i32", shift)
        log.setdefault("requant", {})[tag] = {
            "s_in": float(s_in), "w_absmax": float(np.max(np.abs(w))),
            "w_l2": float(np.linalg.norm(w, axis=1).max()),
        }
        if requant:
            log["requant"][tag]["shift_min"] = int(shift.min())
            log["requant"][tag]["shift_max"] = int(shift.max())
            log["requant"][tag]["mult_min"] = int(mult.min())
            log["requant"][tag]["mult_max"] = int(mult.max())
        return qw, sw, bq

    # ---- conv 层 ----
    for bi in (0, 1, 2):
        in_ch = IN_CH if bi == 0 else CH
        add_conv("tcn_%d_conv1" % bi, W["w_tcn_%d_conv1" % bi], W["b_tcn_%d_conv1" % bi],
                 q["s_x"] if bi == 0 else q["s_add_%d" % (bi - 1)],
                 q["s_c1_%d" % bi], q["zp_c1_%d" % bi], q["zp_c1_%d" % bi], 127)
        add_conv("tcn_%d_conv2" % bi, W["w_tcn_%d_conv2" % bi], W["b_tcn_%d_conv2" % bi],
                 q["s_c1_%d" % bi], q["s_c2_%d" % bi], q["zp_c2_%d" % bi], q["zp_c2_%d" % bi], 127)
        if bi == 0:
            # residual 1x1 conv: 直接输出到 block0 的 add 域 (省掉一次 requant)
            add_conv("tcn_0_residual", W["w_tcn_0_residual"], W["b_tcn_0_residual"],
                     q["s_x"], q["s_add_0"], q["zp_add_0"], -128, 127)

    # ---- FC 层 ----
    for bi in (0, 1, 2):
        # SE fc1: (s_c2) -> (s_seh), post-ReLU
        add_fc("tcn_%d_se_fc1" % bi, W["w_tcn_%d_se_fc1" % bi], W["b_tcn_%d_se_fc1" % bi],
               q["s_c2_%d" % bi], s_out=q["s_seh_%d" % bi], zp_out=q["zp_seh_%d" % bi])
        # SE fc2: -> int32 累加器 (logit 不量化, sigmoid 前保持精度)
        add_fc("tcn_%d_se_fc2" % bi, W["w_tcn_%d_se_fc2" % bi], W["b_tcn_%d_se_fc2" % bi],
               q["s_seh_%d" % bi], requant=False)
    # classifier fc1: (s_add2) -> (s_fc1), post-ReLU
    add_fc("fc1", W["w_fc1"], W["b_fc1"], q["s_add_2"], s_out=q["s_fc1"], zp_out=q["zp_fc1"])
    # classifier fc2: -> int32 累加器 -> logit
    add_fc("fc2", W["w_fc2"], W["b_fc2"], q["s_fc1"], requant=False)

    # ---- 标量 ----
    for k in ("s_x", "s_c1_0", "s_c2_0", "s_add_0", "s_seh_0",
              "s_c1_1", "s_c2_1", "s_add_1", "s_seh_1",
              "s_c1_2", "s_c2_2", "s_add_2", "s_seh_2", "s_fc1"):
        S[k] = ("f32", float(q[k]))
    for k in ("zp_x", "zp_c1_0", "zp_c2_0", "zp_add_0", "zp_seh_0",
              "zp_c1_1", "zp_c2_1", "zp_add_1", "zp_seh_1",
              "zp_c1_2", "zp_c2_2", "zp_add_2", "zp_seh_2", "zp_fc1"):
        S[k] = ("i32", int(q[k]))
    for bi in (0, 1, 2):
        S["mult_se_%d" % bi] = ("i32", int(q["mult_se_%d" % bi]))
        S["shift_se_%d" % bi] = ("i32", int(q["shift_se_%d" % bi]))
        if bi > 0:
            S["mult_res_%d" % bi] = ("i32", int(q["mult_res_%d" % bi]))
            S["shift_res_%d" % bi] = ("i32", int(q["shift_res_%d" % bi]))
    S["sig_q"] = ("i32", SIG_Q)
    return A, S



# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------


def parse_input_range(argv, rng):
    """决定模型输入的量化范围. 只允许用 train 区信息.

    返回 (s_x, zp_x, lo, hi).
      --clip V        : 对称 [-V, V]
      --ilo L --ihi H : 非对称 [L, H]
      --ipct LO,HI    : 按 train 区实测分位数 (需要 --pct-source 提供的窗口)
    """
    def get(a, d=None):
        return d if a is None else float(a)
    lo = get(_arg(argv, "--ilo"))
    hi = get(_arg(argv, "--ihi"))
    clip = get(_arg(argv, "--clip"))
    if lo is not None and hi is not None:
        s, zp = asymmetric_activation(lo, hi)
        return np.float32(s), int(zp), lo, hi
    # 对称: s = V/127, zp = 0 (原基线的做法)
    v = CLIP_VAL if clip is None else clip
    return np.float32(v / 127.0), 0, -v, v


def _arg(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def main():
    ranges_only = "--ranges" in sys.argv
    out_hdr = _arg(sys.argv, "--out") or INT8_HDR
    out_json = _arg(sys.argv, "--json") or QLOG_JSON
    from_json = _arg(sys.argv, "--from-json")
    W = load_fp32_model()
    if from_json:
        # 复用已校准的激活范围 (换输入范围不用重跑校准)
        with open(from_json, encoding="utf-8") as f:
            prev = json.load(f)
        rng = {k: (float(v[0]), float(v[1])) for k, v in prev["ranges"].items()}
        pct = prev.get("ranges_pct")
        X = None
        print("[calib] 复用 %s 里的激活范围" % from_json)
    else:
        X, frames = build_calib()
        rng, pct = collect_ranges(W, X)

    print("\n=== 激活范围 (train 区 %d 窗) ===" % (len(X) if X is not None else 0))
    for k in ("x", "c1_0", "c2_0", "se_0", "res_0", "add_0", "seh_0",
              "c1_1", "c2_1", "se_1", "add_1", "seh_1",
              "c1_2", "c2_2", "se_2", "add_2", "seh_2", "gap", "fc1", "logit"):
        lo, hi = rng[k]
        print("  %-8s [%12.5f, %12.5f]  absmax=%.4f" % (k, lo, hi, max(abs(lo), abs(hi))))
    if ranges_only:
        return

    # ---- 设计量化参数 ----
    input_range = parse_input_range(sys.argv, rng)
    pct_arg = _arg(sys.argv, "--act-pct")
    pct_sel = float(pct_arg) if pct_arg else 99.9
    use_pct = pct if (pct_arg and pct is not None) else None
    q, log = design(W, rng, input_range, pct=use_pct, pct_sel=pct_sel)
    A, S = build_arrays(W, q, log)

    # ---- 溢出体检: CMSIS 的 requantize 在 shift>0 时做 int32 左移, 不能溢出 ----
    print("\n=== requant 检查 (shift>0 -> int32 左移, 需 acc_max*2^shift < 2^31) ===")
    worst = 0
    for tag, d in log["requant"].items():
        if "shift_max" not in d:
            continue
        nterms = {"tcn_0_conv1": 3 * IN_CH, "tcn_0_residual": 1 * IN_CH,
                  "tcn_0_conv2": 3 * CH, "tcn_1_conv1": 3 * CH, "tcn_1_conv2": 3 * CH,
                  "tcn_2_conv1": 3 * CH, "tcn_2_conv2": 3 * CH,
                  "fc1": CH, "tcn_0_se_fc1": CH, "tcn_1_se_fc1": CH, "tcn_2_se_fc1": CH}.get(tag, 1)
        acc_max = nterms * 255 * 127
        ls = max(0, d["shift_max"])
        v = acc_max * (1 << ls)
        worst = max(worst, v)
        flag = "OK " if v < 2 ** 31 else "!!! OVERFLOW"
        print("  %-16s nterms=%3d shift=[%d,%d] acc_max=%9d -> %12d  %s"
              % (tag, nterms, d["shift_min"], d["shift_max"], acc_max, v, flag))
    assert worst < 2 ** 31, "requantize int32 溢出!"

    # ---- 写 header ----
    meta = [
        "全量 INT8 权重 (per-channel 对称权重 + int32 bias + per-channel requant)",
        "",
        "生成: quantize_fp32_to_int8.py (2026-10-21)",
        "校准集: train 区 (frame <= %d) 的 H743-view 窗口, 采样 %s 窗"
        % (TRAIN_END, len(X) if X is not None else "复用"),
        "部署区 (frame >= 200001) 未参与校准",
        "",
        "模型输入量化范围: real [%.4f, %.4f]  s_x=%.9g  zp_x=%d"
        % (input_range[2], input_range[3], float(input_range[0]), int(input_range[1])),
        "",
        "定点语义: arm_nn_requantize (CMSIS-NN double-rounding 分支)",
        "  requantize(v, mult, shift) = div_pow2(  (v*mult + 2^30) >> 31 ,  max(0,-shift) )",
        "  真实乘数 = mult * 2^(shift-31)",
        "",
        "SE scale-back: (q_c2 - zp_c2) * q_sig  --->  requantize(mult_se_b, shift_se_b)",
        "  q_sig = round(sigmoid(logit) * %d)  (Q12, runtime)" % SIG_Q,
        "",
        "classifier fc2 / SE fc2: 输出保持 int32 累加器 (logit 不量化, sigmoid 前无损)",
    ]
    write_int8_header(out_hdr, A, S, meta)
    nbytes = sum(len(np.asarray(v[1]).ravel()) * {"i8": 1, "i32": 4, "f32": 4}[v[0]]
                 for v in A.values())
    print("\n[write] %s" % out_hdr)
    print("[write] 数组 %d 个, 权重+参数共 %d B (%.1f KB) [FP32 权重是 20493*4 = %d B]"
          % (len(A), nbytes, nbytes / 1024.0, 20493 * 4))

    # ---- 写校准日志 ----
    if pct is not None:
        log["ranges_pct"] = pct
    log["act_pct_used"] = pct_sel if use_pct is not None else None
    log["calib"] = {"windows": len(X) if X is not None else None, "train_end_frame": TRAIN_END,
                    "csv": "IanArffDatasetF32.csv", "n_total_train_windows": 12500}
    log["quant"] = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                    for k, v in q.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)
    print("[write] %s" % out_json)


if __name__ == "__main__":
    main()
