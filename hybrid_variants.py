"""
hybrid_variants.py -- 针对 "权重量化导致 FPR 上升" 的三个修正方案 (PC 侧).

背景 (compare_fp32_vs_hybrid.py + diag_hybrid_shift.py 的结论)
------------------------------------------------------------
  2x2 归因: 只量化激活 -> FPR 完全不变 (0.0591); 只量化权重 -> FPR 涨到 0.0689.
  => FPR 上升的元凶是 **权重量化**.
  机制: int8 舍入误差跟数据的均值激活相关 -> 系统性把 logit 上推 (+0.186, 中位数同号).
  对照: 等幅随机高斯噪声的 Δlogit 均值在 [-0.085, +0.067] (无偏), int8 舍入是 +0.186.

三个修正方案 (权重存储不变或变大, 都不增加 RAM 拷贝)
---------------------------------------------------
  V1 int16 权重   : scale = absmax/32767.  量化误差比 int8 小 256 倍 -> 应 ≈ FP32.
                     代价: Flash 权重 25 KB -> 41 KB (仍比 FP32 的 82 KB 少一半, 且无 RAM 拷贝)
  V2 MSE 最优裁剪 : per-channel 搜索截断阈值 p, 最小化 ||w - dequant(w,p)||^2.
                     int8 大小不变, 只改 scale. 误差比 absmax 裁剪小.
  V3 偏置校正     : 逐层 greedy 标定, 让每层输出的均值跟 FP32 参考对齐.
                     只改 bias 常量, 0 字节额外开销.

标定纪律: V2 / V3 的所有标定量 **只在 train 区 (frame<200001) 上算**,
          部署区 (frame>=200001) 只做最终评估, 绝不参与拟合.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim_hybrid_kernel import (CH, DIL, FC1_OUT, IN_CH, SE_H, T, _SHAPES,
                               _TENSORS, conv1d_same_batch, load_weights,
                               se_block)

# 需要量化 + 可校正的层 (顺序 = 前向顺序)
CONV_ORDER = ["tcn_0_residual", "tcn_0_conv1", "tcn_0_conv2",
               "tcn_1_conv1", "tcn_1_conv2",
               "tcn_2_conv1", "tcn_2_conv2"]
SE1_ORDER = ["tcn_0_se_fc1", "tcn_1_se_fc1", "tcn_2_se_fc1"]
SE2_ORDER = ["tcn_0_se_fc2", "tcn_1_se_fc2", "tcn_2_se_fc2"]


# ---------------------------------------------------------------------------
# 权重量化 (三种)
# ---------------------------------------------------------------------------


def quantize_dequant(w, bits=8, mse_search=False, n_search=64):
    """per-channel 对称量化后立刻反量化 -> 返回 (w_q, scale).

    bits=8  : scale = absmax/127
    bits=16 : scale = absmax/32767
    mse_search=True: 逐通道搜索截断阈值 p*[0.5,1.0], 最小化 ||w - dequant||^2
    """
    O = w.shape[0]
    w = np.asarray(w, dtype=np.float64)
    flat = w.reshape(O, -1)
    qmax = (1 << (bits - 1)) - 1
    out = np.empty_like(flat)
    scales = np.empty(O, dtype=np.float64)
    for oc in range(O):
        v = flat[oc]
        amax = np.abs(v).max()
        if amax <= 0:
            scales[oc] = 1.0
            out[oc] = v
            continue
        best_p, best_err = 1.0, None
        if mse_search:
            for p in np.linspace(0.5, 1.0, n_search):
                s = (amax * p) / qmax
                if s <= 0:
                    continue
                dq = np.clip(np.round(v / s), -qmax, qmax) * s
                err = float(((dq - v) ** 2).sum())
                if best_err is None or err < best_err:
                    best_err, best_p = err, p
        s = (amax * best_p) / qmax
        scales[oc] = s
        out[oc] = np.clip(np.round(v / s), -qmax, qmax) * s
    return out.reshape(w.shape), scales


def build_variant(W32, bits=8, mse_search=False):
    """从 FP32 权重构造一个量化变体 (返回同结构的 W dict)."""
    W = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in W32.items()}
    for nm in _TENSORS:
        wq, sc = quantize_dequant(W32["w_" + nm], bits=bits, mse_search=mse_search)
        W["w_" + nm] = wq
        W["_scale_" + nm] = sc
    return W


# ---------------------------------------------------------------------------
# 前向 (带逐层 hook, 供偏置校正用)
# ---------------------------------------------------------------------------


def forward_hooks(W, X):
    """返回一个 dict.

    h[层名] = 该层的 **pre-activation** (ReLU/sigmoid 之前), 供偏置校正对齐均值.
    h["se_%d"] = SE 缩放后的输出 (c2 * sigmoid), 供 se_fc2 的 Newton 修正用.
    """
    x = np.asarray(X, dtype=np.float32).astype(np.float64)
    h = {}

    def conv(inp, w, b, dil, tag):
        O = w.shape[0]
        Tx = inp.shape[2]
        out = np.broadcast_to(np.asarray(b, dtype=np.float64)[None, :, None],
                              (inp.shape[0], O, Tx)).copy()
        for k in range(w.shape[2]):
            from sim_hybrid_kernel import t_in_index
            idx = t_in_index(Tx, w.shape[2], dil, k)
            valid = ((idx >= 0) & (idx < Tx)).astype(np.float64)
            ic = np.clip(idx, 0, Tx - 1)
            out += np.einsum("oc,nct->not", np.asarray(w[:, :, k], dtype=np.float64),
                             inp[:, :, ic] * valid[None, None, :])
        h[tag] = out
        return out

    def se(c2, bi):
        """返回 (se_fc1 的 pre-activation, se_fc2 的 pre-activation, 缩放后的输出)."""
        g = c2.sum(axis=2) / T
        w1 = np.asarray(W["w_tcn_%d_se_fc1" % bi], dtype=np.float64)
        b1 = np.asarray(W["b_tcn_%d_se_fc1" % bi], dtype=np.float64)
        pre1 = g @ w1.T + b1
        h["tcn_%d_se_fc1" % bi] = pre1
        hh = np.maximum(pre1, 0.0)
        w2 = np.asarray(W["w_tcn_%d_se_fc2" % bi], dtype=np.float64)
        b2 = np.asarray(W["b_tcn_%d_se_fc2" % bi], dtype=np.float64)
        pre2 = hh @ w2.T + b2
        h["tcn_%d_se_fc2" % bi] = pre2
        sig = 1.0 / (1.0 + np.exp(-pre2))
        out = c2 * sig[:, :, None]
        h["se_%d" % bi] = out
        return out

    # 注意: residual 旁路是 1x1 conv, **不接 ReLU** (ReLU 在相加之后才做)
    res = conv(x, W["w_tcn_0_residual"], W["b_tcn_0_residual"], 1, "tcn_0_residual")
    c1 = np.maximum(conv(x, W["w_tcn_0_conv1"], W["b_tcn_0_conv1"], DIL[0], "tcn_0_conv1"), 0)
    c2 = np.maximum(conv(c1, W["w_tcn_0_conv2"], W["b_tcn_0_conv2"], DIL[0], "tcn_0_conv2"), 0)
    cur = se(c2, 0)
    cur = np.maximum(cur + res, 0)

    for bi in (1, 2):
        res = cur
        c1 = np.maximum(conv(cur, W["w_tcn_%d_conv1" % bi], W["b_tcn_%d_conv1" % bi],
                             DIL[bi], "tcn_%d_conv1" % bi), 0)
        c2 = np.maximum(conv(c1, W["w_tcn_%d_conv2" % bi], W["b_tcn_%d_conv2" % bi],
                             DIL[bi], "tcn_%d_conv2" % bi), 0)
        cur = se(c2, bi)
        cur = np.maximum(cur + res, 0)

    g = cur.sum(axis=2) / T
    h["gap"] = g
    pre_fc1 = g @ np.asarray(W["fc1"], dtype=np.float64).T + np.asarray(W["b_fc1"], dtype=np.float64)
    h["fc1"] = pre_fc1
    hh = np.maximum(pre_fc1, 0.0)
    logit = (hh @ np.asarray(W["fc2"], dtype=np.float64).T
             + np.asarray(W["b_fc2"], dtype=np.float64))[:, 0]
    h["logit"] = logit
    return h


def forward(W, X):
    h = forward_hooks(W, X)
    lg = h["logit"]
    return 1.0 / (1.0 + np.exp(-lg)), lg


# ---------------------------------------------------------------------------
# V3: 逐层偏置校正 (标定集 = train 区)
# ---------------------------------------------------------------------------


def bias_correct(W32, Wq, Xcal, verbose=True, mode="per_channel", se2_mode="pre"):
    """逐层 greedy 偏置校正: 让量化模型每层的输出跟 FP32 参考对齐.

    mode="per_channel" (推荐):
        对每个输出通道 oc 单独修 bias:
            Δb[oc] = mean_{N,T}( ref_pre[oc] ) - mean_{N,T}( q_pre[oc] )
    mode="scalar":
        只用一个标量对齐全局均值 (早期版本, 太弱且不稳).

    se2_mode 控制 SE fc2 (sigmoid 那一层) 的处理:
        "skip" : 不动 (只修线性层)
        "pre"  : 对齐 **pre-sigmoid** 均值 (线性, 目标必然可达, 稳定)
        "out"  : 对齐 **缩放后输出** c2*sigmoid(pre2) 的均值 (逐通道二分;
                 当量化模型的 c2 本身有偏时目标可能不可达 -> 会撞边界)

    所有标定量都在标定集 (train 区) 上算, 部署区不参与.
    """
    W = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Wq.items()}
    h32 = forward_hooks(W32, Xcal)
    hq = forward_hooks(W, Xcal)
    adj = {}

    def chan_mean(a):
        """(N, OC, T) -> (OC,) 每个通道在 N,T 上的均值; (N, OC) -> (OC,)"""
        if a.ndim == 3:
            return a.mean(axis=(0, 2))
        return a.mean(axis=0)

    # --- 线性层: 逐通道对齐 pre-activation 均值 ---
    for nm in CONV_ORDER + SE1_ORDER:
        d = chan_mean(h32[nm]) - chan_mean(hq[nm])
        if mode == "scalar":
            d = np.array([float(d.mean())] * len(d))
        W["b_" + nm] = W["b_" + nm] + d
        adj[nm] = float(np.abs(d).max())
        hq = forward_hooks(W, Xcal)          # 重算, 让后续层用新的 bias

    # --- SE fc2 ---
    if se2_mode != "skip":
        for bi, nm in enumerate(SE2_ORDER):
            for _ in range(3):
                hq = forward_hooks(W, Xcal)
                pre2 = hq["tcn_%d_se_fc2" % bi]       # (N, OC)
                if se2_mode == "pre":
                    # 对齐 pre-sigmoid 均值 (线性, 一步到位)
                    d = chan_mean(h32[nm]) - chan_mean(pre2)
                    W["b_" + nm] = W["b_" + nm] + d
                    adj[nm] = float(np.abs(d).max())
                    if np.abs(d).max() < 1e-9:
                        break
                    continue

                # se2_mode == "out": 逐通道二分, 目标 = SE 缩放后输出的均值
                c2 = hq["tcn_%d_conv2" % bi]          # (N, OC, T)
                ref = h32["se_%d" % bi]
                target = chan_mean(ref) if ref.ndim == 3 else ref.mean(axis=0)
                nch = pre2.shape[1]
                delta = np.zeros(nch)
                for oc in range(nch):
                    c_oc = c2[:, oc, :]                # (N, T)
                    p_oc = pre2[:, oc]                 # (N,)

                    def f(d):
                        s = 1.0 / (1.0 + np.exp(-(p_oc + d)))
                        return float((c_oc * s[:, None]).mean()) - float(target[oc])

                    lo, hi = -20.0, 20.0
                    flo, fhi = f(lo), f(hi)
                    if flo > 0:
                        delta[oc] = lo
                    elif fhi < 0:
                        delta[oc] = hi
                    else:
                        for _ in range(40):
                            mid = 0.5 * (lo + hi)
                            if f(mid) < 0:
                                lo = mid
                            else:
                                hi = mid
                        delta[oc] = 0.5 * (lo + hi)
                W["b_" + nm] = W["b_" + nm] + delta
                adj[nm] = float(np.abs(delta).max())
                if np.abs(delta).max() < 1e-9:
                    break

    if verbose:
        print("  [偏置校正 mode=%s se2=%s] 各层 bias 调整量 (max|Δb| over 通道):"
              % (mode, se2_mode))
        for nm in CONV_ORDER + SE1_ORDER + SE2_ORDER:
            print("    %-16s max|Δb| = %.6f" % (nm, adj.get(nm, 0.0)))
    return W, adj
