"""
diag_hybrid_shift.py -- 诊断 Hybrid (INT8 权重/FP32 激活) 相对 FP32 的 logit 上偏机制.

观察到的现象 (compare_fp32_vs_hybrid.py):
  部署区 4664 窗, thr=0.5:
    Hybrid vs FP32 决策翻转 42 窗, 其中 39 窗"变正" / 3 窗"变负"  -> 单向上偏
    ΔFPR = +0.98pp  (bootstrap CI [+0.0057, +0.0142], 显著)
    ΔF1  = -0.0008  (bootstrap CI [-0.0035, +0.0021], 不可区分)

假设: **权重量化噪声 -> pre-ReLU 激活方差变大 -> ReLU 之后激活整体抬高 -> logit 上偏**

  conv 输出 y = sum_ic w_ic * x_ic.  权重量化给 w 加了零均值噪声 eps:
      y' = y + sum_ic eps_ic * x_ic
  => y' 的方差比 y 大 (方差 = sum_ic eps_ic^2 * x_ic^2 + 交叉项)
  => ReLU 之后: E[ReLU(y')] 随 Var(y') 单调增 (正半轴上的部分被放大)
  => GAP 变大 -> logit 变大  => 阈值判定整体往"正"的方向挪

  这个机制能解释:
    1. 为什么是单向上偏 (39 正 / 3 负), 而不是对称噪声的随机翻转
    2. 为什么差异集中在决策边界带 (饱和窗 prob 不变)
    3. 为什么去掉激活量化没用 (瓶颈在权重量化)

本脚本做三件事:
  [1] 测 logit 偏移的分布 (均值/中位数/分位数), 确认是否系统性为正
  [2] 测 pre-ReLU 激活的幅度在 FP32 vs Hybrid 下的变化, 验证方差膨胀假设
  [3] 做一个反事实实验: 把权重量化噪声"注入"FP32 模型 (FP32 权重 + 人工加噪),
      看能否复现同样的上偏 -> 闭环证明机制
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim_hybrid_kernel import (DIL, SE_H, T, conv1d_same_batch, load_weights,
                               se_block, t_in_index)

WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
DEPLOY_FROM = 200001


def forward_debug(W, X):
    """跟 sim_hybrid_kernel.forward 同语义, 但额外返回 pre-ReLU 激活统计."""
    x = np.asarray(X, dtype=np.float32).astype(np.float64)
    pre = []          # 每层 conv 输出 (ReLU 之前)
    post = []         # 每层 conv 输出 (ReLU 之后)

    def conv(inp, w, b, dil, tag):
        O, _, K = w.shape
        Tx = inp.shape[2]
        out = np.broadcast_to(np.asarray(b, dtype=np.float64)[None, :, None],
                              (inp.shape[0], O, Tx)).copy()
        for k in range(K):
            idx = t_in_index(Tx, K, dil, k)
            valid = ((idx >= 0) & (idx < Tx)).astype(np.float64)
            ic = np.clip(idx, 0, Tx - 1)
            xs = inp[:, :, ic] * valid[None, None, :]
            out += np.einsum("oc,nct->not", np.asarray(w[:, :, k], dtype=np.float64), xs)
        pre.append(out.copy())
        r = np.maximum(out, 0)
        post.append(r.copy())
        return r, out

    res, _ = conv(x, W["w_tcn_0_residual"], W["b_tcn_0_residual"], 1, "res0")
    c1, _ = conv(x, W["w_tcn_0_conv1"], W["b_tcn_0_conv1"], DIL[0], "c1_0")
    c2, _ = conv(c1, W["w_tcn_0_conv2"], W["b_tcn_0_conv2"], DIL[0], "c2_0")
    cur = se_block(c2, W, 0)
    cur = np.maximum(cur + res, 0)
    for bi in (1, 2):
        res = cur
        c1, _ = conv(cur, W["w_tcn_%d_conv1" % bi], W["b_tcn_%d_conv1" % bi], DIL[bi], "c1_%d" % bi)
        c2, _ = conv(c1, W["w_tcn_%d_conv2" % bi], W["b_tcn_%d_conv2" % bi], DIL[bi], "c2_%d" % bi)
        cur = se_block(c2, W, bi)
        cur = np.maximum(cur + res, 0)

    g = cur.sum(axis=2) / T
    h = np.maximum(g @ np.asarray(W["fc1"], dtype=np.float64).T
                   + np.asarray(W["b_fc1"], dtype=np.float64), 0.0)
    logit = (h @ np.asarray(W["fc2"], dtype=np.float64).T
             + np.asarray(W["b_fc2"], dtype=np.float64))[:, 0]
    return logit, pre, post


def noise_injected_weights(W32, Why, rng, alpha=1.0):
    """反事实: FP32 权重 + (Hybrid 量化噪声)*alpha.

    noise = w_hybrid_dequant - w_fp32 ; 注入后 w' = w_fp32 + alpha*noise.
    alpha=1 等价于 Hybrid 的权重; alpha=0 等价于 FP32.
    """
    W = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in W32.items()}
    for k in W32:
        if k.startswith("w_") and k in Why:
            W[k] = W32[k] + alpha * (Why[k] - W32[k])
    return W


def main():
    d = np.load(WIN_NPZ)
    X = d["X"].astype(np.float32)
    frames = d["frames"]
    dep = frames >= DEPLOY_FROM
    Xd = X[dep]

    W32 = load_weights("fp32")
    Why = load_weights("int8")

    print("[1] logit 偏移 (部署区 %d 窗)" % int(dep.sum()))
    lg32, pre32, post32 = forward_debug(W32, Xd)
    lg_h, pre_h, post_h = forward_debug(Why, Xd)
    d_lg = lg_h - lg32
    print("    Δlogit: 均值=%+.4f  中位数=%+.4f  std=%.4f  P(>0)=%.4f"
          % (d_lg.mean(), np.median(d_lg), d_lg.std(), (d_lg > 0).mean()))
    print("    分位数: p1=%+.3f p10=%+.3f p25=%+.3f p75=%+.3f p90=%+.3f p99=%+.3f"
          % tuple(np.percentile(d_lg, [1, 10, 25, 75, 90, 99])))
    print("    logit 幅度对比: FP32 mean|logit|=%.4f  Hybrid mean|logit|=%.4f"
          % (np.abs(lg32).mean(), np.abs(lg_h).mean()))
    print("    正 logit 比例: FP32=%.4f  Hybrid=%.4f" % ((lg32 > 0).mean(), (lg_h > 0).mean()))

    print("\n[2] pre-ReLU 激活幅度 (方差膨胀假设)")
    for i, (a, b) in enumerate(zip(pre32, pre_h)):
        print("    层%d  FP32: mean|x|=%.5f  std=%.5f  P(x>0)=%.4f  |  Hybrid: mean|x|=%.5f  std=%.5f  P(x>0)=%.4f"
              % (i, np.abs(a).mean(), a.std(), (a > 0).mean(),
                 np.abs(b).mean(), b.std(), (b > 0).mean()))
    print("    post-ReLU (GAP 输入) 总量:")
    for i, (a, b) in enumerate(zip(post32, post_h)):
        print("      层%d  FP32 mean=%.5f  Hybrid mean=%.5f  相对变化=%+.3f%%"
              % (i, a.mean(), b.mean(), 100 * (b.mean() / a.mean() - 1)))

    print("\n[3] 反事实: FP32 权重 + alpha * 量化噪声")
    print("    (alpha=1 应完全复现 Hybrid)")
    rng = np.random.default_rng(0)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        Wa = noise_injected_weights(W32, Why, rng, alpha)
        lg_a, _, _ = forward_debug(Wa, Xd)
        d_a = lg_a - lg32
        up = int(((lg_a > 0) & (lg32 <= 0)).sum())
        dn = int(((lg_a <= 0) & (lg32 > 0)).sum())
        print("    alpha=%.2f  Δlogit 均值=%+.4f 中位数=%+.4f  变正=%d 变负=%d"
              % (alpha, d_a.mean(), np.median(d_a), up, dn))

    print("\n[4] 对照: 权重量化噪声的相对幅度")
    for k in ("w_tcn_0_conv1", "w_tcn_1_conv2", "w_tcn_2_conv2"):
        a = Why[k].astype(np.float64).ravel()
        b = W32[k].astype(np.float64).ravel()
        rel = np.abs(a - b) / (np.abs(b).max() + 1e-12)
        print("    %-14s 量化相对误差: mean=%.5f  max=%.5f  (int8 absmax 量化理论上限 ~1/254=%.5f)"
              % (k, rel.mean(), rel.max(), 1 / 254))


if __name__ == "__main__":
    main()
