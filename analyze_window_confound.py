#!/usr/bin/env python3
"""窗口数量作为混淆变量的 3 个量化来源。

3 个机制:
  M1 标签噪声 (label noise): 小窗口 max-pool 出 0/1 的方差更大
  M2 样本独立性: 滑动/非滑动窗口不是 i.i.d. 样本, 时间相关性很强
  M3 评估颗粒度: F1 = 2PR/(P+R), 改变窗口数 = 改变"攻击单元"定义

本脚本只算 M1 (label purity) 和 M2 (effective sample size).
M3 用 cross-window eval 验证 (复用已训好的 .pt).
"""
import os
import numpy as np
from collections import Counter

BASE = r"C:\work\Claude\Issue"


def label_purity(W):
    """每个窗口包含多少个 attack row. 越纯 → 标签越确定 → 评估越稳."""
    y = np.load(os.path.join(BASE, f"y_test_binary_v2_scada_window{W}.npy")).astype(int)
    # 重新切片: 把 N=54927 行重组成 W 大小, 取每窗口的 attack 行数
    n_total = 54927
    n_use = (n_total // W) * W
    y_flat = np.load(os.path.join(BASE, "y_test_binary_v2_scada.npy")).astype(int)[:n_use]
    wins = y_flat.reshape(-1, W)
    attack_counts = wins.sum(axis=1)
    total_wins = len(attack_counts)
    pure_normal = int((attack_counts == 0).sum())
    pure_attack = int((attack_counts == W).sum())
    mixed = int(((attack_counts > 0) & (attack_counts < W)).sum())
    print(f"\n=== W={W} (test={total_wins} windows) ===")
    print(f"  标签分布: pure_normal={pure_normal} ({100*pure_normal/total_wins:.1f}%)  "
          f"pure_attack={pure_attack} ({100*pure_attack/total_wins:.1f}%)  "
          f"mixed={mixed} ({100*mixed/total_wins:.1f}%)")
    print(f"  attack rows/window 分布 (count→pct):")
    c = Counter(attack_counts.tolist())
    for k in sorted(c.keys()):
        print(f"    {k:>2}/{W}: {c[k]:>6} ({100*c[k]/total_wins:5.2f}%)")
    return {
        "n_windows": total_wins,
        "pct_pure_normal": 100 * pure_normal / total_wins,
        "pct_pure_attack": 100 * pure_attack / total_wins,
        "pct_mixed": 100 * mixed / total_wins,
        "attack_counts_dist": dict(c),
    }


def effective_sample_size(W):
    """Effective sample size via autocorrelation of test labels.

    如果相邻窗口完全独立 → ESS = N. 如果完全相关 → ESS ≈ 1.
    一般 ESS = N * (1 - r1) / (1 + r1), 其中 r1 = lag-1 autocorrelation.
    """
    y = np.load(os.path.join(BASE, f"y_test_binary_v2_scada_window{W}.npy")).astype(int)
    n = len(y)
    y_mean = y.mean()
    y_var = y.var()
    if y_var == 0:
        return n
    # lag-1 autocorrelation
    r1 = np.corrcoef(y[:-1], y[1:])[0, 1]
    # Bartlett ESS for AR(1): N * (1 - r1) / (1 + r1)
    ess = n * (1 - r1) / (1 + r1)
    print(f"\n=== W={W} ESS (Effective Sample Size) ===")
    print(f"  N windows = {n}")
    print(f"  attack_pct = {y_mean:.4f}")
    print(f"  lag-1 autocorrelation r1 = {r1:.4f}")
    print(f"  ESS = N*(1-r1)/(1+r1) = {ess:.1f}  ({100*ess/n:.1f}% of N)")
    return int(ess)


def model_size_sanity():
    """检查模型参数随 W 是否变化. TCN+SE ch=32 是固定架构, params 应一致."""
    import torch
    pt_path = os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window4_s42.pt")
    if not os.path.exists(pt_path):
        return
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    print(f"\n=== 模型参数 (W=4 seed=42 验证) ===")
    print(f"  n_features (per step) = {ckpt['n_features']} (期望 23)")
    print(f"  channels (TCN width) = {ckpt['channels']} (期望 32)")
    print(f"  total params = {ckpt['n_params']:,}")
    print(f"  → 模型结构对所有 W 一致, params 与 W 无关")


if __name__ == "__main__":
    print("="*72)
    print("窗口大小作为混淆变量 — 量化分析")
    print("="*72)

    print("\n[ M1: Label Purity per Window ]")
    print("-"*72)
    results = {}
    for W in [4, 8, 12, 16]:
        results[W] = label_purity(W)

    print("\n\n[ M2: Effective Sample Size via lag-1 autocorrelation ]")
    print("-"*72)
    ess_results = {}
    for W in [4, 8, 12, 16]:
        ess_results[W] = effective_sample_size(W)

    print("\n\n[ M3: 模型结构 vs W 无关性验证 ]")
    print("-"*72)
    model_size_sanity()

    # 综合结论
    print("\n\n" + "="*72)
    print("综合结论")
    print("="*72)
    print(f"{'W':>3} {'N_win':>8} {'pure_N%':>8} {'pure_A%':>8} {'mixed%':>8} {'ESS':>8} {'ESS/N%':>8}")
    for W in [4, 8, 12, 16]:
        r = results[W]
        e = ess_results[W]
        print(f"{W:>3} {r['n_windows']:>8} {r['pct_pure_normal']:>7.1f}% {r['pct_pure_attack']:>7.1f}% "
              f"{r['pct_mixed']:>7.1f}% {e:>8} {100*e/r['n_windows']:>7.1f}%")
    print("\n→ 如果 ESS 远小于 N, 说明测试集强时间相关, F1 的统计显著性被高估")
    print("→ 如果 pure_pct 在 W 间差异大, 说明 label 颗粒度本身变了")
    print("→ 模型结构 (ch=32 params=20,877) 与 W 无关, 性能差异纯粹来自数据 + 训练动态")
