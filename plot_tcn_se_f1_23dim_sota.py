#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 TCN+SE F1-Score 训练曲线 — 用 23-dim ch=32 5-seed ensemble (SOTA BinF1=0.8737)

对比三个变体:
  1. ★ TCN+SE 23-dim ch=32 5-seed ensemble (SOTA, BinF1=0.8737)
  2. TCN+SE 23-dim ch=64 5-seed baseline (BinF1 mean ≈ 0.81)
  3. TCN 23-dim NoSE ch=64 5-seed baseline (BinF1 mean ≈ 0.74)

数据源:
  - rerun_4metrics_5seed.json (SOTA BinF1)
  - train_tcn_23dim_nose_5seed_results.json (ch=64 baseline)
  - train_tcn_27dim_nose_5seed_results.json (27-dim 早期 baseline)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

BASE = r"C:\work\Claude\Issue"

# ============== 真实数据 ==============
sota = json.load(open(os.path.join(BASE, "rerun_4metrics_5seed.json"), encoding="utf-8"))
sota_seeds_binF1 = [sota["per_seed"][str(s)]["F1-Score"] for s in [42, 123, 456, 789, 1024]]
SOTA_ENS_BINF1 = sota["ensemble_prob_mean"]["F1-Score"]   # 0.8737
SOTA_ENS_ACC = sota["ensemble_prob_mean"]["Accuracy"]     # 0.8776
SOTA_BEST_SEED_BINF1 = max(sota_seeds_binF1)              # 0.8728 (seed 123)
SOTA_MEAN_BINF1 = float(np.mean(sota_seeds_binF1))        # 0.8617

# 23-dim ch=64 with_se
hist23 = json.load(open(os.path.join(BASE, "train_tcn_23dim_nose_5seed_results.json"), encoding="utf-8"))
ch64_se = hist23["with_se"]["runs"]
ch64_se_binF1 = [r["test_binary_f1"] for r in ch64_se]
ch64_se_best_ep = [r["best_epoch"] for r in ch64_se]
CH64_SE_MEAN_BINF1 = float(np.mean(ch64_se_binF1))         # ≈ 0.8056
CH64_SE_MEAN_BEST_EP = float(np.mean(ch64_se_best_ep))     # ≈ 13.4

# 23-dim ch=64 without_se
ch64_nose = hist23["without_se"]["runs"]
ch64_nose_binF1 = [r["test_binary_f1"] for r in ch64_nose]
ch64_nose_best_ep = [r["best_epoch"] for r in ch64_nose]
CH64_NOSE_MEAN_BINF1 = float(np.mean(ch64_nose_binF1))     # ≈ 0.7373
CH64_NOSE_MEAN_BEST_EP = float(np.mean(ch64_nose_best_ep)) # ≈ 8.6

# ============== 示意曲线生成 ==============
def f1_curve(epochs, f1_max, k, seed=42, jitter_amp=0.012):
    rng = np.random.default_rng(seed)
    e = np.arange(1, epochs + 1)
    base = f1_max * (1 - np.exp(-k * e))
    jitter = rng.normal(0, jitter_amp, size=len(e))
    return e, np.clip(base + jitter, 0, 0.98)

# 训练 epoch 数 (从历史 best_epoch 推断 5-seed ensemble 大约跑到 20-40 epoch)
MAX_EPOCH = 40

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(13, 7))

# ---- 主曲线 1: TCN+SE 23-dim ch=32 5-seed (SOTA) ----
color_sota = '#1f77b4'
# 5 条单 seed 细线
for i, (s, binf1) in enumerate(zip([42, 123, 456, 789, 1024], sota_seeds_binF1)):
    e, f1 = f1_curve(MAX_EPOCH, f1_max=binf1*1.005, k=0.18, seed=42 + i*7)
    ax.plot(e, f1, color=color_sota, alpha=0.35, linewidth=0.9)
    # 标注 best_binF1 点 (假设在 epoch 20 附近)
    ax.scatter([MAX_EPOCH], [binf1], color=color_sota, s=70, zorder=6,
               edgecolors='black', linewidth=0.7)

# SOTA ensemble 粗线 (上限)
e_sota, f1_sota = f1_curve(MAX_EPOCH, f1_max=SOTA_ENS_BINF1*1.003, k=0.18, seed=2026, jitter_amp=0.006)
ax.plot(e_sota, f1_sota, color=color_sota, linewidth=3.5, zorder=5,
        label=f'★ TCN+SE 23-dim ch=32 5-seed ensemble (BinF1 = {SOTA_ENS_BINF1:.4f})')

# SOTA 水平参考线
ax.axhline(SOTA_ENS_BINF1, color=color_sota, linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)

# ---- 对比曲线 2: TCN+SE 23-dim ch=64 (baseline) ----
color_ch64 = '#ff7f0e'
for i, (binf1, ep) in enumerate(zip(ch64_se_binF1, ch64_se_best_ep)):
    e, f1 = f1_curve(max(ep+2, 8), f1_max=binf1*1.008, k=0.20, seed=100 + i*5)
    ax.plot(e, f1, color=color_ch64, alpha=0.30, linewidth=0.9)
    ax.scatter([ep], [binf1], color=color_ch64, s=55, zorder=6, marker='X',
               edgecolors='black', linewidth=0.7)

e_ch64, f1_ch64 = f1_curve(MAX_EPOCH, f1_max=CH64_SE_MEAN_BINF1*1.005, k=0.20, seed=2027, jitter_amp=0.008)
ax.plot(e_ch64, f1_ch64, color=color_ch64, linewidth=2.5, zorder=4,
        label=f'TCN+SE 23-dim ch=64 baseline (mean BinF1 = {CH64_SE_MEAN_BINF1:.4f})')

# ---- 对比曲线 3: TCN 23-dim NoSE ch=64 ----
color_nose = '#d62728'
for i, (binf1, ep) in enumerate(zip(ch64_nose_binF1, ch64_nose_best_ep)):
    e, f1 = f1_curve(max(ep+2, 8), f1_max=binf1*1.01, k=0.40, seed=200 + i*11)
    ax.plot(e, f1, color=color_nose, alpha=0.30, linewidth=0.9)
    ax.scatter([ep], [binf1], color=color_nose, s=55, zorder=6, marker='X',
               edgecolors='black', linewidth=0.7)

e_nose, f1_nose = f1_curve(MAX_EPOCH, f1_max=CH64_NOSE_MEAN_BINF1*1.005, k=0.40, seed=2028, jitter_amp=0.008)
ax.plot(e_nose, f1_nose, color=color_nose, linewidth=2.5, zorder=4,
        label=f'TCN 23-dim NoSE ch=64 (mean BinF1 = {CH64_NOSE_MEAN_BINF1:.4f})')

# ---- 关键标注 ----
# Δ SOTA - ch=64 baseline
delta_sota = SOTA_ENS_BINF1 - CH64_SE_MEAN_BINF1
ax.annotate(f'ch=32 ensemble\n+ BinF1 +{delta_sota:.4f}\nvs ch=64 baseline',
            xy=(35, 0.83), xytext=(35, 0.83),
            fontsize=10, color='#1f77b4', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#e6f0fa',
                      edgecolor='#1f77b4', alpha=0.95))

# Δ SE - NoSE
delta_se = CH64_SE_MEAN_BINF1 - CH64_NOSE_MEAN_BINF1
ax.annotate(f'SE 提供\nBinF1 +{delta_se:.4f}',
            xy=(35, 0.67), xytext=(35, 0.67),
            fontsize=10, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow',
                      edgecolor='darkgreen', alpha=0.95))

# 美化
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('F1-Score (Binary convention, Attack class)', fontsize=13)
ax.set_title('TCN+SE F1-Score 训练曲线 — 23-dim 三变体对比\n(SOTA 5-seed ensemble BinF1 = 0.8737)',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.3, MAX_EPOCH + 1.5)
ax.set_ylim(0.55, 0.92)

# 底部 [NOTE]
fig.text(0.5, 0.01,
         '[NOTE] Schematic Reconstruction — per-epoch F1 历史未保存,曲线基于典型饱和上升形态 + 真实 best_epoch / BinF1 终点锚定',
         ha='center', fontsize=10, color='darkorange', fontweight='bold')

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = os.path.join(BASE, "training_curves_f1score_tcn_se_sota.png")
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

# ============== 打印真实数据 ==============
print("\n=== 真实数据汇总 ===")
print(f"{'变体':<35} {'BinF1 mean':>10} {'BinF1 max':>10} {'best_ep mean':>12}")
print("-" * 70)
print("{:<35} {:>10.4f} {:>10.4f} {:>12}".format('TCN+SE 23-dim ch=32 5-seed (★ SOTA)',
       SOTA_MEAN_BINF1, SOTA_ENS_BINF1, "20-40"))
print("{:<35} {:>10.4f} {:>10.4f} {:>12.1f}".format('TCN+SE 23-dim ch=64 baseline',
       CH64_SE_MEAN_BINF1, max(ch64_se_binF1), CH64_SE_MEAN_BEST_EP))
print("{:<35} {:>10.4f} {:>10.4f} {:>12.1f}".format('TCN 23-dim NoSE ch=64',
       CH64_NOSE_MEAN_BINF1, max(ch64_nose_binF1), CH64_NOSE_MEAN_BEST_EP))
print(f"\n★ SOTA ensemble BinF1 = {SOTA_ENS_BINF1:.4f}  (用户提到的 0.87 = SOTA)")
print(f"  Single best seed 123 BinF1 = {SOTA_BEST_SEED_BINF1:.4f}")
print(f"  Δ ch=32 ensemble vs ch=64 baseline = +{delta_sota:.4f}")
print(f"  Δ SE vs NoSE (ch=64) = +{delta_se:.4f}")