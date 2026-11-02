#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 TCN vs TCN+SE 的 F1-Score 训练曲线 (单图聚焦)

数据源:
  - train_tcn_27dim_nose_5seed_results.json (27-dim × {SE, NoSE} × 5-seed 公平消融)
  - rerun_4metrics_5seed.json (当前 SOTA 23-dim ch=32 5-seed ensemble BinF1=0.8737)

⚠️ Schematic Reconstruction: per-epoch F1 历史未保存,曲线基于典型收敛形态
   (F1 ~ F1_max * (1 - exp(-k*epoch))) + 真实 best_epoch/F1m 锚定终点

口径: 训练时使用 val_macro_f1 (与 train_tcn_*.py 一致); F1-Score 默认 Macro
       上限参考线用 BinF1 (Binary convention, 论文标准)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

BASE = r"C:\work\Claude\Issue"

# ============== 历史真实数据 ==============
hist27 = json.load(open(os.path.join(BASE, "train_tcn_27dim_nose_5seed_results.json"), encoding="utf-8"))
with_se = hist27["with_se"]["runs"]
without_se = hist27["without_se"]["runs"]

F1M_SE = hist27["with_se"]["mean_f1m"]            # 0.8205
F1M_NOSE = hist27["without_se"]["mean_f1m"]       # 0.7612
DELTA_F1M = F1M_SE - F1M_NOSE                     # +0.0594

# SOTA 23-dim 5-seed ensemble BinF1 (Binary convention, 上限参考)
sota = json.load(open(os.path.join(BASE, "rerun_4metrics_5seed.json"), encoding="utf-8"))
SOTA_BINF1 = sota["ensemble_prob_mean"]["F1-Score"]   # 0.8737

# ============== 示意曲线生成 ==============
def f1_curve(epochs, f1_max, k, seed=42, jitter_amp=0.012):
    """F1 饱和上升曲线: f1_max * (1 - exp(-k*epoch)) + 小幅 jitter"""
    rng = np.random.default_rng(seed)
    e = np.arange(1, epochs + 1)
    base = f1_max * (1 - np.exp(-k * e))
    jitter = rng.normal(0, jitter_amp, size=len(e))
    return e, np.clip(base + jitter, 0, 0.98)

MAX_EPOCH = 20

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(11, 6.5))

# 5 条细线 + 1 条平均粗线 - TCN+SE
for i, r in enumerate(with_se):
    best_ep = r["best_epoch"]; best_f1 = r["test_macro_f1"]
    # 调整 k 让曲线在 best_epoch 接近 best_f1 (粗略)
    k = 0.20 if best_f1 > 0.81 else 0.18
    e, f1 = f1_curve(MAX_EPOCH, f1_max=best_f1*1.008, k=k, seed=42 + i*7)
    ax.plot(e, f1, color='#1f77b4', alpha=0.30, linewidth=0.9)
    ax.scatter([best_ep], [best_f1], color='#1f77b4', s=55, zorder=5,
               edgecolors='black', linewidth=0.7, alpha=0.85)

# 5 条细线 + 1 条平均粗线 - TCN 无 SE
for i, r in enumerate(without_se):
    best_ep = r["best_epoch"]; best_f1 = r["test_macro_f1"]
    k = 0.45 if best_f1 > 0.75 else 0.55
    e, f1 = f1_curve(MAX_EPOCH, f1_max=best_f1*1.01, k=k, seed=99 + i*11)
    ax.plot(e, f1, color='#d62728', alpha=0.30, linewidth=0.9)
    ax.scatter([best_ep], [best_f1], color='#d62728', s=55, zorder=5,
               edgecolors='black', linewidth=0.7, marker='X', alpha=0.85)

# 平均线 (粗) - 用真实 mean F1m 锚定
e_se, f1_se = f1_curve(MAX_EPOCH, f1_max=F1M_SE*1.005, k=0.20, seed=2026, jitter_amp=0.008)
ax.plot(e_se, f1_se, color='#1f77b4', linewidth=3.0, zorder=4,
        label=f'TCN+SE (5-seed mean F1m = {F1M_SE:.4f})')

e_nose, f1_nose = f1_curve(MAX_EPOCH, f1_max=F1M_NOSE*1.005, k=0.45, seed=2027, jitter_amp=0.008)
ax.plot(e_nose, f1_nose, color='#d62728', linewidth=3.0, zorder=4,
        label=f'TCN 无 SE (5-seed mean F1m = {F1M_NOSE:.4f})')

# SOTA 上限参考线 (23-dim ch=32 5-seed ensemble BinF1)
ax.axhline(SOTA_BINF1, color='green', linestyle='--', linewidth=1.8, alpha=0.8, zorder=3,
           label=f'当前 SOTA 23-dim ch=32 ensemble BinF1 = {SOTA_BINF1:.4f}')

# Δ 标注
ax.annotate(f'Δ F1m = +{DELTA_F1M:.4f}\n(SE 提供)',
            xy=(14, 0.62), xytext=(14, 0.62),
            fontsize=12, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow',
                      edgecolor='darkgreen', alpha=0.95))

# 美化
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('F1-Score (Validation, Macro convention)', fontsize=13)
ax.set_title('TCN+SE vs TCN — F1-Score 训练曲线 (27-dim, 5-seed 公平消融)',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.3, MAX_EPOCH + 0.5)
ax.set_ylim(0.50, 0.90)

# 在底部加 SCHEMATIC 警示
fig.text(0.5, 0.01,
         '[NOTE] Schematic Reconstruction — 训练历史未保存,曲线基于典型饱和上升形态 + 真实 best_epoch / F1m 终点锚定',
         ha='center', fontsize=10, color='darkorange', fontweight='bold')

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = os.path.join(BASE, "training_curves_f1score_tcn_vs_tcnse.png")
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

# ============== 同步输出真实 best_epoch 表 ==============
print("\n=== 真实数据 (来自 train_tcn_27dim_nose_5seed_results.json) ===")
print(f"{'变体':<12} {'best_ep':>8} {'F1m':>8} {'BinF1':>8} {'PR-AUC':>8} {'Acc':>8}")
print("-" * 60)
for r in with_se:
    print("{:<12} {:>8} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        'TCN+SE', r['best_epoch'], r['test_macro_f1'], r['test_binary_f1'],
        r['test_pr_auc'], r['test_accuracy']))
for r in without_se:
    print("{:<12} {:>8} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        'TCN 无 SE', r['best_epoch'], r['test_macro_f1'], r['test_binary_f1'],
        r['test_pr_auc'], r['test_accuracy']))

print(f"\n当前 SOTA 23-dim ch=32 5-seed ensemble BinF1 = {SOTA_BINF1:.4f} (上限参考线)")