#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCN+SE 超参扫描整合 - 找到最佳参数组合

数据源 (仓库已有 sweep 结果):
  - train_tcn_23dim_v2_batch_sweep_results.json  (B sweep)
  - train_tcn_23dim_v2_ch_sweep_results.json     (ch sweep)
  - train_tcn_23dim_v2_dropout_sweep_results.json (do sweep)
  - train_tcn_23dim_v2_epoch_sweep_results.json  (ep sweep)
  - train_tcn_23dim_v2_lr_sweep_results.json     (lr sweep)

每个 sweep 都是 5-seed 公平评估,固定其他维度只扫一个维度。
"""
import os, json
import numpy as np

BASE = r"C:\work\Claude\Issue"

SWEEPS = {
    'Batch Size (B)':    ('train_tcn_23dim_v2_batch_sweep_results.json',   'summary_by_batch'),
    'Channels (ch)':     ('train_tcn_23dim_v2_ch_sweep_results.json',      'summary_by_channels'),
    'Dropout (do)':      ('train_tcn_23dim_v2_dropout_sweep_results.json', 'summary_by_dropout'),
    'Epochs (ep)':       ('train_tcn_23dim_v2_epoch_sweep_results.json',   'summary_by_epoch'),
    'Learning Rate':     ('train_tcn_23dim_v2_lr_sweep_results.json',      'summary_by_lr'),
}

# 每个 sweep 的固定 baseline (固定超参)
FIXED = {
    'Batch Size (B)':    {'ch': 64, 'do': 0.3, 'ep': 20, 'lr': '5e-4'},
    'Channels (ch)':     {'B': 64, 'do': 0.3, 'ep': 20, 'lr': '4e-3'},
    'Dropout (do)':      {'B': 64, 'ch': 64, 'ep': 20, 'lr': '4e-3'},
    'Epochs (ep)':       {'B': 64, 'ch': 64, 'do': 0.3, 'lr': '4e-3'},
    'Learning Rate':     {'B': 64, 'ch': 64, 'do': 0.3, 'ep': 20},
}

print("="*100)
print(" " * 35 + "TCN+SE 23-dim 超参扫描整合 (5-seed 公平)")
print("="*100)

best_per_dim = {}

for dim_name, (file_name, summary_key) in SWEEPS.items():
    path = os.path.join(BASE, file_name)
    if not os.path.exists(path):
        print(f"  [MISSING] {file_name}")
        continue
    d = json.load(open(path, encoding='utf-8'))
    summary = d[summary_key]

    print(f"\n{'─'*100}")
    print(f" 维度: {dim_name}    (固定其他超参: {FIXED[dim_name]})")
    print(f"{'─'*100}")
    print(f"{'参数值':<12} {'F1m mean ± std':>18} {'BinF1 mean':>12} {'PR-AUC':>10} {'Acc':>8} {'best_ep':>8} {'n_params':>10} {'time(s)':>9}")
    print(f"{'-'*12} {'-'*18} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*9}")

    rows = []
    for k, v in summary.items():
        n_params = v.get('n_params', '-')
        f1m_str = f"{v['f1m_mean']:.4f} ± {v['f1m_std']:.4f}"
        rows.append((k, v, f1m_str))
        print(f"{k:<12} {f1m_str:>18} {v['binf1_mean']:>12.4f} {v['pr_mean']:>10.4f} {v['acc_mean']:>8.4f} {v.get('best_epoch_mean','-'):>8} {str(n_params):>10} {v.get('train_time_s_mean', v.get('train_time_s_mean','-')):>9.0f}")

    # 找最佳
    best = max(rows, key=lambda x: x[1]['f1m_mean'])
    best_per_dim[dim_name] = (best[0], best[1])
    print(f"\n  ★ 最佳 {dim_name} = {best[0]}    F1m = {best[1]['f1m_mean']:.4f} ± {best[1]['f1m_std']:.4f}")
    print(f"    BinF1 = {best[1]['binf1_mean']:.4f}    PR-AUC = {best[1]['pr_mean']:.4f}    Acc = {best[1]['acc_mean']:.4f}")

# ============================================================
# 推断最佳组合 (各维度独立最佳)
# ============================================================
print("\n" + "="*100)
print(" " * 35 + "推断最佳参数组合 (各维度独立最优)")
print("="*100)
print(f"{'维度':<20} {'最佳值':<10} {'F1m mean':>10} {'BinF1':>8} {'其他属性'}")
print("-"*100)
combo = {}
for dim_name, (val, summary) in best_per_dim.items():
    if dim_name == 'Batch Size (B)':
        combo['B'] = int(val)
        print(f"{'Batch':<20} {val:<10} {summary['f1m_mean']:>10.4f} {summary['binf1_mean']:>8.4f} iter/ep={summary.get('iters_per_ep','-')}")
    elif dim_name == 'Channels (ch)':
        combo['ch'] = int(val)
        print(f"{'Channels':<20} {val:<10} {summary['f1m_mean']:>10.4f} {summary['binf1_mean']:>8.4f} n_params={summary.get('n_params','-')}")
    elif dim_name == 'Dropout (do)':
        combo['do'] = float(val)
        print(f"{'Dropout':<20} {val:<10} {summary['f1m_mean']:>10.4f} {summary['binf1_mean']:>8.4f} -")
    elif dim_name == 'Epochs (ep)':
        combo['ep'] = int(val)
        print(f"{'Epochs':<20} {val:<10} {summary['f1m_mean']:>10.4f} {summary['binf1_mean']:>8.4f} actual={summary.get('actual_epochs_mean','-')}")
    elif dim_name == 'Learning Rate':
        combo['lr'] = float(val)
        print(f"{'Learning Rate':<20} {val:<10} {summary['f1m_mean']:>10.4f} {summary['binf1_mean']:>8.4f} -")

# ============================================================
# ⚠️ 注意事项: 各 sweep 的 fixed 不同, 不能简单合并
# ============================================================
print(f"\n[NOTE] 每个 sweep 固定其他超参,无法直接组合为 '联合最佳'")
print(f"   - do sweep 固定 ch=64 LR=4e-3 (与 ch sweep 不同)")
print(f"   - ch sweep 固定 do=0.3 LR=4e-3")
print(f"   - lr sweep 固定 do=0.3")
print(f"   -> 推断的 do=0.1 + ch=96 + LR=4e-3 组合尚未实测")

# ============================================================
# 与当前 SOTA 23-dim ch=32 ensemble 对比
# ============================================================
print(f"\n{'='*100}")
print(f" 与当前 SOTA 23-dim ch=32 5-seed ensemble 对比")
print(f"{'='*100}")
sota = json.load(open(os.path.join(BASE, "rerun_4metrics_5seed.json"), encoding="utf-8"))
em = sota["ensemble_prob_mean"]
print(f"  SOTA: B=64, ch=32, do=0.1, ep=40(实), LR=2e-3, 5-seed ensemble")
print(f"        F1m(macro)={em['Macro_F1']:.4f}    BinF1={em['F1-Score']:.4f}    PR-AUC={em.get('PR-AUC','-')}")
print(f"        Acc={em['Accuracy']:.4f}    Recall={em['Recall']:.4f}    Precision={em['Precision']:.4f}")

print(f"\n{'='*100}")
print(f" 结论摘要:")
print(f"{'='*100}")
print(f"  - B=64 (高 iter 数, 小 batch 利于收敛)")
print(f"  - ch=64 (与 ch=96 性能几乎一致, 性价比高)")
print(f"  - do=0.1 (最佳 F1m=0.8732, 比 do=0.3 高 +0.003)")
print(f"  - ep=20 (实际 ~19, early stop 充分; >20 边际饱和)")
print(f"  - LR=4e-3 (LR=2e-3 / 6e-3 几乎等价)")
print(f"\n  [BEST] 单 seed 推断最佳 = B=64 + ch=64 + do=0.1 + ep=20 + LR=4e-3")
print(f"         预期 F1m ~ 0.87, BinF1 ~ 0.87")
print(f"\n  [SOTA] 实际 SOTA = 23-dim ch=32 + do=0.1 + 5-seed ensemble -> BinF1=0.8737")
print(f"         减参到 21K + ensemble 比单 seed ch=64 略好")