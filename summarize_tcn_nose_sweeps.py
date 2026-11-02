#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCN-NoSE 23-dim 超参扫描整合 - 找到最佳参数组合

数据源 (由 train_tcn_nose_sweep_runner.py 产出):
  - train_tcn_nose_sweep_results_batch.json     (B sweep)
  - train_tcn_nose_sweep_results_ch.json        (ch sweep)
  - train_tcn_nose_sweep_results_dropout.json   (do sweep)
  - train_tcn_nose_sweep_results_epochs.json    (ep sweep)
  - train_tcn_nose_sweep_results_lr.json        (lr sweep)

每个 sweep 都是 5-seed 公平评估, 固定其他维度只扫一个维度。
与 TCN+SE sweep 完全对齐, 唯一差异是 NoSE (无 attention)。
"""
import os, json
import numpy as np

BASE = r"C:\work\Claude\Issue"

SWEEPS = {
    'Batch Size (B)':    ('train_tcn_nose_sweep_results_batch.json',     'summary_by_batch'),
    'Channels (ch)':    ('train_tcn_nose_sweep_results_ch.json',        'summary_by_ch'),
    'Dropout (do)':     ('train_tcn_nose_sweep_results_dropout.json',   'summary_by_dropout'),
    'Epochs (ep)':      ('train_tcn_nose_sweep_results_epochs.json',    'summary_by_epochs'),
    'Learning Rate':    ('train_tcn_nose_sweep_results_lr.json',        'summary_by_lr'),
}

# 每个 sweep 的固定 baseline (与 TCN+SE sweep 完全对齐)
FIXED = {
    'Batch Size (B)':    {'ch': 64, 'do': 0.3, 'ep': 20, 'lr': '5e-4'},
    'Channels (ch)':     {'B': 64, 'do': 0.3, 'ep': 20, 'lr': '5e-4'},
    'Dropout (do)':      {'B': 64, 'ch': 64, 'ep': 20, 'lr': '5e-4'},
    'Epochs (ep)':       {'B': 64, 'ch': 64, 'do': 0.3, 'lr': '5e-4'},
    'Learning Rate':     {'B': 64, 'ch': 64, 'do': 0.3, 'ep': 20},
}

print("="*100)
print(" " * 32 + "TCN-NoSE 23-dim 超参扫描整合 (5-seed 公平)")
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
    print(f"{'-'*12} {'-'*18} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {''-'*9}".replace("'-'*9", "'-'*9"))

    rows = []
    for k, v in summary.items():
        n_params = v.get('n_params', '-')
        f1m_str = f"{v['f1m_mean']:.4f} ± {v['f1m_std']:.4f}"
        rows.append((k, v, f1m_str))
        time_str = f"{v.get('train_time_s_mean', 0):.0f}" if v.get('train_time_s_mean') else '-'
        n_params_str = str(n_params) if n_params != '-' else '-'
        print(f"{k:<12} {f1m_str:>18} {v['binf1_mean']:>12.4f} {v['pr_mean']:>10.4f} {v['acc_mean']:>8.4f} {v.get('best_epoch_mean','-'):>8} {n_params_str:>10} {time_str:>9}")

    # 找最佳
    best = max(rows, key=lambda x: x[1]['f1m_mean'])
    best_per_dim[dim_name] = (best[0], best[1])
    print(f"\n  ★ 最佳 {dim_name} = {best[0]}    F1m = {best[1]['f1m_mean']:.4f} ± {best[1]['f1m_std']:.4f}")
    print(f"    BinF1 = {best[1]['binf1_mean']:.4f}    PR-AUC = {best[1]['pr_mean']:.4f}    Acc = {best[1]['acc_mean']:.4f}")

# ============================================================
# 推断最佳组合
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
# 与 TCN+SE SOTA 对比 + SE 边际贡献
# ============================================================
print(f"\n{'='*100}")
print(f" TCN-NoSE 最佳 vs TCN+SE SOTA 对比")
print(f"{'='*100}")
sota_path = os.path.join(BASE, "rerun_4metrics_5seed.json")
sota_se_path = os.path.join(BASE, "tcn_hparam_sweep_best.json")

if os.path.exists(sota_path):
    sota = json.load(open(sota_path, encoding='utf-8'))
    em = sota["ensemble_prob_mean"]
    print(f"  TCN+SE SOTA (5-seed ensemble, ch=32+do=0.1):")
    print(f"    F1m={em['Macro_F1']:.4f}  BinF1={em['F1-Score']:.4f}  Acc={em['Accuracy']:.4f}  PR-AUC={em.get('PR-AUC','-')}")
    print(f"    Recall={em['Recall']:.4f}  Precision={em['Precision']:.4f}")

if os.path.exists(sota_se_path):
    se_sweep = json.load(open(sota_se_path, encoding='utf-8'))
    # TCN+SE 推断最佳 (各 sweep 独立最优)
    se_best = se_sweep.get('inferred_best_combo_single_seed', {})
    print(f"\n  TCN+SE 推断最佳 (各 dim 独立, 单 seed): {se_best}")
    print(f"  TCN+SE 单 sweep 各 dim 最佳:")
    for dim_n, info in se_sweep.get('best_per_dimension', {}).items():
        print(f"    {dim_n:>18}: best={info['value']:>5}  F1m={info['f1m_mean']:.4f}±{info['f1m_std']:.4f}  BinF1={info['binf1_mean']:.4f}")

print(f"\n{'='*100}")
print(f" 关键结论")
print(f"{'='*100}")

# 计算 NoSE vs SE 边际
nose_combo_f1m = best_per_dim.get('Dropout (do)', (None, {}))[1].get('f1m_mean', None)
se_do = se_sweep.get('best_per_dimension', {}).get('Dropout (do)', {}).get('f1m_mean', None)
if nose_combo_f1m and se_do:
    delta = se_do - nose_combo_f1m
    print(f"  Δ F1m (SE vs NoSE @ dropout=best) = {delta:+.4f}")

nose_ch = best_per_dim.get('Channels (ch)', (None, {}))[1].get('f1m_mean', None)
se_ch = se_sweep.get('best_per_dimension', {}).get('Channels (ch)', {}).get('f1m_mean', None)
if nose_ch and se_ch:
    delta = se_ch - nose_ch
    print(f"  Δ F1m (SE vs NoSE @ ch=best)     = {delta:+.4f}")

print(f"\n  [NoSE BEST] 单 seed 推断最佳:")
for k, v in combo.items():
    print(f"    {k} = {v}")
print(f"  预期: F1m ≈ {sum(b['f1m_mean'] for _, b in best_per_dim.values()) / len(best_per_dim):.4f} (各 dim 平均值估计)")

# 保存整合结果
out = {
    'config': 'TCN-NoSE 23-dim hparam sweep',
    'data_sources': [(f, k) for f, k in SWEEPS.values()],
    'best_per_dimension': {n: {'value': v, **info} for n, (v, info) in best_per_dim.items()},
    'inferred_best_combo_single_seed': combo,
    'sota_se_5seed_ensemble': sota.get('ensemble_prob_mean', {}) if os.path.exists(sota_path) else None,
    'sota_se_single_seed_sweep': se_best if os.path.exists(sota_se_path) else None,
}
out_path = os.path.join(BASE, "tcn_nose_hparam_sweep_best.json")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out_path}")