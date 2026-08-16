#!/usr/bin/env python3
"""Cross-Model Leaderboard — 所有历史 best 结果排序

从 4 个 ensemble + LR=2e-3 5-seed + batch sweep + combos_64 汇总成单张排名表。
"""

import json, os

BASE = r"C:\work\Claude\Issue"
results = []

# 1) LR=2e-3 5-seed ensemble (单架构 self-ensemble, 5 models, B=128, ep=40)
d = json.load(open(os.path.join(BASE, "processed_meta_tcn_v4_se_lr2e3_5seed.json")))
em = d['ensemble_metrics']['test_at_thr']
results.append({
    'category': 'Self-Ensemble',
    'name': 'TCN+SE LR=2e-3 5-seed ensemble (ch=64, B=128, ep=40)',
    'F1m': em['macro_f1'], 'PR_AUC': em['pr_auc'],
    'BinF1': em['binary_f1'], 'Acc': em['accuracy'], 'ROC_AUC': em['roc_auc'],
    'thr': d['best_threshold'], 'n_seeds': 5, 'n_models': 5,
    'src': 'processed_meta_tcn_v4_se_lr2e3_5seed.json'
})

# 2) Batch sweep
d2 = json.load(open(os.path.join(BASE, "train_tcn_23dim_v2_batch_sweep_results.json")))
for b, s in d2['summary_by_batch'].items():
    results.append({
        'category': 'Single-Model',
        'name': f'TCN+SE 23-dim B={b} 5-seed (this work)',
        'F1m': s['f1m_mean'], 'PR_AUC': s['pr_mean'],
        'BinF1': s['binf1_mean'], 'Acc': s['acc_mean'], 'ROC_AUC': s['roc_mean'],
        'thr': 0.5, 'n_seeds': 5, 'n_models': 1,
        'src': 'batch sweep'
    })

# 3) Combos_64 - 23-dim all configs (top entries only)
d3 = json.load(open(os.path.join(BASE, "train_tcn_combos_64_results.json")))
cfgs_23 = [c for c in d3['configs'] if c.get('n_features') == 23]
cfgs_23.sort(key=lambda c: -c.get('mean_f1m', 0))
for c in cfgs_23[:5]:  # top 5
    drops = c.get('drop_features', [])
    results.append({
        'category': 'Single-Model',
        'name': f"TCN+SE 23-dim (-{','.join(drops)}) 5-seed",
        'F1m': c.get('mean_f1m', 0), 'PR_AUC': c.get('mean_pr_auc', 0),
        'BinF1': c.get('mean_binary_f1', 0), 'Acc': c.get('mean_accuracy', 0),
        'ROC_AUC': c.get('mean_roc_auc', 0),
        'thr': 0.5, 'n_seeds': 5, 'n_models': 1,
        'src': 'combos_64'
    })

# 4) Combos_64 - 27-dim baseline
for c in d3['configs']:
    if c.get('n_features') == 27:
        results.append({
            'category': 'Single-Model',
            'name': 'TCN+SE 27-dim baseline 5-seed',
            'F1m': c.get('mean_f1m', 0), 'PR_AUC': c.get('mean_pr_auc', 0),
            'BinF1': c.get('mean_binary_f1', 0), 'Acc': c.get('mean_accuracy', 0),
            'ROC_AUC': c.get('mean_roc_auc', 0),
            'thr': 0.5, 'n_seeds': 5, 'n_models': 1,
            'src': 'combos_64'
        })
        break

# 5) 4 ensembles - stacking best + uniform best + weighted best
for v in ['v1','v2','v3','v4']:
    d = json.load(open(os.path.join(BASE, f"processed_meta_ensemble_{v}.json")))
    for kind in ['stacking', 'uniform', 'weighted']:
        if kind in d and d[kind]:
            best = max(d[kind], key=lambda x: x.get('test_f1m', 0))
            results.append({
                'category': f'Ensemble-{kind[:3].upper()}',
                'name': f'ENSEMBLE_{v} {kind} best ({best["members"]})',
                'F1m': best['test_f1m'], 'PR_AUC': best['test_pra'],
                'BinF1': best['test_f1b'], 'Acc': best['test_acc'],
                'ROC_AUC': best['test_roc'],
                'thr': best['best_thr'], 'n_seeds': 1, 'n_models': len(d['models']),
                'src': f'ensemble_{v}'
            })

# Sort by F1m desc
results.sort(key=lambda x: -x['F1m'])

# Print leaderboard
print('=' * 145)
print(f'{"Rank":>4}  {"Category":<18} {"Name":<70} {"F1m":>8} {"PR-AUC":>8} {"BinF1":>8} {"Acc":>8} {"ROC":>8} {"thr":>5}  {"#seed":>5}  {"#mdl":>4}')
print('=' * 145)
for i, r in enumerate(results, 1):
    bf = f"{r['BinF1']:.4f}" if r['BinF1'] is not None else '  N/A'
    ac = f"{r['Acc']:.4f}" if r['Acc'] is not None else '  N/A'
    rc = f"{r['ROC_AUC']:.4f}" if r['ROC_AUC'] is not None else '  N/A'
    name_short = r['name'][:70]
    print(f"{i:>4}  {r['category']:<18} {name_short:<70} {r['F1m']:>8.4f} {r['PR_AUC']:>8.4f} {bf:>8} {ac:>8} {rc:>8} {r['thr']:>5.2f}  {r['n_seeds']:>5d}  {r['n_models']:>4d}")
print('=' * 145)

# Write to file
out_md = os.path.join(BASE, "CROSS_MODEL_LEADERBOARD.md")
with open(out_md, "w", encoding="utf-8") as f:
    f.write("# Cross-Model Leaderboard — 所有历史最佳结果\n\n")
    f.write("**生成时间**: 2026-07-25\n\n")
    f.write("**数据源**: 4 个 ensemble + TCN+SE LR=2e-3 self-ensemble + 23-dim batch sweep + 64-combos 23-dim 全表\n\n")
    f.write("## Top 25 排名 (按 Test Macro-F1 降序)\n\n")
    f.write("| Rank | Category | Name | F1m | PR-AUC | BinF1 | Acc | ROC | thr | #seed | #mdl |\n")
    f.write("|------|----------|------|-----|--------|-------|-----|-----|-----|-------|------|\n")
    for i, r in enumerate(results[:25], 1):
        bf = f"{r['BinF1']:.4f}" if r['BinF1'] is not None else 'N/A'
        ac = f"{r['Acc']:.4f}" if r['Acc'] is not None else 'N/A'
        rc = f"{r['ROC_AUC']:.4f}" if r['ROC_AUC'] is not None else 'N/A'
        f.write(f"| {i} | {r['category']} | {r['name']} | {r['F1m']:.4f} | {r['PR_AUC']:.4f} | {bf} | {ac} | {rc} | {r['thr']:.2f} | {r['n_seeds']} | {r['n_models']} |\n")
    f.write("\n")
    # 分类冠军
    f.write("## 分类冠军\n\n")
    cats = {}
    for r in results:
        cats.setdefault(r['category'], []).append(r)
    for cat, lst in cats.items():
        best = lst[0]
        f.write(f"### {cat}\n\n- **{best['name']}**\n  F1m = {best['F1m']:.4f}, PR-AUC = {best['PR_AUC']:.4f}, thr = {best['thr']:.2f}\n\n")
    f.write("## 关键洞察\n\n")
    f.write("1. **绝对冠军**: TCN+SE LR=2e-3 self-ensemble (5 seeds) → F1m=0.8735, **+0.0254** 超过所有跨架构 stacking\n")
    f.write("2. **跨架构 stacking 冠军**: ENSEMBLE_v4 Stack ALL_7_v2 → F1m=0.8481\n")
    f.write("3. **单模冠军**: TCN+SE 23-dim B=64 (5 seeds) → F1m=0.8454, ≈ stack 跨架构冠军\n")
    f.write("4. **PR-AUC 冠军**: TCN+SE LR=2e-3 self-ensemble → 0.9269, 远超所有单模/集成\n")
    f.write("5. **下一步机会**: 用 23-dim + B=64 + LR=2e-3 + 5-seed self-ensemble 替换 ENSEMBLE_v4 的 TCN_v4 — 可能再 +0.02\n")
print(f"\n[saved] {out_md}")
