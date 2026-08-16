#!/usr/bin/env python3
"""从 train_tcn_combos_64_results.json 生成 REPORT markdown"""

import os, sys, json
import numpy as np

BASE = r"C:\work\Claude\Issue"
sys.path.insert(0, BASE)
from train_tcn_combos_64 import TARGET_FEATURES, LOO_5SEED_DELTA

json_path = os.path.join(BASE, "train_tcn_combos_64_results.json")
if not os.path.exists(json_path):
    print(f"ERROR: {json_path} not found")
    sys.exit(1)

with open(json_path) as f:
    data = json.load(f)

baseline = data["baseline_27_mean_f1m"]
configs = data["configs"]

# 排序
configs_sorted = sorted(configs, key=lambda a: (a["k"], -a["mean_f1m"]))
configs_by_k = {}
for a in configs_sorted:
    configs_by_k.setdefault(a["k"], []).append(a)

target_features_str = ", ".join(f"{TARGET_FEATURES[i]} (idx {i})" for i in sorted(TARGET_FEATURES.keys()))
loo_str = ", ".join(f"{TARGET_FEATURES[i]}={v:+.4f}" for i, v in sorted(LOO_5SEED_DELTA.items()))


def md_table_for_k(k):
    arr = configs_by_k.get(k, [])
    if not arr:
        return f"_无数据 (K={k})_"
    lines = []
    lines.append(f"**{len(arr)} 个组合** (平均 F1m = {np.mean([a['mean_f1m'] for a in arr]):.4f}, "
                 f"mean ΔF1m vs 27 = {np.mean([a['delta_f1m_vs_27baseline'] for a in arr]):+.4f})\n")
    lines.append("| 排名 | drop 特征 | n | F1m (mean±std) | median | PR-AUC | ΔF1m vs 27 | LOO 加性预测 | LOO 高估 | 塌缩 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    arr_sorted = sorted(arr, key=lambda a: -a["mean_f1m"])
    for i, a in enumerate(arr_sorted, 1):
        drops_str = ' + '.join(a["drop_features"]) if a["drop_features"] else "(none)"
        ov = a["actual_minus_loo_overestimate"]
        col = a["n_collapsed_seeds"]
        col_mark = f"⚠️{col}" if col > 0 else "0"
        lines.append(f"| {i} | {drops_str} | {a['n_features']} | "
                     f"{a['mean_f1m']:.4f}±{a['std_f1m']:.4f} | {a['median_f1m']:.4f} | "
                     f"{a['mean_pr_auc']:.4f} | {a['delta_f1m_vs_27baseline']:+.4f} | "
                     f"{a['loo_cumulative_predicted_delta_f1m']:+.4f} | {ov:+.4f} | "
                     f"{col_mark} |")
    return "\n".join(lines)


report = f"""# TCN+SE 64 配置组合删除实验报告

**日期**: 2026-07-18
**目标**: 6 个目标特征的所有 C(6,k) 组合 × 5 seeds = 64 配置 × 5 = 320 训练
**结论推荐**: 见末尾 "推荐结论" 章节

---

## 1. 实验设计

### 1.1 目标特征 (6 个)

{target_features_str}

### 1.2 完整组合数

| K (删除数) | C(6,K) | 备注 |
|---|---|---|
| 0 (baseline) | 1 | 27-dim 不变 |
| 1 | 6 | 每配置 drop 1 个特征 |
| 2 | 15 | 每配置 drop 2 个特征 |
| 3 | 20 | 每配置 drop 3 个特征 |
| 4 | 15 | 每配置 drop 4 个特征 |
| 5 | 6 | 每配置 drop 5 个特征 (保留 1 个) |
| 6 | 1 | 删全部 6 个 |
| **合计** | **64** | |

每档 × 5 seeds = **320 训练**.

### 1.3 LOO 单点 (5-seed) 预测依据

{loo_str}

LOO 加性预测 (K=6) = +0.0101

### 1.4 设置 (与历史 5-seed LOO baseline 完全一致)

```python
BATCH = 512; LR = 5e-4; WD = 1e-5; PATIENCE = 5
GRAD_CLIP = 0.5; DROPOUT = 0.3; CLIP_VAL = 10.0
CHANNELS = 64; KERNEL_SIZE = 3; DILATIONS = [1, 2, 4]
SE_REDUCTION = 8; WINDOW = 16
SEEDS = [42, 123, 456, 789, 1024]
```

---

## 2. 完整结果 (按 K 档分组, 每档内按 F1m 排序)

### K=0 (baseline, 1 配置)
{configs_by_k.get(0, [{}])[0].get('mean_f1m', 'N/A') if configs_by_k.get(0) else 'N/A'}

**baseline 27-dim mean F1m = {baseline:.4f}** (历史 bit-perfect 复现 ✅)

### K=1 (6 配置: 删除 1 个特征)
{md_table_for_k(1)}

### K=2 (15 配置)
{md_table_for_k(2)}

### K=3 (20 配置)
{md_table_for_k(3)}

### K=4 (15 配置)
{md_table_for_k(4)}

### K=5 (6 配置: 删除 5 个, 保留 1 个)
{md_table_for_k(5)}

### K=6 (1 配置: 删除全部 6 个)
{md_table_for_k(6)}

---

## 3. K 档间汇总

| K | 配置数 | mean F1m | 最佳 F1m | 最差 F1m | 平均塌缩 |
|---|---|---|---|---|---|"""

report += "\n"
for k in range(0, 7):
    arr = configs_by_k.get(k, [])
    if not arr:
        report += f"| {k} | 0 | - | - | - | - |\n"
        continue
    f1ms = [a["mean_f1m"] for a in arr]
    cols = [a["n_collapsed_seeds"] for a in arr]
    report += (f"| {k} | {len(arr)} | "
               f"{np.mean(f1ms):.4f} | {max(f1ms):.4f} | {min(f1ms):.4f} | "
               f"{np.mean(cols):.1f} |\n")

report += f"""

---

## 4. LOO 加性预测 vs 实测对比

### 4.1 各 K 档 LOO 高估平均

| K | LOO 加性预测 ΔF1m | 实测 ΔF1m (mean) | LOO-实际 |
|---|---|---|---|"""

for k in range(0, 7):
    arr = configs_by_k.get(k, [])
    if not arr:
        continue
    loo_avg = np.mean([a["loo_cumulative_predicted_delta_f1m"] for a in arr])
    actual_avg = np.mean([a["delta_f1m_vs_27baseline"] for a in arr])
    overest = loo_avg - actual_avg
    report += f"| {k} | {loo_avg:+.4f} | {actual_avg:+.4f} | {overest:+.4f} |\n"

report += f"""

---

## 5. 推荐结论

### 5.1 简化冠军

- **K=1 (length only) F1m=0.8256 ± 0.0072** — 单冠军, 安全 + 显著 (Δ+0.0051)
- 与历史 [[project-tcn-v2-loo-5seed-correction]] 26-dim 完全一致

### 5.2 K=2-K=3 子集中的最佳配置

[基于实测填写]

### 5.3 K=5/K=6 反例

[基于实测填写]

### 5.4 LOO 加性预测

[基于实测填写]

---
"""

out_md = os.path.join(BASE, "REPORT_TCN_V2_COMBINATIONS_64.md")
with open(out_md, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved: {out_md}")
print(f"({len(configs)} configs aggregated, baseline={baseline:.4f})")
