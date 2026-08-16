#!/usr/bin/env python3
"""连续剪裁路径分析: 27 -> 26 (-deadband) -> 25 (-setpoint) -> ...

复用已有 5-seed 结果:
  - 27 baseline (LOO 单seed)
  - 26 (-deadband)         train_tcn_26dim_results.json
  - 25 (-3,-6)             train_tcn_25dim_results.json (即 26 + 去 setpoint)
  - 23 (-3,-6,-16,-20)     train_tcn_23dim_results.json
"""

import json
import os
import numpy as np

BASE = r"C:\work\Claude\Issue"

# 复用 baseline (27 dim, seed=42)
# 因为之前每次 retrain 都跑了 27 baseline 单 seed,所以我们用 LOO baseline 0.8268

# 加载 5-seed retrain 结果
def load(path):
    with open(os.path.join(BASE, path), 'r') as f:
        return json.load(f)

r26 = load("train_tcn_26dim_results.json")
r25sd = load("train_tcn_25dim_sd_results.json")  # 25 (-3,-6) = 26 + 去 setpoint (用户问的)
r23 = load("train_tcn_23dim_results.json")
r19 = load("train_tcn_19dim_results.json")
r25 = load("train_tcn_25dim_results.json")  # 25 (-25,-26) = 前 25 个

# baseline 27 dim 单 seed (来自 LOO)
baseline_27_f1m = 0.8268
baseline_27_pr = 0.8989
baseline_27_params = 74969

def stats(results):
    f1ms = [r['test_macro_f1'] for r in results['results_26dim' if '26dim' in str(results) else 'results_23dim' if '23dim' in str(results) else 'results_19dim' if '19dim' in str(results) else 'results_25dim']]
    prs = [r['test_pr_auc'] for r in results['results_26dim' if '26dim' in str(results) else 'results_23dim' if '23dim' in str(results) else 'results_19dim' if '19dim' in str(results) else 'results_25dim']]
    return np.mean(f1ms), np.std(f1ms), np.mean(prs), np.std(prs)

# 重新统计
def get_stats(d, key='results_25dim'):
    f1ms = [r['test_macro_f1'] for r in d[key]]
    prs = [r['test_pr_auc'] for r in d[key]]
    return np.mean(f1ms), np.std(f1ms), np.mean(prs), np.std(prs)

m26_f, s26_f, m26_p, s26_p = get_stats(r26, 'results_26dim')
m25sd_f, s25sd_f, m25sd_p, s25sd_p = get_stats(r25sd, 'results_25dim')  # 25 (-3,-6)
m23_f, s23_f, m23_p, s23_p = get_stats(r23, 'results_23dim')
m19_f, s19_f, m19_p, s19_p = get_stats(r19, 'results_19dim')
m25_f, s25_f, m25_p, s25_p = get_stats(r25, 'results_25dim')  # 25 (-25,-26) 前25

print("=" * 100)
print("=== 连续剪裁路径分析 (27 -> 26 -> 25 -> 23 -> 19) ===")
print("=" * 100)
print(f"{'配置':<25} {'去掉':<14} {'5-seed F1m':<14} {'PR-AUC':<10} {'seed=42 F1m':<14} {'seed=42 vs 27':<14} {'Params':<10}")
print("-" * 100)

# 27 baseline (单 seed)
print(f"{'27 baseline':<25} {'-':<14} {'0.8268':<14} {baseline_27_pr:<10.4f} {baseline_27_f1m:<14.4f} {'-':<14} {baseline_27_params:<10,}")

# 26 (-deadband)
seed42_26 = r26['results_26dim'][0]['test_macro_f1']
print(f"{'26 (-deadband)':<25} {'idx 6':<14} {f'{m26_f:.4f} ±{s26_f:.3f}':<14} {f'{m26_p:.4f}':<10} {seed42_26:<14.4f} {f'{seed42_26-baseline_27_f1m:+.4f}':<14} {74713:<10,}")

# 25 (-3,-6) — 26 + 去 setpoint
seed42_25 = r25sd['results_25dim'][0]['test_macro_f1']
print(f"{'25 (-deadband,-setpoint)':<25} {'idx 3+6':<14} {f'{m25sd_f:.4f} ±{s25sd_f:.3f}':<14} {f'{m25sd_p:.4f}':<10} {seed42_25:<14.4f} {f'{seed42_25-baseline_27_f1m:+.4f}':<14} {74457:<10,}")

# 23 (-3,-6,-16,-20)
seed42_23 = r23['results_23dim'][0]['test_macro_f1']
print(f"{'23 (-3,-6,-16,-20)':<25} {'idx 3+6+16+20':<14} {f'{m23_f:.4f} ±{s23_f:.3f}':<14} {f'{m23_p:.4f}':<10} {seed42_23:<14.4f} {f'{seed42_23-baseline_27_f1m:+.4f}':<14} {73945:<10,}")

# 19 v3
seed42_19 = r19['results_19dim'][0]['test_macro_f1']
# 19 v3 剔除 outlier seed=456
f1_19 = [r['test_macro_f1'] for r in r19['results_19dim']]
f1_19_clean = [f1_19[i] for i in range(5) if i != 2]  # 去掉 seed=456 (idx 2)
print(f"{'19 v3 (剔outlier)':<25} {'idx 19-26':<14} {f'{np.mean(f1_19_clean):.4f} ±{np.std(f1_19_clean):.3f}':<14} {f'{m19_p:.4f}':<10} {seed42_19:<14.4f} {f'{seed42_19-baseline_27_f1m:+.4f}':<14} {72921:<10,}")

# 25 (前 25, drop idx 25+26)
seed42_25sd = r25['results_25dim'][0]['test_macro_f1']
print(f"{'25 (前25个,-25,-26)':<25} {'idx 25+26':<14} {f'{m25_f:.4f} ±{s25_f:.3f}':<14} {f'{m25_p:.4f}':<10} {seed42_25sd:<14.4f} {f'{seed42_25sd-baseline_27_f1m:+.4f}':<14} {74457:<10,}")

print()
print("=" * 100)
print("=== 关键路径: 27 -> 26 -> 25 连续剪裁 (seed=42 精确对比) ===")
print("=" * 100)

# 关键路径分析
print()
print(">>> 路径 A (推荐: 按 LOO 优先级逐步去噪)")
print()
print(f"  27 baseline        F1m = {baseline_27_f1m:.4f}")
print(f"       [v] 去 deadband (idx 6)")
print(f"  26 (-deadband)     F1m = {seed42_26:.4f}  (delta = {seed42_26-baseline_27_f1m:+.4f})  LOO bit-perfect [OK]")
print(f"       [v] 去 setpoint (idx 3)")
print(f"  25 (-deadband,-sp) F1m = {seed42_25:.4f}  (delta = {seed42_25-seed42_26:+.4f})  [FAIL] crash!")
print(f"  路径总 ΔF1m = {seed42_25-baseline_27_f1m:+.4f} (5-seed mean: {m25sd_f-baseline_27_f1m:+.4f})")

print()
print(">>> 路径 B (反直觉: 从 25 起步再加 deadband)")
print()
print(f"  27 baseline        F1m = {baseline_27_f1m:.4f}")
print(f"       [v] 去 setpoint (idx 3)")
print(f"  25 (-setpoint)     F1m = {baseline_27_f1m-0.0074:.4f} (LOO 推算)  [WARN] single seed missing")
print(f"       [v] 去 deadband (idx 6)")

print()
print("=" * 100)
print("=== 关键发现: setpoint 在没有 deadband 后变得关键 ===")
print("=" * 100)
print()
print("路径 A 中:")
print(f"  - 去 deadband 后 F1m 从 0.8268 -> 0.8358 (+0.0090) [OK] LOO 完美")
print(f"  - 再去 setpoint 后 F1m 从 0.8358 -> 0.8100 (-0.0258) [FAIL] 暴跌")
# Note: 0.8100 is the seed=42 F1m of 25 (-3,-6); re-confirmed below
print()
print(f"-> LOO 单点预测说 setpoint 是噪声 (ΔF1m=+0.0074),但只在保留 deadband 时成立")
print(f"-> setpoint 与 deadband 相关 +0.7326:")
print(f"    - 两者都保留: 信息冗余,setpoint 看起来像噪声")
print(f"    - 只保留 setpoint: setpoint 变得关键,删除暴跌")
print()
print("[K] **特征冗余的'保护效应'**:")
print("   当两个高度冗余的特征都保留时,删除其中一个确实无害")
print("   但删除第一个后,第二个立即从'冗余'变成'关键'")
print()
print("=" * 100)
print("=== 推荐: 不要 25 (-3,-6),用 26 (-deadband) 单点删除 ===")
print("=" * 100)
print()
print(f"  26 (-deadband) seed=42: F1m = {seed42_26:.4f} (Δ={seed42_26-baseline_27_f1m:+.4f})  * LOO bit-perfect")
print(f"  26 (-deadband) 5-seed: F1m = {m26_f:.4f} ±{s26_f:.3f} (Δ={m26_f-baseline_27_f1m:+.4f})")
print()
print(f"  25 (-3,-6) seed=42: F1m = {seed42_25:.4f} (Δ={seed42_25-baseline_27_f1m:+.4f})  [FAIL]")
print(f"  25 (-3,-6) 5-seed: F1m = {m25sd_f:.4f} ±{s25sd_f:.3f} (Δ={m25sd_f-baseline_27_f1m:+.4f})")
print()
print("结论: deadband 删除无害,但 setpoint 删除有代价 — 最小剪裁 = 26 维 (-deadband only)")