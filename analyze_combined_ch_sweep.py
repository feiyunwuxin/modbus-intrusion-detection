#!/usr/bin/env python3
"""
合并 ch=8-24 (新) + ch=32-128 (旧) 两份扫描结果,做完整横评和 Pareto 分析。
"""
import os
import pandas as pd
import numpy as np

BASE = r"C:\work\Claude\Issue"
df_big = pd.read_csv(os.path.join(BASE, "tcn_v4_se_ch_sweep_lr2e3_summary.csv"))
df_small = pd.read_csv(os.path.join(BASE, "tcn_v4_se_ch_sweep_lr2e3_small_summary.csv"))

# 对齐列名
df_small = df_small.rename(columns={"n_blocks": "n_blocks"})[["channels", "n_params", "best_val_f1m",
                                                                "best_val_epoch", "best_threshold",
                                                                "test_f1m", "test_binary_f1",
                                                                "test_pr_auc", "test_acc", "train_time"]]
df_big["n_blocks"] = 3
df_all = pd.concat([df_small, df_big], ignore_index=True).sort_values("n_params").reset_index(drop=True)

# 计算性价比指标
df_all["f1m_per_kp"] = df_all["test_f1m"] / (df_all["n_params"] / 1000)
df_all["prauc_per_kp"] = df_all["test_pr_auc"] / (df_all["n_params"] / 1000)
df_all["log_params"] = np.log10(df_all["n_params"])

print("=" * 100)
print("  完整 10-ch 扫描合并结果 (LR=2e-3, B=128, b=3, 19-dim SCADA)")
print("=" * 100)
print(f"{'ch':>4} {'Params':>8} {'F1m':>8} {'Bin-F1':>8} {'PR-AUC':>8} {'Acc':>8} "
      f"{'Time(s)':>8} {'F1m/Kp':>8} {'PRAUC/Kp':>10} {'TrainEp':>8}")
print("-" * 100)
for _, r in df_all.iterrows():
    print(f"{int(r['channels']):>4} {int(r['n_params']):>8,} {r['test_f1m']:>8.4f} "
          f"{r['test_binary_f1']:>8.4f} {r['test_pr_auc']:>8.4f} {r['test_acc']:>8.4f} "
          f"{r['train_time']:>8.1f} {r['f1m_per_kp']:>8.4f} {r['prauc_per_kp']:>10.4f} "
          f"{int(r['best_val_epoch']):>8}")

print()
print("=" * 100)
print("  各项冠军")
print("=" * 100)
f1m_champ = df_all.loc[df_all["test_f1m"].idxmax()]
prauc_champ = df_all.loc[df_all["test_pr_auc"].idxmax()]
params_min = df_all.loc[df_all["n_params"].idxmin()]
f1m_per_kp_champ = df_all.loc[df_all["f1m_per_kp"].idxmax()]
prauc_per_kp_champ = df_all.loc[df_all["prauc_per_kp"].idxmax()]

print(f"F1m 冠军     : ch={int(f1m_champ['channels']):>3}  F1m={f1m_champ['test_f1m']:.4f}  "
      f"Params={int(f1m_champ['n_params']):,}  PR-AUC={f1m_champ['test_pr_auc']:.4f}")
print(f"PR-AUC 冠军  : ch={int(prauc_champ['channels']):>3}  PR-AUC={prauc_champ['test_pr_auc']:.4f}  "
      f"Params={int(prauc_champ['n_params']):,}  F1m={prauc_champ['test_f1m']:.4f}")
print(f"最小模型     : ch={int(params_min['channels']):>3}  Params={int(params_min['n_params']):,}  "
      f"F1m={params_min['test_f1m']:.4f}  PR-AUC={params_min['test_pr_auc']:.4f}")
print(f"F1m 性价比   : ch={int(f1m_per_kp_champ['channels']):>3}  "
      f"F1m/Kp={f1m_per_kp_champ['f1m_per_kp']:.4f}  F1m={f1m_per_kp_champ['test_f1m']:.4f}  "
      f"Params={int(f1m_per_kp_champ['n_params']):,}")
print(f"PR-AUC 性价比: ch={int(prauc_per_kp_champ['channels']):>3}  "
      f"PRAUC/Kp={prauc_per_kp_champ['prauc_per_kp']:.4f}  PR-AUC={prauc_per_kp_champ['test_pr_auc']:.4f}  "
      f"Params={int(prauc_per_kp_champ['n_params']):,}")

print()
print("=" * 100)
print("  Pareto 前沿 (在 log(Params)-F1m 平面)")
print("=" * 100)
# 找 Pareto 最优点: 不能被任何 (params 更小 且 F1m 更高) 的点支配
pareto_f1m = []
sorted_df = df_all.sort_values("n_params")
for i, row in sorted_df.iterrows():
    is_dominated = False
    for j, other in sorted_df.iterrows():
        if (other["n_params"] < row["n_params"] and other["test_f1m"] >= row["test_f1m"]):
            is_dominated = True
            break
    if not is_dominated:
        pareto_f1m.append(row)
print("F1m-Pareto:")
for r in pareto_f1m:
    print(f"  ch={int(r['channels']):>3}  Params={int(r['n_params']):>7,}  "
          f"F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pr_auc']:.4f}")

print()
print("PR-AUC Pareto:")
pareto_prauc = []
sorted_df = df_all.sort_values("n_params")
for i, row in sorted_df.iterrows():
    is_dominated = False
    for j, other in sorted_df.iterrows():
        if (other["n_params"] < row["n_params"] and other["test_pr_auc"] >= row["test_pr_auc"]):
            is_dominated = True
            break
    if not is_dominated:
        pareto_prauc.append(row)
for r in pareto_prauc:
    print(f"  ch={int(r['channels']):>3}  Params={int(r['n_params']):>7,}  "
          f"PR-AUC={r['test_pr_auc']:.4f}  F1m={r['test_f1m']:.4f}")

# 保存合并 CSV
df_all.to_csv(os.path.join(BASE, "tcn_v4_se_ch_sweep_lr2e3_FULL.csv"), index=False)
print(f"\n合并结果已保存: tcn_v4_se_ch_sweep_lr2e3_FULL.csv")
