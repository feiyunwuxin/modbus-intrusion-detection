#!/usr/bin/env python3
"""
analyze_h743_alarms_20260928.py
===============================
分析 test_20260928-225115_manual-274628rows.csv 的 77 个 ALARM,
把每个 alarm 映射回 CSV 的 attack type / FC / length, 找 root cause.
"""
import csv
import sys
import pandas as pd

CSV_PATH = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"
LOG_PATH = r"C:\work\Claude\Issue\logs\test_20260928-225115_manual-274628rows.csv"


def main():
    # 1. 读 CSV (765 行)
    df = pd.read_csv(CSV_PATH, nrows=765)
    print(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
    print()
    print("=== Attack 类型分布 (specific result, 前 765 行) ===")
    print(df["specific result"].value_counts().sort_index())
    print()
    print("=== FC 分布 (前 765 行) ===")
    print("Normal FC:", df[df["binary result"] == 0]["function"].value_counts().to_dict())
    print("Attack FC:", df[df["binary result"] == 1]["function"].value_counts().to_dict())
    print()
    # attack frame_idx 范围
    attack_frames = df.index[df["binary result"] == 1].tolist()  # 0-based
    print(f"Attack frames (0-based): {attack_frames[0]}..{attack_frames[-1]} (total {len(attack_frames)})")
    print()

    # 2. 解析 ALARM 行
    alarms = []
    with open(LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["event"] == "ALARM":
                alarms.append(row)
    print(f"Total ALARMs: {len(alarms)}")
    print()

    # 3. 每个 alarm 映射回 CSV 行
    # running_total 是 alarm 触发时已发送的总帧数 (1-based)
    # running_attack 是其中 attack 数
    # running_normal 是 normal 数
    # binary = 1 → attack (TP), binary = 0 → normal (FP)
    print("=== Alarm → CSV 行映射 ===")
    print(f"{'#':<4} {'run_total':<10} {'is_fp':<5} {'prob':<6} {'csv_row':<8} {'attack_type':<6} {'fc':<4} {'len':<5} {'addr':<5} {'cmd_resp':<8}")
    tp_types = {}
    fp_types = {}
    for a in alarms:
        run_total = int(a["running_total"])
        is_fp = a["is_fp"]
        prob = int(a["prob_x1000"]) / 1000.0
        # running_total 是 alarm 触发时累计发送数
        # alarm 是基于前面 16 帧做的推理, alarm 触发时这帧刚发完
        csv_row = run_total  # 1-based
        if csv_row > 0 and csv_row <= 765:
            cs = df.iloc[csv_row - 1]
            attack_type = int(cs["specific result"])
            fc = int(cs["function"])
            length = int(cs["length"])
            addr = int(cs["address"])
            cmd_resp = int(cs["command response"])
        else:
            attack_type = fc = length = addr = cmd_resp = -1
        prob_str = f"{prob:.3f}"
        print(f"{a['alarm_count']:<4} {run_total:<10} {is_fp:<5} {prob_str:<6} {csv_row:<8} {attack_type:<6} {fc:<4} {length:<5} {addr:<5} {cmd_resp:<8}")
        if is_fp == "TP":
            tp_types[attack_type] = tp_types.get(attack_type, 0) + 1
        else:
            fp_types[attack_type] = fp_types.get(attack_type, 0) + 1

    print()
    print(f"TP attack type counts: {tp_types}")
    print(f"FP attack type counts (binary=0 → specific result 应该是 0): {fp_types}")
    print()
    # 4. CSV 里没被检测到的 attack 类型
    detected = set()
    for a in alarms:
        if a["is_fp"] == "TP":
            run_total = int(a["running_total"])
            if 0 < run_total <= 765:
                detected.add(int(df.iloc[run_total - 1]["specific result"]))
    all_attack_types = set(df[df["binary result"] == 1]["specific result"].astype(int).unique())
    missed = all_attack_types - detected
    print(f"All attack types in 765 frames: {sorted(all_attack_types)}")
    print(f"Detected attack types: {sorted(detected)}")
    print(f"Missed attack types: {sorted(missed)}")
    print()
    # 5. missed attack 的统计
    print("=== Missed attacks (per-type counts in 765 rows) ===")
    for at in sorted(all_attack_types):
        cnt_in_csv = (df["specific result"] == at).sum()
        tp_cnt = tp_types.get(at, 0)
        print(f"  type={at}: in_csv={cnt_in_csv}, detected={tp_cnt}, missed={cnt_in_csv - tp_cnt}")
    print()
    # 6. FP 段的 normal 帧特征
    print("=== FP frames (运行到 alarm 时, 哪个 CSV 行) ===")
    print(f"  total FP: {sum(1 for a in alarms if a['is_fp'] == 'FP')}")
    # FP 应该 binary=0, 但我们查 CSV 看看
    fp_rows = []
    for a in alarms:
        if a["is_fp"] == "FP":
            run_total = int(a["running_total"])
            if 0 < run_total <= 765:
                fp_rows.append(run_total - 1)  # 0-based
    print(f"  FP CSV rows (0-based): {fp_rows[:20]}...")
    if fp_rows:
        # FP 段 binary=0 的 normal frame, 看 fc 分布
        fp_df = df.iloc[fp_rows]
        print(f"  FP CSV rows binary result distribution: {fp_df['binary result'].value_counts().to_dict()}")
        print(f"  FP CSV rows function distribution: {fp_df['function'].value_counts().to_dict()}")
        print(f"  FP CSV rows length distribution: {fp_df['length'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()