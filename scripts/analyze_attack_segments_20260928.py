#!/usr/bin/env python3
"""
analyze_attack_segments_20260928.py
==================================
分析攻击段 vs FP 段的对应关系.
"""
import pandas as pd

CSV_PATH = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"
df = pd.read_csv(CSV_PATH, nrows=765)

print("=== Attack type 在 765 行 frame 上的分布 (1-based) ===")
for at in sorted(df[df["binary result"] == 1]["specific result"].unique()):
    rows = df.index[df["specific result"] == at].tolist()  # 0-based
    rows1 = [r + 1 for r in rows]
    # 按 fc 拆
    fc_groups = df.iloc[rows].groupby("function").size().to_dict()
    print(f"  type={at}: count={len(rows1)}, frames (1-based)={rows1[:5]}...{rows1[-3:] if len(rows1)>5 else ''}, fc={fc_groups}")
print()
# 看 49 FP 对应的 frame 段 vs 真实 attack 段
print("=== Attack 段连续性 (按 binary=1 切段) ===")
attack_idx = df.index[df["binary result"] == 1].tolist()
# 切连续段
segs = []
cur_start = attack_idx[0]
cur_end = attack_idx[0]
for i in range(1, len(attack_idx)):
    if attack_idx[i] == attack_idx[i-1] + 1:
        cur_end = attack_idx[i]
    else:
        segs.append((cur_start, cur_end))
        cur_start = attack_idx[i]
        cur_end = attack_idx[i]
segs.append((cur_start, cur_end))
for s, e in segs:
    seg_df = df.iloc[s:e+1]
    types = seg_df["specific result"].value_counts().to_dict()
    fcs = seg_df["function"].value_counts().to_dict()
    print(f"  frame {s+1}..{e+1} (count={e-s+1}): types={types}, fcs={fcs}")
print()
# 49 FP 段的 frame 范围
fp_frames_1based = [167, 171, 175, 179, 183, 187, 191, 195, 231, 235, 241, 245, 249, 253, 255,
                    523, 527, 531, 535, 539, 543, 547, 551, 555, 559, 563, 567, 571, 575, 579,
                    583, 587, 591, 595, 599, 603, 607, 611, 615, 619, 623, 627, 631, 635, 639, 643,
                    757, 761, 765]
print(f"=== 49 FP frames (1-based) ===")
print(f"  FP frame 范围: {min(fp_frames_1based)}..{max(fp_frames_1based)}, count={len(fp_frames_1based)}")
# 看每个 FP frame 周围 ±2 的 attack 情况
print()
print("=== FP frame ±2 邻域的 attack type ===")
for f in fp_frames_1based[:10]:
    f0 = f - 1  # 0-based
    near = df.iloc[max(0, f0-2):f0+3]
    near_types = near["specific result"].value_counts().to_dict()
    print(f"  FP frame {f}: ±2 types={near_types}")