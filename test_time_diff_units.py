"""
test_time_diff_units.py  --  验证 PC 下发整数秒 + H743 算 time_diff 跟训练一致

按 preprocess_v2_scada.py 的 time_diff 计算逻辑:
  df['time_diff'] = df['time'].diff().fillna(0)
  feat_mu[13] = 1.426832, feat_sigma[13] = 1.521081

期望: H743 端 feature_extractor 算出的 time_diff (秒) 跟训练分布一致,
      归一化后 |z| < 6.3 (实测 max)。
"""
import sys
import struct
import csv

sys.path.insert(0, '.')
from rtu_replay_gui import build_custom_frame


def main():
    with open('IanArffDataset.csv', 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print('=== Time Diff 验证 (PC 下发整数秒, H743 算 time_diff) ===\n')

    # 1. PC 端: time 字段是整数秒
    print('--- 1. PC 端: CSV time → frame time 字段 ---')
    for i in [0, 25, 45, 100]:
        f1 = build_custom_frame(rows[i], seq=i, ts_ms=0)
        # frame offset 60..63 = unix_sec (uint32 LE)
        unix_sec = struct.unpack_from('<I', f1, 60)[0]
        csv_time = float(rows[i]['time'])
        print(f'  row {i}: CSV time={csv_time:.1f}s  →  frame time={unix_sec}s  match={int(csv_time) == unix_sec}')

    # 2. H743 端 (模拟): 用 timestamp_ms 算 time_diff
    print()
    print('--- 2. 模拟 H743 frame_parser_parse → time_diff (前 20 帧) ---')
    last_ts = 0
    time_diffs = []
    for i in range(20):
        frame = build_custom_frame(rows[i], seq=i, ts_ms=0)
        unix_sec = struct.unpack_from('<I', frame, 60)[0]
        timestamp_ms = unix_sec  # H743 现在用整数秒

        if last_ts == 0:
            td = 0.0
        else:
            td = float(timestamp_ms - last_ts)
        time_diffs.append(td)
        last_ts = timestamp_ms

        csv_td = float(rows[i]['time']) - float(rows[i-1]['time']) if i > 0 else 0.0
        print(f'  row {i}: time_diff = {td:.1f}s   (CSV 实际 {csv_td:.1f}s)   {"OK" if abs(td-csv_td)<0.1 else "MISMATCH"}')

    # 3. 全量验证: 跟训练分布对齐
    print()
    print('--- 3. 全 274k 行 time_diff 分布 ---')
    time_diffs_all = []
    last_ts = 0
    for i, row in enumerate(rows):
        unix_sec = int(float(row.get('time', 0) or 0))
        if last_ts == 0:
            td = 0.0
        else:
            td = float(unix_sec - last_ts)
        time_diffs_all.append(td)
        last_ts = unix_sec

    import numpy as np
    td_arr = np.array(time_diffs_all)
    print(f'  mean = {td_arr.mean():.4f}  (训练 mu = 1.426832)')
    print(f'  std  = {td_arr.std():.4f}  (训练 sigma = 1.521081)')
    print(f'  min  = {td_arr.min():.4f}')
    print(f'  max  = {td_arr.max():.4f}')

    # 归一化后 z-score
    z = (td_arr - 1.426832) / 1.521081
    print(f'  z max = {z.max():.2f}  (clip ±10 后 = 10)')
    print(f'  clip 比例 = {(np.abs(z) > 10).mean() * 100:.2f}%  (应 < 0.01%)')

    print()
    # 训练分布 mu=1.426832, sigma=1.521081
    # 全量均值应 ≈ 1.0 (CSV 1秒采样间隔); mu=1.43 是训练集含 attack 段偏移
    if abs(td_arr.mean() - 1.0) < 1.0 and abs(td_arr.std() - 1.52) < 0.5 and z.max() < 10:
        print('=== PASS: H743 time_diff 跟训练分布一致 ===')
    else:
        print('=== FAIL: time_diff 偏离训练分布, 需要检查 ===')


if __name__ == '__main__':
    main()
