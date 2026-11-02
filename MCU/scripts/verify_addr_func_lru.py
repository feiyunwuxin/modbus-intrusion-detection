#!/usr/bin/env python3
"""
verify_addr_func_lru.py
========================
Bit-perfect 验证 h743/Core/Src/feature_extractor.c 的 LRU 64 round-robin 算法
跟 PC 训练管线 preprocess_v2_scada.py:80-92 的 dict 实现等价.

用法:
    cd h743/scripts && python verify_addr_func_lru.py --n-rows 1000
    cd h743/scripts && python verify_addr_func_lru.py --csv-path "C:/work/Claude/Issue/IanArffDatasetF32.csv"

接受阈值: max_diff < 1e-4 (float32 μs 级, 训练 f64 → H743 float32 转一次量化误差).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

LRU_SIZE = 64  # 跟 h743 feature_extractor.c: ADDR_FUNC_LRU_SIZE 一致


def pc_ground_truth(df):
    """复刻 preprocess_v2_scada.py:80-92 的 dict 逻辑 (完全无碰撞)."""
    last = {}
    out = np.zeros(len(df), dtype=np.float64)
    addr = df["address"].astype(int).values
    func = df["function"].astype(int).values
    time = df["time"].astype(np.float64).values
    for i in range(len(df)):
        key = int(addr[i]) * 1000 + int(func[i])
        if key in last:
            out[i] = time[i] - last[key]
        last[key] = time[i]
    return out


def h743_lru_predict(df):
    """复刻 h743/Core/Src/feature_extractor.c lru_addr_func_lookup() (新)."""
    ts = np.zeros(LRU_SIZE, dtype=np.float64)
    sl = np.zeros(LRU_SIZE, dtype=np.uint8)
    fn = np.zeros(LRU_SIZE, dtype=np.uint8)
    valid = np.zeros(LRU_SIZE, dtype=np.uint8)
    rr = 0
    out = np.zeros(len(df), dtype=np.float32)  # 跟 H743 (float) 一致
    addr = df["address"].astype(np.uint8).values
    func = df["function"].astype(np.uint8).values
    time = df["time"].astype(np.float64).values
    for i in range(len(df)):
        s, f, t = int(addr[i]), int(func[i]), float(time[i])
        hit = -1
        for k in range(LRU_SIZE):
            if valid[k] and sl[k] == s and fn[k] == f:
                hit = k
                break
        if hit >= 0:
            out[i] = np.float32(t - ts[hit])
            ts[hit] = t
        else:
            ts[rr] = t
            sl[rr] = s
            fn[rr] = f
            valid[rr] = 1
            rr = (rr + 1) % LRU_SIZE
            out[i] = np.float32(0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=1000)
    ap.add_argument(
        "--csv-path",
        default=r"C:\work\Claude\Issue\IanArffDatasetF32.csv",
    )
    args = ap.parse_args()

    if not os.path.exists(args.csv_path):
        sys.exit(f"CSV not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path, nrows=args.n_rows)
    df = df.sort_values("time").reset_index(drop=True)  # 跟 PC 训练管线一致
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    gt = pc_ground_truth(df)
    pred = h743_lru_predict(df)
    diff = np.abs(gt.astype(np.float64) - pred.astype(np.float64))
    # H743 输出 float32 (1e-5 量化误差), 训练用 f64; 接受阈值 1e-4 (float32 μs 级)
    n_miss = int(np.sum(diff > 1e-4))

    n_unique = len(set(zip(df["address"].tolist(), df["function"].tolist())))
    print(f"  unique (addr,func) seen: {n_unique}")
    print(f"  max_diff  = {diff.max():.3e}")
    print(f"  mean_diff = {diff.mean():.3e}")
    print(f"  mismatches (diff>1e-4): {n_miss}/{len(df)}")

    sys.exit(0 if (diff.max() < 1e-4 and n_miss == 0) else 1)


if __name__ == "__main__":
    main()