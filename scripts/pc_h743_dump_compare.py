#!/usr/bin/env python3
"""
pc_h743_dump_compare.py
========================
PC 端接收 H743 DUMP 输出, 解析 sw_buf 23*16 floats, 跟 PC 端算的 ch32_input
bit-perfect 对比, 定位差异源.

用法:
  python pc_h743_dump_compare.py --port COM5 --csv "C:/work/Claude/Issue/IanArffDatasetF32.csv"

流程:
  1. 打开串口, 监听 [ALARM] + [DUMP_*] 输出
  2. 解析 1 个完整 dump (10 行)
  3. PC 端算同一行的 ch32_input (从 CSV 走 preprocess_v2_scada.py 全流程)
  4. 对比 max_diff, 输出报告
"""
import argparse
import re
import sys
import time
import numpy as np
import pandas as pd
import joblib
import serial

CLIP_VAL = 10.0
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}
DUMP_NCHUNK = 8
DUMP_CHUNK = 46


def build_18(df):
    """复刻 H743 feature_extractor.c + PC preprocess_v2_scada.py:73-105, 输出 RAW 18-dim."""
    df = df.sort_values("time").reset_index(drop=True)
    df["time_diff"] = df["time"].diff().fillna(0)
    last = {}
    tsl = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        k = int(df["address"].iat[i]) * 1000 + int(df["function"].iat[i])
        if k in last:
            tsl[i] = df["time"].iat[i] - last[k]
        last[k] = df["time"].iat[i]
    df["time_since_last_same_addr_func"] = tsl
    df["is_unusual_fc"] = df["function"].apply(lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)
    df["is_response"] = df["command response"].astype(np.float32)
    df["pressure measurement"] = pd.to_numeric(df["pressure measurement"], errors="coerce").fillna(0).clip(0, 100)
    keep = ["address", "function", "length", "gain", "reset rate", "deadband", "cycle time",
            "rate", "system mode", "control scheme", "pump", "solenoid",
            "pressure measurement", "crc rate", "time_diff",
            "time_since_last_same_addr_func", "is_unusual_fc", "is_response"]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[keep].astype(np.float32).values  # (N, 18)


def sliding_window_get(norm18, t0):
    """复刻 H743 sliding_window.c sliding_window_get() Python 版.
    输入: norm18 (N, 18) 归一化, t0 (window 中心 = 最后一帧的 0-based idx).
    输出: ch32_input (23, 16) numpy, 跟 H743 ch32_forward 期望一致.
    """
    SW_WINDOW = 16
    if t0 < SW_WINDOW - 1 or t0 >= len(norm18):
        return None
    win = norm18[t0 - SW_WINDOW + 1 : t0 + 1]  # (16, 18)
    press_sum = win[:, 12].sum()
    crc_max = win[:, 13].max()
    resp_signed = win[:, 17].sum()
    unusual_sign = win[:, 16].sum()
    seen = set()
    n = 0
    for v in win[:, 2]:
        k = int(round(float(v)))
        if k not in seen:
            seen.add(k)
            n += 1
    press_mean = press_sum / SW_WINDOW
    balance = 1.0 - resp_signed / 8.0
    crc_mean = win[:, 13].mean()
    cmd_count = float(SW_WINDOW - unusual_sign - resp_signed)
    frame27 = np.zeros((SW_WINDOW, 27), dtype=np.float32)
    for c18 in range(18):
        frame27[:, KEEP_18[c18]] = win[:, c18]
    frame27[:, 19] = press_mean
    frame27[:, 20] = crc_mean
    frame27[:, 21] = crc_max
    frame27[:, 22] = cmd_count
    frame27[:, 23] = resp_signed
    frame27[:, 24] = balance
    frame27[:, 25] = n
    frame27[:, 26] = unusual_sign
    sliced = frame27[:, KEEP_23]
    sliced = np.clip(sliced, -CLIP_VAL, CLIP_VAL)
    return sliced.T.astype(np.float32)  # (23, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv-path", default=r"C:\work\Claude\Issue\IanArffDatasetF32.csv")
    ap.add_argument("--scaler-path", default=r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib")
    ap.add_argument("--out", default=r"C:\work\Claude\Issue\logs\h743_dump_compare.csv")
    ap.add_argument("--n-dumps", type=int, default=20, help="最大接收 dump 数")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    raw18 = build_18(df)
    scaler = joblib.load(args.scaler_path)
    mu_18 = scaler.center_[KEEP_18].astype(np.float32)
    sg_18 = scaler.scale_[KEEP_18].astype(np.float32)
    norm18 = (raw18 - mu_18) / sg_18  # (N, 18)

    # 串口监听
    ser = serial.Serial(args.port, args.baud, timeout=2)
    print(f"Listening on {args.port} @ {args.baud}...")
    print(f"Press 'q'+Enter to quit, waiting for [DUMP_S] chunks...")
    sys.stdout.flush()

    in_dump = False
    alarm_count = -1
    chunks = []
    n_done = 0
    results = []
    while n_done < args.n_dumps:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith("[DUMP_S]"):
            m = re.match(r"\[DUMP_S\] c=(\d+) n=(\d+)", line)
            if m:
                alarm_count = int(m.group(1))
                in_dump = True
                chunks = []
        elif in_dump and line.startswith("[DUMP_"):
            m = re.match(r"\[DUMP_(\d+)\]\s*(.*)", line)
            if m:
                idx = int(m.group(1))
                vals = [float(x) for x in m.group(2).split(",") if x.strip()]
                chunks.append((idx, vals))
        elif line.startswith("[DUMP_E]"):
            # 拼回 368 floats
            chunks.sort(key=lambda x: x[0])
            flat = []
            for _, vals in chunks:
                flat.extend(vals)
            if len(flat) != 368:
                print(f"  ✗ chunk 不全: got {len(flat)} floats (expected 368)")
                in_dump = False
                continue
            ch32_h743 = np.array(flat, dtype=np.float32).reshape(23, 16)

            # 跟 PC 算的同一帧对比
            # H743 alarm_count = g_stats.frames_attack 累计值, 不是 frame_idx
            # 我们用 PC 端 norm18 模拟滑动窗口, 找 ch32_h743 中特征最匹配的 frame
            # 简化: 假设 manual 模式 frame_idx=1,2,3... 对应 norm18[0..N-1]
            # 找最近 100 帧里 max_diff 最小的
            best = None
            for t0 in range(15, min(len(norm18), 200)):
                pc = sliding_window_get(norm18, t0)
                if pc is None:
                    continue
                d = np.abs(pc - ch32_h743).max()
                if best is None or d < best[0]:
                    best = (d, t0, pc)
            if best is None:
                print(f"  ✗ alarm_count={alarm_count}: 没找到匹配 PC 帧")
                in_dump = False
                continue
            d, t0, pc = best
            frame_idx_1based = t0 + 1  # 0-based → 1-based
            attack_type = int(df.iloc[t0]["specific result"])
            fc = int(df.iloc[t0]["function"])
            binary = int(df.iloc[t0]["binary result"])
            status = "✓" if d < 1e-3 else "✗"
            print(f"  {status} alarm#{alarm_count} → PC frame {frame_idx_1based} "
                  f"(binary={binary}, type={attack_type}, fc={fc}), "
                  f"max_diff={d:.4e}")
            results.append({
                "alarm_count": alarm_count,
                "pc_frame_idx": frame_idx_1based,
                "binary": binary,
                "attack_type": attack_type,
                "fc": fc,
                "max_diff": d,
            })
            in_dump = False
            n_done += 1
            sys.stdout.flush()
        elif line.startswith("[ALARM]"):
            m = re.search(r"prob=0\.(\d{3})", line)
            if m:
                prob_h743 = int(m.group(1)) / 1000.0
                print(f"  [H743 ALARM] #{alarm_count} prob={prob_h743:.3f} (dump follows)")

    # 写 CSV
    if results:
        out = pd.DataFrame(results)
        out.to_csv(args.out, index=False)
        print(f"\nWrote {len(results)} dump records to {args.out}")
        print(f"\n=== Summary ===")
        print(f"  total dumps: {len(results)}")
        print(f"  max_diff range: [{out['max_diff'].min():.4e}, {out['max_diff'].max():.4e}]")
        print(f"  dumps with max_diff < 1e-3: {(out['max_diff'] < 1e-3).sum()} / {len(results)}")
        print(f"  dumps with max_diff > 0.1: {(out['max_diff'] > 0.1).sum()} / {len(results)}")
        if (out['max_diff'] > 1e-3).any():
            print(f"\n  ⚠️ sw_buf drift detected! H743 sw_buf ≠ PC sw_buf")
            print(f"     Root cause is in feature_extractor.c / sliding_window.c, NOT ch32_inference.c")
        else:
            print(f"\n  ✓ sw_buf bit-perfect, forward drift (if any) is in ch32_inference.c")


if __name__ == "__main__":
    main()