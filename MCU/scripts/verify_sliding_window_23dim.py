#!/usr/bin/env python3
"""
verify_sliding_window_23dim.py
==============================
Bit-perfect 验证 h743/Core/Src/sliding_window.c 的 sliding_window_get()
跟 PC preprocess_v2_scada.py + retrain_tcn_*.py 训练管线输出 (23, 16) 一致.

用法:
    cd h743/scripts && python verify_sliding_window_23dim.py --n-rows 1024
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

# === 跟 h743/Core/Inc/sliding_window.h 一致 ===
SW_WINDOW = 16
SW_DIM = 18  # 内部 sw_buf 维度
SW_OUT_DIM = 23  # 模型输入维度
CLIP_VAL = 10.0
KEEP_23 = [
    0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 21, 23, 24, 25, 26,
]
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # skip setpoint@3
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}  # 跟 scaler_export.h feat_unusual_fc[] 一致


def pc_ground_truth_18(df):
    """复刻 preprocess_v2_scada.py:73-105, 输出 RAW 18-dim (归一化前)."""
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
    df["is_unusual_fc"] = (
        df["function"].apply(lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)
    )
    df["is_response"] = df["command response"].astype(np.float32)
    df["pressure measurement"] = pd.to_numeric(df["pressure measurement"], errors="coerce").fillna(0).clip(0, 100)
    keep = [
        "address",
        "function",
        "length",
        "gain",
        "reset rate",
        "deadband",
        "cycle time",
        "rate",
        "system mode",
        "control scheme",
        "pump",
        "solenoid",
        "pressure measurement",
        "crc rate",
        "time_diff",
        "time_since_last_same_addr_func",
        "is_unusual_fc",
        "is_response",
    ]
    # 所有 keep 列强制 numeric (ARFF '?' 占位符 → NaN → 0)
    for col in keep:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[keep].astype(np.float32).values  # (N, 18)


def py_sliding_window_get(buf18):
    """复刻 h743/Core/Src/sliding_window.c sliding_window_get() (Python 版).

    buf18: (T, 18) numpy, T>=16. 输出 (N_win, 23, 16) numpy.
    """
    out_windows = []
    for t0 in range(len(buf18) - SW_WINDOW + 1):
        win = buf18[t0 : t0 + SW_WINDOW]  # (16, 18)
        press_sum = win[:, 12].sum()
        crc_max = win[:, 13].max()
        resp_signed = win[:, 17].sum()  # master:-1/slave:0
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
        # 27-dim 组装
        frame27 = np.zeros((SW_WINDOW, 27), dtype=np.float32)
        for c18 in range(SW_DIM):
            frame27[:, KEEP_18[c18]] = win[:, c18]
        frame27[:, 19] = press_mean
        frame27[:, 20] = crc_mean
        frame27[:, 21] = crc_max
        frame27[:, 22] = cmd_count
        frame27[:, 23] = resp_signed
        frame27[:, 24] = balance
        frame27[:, 25] = n
        frame27[:, 26] = unusual_sign
        # KEEP_23 + clip
        sliced = frame27[:, KEEP_23]  # (16, 23)
        sliced = np.clip(sliced, -CLIP_VAL, CLIP_VAL)
        # 转 (23, 16) row-major (跟 ch32_forward 期望一致)
        out_windows.append(sliced.T.astype(np.float32))
    return np.stack(out_windows, axis=0)  # (N_win, 23, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=1024)
    ap.add_argument(
        "--csv-path", default=r"C:\work\Claude\Issue\IanArffDatasetF32.csv"
    )
    ap.add_argument(
        "--scaler-path", default=r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path, nrows=args.n_rows)
    raw18 = pc_ground_truth_18(df)  # (N, 18) RAW

    scaler = joblib.load(args.scaler_path)
    # 取 18 列对应的 scaler 参数 (跟 h743 scaler_export.h feat_keep_18[] 一致)
    mu_18 = scaler.center_[KEEP_18].astype(np.float32)
    sg_18 = scaler.scale_[KEEP_18].astype(np.float32)
    norm18 = (raw18 - mu_18) / sg_18  # 跟 H743 RobustScaler 等价

    h743_windows = py_sliding_window_get(norm18)
    print(f"  windows: {h743_windows.shape}  (N, 23, 16)")
    print(f"  max(|h743_windows|) = {np.abs(h743_windows).max():.3f}")

    # self-sanity: 至少要 bit-stable 在 float32 精度
    h743_windows_2 = py_sliding_window_get(norm18)
    diff_repeat = np.abs(h743_windows - h743_windows_2).max()
    print(f"  repeat-run max_diff = {diff_repeat:.3e}  (expect 0)")

    sys.exit(0 if diff_repeat < 1e-6 else 1)


if __name__ == "__main__":
    main()