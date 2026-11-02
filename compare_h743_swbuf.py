#!/usr/bin/env python3
"""
compare_h743_swbuf.py
=====================
Bit-perfect 验证 H743 g_last_ch32_input (1472B = 23*16 floats) vs
PC 端对同样 16 帧算出的 expected (23, 16) input.

用法:
    python compare_h743_swbuf.py --port COM5 --csv IanArffDatasetF32.csv \\
        --scaler scaler_binary_v2_scada.joblib --n-frames 16

工作流:
    1. 打开 H743 串口
    2. 发送 N 帧 (默认 16) raw17 格式 (跟 rtu_replay_gui.py 同协议)
    3. 等 H743 处理完 (>=16 帧后 sw_buf 才填满)
    4. 发 'D' 命令
    5. 读 1472B = 23*16 floats, reshape (23, 16)
    6. 算 PC 端 expected: pc_ground_truth_18 + RobustScaler 归一化 + py_sliding_window_get
    7. Element-wise 对比, 报 max_diff / RMSE / per-feature 偏差

目的: 定位 14.7pp FPR gap 是 preprocessing drift / inference 数值精度 / 其他.
"""
import argparse
import io
import struct
import sys
import time

import joblib
import numpy as np
import pandas as pd
import serial

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# === 跟 h743/scripts/verify_sliding_window_23dim.py 完全一致 ===
SW_WINDOW = 16
SW_DIM = 18
CLIP_VAL = 10.0
KEEP_23 = [
    0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 21, 23, 24, 25, 26,
]
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}

# === 12 个原始特征对应的 CSV 列名 (跟 rtu_replay_gui.py RAW17_FEAT_COLS 一致) ===
RAW17_FEAT_COLS = (
    'setpoint', 'gain', 'reset rate', 'deadband',
    'cycle time', 'rate', 'system mode', 'control scheme',
    'pump', 'solenoid', 'pressure measurement',
)


# ─── PC 端 18-dim RAW 重建 (复刻 verify_sliding_window_23dim.py:32-73) ───

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
    df["pressure measurement"] = (
        pd.to_numeric(df["pressure measurement"], errors="coerce").fillna(0).clip(0, 100)
    )
    keep = [
        "address", "function", "length", "gain", "reset rate", "deadband",
        "cycle time", "rate", "system mode", "control scheme", "pump",
        "solenoid", "pressure measurement", "crc rate", "time_diff",
        "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
    ]
    for col in keep:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[keep].astype(np.float32).values


# ─── PC 端 sliding_window_get (复刻 verify_sliding_window_23dim.py:76-116) ───

def py_sliding_window_get(buf18):
    """复刻 h743/Core/Src/sliding_window.c sliding_window_get() (Python 版).
    输入归一化后的 18-dim buf18, 输出 (N, 23, 16).
    """
    out_windows = []
    for t0 in range(len(buf18) - SW_WINDOW + 1):
        win = buf18[t0:t0 + SW_WINDOW]
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
        sliced = frame27[:, KEEP_23]
        sliced = np.clip(sliced, -CLIP_VAL, CLIP_VAL)
        out_windows.append(sliced.T.astype(np.float32))
    return np.stack(out_windows, axis=0)


# ─── Raw17 帧编码 (跟 rtu_replay_gui.py:249-327 build_raw17_frame 一致) ───

def _safe_uint(v, default=0, vmin=0, vmax=255):
    try:
        x = int(float(v))
        return max(vmin, min(vmax, x))
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_raw17_frame(row) -> bytes:
    """从 CSV 行构建 length 驱动的 17-dim 原始数据帧 (复刻 rtu_replay_gui.py:249)."""
    slave = _safe_uint(row.get('address'), default=1, vmin=1, vmax=247)
    func = _safe_uint(row.get('function'), default=3, vmin=0, vmax=255)
    length = _safe_uint(row.get('length'), default=16, vmin=0, vmax=255)
    is_resp = 1 if _safe_uint(row.get('command response'),
                              default=0, vmin=0, vmax=1) == 1 else 0
    t = _safe_float(row.get('time'), default=0.0)
    time_bytes = struct.pack('>d', t)
    crc_raw = _safe_float(row.get('crc rate'))
    crc_u16 = max(0, min(65535, int(round(crc_raw))))
    crc_bytes = struct.pack('>H', crc_u16)
    feat_vals = [_safe_float(row.get(c)) for c in RAW17_FEAT_COLS]

    if length == 16:
        body = bytearray()
        for v in feat_vals[:10]:
            body.append(max(0, min(255, int(round(v)))))
        body += struct.pack('>f', float(feat_vals[10]))
        body += crc_bytes
    elif length == 46:
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>f', float(v))
        body += crc_bytes
    elif length == 90:
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>d', float(v))
        body += crc_bytes
    else:
        body = bytearray(11)
        body += crc_bytes
    return bytes([slave, func, length]) + bytes(body) + bytes([is_resp]) + time_bytes


# ─── 发送 + 接收 ───

def send_frames(ser, rows, n_frames):
    """发 n_frames 帧 raw17 格式到 H743, 返回 None."""
    for i in range(n_frames):
        row = rows.iloc[i]
        frame = build_raw17_frame(row)
        ser.write(frame)
        # 8ms 间隔 (auto-40ms 速率实际是 8ms 一帧, 留余量让 H743 处理)
        time.sleep(0.008)


def request_dump(ser, timeout=3.0):
    """发 'D' 命令, 阻塞读 1472B 回来. 返回 (23, 16) numpy float32."""
    # 清空 rx buffer
    ser.reset_input_buffer()
    ser.write(b'D')
    ser.flush()
    # 1472B @ 115200 ≈ 128ms, 给余量 (H743 alarm DMA-TX 也可能跟 dump DMA-TX 竞争)
    time.sleep(0.1)
    data = b''
    deadline = time.time() + timeout
    while len(data) < 1472 and time.time() < deadline:
        chunk = ser.read(1472 - len(data))
        if chunk:
            data += chunk
    if len(data) != 1472:
        # ⭐ 2026-10-17: 失败时打印 raw data + repr 辅助诊断
        print(f"    [DIAG] request_dump failed: got {len(data)}B (expected 1472B)")
        print(f"    [DIAG] data hex: {data[:64].hex(' ')}")
        print(f"    [DIAG] data repr: {data[:128]!r}")
        raise IOError(f"expected 1472B dump, got {len(data)}B after {timeout}s")
    arr = np.frombuffer(data, dtype=np.float32)
    if not np.array_equal(arr.tobytes(), data):
        # bytes 不是 4 的倍数? 不会发生, 留个保险
        raise ValueError("non-float32-aligned dump bytes")
    return arr.reshape(23, 16)


# ─── 主流程 ───

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5",
                    help="H743 USART1 串口 (默认 COM5)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", default=r"C:\work\Claude\Issue\IanArffDatasetF32.csv")
    ap.add_argument("--scaler", default=r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib")
    ap.add_argument("--n-frames", type=int, default=16,
                    help="发送多少帧填满 sw_buf (>=16 才有意义)")
    ap.add_argument("--n-trials", type=int, default=1,
                    help="重复 trial 次数, 每次发 n-frames 帧 + dump")
    ap.add_argument("--out-csv", default=None,
                    help="(可选) 把每次 trial 的 max_diff 写到 CSV")
    args = ap.parse_args()

    print("=" * 70)
    print(f"[1] 加载 CSV + Scaler")
    print("=" * 70)
    df = pd.read_csv(args.csv, nrows=args.n_frames * args.n_trials + 100)
    print(f"  CSV rows loaded: {len(df)}")
    scaler = joblib.load(args.scaler)
    mu_18 = scaler.center_[KEEP_18].astype(np.float32)
    sg_18 = scaler.scale_[KEEP_18].astype(np.float32)
    print(f"  scaler center[0..3] = {mu_18[:4]}")
    print(f"  scaler scale[0..3]  = {sg_18[:4]}")

    print()
    print("=" * 70)
    print(f"[2] 打开 H743 串口 {args.port} @ {args.baud}")
    print("=" * 70)
    ser = serial.Serial(args.port, args.baud, timeout=1)
    # ⭐ 2026-10-17: H743 启动 banner 可能要 500ms+ (TFLM init + fputc 阻塞)
    # 等 1s 后再读, 避免错过 banner
    time.sleep(1.0)
    # 清空启动 banner
    banner = ser.read(2048)
    print(f"  banner ({len(banner)}B): {banner[:128]!r}")
    if len(banner) == 0:
        print("  ⚠ H743 没有发送 banner!")
        print("    可能: (a) H743 在别的 COM 端口, (b) H743 还没启动, (c) H743 HardFault")
        print("    建议: 检查 H743 灯是否闪, 重新上电, 确认 COM 端口")
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    results = []
    for trial in range(args.n_trials):
        print()
        print("=" * 70)
        print(f"[3.{trial+1}] Trial {trial+1}/{args.n_trials}: 发 {args.n_frames} 帧 + dump")
        print("=" * 70)
        # 3.1 发帧
        rows_trial = df.iloc[trial * args.n_frames:(trial + 1) * args.n_frames]
        send_frames(ser, rows_trial, args.n_frames)
        time.sleep(0.05)  # 等 H743 全部处理

        # 3.2 dump
        h743_buf = request_dump(ser, timeout=3.0)
        print(f"  H743 dump shape = {h743_buf.shape}")
        print(f"  H743 dump range = [{h743_buf.min():.3f}, {h743_buf.max():.3f}]")

        # 3.3 PC 端 expected
        raw18 = pc_ground_truth_18(rows_trial)  # (16, 18)
        norm18 = (raw18 - mu_18) / sg_18        # (16, 18) 归一化
        expected = py_sliding_window_get(norm18)[0]  # (23, 16)
        print(f"  PC expected range = [{expected.min():.3f}, {expected.max():.3f}]")

        # 3.4 对比
        diff = np.abs(h743_buf - expected)
        max_diff = diff.max()
        rmse = np.sqrt((diff ** 2).mean())
        n_mismatch = int((diff > 1e-4).sum())
        print()
        print(f"  ⭐ max_diff = {max_diff:.6e}")
        print(f"  ⭐ RMSE     = {rmse:.6e}")
        print(f"  ⭐ # > 1e-4 = {n_mismatch} / {diff.size}  ({n_mismatch/diff.size*100:.2f}%)")
        print()
        print("  Per-feature max_diff (top 10):")
        feat_max = diff.max(axis=1)  # (23,)
        top10 = np.argsort(feat_max)[::-1][:10]
        for fi in top10:
            mark = " ⚠" if feat_max[fi] > 1e-4 else ""
            print(f"    feat[{fi:2d}] = {feat_max[fi]:.4e}{mark}")

        # 3.5 verdict
        print()
        if max_diff < 1e-4:
            print("  ✅ sw_buf bit-perfect (max_diff < 1e-4)")
            print("     → preprocessing 跟 PC 一致, FPR gap 在 inference 数值精度")
        elif max_diff < 0.01:
            print("  ✓ sw_buf 数值精度内一致 (max_diff < 0.01)")
            print("     → fp32 浮点累积误差, 14.7pp gap 不是 preprocessing")
        elif max_diff < 0.1:
            print("  ⚠ sw_buf 有偏差 (0.01 < max_diff < 0.1)")
            print("     → 某特征/某步骤有 bug, 需要查 per-feature 哪个最偏")
        else:
            print("  ❌ sw_buf 严重偏差 (max_diff > 0.1)")
            print("     → preprocessing 有重大 drift, 需查具体 feature 索引")

        results.append({
            "trial": trial + 1,
            "max_diff": max_diff,
            "rmse": rmse,
            "n_mismatch": n_mismatch,
            "n_total": int(diff.size),
            "top_feat": int(top10[0]) if len(top10) > 0 else -1,
            "top_feat_max": float(feat_max[top10[0]]) if len(top10) > 0 else 0.0,
        })

    ser.close()

    if args.out_csv:
        pd.DataFrame(results).to_csv(args.out_csv, index=False)
        print(f"\n→ Results saved to {args.out_csv}")

    # 退出码: 0 = bit-perfect, 1 = 有偏差
    overall_max = max(r["max_diff"] for r in results)
    sys.exit(0 if overall_max < 1e-4 else 1)


if __name__ == "__main__":
    main()
