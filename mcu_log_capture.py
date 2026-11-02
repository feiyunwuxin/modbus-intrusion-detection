#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mcu_log_capture.py  --  串口读取 H743 [ALARM] 日志, 解析为 JSON

⏵ 2026-10-15 TFLite Micro H743 部署配套脚本
   H743 通过 USART1 (115200 8N1) 输出 [ALARM] 行, 格式:
     [ALARM] N prob=0.NNN frame=K
   我们把这些行读出, 解析出每个 window 的 prob + frame 编号, 保存为 mcu_runs.json.

   rtu_replay_gui.py 已经做了类似的事 (IdsReportPanel), 但这个脚本是独立的,
   给 parity_diff.py 用, 不依赖 PyQt.

依赖:
   pip install pyserial
"""
import argparse
import json
import os
import re
import sys
import time

try:
    import serial
except ImportError:
    sys.stderr.write("[ERROR] 需要 pyserial: pip install pyserial\n")
    sys.exit(1)


# [ALARM] 1234 prob=0.456 frame=5678
ALARM_RE = re.compile(
    r'\[ALARM\]\s+'
    r'(?P<win_idx>\d+)\s+'
    r'prob=(?P<prob>[\d.]+)\s+'
    r'frame=(?P<frame>\d+)'
)


def capture(port: str, baud: int, duration: float, out_path: str) -> dict:
    """串口监听, 抓 [ALARM] 行, 写 JSON."""
    print(f"[open] {port} @ {baud}")
    ser = serial.Serial(port, baud, timeout=1)
    ser.reset_input_buffer()

    t0 = time.time()
    runs = []  # list of {win_idx, prob, frame, t_recv}
    raw_lines = 0

    while time.time() - t0 < duration:
        line = ser.readline().decode('ascii', errors='replace').rstrip()
        if not line:
            continue
        raw_lines += 1
        m = ALARM_RE.search(line)
        if m:
            runs.append({
                'win_idx': int(m.group('win_idx')),
                'prob': float(m.group('prob')),
                'frame': int(m.group('frame')),
                't_recv': time.time() - t0,
            })

    ser.close()
    print(f"[done] {duration}s 内收 {raw_lines} 行, 解析 {len(runs)} 个 [ALARM]")

    result = {
        'port': port,
        'baud': baud,
        'duration': duration,
        'n_runs': len(runs),
        'runs': runs,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] 写入 {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM5',
                    help='H743 串口 (默认 COM5, 跟 rtu_replay_gui.py 保持一致)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--duration', type=float, default=60.0,
                    help='监听秒数')
    ap.add_argument('--out', default=os.path.join(r'C:\work\Claude\Issue', 'mcu_runs.json'))
    args = ap.parse_args()
    capture(args.port, args.baud, args.duration, args.out)


if __name__ == "__main__":
    main()
