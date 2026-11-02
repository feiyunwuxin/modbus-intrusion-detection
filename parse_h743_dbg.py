#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 H743 USART1 串口输出的 [DBG] + [DBG_F] 行, 跟 PC sim_pc_align 对齐.

H743 输出格式 (modbus_ids.c):
  [DBG]   idx|slave|func|gain|reset|deadband|cyc|rate|sm|cs|pump|sol|press|crc|time|payload|is_resp
            (所有 float 字段 *10000 → int32)
  [DBG_F] idx|f0|f1|...|f22
            (23 维 feats *10000 → int32)

用法:
  # 1. 串口抓 H743 输出到文件 (PuTTY/超级终端 log):
  python parse_h743_dbg.py h743_serial.log

  # 2. 或直接给 - 走 stdin:
  python parse_h743_dbg.py -

输出:
  h743_dbg_parsed.csv — 每行 idx + 17 raw 字段 + 23 feats
  h743_dbg_vs_pc.csv  — 跟 sim_pc_align_A 的 PC 端 X_keep 对齐
"""
import sys, re, csv, argparse, os
import numpy as np

BASE = r"C:\work\Claude\Issue"

DBG_RE   = re.compile(rb'\[DBG\]\s+(\d+)\|(\d+)\|(\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(-?\d+)\|(-?\d+)\|(\d+)\|(\d+)\|(\d+)')
DBG_F_RE = re.compile(rb'\[DBG_F\]\s+(\d+)\|(.+)')

RAW_FIELDS = ['idx', 'slave', 'func', 'gain', 'reset_rate', 'deadband',
              'cycle_time', 'rate', 'system_mode', 'control_scheme',
              'pump', 'solenoid', 'pressure_measurement', 'crc_rate',
              'timestamp_ms', 'payload_len', 'is_response']


def parse_log(path):
    """读 H743 串口 log (text 或 binary), 返回 [{raw: dict, feats: list or None}, ...]."""
    by_idx = {}
    if path == '-':
        data = sys.stdin.buffer.read()
    else:
        with open(path, 'rb') as f:
            data = f.read()
    for line in data.split(b'\n'):
        line = line.strip()
        if not line: continue
        m = DBG_RE.search(line)
        if m:
            vals = [int(x) for x in m.groups()]
            idx = vals[0]
            d = dict(zip(RAW_FIELDS, vals))
            # float 字段 (gain..pressure) 还原成 float (除 10000)
            for k in ['gain','reset_rate','deadband','cycle_time','rate','pressure_measurement']:
                d[k] = d[k] / 10000.0
            # crc_rate 现在是 uint16 raw (H743 2026-09-21 改成 uint16, 跟 PC struct.pack('>H') 一致),
            # 不再 /10000, 保持整数语义 (CSV 累计 CRC 错误数, 范围 12336..17990).
            # 其他 uint 字段 (slave/func/system_mode/.../timestamp_ms/payload_len/is_response) 保持整数.
            by_idx.setdefault(idx, {'raw': d, 'feats': None})
            continue
        m = DBG_F_RE.search(line)
        if m:
            idx = int(m.group(1))
            vals = [int(x) for x in m.group(2).strip().split(b'|')]
            if len(vals) == 23:
                by_idx.setdefault(idx, {'raw': None, 'feats': None})
                by_idx[idx]['feats'] = [v / 10000.0 for v in vals]
    # 按 idx 排序
    return [by_idx[i] for i in sorted(by_idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logfile')
    ap.add_argument('--pc-feats', default='sim_pc_align_A_thr0.5_feats.csv',
                    help='PC 端 sim 输出的 feats CSV (含 23 维 f0..f22)')
    ap.add_argument('--out-prefix', default='h743_dbg')
    args = ap.parse_args()

    entries = parse_log(args.logfile)
    print(f"parsed {len(entries)} frame(s) from {args.logfile}", file=sys.stderr)

    # dump 原始 parsed
    out_parsed = f"{args.out_prefix}_parsed.csv"
    with open(out_parsed, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(RAW_FIELDS + [f'feat_{i}' for i in range(23)])
        for e in entries:
            if e['raw'] is None: continue
            row = [e['raw'][k] for k in RAW_FIELDS]
            row += e['feats'] if e['feats'] else [''] * 23
            w.writerow(row)
    print(f"dumped {out_parsed}", file=sys.stderr)

    # 跟 PC feats 对比
    if not os.path.exists(args.pc_feats):
        print(f"WARN: {args.pc_feats} not found, skip PC feats compare", file=sys.stderr)
        return
    pc = []
    with open(args.pc_feats) as f:
        for row in csv.DictReader(f):
            pc.append([float(row[f'f{i}']) for i in range(23)])
    print(f"PC feats entries: {len(pc)} (frames 0..{len(pc)*16-1})", file=sys.stderr)

    # H743 idx 是调试序号 (1..16), 跟 PC 的 frame_idx (窗最后一步行号) 不直接对应
    # 但 H743 raw 里有 timestamp_ms (unix sec ×1000), 可用它查 PC 的 last_idx
    # 现在做"按 row 顺序对齐": H743 entries[0] 对 PC entries[i_window] (i_window 同 sent 顺序)
    # 简单起见: 假设 H743 前 N 帧的 raw_idx 1..N 对应 PC 顺序回放的前 N 窗
    print("\n=== frame-by-frame 23 维 feats 对比 (H743 vs PC A 变体) ===", file=sys.stderr)
    print(f"{'H743#':>5} | {'PC#':>5} | "
          + " | ".join([f"f{i:02d}" for i in range(23)]), file=sys.stderr)
    print('-' * 140, file=sys.stderr)
    for i, e in enumerate(entries):
        if e['feats'] is None: continue
        if i >= len(pc): break
        pc_f = pc[i]
        diffs = [abs(e['feats'][j] - pc_f[j]) for j in range(23)]
        max_diff = max(diffs)
        bad_dims = [j for j in range(23) if diffs[j] > 1.0]
        bad_marker = '*' if bad_dims else ' '
        bad_str = f"  diff>1: {bad_dims}" if bad_dims else ""
        line = f"{int(e['raw']['idx']):5d} | {i:5d} | "
        line += " | ".join(f"{e['feats'][j]:6.2f}/{pc_f[j]:6.2f}{bad_marker if diffs[j]>1 else ' '}"
                            for j in range(23))
        line += f"  max_diff={max_diff:.2f}{bad_str}"
        print(line, file=sys.stderr)
        # stdout 用 utf-8 写 (避免 GBK)
        sys.stdout.buffer.write((line + '\n').encode('utf-8'))


if __name__ == "__main__":
    main()
