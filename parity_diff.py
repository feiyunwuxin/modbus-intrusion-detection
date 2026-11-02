#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parity_diff.py  --  PC 参考 vs MCU 实测 概率对比

⏵ 2026-10-15 TFLite Micro H743 部署配套脚本
   读 mcu_pc_parity_audit.py 写的 pc_refs.json 和 mcu_log_capture.py 写的
   mcu_runs.json, 对齐 window 索引, 计算:
     - max_diff   (最大 |pc_prob - mcu_prob|)
     - mean_diff  (平均差)
     - n_compared (对齐的窗口数)
     - pc/mcu disagreement (>1e-3) 计数

   期望: max_diff = 0.0 (两者都跑同一 .tflite, 同一 int8 kernel).
   实际: 可能 < 1e-6 (sigmoid/exp 在 AC6 vs numpy 的最后一位舍入差异).
   如果 max_diff > 0.01, 立即报警: 可能是 MCU 端预处理 / 量化 / 推理 bug.
"""
import argparse
import json
import os
import sys


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def diff_one_seed(pc_seed_block: dict, mcu_runs: list, threshold_diff: float = 1e-3):
    """对齐 win_idx, 算 4 个指标."""
    pc_probs = pc_seed_block['probs']
    pc_starts = pc_seed_block['window_starts']

    # mcu_runs 索引: list of dicts
    # 对齐: 假设 mcu 顺序输出, win_idx 跟 pc_probs 索引一一对应
    mcu_probs = [r['prob'] for r in mcu_runs]
    mcu_frames = [r['frame'] for r in mcu_runs]

    n = min(len(pc_probs), len(mcu_probs))
    if n == 0:
        return {'n_compared': 0, 'max_diff': None, 'mean_diff': None,
                'disagreements': 0, 'msg': 'no overlapping windows'}

    diffs = [abs(pc_probs[i] - mcu_probs[i]) for i in range(n)]
    max_diff  = max(diffs)
    mean_diff = sum(diffs) / n
    dis = sum(1 for d in diffs if d > threshold_diff)

    return {
        'n_compared': n,
        'pc_probs_first10':  [round(p, 6) for p in pc_probs[:10]],
        'mcu_probs_first10': [round(p, 6) for p in mcu_probs[:10]],
        'mcu_frames_first10': mcu_frames[:10],
        'pc_window_starts_first10': pc_starts[:10],
        'max_diff':  round(max_diff, 9),
        'mean_diff': round(mean_diff, 9),
        'disagreements_gt_1e-3': dis,
        'disagreement_rate': round(dis / n, 6) if n else 0.0,
        'verdict': 'PASS' if (max_diff < 0.01 and dis / n < 0.01) else 'FAIL',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc', default=os.path.join(r'C:\work\Claude\Issue', 'pc_refs.json'))
    ap.add_argument('--mcu', default=os.path.join(r'C:\work\Claude\Issue', 'mcu_runs.json'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threshold_diff', type=float, default=1e-3)
    args = ap.parse_args()

    if not os.path.exists(args.pc):
        sys.stderr.write(f"[ERROR] 缺 {args.pc} (先跑 mcu_pc_parity_audit.py)\n")
        sys.exit(1)
    if not os.path.exists(args.mcu):
        sys.stderr.write(f"[ERROR] 缺 {args.mcu} (先跑 mcu_log_capture.py)\n")
        sys.exit(1)

    pc = load_json(args.pc)
    mcu = load_json(args.mcu)

    if args.seed not in pc:
        sys.stderr.write(f"[ERROR] {args.pc} 没 seed={args.seed} 块\n")
        sys.exit(1)

    print(f"=== Parity Diff: PC (seed={args.seed}) ↔ MCU ===\n")
    res = diff_one_seed(pc[str(args.seed)], mcu['runs'], args.threshold_diff)

    for k, v in res.items():
        if isinstance(v, list):
            print(f"  {k}: {v[:5]}{'...' if len(v) > 5 else ''}")
        else:
            print(f"  {k}: {v}")

    print(f"\n  期望: max_diff = 0.0 (bit-perfect, 同一 .tflite)")
    print(f"  实际: max_diff = {res['max_diff']}")
    print(f"  结论: {res['verdict']}  ({'✅' if res['verdict']=='PASS' else '❌'})")

    sys.exit(0 if res['verdict'] == 'PASS' else 1)


if __name__ == "__main__":
    main()
