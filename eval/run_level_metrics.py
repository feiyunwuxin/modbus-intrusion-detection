#!/usr/bin/env python3
"""Run-level IDS 性能评估工具 (Round 24+ 标准化).

对 H743 测试日志 CSV 计算业务对齐的指标:
  - Run-level Recall: 每个 attack run 至少 1 次 alarm 命中
  - Time-to-Detect (TTD): 攻击开始到首次报警的延迟
  - Per-length bucket: 按 run 长度分组评估
  - Missed runs 详情

用法:
    # 单文件评估
    python eval/run_level_metrics.py logs/test_20261001-090703_auto-40ms.csv

    # 批量评估 (logs/ 目录下所有 *.csv, 输出对比表)
    python eval/run_level_metrics.py --batch

    # 自定义参数
    python eval/run_level_metrics.py --max-delay 8 --json out.json <csv>

为什么 run-level 而非 per-frame:
  - 短脉冲攻击 (avg 2 帧) 下, per-frame Recall 偏低不反映真实能力
  - 实际安全用例只需要"每次攻击都触发 1 次报警"
  - Run-level 才是与业务对齐的指标
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def find_attack_runs(binary_seq: List[int]) -> List[Tuple[int, int]]:
    """从 binary 序列中找出所有 attack run. 返回 [(start_idx, end_idx), ...]."""
    runs = []
    i = 0
    n = len(binary_seq)
    while i < n:
        if binary_seq[i] == 1:
            start = i
            while i < n and binary_seq[i] == 1:
                i += 1
            runs.append((start, i - 1))
        else:
            i += 1
    return runs


def reconstruct_binary_and_alarms(
    csv_path: str,
) -> Tuple[List[int], List[int], List[Optional[bool]], Dict[str, int]]:
    """重建 PC 端 binary 序列 + 每个 [ALARM] 预测的帧位置 + legacy is_fp.

    Returns:
        binary_seq: PC 端所有 SEND 帧的 binary (按时间顺序)
        alarm_pred_idx: 每个 [ALARM] 预测的 PC 帧在 binary_seq 中的位置
        alarm_is_fp_legacy: 每个 [ALARM] 的 legacy is_fp (Round 22 窗内 any())
        csv_summary: 从 summary.txt 提取的 TP/FP/Attack/Normal 等
    """
    df = pd.read_csv(csv_path)
    binary_seq: List[int] = []
    alarm_pred_idx: List[int] = []
    alarm_is_fp_legacy: List[Optional[bool]] = []

    for _, row in df.iterrows():
        event = row['event']
        if event == 'SEND':
            b = int(row['binary']) if not pd.isna(row['binary']) else 0
            binary_seq.append(b)
        elif event == 'ALARM':
            if len(binary_seq) > 0:
                alarm_pred_idx.append(len(binary_seq) - 1)
                is_fp_str = row['is_fp']
                if pd.isna(is_fp_str):
                    alarm_is_fp_legacy.append(None)
                elif is_fp_str == 'TP':
                    alarm_is_fp_legacy.append(False)
                elif is_fp_str == 'FP':
                    alarm_is_fp_legacy.append(True)
                else:
                    alarm_is_fp_legacy.append(None)

    # 从 CSV 最后一行读 running_* (final values)
    csv_summary = {}
    if len(df) > 0:
        last = df.iloc[-1]
        for col in ('running_total', 'running_attack', 'running_normal',
                    'running_tp', 'running_fp'):
            if col in df.columns and not pd.isna(last[col]):
                csv_summary[col.replace('running_', '')] = int(last[col])

    return binary_seq, alarm_pred_idx, alarm_is_fp_legacy, csv_summary


def compute_run_level_metrics(
    binary_seq: List[int],
    alarm_pred_idx: List[int],
    max_delay: int = 16,
) -> Dict[str, Any]:
    """计算 run-level 指标."""
    runs = find_attack_runs(binary_seq)
    n_runs = len(runs)

    detected_runs = []
    ttds = []
    missed_runs = []

    for run_idx, (start, end) in enumerate(runs):
        # 找最近的 alarm (在 run 内 或 run 后 max_delay 帧内)
        best_ai = None
        for ai in alarm_pred_idx:
            if start <= ai <= end + max_delay:
                if best_ai is None or ai < best_ai:
                    best_ai = ai
        if best_ai is not None:
            detected_runs.append(run_idx)
            ttds.append(best_ai - start)
        else:
            missed_runs.append({
                'idx': run_idx,
                'start': start,
                'end': end,
                'length': end - start + 1,
            })

    n_detected = len(detected_runs)

    if ttds:
        avg_ttd = sum(ttds) / len(ttds)
        min_ttd = min(ttds)
        max_ttd = max(ttds)
        immediate = sum(1 for t in ttds if t == 0)
        within_3 = sum(1 for t in ttds if t <= 3)
        within_16 = sum(1 for t in ttds if t <= 16)
    else:
        avg_ttd = min_ttd = max_ttd = None
        immediate = within_3 = within_16 = 0

    return {
        'n_runs': n_runs,
        'n_detected': n_detected,
        'n_missed': n_runs - n_detected,
        'run_recall': n_detected / n_runs if n_runs > 0 else 0,
        'avg_ttd': avg_ttd,
        'min_ttd': min_ttd,
        'max_ttd': max_ttd,
        'immediate_pct': immediate / n_runs * 100 if n_runs > 0 else 0,
        'within_3_pct': within_3 / n_runs * 100 if n_runs > 0 else 0,
        'within_16_pct': within_16 / n_runs * 100 if n_runs > 0 else 0,
        'missed_runs': missed_runs,
    }


def compute_per_length_metrics(
    binary_seq: List[int],
    alarm_pred_idx: List[int],
    max_delay: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """按 attack run 长度分组."""
    runs = find_attack_runs(binary_seq)
    buckets = defaultdict(lambda: {'runs': 0, 'detected': 0, 'ttds': []})

    for start, end in runs:
        run_len = end - start + 1
        if run_len == 1:
            bucket = '1'
        elif run_len == 2:
            bucket = '2'
        elif run_len == 3:
            bucket = '3'
        elif run_len <= 7:
            bucket = '4-7'
        elif run_len <= 15:
            bucket = '8-15'
        else:
            bucket = '16+'

        buckets[bucket]['runs'] += 1

        best_ai = None
        for ai in alarm_pred_idx:
            if start <= ai <= end + max_delay:
                if best_ai is None or ai < best_ai:
                    best_ai = ai
        if best_ai is not None:
            buckets[bucket]['detected'] += 1
            buckets[bucket]['ttds'].append(best_ai - start)

    result = {}
    for bucket, data in buckets.items():
        recall = data['detected'] / data['runs'] if data['runs'] > 0 else 0
        avg_ttd = sum(data['ttds']) / len(data['ttds']) if data['ttds'] else None
        result[bucket] = {
            'runs': data['runs'],
            'detected': data['detected'],
            'recall': recall,
            'avg_ttd': avg_ttd,
        }
    return result


def compute_alarm_classification(
    binary_seq: List[int],
    alarm_pred_idx: List[int],
) -> Dict[str, Any]:
    """按预测帧 binary 分类 TP/FP (Option A 视角)."""
    tp = fp = 0
    for ai in alarm_pred_idx:
        if 0 <= ai < len(binary_seq):
            if binary_seq[ai] == 1:
                tp += 1
            else:
                fp += 1
    total = tp + fp
    return {
        'tp': tp,
        'fp': fp,
        'total': total,
        'precision': tp / total if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 报告打印
# ---------------------------------------------------------------------------

def print_report(
    csv_path: str,
    binary_seq: List[int],
    alarm_pred_idx: List[int],
    csv_summary: Dict[str, int],
    max_delay: int,
    quiet: bool = False,
) -> Dict[str, Any]:
    """打印评估报告, 返回 JSON dict."""
    n_total = len(binary_seq)
    n_attack = sum(binary_seq)
    n_normal = n_total - n_attack

    runs = find_attack_runs(binary_seq)
    run_metrics = compute_run_level_metrics(binary_seq, alarm_pred_idx, max_delay)
    per_length = compute_per_length_metrics(binary_seq, alarm_pred_idx, max_delay)
    alarm_cls = compute_alarm_classification(binary_seq, alarm_pred_idx)

    if not quiet:
        print(f"\n=== Run-level IDS 性能评估 ===")
        print(f"Input: {csv_path}")
        print(f"Max detection delay: {max_delay} frames")
        print()
        print(f"Total frames: {n_total}  (Attack: {n_attack}, Normal: {n_normal})")
        print(f"Total [ALARM] messages: {len(alarm_pred_idx)}")
        print()
        print("=" * 70)
        print("【Run-level 检测指标】")
        print("=" * 70)
        print(f"Attack runs:                {run_metrics['n_runs']}")
        print(f"Detected runs:              {run_metrics['n_detected']}")
        print(f"Missed runs:                {run_metrics['n_missed']}")
        print(f"** Run-level Recall:        {run_metrics['run_recall']*100:.2f}% **")
        print()
        if run_metrics['avg_ttd'] is not None:
            print(f"Time-to-Detect (TTD):")
            print(f"  Average:    {run_metrics['avg_ttd']:.2f} frames")
            print(f"  Min:        {run_metrics['min_ttd']} frames")
            print(f"  Max:        {run_metrics['max_ttd']} frames")
            print(f"  Immediate:  {run_metrics['immediate_pct']:.1f}% (TTD=0)")
            print(f"  Within 3:   {run_metrics['within_3_pct']:.1f}% (TTD<=3)")
            print(f"  Within {max_delay}: {run_metrics['within_16_pct']:.1f}% (TTD<={max_delay})")

        print()
        print("=" * 70)
        print("【按 attack run 长度分桶】")
        print("=" * 70)
        print(f"{'长度':<10}{'总数':<10}{'检出':<10}{'Recall':<12}{'Avg TTD':<10}")
        print("-" * 70)
        for bucket in ['1', '2', '3', '4-7', '8-15', '16+']:
            if bucket in per_length:
                m = per_length[bucket]
                ttd_str = f"{m['avg_ttd']:.1f}" if m['avg_ttd'] is not None else 'N/A'
                print(f"{bucket:<10}{m['runs']:<10}{m['detected']:<10}"
                      f"{m['recall']*100:<12.1f}{ttd_str:<10}")

        print()
        print("=" * 70)
        print("【对比 Per-frame 旧指标】")
        print("=" * 70)
        recall_frame = alarm_cls['tp'] / n_attack if n_attack else 0
        f1_num = 2 * recall_frame * alarm_cls['precision']
        f1_den = recall_frame + alarm_cls['precision']
        f1 = f1_num / f1_den if f1_den > 0 else 0
        print(f"Per-frame Recall:     {recall_frame*100:.2f}%")
        print(f"Precision:            {alarm_cls['precision']*100:.2f}%")
        print(f"Per-frame F1:         {f1*100:.2f}%")
        print(f"Per-run Recall:       {run_metrics['run_recall']*100:.2f}%  (推荐)")
        if run_metrics['avg_ttd'] is not None:
            ttd_ms = run_metrics['avg_ttd'] * 40  # 假设 25fps
            print(f"Avg TTD:              {run_metrics['avg_ttd']:.2f} frames (~{ttd_ms:.0f}ms @ 25fps)")

    # 返回 JSON dict
    return {
        'input': csv_path,
        'max_delay': max_delay,
        'n_frames': n_total,
        'n_attack_frames': n_attack,
        'n_normal_frames': n_normal,
        'n_alarms': len(alarm_pred_idx),
        'run_level': {
            'n_runs': run_metrics['n_runs'],
            'n_detected': run_metrics['n_detected'],
            'n_missed': run_metrics['n_missed'],
            'recall': run_metrics['run_recall'],
            'avg_ttd': run_metrics['avg_ttd'],
            'immediate_pct': run_metrics['immediate_pct'],
            'within_3_pct': run_metrics['within_3_pct'],
        },
        'per_length': per_length,
        'per_frame': {
            'recall': alarm_cls['tp'] / n_attack if n_attack else 0,
            'precision': alarm_cls['precision'],
            'tp': alarm_cls['tp'],
            'fp': alarm_cls['fp'],
        },
        'csv_summary': csv_summary,
    }


def batch_evaluate(logs_dir: str, max_delay: int) -> List[Dict[str, Any]]:
    """批量评估 logs/ 下所有 test_*.csv."""
    csv_files = sorted(Path(logs_dir).glob('test_*.csv'))
    if not csv_files:
        print(f"[ERROR] No test_*.csv files in {logs_dir}")
        return []

    print(f"Found {len(csv_files)} test CSVs in {logs_dir}")
    print()

    results = []
    for csv_path in csv_files:
        try:
            binary_seq, alarm_pred_idx, _, csv_summary = reconstruct_binary_and_alarms(str(csv_path))
            if len(binary_seq) < 16:
                # ramp-up 阶段, 跳过
                continue
            result = print_report(str(csv_path), binary_seq, alarm_pred_idx,
                                  csv_summary, max_delay, quiet=True)
            results.append(result)
        except Exception as e:
            print(f"[SKIP] {csv_path.name}: {e}")

    # 对比表
    if results:
        print("\n" + "=" * 110)
        print(f"{'Test':<35}{'Frames':<10}{'Runs':<8}{'Det':<6}"
              f"{'R-Run':<10}{'F1-frm':<10}{'TTD':<8}{'Miss':<6}")
        print("-" * 110)
        for r in results:
            name = Path(r['input']).stem
            ttd = f"{r['run_level']['avg_ttd']:.1f}" if r['run_level']['avg_ttd'] else 'N/A'
            f1_num = 2 * r['per_frame']['recall'] * r['per_frame']['precision']
            f1_den = r['per_frame']['recall'] + r['per_frame']['precision']
            f1 = (f1_num / f1_den * 100) if f1_den > 0 else 0
            print(f"{name:<35}{r['n_frames']:<10}{r['run_level']['n_runs']:<8}"
                  f"{r['run_level']['n_detected']:<6}"
                  f"{r['run_level']['recall']*100:<10.2f}"
                  f"{f1:<10.2f}{ttd:<8}{r['run_level']['n_missed']:<6}")

    return results


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run-level IDS 性能评估工具 (Round 24+)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('csv', nargs='?', default=None,
                        help='单个 CSV 路径 (默认: latest logs/test_*.csv)')
    parser.add_argument('--batch', action='store_true',
                        help='批量评估 logs/ 目录下所有 test_*.csv')
    parser.add_argument('--max-delay', type=int, default=16,
                        help='最大报警延迟 (帧, 默认 16 = window size)')
    parser.add_argument('--json', type=str, default=None,
                        help='输出 JSON 报告到文件')
    parser.add_argument('--logs-dir', type=str, default='logs',
                        help='批量模式的目录 (默认 logs/)')

    args = parser.parse_args()

    if args.batch:
        results = batch_evaluate(args.logs_dir, args.max_delay)
        if args.json and results:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n[JSON] Saved to {args.json}")
        return

    # 单文件模式
    if args.csv is None:
        # 默认: 最新的 logs/test_*.csv
        logs = sorted(Path(args.logs_dir).glob('test_*.csv'))
        if not logs:
            print("[ERROR] No test_*.csv found")
            sys.exit(1)
        args.csv = str(logs[-1])
        print(f"[AUTO] Using latest: {args.csv}")

    binary_seq, alarm_pred_idx, _, csv_summary = reconstruct_binary_and_alarms(args.csv)
    result = print_report(args.csv, binary_seq, alarm_pred_idx, csv_summary, args.max_delay)

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[JSON] Saved to {args.json}")


if __name__ == '__main__':
    main()
