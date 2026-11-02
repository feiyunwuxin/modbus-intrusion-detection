#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tflm_deploy_parity.py  --  MCU <-> PC 逐窗对拍 + 部署区 4 指标 (G3)

对比对象 (同一次回放、同一口径):
    PC  : TFLite (BUILTIN_REF) 在 F32 数据集部署区窗口上的输出
          -> experiments/tflm_fp32/artifacts/pc_parity_logits.npz (lg_tflite)
    MCU : 实机日志 (rtu_replay_gui.py 写的 logs/test_<ts>.csv) 里 ALARM 行的
          prob_x1000 (已经是 sigmoid 后的概率 x1000 取整)

窗口对齐 (跟 h743_error_analysis.py 完全一致, 不重新发明):
    日志里的 frame_idx 是**回放内**的帧号 (1-based, 回放从数据集行 200000 起).
    部署区窗口的全局窗号 w_local = frame_idx // 16 - 1
    对应 PC 参考数组的下标 = w_local (即全局窗 G0 + w_local).
    ⚠ 这个对齐是**实测**出来的: 用 Route E 的基线日志对 -1/0/+1/+2 四种
      offset 分别算 max|Δprob|, 只有 offset=0 给出 0 次决策翻转
      (其余 offset 有 ~300 次翻转). 见报告 "窗口对齐" 一节.

判定:
    - 决策翻转数 = 0  -> MCU 与 PC 在部署区行为一致, 数字可以直接对比
    - max|Δprob| 的下界是 log 的量化误差 (prob_x1000 整数化, 1/1000)

用法:
    python experiments/tflm_fp32/tools/tflm_deploy_parity.py --log logs/test_XXX.csv
    python experiments/tflm_fp32/tools/tflm_deploy_parity.py \
        --log logs/tflm.csv --label TFLM --baseline-log logs/routee.csv --label-baseline RouteE
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ART = os.path.join(ROOT, "experiments", "tflm_fp32", "artifacts")

W = 16
DEPLOY_START_ROW = 200000
G0 = DEPLOY_START_ROW // W          # 部署区第一个全局窗号
THR = 0.5


def log(msg):
    print(msg, flush=True)


def load_reference():
    """PC 侧参考: 逐窗 logit (TFLite BUILTIN_REF) + 窗标签."""
    logits = np.load(os.path.join(ART, "pc_parity_logits.npz"))
    windows = np.load(os.path.join(ART, "windows_f32_mcu.npz"))
    lg = logits["lg_tflite"].astype(np.float64)
    frames = logits["frames"]
    atk = windows["atk"].astype(np.int64)
    p_all = 1.0 / (1.0 + np.exp(-lg))
    return frames, lg, p_all, atk


def parse_log(path):
    """返回 {回放内 frame_idx: [prob_x1000, ...]} (只取 ALARM 行)."""
    alarms = {}
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("event") != "ALARM":
                continue
            if not r.get("prob_x1000"):
                continue
            alarms.setdefault(int(r["frame_idx"]), []).append(int(r["prob_x1000"]))
    return alarms


def align(alarms, n_deploy):
    """回放帧号 -> 部署区窗号 (w_local = fr//16 - 1), 返回 (prob 数组, 命中掩码)."""
    hp = np.zeros(n_deploy, dtype=np.float64)
    hit = np.zeros(n_deploy, dtype=bool)
    for fr, ps in alarms.items():
        w = fr // W - 1
        if 0 <= w < n_deploy:
            hp[w] = max(hp[w], max(ps) / 1000.0)
            hit[w] = True
    return hp, hit


def metrics(pred, g):
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    tn = int(((pred == 0) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    R = tp / (tp + fn) if tp + fn else 0.0
    FPR = fp / (fp + tn) if fp + tn else 0.0
    P = tp / (tp + fp) if tp + fp else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=P, recall=R, f1=F1, fpr=FPR)


def analyse(log_path, label, p_pc_dep, atk_dep):
    alarms = parse_log(log_path)
    hp, hit = align(alarms, len(p_pc_dep))

    m = hit
    dp = np.abs(hp[m] - p_pc_dep[m])
    flips = int(((hp[m] >= THR) != (p_pc_dep[m] >= THR)).sum())

    mcu = metrics((hp >= THR).astype(int), atk_dep)
    pc = metrics((p_pc_dep >= THR).astype(int), atk_dep)

    # log 量化下界: prob_x1000 是整数, 所以 |Δ| 天然 >= 量化误差
    quant_err = np.abs(hp[m] * 1000.0 - np.round(hp[m] * 1000.0)) / 1000.0

    res = dict(
        label=label,
        log=os.path.relpath(log_path, ROOT),
        n_deploy_windows=int(len(p_pc_dep)),
        n_alarm_windows=int(m.sum()),
        parity_vs_pc=dict(
            max_abs_dprob=float(dp.max()) if dp.size else 0.0,
            median_abs_dprob=float(np.median(dp)) if dp.size else 0.0,
            p99_abs_dprob=float(np.percentile(dp, 99)) if dp.size else 0.0,
            mean_abs_dprob=float(dp.mean()) if dp.size else 0.0,
            decision_flips=flips,
            log_quantization_floor=float(quant_err.max()) if quant_err.size else 0.0,
        ),
        deploy_zone_mcu=mcu,
        deploy_zone_pc=pc,
    )
    return res


def print_res(r):
    p = r["parity_vs_pc"]
    log(f"\n===== {r['label']} =====")
    log(f"  log                : {r['log']}")
    log(f"  部署区窗口          : {r['n_deploy_windows']}")
    log(f"  MCU 报警窗 (prob>=thr): {r['n_alarm_windows']}")
    log(f"  MCU<->PC |Δprob|   : max {p['max_abs_dprob']:.6f}  "
        f"median {p['median_abs_dprob']:.6f}  p99 {p['p99_abs_dprob']:.6f}")
    log(f"  log 量化下界        : {p['log_quantization_floor']:.6f}  "
        f"(prob_x1000 整数化)")
    log(f"  决策翻转 (thr={THR}) : {p['decision_flips']}")
    m, c = r["deploy_zone_mcu"], r["deploy_zone_pc"]
    log("  部署区 4 指标:")
    log("    %-8s %5s %5s %5s %5s %8s %8s %8s %8s" %
        ("source", "TP", "FP", "TN", "FN", "P", "R", "F1", "FPR"))
    log("    %-8s %5d %5d %5d %5d %8.4f %8.4f %8.4f %8.4f" %
        ("MCU", m["tp"], m["fp"], m["tn"], m["fn"], m["precision"], m["recall"], m["f1"], m["fpr"]))
    log("    %-8s %5d %5d %5d %5d %8.4f %8.4f %8.4f %8.4f" %
        ("PC", c["tp"], c["fp"], c["tn"], c["fn"], c["precision"], c["recall"], c["f1"], c["fpr"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="MCU 侧日志 (rtu_replay_gui 写的 CSV)")
    ap.add_argument("--label", default="MCU")
    ap.add_argument("--baseline-log", default=None, help="对照组日志 (例如 Route E 基线)")
    ap.add_argument("--label-baseline", default="Baseline")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    frames, lg, p_all, atk = load_reference()
    p_dep = p_all[G0:]
    atk_dep = atk[G0:]

    log("=" * 78)
    log("TFLM <-> PC 部署区逐窗对拍")
    log("=" * 78)
    log(f"  PC 参考   : experiments/tflm_fp32/artifacts/pc_parity_logits.npz")
    log(f"  部署区    : 全局窗 {G0}..{len(frames)-1}  ({len(p_dep)} 窗)")
    log(f"  窗口对齐  : w_local = frame_idx//16 - 1  (实测唯一 0-flip 的对齐)")

    res = analyse(args.log, args.label, p_dep, atk_dep)
    print_res(res)

    if args.baseline_log:
        base = analyse(args.baseline_log, args.label_baseline, p_dep, atk_dep)
        print_res(base)
        res["baseline"] = base
        log("\n----- 汇总 -----")
        log(f"  {args.label} vs {args.label_baseline}: "
            f"决策翻转 {res['parity_vs_pc']['decision_flips']} vs "
            f"{base['parity_vs_pc']['decision_flips']}")
        log(f"  F1  {res['deploy_zone_mcu']['f1']:.4f} (MCU) vs "
            f"{base['deploy_zone_mcu']['f1']:.4f} ({args.label_baseline})")

    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        log(f"\n  -> {os.path.relpath(args.out, ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
