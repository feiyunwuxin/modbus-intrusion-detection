#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
int8_vs_fp32_diff.py  --  同一回放口径下 INT8 固件 vs FP32 固件 的部署区逐窗对比

两次回放都从数据集行 200000 起, 帧号口径一致, 所以可以逐窗对齐.

输出:
    - 4 指标对比 (各自算, 以及相对 FP32 的增量)
    - 决策翻转分解: FP32->INT8 的翻转分成 4 类
        FP32 TP -> INT8 FN   (漏报, 变差)
        FP32 TN -> INT8 FP   (误报, 变差)
        FP32 FP -> INT8 TN   (误报被治好, 变好)
        FP32 FN -> INT8 TP   (漏报被治好, 变好)
    - 概率分布对比 (量化把 logit 压到什么程度)
    - FP / FN 集合的交集/新增/消失
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
G0 = DEPLOY_START_ROW // W
THR = 0.5


def log(msg):
    print(msg, flush=True)


def load_reference():
    logits = np.load(os.path.join(ART, "pc_parity_logits.npz"))
    windows = np.load(os.path.join(ART, "windows_f32_mcu.npz"))
    lg = logits["lg_tflite"].astype(np.float64)
    atk = windows["atk"].astype(np.int64)
    p = 1.0 / (1.0 + np.exp(-lg))
    return p[G0:], atk[G0:]


def parse_log(path):
    alarms = {}
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("event") != "ALARM" or not r.get("prob_x1000"):
                continue
            alarms.setdefault(int(r["frame_idx"]), []).append(int(r["prob_x1000"]))
    return alarms


def align(alarms, n):
    hp = np.zeros(n, dtype=np.float64)
    hit = np.zeros(n, dtype=bool)
    for fr, ps in alarms.items():
        w = fr // W - 1
        if 0 <= w < n:
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
    Acc = (tp + tn) / max(1, len(g))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=P, recall=R,
                f1=F1, fpr=FPR, acc=Acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8-log", required=True)
    ap.add_argument("--fp32-log", required=True)
    ap.add_argument("--label-int8", default="INT8")
    ap.add_argument("--label-fp32", default="FP32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p_pc, atk = load_reference()
    n = len(p_pc)

    h8, hit8 = align(parse_log(args.int8_log), n)
    h32, hit32 = align(parse_log(args.fp32_log), n)

    d8 = (h8 >= THR).astype(int)
    d32 = (h32 >= THR).astype(int)

    m8 = metrics(d8, atk)
    m32 = metrics(d32, atk)

    log("=" * 78)
    log("INT8 固件 vs FP32 固件 —— 部署区逐窗对比 (同一回放口径)")
    log("=" * 78)
    log(f"  INT8 日志 : {os.path.relpath(args.int8_log, ROOT)}")
    log(f"  FP32 日志 : {os.path.relpath(args.fp32_log, ROOT)}")
    log(f"  部署区    : {n} 窗 (数据集行 {DEPLOY_START_ROW} 起)")
    log(f"  报警窗覆盖: INT8 {int(hit8.sum())} / FP32 {int(hit32.sum())}")

    log("")
    log("  ---- 4 指标 ----")
    log("  %-8s %5s %5s %5s %5s %8s %8s %8s %8s %8s" %
        ("source", "TP", "FP", "TN", "FN", "P", "R", "F1", "FPR", "Acc"))
    for lbl, m in ((args.label_fp32, m32), (args.label_int8, m8)):
        log("  %-8s %5d %5d %5d %5d %8.4f %8.4f %8.4f %8.4f %8.4f" %
            (lbl, m["tp"], m["fp"], m["tn"], m["fn"],
             m["precision"], m["recall"], m["f1"], m["fpr"], m["acc"]))
    log("  %-8s %5s %5s %5s %5s %8s %8s %8s %8s %8s" %
        ("delta", m8["tp"] - m32["tp"], m8["fp"] - m32["fp"],
         m8["tn"] - m32["tn"], m8["fn"] - m32["fn"],
         "%+.4f" % (m8["precision"] - m32["precision"]),
         "%+.4f" % (m8["recall"] - m32["recall"]),
         "%+.4f" % (m8["f1"] - m32["f1"]),
         "%+.4f" % (m8["fpr"] - m32["fpr"]),
         "%+.4f" % (m8["acc"] - m32["acc"])))

    # ---- 决策翻转分解 ----
    log("")
    log("  ---- 决策翻转分解 (thr=0.5) ----")
    flips = d8 != d32
    cats = {
        "FP32 TP -> INT8 FN  (变差: 漏报)": (d32 == 1) & (d8 == 0) & (atk == 1),
        "FP32 TN -> INT8 FP  (变差: 误报)": (d32 == 0) & (d8 == 1) & (atk == 0),
        "FP32 FP -> INT8 TN  (变好: 误报消失)": (d32 == 1) & (d8 == 0) & (atk == 0),
        "FP32 FN -> INT8 TP  (变好: 漏报消失)": (d32 == 0) & (d8 == 1) & (atk == 1),
    }
    for k, mask in cats.items():
        log(f"    {k:34s} {int(mask.sum()):5d}")
    log(f"    {'合计翻转':34s} {int(flips.sum()):5d}")

    # ---- 概率分布 ----
    log("")
    log("  ---- 窗口最大概率分布 (仅两方都有报警的窗) ----")
    both = hit8 & hit32
    if both.any():
        q = np.percentile(h8[both], [0, 25, 50, 75, 90, 99, 100])
        log("    INT8  : " + "  ".join(f"{x:.3f}" for x in q))
        q = np.percentile(h32[both], [0, 25, 50, 75, 90, 99, 100])
        log("    FP32  : " + "  ".join(f"{x:.3f}" for x in q))
        d = np.abs(h8[both] - h32[both])
        log(f"    |Δprob|: max {d.max():.4f}  median {np.median(d):.4f}  "
            f"p99 {np.percentile(d, 99):.4f}  mean {d.mean():.4f}")
        # 量化偏差 (有符号)
        sd = h8[both] - h32[both]
        log(f"    有符号 Δ (INT8-FP32): mean {sd.mean():+.4f}  "
            f"median {np.median(sd):+.4f}  "
            f"偏大窗数 {(sd > 0).sum()}  偏小窗数 {(sd < 0).sum()}")
        # 边界带: 概率在 thr 附近的窗最容易翻转
        band = (np.minimum(h8[both], h32[both]) < 0.6) & \
               (np.maximum(h8[both], h32[both]) > 0.4)
        log(f"    落在 [0.4, 0.6] 边界带的窗: {int(band.sum())} / {int(both.sum())} "
            f"({100.0 * band.mean():.1f}%)")

    # ---- FP / FN 集合变化 ----
    log("")
    log("  ---- 错误集合变化 ----")
    fp32_fp = set(np.where((d32 == 1) & (atk == 0))[0].tolist())
    int8_fp = set(np.where((d8 == 1) & (atk == 0))[0].tolist())
    fp32_fn = set(np.where((d32 == 0) & (atk == 1))[0].tolist())
    int8_fn = set(np.where((d8 == 0) & (atk == 1))[0].tolist())
    log(f"    FP 集合: FP32 {len(fp32_fp)}  INT8 {len(int8_fp)}  "
        f"共有 {len(fp32_fp & int8_fp)}  新增 {len(int8_fp - fp32_fp)}  "
        f"消失 {len(fp32_fp - int8_fp)}")
    log(f"    FN 集合: FP32 {len(fp32_fn)}  INT8 {len(int8_fn)}  "
        f"共有 {len(fp32_fn & int8_fn)}  新增 {len(int8_fn - fp32_fn)}  "
        f"消失 {len(fp32_fn - int8_fn)}")

    # ---- 跟 PC FP32 参考的偏离 ----
    # 注意: 只能在该 MCU **自己报了警**的窗上比 (没报警的窗日志里没有 prob,
    # 拿不到值; 这些窗的后果已经体现在 FN 计数里).
    # 分母不同, 跟上面的"翻转分解"不是同一个口径:
    #   上面的翻转 = 两个 MCU 之间的**决策**分歧 (全 4664 窗)
    #   这里的翻转 = 该 MCU 与 PC 参考的**决策**分歧 (仅该 MCU 报警的窗)
    log("")
    log("  ---- 各自跟 PC FP32 参考的偏离 (仅该 MCU 报警的窗) ----")
    for lbl, h, hit in ((args.label_fp32, h32, hit32), (args.label_int8, h8, hit8)):
        msk = hit
        dp = np.abs(h[msk] - p_pc[msk])
        fl = int(((h[msk] >= THR) != (p_pc[msk] >= THR)).sum())
        log(f"    {lbl:8s} 报警窗 {int(msk.sum()):5d}  "
            f"max|Δprob| {dp.max():.4f}  median {np.median(dp):.4f}  "
            f"p99 {np.percentile(dp, 99):.4f}  决策翻转 {fl}")

    # ---- 全窗决策分歧 (INT8 vs PC 参考, 把"沉默"也算进来) ----
    # PC 参考在所有部署区窗都有概率; MCU 没报警 => 决策 0.
    # 前提: 该窗确实被 MCU 推理过 (在 hit 或在"沉默但被推理"的集合里).
    # 这里用 PC 参考作为 FP32 的替身 (已证 FP32 与 PC 决策完全一致).
    log("")
    log("  ---- INT8 全窗决策 vs PC 参考 (把沉默窗算作决策 0) ----")
    d_pc = (p_pc >= THR).astype(int)
    fl_all = int((d8 != d_pc).sum())
    tp_f = int(((d8 == 0) & (d_pc == 1)).sum())   # PC 说攻击, INT8 沉默
    fp_f = int(((d8 == 1) & (d_pc == 0)).sum())
    log(f"    分歧总数 {fl_all}  (其中 INT8 沉默漏掉的 {tp_f}, INT8 多报的 {fp_f})")
    log(f"    其中 INT8 沉默的那 {tp_f} 窗, 在 INT8 vs FP32 翻转表里记为 "
        f"'FP32 TP -> INT8 FN'")

    # ---- 最大的分歧窗 (给定位用) ----
    log("")
    log("  ---- 分歧最大的窗 (前 12, 按 |Δprob| 降序) ----")
    both = hit8 & hit32
    if both.any():
        idx = np.where(both)[0]
        d = np.abs(h8[idx] - h32[idx])
        order = np.argsort(-d)[:12]
        log("    %6s %10s %10s %10s %6s" % ("w_local", "PC", "INT8", "FP32", "atk"))
        for k in order:
            i = idx[k]
            log("    %6d %10.6f %10.6f %10.6f %6d" %
                (i, p_pc[i], h8[i], h32[i], atk[i]))

    res = dict(
        n_windows=int(n),
        int8=dict(label=args.label_int8, log=os.path.relpath(args.int8_log, ROOT),
                  metrics=m8, n_alarm_windows=int(hit8.sum())),
        fp32=dict(label=args.label_fp32, log=os.path.relpath(args.fp32_log, ROOT),
                  metrics=m32, n_alarm_windows=int(hit32.sum())),
        flips={k: int(v.sum()) for k, v in cats.items()},
        n_flips=int(flips.sum()),
    )
    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        log(f"\n  -> {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
