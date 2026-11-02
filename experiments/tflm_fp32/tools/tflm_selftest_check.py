#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tflm_selftest_check.py  --  G2: 校验实机 TFLM controlled-input 自检输出

对照对象:
    PC 参考 : experiments/tflm_fp32/artifacts/tflm_selftest_ref.json
              (TFLite BUILTIN_REF, float32 精确值)
    MCU 实机: 串口日志里 [TFLM-TEST] <case> <logit> <prob> 几行
              (串口工具抓的 .txt/.log, 或直接 stdin 管道)

用法:
    python experiments/tflm_fp32/tools/tflm_selftest_check.py capture.txt
    python experiments/tflm_fp32/tools/tflm_selftest_check.py --stdin < capture.txt
    python experiments/tflm_fp32/tools/tflm_selftest_check.py capture.txt --tol 1e-4

退出码:
    0 = 全部 case 与 PC 参考一致 (且 5 个输出互不相同)
    1 = 有 case 超差 / 缺失
    2 = 5 个输出恒定 (图或运行时在 MCU 上坏了) —— 旧 TFLM 路线的失败签名
    3 = 日志里根本找不到 [TFLM-TEST] 行

容差默认 1e-4: MCU 打印的是 %14.6f, 6 位小数本身就有 <=5e-7 的截断,
再加 float32 累加顺序差异, 1e-4 是既能判真差异又不误报的量级。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REF = os.path.join(ROOT, "experiments", "tflm_fp32", "artifacts", "tflm_selftest_ref.json")

# [TFLM-TEST] ZEROS    42.251404      1.00000000
LINE_RE = re.compile(
    r"\[TFLM-TEST\]\s+(?P<case>[A-Z0-9_]+)\s+"
    r"(?P<logit>[-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+"
    r"(?P<prob>[-+]?\d+\.\d+(?:[eE][-+]?\d+)?)"
)


def log(msg):
    print(msg, flush=True)


def parse(text):
    out = {}
    for m in LINE_RE.finditer(text):
        name = m.group("case")
        # 表头行 "case logit prob" 不会匹配 (case 是小写), 这里再挡一次
        out[name] = (float(m.group("logit")), float(m.group("prob")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", help="串口抓取的日志文件")
    ap.add_argument("--stdin", action="store_true", help="从 stdin 读")
    ap.add_argument("--tol", type=float, default=1e-4, help="logit 绝对容差 (默认 1e-4)")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    if args.stdin:
        text = sys.stdin.read()
    elif args.capture:
        with io.open(args.capture, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        ap.error("要给 capture 文件或 --stdin")

    with io.open(REF, encoding="utf-8") as fh:
        ref = json.load(fh)["cases"]

    got = parse(text)
    log("=" * 78)
    log("G2: TFLM 实机 controlled-input 自检 vs PC 参考")
    log("=" * 78)
    log(f"  参考 : {os.path.relpath(REF, ROOT)}")
    log(f"  容差 : {args.tol:g} (logit 绝对误差)")

    if not got:
        log("\n  ✗ 日志里没有 [TFLM-TEST] 行 —— 自检没跑起来?")
        log("    (检查: 串口发的是不是单字节 'T' + 至少 1 char time 静默)")
        return 3

    rows = []
    all_ok = True
    log("")
    log("  %-8s %16s %16s %12s %s" % ("case", "PC logit", "MCU logit", "|delta|", "verdict"))
    for name, r in ref.items():
        if name not in got:
            rows.append(dict(case=name, status="MISSING"))
            all_ok = False
            log("  %-8s %16.8f %16s %12s %s" % (name, r["logit"], "-", "-", "MISSING"))
            continue
        m_logit, m_prob = got[name]
        d = abs(m_logit - r["logit"])
        ok = d <= args.tol
        all_ok = all_ok and ok
        rows.append(dict(case=name, pc_logit=r["logit"], mcu_logit=m_logit,
                         pc_prob=r["prob"], mcu_prob=m_prob, abs_delta=d,
                         status="OK" if ok else "MISMATCH"))
        log("  %-8s %16.8f %16.8f %12.3e %s" %
            (name, r["logit"], m_logit, d, "OK" if ok else "MISMATCH"))

    # 5 个输出必须互不相同 (旧 TFLM 路线恒定输出的失败签名)
    logits = [v[0] for v in got.values()]
    all_same = len(logits) >= 5 and len(set(logits)) == 1

    log("")
    if all_same:
        log("  ✗ 5 个输出**恒定** -> 图 / 运行时在 MCU 上坏了 (旧 TFLM 失败签名)")
        verdict = 2
    elif not all_ok:
        log("  ✗ 有 case 与 PC 参考不一致 -> 算子数值实现差异, 需逐层 dump 定位")
        verdict = 1
    else:
        log("  ✓ 5 个输出互不相同, 且与 PC 参考一致 -> 运行时 OK, "
            "问题在数据通路 (特征 / 窗口)")
        verdict = 0

    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as fh:
            json.dump(dict(verdict=verdict, tol=args.tol, rows=rows),
                      fh, ensure_ascii=False, indent=2)
        log(f"  -> {os.path.relpath(args.out, ROOT)}")

    return verdict


if __name__ == "__main__":
    sys.exit(main())
