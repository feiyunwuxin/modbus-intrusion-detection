"""
exp_hybrid_conclusion.py -- Hybrid 工程选型的 **PC 侧最终结论**.

四个候选 (都在同一份 23-dim / ch=32 / seed42 checkpoint 上):
  FP32        : Route E 部署真值 (手写 C kernel, 全 FP32 权重, 启动时 memcpy 到 RAM)
  Hybrid-INT8 : 权重 int8 per-channel 存 Flash, 激活 FP32, 运行期 dequant
  Hybrid-INT16: 权重 int16 per-channel 存 Flash, 激活 FP32, 运行期 dequant  ← 本次的方案
  全量 INT8   : CMSIS-NN arm_convolve_s8 (激活也量化), 对照组

输出:
  1) 分区域 4-metrics (Binary, thr=0.5)
  2) 阈值扫描
  3) 相对 FP32 的 paired bootstrap CI + 精确 McNemar
  4) 决策翻转分解
  5) 资源占用表 (Flash/RAM: 来自 Keil .map + 头文件字节数)

口径 (与项目 4-metrics 口径一致):
  部署区   frame >= 200001   (对外报这个)
  官方test frame >= 219701
  train区  frame <  200001   (标定只用这里)
  Binary 指标, thr = 0.5

用法: python exp_hybrid_conclusion.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compare_fp32_vs_hybrid import (ZONES, bootstrap_delta, mcnemar_exact,
                                    metrics)
from sim_hybrid_kernel import load_weights, run_all

WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
FP32_NPZ = os.path.join(HERE, "fullsim_f32.npz")
INT8_NPZ = os.path.join(HERE, "fullsim_int8.npz")
OUT_JSON = os.path.join(HERE, "hybrid_conclusion.json")

# Keil .map 实测 (2026-10-27 编译)
#   ROM = Total ROM Size (Code + RO Data + RW Data)  -> Flash
#   RAM = Total RW  Size (RW Data + ZI Data)         -> SRAM
MAP = {
    "FP32": {"rom": 149684, "ram": 241888},
    "全量 INT8": {"rom": 97292, "ram": 157408},
    "Hybrid-INT16": {"rom": 112656, "ram": 159912},
}
FP32_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h")
I16_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int16_s42.h")
HYB_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_hybrid_s42.h")


def rodata_bytes(header_path):
    """从 header 里算 .rodata 字节数 (权重 + scale + bias)."""
    import re
    txt = open(header_path, encoding="utf-8", errors="replace").read()
    tot = 0
    for m in re.finditer(r"static const\s+(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]", txt):
        ctype, n = m.group(1), int(m.group(3))
        tot += n * ({"int8_t": 1, "int16_t": 2, "int32_t": 4}.get(ctype, 4))
    return tot


def main():
    d = np.load(WIN_NPZ)
    X = d["X"].astype(np.float32)
    atk = d["atk"].astype(bool)
    frames = d["frames"]
    masks = {name: fn(frames) for name, fn in ZONES}
    dep = masks["部署区 (frame>=200001)"]
    print("窗口 %d 个 (部署区 %d / train区 %d)"
          % (len(X), int(dep.sum()), int(masks["train/val 区 (frame<200001)"].sum())))

    # ---------- 跑仿真 ----------
    print("\n[仿真] Hybrid-INT16 (读 gen_int16_weights.py 生成的头) ...")
    t0 = time.time()
    p_i16 = run_all(X, weights_mode="int16", act_mode="fp32", verbose=False)
    print("  用时 %.1fs" % (time.time() - t0))

    p32 = np.load(FP32_NPZ)["probs"]
    p_i8_all = np.load(INT8_NPZ)["probs"]
    print("[仿真] Hybrid-INT8 (现有 hybrid 头) ...")
    p_i8 = run_all(X, weights_mode="int8", act_mode="fp32", verbose=False)

    variants = [
        ("FP32 (Route E 真值)", p32),
        ("Hybrid-INT16", p_i16),
        ("Hybrid-INT8", p_i8),
        ("全量 INT8", p_i8_all),
    ]

    # ---------- 1. 分区域指标 ----------
    print("\n" + "=" * 100)
    print("[1] 分区域 4-metrics (Binary, thr=0.5)")
    print("=" * 100)
    allm = {}
    for zname, m in masks.items():
        print("  --- %s ---" % zname)
        for tag, p in variants:
            mm = metrics(p[m] >= 0.5, atk[m])
            allm["%s | %s" % (tag, zname)] = mm
            print("    %-22s n=%5d TP=%5d FP=%4d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
                  % (tag, mm["n"], mm["TP"], mm["FP"], mm["FN"],
                     mm["P"], mm["R"], mm["F1"], mm["FPR"]))

    # ---------- 2. 阈值扫描 ----------
    print("\n" + "=" * 100)
    print("[2] 阈值扫描 (部署区 frame>=200001)")
    print("=" * 100)
    print("    thr  " + "".join("%-24s" % t for t, _ in variants))
    for thr in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        row = "%-6.2f" % thr
        for tag, p in variants:
            m = metrics(p[dep] >= thr, atk[dep])
            row += "%-24s" % ("F1=%.4f FPR=%.4f" % (m["F1"], m["FPR"]))
        print("    " + row)

    # ---------- 3. 统计检验 ----------
    print("\n" + "=" * 100)
    print("[3] 相对 FP32 的统计检验 (部署区 %d 窗, thr=0.5)" % int(dep.sum()))
    print("=" * 100)
    stats = {}
    for tag, p in variants[1:]:
        f1lo, f1hi, fprlo, fprhi, pbet = bootstrap_delta(p[dep], p32[dep], atk[dep])
        b, c, pv = mcnemar_exact(p[dep], p32[dep], atk[dep])
        flip = int(((p[dep] >= 0.5) != (p32[dep] >= 0.5)).sum())
        stats[tag] = dict(dF1_ci=[f1lo, f1hi], dFPR_ci=[fprlo, fprhi],
                          p_better=pbet, mcnemar_b=b, mcnemar_c=c,
                          mcnemar_p=pv, flips=flip)
        print("  %s" % tag)
        print("    ΔF1  95%%CI = [%+.4f, %+.4f]   P(该变体 F1 更好)=%.3f" % (f1lo, f1hi, pbet))
        print("    ΔFPR 95%%CI = [%+.4f, %+.4f]%s" % (fprlo, fprhi,
              "   ← 不含 0, 差异显著" if (fprlo > 0 or fprhi < 0) else "   ← 含 0, 不可区分"))
        print("    McNemar: 变体独正=%d, FP32独正=%d, p=%s ; 决策翻转=%d"
              % (b, c, ("<1e-6" if pv < 1e-6 else "%.4f" % pv), flip))

    # ---------- 4. 翻转分解 ----------
    print("\n" + "=" * 100)
    print("[4] 决策翻转分解 (部署区, 相对 FP32)")
    print("=" * 100)
    for tag, p in variants[1:]:
        dv = p[dep] >= 0.5
        db = p32[dep] >= 0.5
        a = atk[dep]
        print("  %s:" % tag)
        print("    TP->FN (变差·漏报) %4d    FP->TN (变好) %4d" % ((db & a & ~dv).sum(), (db & ~a & ~dv).sum()))
        print("    TN->FP (变差·误报) %4d    FN->TP (变好) %4d" % ((~db & ~a & dv).sum(), (~db & a & dv).sum()))

    # ---------- 5. 概率漂移 ----------
    print("\n" + "=" * 100)
    print("[5] 概率漂移 (部署区, 相对 FP32)")
    print("=" * 100)
    for tag, p in variants[1:]:
        d = p[dep] - p32[dep]
        lg = np.log(np.clip(p[dep], 1e-12, 1)) - np.log(np.clip(p32[dep], 1e-12, 1))
        print("  %-14s MAE=%.6f  max|Δ|=%.6f  r=%.8f   log比值偏移 均值=%+.5f 中位数=%+.5f"
              % (tag, np.abs(d).mean(), np.abs(d).max(),
                 np.corrcoef(p32[dep], p[dep])[0, 1], lg.mean(), np.median(lg)))

    # ---------- 6. 资源表 ----------
    print("\n" + "=" * 100)
    print("[6] 资源占用")
    print("=" * 100)
    ro32 = rodata_bytes(FP32_HDR)
    ro16 = rodata_bytes(I16_HDR)
    ro8 = rodata_bytes(HYB_HDR)
    print("  权重头 .rodata 字节数:")
    print("    FP32 头         %6d B (%.1f KB)" % (ro32, ro32 / 1024))
    print("    Hybrid-INT16 头 %6d B (%.1f KB)" % (ro16, ro16 / 1024))
    print("    Hybrid-INT8  头 %6d B (%.1f KB)" % (ro8, ro8 / 1024))
    print("  Keil .map 实测 (ROM=Flash, RW=RAM):")
    for k, v in MAP.items():
        print("    %-12s ROM %7d B (%.1f KB)   RAM %7d B (%.1f KB)"
              % (k, v["rom"], v["rom"] / 1024, v["ram"], v["ram"] / 1024))
    est_rom = MAP["FP32"]["rom"] - ro32 + ro16
    est_ram = MAP["FP32"]["ram"] - 20128 * 4
    print("    %-12s ROM %7d B (%.1f KB)   RAM %7d B (%.1f KB)   ← 估算 (仅供对比)"
          % ("(估算参考)", est_rom, est_rom / 1024, est_ram, est_ram / 1024))
    h = MAP["Hybrid-INT16"]
    print("    %-12s ROM %7d B (%.1f KB)   RAM %7d B (%.1f KB)   ← Keil .map 实测 (H743test_HYBRID)"
          % ("Hybrid-INT16", h["rom"], h["rom"] / 1024, h["ram"], h["ram"] / 1024))
    f = MAP["FP32"]
    i8 = MAP["全量 INT8"]
    print("\n  相对 FP32:")
    print("    Flash  %+7d B (%.1f%%)   [FP32 %d -> Hybrid %d]"
          % (h["rom"] - f["rom"], 100.0 * (h["rom"] - f["rom"]) / f["rom"], f["rom"], h["rom"]))
    print("    RAM    %+7d B (%.1f%%)   [FP32 %d -> Hybrid %d]  (FP32 权重 memcpy 到 RAM 的 80KB 省掉了)"
          % (h["ram"] - f["ram"], 100.0 * (h["ram"] - f["ram"]) / f["ram"], f["ram"], h["ram"]))
    print("  相对 全量 INT8:")
    print("    Flash  %+7d B (%.1f%%)   [INT8 %d -> Hybrid %d]"
          % (h["rom"] - i8["rom"], 100.0 * (h["rom"] - i8["rom"]) / i8["rom"], i8["rom"], h["rom"]))
    print("    RAM    %+7d B (%.1f%%)   [INT8 %d -> Hybrid %d]"
          % (h["ram"] - i8["ram"], 100.0 * (h["ram"] - i8["ram"]) / i8["ram"], i8["ram"], h["ram"]))

    out = {
        "zones": allm,
        "stats_vs_fp32_deploy": stats,
        "resources": {
            "rodata_fp32": ro32, "rodata_int16": ro16, "rodata_hybrid_int8": ro8,
            "map": MAP,
            "hybrid_int16_estimate": {"rom": est_rom, "ram": est_ram},
            "hybrid_int16_measured": {"rom": h["rom"], "ram": h["ram"],
                                      "target": "H743test_HYBRID"},
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n已写入 %s" % OUT_JSON)


if __name__ == "__main__":
    main()
