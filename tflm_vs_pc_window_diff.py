"""
tflm_vs_pc_window_diff.py -- MCU TFLM 上机日志 vs PC 侧各方案的 **逐窗** 决策对比.

背景
----
2026-10-28 用户用 TFLM 固件跑了全量 (17164 窗), 摘要:
    TP 7097 FP 189 TN 8154 FN 1724  F1 0.8812  R 0.8046  P 0.9741  FPR 0.0227
这跟 PC 侧 **Hybrid-INT16** 的全量预测逐个计数完全一样, 跟 Route E FP32 差 1 个 TP。
到底是哪一窗? 本脚本定位到窗。

做法
----
* 窗口标签 (attack) 和 MCU 报警 (alarm) 用 **真源码** 回放器
  (experiments/tflm_fp32/tools/replay_window_eval.py, 从 rtu_replay_gui.py exec 出
  WindowEvaluator 类), 不 Python 复刻。
* PC 侧决策: 四个方案的 prob 在 thr=0.5 下的判定。
* 窗口网格一致: 都是 phase=0 的 17164 窗, 第 W 窗覆盖帧 [16W, 16W+15]。
  部署区 = W >= 12500 (frame >= 200001)。

用法: python tflm_vs_pc_window_diff.py <tflm_log.csv>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments/tflm_fp32/tools"))

from replay_window_eval import load_evaluator, replay   # 真源码回放器

WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
OUT = os.path.join(HERE, "tflm_vs_pc_window_diff.json")


def pc_probs():
    """四个方案的 PC 侧 prob (全 17164 窗)."""
    out = {}
    out["Route E FP32"] = np.load(os.path.join(HERE, "fullsim_f32.npz"))["probs"]
    out["Hybrid-INT8"] = np.load(os.path.join(HERE, "fullsim_hybrid.npz"))["probs"]
    out["全量 INT8"] = np.load(os.path.join(HERE, "fullsim_int8.npz"))["probs"]
    out["Hybrid-INT16"] = np.load(os.path.join(HERE, "fullsim_int16.npz"))["probs"]
    return out


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "logs/test_20261027-232828_auto-40ms.csv")
    cls = load_evaluator(r"C:/work/Claude/Issue/rtu_replay_gui.py")
    ev, pw = replay(cls, log)
    m = ev.get_metrics()
    print("=== MCU 日志回放 (真源码 WindowEvaluator): %s" % os.path.basename(log))
    print("  windows=%d TP=%d FP=%d TN=%d FN=%d F1=%.4f R=%.4f P=%.4f"
          % (m["n_windows"], m["tp"], m["fp"], m["tn"], m["fn"],
             m["f1"], m["recall"], m["precision"]))

    d = np.load(WIN_NPZ)
    frames = d["frames"]
    atk = d["atk"]
    n = min(len(pw), len(frames))
    mcu = np.array([bool(al) for _, _, al in pw[:n]])
    mcu_att = np.array([bool(a) for _, a, _ in pw[:n]])
    print("  日志窗数 %d, PC 窗数 %d" % (len(pw), len(frames)))

    # 攻击标签一致性 (数据回放是否对齐)
    mism_att = int((mcu_att[:n] != atk[:n]).sum())
    print("  attack 标签跟 PC 不一致的窗: %d / %d" % (mism_att, n))
    if mism_att:
        bad = np.where(mcu_att[:n] != atk[:n])[0][:10]
        print("    前 10 个:", [(int(b), bool(mcu_att[b]), bool(atk[b])) for b in bad])
        return

    P = pc_probs()
    dep = frames[:n] >= 200001
    print("\n%-14s %-28s %-28s" % ("PC 方案", "全量 (n=%d)" % n, "部署区 (n=%d)" % int(dep.sum())))
    print("%-14s %8s %8s %8s %8s %8s %8s %8s %8s"
          % ("", "F1", "FPR", "TP", "FP", "F1", "FPR", "TP", "FP"))
    rows = []
    for name, p in P.items():
        det = p[:n] >= 0.5
        flip = det != mcu
        # 全量
        tp = int((det & atk[:n]).sum()); fp = int((det & ~atk[:n]).sum())
        fn = int((~det & atk[:n]).sum()); tn = n - tp - fp - fn
        Pm = tp / max(tp + fp, 1); Rm = tp / max(tp + fn, 1)
        f1 = 2 * Pm * Rm / max(Pm + Rm, 1e-9); fpr = fp / max(tn + fp, 1)
        # 部署区
        dd = det[dep]; aa = atk[:n][dep]
        tp2 = int((dd & aa).sum()); fp2 = int((dd & ~aa).sum())
        fn2 = int((~dd & aa).sum()); tn2 = int(dep.sum()) - tp2 - fp2 - fn2
        P2 = tp2 / max(tp2 + fp2, 1); R2 = tp2 / max(tp2 + fn2, 1)
        f1_2 = 2 * P2 * R2 / max(P2 + R2, 1e-9); fpr2 = fp2 / max(tn2 + fp2, 1)
        print("%-14s %8.4f %8.4f %8d %8d %8.4f %8.4f %8d %8d"
              % (name, f1, fpr, tp, fp, f1_2, fpr2, tp2, fp2))
        rows.append((name, int(flip.sum()),
                     [int(w) for w in np.where(flip)[0][:10]]))

    print("\n--- MCU vs PC 逐窗决策翻转 ---")
    for name, nf, sample in rows:
        print("  %-14s 翻转 %d / %d   %s" % (name, nf, n, sample))

    import json
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"log": log, "mcu_metrics": m,
                   "flips": {r[0]: r[1] for r in rows},
                   "flip_samples": {r[0]: r[2] for r in rows},
                   "n": n, "n_deploy": int(dep.sum())}, f, indent=2, ensure_ascii=False)
    print("\n已写入 %s" % OUT)


if __name__ == "__main__":
    main()
