"""
analyze_20261021_test.py -- 校验 2026-10-21 全量回放测试.

三件事:
  1. 仿真 (F32 数据集 + 修复后 conv) 的全量 per-window 指标
  2. 仿真 prob >= 0.5 的窗口数 vs H743 实测 alarm 数, 以及 prob 分布是否逐个吻合
  3. 用 H743 实测的 alarm frame + CSV ground truth 独立算一遍真值指标
     (绕开 GUI WindowEvaluator 的记账 bug)
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOG = os.path.join(HERE, "logs", "test_20261021-003650_auto-40ms.csv")
CSV = os.path.join(HERE, "IanArffDatasetF32.csv")
NPZ = os.path.join(HERE, "fullsim_f32.npz")


def metrics(pred, atk, thr=0.5):
    det = pred >= thr
    tp = int((det & atk).sum()); fp = int((det & ~atk).sum())
    fn = int((~det & atk).sum()); tn = int((~det & ~atk).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=P, R=R, F1=2 * P * R / max(P + R, 1e-9),
                FPR=fp / max(fp + tn, 1), Acc=(tp + tn) / len(atk))


d = np.load(NPZ)
probs, atk, frames = d["probs"], d["atk"].astype(bool), d["frames"]
print("=" * 100)
print("[1] 仿真 (IanArffDatasetF32.csv + 修复后 conv), 全量 %d window" % len(probs))
print("=" * 100)
m = metrics(probs, atk, 0.5)
print("  thr=0.50  TP=%5d FP=%5d FN=%5d TN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f Acc=%.4f"
      % (m["TP"], m["FP"], m["FN"], m["TN"], m["P"], m["R"], m["F1"], m["FPR"], m["Acc"]))
print()
for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    mm = metrics(probs, atk, thr)
    print("  thr=%.2f  TP=%5d FP=%5d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (thr, mm["TP"], mm["FP"], mm["FN"], mm["P"], mm["R"], mm["F1"], mm["FPR"]))

# ── 2. H743 实测 alarm ────────────────────────────────────────────────────
print()
print("=" * 100)
print("[2] H743 实测 alarm vs 仿真")
print("=" * 100)
h743 = []
with open(LOG, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["event"] == "ALARM":
            h743.append((int(r["frame_idx"]), int(r["prob_x1000"])))
hprob = np.array([p / 1000.0 for _, p in h743])
hframe = np.array([f for f, _ in h743])
print("  H743 alarm 数: %d   仿真 prob>=0.5 的窗口数: %d"
      % (len(hprob), int((probs >= 0.5).sum())))

sim_sel = np.sort(probs[probs >= 0.5])
h743_sel = np.sort(hprob)
print("  alarm prob 范围: H743 [%.3f, %.3f]   仿真 [%.3f, %.3f]"
      % (hprob.min(), hprob.max(), sim_sel.min(), sim_sel.max()))
print("  全体 prob 直方图对比 (H743 / 仿真):")
for lo, hi in ((0.5, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.01)):
    a = int(((hprob >= lo) & (hprob < hi)).sum())
    b = int(((sim_sel >= lo) & (sim_sel < hi)).sum())
    print("    [%.2f,%.2f)  H743=%5d  仿真=%5d  差=%+d" % (lo, hi, a, b, b - a))

# 按 prob 值做多重集匹配 (H743 的 prob 是 3 位小数)
hkey = np.round(hprob, 3)
skey = np.round(sim_sel, 3)
hd = {round(float(v), 3): int(c) for v, c in zip(*np.unique(hkey, return_counts=True))}
sd = {round(float(v), 3): int(c) for v, c in zip(*np.unique(skey, return_counts=True))}
allv = sorted(set(hd) | set(sd))
diff = {v: sd.get(v, 0) - hd.get(v, 0) for v in allv
        if sd.get(v, 0) != hd.get(v, 0)}
print("  prob 多重集差异 (仅列出不相等的值, 最多 15 个):")
for v, dd in sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:15]:
    print("    prob=%.3f  仿真比 H743 多 %+d" % (v, dd))
print("  不相等的 prob 值个数: %d / %d  (完全一致 = 逐窗口 bit-level 复现)" % (len(diff), len(allv)))

# ── 3. 用 alarm frame + ground truth 独立算真值 ──────────────────────────
print()
print("=" * 100)
print("[3] 用 H743 实测 alarm frame + CSV ground truth 独立算 (绕开 GUI WindowEvaluator)")
print("=" * 100)
# ground truth: CSV 的 binary 列
gt = np.zeros(274629, dtype=np.int8)
with open(CSV, encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for i, r in enumerate(rd, start=1):
        gt[i] = int((r.get("binary result") or "0").strip() or 0)

# H743 的 window 网格: window k (1-indexed) = 帧 [16k-15 .. 16k]
# alarm 报的 frame 可能 = 16k (准时) 或 16k+1 (主循环晚 1 帧)
# 两种归一: (a) frame//16  (b) (frame-1)//16
for name, fn in (("frame//16", lambda fr: fr // 16),
                 ("(frame-1)//16", lambda fr: (fr - 1) // 16)):
    w = np.array([fn(fr) for fr in hframe])
    w = np.clip(w, 1, 17164)
    pred = np.zeros(17165, dtype=bool)
    pred[w] = True
    atk_w = np.zeros(17165, dtype=bool)
    # window k 的 attack = 帧 [16k-15..16k] 任一 attack
    csum = np.concatenate([[0], np.cumsum(gt[:274628])])
    for k in range(1, 17165):
        atk_w[k] = (csum[16 * k] - csum[16 * k - 16]) > 0
    mm = metrics(pred[1:], atk_w[1:], 0.5)
    print("  %-14s TP=%5d FP=%5d FN=%5d TN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (name, mm["TP"], mm["FP"], mm["FN"], mm["TN"], mm["P"], mm["R"], mm["F1"], mm["FPR"]))

print()
print("  GUI summary 报的:      TP= 6282 FP= 391 FN=2539  R=0.7122 FPR=0.0469 F1=0.8109")
