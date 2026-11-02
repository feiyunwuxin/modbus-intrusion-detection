"""
analyze_20261021_align.py -- 逐窗口对齐 H743 实测 alarm 与仿真.

思路: 既然 prob 直方图逐档零误差, 就把两边 prob>=0.5 的序列按顺序一一对齐,
      反推出每个 alarm 对应的仿真窗口号, 再用仿真标签算真值指标.
      这样完全绕开 GUI WindowEvaluator 的记账逻辑.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOG = os.path.join(HERE, "logs", "test_20261021-003650_auto-40ms.csv")
NPZ = os.path.join(HERE, "fullsim_f32.npz")

d = np.load(NPZ)
probs, atk, frames = d["probs"], d["atk"].astype(bool), d["frames"]

rows = []
with open(LOG, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

alarms = [(int(r["frame_idx"]), int(r["prob_x1000"])) for r in rows if r["event"] == "ALARM"]
sends = [r for r in rows if r["event"] == "SEND"]
print("SEND 行数 = %d, ALARM 行数 = %d" % (len(sends), len(alarms)))

hp = np.array([p for _, p in alarms])          # H743 报的 prob*1000
hf = np.array([f for f, _ in alarms])          # H743 报的 frame

# ── A. 截断 vs 四舍五入 ──────────────────────────────────────────────────
sim = probs[probs >= 0.5]
floor_k = np.floor(sim * 1000).astype(int)
round_k = np.round(sim * 1000).astype(int)
print()
print("[A] H743 prob_x1000 的取整方式")
for name, k in (("截断 floor", floor_k), ("四舍五入 round", round_k)):
    m = np.bincount(k, minlength=1001)
    h = np.bincount(hp, minlength=1001)
    print("  %-14s 与 H743 逐值计数完全一致 = %s   (不同值个数 %d)"
          % (name, bool((m == h).all()), int((m != h).sum())))
    if not (m == h).all():
        bad = np.nonzero(m != h)[0][:8]
        print("       例: " + ", ".join("p=%.3f sim=%d h743=%d" % (v / 1000, m[v], h[v]) for v in bad))

# ── B. 逐 alarm 对齐仿真窗口 ────────────────────────────────────────────
sel = np.nonzero(probs >= 0.5)[0]              # 仿真里被报警的窗口号
print()
print("[B] alarm 序列 vs 仿真窗口序列")
print("  报警窗口数: H743=%d  仿真=%d" % (len(hp), len(sel)))

# H743 报的 frame 应该等于该窗口的结束帧; 先看 frame 与仿真 frames 的关系
sf = frames[sel]
print("  仿真窗口结束帧 (前 8 个): %s ..." % sf[:8].tolist())
print("  H743 alarm frame (前 8 个): %s ..." % hf[:8].tolist())

# 逐个比较: H743 frame 与仿真窗口结束帧
diffs = hf - sf
vals, cnts = np.unique(diffs, return_counts=True)
print("  H743_frame - 仿真窗口结束帧 的取值分布:")
for v, c in zip(vals, cnts):
    print("     %+8d  : %5d 次" % (v, c))

# 能不能靠 frame 直接定位窗口?
if (diffs == 0).all():
    print("  ✅ 完全对齐: H743 的 frame == 仿真窗口结束帧")
else:
    # 用 (frame // 16) 反推窗口序号, 看看是否和 sel 一致
    w_from_frame = hf // 16
    ok = (w_from_frame == sel).sum()
    print("  frame//16 == 仿真窗口序号 的匹配数: %d / %d" % (ok, len(sel)))
    w2 = (hf - 1) // 16
    ok2 = (w2 == sel).sum()
    print("  (frame-1)//16 == 仿真窗口序号 的匹配数: %d / %d" % (ok2, len(sel)))

# ── C. 用仿真标签算真值 ─────────────────────────────────────────────────
print()
print("[C] 真值指标 (用仿真窗口标签, 阈值 0.5)")
det = np.zeros(len(probs), dtype=bool)
det[sel] = True
tp = int((det & atk).sum()); fp = int((det & ~atk).sum())
fn = int((~det & atk).sum()); tn = int((~det & ~atk).sum())
P = tp / (tp + fp); R = tp / (tp + fn)
print("  TP=%5d FP=%5d FN=%5d TN=%5d" % (tp, fp, fn, tn))
print("  Precision=%.4f Recall=%.4f F1=%.4f FPR=%.4f Acc=%.4f"
      % (P, R, 2 * P * R / (P + R), fp / (fp + tn), (tp + tn) / len(atk)))

# ── D. GUI summary 对比 ─────────────────────────────────────────────────
print()
print("[D] GUI summary 对比")
print("  GUI:        Detected=7287 TP=6282 FP=391 FN=2539 TN=7952  R=0.7122 FPR=0.0469 F1=0.8109")
print("  真值(仿真):Detected=%d TP=%d FP=%d FN=%d TN=%d  R=%.4f FPR=%.4f F1=%.4f"
      % (det.sum(), tp, fp, fn, tn, R, fp / (fp + tn), 2 * P * R / (P + R)))
print("  差: Detected %+d   TP %+d   FP %+d   FN %+d   TN %+d"
      % (int(det.sum()) - 7287, tp - 6282, fp - 391, fn - 2539, tn - 7952))

# ── E. FN 分析: 漏报的 attack window 都在哪 ────────────────────────────
miss = np.nonzero(atk & ~det)[0]
print()
print("[E] 漏报 (FN) 的 %d 个 attack window" % len(miss))
print("  窗口号前 12 个: %s" % miss[:12].tolist())
print("  这些窗口的 prob 分布:")
allp = probs
for lo, hi in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 1.01)):
    c = int(((allp[miss] >= lo) & (allp[miss] < hi)).sum())
    print("     [%.2f,%.2f) : %5d  (%.1f%%)" % (lo, hi, c, 100 * c / len(miss)))
