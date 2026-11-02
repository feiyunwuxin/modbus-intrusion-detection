"""
analyze_20261021_final.py -- 收尾:
  A. 验证 H743 prob_x1000 = min(floor(prob*1000), 999)
  B. 分区域指标 (全量 / 部署区 200001+) 与阈值扫描
  C. GUI 记账差额的来源
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

hp, hf = [], []
with open(LOG, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["event"] == "ALARM":
            hp.append(int(r["prob_x1000"])); hf.append(int(r["frame_idx"]))
hp = np.array(hp); hf = np.array(hf)

print("=" * 90)
print("[A] H743 prob_x1000 的取整方式")
print("=" * 90)
sel = np.nonzero(probs >= 0.5)[0]
sim = probs[sel]
for name, k in (("floor", np.floor(sim * 1000).astype(int)),
                ("floor + clamp999", np.clip(np.floor(sim * 1000).astype(int), 0, 999))):
    m = np.bincount(k, minlength=1001); h = np.bincount(hp, minlength=1001)
    bad = np.nonzero(m != h)[0]
    print("  %-16s 完全一致 = %-5s  不同值 %d 个 %s"
          % (name, bool((m == h).all()), len(bad),
             ("(" + ", ".join("p=%.3f sim=%d h743=%d" % (v / 1000, m[v], h[v]) for v in bad[:5]) + ")") if len(bad) else ""))

print()
print("=" * 90)
print("[B] 分区域指标")
print("=" * 90)


def rep(tag, mask):
    det = np.zeros(len(probs), dtype=bool)
    det[sel] = True
    det = det & mask; a = atk & mask
    tp = int((det & a).sum()); fp = int((det & ~a).sum())
    fn = int((~det & a).sum())
    n_norm = int((mask & ~atk).sum())
    tn = n_norm - fp
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    print("  %-34s n=%5d  TP=%5d FP=%4d FN=%5d TN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (tag, mask.sum(), tp, fp, fn, tn, P, R, 2 * P * R / max(P + R, 1e-9), fp / max(n_norm, 1)))


allm = np.ones(len(probs), dtype=bool)
dep = frames >= 200001          # 部署区 (H743 部署网格起点)
testreg = frames >= 219701      # 官方 test split 区
rep("全量 274628 帧", allm)
rep("部署区 (frame>=200001)", dep)
rep("官方 test 区 (frame>=219701)", testreg)
rep("train/val 区 (frame<200001)", ~dep)

print()
print("  部署区阈值扫描 (H743 部署网格, 亚秒数据集):")
det_full = np.zeros(len(probs), dtype=bool); det_full[sel] = True
for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    det = (probs >= thr) & dep
    a = atk & dep
    tp = int((det & a).sum()); fp = int((det & ~a).sum())
    fn = int((~det & a).sum())
    n_norm = int((dep & ~atk).sum()); tn = n_norm - fp
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    print("    thr=%.2f  TP=%5d FP=%4d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (thr, tp, fp, fn, P, R, 2 * P * R / max(P + R, 1e-9), fp / max(n_norm, 1)))

print()
print("=" * 90)
print("[C] GUI 记账差额来源")
print("=" * 90)
det = np.zeros(len(probs), dtype=bool); det[sel] = True
true_tp = int((det & atk).sum()); true_fp = int((det & ~atk).sum())
gui_tp, gui_fp, gui_det = 6282, 391, 7287
print("  GUI 报警数 %d, 真值报警数 %d" % (gui_det, int(det.sum())))
print("  GUI 计入 TP+FP = %d,  有 %d 个报警**根本没进 TP/FP 任何一边**" % (gui_tp + gui_fp, gui_det - gui_tp - gui_fp))
print("  真值: TP=%d FP=%d" % (true_tp, true_fp))
print("  → GUI 把 %d 个真 TP 记成了 FP, %d 个真 TP 直接丢弃 (GUI FN 比真值多 816)"
      % (gui_fp - true_fp, true_tp - gui_tp))
print("  → 净效果: F1 被低报 %.4f, FPR 被高报 %.4f"
      % (2 * (true_tp / (true_tp + true_fp)) * (true_tp / (true_tp + 1723))
         / ((true_tp / (true_tp + true_fp)) + (true_tp / (true_tp + 1723))) - 0.8109,
         gui_fp / (gui_fp + 7952) - true_fp / (true_fp + 8154)))
