"""
analyze_20261021_gui.py -- 证明 GUI summary 低估的机制:

  H743 报的 frame = 窗口结束帧 + {0 或 +1} (主循环晚 1 帧)
  GUI 归窗: W = (frame_idx - 1) // 16
    frame = 16k   -> W = k-1  ✅ 正确
    frame = 16k+1 -> W = k    ❌ 落到下一个窗口
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
sel = np.nonzero(probs >= 0.5)[0]          # 报警窗口的 0-based 序号
sf = frames[sel]                            # 对应的窗口结束帧

hp, hf = [], []
with open(LOG, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["event"] == "ALARM":
            hp.append(int(r["prob_x1000"])); hf.append(int(r["frame_idx"]))
hp = np.array(hp); hf = np.array(hf)

print("=" * 96)
print("[1] alarm 逐条对齐 (按时间序一一对应)")
print("=" * 96)
print("  alarm 数 %d, 仿真报警窗口数 %d, 顺序对齐后 frame 差值分布:" % (len(hf), len(sel)))
delta = hf - sf
u, c = np.unique(delta, return_counts=True)
for v, n in zip(u, c):
    print("     H743_frame - 窗口结束帧 = %+d : %5d 次 (%.1f%%)" % (v, n, 100 * n / len(delta)))

print()
print("=" * 96)
print("[2] GUI 归窗公式的后果")
print("=" * 96)
w_gui = (hf - 1) // 16                     # GUI: W = (frame-1)//16
w_true = sel                               # 真值: 仿真窗口序号
off = w_gui - w_true
u, c = np.unique(off, return_counts=True)
print("  GUI 归的窗 - 真值窗 的偏移:")
for v, n in zip(u, c):
    kind = "✅ 归对" if v == 0 else "❌ 归错 (落到下一个窗)"
    print("     %+d : %5d 次 (%.1f%%)  %s" % (v, n, 100 * n / len(off), kind))

late = delta == 1                          # H743 迟报 1 帧的 alarm
early = delta == 0
print()
print("  交叉验证: H743 迟报(+1) 的 alarm 是否 100%% 被 GUI 归错?")
print("     迟报 %d 个, 其中 GUI 归错 %d 个, 归对 %d 个"
      % (late.sum(), (off[late] != 0).sum(), (off[late] == 0).sum()))
print("     准时 %d 个, 其中 GUI 归错 %d 个" % (early.sum(), (off[early] != 0).sum()))

print()
print("=" * 96)
print("[3] 归错的后果: 816 个真 TP 怎么丢的")
print("=" * 96)
mis = off != 0
print("  被 GUI 归错的 alarm: %d 个" % mis.sum())
print("     其中真窗是 attack 的: %5d  <- 这些真 TP 被记成别的东西" % int((mis & atk[w_true]).sum()))
print("     其中真窗是 normal 的: %5d  <- 这些是本该记的 FP" % int((mis & ~atk[w_true]).sum()))
print()
print("  GUI summary: Detected=7287  TP=6282 FP=391 FN=2539 TN=7952")
print("  真值(仿真): Detected=7287  TP=7098 FP=189 FN=1723 TN=8154")
print()
print("  差额分解:")
print("     真 TP 少了 7098-6282 = %d" % (7098 - 6282))
print("     GUI FP 多了 391-189 = %d   <- 归错到 normal 窗, 记成 FP" % (391 - 189))
print("     报警里没进 TP/FP 的 = 7287-6673 = %d   <- 归错到已评估的窗, rescue 缓冲(maxlen=4)够不着, 直接丢"
      % (7287 - 6673))
print("     两者之和 = %d + %d = %d  == 真 TP 差额 %d ✔"
      % (391 - 189, 7287 - 6673, (391 - 189) + (7287 - 6673), 7098 - 6282))

print()
print("=" * 96)
print("[4] 修掉归窗错位后的预期 summary (离线重算)")
print("=" * 96)
for name, wfix in (("GUI 原样 (frame-1)//16", w_gui),
                   ("修正: 扣掉 frame%16!=0 的迟报", (hf - 1) // 16 - (hf % 16 != 0).astype(int)),
                   ("假设 H743 永不迟报 (frame 恒 = 末帧)", np.minimum((hf - 1) // 16, len(probs) - 1))):
    wfix = np.clip(wfix, 0, len(probs) - 1)
    pred = np.zeros(len(probs), dtype=bool)
    pred[np.unique(wfix)] = True
    tp = int((pred & atk).sum()); fp = int((pred & ~atk).sum())
    fn = int((~pred & atk).sum()); tn = int((~pred & ~atk).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    print("  %-36s Detected=%5d TP=%5d FP=%4d FN=%5d TN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (name, pred.sum(), tp, fp, fn, tn, P, R, 2 * P * R / max(P + R, 1e-9),
             fp / max(fp + tn, 1)))
print()
print("  (注: GUI 的 'Detected' KPI 数的是 alarm 条数, TP/FP 数的是窗口数, 两者本就不同口径)")
print("  (真值 Detected 窗口数 = %d, GUI 原样 = %d, 差 %d = 归窗碰撞)" % (len(sel), len(np.unique(w_gui)), len(sel) - len(np.unique(w_gui))))
