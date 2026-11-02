"""
pc_canonical_eval.py -- PC sim 的规范评估口径.

两个问题:
 1) PC sim (训练模型) 在同一份 csv (frames 200000-274628) 上的真实 per-window 指标是多少?
    (sliding 每帧推理 + per-window max 归并, 跟 ids_worker / WindowEvaluator 一致)
 2) 之前记忆里的 "PC FPR 2.27% / Recall 95%" 是哪来的? -> 训练区 vs 测试区数据漂移
"""
import csv
import math

import numpy as np

W = 16
d = np.load("parity_pysim_probs.npz")          # PyTorch .pt 模型, 每帧 sliding prob
frames = d["frames"]
probs = d["probs"]
lab_frames = d["label_frames"]
labels = d["labels"]

pc = dict(zip(frames.tolist(), probs.tolist()))
gl = dict(zip(lab_frames.tolist(), labels.tolist()))

n_win = int(frames[-1]) // W
atk = [any(gl.get(f, 0) == 1 for f in range(w * W + 1, w * W + W + 1)) for w in range(n_win)]
n_atk = sum(atk)
n_norm = n_win - n_atk
print("windows=%d attack=%d normal=%d" % (n_win, n_atk, n_norm))


def metrics(prob_win, thr, tag):
    tp = fp = fn = 0
    for w in range(n_win):
        p = prob_win[w]
        det = p is not None and p >= thr
        if atk[w]:
            if det:
                tp += 1
            else:
                fn += 1
        elif det:
            fp += 1
    P = tp / max(tp + fp, 1)
    R = tp / max(n_atk, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    print("  %-38s thr=%.2f TP=%5d FP=%5d FN=%5d P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (tag, thr, tp, fp, fn, P, R, F1, fp / max(n_norm, 1)))
    return tp, fp, fn


print()
print("[1] PC sim (PyTorch 训练模型) 在 frames 200000+ 上的 per-window 指标")
last = [pc.get((w + 1) * W) for w in range(n_win)]
wmax = []
for w in range(n_win):
    vals = [pc.get(f) for f in range(w * W + 1, w * W + W + 1) if f in pc]
    wmax.append(max(vals) if vals else None)

for thr in (0.5, 0.7, 0.9):
    metrics(last, thr, "non-overlap 末帧 (H743 同语义)")
for thr in (0.5, 0.7, 0.9):
    metrics(wmax, thr, "sliding + per-window max (PC 规范口径)")

print()
print("[2] 数据漂移检查: 训练区 (frames 0..192k) vs 测试区 (200k..274k)")


def eval_range(start, end, tag):
    sub = [(f, p) for f, p in zip(frames.tolist(), probs.tolist()) if start <= f <= end]
    if not sub:
        print("   %s: 无数据" % tag)
        return
    lab = {f: gl.get(f, 0) for f, _ in sub}
    mx = max(f for f, _ in sub)
    nw = mx // W
    a = [any(lab.get(f, 0) == 1 for f in range(w * W + 1, w * W + W + 1)) for w in range(nw)]
    na = sum(a)
    nn = nw - na
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tp = fp = fn = 0
        for w in range(nw):
            vals = [p for f, p in sub if w * W + 1 <= f <= w * W + W]
            if not vals:
                continue
            det = max(vals) >= thr
            if a[w]:
                tp += det
                fn += (not det)
            else:
                fp += det
        P = tp / max(tp + fp, 1)
        R = tp / max(na, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        FPR = fp / max(nn, 1)
        if best is None or F1 > best[1]:
            best = (thr, F1, tp, fp, fn, P, R, FPR)
    thr, F1, tp, fp, fn, P, R, FPR = best
    print("   %-24s windows=%5d atk=%5d | best thr=%.2f TP=%5d FP=%5d FN=%5d "
          "P=%.4f R=%.4f F1=%.4f FPR=%.4f" % (tag, nw, na, thr, tp, fp, fn, P, R, F1, FPR))


# 测试区 = 200000+ (本次 replay 用的段)
eval_range(1, 200000, "训练区 frames 1..200k")
eval_range(200000, 74628 + 0, "测试区 frames 200k..274k")
