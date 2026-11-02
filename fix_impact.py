"""
fix_impact.py -- 量化 conv 对齐 bug 的影响.

对同一批 4664 个 window (H743 non-overlap 语义) 分别用:
  (a) C 语义   (H743 现在实际跑的, 因果对齐)
  (b) 训练语义 (PyTorch 对称 same padding, 修好 kernel 后)
算 per-window TP/FP/FN, 跟 H743 实测 summary 对齐, 并给出修复后的指标.
"""
import csv
import math
import sys
from collections import deque

import joblib
import numpy as np
import torch

sys.path.insert(0, ".")
from h743_view_sim import (SW_WINDOW, KEEP_18, KEEP_23, MCU_UNUSUAL, build23,
                           load_header_weights, mcu_decode, pc_encode, St,
                           featurize_pc, CSV_PATH, SCALER, H743_CSV, START, N)
from conv_sem_test import forward_c_sem, forward_torch

sc = joblib.load(SCALER)
center = np.asarray(sc.center_, dtype=np.float32)
scale = np.asarray(sc.scale_, dtype=np.float32)
mu18 = center[np.asarray(KEEP_18)]
sg18 = scale[np.asarray(KEEP_18)]
A = load_header_weights()

st_pc = St()
lru = {}
last_ts = None
first = True
win_pc = deque(maxlen=SW_WINDOW)
win_h7 = deque(maxlen=SW_WINDOW)
wins = []
with open(CSV_PATH, encoding="utf-8") as f:
    for i, row in enumerate(csv.DictReader(f), start=1):
        if i <= START:
            continue
        if i > START + N:
            break
        fidx = i - START
        raw_pc = featurize_pc(st_pc, row)
        win_pc.append((raw_pc - mu18) / sg18)
        dec = mcu_decode(pc_encode(row))
        ts = dec["time"]
        dt = 0.0 if first else float(np.float32(ts - last_ts))
        first = False
        last_ts = ts
        k = (dec["address"], dec["function"])
        tsl = 0.0 if k not in lru else float(np.float32(ts - lru[k]))
        lru[k] = ts
        raw_h7 = np.array([dec["address"], dec["function"], dec["length"],
                           dec["gain"], dec["reset"], dec["deadband"], dec["cycle"],
                           dec["rate"], dec["system_mode"], dec["control_scheme"],
                           dec["pump"], dec["solenoid"],
                           min(100.0, max(0.0, dec["pressure"])), dec["crc"], dt, tsl,
                           1.0 if dec["function"] in MCU_UNUSUAL else 0.0,
                           float(dec["is_response"])], dtype=np.float32)
        win_h7.append((raw_h7 - mu18) / sg18)
        if len(win_pc) < SW_WINDOW:
            continue
        if fidx % SW_WINDOW == 0:
            wins.append((fidx, build23(np.stack(list(win_pc), 0)),
                         build23(np.stack(list(win_h7), 0))))

n_win = len(wins)
frame_end = np.array([w[0] for w in wins])

# ground truth
lab = np.load("gui_labels.npz")
gl = dict(zip(lab["frames"].tolist(), lab["gui"].tolist()))
atk = [any(gl.get(f, 0) for f in range(w * 16 + 1, w * 16 + 17)) for w in range(n_win)]
n_atk = sum(atk)
n_norm = n_win - n_atk

c_prob = np.array([forward_c_sem(A, xh7) for (_, _, xh7) in wins])
t_prob = np.array([forward_torch(A, xpc) for (_, xpc, _) in wins])

np.savez_compressed("fix_impact_probs.npz", frames=frame_end,
                    c_sem=c_prob, trained=t_prob, atk=np.array(atk, dtype=np.int8))


def metrics(probs, thr, tag):
    tp = fp = fn = 0
    for w in range(n_win):
        det = probs[w] >= thr
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
    print("  %-34s thr=%.2f  TP=%5d FP=%5d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f Acc=%.4f"
          % (tag, thr, tp, fp, fn, P, R, F1, fp / max(n_norm, 1),
             (tp + (n_norm - fp)) / n_win))
    return tp, fp, fn


print("windows=%d  attack=%d  normal=%d" % (n_win, n_atk, n_norm))
print()
print("H743 实测 summary:  TP=1482 FP=58  FN=933  P=0.9623 R=0.6137 F1=0.7494 FPR=0.0258")
print()
print("--- (a) C 语义 (H743 当前 kernel, 因果对齐) ---")
for thr in (0.36, 0.5):
    metrics(c_prob, thr, "C 语义")
print()
print("--- (b) 训练语义 (修好 conv 对齐后) ---")
for thr in (0.36, 0.5, 0.6, 0.7):
    metrics(t_prob, thr, "训练语义")

# 与 H743 实测逐 window 复核
h743 = {}
with open(H743_CSV, encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        if r["event"] == "ALARM" and r["frame_idx"] and r["prob_x1000"]:
            h743[int(r["frame_idx"])] = int(r["prob_x1000"]) / 1000.0
fe2c = dict(zip(frame_end.tolist(), c_prob.tolist()))
fe2t = dict(zip(frame_end.tolist(), t_prob.tolist()))
hp, cp, tp_ = [], [], []
for fr_i, pv in h743.items():
    cand = [f for f in frame_end.tolist() if fr_i - 15 <= f <= fr_i]
    if not cand:
        continue
    f = cand[-1]
    hp.append(pv)
    cp.append(fe2c.get(f))
    tp_.append(fe2t.get(f))
hp = np.array(hp)
cp = np.array(cp)
tp_ = np.array(tp_)
m = ~np.isnan(cp)
print()
print("在 H743 报警的 %d 个 window 上:" % len(hp))
print("  C 语义复现 vs 实测 : MAE=%.5f  corr=%.5f" % (np.mean(np.abs(hp[m] - cp[m])),
                                                     np.corrcoef(hp[m], cp[m])[0, 1]))
print("  训练语义 vs 实测   : MAE=%.5f  corr=%.5f" % (np.mean(np.abs(hp[m] - tp_[m])),
                                                     np.corrcoef(hp[m], tp_[m])[0, 1]))
print("  训练语义 prob 均值 %.4f median %.4f   (实测 %.4f / %.4f)"
      % (np.nanmean(tp_), np.nanmedian(tp_), hp.mean(), np.median(hp)))
