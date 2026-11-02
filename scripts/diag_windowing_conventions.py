"""Compare two per-window counting conventions on the same CSV:
  A) per-alarm-event: each ALARM row in CSV = 1 detection (TP if binary=1, FP if binary=0)
  B) per-window (non-overlap W=16): one decision per window, alarmed if ANY ALARM in window range

User wants this to match PC sim. PC sim = 1 prob per (sliding? non-overlap?) window.
H743 CSV has 7077 ALARM events — each is one window's inference result.
"""
import csv
from collections import defaultdict

CSV = "C:/work/Claude/Issue/logs/test_20261018-231733_auto-40ms.csv"
W = 16

rows = []
with open(CSV, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

binary = {}
for r in rows:
    if r["event"] == "SEND":
        binary[int(r["frame_idx"])] = int(r["binary"])

alarms = [r for r in rows if r["event"] == "ALARM"]
print(f"Total alarms: {len(alarms)}")
print(f"Total frames: {len(binary)}")
print(f"Attack frames: {sum(1 for v in binary.values() if v == 1)}")
print(f"Normal frames: {sum(1 for v in binary.values() if v == 0)}")

# === Convention A: per-alarm-event (each ALARM = 1 detection) ===
print(f"\n=== Convention A: per-ALARM-event ===")
tp_a, fp_a = 0, 0
for r in alarms:
    fidx = int(r["frame_idx"])
    if binary.get(fidx, 0) == 1:
        tp_a += 1
    else:
        fp_a += 1
print(f"  TP={tp_a} FP={fp_a}  (FP rate over all alarms = {fp_a/len(alarms)*100:.2f}%)")

# Per-window aggregate (non-overlap W=16) — frame label = window is "attack" if any frame in window is attack
def agg_nonoverlap(W_):
    n_total = max(binary.keys()) + 1
    n_win = n_total // W_
    win_label = {}
    for w in range(n_win):
        start = w * W_
        end = start + W_
        is_atk = any(binary.get(f, 0) == 1 for f in range(start, end))
        win_label[w] = is_atk
    return n_win, win_label

# === Convention B: per non-overlap W=16 window (any alarm in window's frame range → alarmed) ===
print(f"\n=== Convention B: per non-overlap W=16 window ===")
n_win, win_label = agg_nonoverlap(W)
n_atk_win = sum(1 for v in win_label.values() if v)
n_norm_win = n_win - n_atk_win

window_alarm = defaultdict(list)
for r in alarms:
    fidx = int(r["frame_idx"])
    prob = int(r["prob_x1000"])
    win = fidx // W
    window_alarm[win].append(prob)

tp_b, fp_b, fn_b, tn_b = 0, 0, 0, 0
for w in range(n_win):
    is_atk = win_label[w]
    has_alarm = len(window_alarm.get(w, [])) > 0
    if has_alarm:
        if is_atk: tp_b += 1
        else: fp_b += 1
    else:
        if is_atk: fn_b += 1
        else: tn_b += 1

print(f"  Total windows: {n_win}  attack: {n_atk_win}  normal: {n_norm_win}")
print(f"  TP={tp_b} FP={fp_b} FN={fn_b} TN={tn_b}")
prec = tp_b/max(tp_b+fp_b,1); fpr = fp_b/max(n_norm_win,1); rec = tp_b/max(n_atk_win,1)
f1 = 2*prec*rec/max(prec+rec,1e-9)
print(f"  P={prec*100:.2f}% R={rec*100:.2f}% FPR={fpr*100:.2f}% F1={f1*100:.2f}%")

# === Convention C: same as B but attack if MAJORITY of frames in window are attack ===
print(f"\n=== Convention C: per non-overlap W=16 window, attack if >=50% frames in window are attack ===")
n_win2 = (max(binary.keys()) + 1) // W
win_label_c = {}
for w in range(n_win2):
    start = w * W
    end = start + W
    n_atk = sum(1 for f in range(start, end) if binary.get(f, 0) == 1)
    win_label_c[w] = (n_atk >= 8)  # majority

n_atk_c = sum(1 for v in win_label_c.values() if v)
tp_c, fp_c, fn_c, tn_c = 0, 0, 0, 0
for w in range(n_win2):
    is_atk = win_label_c[w]
    has_alarm = len(window_alarm.get(w, [])) > 0
    if has_alarm:
        if is_atk: tp_c += 1
        else: fp_c += 1
    else:
        if is_atk: fn_c += 1
        else: tn_c += 1

print(f"  Total windows: {n_win2}  attack: {n_atk_c}  normal: {n_win2 - n_atk_c}")
print(f"  TP={tp_c} FP={fp_c} FN={fn_c} TN={tn_c}")
prec = tp_c/max(tp_c+fp_c,1); fpr = fp_c/max(n_win2-n_atk_c,1); rec = tp_c/max(n_atk_c,1)
f1 = 2*prec*rec/max(prec+rec,1e-9)
print(f"  P={prec*100:.2f}% R={rec*100:.2f}% FPR={fpr*100:.2f}% F1={f1*100:.2f}%")

# === Convention D: per-alarm-event, attacked window if alarm's frame is attack ===
# Memory: Attacked=8821, TP=6250, FP=757, FN=2571
# Try: TP+FP = 7007, sum of all alarms. Memory shows 7077 alarms but 7007 detection events.
# So maybe Convention D counts each alarm as a detection, and the 70 alarms missing are "dedup"
print(f"\n=== Memory says: Attacked=8821 TP=6250 FP=757 FN=2571 ===")
print(f"  8821 = TP+FN = 6250+2571")
print(f"  Normal = n_uniq - 8821 = ???")
# 17164 - 8821 = 8343 normal windows (per memory)
# FPR = 757 / 8343 = 9.07%
# Recall = 6250 / 8821 = 70.85%
# F1 = 2*6250 / (2*6250+757+2571) = 12500 / 15828 = 78.97%

# What gives attacked windows = 8821?
# If we count windows where any frame is attack, sliding-style: many more than 8821.
# If we count windows where ANY frame in window [i, i+16) is attack, with stride=16 non-overlap: 8806.
# Memory's 8821 is close to 8806 but off by 15. Could be a 15-frame offset?

# Try with 15-frame stride offset (window starts at frame 15 instead of 0):
print(f"\n=== Convention E: window starts at frame 15 (offset variant) ===")
n_atk_e = 0
for w in range(n_win):
    start = w * W + 15  # offset by 15
    end = start + W
    is_atk = any(binary.get(f, 0) == 1 for f in range(start, min(end, max(binary.keys())+1)))
    if is_atk: n_atk_e += 1
print(f"  Attacked windows (offset=15): {n_atk_e}")

# Try yet another: stride=16 windows, but counting ALL frames in dataset (including last partial)
n_total_frame = max(binary.keys()) + 1
n_atk_strict = 0
for w in range(n_total_frame // W):
    start = w * W
    end = start + W
    is_atk = any(binary.get(f, 0) == 1 for f in range(start, end))
    if is_atk: n_atk_strict += 1
print(f"\n=== Strict non-overlap with 17164 windows: attacked = {n_atk_strict} ===")