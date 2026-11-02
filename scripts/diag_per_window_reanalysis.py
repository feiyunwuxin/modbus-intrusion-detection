"""Re-analyze auto-40ms CSV at PER-WINDOW level (match PC sim).

H743 reality:
  - 1 inference per window (16 frames)
  - 1 prob per window → sigmoid output → compare with threshold → 1 alarm decision
  - Each ALARM event in CSV = 1 WINDOW detected (not 1 frame)

Per-window metrics must match PC sim conventions:
  - window_size = 16
  - attack window: any frame in window has binary=1
  - normal window: all 16 frames have binary=0
  - alarmed window: there's an ALARM event with frame_idx in window's range

PC sim test set: 17164 windows total, 8821 attacked windows.
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

# Step 1: Build per-frame label map (frame_idx → binary)
binary = {}
for r in rows:
    if r["event"] == "SEND":
        binary[int(r["frame_idx"])] = int(r["binary"])
print(f"Total frames sent: {len(binary)}")
n_atk_frame = sum(1 for v in binary.values() if v == 1)
n_norm_frame = sum(1 for v in binary.values() if v == 0)
print(f"  attack frames: {n_atk_frame}  normal frames: {n_norm_frame}")

# Step 2: Aggregate to per-window (non-overlapping W=16)
#   Window i covers frames [i*16, (i+1)*16)
n_total_frame = max(binary.keys()) + 1
n_window = (n_total_frame + W - 1) // W
print(f"\nTotal windows (non-overlap W={W}): {n_window}")

window_label = {}  # 1 if any attack in window, else 0
window_max_prob = {}  # max prob in window (for re-evaluating at different thresholds)
for w in range(n_window):
    start = w * W
    end = min(start + W, n_total_frame)
    is_atk = 0
    max_prob = None
    for f in range(start, end):
        if binary.get(f, 0) == 1:
            is_atk = 1
        if f in binary:
            pass  # prob only set by ALARM, not SEND
    window_label[w] = is_atk

# Step 3: Aggregate ALARM events per window
#   each ALARM has frame_idx → which window does it belong to?
#   H743 typically fires once per completed window (sliding window) → only the last frame of window triggers alarm
window_alarm = defaultdict(list)  # win → list of (frame_idx, prob_x1000)
for r in rows:
    if r["event"] == "ALARM":
        fidx = int(r["frame_idx"])
        prob = int(r["prob_x1000"])
        win = fidx // W  # window that contains this frame
        window_alarm[win].append((fidx, prob))

# Some windows may have multiple alarms (carry-over), some have 0
print(f"\nWindow alarm distribution:")
n_alarm = sum(1 for v in window_alarm.values() if v)
multi_alarm = sum(1 for v in window_alarm.values() if len(v) > 1)
print(f"  windows with at least 1 alarm: {n_alarm}")
print(f"  windows with multiple alarms: {multi_alarm}")

# Step 4: Compute TP/FP/FN at per-window level using current threshold (0.36)
thr_x1000 = 360
tp_win = 0
fp_win = 0
fn_win = 0
tn_win = 0
tp_max_prob = []
fp_max_prob = []
for w in range(n_window):
    is_atk = window_label[w]
    alarms = window_alarm.get(w, [])
    has_alarm = any(p >= thr_x1000 for f, p in alarms)
    if has_alarm:
        max_p = max((p for _, p in alarms), default=0)
        if is_atk:
            tp_win += 1
            tp_max_prob.append(max_p)
        else:
            fp_win += 1
            fp_max_prob.append(max_p)
    else:
        if is_atk:
            fn_win += 1
        else:
            tn_win += 1

n_atk_win = sum(1 for v in window_label.values() if v == 1)
n_norm_win = sum(1 for v in window_label.values() if v == 0)

print(f"\n=== Per-window metrics (thr=0.36) ===")
print(f"Total windows:    {n_window}")
print(f"Attack windows:   {n_atk_win}")
print(f"Normal windows:   {n_norm_win}")
print(f"TP: {tp_win}, FP: {fp_win}, FN: {fn_win}, TN: {tn_win}")

prec = tp_win / max(tp_win + fp_win, 1)
rec = tp_win / max(n_atk_win, 1)
fpr = fp_win / max(n_norm_win, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(f"\nPrecision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"FPR:       {fpr*100:.2f}%")
print(f"F1:        {f1*100:.2f}%")

# Compare with memory
print(f"\n=== vs memory (project-2026-10-19-h743-fp32-full274k.md) ===")
print(f"  memory:    F1=78.97% Recall=70.85% FPR=9.07% Attacked=8821 TP=6250 FP=757 FN=2571")
print(f"  this CSV:  F1={f1*100:.2f}% Recall={rec*100:.2f}% FPR={fpr*100:.2f}% "
      f"Attacked={n_atk_win} TP={tp_win} FP={fp_win} FN={fn_win}")

# Step 5: Threshold sweep (per-window)
print(f"\n=== Threshold sweep (per-window) ===")
print(f"{'thr':>6} {'TP':>5} {'FP':>5} {'Prec':>7} {'Recall':>7} {'FPR':>7} {'F1':>7}")
for thr in [200, 250, 300, 350, 360, 380, 400, 420, 450, 500, 550, 600, 700, 800, 900]:
    thr_ = thr
    tp_, fp_, fn_ = 0, 0, 0
    for w in range(n_window):
        is_atk = window_label[w]
        alarms = window_alarm.get(w, [])
        has_alarm = any(p >= thr_ for f, p in alarms)
        if has_alarm:
            if is_atk: tp_ += 1
            else: fp_ += 1
        else:
            if is_atk: fn_ += 1
    P = tp_ / max(tp_+fp_, 1)
    R = tp_ / max(n_atk_win, 1)
    FPR = fp_ / max(n_norm_win, 1)
    F1 = 2*P*R / max(P+R, 1e-9)
    print(f"{thr/1000:>6.3f} {tp_:>5} {fp_:>5} {P*100:>6.2f}% {R*100:>6.2f}% {FPR*100:>6.2f}% {F1*100:>6.2f}%")

# Step 6: FP prob distribution (per-window)
print(f"\n=== FP prob distribution (per-window) ===")
import collections
fp_hist = collections.Counter()
for p in fp_max_prob:
    fp_hist[p // 50 * 50] += 1
for b in sorted(fp_hist.keys()):
    bar = "#" * (fp_hist[b] // 5)
    print(f"  [{b:4}-{b+50}): {fp_hist[b]:4} {bar}")

print(f"\n=== TP prob distribution (per-window) ===")
tp_hist = collections.Counter()
for p in tp_max_prob:
    tp_hist[p // 50 * 50] += 1
for b in sorted(tp_hist.keys()):
    bar = "#" * (tp_hist[b] // 5)
    print(f"  [{b:4}-{b+50}): {tp_hist[b]:4} {bar}")

# Step 7: FP concentrated where? Per-bin (50 windows)
print(f"\n=== FP concentration by window bin (1000 win bins) ===")
fp_bins = collections.Counter()
for w in range(n_window):
    is_atk = window_label[w]
    alarms = window_alarm.get(w, [])
    has_alarm = any(p >= thr_x1000 for f, p in alarms)
    if has_alarm and not is_atk:
        bin_ = w // 1000
        fp_bins[bin_] += 1
for b in sorted(fp_bins.keys()):
    print(f"  win[{b*1000:>5}:{(b+1)*1000:>5}): {fp_bins[b]} FPs")
print(f"  Total FP windows: {sum(fp_bins.values())}")