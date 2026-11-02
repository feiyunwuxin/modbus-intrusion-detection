"""Investigate the 120 'other' alarms and threshold-tuning opportunities."""
import csv

CSV = "C:/work/Claude/Issue/logs/test_20261018-231733_auto-40ms.csv"
rows = []
with open(CSV, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

# Frame-level label distribution
binary_per_frame = {}
for r in rows:
    if r["event"] == "SEND":
        idx = int(r["frame_idx"])
        binary_per_frame[idx] = r["binary"]

# Re-run the alarm analysis
alarms = [r for r in rows if r["event"] == "ALARM"]
prev_tp = 0
prev_fp = 0
tp_alarms = []
fp_alarms = []
other = []
for r in alarms:
    cur_tp = int(r["running_tp"])
    cur_fp = int(r["running_fp"])
    if cur_tp > prev_tp:
        tp_alarms.append(r)
    elif cur_fp > prev_fp:
        fp_alarms.append(r)
    else:
        other.append(r)
    prev_tp = cur_tp
    prev_fp = cur_fp

# Other analysis: see how often they're consecutive (same frame_idx)
print(f"=== 120 'other' alarms (running_tp/fp didn't change) ===")
print(f"  unique frame_idxs: {len(set(r['frame_idx'] for r in other))}")
print(f"  with binary in binary_per_frame: {sum(1 for r in other if binary_per_frame.get(r['frame_idx']) is not None)}")

# Most are follow-ups after TP (frame=None means binary wasn't found — meaning the alarm was raised
# but the running_tp was already > 0 indicating TP already counted)
# Actually binary=None could mean that frame_idx wasn't sent yet at the time of alarm. Hmm.
# Let me check: are these "other" alarms always at high frame_idx where many TPs preceded?

# How many are 'none' binary?
none_binary = sum(1 for r in other if binary_per_frame.get(r['frame_idx']) is None)
print(f"  frames not in SEND map: {none_binary}")

# See prob distribution for these
other_probs = [int(r["prob_x1000"]) for r in other]
print(f"  prob stats: min={min(other_probs)} max={max(other_probs)} median={sorted(other_probs)[len(other_probs)//2]}")

# Threshold sweep
print(f"\n=== Threshold sweep ===")
print(f"At each thr, count how many alarms would be raised (FP reduction).")
# For each potential threshold, compute # alarms above threshold for FPs vs TPs

tp_probs = sorted([int(r["prob_x1000"]) for r in tp_alarms])
fp_probs = sorted([int(r["prob_x1000"]) for r in fp_alarms])

# Combined alarm probs: each alarm has prob. Track whether the alarm registered as TP or FP.
# Threshold sweep: count kept alarms.
print(f"FP prob histogram (max=999%):")
import collections
fp_hist = collections.Counter()
for p in fp_probs:
    fp_hist[p // 50 * 50] += 1  # bucket by 50
for b in sorted(fp_hist.keys()):
    bar = "#" * (fp_hist[b] // 5)
    print(f"  [{b:4}-{b+50}): {fp_hist[b]:4} {bar}")

print(f"\nTP prob histogram (max=999%):")
tp_hist = collections.Counter()
for p in tp_probs:
    tp_hist[p // 50 * 50] += 1
for b in sorted(tp_hist.keys()):
    bar = "#" * (tp_hist[b] // 5)
    print(f"  [{b:4}-{b+50}): {tp_hist[b]:4} {bar}")

# Compute F1 at each threshold (frame-level)
# At thr T: keep alarms with prob >= T
# TP at thr T = alarms with prob>=T on attack frame (running_tp incremented)
# FP at thr T = alarms with prob>=T on normal frame (running_fp incremented)
# Frame-level binary = 60048 attacks, 214580 normal
# Each alarm is one "detection" (1 window?)
# Actually each alarm fires once. Total alarms that count = those above threshold.
# For F1: TP/FP from alarm perspective.

# F1 frame-level: precision = TP/(TP+FP), recall = TP/total_attack
# Note: this is FRAME-level. The per-window calculation is different.

print(f"\n=== F1 at different thresholds (using tp_alarms/fp_alarms) ===")
total_attack = 60048
total_normal = 214580
print(f"baseline (thr=0.36): TP={len(tp_alarms)} FP={len(fp_alarms)}")
print(f"  Precision = {len(tp_alarms)/(len(tp_alarms)+len(fp_alarms))*100:.2f}%")
print(f"  Recall = {len(tp_alarms)/total_attack*100:.2f}%")

# At thr=0.4: drop alarms with prob<400
def f1_at_thr(thr_x1000):
    tp_kept = sum(1 for r in tp_alarms if int(r["prob_x1000"]) >= thr_x1000)
    fp_kept = sum(1 for r in fp_alarms if int(r["prob_x1000"]) >= thr_x1000)
    P = tp_kept / max(tp_kept+fp_kept, 1)
    R = tp_kept / total_attack
    F1 = 2*P*R / max(P+R, 1e-9)
    return P, R, F1, tp_kept, fp_kept

print(f"\nThreshold sweep (frame-level approximation):")
print(f"{'thr':>6} {'TP':>5} {'FP':>5} {'Prec':>6} {'Recall':>7} {'F1':>6}")
for thr in [200, 250, 300, 350, 360, 380, 400, 420, 450, 500, 550, 600, 700, 800, 900]:
    P, R, F1, tp, fp = f1_at_thr(thr)
    print(f"{thr/1000:>6.3f} {tp:>5} {fp:>5} {P*100:>5.2f}% {R*100:>6.2f}% {F1*100:>5.2f}%")