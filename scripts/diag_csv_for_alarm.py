"""Analyze test_20261018-231733_auto-40ms.csv for FP/TP pattern.

Frame-level stats from auto-40ms test:
  frames=274628  attack=60048  normal=214580
  TP=6250  FP=757

But 7077 ALARM events total, all marked is_fp="TP" in column 8.
running_tp=6250, running_fp=757. So 7077-7007=70 alarms "unaccounted".

Goal: figure out the gap and characterize the F1 gap.
"""
import csv

CSV = "C:/work/Claude/Issue/logs/test_20261018-231733_auto-40ms.csv"
rows = []
with open(CSV, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

print(f"Total rows: {len(rows)}")
events = {}
for r in rows:
    events[r["event"]] = events.get(r["event"], 0) + 1
print(f"Event breakdown: {events}")

# Frame-level label distribution
binary_per_frame = {}
for r in rows:
    if r["event"] == "SEND":
        idx = int(r["frame_idx"])
        binary_per_frame[idx] = r["binary"]
print(f"\nFrame-level: {len(binary_per_frame)} unique frames sent")
n_atk = sum(1 for v in binary_per_frame.values() if v == "1")
n_norm = sum(1 for v in binary_per_frame.values() if v == "0")
print(f"  binary=1 (attack): {n_atk}")
print(f"  binary=0 (normal): {n_norm}")

# Alarms analysis
alarms = [r for r in rows if r["event"] == "ALARM"]
print(f"\nAlarms total: {len(alarms)}")
# Each alarm has prob_x1000 + frame_idx
# running_tp and running_fp are in row

prev_tp = 0
prev_fp = 0
tp_alarms = []
fp_alarms = []
for r in alarms:
    cur_tp = int(r["running_tp"])
    cur_fp = int(r["running_fp"])
    if cur_tp > prev_tp:
        tp_alarms.append(r)
    if cur_fp > prev_fp:
        fp_alarms.append(r)
    prev_tp = cur_tp
    prev_fp = cur_fp

print(f"  TP alarms (running_tp incremented): {len(tp_alarms)}")
print(f"  FP alarms (running_fp incremented): {len(fp_alarms)}")
print(f"  Other alarms (running didn't increment): {len(alarms) - len(tp_alarms) - len(fp_alarms)}")

# Check if FP alarms align with binary=0 frames
fp_on_attack = sum(1 for r in fp_alarms if binary_per_frame.get(r["frame_idx"]) == "1")
fp_on_normal = sum(1 for r in fp_alarms if binary_per_frame.get(r["frame_idx"]) == "0")
print(f"\nFP alarms: {fp_on_attack} on attack frames, {fp_on_normal} on normal frames")

tp_on_attack = sum(1 for r in tp_alarms if binary_per_frame.get(r["frame_idx"]) == "1")
tp_on_normal = sum(1 for r in tp_alarms if binary_per_frame.get(r["frame_idx"]) == "0")
print(f"TP alarms: {tp_on_attack} on attack frames, {tp_on_normal} on normal frames")

# is_fp column distribution
is_fp_vals = {}
for r in alarms:
    v = r["is_fp"]
    is_fp_vals[v] = is_fp_vals.get(v, 0) + 1
print(f"\nis_fp column distribution: {is_fp_vals}")

# Other alarms: which frame?
other = []
prev_tp = 0
prev_fp = 0
for r in alarms:
    cur_tp = int(r["running_tp"])
    cur_fp = int(r["running_fp"])
    if cur_tp == prev_tp and cur_fp == prev_fp:
        other.append(r)
    prev_tp = cur_tp
    prev_fp = cur_fp

if other:
    print(f"\nFirst 5 'other' alarms (no TP/FP counter increment):")
    for r in other[:5]:
        print(f"  frame={r['frame_idx']} prob={r['prob_x1000']} is_fp={r['is_fp']} binary={binary_per_frame.get(r['frame_idx'])} running_tp={r['running_tp']} running_fp={r['running_fp']}")

# Prob distribution of FP alarms
fp_probs = [int(r["prob_x1000"]) for r in fp_alarms]
tp_probs = [int(r["prob_x1000"]) for r in tp_alarms]
print(f"\nFP prob stats: min={min(fp_probs)} max={max(fp_probs)} median={sorted(fp_probs)[len(fp_probs)//2]}")
print(f"TP prob stats: min={min(tp_probs)} max={max(tp_probs)} median={sorted(tp_probs)[len(tp_probs)//2]}")

# where do FPs concentrate? by frame_idx
fp_bins = {}
for r in fp_alarms:
    fidx = int(r["frame_idx"])
    bin_50k = fidx // 50000
    fp_bins[bin_50k] = fp_bins.get(bin_50k, 0) + 1
print(f"\nFP distribution by 50k-frame bins:")
for b in sorted(fp_bins.keys()):
    print(f"  [{b*50}k:{(b+1)*50}k) = {fp_bins[b]} FPs")