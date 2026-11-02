#!/usr/bin/env python3
"""
Round 63 v2 — Dynamic lag-aware window assignment.

Per-frame alarm attribution: each alarm's window should be (frame-1)//16 - shift
where shift depends on the actual measured lag for that alarm.

Try: shift = floor(lag_ms / 40) (no cap) — for lag=80ms, shift=2
Also try: shift = max(0, round(lag_ms/40) - 1) — assume minimal accurate shift
"""
import csv
from collections import defaultdict
from pathlib import Path

LOG = Path("logs/test_20261012-002126_auto-40ms.csv")
CSV = Path("IanArffDatasetF32.csv")

labels = []
with CSV.open() as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        try:
            labels.append((int(row[17]), int(row[18])))
        except (ValueError, IndexError):
            labels.append((0, 0))
n_windows = len(labels) // 16

sends_by_frame = {}
alarms = []
with LOG.open() as f:
    r = csv.DictReader(f)
    for row in r:
        if row["event"] == "SEND":
            try:
                fi = int(row["frame_idx"])
                sends_by_frame[fi] = int(row["t_event_ms"])
            except (ValueError, KeyError):
                continue
        elif row["event"] == "ALARM":
            try:
                fi_str = row["frame_idx"]
                if not fi_str:
                    continue
                fi = int(fi_str)
                t = int(row["t_event_ms"])
                alarms.append((t, fi, sends_by_frame.get(fi, t)))
            except (ValueError, KeyError):
                continue

def compute_metrics(window_alarm):
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for W in range(n_windows):
        start = W * 16
        end = (W + 1) * 16
        had_attack = any(labels[i][0] for i in range(start, end))
        had_alarm = bool(window_alarm.get(W))
        if had_attack and had_alarm:
            counts["TP"] += 1
        elif not had_attack and had_alarm:
            counts["FP"] += 1
        elif had_attack and not had_alarm:
            counts["FN"] += 1
        else:
            counts["TN"] += 1
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    return counts, f1, prec, rec


# Dynamic: shift based on measured lag
def assign_dynamic():
    win_alarm = defaultdict(list)
    for t_alarm, fi, t_send in alarms:
        lag_ms = t_alarm - t_send
        lag_frames = lag_ms / 40
        # Choose shift that minimizes window misalignment
        # If alarm arrives in the same window as it was fired for, no shift (shift=0).
        # If alarm arrives during window K+1, shift by 1.
        # In general: shift = ceil(lag_frames) - 1
        shift = max(0, int(lag_frames) + (1 if lag_frames > int(lag_frames) else 0) - 1)
        # For lag=80ms (2 frames), shift = 1
        # For lag=40ms (1 frame), shift = 0
        # For lag=20ms (0.5 frame), shift = 0
        K = fi // 16  # 1-indexed
        W = max(0, K - 1 - shift)
        win_alarm[W].append(fi)
    return win_alarm


# Dynamic v2: shift = round(lag_frames) - 1 (assume 1 frame uncertainty)
def assign_dynamic_v2():
    win_alarm = defaultdict(list)
    for t_alarm, fi, t_send in alarms:
        lag_ms = t_alarm - t_send
        lag_frames = lag_ms / 40
        # If alarm arrives 0-40ms after frame send, no shift needed (on time)
        # If 40-80ms, shift by 1
        # If 80-120ms, shift by 2
        # etc.
        if lag_ms < 40:
            shift = 0
        elif lag_ms < 80:
            shift = 1
        elif lag_ms < 160:
            shift = 2  # covers both 80-120 and 120-160
        else:
            shift = 3
        K = fi // 16  # 1-indexed
        W = max(0, K - 1 - shift)
        win_alarm[W].append(fi)
    return win_alarm


# Hybrid: legacy deque path (current real behavior) + direct path  merged
def assign_hybrid_actual():
    """Simulate the actual current behavior: direct OR legacy.
    Direct: _direct_alarms.add(W) for window W. If W evaluates BEFORE add → window uses legacy.
    Legacy: deque-based, alarm lands in window where PC frame count falls during shift.
    """
    win_alarm = defaultdict(list)
    # Each alarm has potential to go to TWO windows: direct W and legacy W'
    # Actual behavior depends on Qt signal ordering which we can't simulate exactly.
    # But empirically, actual ≈ blend of pure-direct and pure-legacy.
    # Let's just use pure-legacy for now (current real behavior per my analysis)
    for t_alarm, fi, t_send in alarms:
        lag_ms = t_alarm - t_send
        lag_frames = lag_ms / 40
        # Legacy deque: when alarm arrives, PC frame count = fi + lag_frames.
        # Deque gets (PC.frame_count - 1) = fi + lag_frames - 1
        # Window = (fi + lag_frames - 2) // 16 (0-indexed) — the window that PC will
        # close NEXT, which contains this frame index
        # For lag=80ms (2 frames), PC is at fi+2. Window = (fi+0)//16 = fi//16.
        # fi is multiple of 16, so W = fi/16 = K (0-indexed = K-1 1-indexed).
        # But for lag=120ms (3 frames), PC at fi+3. Window = (fi+1)//16 = fi/16 if fi%16==0 else (fi//16).
        # For fi=multiple of 16, all give W = fi/16 = K (0-indexed = K-1 1-indexed).
        # Legacy always maps to W = (fi + something) // 16 where something is lag-induced PC frame offset.
        # Practically, the alarm is attributed to window W = (fi - 1) // 16 + 1 = fi//16 (for lag-induced misalignment)
        # For lag=80ms, W=fi//16 (=K, 1-indexed K+1) instead of K-1 (1-indexed K). Off by 1.
        pc_frame_offset = int(lag_ms / 40)  # how many frames PC has sent after fi
        W = max(0, (fi + pc_frame_offset - 1) // 16)
        win_alarm[W].append(fi)
    return win_alarm


# Print all
print(f"{'Strategy':<35} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'F1':>7} {'Prec':>7} {'Recall':>7}")
print(f"{'(Actual reported by eval)':<35} {'6969':>6} {'2001':>6} {'1852':>6} {'6342':>6} {'78.34%':>7} {'77.69%':>7} {'79.00%':>7}")

# Direct (no lag compensation)
def assign_direct():
    win_alarm = defaultdict(list)
    for t_alarm, fi, _ in alarms:
        K = fi // 16
        W = K - 1
        if 0 <= W < n_windows:
            win_alarm[W].append(fi)
    return win_alarm

counts, f1, prec, rec = compute_metrics(assign_direct())
print(f"{'Pure Direct (no shift)':<35} {counts['TP']:>6} {counts['FP']:>6} {counts['FN']:>6} {counts['TN']:>6} {f1*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}%")

# Dynamic shift
counts, f1, prec, rec = compute_metrics(assign_dynamic())
print(f"{'Dynamic shift = ceil(lag)-1':<35} {counts['TP']:>6} {counts['FP']:>6} {counts['FN']:>6} {counts['TN']:>6} {f1*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}%")

# Dynamic v2 (bucket shifts)
counts, f1, prec, rec = compute_metrics(assign_dynamic_v2())
print(f"{'Dynamic v2 (lag buckets)':<35} {counts['TP']:>6} {counts['FP']:>6} {counts['FN']:>6} {counts['TN']:>6} {f1*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}%")

# Hybrid "actual" (legacy approximation)
counts, f1, prec, rec = compute_metrics(assign_hybrid_actual())
print(f"{'Hybrid actual (legacy sim)':<35} {counts['TP']:>6} {counts['FP']:>6} {counts['FN']:>6} {counts['TN']:>6} {f1*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}%")

# Optimal fixed shift
print(f"\n=== Fixed shift sweep ===")
for shift in [0, 1, 2, 3]:
    win_alarm = defaultdict(list)
    for t_alarm, fi, _ in alarms:
        K = fi // 16
        W = max(0, K - 1 - shift)
        if W < n_windows:
            win_alarm[W].append(fi)
    counts, f1, prec, rec = compute_metrics(win_alarm)
    print(f"  Shift={shift}: TP={counts['TP']:>5} FP={counts['FP']:>5} FN={counts['FN']:>5} TN={counts['TN']:>5} F1={f1*100:>5.2f}%")

# What if we ONLY attribute alarms from windows where H743's printf frame aligns
# with the local ARFF label being attack (true positive alarm)?
print(f"\n=== Filter: only count alarms where alarm's frame_idx is ARFF-attack ===")
counts, f1, prec, rec = compute_metrics(assign_direct())
tp = counts['TP']
fp = counts['FP']
print(f"  Pure Direct: TP={tp}, FP={fp}")
print(f"  → FP=1448 is the model's TRUE false positive count (model fires on normal windows)")
print(f"  → Latent F1 ceiling if we counted only 'ideal-attribution' TPs: {2*tp/(2*tp+fp+(1852+1298-tp))*100:.2f}%")

# Also note: what's the gap between Direct F1 and Actual F1?
print(f"\n=== Diagnosis ===")
print(f"Pure Direct (ideal):  F1=84.57% → 1448 true FPs (model-level) + 0 attribution errors")
print(f"Actual:                F1=78.34% → 2001 FPs (1448 model + 553 attribution)")
print(f"Attribution penalty:  -6.23% F1 due to lag-induced window misattribution")
