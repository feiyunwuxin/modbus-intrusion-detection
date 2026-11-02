#!/usr/bin/env python3
"""
Round 63 — Reproduce per-window eval WITH proper signal ordering to find the REAL bug.

The Round 62 padded3 fix didn't change F1. Per-alarm analysis shows 0 FP at frame level
but 2001 FP at window level. The window attribution mechanism is suspect.

Approach: simulate the SAME timing race with multiple plausible scenarios:
  Scenario A: alarm always arrives BEFORE _evaluate_window (impossible but ideal)
  Scenario B: alarm arrives AFTER _evaluate_window (lag ≥ 1 frame)
  Scenario C: race — sometimes alarm before, sometimes after

Also check: do the 2001 FP windows all share common characteristics?
"""
import csv
from collections import defaultdict
from pathlib import Path

LOG = Path("logs/test_20261012-002126_auto-40ms.csv")
CSV = Path("IanArffDatasetF32.csv")

CAT_NAMES = {0: "Normal", 1: "DoS", 2: "Recon", 3: "MPCI", 4: "MSCI", 5: "NMRI", 6: "CMRI", 7: "MFCI"}

# Load ARFF labels
labels = []
with CSV.open() as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        try:
            labels.append((int(row[17]), int(row[18])))
        except (ValueError, IndexError):
            labels.append((0, 0))
print(f"[load] {len(labels):,} ARFF rows")

# Alarms with frame and lag
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
                alarms.append((t, fi))
            except (ValueError, KeyError):
                continue
print(f"[load] {len(alarms):,} alarms")

# Per-window analysis: For each window K-1 (0-indexed), check:
#  - Was any of frames 16(K-1)+1 .. 16K attack in ARFF? (ground truth attack)
#  - Was any alarm from H743 associated with this window? (alarm)
#
# H743 associates alarm with window K-1 if H743's frames_total=16K (last frame of window K)
# PC associates alarm with window K-1 if PC receives alarm with frame=16K BEFORE
#    PC's _evaluate_window for W=K-1 fires (which happens at PC sends frame 16K).

# Simulate the lag effect:
# Window K-1 (1-indexed K) is FP if:
#   - any PC frame 16(K-1)+1..16K was attack  → window has attack in ground truth
#   - alarm from H743 lands in W=K-1 OR in W≠K-1 (FP means alarm in wrong window)
# Actually for window to be TP/FP at PC side:
#   - PC's _evaluate_window for W=K-1 fires with attack=True/False (computed from PC's frames)
#   - PC's _evaluate_window for W=K-1 fires with alarm=True iff _direct_alarms has K-1
#     OR _recent_alarm_frames is non-empty OR _current_window_alarm was set

# For each alarm, predict which window PC ATTRIBUTES it to.
# Window index in _direct_alarms: (h743_frame-1)//16
# This window must be the SAME window that PC's _evaluate_window has just closed
# OR has yet to close (depending on whether alarm arrives before/after eval).

# Hypothesis: with lag=80ms (2 frames), alarm arrives 2 frames AFTER PC sent the
# triggering frame. By then PC has moved to frame K+2. So _direct_alarms.add(K-1)
# fires 2 frames late. When PC's next _evaluate_window fires (at frame K+16),
# W_now=K (not K-1). The alarm for K-1 is stale.

# Counter-hypothesis (this is what produces TPs):
# Some alarms arrive BEFORE PC's eval (lag < frame interval ~40ms).
# For those, on_alarm_with_frame fires before on_frame → _direct_alarms has K-1
#   before _evaluate_window closes window K-1 → alarm=True → TP.

# So the 6969 TPs come from alarms with lag < 40ms (or races).
# The 2001 FPs come from alarms where the alarm landed in a WRONG window:
#   - _direct_alarms has K-1 (stale, window K-1 already evaluated)
#   - Next alarm K+1 sends K (new window index). For delta=1 → only one K+1 in set.
#     If this is added BEFORE _evaluate_window for K fires, K is hit.
#     If after, K-1 (already evaluated) and K (still evaluated as TN/FN).
#   - Effectively, each "inter-window" gap creates misalignment.

# But wait — _direct_alarms.discard(W_now) is called in _evaluate_window AFTER checking
# membership. So once W_now is checked, W_now is removed.
#
# The order of operations in _evaluate_window:
#   1. Check W_now in _direct_alarms
#   2. Apply TP/FP/TN/FN
#   3. _n_windows++ (becomes K+1)
#   4. _direct_alarms.discard(W_now=K)
#
# So _direct_alarms is read BEFORE discard. If between step 1 and step 4 we add
# a NEW alarm, it could affect the NEXT window's eval.

# Simulation approach:
# - Each alarm with frame=16K, if it arrives BEFORE PC sends frame 16K, contributes to W=K-1 eval
# - If it arrives AFTER PC sends frame 16K, contributes to W=K-1 set (stale, never read)

# Lag distribution from earlier analysis:
#   0-20ms: 387 (4.3%)  — alarm arrives BEFORE next frame (40ms interval)
#   20-40ms: 33 (0.4%)  — at the boundary
#   40-160ms: 8508 (94.9%) — alarm arrives AFTER next frame, definitely late
#   160-500ms: 43 (0.5%)
#
# For 4.3% + 0.4% = 4.7% with lag < 40ms → potentially TP through direct path
# But we see 6969 TPs = 77.6% of 8971. Way higher than 4.7%.

# Something else is happening. Let me check: are there windows where PC missed the
# attack but later got a LATE alarm that was attributed to next window?
# That's an FP — alarm landed on window W+1 (or different) instead of W (where attack was).

# Let me check: for each PC window (1-indexed), what's the ATTACK state of its 16 frames?
# Then: what H743 alarms correspond to that window (by frames_total=16K where K=window index 1-indexed)?
# Cross-check: are those alarms TRUE (the window actually had attack) or FALSE (the window was normal)?

# Window K (1-indexed): PC frames 16(K-1)+1 .. 16K
# ARFF labels for those frames are labels[(K-1)*16 + 0 .. (K-1)*16 + 15]
# H743 alarm with frame=16K corresponds to this window.

n_windows = len(labels) // 16
print(f"\n[sim] {n_windows} windows")

# For each window K-1 (0-indexed), compute:
#   - had_attack = max(ARFF binary in 16 frames)
#   - had_h743_alarm = any alarm with frame=16(K)
#   - if had_h743_alarm, alarm's ARFF binary for that exact frame
windows = []
h743_alarms_by_window = defaultdict(list)  # window_index (1-indexed) → list of (frame_idx, pc_lag_ms)
for t, fi in alarms:
    W = fi // 16  # 1-indexed (16 → 1, 32 → 2)
    h743_alarms_by_window[W].append((fi, t - sends_by_frame.get(fi, t)))

for K1 in range(1, n_windows + 1):  # 1-indexed
    start = (K1 - 1) * 16
    end = K1 * 16
    window_binary = [labels[i][0] for i in range(start, end)]
    had_attack = any(window_binary)
    cats = set(labels[i][1] for i in range(start, end))
    h743_alarms = h743_alarms_by_window.get(K1, [])
    had_alarm = bool(h743_alarms)
    windows.append({
        "K1": K1,
        "had_attack": had_attack,
        "had_alarm": had_alarm,
        "n_attack_frames": sum(window_binary),
        "categories": cats,
        "n_h743_alarms": len(h743_alarms),
    })

# Cross-tab
counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
for w in windows:
    if w["had_attack"] and w["had_alarm"]:
        counts["TP"] += 1
    elif not w["had_attack"] and w["had_alarm"]:
        counts["FP"] += 1
    elif w["had_attack"] and not w["had_alarm"]:
        counts["FN"] += 1
    else:
        counts["TN"] += 1

print("\n=== Cross-tabulation IF H743 alarms correctly attributed to their window ===")
print(f"{'Category':<10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
print(f"{'Direct':<10} {tp:>6} {fp:>6} {fn:>6} {tn:>6}")
print(f"  → If H743 alarms map perfectly to ARFF-labeled windows, this is the answer.")
print(f"  Recall={tp/(tp+fn)*100:.2f}% Prec={tp/(tp+fp)*100:.2f}% F1={2*tp/(2*tp+fp+fn)*100:.2f}%")

# But the actual eval gave: TP=6969, FP=2001, FN=1852, TN=6342
# Diff = (FP-FP_direct, FN-FN_direct, etc.)
# Let's see how much the eval's numbers diverge from "ideal direct" mapping

actual = {"TP": 6969, "FP": 2001, "FN": 1852, "TN": 6342}
print(f"\n=== Diff (Actual - Ideal Direct) ===")
print(f"  TP_direct={tp}, Actual_TP=6969, Diff={tp-6969}")
print(f"  FP_direct={fp}, Actual_FP=2001, Diff={fp-2001}")
print(f"  FN_direct={fn}, Actual_FN=1852, Diff={fn-1852}")
print(f"  TN_direct={tn}, Actual_TN=6342, Diff={tn-6342}")

# H743 has 8971 alarms. Direct mapping says 7447 (= TP+FP) hit windows. Where are the other 1524?
# Hmm let's check. had_alarm counts.
n_with_alarm = sum(1 for w in windows if w["had_alarm"])
n_with_attack = sum(1 for w in windows if w["had_attack"])
n_alarm_attack = sum(1 for w in windows if w["had_alarm"] and w["had_attack"])
print(f"\n[stats] windows with H743 alarm: {n_with_alarm}")
print(f"[stats] windows with attack (ARFF): {n_with_attack}")
print(f"[stats] windows with both: {n_alarm_attack}")
