#!/usr/bin/env python3
"""
Round 63 fix simulation — Test lag-compensated window assignment.

Hypothesis: H743 alarm with frame=K is intended for window (K-1)//16.
But it arrives at PC 80ms (2 frames) late, so PC's _evaluate_window
for that window has already fired with empty _direct_alarms (becoming FN).

What if we attribute the alarm to W=(K-1)//16 - SHIFT, where SHIFT=1 or 2,
to compensate for the H743→PC pipeline lag?

Test cases on test_20261012-002126_auto-40ms:
  Per-alarm: 8971 alarms, all on attack frames (TP at frame level)
  Per-window actual: TP=6969, FP=2001, FN=1852, TN=6342 → F1=78.34%
  Per-window ideal (alarm → (frame-1)//16): TP=7523, FP=1448, FN=1298, TN=6895 → F1=84.57%
  Lag distribution: mean=80ms (2 frames), 95% lag > 40ms
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
print(f"[load] {len(labels):,} ARFF rows, {n_windows} windows")

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

# For each window W (0-indexed), compute:
#   - had_attack = max(binary in W*16 .. W*16+15)
# And map each alarm (frame=16K) to window assignment:
#   - Direct: W=(K-1)//16 = K-1
#   - Shift-1: W=(K-1)//16 - 1
#   - Shift-2: W=(K-1)//16 - 2
#   - Shift-3: W=(K-1)//16 - 3
#   - Legacy (deque) approximation: frame_count-1, where frame_count=send_time+lag/40ms
#     We approximate frame_count = (sends_by_frame[16K]+lag)/40 = K + lag_frames
#     Window = ceil(K + lag_frames - 1, /16) - 1

def compute_metrics(window_alarm, n_windows):
    """window_alarm[w] = set of (frame_idx, h743_frame) seen for window w"""
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
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    return counts, f1, prec, rec


def assign_alarms(shift=0):
    """Assign alarms to windows with optional shift compensation."""
    win_alarm = defaultdict(list)
    for t_alarm, fi in alarms:
        K = fi // 16  # 1-indexed target window
        W = max(0, K - 1 - shift)  # shift windows earlier
        if W < n_windows:
            win_alarm[W].append(fi)
    return win_alarm


def assign_alarms_legacy_path():
    """Simulate the legacy deque path: alarm lands in window that PC's
    _evaluate_window will close next."""
    win_alarm = defaultdict(list)
    for t_alarm, fi in alarms:
        # Lag in frames ≈ (t_alarm - t_send_at_fi) / 40
        t_send_fi = sends_by_frame.get(fi, 0)
        lag_ms = t_alarm - t_send_fi
        # PC has sent ~lag_ms/40 more frames when alarm arrives
        # The deque gets _frame_count-1, and _frame_count = fi + lag_frames
        lag_frames = lag_ms / 40
        pc_frame_at_alarm = fi + lag_frames - 1
        # Window containing this frame: (pc_frame_at_alarm - 1) // 16 (1-indexed)
        # 0-indexed W = (pc_frame_at_alarm - 1) // 16
        W = int(pc_frame_at_alarm - 1) // 16
        W = max(0, min(W, n_windows - 1))
        win_alarm[W].append(fi)
    return win_alarm


# Compute for various shift values
print("\n=== Window assignment strategies ===")
print(f"{'Strategy':<25} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'F1':>7} {'Prec':>7} {'Recall':>7}")

# Strategy 1: direct (current Round 61, no compensation)
counts, f1, prec, rec = compute_metrics(assign_alarms(shift=0), n_windows)
print(f"{'Direct (Round 61)':<25} {counts['TP']:>6} {counts['FP']:>6} {counts['FN']:>6} {counts['TN']:>6} {f1*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}%")

# Strategy 2: legacy deque approximation (what PC actually does today)
counts, f1, prec, rec = compute_metrics(assign_alarms_legacy_path(), n_windows)
print(f"{'Legacy deque (current)':<25} {counts['TP']:>6} {counts['FP']:>6} {counts['FN']:>6} {counts['TN']:>6} {f1*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}%")

# Strategy 3: shift compensation
for shift in [1, 2, 3]:
    counts, f1, prec, rec = compute_metrics(assign_alarms(shift=shift), n_windows)
    label = f"Direct + Shift-{shift}"
    print(f"{label:<25} {counts['TP']:>6} {counts['FP']:>6} {counts['FN']:>6} {counts['TN']:>6} {f1*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}%")

# Strategy 4: lag-aware assignment (window based on arrival time)
# Map to ceil((fi + lag_frames - 1) / 16) - 1 (mimics deque)
print(f"\n[NOTE] Both Direct (7523→FP+FN expected) and Legacy-deque (CURRENT PC behavior with shift-back)")
print(f"      can produce F1 ~84% if attribution aligns with ground truth.")
