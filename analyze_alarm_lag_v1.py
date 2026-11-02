#!/usr/bin/env python3
"""
Round 63 diagnostic: Determine ordering of ALARM vs SEND for the SAME frame.

If H743 fires for window K (frames 16(K-1)+1..16K) and printf carries frame=16K,
the alarm arrives at PC AFTER PC already evaluated window K-1 (at frame 16K send).

Test: pair each ALARM with the SEND that triggered it (frame=16K).
Compare t_event_ms of ALARM vs SEND-with-frame-16K.
If ALARM > SEND by 5-50ms, the lag-induced FP analysis is correct.
If ALARM < SEND, then on_alarm_with_frame happens BEFORE _evaluate_window.

Also: for each ALARM frame=16K, what's window index that should be tagged?
Compute W_tag = (16K-1)//16 = K-1.
Check if at the time of ALARM (PC t_event_ms), what was PC's _frame_count?
"""
import csv
from pathlib import Path

LOG = Path("logs/test_20261012-002126_auto-40ms.csv")

sends_by_frame = {}  # frame_idx -> t_event_ms
alarms = []  # list of (t_event_ms, frame_idx, alarm_count)
with LOG.open() as f:
    r = csv.DictReader(f)
    for row in r:
        if row["event"] == "SEND":
            try:
                fi = int(row["frame_idx"])
                t = int(row["t_event_ms"])
                sends_by_frame[fi] = t
            except (ValueError, KeyError):
                continue
        elif row["event"] == "ALARM":
            try:
                fi_str = row["frame_idx"]
                if not fi_str:
                    continue
                fi = int(fi_str)
                t = int(row["t_event_ms"])
                ac = int(row["alarm_count"])
                alarms.append((t, fi, ac))
            except (ValueError, KeyError):
                continue

print(f"[load] {len(sends_by_frame):,} SEND events; {len(alarms):,} ALARM events")

# For each alarm, find the matching send (frame=16K, the frame that triggered H743 inference)
# alarm frame=16K means H743 processed frame 16K
# Pair: SEND(t_send, fi=16K) and ALARM(t_alarm, fi=16K)
# Lag = t_alarm - t_send
lags = []
for t_alarm, fi, ac in alarms:
    if fi in sends_by_frame:
        t_send = sends_by_frame[fi]
        lags.append((t_alarm - t_send, fi, ac, t_send, t_alarm))

if not lags:
    print("[NO MATCH] No ALARM frame matches SEND frame")
else:
    lag_vals = [x[0] for x in lags]
    print(f"\n=== ALARM arrival lag from H743 inference-frame SEND ===")
    print(f"Count: {len(lags)}")
    print(f"Min: {min(lag_vals)} ms")
    print(f"Max: {max(lag_vals)} ms")
    print(f"Mean: {sum(lag_vals)/len(lag_vals):.1f} ms")
    sorted_lags = sorted(lag_vals)
    median = sorted_lags[len(sorted_lags)//2]
    p25 = sorted_lags[len(sorted_lags)//4]
    p75 = sorted_lags[3*len(sorted_lags)//4]
    p95 = sorted_lags[int(0.95*len(sorted_lags))]
    print(f"Median: {median} ms")
    print(f"P25/P75: {p25}/{p75} ms")
    print(f"P95: {p95} ms")

    # Bucket
    buckets = [(0, 20), (20, 40), (40, 80), (80, 160), (160, 500), (500, 99999)]
    print("\n=== Lag bucket distribution ===")
    for lo, hi in buckets:
        c = sum(1 for l in lag_vals if lo <= l < hi)
        print(f"  {lo:>4}-{hi:>4}ms: {c:>5} ({100*c/len(lags):.1f}%)")

    # Distribution of alarm arrival vs PC window evaluation
    # Window K-1 evaluation happens at t_send(16K) (when PC sends frame 16K)
    # If lag > 0, alarm arrives AFTER window K-1 was evaluated
    # If lag < 0, alarm arrived BEFORE window K-1 was evaluated (impossible? H743 needs to receive frame first)
    print("\n=== Critical: Lag sign ===")
    positive = sum(1 for l in lag_vals if l > 0)
    negative = sum(1 for l in lag_vals if l < 0)
    zero = sum(1 for l in lag_vals if l == 0)
    print(f"Positive lag (alarm after send): {positive} ({100*positive/len(lags):.1f}%)")
    print(f"Zero lag: {zero}")
    print(f"Negative lag (alarm before send??): {negative}")

    # For alarms with lag > 0:
    # Window K-1 (0-indexed, 1-indexed K) closes at t_send(16K).
    # Alarm arrives at t_send(16K) + lag.
    # By then PC has sent more frames (lag/40ms more).
    # If 0 < lag < 40ms, PC is filling next window (frames 16K+1..16K+16).
    # If lag >= 40ms, PC has evaluated NEXT WINDOW also.
    # If lag >= 80ms, two more windows evaluated.
    print("\n===  Lag buckets mapped to PC state at alarm arrival ===")
    print("If lag=37ms, PC has sent ~1 more frame (16K+1). Window K-1 already evaluated.")
    print("If lag=80ms, PC has sent 2 more frames (16K+2). Window K-1 evaluated.")
    print("If lag=120ms, PC has sent 3 more frames. Window K-1 evaluated.")
