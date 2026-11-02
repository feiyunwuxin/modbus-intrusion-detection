#!/usr/bin/env python3
"""Round 63 analysis — Alarm burst patterns.

If multiple alarms arrive in the same SerialReader poll (100ms cycle),
they get parsed in one batch and trigger delta>1 logic.
Let's check.
"""
import csv
from pathlib import Path
from collections import Counter

LOG = Path("logs/test_20261012-002126_auto-40ms.csv")

# Look at delta between consecutive alarms
alarms = []
with LOG.open() as f:
    r = csv.DictReader(f)
    for row in r:
        if row["event"] == "ALARM":
            try:
                fi = int(row["frame_idx"])
                t = int(row["t_event_ms"])
                ac = int(row["alarm_count"])
                alarms.append((t, fi, ac))
            except (ValueError, KeyError):
                continue

print(f"[load] {len(alarms)} alarms")

# Compute time deltas between consecutive alarms
deltas_t = []
deltas_count = []
for i in range(1, len(alarms)):
    dt = alarms[i][0] - alarms[i-1][0]
    dc = alarms[i][2] - alarms[i-1][2]
    deltas_t.append(dt)
    deltas_count.append(dc)

# Counter of count-deltas (number of alarms between consecutive ARFs)
delta_count_dist = Counter(deltas_count)
print("\n=== Internal alarm count deltas ===")
for d in sorted(delta_count_dist):
    print(f"  delta_count={d}: {delta_count_dist[d]}")

# Number of alarms arriving in <100ms batch
near_bursts = sum(1 for dt in deltas_t if dt < 100)
print(f"\n[burst] alarms within 100ms of previous: {near_bursts}/{len(deltas_t)} ({100*near_bursts/len(deltas_t):.1f}%)")
near_bursts_50 = sum(1 for dt in deltas_t if dt < 50)
print(f"[burst] alarms within 50ms: {near_bursts_50}/{len(deltas_t)} ({100*near_bursts_50/len(deltas_t):.1f}%)")

# How many alarms per ~100ms window (burst size)?
print("\n=== Alarm arrival rate per 100ms ===")
buckets_100ms = Counter()
for t, fi, ac in alarms:
    bucket = t // 100
    buckets_100ms[bucket] += 1

burst_dist = Counter(buckets_100ms.values())
print("Distribution of alarms per 100ms bucket:")
for k in sorted(burst_dist)[:20]:
    print(f"  {k} alarms/100ms: {burst_dist[k]} buckets")
