#!/usr/bin/env python3
"""Round 64 diagnostic: pair each ALARM frame with its ground-truth binary label.

For each H743 alarm, check whether the alarm's reported frame_idx is actually
an attack frame (binary=1) or a normal frame (binary=0) in the test log.

Why this matters:
  Round 61+62 reports F1=78.34% FPR=23.98%, but the legacy 'is_fp' field uses
  NEXT-window attribution (alarm on frame K → window (K+1)//16) and max-aggregation
  (window attack if any frame in 16 is attack). This HIDES the model's actual
  frame-level accuracy.

  Frame-level pairing reveals: when H743 fires an alarm, what fraction are on
  truly-attack frames vs truly-normal frames? This is the model's *real* precision
  before window-level max-aggregation inflates TPs.
"""
import csv
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python analyze_alarm_vs_groundtruth.py <test_log.csv>")
    sys.exit(1)

log_path = Path(sys.argv[1])
print(f"Analyze: {log_path.name}")
print("=" * 70)

# 1) Read all SEND events → build frame_idx → binary map
frame_binary = {}
with open(log_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('event') == 'SEND' and row.get('frame_idx'):
            fi = int(row['frame_idx'])
            bin_ = int(row.get('binary', 0) or 0)
            frame_binary[fi] = bin_

n_attack = sum(1 for b in frame_binary.values() if b == 1)
n_normal = sum(1 for b in frame_binary.values() if b == 0)
print(f"SEND events: {len(frame_binary)}  (attack={n_attack}, normal={n_normal})")
print()

# 2) Read all ALARM events, pair with frame's binary
alarms = []
with open(log_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('event') == 'ALARM' and row.get('frame_idx'):
            fi = int(row['frame_idx'])
            bin_ = frame_binary.get(fi, -1)
            alarms.append({
                'frame': fi,
                'binary': bin_,
                'prob_x1000': int(row.get('prob_x1000', 0) or 0),
                'log_is_fp': row.get('is_fp', ''),
                't_ms': int(row.get('t_event_ms', 0) or 0),
            })

print(f"ALARM events: {len(alarms)}")
print()

# 3) Distribution
on_attack = sum(1 for a in alarms if a['binary'] == 1)
on_normal = sum(1 for a in alarms if a['binary'] == 0)
on_missing = sum(1 for a in alarms if a['binary'] == -1)
n = max(len(alarms), 1)
print(f"Alarm frame landing distribution:")
print(f"  attack (binary=1): {on_attack:4d}  ({on_attack/n*100:.1f}%)")
print(f"  normal (binary=0): {on_normal:4d}  ({on_normal/n*100:.1f}%)")
print(f"  frame missing:     {on_missing:4d}  ({on_missing/n*100:.1f}%)")
print()

# 4) Frame-level classification view
print(f"Model classification view (by alarm's own frame binary):")
print(f"  True positive (alarm on attack frame):  {on_attack}")
print(f"  False positive (alarm on normal frame): {on_normal}")
if on_attack + on_normal > 0:
    precision_model = on_attack / (on_attack + on_normal)
    print(f"  Frame-level Precision: {precision_model:.4f} ({precision_model*100:.2f}%)")
print()

# 5) Log legacy tagging comparison
log_tp = sum(1 for a in alarms if a['log_is_fp'] == 'TP')
log_fp = sum(1 for a in alarms if a['log_is_fp'] == 'FP')
print(f"Log legacy tagging (is_fp field):")
print(f"  TP: {log_tp}    FP: {log_fp}")
print(f"  → Legacy 'next window' attribution + max-aggregation masks real FP")
print()

# 6) Threshold sweep: re-classify by prob threshold
print("Threshold sweep on alarm prob (frame-level pairing):")
for thr in [0.30, 0.40, 0.46, 0.49, 0.50, 0.60, 0.70]:
    above = sum(1 for a in alarms if a['prob_x1000'] >= thr * 1000)
    on_attack_above = sum(1 for a in alarms
                          if a['prob_x1000'] >= thr * 1000 and a['binary'] == 1)
    on_normal_above = sum(1 for a in alarms
                          if a['prob_x1000'] >= thr * 1000 and a['binary'] == 0)
    if above > 0:
        prec = on_attack_above / above
        print(f"  thr={thr:.2f}: total={above:4d}, on_attack={on_attack_above:4d}, "
              f"on_normal={on_normal_above:4d}, prec={prec:.4f}")

# 7) Per-alarm detail (first 20)
print()
print("First 20 ALARMs detail:")
print(f"{'idx':>4} {'t_ms':>6} {'frame':>5} {'label':>10} {'prob':>6} {'log_is_fp':>10}")
for i, a in enumerate(alarms[:20]):
    prob = a['prob_x1000'] / 1000.0
    if a['binary'] == 1:
        label = "ATTACK"
    elif a['binary'] == 0:
        label = "normal"
    else:
        label = "MISSING"
    print(f"{i:4d} {a['t_ms']:6d} {a['frame']:5d} {label:>10} {prob:6.3f} {a['log_is_fp']:>10}")

# 8) Window-level (per-16-frame) cross-check
print()
print("Window-level cross-check (max-aggregation: attack if ANY frame in 16 is attack):")
window_actual = {}  # W → True if any attack frame
for fi, b in frame_binary.items():
    W = (fi - 1) // 16
    if b == 1:
        window_actual[W] = True

# For each alarm, find which WINDOW its frame falls in
window_correct = 0
window_wrong = 0
for a in alarms:
    if a['binary'] == -1:
        continue
    W = (a['frame'] - 1) // 16
    if window_actual.get(W, False):
        window_correct += 1
    else:
        window_wrong += 1
print(f"  Alarm frame's own window has attack (correct alarm position): {window_correct}")
print(f"  Alarm frame's own window all normal (alarm on quiet window):   {window_wrong}")
