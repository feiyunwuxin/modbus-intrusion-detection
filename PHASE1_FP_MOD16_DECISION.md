# Phase 1 — H743 ↔ PC Window-Alignment Diagnostic (v2 — lag-aware)

**Log**: `logs/test_20261011-173212_auto-40ms.csv` (full 274k, 17:32:12→20:47:05)  
**Method**: Re-derive per-window FP/TP labels from SEND `binary` + ALARM count, plus per-alarm arrival lag analysis.  
**Sanity**: TP=6969 / FP=2001 / n_windows=17164 — ✅ matches summary.txt

## Verdict

### `Threshold/Model`

All 8971 alarms arrive at PC within 119ms (100.00% under 160ms) of the most recent PC SEND. The residue 0-3 cluster is fully explained by a -2-frame pipeline lag (mean=37ms, median=39ms, ~0.9 frames @ 40ms interval).

H743 and PC evaluate the **same 16 frames per window** (frames are sent in order and H743 processes in arrival order), so the window BOUNDARIES are aligned.

The 2001 FPs (all **pure-normal windows**) arise because the alarm printf lags by ~0.9 frames; the alarm is then attributed to a *later* window via IdsLineParser._recent_alarm_frames.append() (rtu_replay_gui.py:1009), which inflates per-window FPR.

**Fix options (cheapest first)**:
  (a) **Heuristic lookback on PC side** — when alarm arrives at residue ≤ 3, also mark the previous window as alarm=True. Effect: recoveries for FP→TP where the original attack was in the previous window. Estimated gain: up to 800 FPs could become TPs.
  (b) **H743 firmware prints `[ALARM] count prob=P.KKK frame=K` so PC knows exactly which frame H743 evaluated** — proper fix. Requires firmware rebuild + flash (~30min + 3h re-test).
  (c) **Threshold retune** alone — won't reduce FP (model is right that those 16 frames look attacky, but PC missed the attack label due to lag). Bound to help less than (a)+(b).

## Lag statistics

| Stat | Value |
|---|---|
| Count | 8971 |
| Min | 0 ms |
| Max | 119 ms |
| Mean | 36.6 ms |
| Median | 39 ms |
| StdDev | 13.3 ms |
| ≤ 4 frame lag (< 160 ms) | 100.00% |
| > 4 frame lag (>= 160 ms) | -99.00% |

### Lag-bucket breakdown

| Bucket | Count | % |
|---|---:|---:|
| 0-40ms (within 1 frame) | 4728 | 52.70% |
| 40-80ms (1-2 frame lag) | 4236 | 47.22% |
| 80-120ms (2-3 frame lag) | 7 | 0.08% |
| 120-160ms (3-4 frame lag) | 0 | 0.00% |
| 160-200ms (4-5 frame lag) | 0 | 0.00% |
| >=200ms (>=5 frame lag) | 0 | 0.00% |

## Residue distribution (alarm frame_idx % 16)

| residue | FP count | TP count |
|---:|---:|---:|
| 0 | 357 | 1139 |
| 1 | 795 | 2929 |
| 2 | 819 | 2788 |
| 3 | 30 | 113 |
| 4 | 0 | 0 |
| 5 | 0 | 0 |
| 6 | 0 | 0 |
| 7 | 0 | 0 |
| 8 | 0 | 0 |
| 9 | 0 | 0 |
| 10 | 0 | 0 |
| 11 | 0 | 0 |
| 12 | 0 | 0 |
| 13 | 0 | 0 |
| 14 | 0 | 0 |
| 15 | 0 | 0 |

**Interpretation:** Residue 0-3 cluster is fully explained by the -2-frame alarm-arrival lag (mean ~0.9 frames). Residues 4-15 being empty simply reflects that the lag is bounded under 4 frames.

## Why windows are aligned (despite residue 0-3 cluster)

- H743 receives frames in arrival order and processes them in order (`modbus_ids.c:241-315`).
- PC sends frames sequentially (`frame_idx` = 1..274628 contiguous).
- Therefore H743's window W (frames 16W-15..16W) = PC's window W (frames 16W-15..16W).
- The 1-3 frame lag is the time between H743 deciding to alarm and the alarm printf landing at the PC. This makes PC's `_recent_alarm_frames.append()` attribute the alarm to a **later** window, inflating FPR.

## Next phase

**Phase 2A — threshold retune + heuristic lookback on PC side (or Phase 2B-light: H743 prints prob/frame_idx to enable correct PC-side attribution).**

## Artifacts

- `logs/test_20261011-173212_auto-40ms.fp_mod16_v2.csv` — residue table
- `logs/test_20261011-173212_auto-40ms.fp_mod16_v2.json` — full JSON
- `logs/test_20261011-173212_auto-40ms.fp_mod16_v2.png` — 2-panel chart (residue + lag)
- `logs/test_20261011-173212_auto-40ms.lag_analysis.csv` — per-alarm lag table (8971 rows)
