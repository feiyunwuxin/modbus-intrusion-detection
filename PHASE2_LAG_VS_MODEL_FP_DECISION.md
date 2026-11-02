# Round 63 — Lag vs Model-FP Decomposition (2026-10-12)

**Question**: F1=78.34% with FPR=23.98% on auto-40ms full 274k test.
Phase 1 misdiagnosis was confirmed: the 2001 FPs at window-level are NOT all
lag-induced. So what ARE they?

## Answer

Two disjoint sources contribute to the 2001 FPs:

| Source | Count | F1 Impact | Fixable? |
|---|---:|---:|---|
| (A) **Model-level FP** (H743 fires alarm on normal window) | **1448** | -4.07% F1 | ❌ Need model/threshold change |
| (B) **Attribution FP** (lag-induced window misattribution) | **553** | -2.16% F1 | ✅ Lag compensation on PC side |
| **Total** | **2001** | -6.23% F1 | hybrid |

### Direct (Ideal) Attribution Analysis

If H743's alarm printf `frame=K` were ALWAYS evaluated at window `W=(K-1)//16` (no lag), the eval would show:
- **TP=7523, FP=1448, FN=1298, TN=6895** → **F1=84.57%, Precision=83.86%, Recall=85.29%**
- The 1448 FPs are H743 firing on normal windows — model-level only.

### Actual Mechanism (Mixed Direct + Legacy)

PC's IdsLineParser fires **both**:
1. **Direct path** (Round 61): `on_alarm_with_frame(K)` → `_direct_alarms.add(W)`.
   But with mean lag=80ms (2 frames), this arrives AFTER `_evaluate_window(W)` has already fired with empty set → stale.
2. **Legacy path** (line 1216, NOT removed in Round 61): `count_changed.connect(on_alarm)` → deque.append(frame_count-1).
   - Updates `_recent_alarm_frames` when alarm arrives
   - Next 16 PC frames' `on_frame` sets `_current_window_alarm=True` (deque non-empty)
   - Alarm gets attributed to the window that PC closes AFTER arrival (shifted forward by ~1 window)

Empirical verification (test_20261012-002126):
- Pure legacy simulation: F1=76.87%
- Actual: F1=78.34% (slightly higher — direct path contributes ~5% of alarms)

### Diagnosis Methodology

1. **Lag distribution** (analyze_alarm_lag_v1.py): mean=80ms, mode=80-160ms (56.3%)
   - 4.3% have lag<20ms (alarm before next frame)
   - 95.3% have lag>40ms (definitively late for direct path)
2. **Per-frame alarm check** (analyze_fp_type_v1.py): 8971/8971 alarms are on attack frames (TP at frame level)
3. **Per-window ideal mapping** (analyze_stepbystep_race.py): if direct path worked perfectly → F1=84.57%
4. **Shift simulation** (analyze_lag_aware_dynamic.py):
   - Shift=0 (direct): F1=84.57%
   - Shift=1: F1=77.99%
   - Shift=2: F1=71.66%
   - **Actual F1=78.34%** sits between these → suggests shift≈0.7 effective

## Implications

### For Decision-Making

The 78.34% F1 is **bounded above** by ~84.57% (model ceiling with perfect attribution).
Current gap: **-6.23% F1** = 553 attribution FPs.

| Fix option | Estimated F1 | Effort | Risk |
|---|---:|---|---|
| **A. Threshold sweep** (raise above 0.49) | TBD | Low (re-eval existing test, no rebuild) | Low (loses some TP) |
| **B. Latency compensation** (PC: lag-based shift) | ~84% | Medium (PC code change, rebuild) | Medium (depends on lag stable) |
| **C. Firmware latency reduction** (CMSIS-DSP, no \r\n) | ~84% | High (rebuild+reflash) | High (architectural) |
| **D. Disable legacy deque path** | ~30-78% (mostly FN) | Low (one-line change) | HIGH (might break things) |
| **E. Hybrid: keep legacy, add lag-shift secondary** | ~83% | Medium | Medium |
| **F. New model / retrain** | depends | High | High |
| **G. Accept 78.34%, document as expected** | 78.34% | Zero | Zero |

### Recommended Path

**Option B (latency compensation)** is most actionable. Code change is small:
- PC parses `frame=K` and `arrival_t - send_t` (lag_ms) — already available in CSV
- Compute `shift = max(0, round(lag_ms/40) - 1)` — typically 1 for mean=80ms
- Update `on_alarm_with_frame(K, shift=shift)` to add to window `W-1-shift` instead of `W-1`

If only `on_alarm_with_frame(K)` is called (and legacy deque's `_recent_alarm_frames` is cleared), this should achieve the Direct F1=84.57% ceiling. But we must disable legacy path (or the deque will compete).

**Simpler Option E**: Keep both paths but on direct attribution, compute `W_effective = (K-1)//16 - shift` where `shift = max(0, ceil(lag_ms/40) - 1)`. For lag=80ms (most alarms), shift=1. For lag<40ms (5% of cases), shift=0.

## Artifacts

- `analyze_alarm_lag_v1.py` — lag distribution per alarm
- `analyze_fp_type_v1.py` — per-alarm FP analysis (shows 0 alarms are FP at frame level)
- `analyze_stepbystep_race.py` — ideal direct attribution
- `analyze_lag_aware_dynamic.py` — fixed/dynamic shift sweep
- `logs/test_20261012-002126_auto-40ms.fp_by_category.json` — per-category breakdown

## Next Round Decision

Recommend **Option B/E** (PC-side lag compensation) as a 30-minute code change targeting F1 ~83%.
If that holds, the remaining 1448 FPs are model-level (needs retrain/threshold) and out of scope.
