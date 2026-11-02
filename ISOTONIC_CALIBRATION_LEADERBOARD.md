# M3 Stacking + Isotonic Calibration Leaderboard

**Date**: 2026-09-16  
**Protocol**: Nested CV — outer 5-fold for M3 stacking + inner 5-fold for isotonic  
**Honesty**: each calibrated prob never saw its own label (fully CV-honest)

## Results

| Version | Best Thr | Acc | Precision | Recall | F1 |
|---------|---------:|----:|----------:|-------:|---:|
| Raw OOF | 0.390 | 0.8904 | 0.9397 | 0.8460 | 0.8904 |
| + Isotonic calibration | 0.320 | 0.8887 | 0.9320 | 0.8504 | 0.8893 |

## Reference Ladder

| Configuration | F1 | Notes |
|---------------|------:|-------|
| TCN baseline (5-seed prob_mean) | 0.8795 | cross-arch v1 |
| M3 stacking (CV-honest, thr=0.5) | 0.8861 | baseline leaderboard |
| M3 stacking (CV-honest, best thr=0.390) | 0.8904 | threshold tuning |
| M3 + isotonic calibration (best thr) | 0.8893 | THIS RUN, delta -0.104pp |

## Verdict

**Isotonic hurts**: -0.104pp F1. Raw M3 with thr=0.390 remains the production candidate.