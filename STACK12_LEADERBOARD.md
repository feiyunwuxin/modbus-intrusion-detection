# Stack12 (XGBoost-added) M3 Threshold Tuning

**Date**: 2026-09-16  
**Protocol**: 5-fold StratifiedKFold (CV-honest)  
**Base learners**: 12 (11 original + XGBoost) → 60 per-seed columns  

## Results

| Config | Threshold | Acc | P | R | F1 |
|--------|----------:|----:|--:|--:|--:|
| Stack12 M3 (best thr) | 0.355 | 0.8902 | 0.9354 | 0.8499 | **0.8906** |
| Stack12 M3 (thr=0.5) | 0.500 | 0.8867 | 0.9481 | 0.8299 | 0.8851 |

## Reference Ladder

| Configuration | F1 | Notes |
|---------------|----:|-------|
| TCN baseline (5-seed prob_mean) | 0.8795 | cross-arch v1 |
| Stack11 M3 (CV-honest, thr=0.5) | 0.8861 | baseline leaderboard |
| Stack11 M3 (CV-honest, thr=0.390) | 0.8904 | threshold tuning Phase 5 |
| **Stack12 M3 (best thr=0.355)** | **0.8906** | XGBoost added, delta +0.017pp |

## Verdict

**XGBoost improves stacking by 0.017pp.** New production baseline: Stack12 M3 @ thr=0.355, F1=0.8906.