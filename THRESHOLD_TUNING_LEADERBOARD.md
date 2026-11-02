# Stacking Threshold Tuning Leaderboard

**Date**: 2026-09-16  
**Protocol**: 5-fold StratifiedKFold OOF (no test contamination)  
**Range**: thr ∈ [0.30, 0.70], step 0.005

## Results

| Method | Best Thr | Best Acc | Best P | Best R | Best F1 | @0.5 F1 | Delta |
|--------|---------:|---------:|-------:|-------:|--------:|--------:|------:|
| M1 Simple average | 0.320 | 0.8770 | 0.9282 | 0.8305 | 0.8766 | 0.8464 | +3.024% |
| M2 Weighted avg (SLSQP) | 0.365 | 0.8785 | 0.9392 | 0.8222 | 0.8768 | 0.8510 | +2.582% |
| M3 LR stacking (L2) | 0.390 | 0.8904 | 0.9397 | 0.8460 | 0.8904 | 0.8861 | +0.431% |

## References

| Method | F1 | Notes |
|--------|------:|-------|
| TCN baseline (5-seed prob_mean) | 0.8795 | cross-arch v1 leaderboard |
| Stacking M3 (CV-honest, thr=0.5) | 0.8861 | current production candidate |
| Stacking M3 (CV-honest, **best thr**) | 0.8904 | delta +0.431% |