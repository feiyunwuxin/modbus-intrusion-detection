# FP32 vs MCU Hybrid INT8 Comparison (seed=42)

**N samples**: 200

| Metric | FP32 (.pt) | MCU (.h) |
|--------|-----------|----------|
| Threshold | 0.50 | 0.49 |
| Accuracy | 0.5900 | 0.5900 |
| F1 | 0.7421 | 0.7421 |
| Inference (ms/sample) | 0.82 | 0.83 |

## Numerical agreement

- Max prob diff: **1.83e-03** (FAIL < 1e-5)
- Mean prob diff: 2.01e-05
- Verdict agreement: **100.00%**
