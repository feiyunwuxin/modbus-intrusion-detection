# Multi-Scale TCN 5-Seed Leaderboard

**Date**: 2026-09-16  
**Architecture**: 3 MultiScaleTCNBlocks × parallel kernels=[1,3,5,7] × dilations=[1,2,4]  
**Fusion**: Concat → 1x1 conv (ch=128→32) + residual  
**Reference**: TCN baseline F1=0.8795; Stacking M3@0.390 F1=0.8904

## 1. 5-Seed 概率平均集成

| Metric | Value |
|--------|------:|
| Accuracy | 0.8622 |
| Precision | 0.9562 |
| Recall | 0.7734 |
| **F1-Score** | **0.8551** |

## 2. 关键对比

| Comparison | Delta |
|------------|------:|
| vs TCN baseline (F1=0.8795) | -2.44% |
| vs Stacking M3@0.390 (F1=0.8904) | -3.53% |
| Target F1 >= 0.88 | MISS |

## 3. Per-Seed F1 详情

| Seed | Accuracy | Precision | Recall | F1-Score |
|-----:|---------:|----------:|-------:|---------:|
| 42 | 0.8558 | 0.9530 | 0.7634 | 0.8477 |
| 123 | 0.8479 | 0.9367 | 0.7623 | 0.8406 |
| 456 | 0.8677 | 0.9543 | 0.7861 | 0.8621 |
| 789 | 0.8712 | 0.9534 | 0.7939 | 0.8664 |
| 1024 | 0.8022 | 0.8379 | 0.7734 | 0.8044 |

## 4. 5-Seed 集成混淆矩阵 (thr=0.5)

```
              预测 Normal    预测 Attack
实际 Normal       1563           64
实际 Attack        409         1396
```

## 5. 关键发现

1. **目标失败**: ❌ F1=0.8551 < 0.88
2. **vs TCN baseline**: -2.44%
3. **TCN 单架构 tuning 已穷尽**: SWA/EMA, augmentation, multi-scale 三轮失败