# Stack11: 11-Architecture × 5-Seed × 3 Meta-Learner Leaderboard

**生成时间**: 2026-09-16  
**数据集**: 23-dim × window=16 (IanArffDataset v2)  
**架构数**: 11 (9 深度 + LightGBM + RandomForest)  
**每架构 seed**: 5 (level-1 列数 = 11 × 5 = 55)  
**阈值**: 0.5  
**评估协议**: 5-fold StratifiedKFold on test — 元学习器 OOF 预测，无 test 泄漏  
**Test 样本数**: N = 3432

## 1. Meta-Learner 4-Metric 对比（CV-based honest test）

| Rank | Method | Accuracy | Precision | Recall | F1-Score |
|------|--------|---------:|----------:|-------:|---------:|
| 1 | M3 LR stacking (L2, 5-fold CV + inner C selection) | 0.8875 | 0.9482 | 0.8316 | 0.8861 |
| 2 | M2 Weighted average (SLSQP, 5-fold CV) | 0.8590 | 0.9577 | 0.7657 | 0.8510 |
| 3 | M1 Simple average | 0.8552 | 0.9573 | 0.7584 | 0.8464 |

## 2. In-Sample Upper Bound (full-test refit, overestimates)

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|---------:|----------:|-------:|---------:|
| M1 Simple average | 0.8552 | 0.9573 | 0.7584 | 0.8464 |
| M2 Weighted average (SLSQP, 5-fold CV) | 0.8590 | 0.9577 | 0.7657 | 0.8510 |
| M3 LR stacking (L2, 5-fold CV + inner C selection) | 0.8899 | 0.9496 | 0.8349 | 0.8886 |

## 3. M2 (Weighted Avg) — Top 8 Arch×Seed Weights (full-data fit)

| Rank | Architecture (seed) | Weight |
|------|---------------------|-------:|
| 1 | LightGBM (s=123) | 0.1477 |
| 2 | TCN (no SE) (s=123) | 0.0581 |
| 3 | LightGBM (s=1024) | 0.0494 |
| 4 | BiLSTM (h128) (s=1024) | 0.0447 |
| 5 | TCN+SE (s=42) | 0.0397 |
| 6 | Pure CNN (s=789) | 0.0375 |
| 7 | TCN+SE (s=123) | 0.0358 |
| 8 | BiGRU (h128) (s=123) | 0.0334 |

## 4. M3 (LR Stacking) — Top 8 Coefs by |coefficient|

| Rank | Architecture (seed) | Coefficient |
|------|---------------------|------------:|
| 1 | TCN (no SE) (s=1024) | +2.2879 |
| 2 | TCN (no SE) (s=123) | +2.1485 |
| 3 | BiGRU (s=789) | +1.4650 |
| 4 | CNN-LSTM (ch256,h128) (s=1024) | +1.3740 |
| 5 | Pure CNN (s=789) | -1.3665 |
| 6 | Random Forest (s=456) | +1.3358 |
| 7 | BiGRU (h128) (s=456) | -1.2465 |
| 8 | BiLSTM (s=456) | -1.1820 |

## 5. 关键发现

1. **最佳 meta-learner (CV honest)**: M3 LR stacking (L2, 5-fold CV + inner C selection) — Test F1 = 0.8861
2. **vs TCN 基线** (F1=0.8795): +0.66%
3. **CV vs in-sample**: 差距反映 meta-learner 对训练集的拟合程度
4. **M2 权重分布**: 反映各架构贡献度；非零权重 arch 都提供独立信号
5. **M3 LR stacking**: 通过 L2 正则 + 内层 CV 选 C