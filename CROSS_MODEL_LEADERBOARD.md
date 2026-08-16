# Cross-Model Leaderboard — 所有历史最佳结果

**生成时间**: 2026-07-25

**数据源**: 4 个 ensemble + TCN+SE LR=2e-3 self-ensemble + 23-dim batch sweep + 64-combos 23-dim 全表

## Top 25 排名 (按 Test Macro-F1 降序)

| Rank | Category | Name | F1m | PR-AUC | BinF1 | Acc | ROC | thr | #seed | #mdl |
|------|----------|------|-----|--------|-------|-----|-----|-----|-------|------|
| 1 | Self-Ensemble | TCN+SE LR=2e-3 5-seed ensemble (ch=64, B=128, ep=40) | 0.8735 | 0.9269 | 0.8711 | 0.8735 | 0.9067 | 0.42 | 5 | 5 |
| 2 | Ensemble-STA | ENSEMBLE_v4 stacking best (ALL_7_v2) | 0.8481 | 0.8473 | 0.7577 | 0.9018 | 0.9248 | 0.54 | 1 | 10 |
| 3 | Ensemble-STA | ENSEMBLE_v2 stacking best (ALL_7) | 0.8478 | 0.8433 | 0.7566 | 0.9026 | 0.9245 | 0.55 | 1 | 7 |
| 4 | Ensemble-STA | ENSEMBLE_v3 stacking best (LGB+RF+TCN_v4+TCN_v4_v2+LSTM+Att+LSTM+Att_v2) | 0.8458 | 0.8541 | 0.7533 | 0.9012 | 0.9250 | 0.52 | 1 | 9 |
| 5 | Ensemble-STA | ENSEMBLE_v1 stacking best (ALL_6) | 0.8455 | 0.8413 | 0.7534 | 0.9003 | 0.9247 | 0.55 | 1 | 6 |
| 6 | Single-Model | TCN+SE 23-dim B=64 5-seed (this work) | 0.8454 | 0.9143 | 0.8362 | 0.8459 | 0.8740 | 0.50 | 5 | 1 |
| 7 | Ensemble-WEI | ENSEMBLE_v1 weighted best (LGB+RF+TCN_v4) | 0.8438 | 0.8502 | 0.7494 | 0.9008 | 0.9248 | 0.57 | 1 | 6 |
| 8 | Ensemble-WEI | ENSEMBLE_v2 weighted best (LGB+RF+LSTM_Att) | 0.8399 | 0.8466 | 0.7429 | 0.8985 | 0.9232 | 0.57 | 1 | 7 |
| 9 | Single-Model | TCN+SE 23-dim B=96 5-seed (this work) | 0.8391 | 0.9098 | 0.8264 | 0.8401 | 0.8689 | 0.50 | 5 | 1 |
| 10 | Single-Model | TCN+SE 23-dim B=128 5-seed (this work) | 0.8383 | 0.9037 | 0.8263 | 0.8392 | 0.8690 | 0.50 | 5 | 1 |
| 11 | Single-Model | TCN+SE 23-dim B=192 5-seed (this work) | 0.8372 | 0.9003 | 0.8251 | 0.8381 | 0.8669 | 0.50 | 5 | 1 |
| 12 | Single-Model | TCN+SE 23-dim B=256 5-seed (this work) | 0.8356 | 0.9085 | 0.8223 | 0.8367 | 0.8706 | 0.50 | 5 | 1 |
| 13 | Ensemble-WEI | ENSEMBLE_v3 weighted best (LGB+RF+TCN_v4+TCN_v4_v2) | 0.8349 | 0.8521 | 0.7384 | 0.8911 | 0.9266 | 0.60 | 1 | 9 |
| 14 | Ensemble-WEI | ENSEMBLE_v4 weighted best (ALL_7_v2) | 0.8316 | 0.8417 | 0.7402 | 0.8811 | 0.9191 | 0.53 | 1 | 10 |
| 15 | Ensemble-UNI | ENSEMBLE_v1 uniform best (LGB+RF+TCN_v4) | 0.8302 | 0.8403 | 0.7381 | 0.8802 | 0.9172 | 0.55 | 1 | 6 |
| 16 | Single-Model | TCN+SE 23-dim (-length,setpoint,crc_mean_w,cmd_count_w) 5-seed | 0.8287 | 0.9079 | 0.8136 | 0.8301 | 0.8665 | 0.50 | 5 | 1 |
| 17 | Ensemble-UNI | ENSEMBLE_v2 uniform best (LGB+RF+LSTM_Att) | 0.8268 | 0.8349 | 0.7328 | 0.8779 | 0.9150 | 0.55 | 1 | 7 |
| 18 | Single-Model | TCN+SE 23-dim (-function,setpoint,crc_mean_w,cmd_count_w) 5-seed | 0.8266 | 0.9044 | 0.8110 | 0.8280 | 0.8633 | 0.50 | 5 | 1 |
| 19 | Single-Model | TCN+SE 23-dim (-function,setpoint,press_mean_w,crc_mean_w) 5-seed | 0.8255 | 0.8977 | 0.8103 | 0.8269 | 0.8565 | 0.50 | 5 | 1 |
| 20 | Single-Model | TCN+SE 23-dim (-function,setpoint,press_mean_w,cmd_count_w) 5-seed | 0.8250 | 0.8932 | 0.8084 | 0.8266 | 0.8554 | 0.50 | 5 | 1 |
| 21 | Single-Model | TCN+SE 23-dim (-length,setpoint,press_mean_w,crc_mean_w) 5-seed | 0.8214 | 0.8964 | 0.8052 | 0.8230 | 0.8539 | 0.50 | 5 | 1 |
| 22 | Single-Model | TCN+SE 27-dim baseline 5-seed | 0.8205 | 0.9052 | 0.8044 | 0.8220 | 0.8624 | 0.50 | 5 | 1 |
| 23 | Ensemble-UNI | ENSEMBLE_v3 uniform best (LGB+RF+TCN_v4+TCN_v4_v2) | 0.8049 | 0.8008 | 0.6949 | 0.8669 | 0.8914 | 0.66 | 1 | 9 |
| 24 | Ensemble-UNI | ENSEMBLE_v4 uniform best (ALL_7_v2) | 0.7930 | 0.7643 | 0.6713 | 0.8646 | 0.8688 | 0.80 | 1 | 10 |

## 分类冠军

### Self-Ensemble

- **TCN+SE LR=2e-3 5-seed ensemble (ch=64, B=128, ep=40)**
  F1m = 0.8735, PR-AUC = 0.9269, thr = 0.42

### Ensemble-STA

- **ENSEMBLE_v4 stacking best (ALL_7_v2)**
  F1m = 0.8481, PR-AUC = 0.8473, thr = 0.54

### Single-Model

- **TCN+SE 23-dim B=64 5-seed (this work)**
  F1m = 0.8454, PR-AUC = 0.9143, thr = 0.50

### Ensemble-WEI

- **ENSEMBLE_v1 weighted best (LGB+RF+TCN_v4)**
  F1m = 0.8438, PR-AUC = 0.8502, thr = 0.57

### Ensemble-UNI

- **ENSEMBLE_v1 uniform best (LGB+RF+TCN_v4)**
  F1m = 0.8302, PR-AUC = 0.8403, thr = 0.55

## 关键洞察

1. **绝对冠军**: TCN+SE LR=2e-3 self-ensemble (5 seeds) → F1m=0.8735, **+0.0254** 超过所有跨架构 stacking
2. **跨架构 stacking 冠军**: ENSEMBLE_v4 Stack ALL_7_v2 → F1m=0.8481
3. **单模冠军**: TCN+SE 23-dim B=64 (5 seeds) → F1m=0.8454, ≈ stack 跨架构冠军
4. **PR-AUC 冠军**: TCN+SE LR=2e-3 self-ensemble → 0.9269, 远超所有单模/集成
5. **下一步机会**: 用 23-dim + B=64 + LR=2e-3 + 5-seed self-ensemble 替换 ENSEMBLE_v4 的 TCN_v4 — 可能再 +0.02
