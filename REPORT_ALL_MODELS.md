# All-Model Comparison — 21+ IDS models (2026-06-12)

Sorted by **Test Macro-F1** (after threshold tuning on val).

| Rank | Model | Family | Macro-F1 | Binary-F1 | Acc | PR-AUC | ROC-AUC | Params | Time | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **Ensemble: Stack ALL_7** | Stacking | 0.8481 | 0.7443 | 0.9018 | 0.8473 | 0.9186 | Σ multi | Σ multi | LGB+RF+TCN_v4+v3a+v3b+v3c+LSTM+Att → LR |
| 2 | **Ensemble: Stack ALL_6** | Stacking | 0.8455 | 0.7417 | 0.9003 | 0.8413 | 0.9186 | Σ multi | Σ multi | LGB+RF+TCN_v4+v3a+v3b+v3c → LR |
| 3 | **Ensemble: Stack ALL_10** | Stacking | 0.8450 | 0.7410 | 0.9000 | 0.8394 | 0.9183 | Σ multi | Σ multi | +Transformer → LR |
| 4 | **Ensemble: Stack ALL_9** | Stacking | 0.8443 | 0.7421 | 0.8993 | 0.8488 | 0.9201 | Σ multi | Σ multi | +TCN_v4_v2+LSTM+Att_v2 → LR |
| 5 | **Ensemble: Weighted 3** | Weighted | 0.8438 | 0.7380 | 0.8912 | 0.8502 | 0.9183 | Σ multi | Σ multi | LGB+RF+TCN_v4 → w=[0.10,0.81,0.09] |
| 6 | **Ensemble: Stack 3** | Stacking | 0.8411 | 0.7338 | 0.8902 | 0.8521 | 0.9192 | Σ multi | Σ multi | LGB+RF+TCN_v4 → LR |
| 7 | **Ensemble: Weighted 4** | Weighted | 0.8403 | 0.7389 | 0.8897 | 0.8521 | 0.9187 | Σ multi | Σ multi | LGB+RF+TCN_v4+v3a → SLSQP |
| 8 | **TCN v4 v2 SCADA w=16** | TCN++ | 0.8367 | 0.8241 | 0.8377 | 0.8990 | 0.8643 | 79321 | 67.0 | v4 + SE on v2 data |
| 9 | **TCN v4 +SE w=16** | TCN++ | 0.8340 | 0.8207 | 0.8351 | 0.9025 | 0.8588 | 79321 | 66.9 | v3b + SE attention |
| 10 | **LGB v2** | Boosted Tree | 0.8337 | 0.7287 | 0.8999 | 0.8166 | 0.9099 | 0 | 0.0 | 16 raw + time_diff, thr=0.64 |
| 11 | **TCN v3a w=8 (deep)** | TCN+ | 0.8336 | 0.8158 | 0.8355 | 0.8813 | 0.8454 | 150913 | 188.0 | 6 blocks, dilations=[1,2,4,8,16,32] |
| 12 | **Random Forest** | Bagged Tree | 0.8330 | 0.7359 | 0.8895 | 0.8364 | 0.9153 | 766289 | 11.0 | 500 trees |
| 13 | **TCN v3b w=16 (wide)** | TCN+ | 0.8326 | 0.8214 | 0.8333 | 0.8999 | 0.8587 | 76033 | 60.9 | 3 blocks, w=16 |
| 14 | **CNN-LSTM w=8** | Hybrid | 0.8296 | 0.8071 | 0.8326 | 0.8904 | 0.8455 | 170305 | 63.0 | Conv1D + BiLSTM |
| 15 | **TCN v2 w=8** | TCN | 0.8263 | 0.8040 | 0.8291 | 0.8952 | 0.8551 | 76033 | 78.0 | 3 blocks, dilations=[1,2,4] |
| 16 | **MobileNet1D w=8** | Mobile | 0.8201 | 0.7954 | 0.8235 | 0.8749 | 0.8497 | 43329 | 71.0 | depthwise-separable |
| 17 | **TCN v3c w=16 deep** | TCN+ | 0.8200 | 0.8133 | 0.8202 | 0.9061 | 0.8596 | 150913 | 126.0 | 6 blocks, w=16 |
| 18 | **GhostNet1D w=8** | Mobile | 0.8178 | 0.7919 | 0.8214 | 0.8762 | 0.8526 | 211137 | 166.0 | ghost features |
| 19 | **LSTM+Att v1 w=16** | RNN+Attn | 0.8168 | 0.8024 | 0.8179 | 0.9034 | 0.8519 | 135233 | 57.4 | BiLSTM + 4-head Self-Attn |
| 20 | **LSTM+Att v2 SCADA w=16** | RNN+Attn | 0.8168 | 0.8024 | 0.8179 | 0.9034 | 0.8519 | 135233 | 57.4 | v2 data, same as v1 |
| 21 | **BiLSTM w=8** | RNN | 0.8144 | 0.7895 | 0.8178 | 0.8873 | 0.8367 | 60481 | 31.0 | 2 Modbus cycles |
| 22 | **ShuffleNet1D w=8** | Mobile | 0.8085 | 0.7875 | 0.8108 | 0.8719 | 0.8452 | 134465 | 150.0 | group conv + shuffle |
| 23 | **BiGRU w=8** | RNN | 0.8037 | 0.7750 | 0.8079 | 0.8766 | 0.8222 | 46401 | 25.0 |  |
| 24 | **FNN (MLP)** | Basic NN | 0.7965 | 0.6633 | 0.8836 | 0.7455 | 0.8607 | 13121 | 79.0 | hidden=[128,64,32] |
| 25 | **GRU Uni w=8** | RNN | 0.7941 | 0.7633 | 0.7987 | 0.8708 | 0.8202 | 23233 | 30.0 |  |
| 26 | **Decision Tree** | Single Tree | 0.7921 | 0.6528 | 0.8855 | 0.6795 | 0.8520 | 92 | 0.4 | max_depth=8 |
| 27 | **1D Transformer w=16** | Transformer | 0.7907 | 0.7694 | 0.7928 | 0.8910 | 0.8337 | 605249 | 571.6 | d_model=128, n_heads=4, n_layers=3 |
| 28 | **Vanilla RNN w=8** | RNN | 0.7704 | 0.7345 | 0.7760 | 0.8508 | 0.7988 | 9153 | 17.0 | 1-gate baseline |
| 29 | **1D-CNN w=4** | CNN | 0.7602 | 0.7126 | 0.7696 | 0.8491 | 0.8027 | 39297 | 54.0 | 1 Modbus cycle |
| 30 | **1D-CNN row** | CNN | 0.7518 | 0.5829 | 0.8668 | 0.6828 | 0.8261 | 95681 | 665.0 | no window |
| 31 | **MobileNetV3-S w=8** | Mobile | 0.7478 | 0.7364 | 0.7483 | 0.8546 | 0.8131 | 876991 | 473.0 | NAS, h-swish, SE |
| 32 | **LSTM w=4** | RNN | 0.7237 | 0.6749 | 0.7323 | 0.8087 | 0.7494 | 30273 | 21.0 | 1 Modbus cycle |

## Per-dimension winners

| Dimension | Champion | Value | Runner-up | Gap |
|---|---|---:|---|---:|
| **Macro-F1** | **Ensemble: Stack ALL_7** | 0.8481 | Ensemble: Stack ALL_6 | 0.0026 |
| **PR-AUC** | **TCN v3c w=16 deep** | 0.9061 | LSTM+Att v2 SCADA w=16 | 0.0028 |
| **Binary-F1** | **TCN v4 v2 SCADA w=16** | 0.8241 | TCN v3b w=16 (wide) | 0.0028 |
| **Accuracy** | **Ensemble: Stack ALL_7** | 0.9018 | Ensemble: Stack ALL_6 | 0.0015 |
| **ROC-AUC** | **Ensemble: Stack ALL_9** | 0.9201 | Ensemble: Stack 3 | 0.0009 |

## Family-level analysis (averages)

| Family | N | F1m (mean) | F1m (max) | PR-AUC (mean) | PR-AUC (max) |
|---|---:|---:|---:|---:|---:|
| Stacking | 5.0 | 0.8448 | **0.8481** | 0.8458 | 0.8521 |
| Weighted | 2.0 | 0.8420 | **0.8438** | 0.8511 | 0.8521 |
| TCN++ | 2.0 | 0.8354 | **0.8367** | 0.9008 | 0.9025 |
| Boosted Tree | 1.0 | 0.8337 | **0.8337** | 0.8166 | 0.8166 |
| TCN+ | 3.0 | 0.8287 | **0.8336** | 0.8958 | 0.9061 |
| Bagged Tree | 1.0 | 0.8330 | **0.8330** | 0.8364 | 0.8364 |
| Hybrid | 1.0 | 0.8296 | **0.8296** | 0.8904 | 0.8904 |
| TCN | 1.0 | 0.8263 | **0.8263** | 0.8952 | 0.8952 |
| Mobile | 4.0 | 0.7985 | **0.8201** | 0.8694 | 0.8762 |
| RNN+Attn | 2.0 | 0.8168 | **0.8168** | 0.9034 | 0.9034 |
| RNN | 5.0 | 0.7812 | **0.8144** | 0.8588 | 0.8873 |
| Basic NN | 1.0 | 0.7965 | **0.7965** | 0.7455 | 0.7455 |
| Single Tree | 1.0 | 0.7921 | **0.7921** | 0.6795 | 0.6795 |
| Transformer | 1.0 | 0.7907 | **0.7907** | 0.8910 | 0.8910 |
| CNN | 2.0 | 0.7560 | **0.7602** | 0.7660 | 0.8491 |

## Pareto front (Macro-F1 vs Params, deep models only)

| Params | Macro-F1 | PR-AUC | Model |
|---:|---:|---:|---|
| 92 | 0.7921 | 0.6795 | Decision Tree |
| 9,153 | 0.7704 | 0.8508 | Vanilla RNN w=8 |
| 13,121 | 0.7965 | 0.7455 | FNN (MLP) |
| 23,233 | 0.7941 | 0.8708 | GRU Uni w=8 |
| 30,273 | 0.7237 | 0.8087 | LSTM w=4 |
| 39,297 | 0.7602 | 0.8491 | 1D-CNN w=4 |
| 43,329 | 0.8201 | 0.8749 | MobileNet1D w=8 |
| 46,401 | 0.8037 | 0.8766 | BiGRU w=8 |
| 60,481 | 0.8144 | 0.8873 | BiLSTM w=8 |
| 76,033 | 0.8326 | 0.8999 | TCN v3b w=16 (wide) |
| 76,033 | 0.8263 | 0.8952 | TCN v2 w=8 |
| 79,321 | 0.8367 | 0.8990 | TCN v4 v2 SCADA w=16 |
| 79,321 | 0.8340 | 0.9025 | TCN v4 +SE w=16 |
| 95,681 | 0.7518 | 0.6828 | 1D-CNN row |
| 134,465 | 0.8085 | 0.8719 | ShuffleNet1D w=8 |
| 135,233 | 0.8168 | 0.9034 | LSTM+Att v1 w=16 |
| 135,233 | 0.8168 | 0.9034 | LSTM+Att v2 SCADA w=16 |
| 150,913 | 0.8200 | 0.9061 | TCN v3c w=16 deep |
| 150,913 | 0.8336 | 0.8813 | TCN v3a w=8 (deep) |
| 170,305 | 0.8296 | 0.8904 | CNN-LSTM w=8 |
| 211,137 | 0.8178 | 0.8762 | GhostNet1D w=8 |
| 605,249 | 0.7907 | 0.8910 | 1D Transformer w=16 |
| 766,289 | 0.8330 | 0.8364 | Random Forest |
| 876,991 | 0.7478 | 0.8546 | MobileNetV3-S w=8 |