# 16-Model Cross-Comparison on 19-Dim SCADA (Macro-F1 ranked)
Yesterday's 14 models + today's 2 new MobileNet variants.

| # | Model | Family | Params | F1m | PR-AUC | Bin-F1 | Acc | ROC-AUC | Train(s) |
|---|-------|--------|--------|-----|--------|--------|-----|---------|----------|
| 1 | **TCN+SE 19-dim (champion)** | TCN | 72,921 | 0.8444 | 0.8892 | 0.8347 | 0.8450 | 0.8680 | 59.2 |
| 2 | **LightGBM** | Tree | N/A | 0.8341 | 0.8243 | 0.7289 | 0.9007 | 0.9125 | nan |
| 3 | **CNN-LSTM** | Hybrid | 51,713 | 0.8329 | 0.8997 | 0.8217 | 0.8336 | 0.8557 | 23.4 |
| 4 | **CNN 1D** | Lightweight | 30,913 | 0.8280 | 0.9031 | 0.8175 | 0.8287 | 0.8666 | 13.3 |
| 5 | **Random Forest** | Tree | N/A | 0.8273 | 0.8262 | 0.7254 | 0.8874 | 0.9101 | nan |
| 6 | **MobileNetV2** | Mobile | 256,129 | 0.8226 | 0.8964 | 0.8085 | 0.8237 | 0.8561 | 170.0 |
| 7 | **ShuffleNet 1D** | Mobile | 5,905 | 0.8189 | 0.8873 | 0.8070 | 0.8196 | 0.8475 | 22.8 |
| 8 | **MobileNet V1** | Mobile | 30,401 | 0.8167 | 0.9058 | 0.8035 | 0.8176 | 0.8628 | 21.4 |
| 9 | **BiLSTM** | Recurrent | 147,009 | 0.8053 | 0.8948 | 0.7916 | 0.8062 | 0.8367 | 42.2 |
| 10 | **SqueezeNet 1D** | Mobile | 61,873 | 0.8028 | 0.8959 | 0.7929 | 0.8033 | 0.8555 | 25.3 |
| 11 | **LSTM** | Recurrent | 57,153 | 0.8003 | 0.8922 | 0.7781 | 0.8027 | 0.8353 | 21.9 |
| 12 | **MobileNetV3-Small** | Mobile | 277,835 | 0.7984 | 0.8876 | 0.7836 | 0.7995 | 0.8523 | 59.8 |
| 13 | **GhostNet 1D** | Mobile | 48,225 | 0.7904 | 0.8808 | 0.7953 | 0.7905 | 0.8537 | 40.6 |
| 14 | **Vanilla RNN** | Recurrent | 15,873 | 0.7895 | 0.8879 | 0.7731 | 0.7908 | 0.8268 | 14.2 |
| 15 | **MobileViT 1D** | Hybrid | 75,617 | 0.7875 | 0.8992 | 0.7908 | 0.7876 | 0.8626 | 49.0 |
| 16 | **LinearSVM** | Classical | N/A | 0.6536 | 0.5366 | 0.4431 | 0.7815 | 0.7466 | nan |
| 17 | **OCC-eSNN** | SNN | N/A | 0.5817 | 0.6874 | 0.5159 | 0.5921 | 0.6105 | nan |

## Family Summary

| Family | Count | Avg F1m | Max F1m | Min F1m |
|--------|-------|---------|---------|---------|
| TCN | 1 | 0.8444 | 0.8444 | 0.8444 |
| Tree | 2 | 0.8307 | 0.8341 | 0.8273 |
| Lightweight | 1 | 0.8280 | 0.8280 | 0.8280 |
| Hybrid | 2 | 0.8102 | 0.8329 | 0.7875 |
| Mobile | 6 | 0.8083 | 0.8226 | 0.7904 |
| Recurrent | 3 | 0.7984 | 0.8053 | 0.7895 |
| Classical | 1 | 0.6536 | 0.6536 | 0.6536 |
| SNN | 1 | 0.5817 | 0.5817 | 0.5817 |

## Highlights

- **Macro-F1 Champion**: TCN+SE 19-dim (champion) (0.8444)
- **PR-AUC Champion**: MobileNet V1 (0.9058)
- **Accuracy Champion**: LightGBM (0.9007)
- **Param Efficiency Champion**: ShuffleNet 1D (F1m=0.8189, 5,905 params)
