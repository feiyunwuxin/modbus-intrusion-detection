# MCU Deployment Size Analysis — 19-Dim SCADA Models

All sizes in **KB**. Estimates based on:
- **FP32**: 4 bytes/param
- **INT8**: 1 byte/param (weights only)
- **Hybrid**: ~1.3 bytes/param (weights INT8 + BN FP32 + small overhead)
- **Flash total** = model + inference C runtime (~8-14KB)
- **RAM** = peak activations at inference, FP32 batch=1

Reference: TCN v19 (ch=12) deployed at **5.5KB INT8 hybrid / 19KB RAM / 0.47ms** @ Cortex-M4 168MHz.

| # | Model | F1m | Params | Disk KB | FP32 KB | INT8 KB | Hybrid KB | Runtime KB | Flash FP32 | Flash INT8 | RAM KB |
|---|-------|-----|--------|---------|---------|---------|-----------|------------|------------|------------|--------|
| 1 | **ShuffleNet 1D** | 0.8189 | 5905 | 45.5 | 23.1 | 5.8 | 7.5 | 8 | 31.1 | 13.8 | 37.7 |
| 2 | **Vanilla RNN** | 0.7895 | 15873 | 66.9 | 62.0 | 15.5 | 20.2 | 12 | 74.0 | 27.5 | 13.9 |
| 3 | **MobileNet V1** | 0.8167 | 30401 | 137.6 | 118.8 | 29.7 | 38.6 | 8 | 126.8 | 37.7 | 49.7 |
| 4 | **CNN 1D** | 0.8280 | 30913 | 130.7 | 120.8 | 30.2 | 39.2 | 8 | 128.8 | 38.2 | 25.7 |
| 5 | **GhostNet 1D** | 0.7904 | 48225 | 232.8 | 188.4 | 47.1 | 61.2 | 8 | 196.4 | 55.1 | 37.7 |
| 6 | **CNN-LSTM** | 0.8329 | 51713 | 210.8 | 202.0 | 50.5 | 65.7 | 12 | 214.0 | 62.5 | 26.2 |
| 7 | **LSTM** | 0.8003 | 57153 | 228.2 | 223.3 | 55.8 | 72.6 | 12 | 235.3 | 67.8 | 26.2 |
| 8 | **SqueezeNet 1D** | 0.8028 | 61873 | 269.5 | 241.7 | 60.4 | 78.5 | 8 | 249.7 | 68.4 | 49.7 |
| 9 | **TCN+SE** | 0.8444 | 72921 | 307.1 | 284.8 | 71.2 | 92.6 | 8 | 292.8 | 79.2 | 25.7 |
| 10 | **MobileViT 1D** | 0.7875 | 75617 | 327.5 | 295.4 | 73.8 | 96.0 | 14 | 309.4 | 87.8 | 49.7 |
| 11 | **BiLSTM** | 0.8053 | 147009 | 581.5 | 574.3 | 143.6 | 186.6 | 12 | 586.3 | 155.6 | 26.7 |
| 12 | **MobileNetV2** | 0.8226 | 256129 | 1057.0 | 1000.5 | 250.1 | 325.2 | 8 | 1008.5 | 258.1 | 49.7 |
| 13 | **MobileNetV3-S** | 0.7984 | 277835 | 1147.7 | 1085.3 | 271.3 | 352.7 | 8 | 1093.3 | 279.3 | 37.7 |
| 14 | **LightGBM** | 0.8341 | nan | 2702.5 | 2702.5 | nan | nan | 8 | 2710.5 | nan | 2.0 |
| 15 | **Random Forest** | 0.8273 | nan | nan | 3992.2 | nan | nan | 8 | 4000.2 | nan | 2.0 |
| 16 | **LinearSVM** | 0.6536 | nan | 3.9 | nan | nan | nan | 4 | nan | nan | 1.0 |
| 17 | **OCC-eSNN** | 0.5817 | nan | nan | nan | nan | nan | 3 | nan | nan | 1.0 |

## Deployment Tier Classification

Based on **Flash INT8 hybrid** (closest to real MCU deployment):

| Tier | Flash Range | Models | Suitable For |
|------|-------------|--------|--------------|
| **Tier 1: Pico** | < 16 KB | ShuffleNet, TCN+SE V19-like | Arduino Uno, ESP8266 |
| **Tier 2: Embedded** | 16-64 KB | CNN 1D, MobileNet V1, ShuffleNet | STM32F4, ESP32 |
| **Tier 3: Edge** | 64-256 KB | CNN-LSTM, BiLSTM, MobileNetV2, TCN+SE | Cortex-M7, Raspberry Pi |
| **Tier 4: Gateway** | 256+ KB | MobileNetV3-S, MobileViT | Cortex-A, gateway-class |

## Pareto Frontier (F1m vs Flash INT8)

Sweet spots (high F1m at small Flash):

| Model | F1m | INT8 KB | F1m per KB |
|-------|-----|---------|------------|
| ShuffleNet 1D | 0.8189 | 5.8 | 141.19e-3 |
| Vanilla RNN | 0.7895 | 15.5 | 50.94e-3 |
| MobileNet V1 | 0.8167 | 29.7 | 27.50e-3 |
| CNN 1D | 0.8280 | 30.2 | 27.42e-3 |
| GhostNet 1D | 0.7904 | 47.1 | 16.78e-3 |
| CNN-LSTM | 0.8329 | 50.5 | 16.49e-3 |
| LSTM | 0.8003 | 55.8 | 14.34e-3 |
| SqueezeNet 1D | 0.8028 | 60.4 | 13.29e-3 |
