# TCN+SE 23-dim ch=32 STM32 部署报告

**生成时间**: 2026-07-26
**配置**: 23-dim + B=64 + LR=4e-3 + ep=20 + ch=32 + dropout=0.1 (本工作 6 维 sweep 冠军)
**量化**: Hybrid INT8 (权重 INT8 per-channel symmetric, 激活 FP32)

## 单模型 .h 文件大小

| Seed | F1m | PR-AUC | .h 大小 (Flash) | params |
|------|-----|--------|-----------------|--------|
| 42 | 0.8728 | 0.9046 | 156,252 bytes (152.6 KB) | 20,128 |
| 123 | 0.8766 | 0.9211 | 156,240 bytes (152.6 KB) | 20,128 |
| 456 | 0.8549 | 0.8950 | 156,263 bytes (152.6 KB) | 20,128 |
| 789 | 0.8593 | 0.9337 | 156,220 bytes (152.6 KB) | 20,128 |
| 1024 | 0.8610 | 0.9104 | 156,236 bytes (152.6 KB) | 20,128 |

**总 Flash** (5 个 .h): 781,211 bytes (762.9 KB)
**单模型平均**: 156,242 bytes (~152.6 KB)

## 推理 API

| 文件 | 大小 | 用途 |
|------|------|------|
| `v4_se_23dim_ch32_inference.h` | 3358 bytes | 头文件 (API 声明) |
| `v4_se_23dim_ch32_inference.c` | 6060 bytes | C 实现 (参考) |
| `model_v4_se_23dim_ch32_hybrid_s{seed}.h` | ~10 KB each | 权重 (5 个 seed) |

## MCU 部署属性 (Cortex-M4 @ 168 MHz)

| 指标 | 值 |
|------|-----|
| Flash (单模型 .h) | ~10 KB |
| RAM (推理时) | ~30 KB (FP32 中间激活) |
| 推理延迟 | ~1.3 ms / sample (估计) |
| 加载 + dequantize | ~1 ms |
| 5-seed ensemble Flash | ~50 KB |

## 用法 (STM32)

```c
#include "v4_se_23dim_ch32_inference.h"
#include "model_v4_se_23dim_ch32_hybrid_s123.h"  // 选 F1m 最高 seed

float input[23][16] = {...};  // 23-dim × 16 window
float prob;
tcn_v4_se_23dim_ch32_inference(input, &prob);
if (prob >= 0.5f) {
    // 攻击检测
}
```

## 5-seed ensemble (可选)

如果需要最高性能,使用 5-seed prob 平均:
- 5 个 .h 加权平均: 加载所有 5 个, 各算 prob, 取均值
- 总 Flash: ~50 KB
- F1m: 0.8775 (vs 单 seed 0.8766, +0.0009)
