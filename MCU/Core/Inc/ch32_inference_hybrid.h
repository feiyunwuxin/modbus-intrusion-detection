/* ============================================================================
 * ch32_inference_hybrid.h  --  TCN+SE ch=32 MCU Inference API (Hybrid INT16 权重 / FP32 激活)
 *                              STM32H743VIT6 (Cortex-M7 @ 480MHz)
 *
 * ⭐⭐⭐ 2026-10-27 Hybrid 工程 (第三个后端, 与 FP32 / 全量 INT8 并存)
 *
 *   方案: 权重 **int16 per-output-channel 对称** 存 Flash (不拷 RAM),
 *         激活全 FP32, 运行期按输出通道 dequant:  w = (float)q * scale[oc]
 *
 *   为什么是 INT16 而不是 INT8
 *   --------------------------
 *   PC 侧 2x2 归因 (exp_hybrid_conclusion.py) 证明:
 *     - 只量化激活 -> FPR 完全不变 (0.0591)
 *     - 只量化权重 -> FPR 涨到 0.0689 (Hybrid-INT8) / 0.0698 (全量 INT8)
 *     根因是 int8 权重量化误差 **系统性地把 logit 上推** (Δlogit 均值 +0.19),
 *     paired bootstrap 的 ΔFPR 95%CI 不含 0, McNemar p<1e-6 -> 统计显著.
 *   INT16 把量化误差缩小 256 倍 (相对误差 1.5e-5):
 *     部署区 F1 0.8671 / FPR 0.0591 / P 0.9361 / R 0.8075
 *     **跟 FP32 逐窗 0 决策翻转**, 每个阈值都完全一致.
 *   => 拿到全量 INT8 的资源, 保住 FP32 的精度.
 *
 *   资源 (ch=32 / 23-dim / seed42)
 *   -----------------------------
 *     权重 .rodata : 44.1 KB  (FP32 头 80.1 KB, Hybrid-INT8 头 25.5 KB)
 *     RAM 权重拷贝 : **0**     (FP32 路线启动时 memcpy 80.5 KB 到 RAM)
 *     激活缓冲     : ~6 KB    (跟 FP32 路线一样)
 *
 *   对外 API 与 FP32 / INT8 版完全同名同义, 由 ch32_inference.c 按
 *   CH32_USE_HYBRID 分派, 调用方 (main.c / modbus_ids.c) 无需改动.
 *   Keil target: H743test_HYBRID (CH32_USE_HYBRID=1).
 *
 * ============================================================================
 */
#ifndef CH32_INFERENCE_HYBRID_H
#define CH32_INFERENCE_HYBRID_H

#include "ch32_inference.h"   /* ch32_mem_info_t 等公共类型 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 初始化. Hybrid 路径 **不需要任何 RAM 权重拷贝** (权重常驻 Flash,
 * 运行期按通道 dequant), 所以这里只做一次常量健全性自检, 几乎零耗时.
 */
void ch32_hybrid_init(void);

/**
 * 前向推理 (Hybrid INT16 权重 / FP32 激活)
 * @param x   输入特征数组, layout = (in_channels=23, window=16) row-major
 *            (跟 FP32 版完全一样)
 * @param len x 的长度 (必须 >= CH32_IN_SIZE)
 * @return    异常概率 in [0, 1] (Sigmoid 输出)
 */
float ch32_hybrid_forward(const float* x, size_t len);

/**
 * 性能基准 (DWT cycle counter), 用法跟 FP32 版一样
 */
uint32_t ch32_hybrid_benchmark(int n_runs);

/**
 * 启用 DWT cycle counter
 */
void ch32_hybrid_dwt_init(void);

/**
 * 内存自检 (Hybrid 版: weights_fp32_bytes 字段复用为 "权重 .rodata 字节数")
 */
ch32_mem_info_t ch32_hybrid_get_mem_info(void);

#ifdef __cplusplus
}
#endif

#endif /* CH32_INFERENCE_HYBRID_H */
