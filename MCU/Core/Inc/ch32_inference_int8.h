/* ============================================================================
 * ch32_inference_int8.h  --  TCN+SE ch=32 MCU Inference API (全量 INT8, 23-dim, w=16)
 *                            STM32H743VIT6 (Cortex-M7 @ 480MHz) + CMSIS-NN
 *
 * ⭐ 2026-10-21 全量 INT8 路线 (与 ch32_inference.c 的 FP32 路线并存)
 *   - 编译开关: ch32_inference.h 里的 CH32_USE_INT8 (0 = FP32, 1 = INT8)
 *   - 卷积: CMSIS-NN arm_convolve_s8 (int32 精确累加 + per-channel requant)
 *   - FC  : 手写 per-channel requant (CMSIS 的 FC 只有 per-tensor, 精度不够)
 *   - 定点语义与 PC 侧 numpy 镜像 (sim_int8_kernel.py) **逐位一致**
 *
 * 对外 API 与 FP32 版完全同名同义 (ch32_init / ch32_forward / ...),
 * 由 ch32_inference.c 按 CH32_USE_INT8 分派, 调用方 (main.c / modbus_ids.c)
 * 无需改动.
 *
 * ============================================================================
 */
#ifndef CH32_INFERENCE_INT8_H
#define CH32_INFERENCE_INT8_H

#include "ch32_inference.h"   /* ch32_mem_info_t 等公共类型 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 初始化. INT8 路径不需要做任何 dequant (权重直接是 int8, 常驻 Flash),
 * 所以这里只做一次必要的自检 (常量健全性), 几乎零耗时.
 */
void ch32_int8_init(void);

/**
 * 前向推理 (INT8)
 * @param x   输入特征数组, layout = (in_channels=23, window=16) row-major
 *            (跟 FP32 版完全一样)
 * @param len x 的长度 (必须 >= CH32_IN_SIZE)
 * @return    异常概率 in [0, 1] (Sigmoid 输出)
 */
float ch32_int8_forward(const float* x, size_t len);

/**
 * 性能基准 (DWT cycle counter), 用法跟 FP32 版一样
 */
uint32_t ch32_int8_benchmark(int n_runs);

/**
 * 启用 DWT cycle counter
 */
void ch32_int8_dwt_init(void);

/**
 * 内存自检 (INT8 版: weights_fp32_bytes 字段复用为 "权重字节数 (int8)")
 */
ch32_mem_info_t ch32_int8_get_mem_info(void);

#ifdef __cplusplus
}
#endif

#endif /* CH32_INFERENCE_INT8_H */
