/* ============================================================================
 * v19_inference.h  —  TCN V19 MCU Inference API
 * ============================================================================
 *
 * Target:   STM32F4 / ESP32 / RP2040 / 任何 Cortex-M4/M7
 * Memory:   静态分配,无需 malloc
 * Header:   请把 model_v19_hybrid.h 放在同目录下
 *
 * 用法 (3 步):
 *   1. v19_init();                       // 启动时调用一次 (~1 ms)
 *   2. 填好 x[44*16] 输入特征           // 一次前向的输入窗口
 *   3. float p = v19_forward(x, 44*16); // 返回 [0,1] 异常概率
 *
 * 集成到 STM32 HAL 项目:
 *   - 在 main.c 开头 #include "v19_inference.h"
 *   - 在 main() 里 SystemClock_Config() 之后调用 v19_init()
 *   - 在检测循环里调 v19_forward(input_window, 44*16)
 *
 * 集成到 ESP-IDF:
 *   - 放在 main/ 目录下,CMakeLists.txt 里加 SRCS "v19_inference.c"
 *   - 在 app_main() 里调 v19_init()
 *
 * 编译选项建议:
 *   -O2 -ffast-math -funroll-loops -DNDEBUG
 *   (STM32: -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16)
 *
 * RAM 占用 (编译后实测):
 *   - 权重 (dequant 后的 FP32):  16,564 B  (16.18 KB)
 *   - 激活缓冲 (峰值):            2,776 B  ( 2.71 KB)
 *   - 总计:                      19,340 B  (18.89 KB)
 *
 * Flash 占用 (编译后):
 *   - 权重数据 (INT8 + scale + bias, 来自 .h):  5,141 B
 *   - 推理代码:                                2-4 KB (取决于优化)
 *   - 总计:                                    7-9 KB
 * ============================================================================
 */
#ifndef V19_INFERENCE_H
#define V19_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 模型维度常量 ──────────────────────────────────────────────────── */
#define V19_IN_CHANNELS    44    /* 输入特征维度 */
#define V19_WINDOW         16    /* 时间窗长度 */
#define V19_CH             12    /* 通道数 (压缩后) */
#define V19_SE_HIDDEN      4     /* SE 块隐藏维度 (max(12/8, 4)) */
#define V19_FC1_OUT        32    /* 分类器第一层输出 */
#define V19_FC2_OUT        1     /* 分类器第二层输出 (= logit) */

/* 派生量 */
#define V19_IN_SIZE        (V19_IN_CHANNELS * V19_WINDOW)   /* 704 */
#define V19_FEAT_SIZE      (V19_CH * V19_WINDOW)            /* 192 */

/* ── 公共 API ─────────────────────────────────────────────────────── */

/**
 * 初始化:把 INT8 权重 dequant 成 FP32 到 RAM 缓冲
 * 必须在 v19_forward 之前调用一次
 * 耗时:~1 ms @ 168 MHz
 */
void v19_init(void);

/**
 * 前向推理
 * @param x   输入特征数组,layout = (in_channels=44, window=16) row-major
 *            x[0..15] = 第 0 个特征在 16 个时间步的值
 *            x[16..31] = 第 1 个特征...
 *            ...
 *            x[704] 之前
 * @param len x 的长度(必须 >= V19_IN_SIZE)
 * @return    异常概率 ∈ [0, 1] (Sigmoid 输出)
 *            > 0.5 视为异常
 */
float v19_forward(const float* x, size_t len);

/**
 * 性能基准测试:在 main loop 之前调用,打印 2000 次推理的平均耗时
 * (串口打印到 stdout,无依赖)
 */
void v19_benchmark(int n_runs);

/**
 * 内存自检:返回权重/激活的字节数(用于打印部署属性)
 */
typedef struct {
    size_t weights_fp32_bytes;   /* dequant 后的 FP32 权重 */
    size_t input_bytes;          /* 1 个输入窗口 */
    size_t peak_act_bytes;       /* 激活峰值 */
} v19_mem_info_t;

v19_mem_info_t v19_get_mem_info(void);

#ifdef __cplusplus
}
#endif

#endif /* V19_INFERENCE_H */
