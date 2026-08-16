/* ============================================================================
 * ch32_inference.h  —  TCN+SE ch=32 MCU Inference API  (19-dim SCADA, window=16)
 * ============================================================================
 *
 * Target:   STM32F407VGT6 (Cortex-M4 @ 168MHz, FPU)
 *           ESP32 / RP2040 / Arduino 同样适用 (改 linker 即可)
 * Memory:   静态分配,无需 malloc
 * Header:   请把 model_ch32_hybrid.h 放在同目录下
 *
 * 模型架构 (与 PyTorch train_quantize_3ch.py ch=32 完全一致):
 *   x (19, 16)
 *     ↓
 *   TCN Block 0  (d=1)  Conv1d(19→32,k=3) + ReLU + Conv1d(32→32,k=3) + ReLU
 *                     + SE(GAP→FC32→4→ReLU→FC4→32→Sigmoid) + Residual(1×1) + ReLU
 *     ↓ (32, 16)
 *   TCN Block 1  (d=2)  同上但无 1×1 residual (identity)
 *     ↓ (32, 16)
 *   TCN Block 2  (d=4)  同上
 *     ↓ (32, 16)
 *   GAP → Linear(32→32) + ReLU + Dropout(0.3) → Linear(32→1) → Sigmoid
 *     ↓
 *   probability ∈ [0, 1]
 *
 * 数据 layout (与 PyTorch Conv1d 一致):
 *   shape (channels, time_steps), row-major
 *   例: input[ic*16 + t]  = 第 ic 个特征在 t 时刻的值
 *
 * 量化策略:
 *   - 权重 INT8 per-channel symmetric (已 scale/zp 在 model_ch32_hybrid.h)
 *   - 激活 FP32
 *   - BN 已 fold 到 Conv (推理时无需 BN)
 *   - init 时一次性 dequant 到 FP32 RAM,推理时纯 FP32 conv
 *
 * 用法 (3 步):
 *   1. ch32_init();                       // 启动时调用一次 (~3 ms @ 168 MHz)
 *   2. 填好 x[19*16] 输入窗口             // 一次前向的输入
 *   3. float p = ch32_forward(x, 19*16);  // 返回 [0,1] 异常概率
 *
 * 集成到 STM32 HAL 项目:
 *   - 在 main.c 开头 #include "ch32_inference.h"
 *   - 在 main() 里 SystemClock_Config() 之后调用 ch32_init()
 *   - 在检测循环里调 ch32_forward(input_window, 19*16)
 *
 * 编译选项建议:
 *   -O2 -ffast-math -funroll-loops -DNDEBUG
 *   (STM32: -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16)
 *
 * 资源占用 (ch=32 Hybrid INT8, 编译后实测):
 *   - 权重 (dequant 后 FP32):  ~79,924 B  ( 78.05 KB)
 *   - 激活缓冲 (峰值):         ~8,256 B   (  8.06 KB)
 *   - 总 RAM:                  ~88,180 B   ( 86.12 KB) ← 已超 STM32F407 SRAM(128KB) 但够用
 *   - Flash (权重 + 代码):    ~30,000 B   ( 29.30 KB)
 *
 * 注: 这是 "Standard Hybrid" 模式 (init 时一次性 dequant).
 *     想要省 RAM 请改用 Lite 模式 (运行期按层 dequant,省 ~52 KB RAM).
 * ============================================================================
 */
#ifndef CH32_INFERENCE_H
#define CH32_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 模型维度常量 ──────────────────────────────────────────────────── */
#define CH32_IN_CHANNELS   19    /* 输入特征维度 */
#define CH32_WINDOW        16    /* 时间窗长度 */
#define CH32_CH            32    /* 通道数 */
#define CH32_N_BLOCKS      3     /* TCN 块数 */
#define CH32_KERNEL        3     /* 卷积核大小 */
#define CH32_SE_HIDDEN     4     /* SE 块隐藏维度 (max(32/8, 4)) */
#define CH32_FC1_OUT       32    /* 分类器第一层输出 */
#define CH32_FC2_OUT       1     /* 分类器第二层输出 (= logit) */
#define CH32_DIL_B0        1     /* Block 0 dilation */
#define CH32_DIL_B1        2     /* Block 1 dilation */
#define CH32_DIL_B2        4     /* Block 2 dilation */

/* 派生量 */
#define CH32_IN_SIZE       (CH32_IN_CHANNELS * CH32_WINDOW)   /* 304 */
#define CH32_FEAT_SIZE     (CH32_CH * CH32_WINDOW)            /* 512 */
#define CH32_SE_REDUCE     (CH32_CH / 8)                      /* 4  (= SE_HIDDEN) */

/* ── 公共 API ─────────────────────────────────────────────────────── */

/**
 * 初始化:把 INT8 权重 dequant 成 FP32 到 RAM 缓冲
 * 必须在 ch32_forward 之前调用一次
 * 耗时:~3 ms @ 168 MHz Cortex-M4
 */
void ch32_init(void);

/**
 * 前向推理
 * @param x   输入特征数组,layout = (in_channels=19, window=16) row-major
 *            x[0..15]   = 第 0 个特征在 16 个时间步的值
 *            x[16..31]  = 第 1 个特征在 16 个时间步的值
 *            ...
 *            x[19*16=304] 之前
 * @param len x 的长度(必须 >= CH32_IN_SIZE)
 * @return    异常概率 ∈ [0, 1] (Sigmoid 输出)
 *            > 0.49 视为异常 (最佳阈值,见 best_threshold_*.json)
 */
float ch32_forward(const float* x, size_t len);

/**
 * 性能基准测试:在 main loop 之前调用,串口打印 n_runs 次推理的平均耗时
 */
void ch32_benchmark(int n_runs);

/**
 * 内存自检:返回权重/激活的字节数(用于打印部署属性)
 */
typedef struct {
    size_t weights_fp32_bytes;   /* dequant 后的 FP32 权重 */
    size_t input_bytes;          /* 1 个输入窗口 */
    size_t peak_act_bytes;       /* 激活峰值 */
    size_t total_ram_bytes;      /* 总 RAM 占用 (权重 + 激活 + 代码) */
} ch32_mem_info_t;

ch32_mem_info_t ch32_get_mem_info(void);

/**
 * 取最佳阈值(从训练时调出来的,部署时可直接用)
 */
#define CH32_BEST_THRESHOLD  0.49f

#ifdef __cplusplus
}
#endif

#endif /* CH32_INFERENCE_H */