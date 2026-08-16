/* ============================================================================
 * v4_se_23dim_ch32_inference.h - TCN+SE ch=32 (23-dim) MCU Inference API
 * ============================================================================
 *
 * Target:   STM32F407VGT6 (Cortex-M4 @ 168MHz, FPU)
 *           ESP32 / RP2040 / Arduino (改 linker 即可)
 * Memory:   静态分配,无需 malloc
 * Header:   请把 model_v4_se_23dim_ch32_hybrid_s{seed}.h 放在同目录
 *
 * 模型架构 (与 PyTorch retrain_tcn_23dim_b64_ch32_do01_savept.py 完全一致):
 *   x (23, 16)
 *     ↓
 *   TCN Block 0 (d=1) Conv1d(23->32,k=3) + ReLU + Conv1d(32->32,k=3) + ReLU
 *                  + SE(GAP->FC32->4->ReLU->FC4->32->Sigmoid) + Residual(1x1) + ReLU
 *     ↓ (32, 16)
 *   TCN Block 1 (d=2) 同上但无 1x1 residual (identity)
 *     ↓ (32, 16)
 *   TCN Block 2 (d=4) 同上
 *     ↓ (32, 16)
 *   GAP -> Linear(32->32) + ReLU + Dropout(0.3) -> Linear(32->1) -> Sigmoid
 *     ↓
 *   probability in [0, 1]
 *
 * 数据 layout (与 PyTorch Conv1d 一致):
 *   shape (channels, time_steps), row-major
 *   例: input[ic*16 + t] = 第 ic 个特征在 t 时刻的值
 *
 * 量化策略 (Hybrid INT8):
 *   - 权重 INT8 per-channel symmetric (scale/zp 在 model_*.h)
 *   - 激活 FP32 (运行期 dequantize INT8->FP32, 推理速度近似 FP32)
 *   - BN 已折叠进 Conv bias
 *
 * 用法 (3 步):
 *   1. v4_se_23dim_ch32_init();                          // 启动时调用一次 (~3 ms @ 168 MHz)
 *   2. 填好 x[23*16] 输入窗口
 *   3. float p = v4_se_23dim_ch32_forward(x, 23*16);     // 单 seed 返回 [0,1] 异常概率
 *   或 float p = v4_se_23dim_ch32_ensemble_forward(x);   // 5-seed 平均 (更准,稍慢)
 *
 * 集成到 STM32 HAL 项目:
 *   - 在 main.c 开头 #include "v4_se_23dim_ch32_inference.h"
 *   - 在 main() 里 SystemClock_Config() 之后调用 v4_se_23dim_ch32_init()
 *   - 在检测循环里调 v4_se_23dim_ch32_forward(input_window)
 *   - (或调 v4_se_23dim_ch32_ensemble_forward() 拿 5-seed 平均概率)
 *
 * 编译选项建议:
 *   -O2 -ffast-math -funroll-loops -DNDEBUG
 *   (STM32: -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16)
 *
 * 资源占用 (ch=32 Hybrid INT8, 23-dim, Standard 模式,编译后实测):
 *   - 权重 (dequant 后 FP32):  ~80,512 B  ( 78.6 KB)
 *   - 激活缓冲 (峰值):         ~6,144 B   (  6.0 KB)
 *   - 输入窗口:                ~1,472 B   (  1.4 KB)
 *   - 总 RAM:                  ~88,128 B  ( 86.1 KB) ← STM32F407 SRAM(128KB) 够用
 *   - Flash (权重 + 代码):    ~30,000 B   ( 29.3 KB)
 *
 * 5-seed ensemble 模式:
 *   - 共享同一块 78.6KB FP32 权重缓冲,每次循环 dequant 一个 seed (约 0.6 ms) →
 *     forward (约 0.9 ms),依次跑 5 个 seed,最终平均概率
 *   - 总耗时约 5 × (0.6 + 0.9) ≈ 7.5 ms / 样本
 *   - RAM 与单 seed 模式相同
 * ============================================================================
 */
#ifndef V4_SE_23DIM_CH32_INFERENCE_H
#define V4_SE_23DIM_CH32_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 模型维度常量 ──────────────────────────────────────────────────── */
#define V4_SE_23DIM_CH32_IN_CHANNELS  23    /* 输入特征维度 */
#define V4_SE_23DIM_CH32_WINDOW       16    /* 时间窗长度 */
#define V4_SE_23DIM_CH32_CHANNELS     32    /* 通道数 */
#define V4_SE_23DIM_CH32_SE_HIDDEN    4     /* SE 块隐藏维度 (max(32/8, 4)) */
#define V4_SE_23DIM_CH32_SE_RED       4     /* 同上,兼容旧宏名 */
#define V4_SE_23DIM_CH32_N_BLOCKS     3     /* TCN 块数 */
#define V4_SE_23DIM_CH32_KERNEL       3     /* 卷积核大小 */
#define V4_SE_23DIM_CH32_FC1_OUT      32    /* 分类器第一层输出 */
#define V4_SE_23DIM_CH32_FC2_OUT      1     /* 分类器第二层输出 (= logit) */
#define V4_SE_23DIM_CH32_DIL_B0       1     /* Block 0 dilation */
#define V4_SE_23DIM_CH32_DIL_B1       2     /* Block 1 dilation */
#define V4_SE_23DIM_CH32_DIL_B2       4     /* Block 2 dilation */

/* 派生量 */
#define V4_SE_23DIM_CH32_IN_SIZE      (V4_SE_23DIM_CH32_IN_CHANNELS * V4_SE_23DIM_CH32_WINDOW)  /* 368 */
#define V4_SE_23DIM_CH32_FEAT_SIZE    (V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_WINDOW)     /* 512 */
#define V4_SE_23DIM_CH32_N_SEEDS      5     /* 5-seed self-ensemble */

/* 阈值 (来自 best_threshold_tcn_v4_se_ensemble5.json, 2026-07-26 训练) */
#define V4_SE_23DIM_CH32_THRESHOLD    0.33f
#define V4_SE_23DIM_CH32_BEST_THRESHOLD  0.33f

/* ── 公共 API ─────────────────────────────────────────────────────── */

/**
 * 初始化:把 INT8 权重 dequant 成 FP32 到 RAM 缓冲 (默认 seed=123,最佳单 seed)
 * 必须在 forward 之前调用一次
 * 耗时:~3 ms @ 168 MHz Cortex-M4
 */
void v4_se_23dim_ch32_init(void);

/**
 * 用指定 seed 重新初始化 (覆盖默认 seed=123)
 * @param seed 必须是 42/123/456/789/1024 之一,且对应 .h 已 include
 */
void v4_se_23dim_ch32_init_seed(int seed);

/**
 * 单 seed 前向推理
 * @param x   输入特征数组,layout = (in_channels=23, window=16) row-major
 * @param len x 的长度(必须 >= V4_SE_23DIM_CH32_IN_SIZE)
 * @return    异常概率 ∈ [0, 1] (Sigmoid 输出)
 */
float v4_se_23dim_ch32_forward(const float* x, size_t len);

/**
 * 5-seed ensemble 前向推理 (循环 dequant + forward + 平均概率)
 * @param x   输入特征数组,layout = (in_channels=23, window=16) row-major
 * @return    异常概率 ∈ [0, 1] (5 个 seed 的平均)
 */
float v4_se_23dim_ch32_ensemble_forward(const float* x);

/**
 * 批量推理 (单 seed)
 * @param input     输入数组,shape=(N, in_channels, window) row-major
 * @param N         样本数
 * @param output    输出数组,shape=(N,)
 * @param input_stride_floats  一个样本的 float 数 (= in_channels*window)
 */
void v4_se_23dim_ch32_batch(
    const float* input, int N, float* output, int input_stride_floats);

/**
 * 5-seed ensemble 批量推理
 */
void v4_se_23dim_ch32_ensemble_batch(
    const float* input, int N, float* output, int input_stride_floats);

/**
 * 性能基准测试:在 main loop 之前调用,串口打印 n_runs 次推理的平均耗时
 */
void v4_se_23dim_ch32_benchmark(int n_runs);

/**
 * 内存自检:返回权重/激活的字节数(用于打印部署属性)
 */
typedef struct {
    size_t weights_fp32_bytes;   /* dequant 后的 FP32 权重 */
    size_t input_bytes;          /* 1 个输入窗口 */
    size_t peak_act_bytes;       /* 激活峰值 */
    size_t total_ram_bytes;      /* 总 RAM 占用 (权重 + 激活 + 代码) */
    int    current_seed;         /* 当前 dequant 的 seed */
} v4_se_23dim_ch32_mem_info_t;

v4_se_23dim_ch32_mem_info_t v4_se_23dim_ch32_get_mem_info(void);

/* ── 内联便捷函数 ─────────────────────────────────────────────────── */

/**
 * 类别预测 (单 seed,直接返回 0/1)
 */
static inline int v4_se_23dim_ch32_predict(const float* x) {
    return v4_se_23dim_ch32_forward(x, V4_SE_23DIM_CH32_IN_SIZE) >= V4_SE_23DIM_CH32_THRESHOLD ? 1 : 0;
}

/**
 * 类别预测 (5-seed ensemble,直接返回 0/1)
 */
static inline int v4_se_23dim_ch32_ensemble_predict(const float* x) {
    return v4_se_23dim_ch32_ensemble_forward(x) >= V4_SE_23DIM_CH32_THRESHOLD ? 1 : 0;
}

#ifdef __cplusplus
}
#endif

#endif /* V4_SE_23DIM_CH32_INFERENCE_H */