/* ============================================================================
 * ch32_inference.h  --  TCN+SE ch=32 MCU Inference API  (FULL FP32, 23-dim, w=16)
 *                        STM32H743VIT6 (Cortex-M7 @ 480MHz, FPU + DSP) 优化版
 *
 * ⭐⭐⭐ 2026-10-18 路线 E 迁移 + FP32 部署 (取代 INT8 Hybrid)
 *   路径:
 *     训练: PyTorch FP32 -> ONNX FP32 -> SavedModel (跟本文件 bit-perfect rel<1e-7)
 *     部署: ch32_inference.c 加载 FP32 weights (跟 ONNX / TFLite PC sim 1:1 一致)
 *   取代原因:
 *     INT8 Hybrid + T=2.0 + thr=0.5 在 H743 4-min 实测 FPR=14.43% (vs PC sim 2.27%)
 *     FP32 完全消除 INT8 dequant drift (0.5-2% per layer 累积)
 *     thr 改回训练默认 0.5, T 改回 1.0 (跟 PyTorch 一致)
 *
 * 模型架构 (与 PyTorch train_quantize_3ch.py ch=32 完全一致):
 *   x (23, 16)
 *     |
 *   TCN Block 0  (d=1)  Conv1d(23->32,k=3) + ReLU + Conv1d(32->32,k=3) + ReLU
 *                     + SE(GAP->FC32->4->ReLU->FC4->32->Sigmoid)
 *                     + Residual(1x1: 23->32) + ReLU
 *     | (32, 16)
 *   TCN Block 1  (d=2)  同上但无 1x1 residual (identity)
 *     | (32, 16)
 *   TCN Block 2  (d=4)  同上
 *     | (32, 16)
 *   GAP -> Linear(32->32) + ReLU -> Linear(32->1) -> Sigmoid
 *     |
 *   probability in [0, 1]
 *
 * 数据 layout (与 PyTorch Conv1d 一致):
 *   shape (channels, time_steps), row-major
 *   例: input[ic*16 + t]  = 第 ic 个特征在 t 时刻的值
 *
 * 资源占用 (ch=32 FP32, H743 @ -O2 -ffast-math):
 *   - 权重 (FP32, .rodata + RAM 拷贝):  ~80 KB
 *   - 激活缓冲 (峰值):                  ~8 KB
 *   - 总 RAM:                          ~88 KB
 *   - Flash (权重 + 代码):            ~190 KB
 *   - 推理时间 @ 480 MHz:               ~0.20 ms (FP32 比 INT8 慢 ~50%, 精度优先)
 *
 * 集成到 STM32 HAL 项目:
 *   - 在 main.c 开头 #include "ch32_inference.h"
 *   - 在 main() 里 SystemClock_Config() 之后调用 ch32_init()
 *   - 在检测循环里调 ch32_forward(input_window, 23*16)
 *
 * 编译选项建议 (Keil AC6):
 *   -O2 -ffast-math -funroll-loops -DNDEBUG
 *   (H743: --cpu Cortex-M7.fp.sp --fpu=FPV5-D16)
 *
 * ============================================================================
 */
#ifndef CH32_INFERENCE_H
#define CH32_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

/* --- 推理后端选择 -------------------------------------------------------- *
 *   0 = FP32 手写 C kernel   (ch32_inference.c,          2026-08 部署真值)
 *   1 = 全量 INT8 + CMSIS-NN (ch32_inference_int8.c,     2026-10-21)
 *   2 = Hybrid INT16 权重    (ch32_inference_hybrid.c,    2026-10-27)
 *       权重 int16 per-channel 存 Flash (不拷 RAM) + 激活 FP32 + 按通道 dequant.
 *       PC 侧结论 (exp_hybrid_conclusion.py): 部署区 F1 0.8671 / FPR 0.0591,
 *       跟 FP32 **逐窗 0 决策翻转** => 拿到 INT8 的资源, 保住 FP32 的精度.
 * 三个后端对外 API 完全一致 (ch32_init / ch32_forward / ...),
 * 由 ch32_inference.c 分派, 调用方 (main.c / modbus_ids.c) 无需改动.
 * Keil 里有三个 target:
 *   H743test        (FP32)   H743test_INT8 (CH32_USE_INT8=1)
 *   H743test_HYBRID (CH32_USE_HYBRID=1)
 */
#ifndef CH32_USE_INT8
#define CH32_USE_INT8 0
#endif

#ifndef CH32_USE_HYBRID
#define CH32_USE_HYBRID 0
#endif

#if CH32_USE_INT8 && CH32_USE_HYBRID
#error "CH32_USE_INT8 与 CH32_USE_HYBRID 不能同时为 1"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- 模型维度常量 -------------------------------------------------------- */
#define CH32_IN_CHANNELS   23    /* 输入特征维度 */
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
#define CH32_IN_SIZE       (CH32_IN_CHANNELS * CH32_WINDOW)   /* 368 */
#define CH32_FEAT_SIZE     (CH32_CH * CH32_WINDOW)            /* 512 */
#define CH32_SE_REDUCE     (CH32_CH / 8)                      /* 4 (= SE_HIDDEN) */

/* 栈上分配的 buffer max (用于 SE / classifier 临时数组) */
#define CH32_CH_MAX         32
#define CH32_SE_HIDDEN_MAX  4
#define CH32_FC1_OUT_MAX    32

/* --- 公共 API ------------------------------------------------------------ */

/**
 * 初始化: 把 const FP32 权重 memcpy 到 RAM 缓冲
 * 必须在 ch32_forward 之前调用一次
 * 耗时: ~3 ms @ 480 MHz Cortex-M7
 *
 * ⭐ 2026-10-18 路线 E: 返回 void (跟 h743/ 一致).
 *   原 TFLM 版本返回 uint8_t (1=OK, 0=FAIL), 因为 TFLM AllocateTensors
 *   可能 arena OOM 失败. 手写 C 路径无此风险, 不需返回值.
 */
void ch32_init(void);

/**
 * 前向推理
 * @param x   输入特征数组, layout = (in_channels=23, window=16) row-major
 *            x[0..15]    = 第 0 个特征在 16 个时间步的值
 *            x[16..31]   = 第 1 个特征在 16 个时间步的值
 *            ...
 * @param len x 的长度 (必须 >= CH32_IN_SIZE)
 * @return    异常概率 in [0, 1] (Sigmoid 输出)
 *            > CH32_BEST_THRESHOLD 视为异常
 */
float ch32_forward(const float* x, size_t len);

/**
 * 性能基准测试: 用 DWT cycle counter 测 n_runs 次推理平均耗时
 * 需要先调用 ch32_dwt_init() 启用 cycle counter
 * @param n_runs  推理次数
 * @return        平均每推理 cycles
 */
uint32_t ch32_benchmark(int n_runs);

/**
 * 启用 DWT cycle counter (主程序调一次)
 */
void ch32_dwt_init(void);

/**
 * 内存自检: 返回权重/激活的字节数
 */
typedef struct {
    size_t weights_fp32_bytes;   /* FP32 权重字节数 */
    size_t input_bytes;          /* 1 个输入窗口 */
    size_t peak_act_bytes;       /* 激活峰值 */
    size_t total_ram_bytes;      /* 总 RAM 占用 (权重 + 激活 + 输入) */
} ch32_mem_info_t;

ch32_mem_info_t ch32_get_mem_info(void);

/**
 * 取最佳阈值
 *
 * ⭐⭐⭐ 2026-10-18 FP32 部署: 改回训练默认 0.5
 *   INT8 Hybrid 时代因为 dequant drift 推到边界, 改 thr=0.7 才稳住 FPR
 *   FP32 部署跟 PyTorch / ONNX / TFLite PC sim bit-perfect, 不需要漂移补偿
 *   训练最佳 thr=0.5 直接可用, 期望 F1 接近 PC sim 0.86, FPR 接近 2.27%
 *   4-min 实测如仍有 FPR gap, 再做 H743-specific threshold sweep
 */
#define CH32_BEST_THRESHOLD  0.5f

/* ⭐⭐⭐ 2026-10-18 FP32 部署: temperature scaling 设成 1.0 (no scaling)
 *   训练时 PyTorch 没做 T scaling, 部署也保持 T=1.0, 让 logit 跟 ONNX/PyTorch 1:1 一致.
 *   INT8 Hybrid 时代的 T=2.0 是 dequant drift 经验值, FP32 没这个 drift.
 *   关闭: 把 CH32_TEMPERATURE 设成 2.0f 即恢复 INT8 Hybrid 行为. */
#define CH32_TEMPERATURE      1.0f

#ifdef __cplusplus
}
#endif

#endif /* CH32_INFERENCE_H */
