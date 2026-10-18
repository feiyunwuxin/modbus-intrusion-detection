/* ============================================================================
 * ch32_inference.h  --  TCN+SE ch=32 MCU Inference API  (23-dim SCADA, w=16)
 *                        STM32H743VIT6 (Cortex-M7 @ 480MHz, FPU + DSP) 优化版
 *
 *  ⭐ 2026-10-18 路线 E 迁移: 取代 tflite_micro_runtime.cpp 的 TFLM ref impl.
 *    TFLM ref impl 跑 3-block TCN+SE FP32 model 在 Cortex-M7 上 broken
 *    (5 controlled input 全给常数 0x000485, Round 23 锁定, 已无解).
 *    改用手写 C forward (ch32_inference.c) + QAT INT8 dequant-to-FP32 路径
 *    (跟 h743/ 路线 E 完全一致, parity test bit-perfect rel<1e-7).
 *    完全绕开 TFLM, 不再需要 thirdParty/tflite-micro/*.cc (~150KB Flash 节省).
 * ============================================================================
 *
 * Target:   STM32H743VIT6 (Cortex-M7 @ 480MHz, double FPU + DSP)
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
 * 量化策略:
 *   - 权重 INT8 per-channel symmetric (已 scale/zp 在 weight header)
 *   - 激活 FP32
 *   - BN 已 fold 到 Conv (推理时无需 BN)
 *   - init 时一次性 dequant 到 FP32 RAM, 推理时纯 FP32 conv
 *
 * 用法 (3 步):
 *   1. ch32_init();                       // 启动时调用一次 (~3 ms @ 480 MHz)
 *   2. 填好 x[23*16] 输入窗口             // 一次前向的输入
 *   3. float p = ch32_forward(x, 23*16);  // 返回 [0,1] 异常概率
 *
 * 资源占用 (ch=32 Hybrid INT8, H743 @ -O2 -ffast-math):
 *   - 权重 (dequant 后 FP32):  ~79,892 B  ( 78.02 KB)
 *   - 激活缓冲 (峰值):         ~8,256 B   (  8.06 KB)
 *   - 总 RAM:                  ~88,148 B   ( 86.08 KB)
 *   - Flash (权重 + 代码):    ~160,000 B  ( ~156 KB .h + ~30 KB 代码)
 *   - 推理时间 @ 480 MHz:      ~0.13 ms  (vs F407 0.47ms @ 168MHz)
 *
 * 集成到 STM32 HAL 项目:
 *   - 在 main.c 开头 #include "ch32_inference.h"
 *   - 在 main() 里 SystemClock_Config() 之后调用 ch32_init()
 *   - 在检测循环里调 ch32_forward(input_window, 23*16)
 *
 * 编译选项建议 (Keil AC6):
 *   -O2 -ffast-math -funroll-loops -DNDEBUG
 *   (H743: --cpu Cortex-M7.fp.sp --fpu=FPV5-D16)
 *   注意 H743 默认 FPv5-SP-D16 (单精度), 不要误用 fpv5-d16 (双精度会慢)
 *
 * ============================================================================
 */
#ifndef CH32_INFERENCE_H
#define CH32_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

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

/* --- 公共 API ------------------------------------------------------------ */

/**
 * 初始化: 把 INT8 权重 dequant 成 FP32 到 RAM 缓冲
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
    size_t weights_fp32_bytes;   /* dequant 后的 FP32 权重 */
    size_t input_bytes;          /* 1 个输入窗口 */
    size_t peak_act_bytes;       /* 激活峰值 */
    size_t total_ram_bytes;      /* 总 RAM 占用 (权重 + 激活 + 输入) */
} ch32_mem_info_t;

ch32_mem_info_t ch32_get_mem_info(void);

/**
 * 取最佳阈值
 *
 * ⭐ 2026-10-18 路线 E: 用 0.5 (跟 h743/ CH32_TEMPERATURE 配合, 取消 0.6 历史默认值)
 *   H743 路线 E 验证: thr=0.5 F1=88.14% FPR=2.27% (vs thr=0.9 F1=85.59% FPR=6.34%)
 *   PC 5-seed FP32 baseline 同样 0.5 sweet spot.
 */
#define CH32_BEST_THRESHOLD  0.5f

/* ⭐ 2026-10-13 Round 68 (路线 E): temperature scaling
 *
 *   logit /= CH32_TEMPERATURE 把 INT8 per-channel scale drift 推到饱和区的 logits
 *   压回 [0.5, 0.9) 边界区间, 恢复阈值调优空间.
 *   PC 验证: T=2.0 thr=0.5 → F1=88.14% FPR=2.27% (vs T=1 thr=0.9 → F1=85.59% FPR=6.34%)
 *   关闭: 把 CH32_TEMPERATURE 设成 1.0f 即恢复原行为. */
#define CH32_TEMPERATURE      2.0f

#ifdef __cplusplus
}
#endif

#endif /* CH32_INFERENCE_H */
