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
 * 用法:
 *   1. include 本头
 *   2. 定义 extern 数组 (在 model_*.h 中已 static const)
 *   3. 调用 tcn_v4_se_23dim_ch32_inference(input, output)
 *
 * 23-dim 特征 (按本工作 6 维 sweep 冠军, drops length/setpoint/crc_mean_w/cmd_count_w):
 *   索引 -> 名称
 *   0: address, 1: function, 4: gain, 5: reset rate, 6: deadband,
 *   7: cycle time, 8: rate, 9: system mode, 10: control scheme, 11: pump,
 *   12: solenoid, 13: pressure measurement, 14: crc rate, 15: time_diff,
 *   16: time_since_last_same_addr_func, 17: is_unusual_fc, 18: is_response,
 *   19: press_mean_w, 21: crc_max_w, 23: resp_count_w,
 *   24: cmd_resp_balance_w, 25: length_nunique_w, 26: unusual_count_w
 */

#ifndef V4_SE_23DIM_CH32_INFERENCE_H
#define V4_SE_23DIM_CH32_INFERENCE_H

#include <stdint.h>
#include <math.h>

#define V4_SE_23DIM_CH32_N_FEATURES  23
#define V4_SE_23DIM_CH32_WINDOW      16
#define V4_SE_23DIM_CH32_CHANNELS    32
#define V4_SE_23DIM_CH32_SE_RED      4    // channels // reduction = 32/8
#define V4_SE_23DIM_CH32_N_BLOCKS    3
#define V4_SE_23DIM_CH32_KERNEL      3
#define V4_SE_23DIM_CH32_FC_HIDDEN   32

// 阈值 (val F1m sweep 找出的最佳分类阈值)
#define V4_SE_23DIM_CH32_THRESHOLD   0.5f

/* 前向推理: input[23,16] (FP32) -> output[1] (FP32, probability) */
#ifdef __cplusplus
extern "C" {
#endif

void tcn_v4_se_23dim_ch32_inference(const float input[23][16], float *output);

/* 批量推理: N 个样本, output[N] */
void tcn_v4_se_23dim_ch32_inference_batch(
    const float *input, int N, float *output, int input_stride_floats);

/* 类别预测 (直接返回 0/1) */
static inline int tcn_v4_se_23dim_ch32_predict(const float input[23][16]) {
    float prob;
    tcn_v4_se_23dim_ch32_inference(input, &prob);
    return prob >= V4_SE_23DIM_CH32_THRESHOLD ? 1 : 0;
}

#ifdef __cplusplus
}
#endif

#endif  // V4_SE_23DIM_CH32_INFERENCE_H
