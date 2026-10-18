/* ============================================================================
 * ch32_inference.c  --  TCN+SE ch=32 MCU Inference Implementation (23-dim SCADA)
 *                       H743 (Cortex-M7 @ 480MHz) 优化版
 * ============================================================================
 *
 * 架构 (与 PyTorch train_quantize_3ch.py ch=32 完全一致):
 *   x (23, 16)
 *     |
 *   Block 0 (d=1): conv1(23->32, k=3) -> ReLU -> conv2(32->32, k=3) -> ReLU
 *                  -> SE -> + Residual(1x1: 23->32) -> ReLU
 *     | (32, 16)
 *   Block 1 (d=2): 同上但 residual 是 Identity
 *     | (32, 16)
 *   Block 2 (d=4): 同上但 residual 是 Identity
 *     | (32, 16)
 *   GAP -> FC1(32->32) + ReLU -> FC2(32->1) -> Sigmoid
 *     |
 *   probability in [0, 1]
 *
 * 关键改动 (vs F407 19-dim 版):
 *   - IN_CHANNELS 19 -> 23
 *   - symbol 前缀 ch32_ -> v4se23_ch32_s42_
 *   - 适配 H743 Cortex-M7 FPv5-SP (单精度 FPU,使用 float 不要 double)
 *   - DWT cycle counter 测推理时间
 *   - fc1/fc2 权重量化是 FP32,不走 dequant 路径
 *
 * 关键优化 (MCU 友好):
 *   - 静态 RAM 分配,无 malloc
 *   - 启动时 dequant 一次性,运行期纯 FP32
 *   - 循环顺序优化 cache 友好 (oc 在最外层 -> 权重连续访问)
 *   - 内联辅助函数避免调用开销
 *
 * 进一步优化选项:
 *   1. 用 CMSIS-DSP 替换 conv1d: arm_conv_f32 / arm_mat_mult_f32
 *      (代码 +10 KB Flash,推理可从 ~0.13 ms 降到 ~0.04 ms @ M7 480MHz)
 *   2. 用 Helium SIMD intrinsics 手写 MAC 循环 (仅 AC6 + ARMv8.1-M)
 *   3. "Lite 模式":运行期按层 dequant (省 ~52 KB RAM,见 ch32_inference_lite.c)
 * ============================================================================
 */

#include "ch32_inference.h"
#include "model_v4_se_23dim_ch32_hybrid_s42.h"
#include <string.h>
#include <math.h>

/* H743 HAL 提供 DWT cycle counter (main.h -> stm32h7xx_hal.h) */
#include "main.h"

/* --- 1. RAM 缓冲:dequant 后的 FP32 权重 (启动时由 ch32_init 填充) --------- */

/* Block 0: conv1 (32,23,3) + conv2 (32,32,3) + residual (32,23,1) + SE(fc1:4,32 + fc2:32,4) */
float w_b0_c1[CH32_CH * CH32_IN_CHANNELS * 3];  /* 32*23*3 = 2208 */
float w_b0_c2[CH32_CH * CH32_CH       * 3];      /* 32*32*3 = 3072 */
float w_b0_rs[CH32_CH * CH32_IN_CHANNELS * 1];   /* 32*23*1 = 736 */
float w_b0_se1[CH32_SE_HIDDEN * CH32_CH];        /* 4*32 = 128 */
float w_b0_se2[CH32_CH * CH32_SE_HIDDEN];        /* 32*4 = 128 */

/* Block 1: 同上但 residual 是 identity */
float w_b1_c1[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b1_c2[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b1_se1[CH32_SE_HIDDEN * CH32_CH];        /* 128 */
float w_b1_se2[CH32_CH * CH32_SE_HIDDEN];        /* 128 */

/* Block 2 */
float w_b2_c1[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b2_c2[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b2_se1[CH32_SE_HIDDEN * CH32_CH];        /* 128 */
float w_b2_se2[CH32_CH * CH32_SE_HIDDEN];        /* 128 */

/* Classifier: fc1 (32,32) + fc2 (1,32) - 注意是 FP32,不走 dequant */
float w_fc1[CH32_FC1_OUT * CH32_CH];             /* 32*32 = 1024 */
float w_fc2[CH32_FC2_OUT * CH32_FC1_OUT];        /* 1*32  = 32 */

/* 全部 bias (FP32) */
float b_b0_c1[CH32_CH];
float b_b0_c2[CH32_CH];
float b_b0_rs[CH32_CH];
float b_b0_se1[CH32_SE_HIDDEN];
float b_b0_se2[CH32_CH];

float b_b1_c1[CH32_CH];
float b_b1_c2[CH32_CH];
float b_b1_se1[CH32_SE_HIDDEN];
float b_b1_se2[CH32_CH];

float b_b2_c1[CH32_CH];
float b_b2_c2[CH32_CH];
float b_b2_se1[CH32_SE_HIDDEN];
float b_b2_se2[CH32_CH];

float b_fc1[CH32_FC1_OUT];
float b_fc2[CH32_FC2_OUT];

/* 激活缓冲 (运行期使用) */
float buf_x[CH32_FEAT_SIZE];   /* (32, 16) 当前块输入/输出 */
float buf_y[CH32_FEAT_SIZE];   /* (32, 16) scratch */
float buf_z[CH32_FEAT_SIZE];   /* (32, 16) scratch */

/* --- 2. 工具函数 --------------------------------------------------------- */

/**
 * INT8 -> FP32 逐通道 dequant
 *   out[i] = q[i] * scale[i_per_channel]
 *   weight layout: (out_ch, in_ch, kernel)
 */
static inline void dequant_per_channel(
    const int8_t* q, const float* scale, float* out,
    int out_ch, int in_ch, int kernel)
{
    int n_per_ch = in_ch * kernel;
    for (int oc = 0; oc < out_ch; oc++) {
        float s = scale[oc];
        const int8_t* q_row = q + oc * n_per_ch;
        float* o_row = out + oc * n_per_ch;
        for (int i = 0; i < n_per_ch; i++) {
            o_row[i] = ((float)q_row[i]) * s;
        }
    }
}

/**
 * Conv1d (same padding, 带 dilation)
 *   in : (in_ch, T)
 *   out: (out_ch, T)
 *   weight: (out_ch, in_ch, kernel)
 *   dilation: d
 *   bias: (out_ch,)
 */
static inline void conv1d_same(
    const float* in, const float* w, const float* b, float* out,
    int in_ch, int out_ch, int T, int kernel, int dilation)
{
    int pad = (kernel - 1) * dilation / 2;

    /* ⭐⭐ 第十三轮 (2026-09-19): 移除 conv1d_same 内部 debug printf.
     * 旧版每层调用都打印一次 w_b0_c1[0..5], 8 层×~30B ≈ 240B / 推理,
     * 严重污染 USART1 (IDS JSON 上行信道).
     * 现场调试请用 SWO / Keil Debug (printf → ITM), 不要走 USART1. */
    for (int oc = 0; oc < out_ch; oc++) {
        const float* w_oc = w + oc * in_ch * kernel;
        float bias = b[oc];
        for (int t_out = 0; t_out < T; t_out++) {
            float sum = bias;
            int t_in_base = t_out - pad;
            for (int ic = 0; ic < in_ch; ic++) {
                const float* w_ic = w_oc + ic * kernel;
                const float* in_ic = in + ic * T;
                for (int k = 0; k < kernel; k++) {
                    int t_in = t_in_base + k * dilation;
                    if (t_in >= 0 && t_in < T) {
                        sum += w_ic[k] * in_ic[t_in];
                    }
                }
            }
            out[oc * T + t_out] = sum;
        }
    }
}

/**
 * 1x1 Conv1d (Pointwise, 等价于矩阵乘)
 *   in : (in_ch, T)
 *   out: (out_ch, T)
 *   weight: (out_ch, in_ch, 1)
 */
static inline void conv1d_1x1(
    const float* in, const float* w, const float* b, float* out,
    int in_ch, int out_ch, int T)
{
    for (int oc = 0; oc < out_ch; oc++) {
        const float* w_oc = w + oc * in_ch;
        float bias = b[oc];
        for (int t = 0; t < T; t++) {
            float sum = bias;
            for (int ic = 0; ic < in_ch; ic++) {
                sum += w_oc[ic] * in[ic * T + t];
            }
            out[oc * T + t] = sum;
        }
    }
}

/**
 * ReLU in-place
 */
static inline void relu(float* x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/**
 * SE Block (Squeeze-and-Excitation)
 */
static inline void se_block(
    float* x, const float* w1, const float* b1,
    const float* w2, const float* b2,
    int ch, int hidden, int T)
{
    float z[CH32_CH];           /* GAP 输出 (ch,) */
    float a[CH32_SE_HIDDEN];    /* FC1 输出 (hidden,) */
    float s[CH32_CH];           /* FC2 输出 / scale (ch,) */

    /* 1. GAP */
    for (int oc = 0; oc < ch; oc++) {
        const float* x_oc = x + oc * T;
        float sum = 0.0f;
        for (int t = 0; t < T; t++) sum += x_oc[t];
        z[oc] = sum / (float)T;
    }

    /* 2. FC1: a = ReLU(z @ W1.T + b1) */
    for (int h = 0; h < hidden; h++) {
        const float* w_h = w1 + h * ch;
        float sum = b1[h];
        for (int oc = 0; oc < ch; oc++) {
            sum += w_h[oc] * z[oc];
        }
        a[h] = sum > 0.0f ? sum : 0.0f;
    }

    /* 3. FC2: s = sigmoid(a @ W2.T + b2) */
    for (int oc = 0; oc < ch; oc++) {
        const float* w_oc = w2 + oc * hidden;
        float sum = b2[oc];
        for (int h = 0; h < hidden; h++) {
            sum += w_oc[h] * a[h];
        }
        s[oc] = 1.0f / (1.0f + expf(-sum));
    }

    /* 4. Scale in-place */
    for (int oc = 0; oc < ch; oc++) {
        float* x_oc = x + oc * T;
        float scale = s[oc];
        for (int t = 0; t < T; t++) {
            x_oc[t] *= scale;
        }
    }
}

/**
 * Add residual + ReLU
 */
static inline void add_residual_relu(float* y, const float* x, const float* res, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i] + res[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/**
 * GAP + FC1 + ReLU + FC2 + Sigmoid
 */
static inline float classifier(
    const float* x,
    const float* w_fc1, const float* b_fc1,
    const float* w_fc2, const float* b_fc2)
{
    float z[CH32_CH];        /* GAP (ch,) */
    float h[CH32_FC1_OUT];   /* FC1 输出 (32,) */
    float logit;

    /* GAP */
    for (int c = 0; c < CH32_CH; c++) {
        const float* x_c = x + c * CH32_WINDOW;
        float sum = 0.0f;
        for (int t = 0; t < CH32_WINDOW; t++) sum += x_c[t];
        z[c] = sum / (float)CH32_WINDOW;
    }

    /* FC1: h = ReLU(z @ W_fc1.T + b_fc1) */
    for (int o = 0; o < CH32_FC1_OUT; o++) {
        const float* w_o = w_fc1 + o * CH32_CH;
        float sum = b_fc1[o];
        for (int ic = 0; ic < CH32_CH; ic++) {
            sum += w_o[ic] * z[ic];
        }
        h[o] = sum > 0.0f ? sum : 0.0f;
        /* 注:训练时 FC1 后有 Dropout(0.1),推理时关闭,无操作 */
    }

    /* FC2: logit = h @ W_fc2.T + b_fc2 */
    {
        const float* w_o = w_fc2;
        float sum = b_fc2[0];
        for (int ic = 0; ic < CH32_FC1_OUT; ic++) {
            sum += w_o[ic] * h[ic];
        }
        logit = sum;
    }

    /* ⭐ 2026-10-13 Round 68: temperature scaling
     *   logit /= CH32_TEMPERATURE 把 INT8 per-channel drift 推到饱和区的 logits
     *   压回 [0.5, 0.9) 边界区间, 恢复阈值调优空间.
     *   PC 验证: T=2.0 thr=0.5 → F1=88.14% FPR=2.27% (vs T=1 thr=0.9 → F1=85.59% FPR=6.34%)
     *   一次 FP32 除法, 几乎无开销.
     *   关闭: 把 CH32_TEMPERATURE 设成 1.0f 即恢复原行为. */
    logit /= CH32_TEMPERATURE;

    return 1.0f / (1.0f + expf(-logit));
}

/* --- 3. 公共 API 实现 ---------------------------------------------------- */

void ch32_init(void)
{
    /* Block 0 */
    dequant_per_channel(v4se23_ch32_s42_w_tcn_0_conv1_weight,    v4se23_ch32_s42_s_tcn_0_conv1_weight,
                        w_b0_c1, CH32_CH, CH32_IN_CHANNELS, 3);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_0_conv2_weight,    v4se23_ch32_s42_s_tcn_0_conv2_weight,
                        w_b0_c2, CH32_CH, CH32_CH, 3);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_0_residual_weight, v4se23_ch32_s42_s_tcn_0_residual_weight,
                        w_b0_rs, CH32_CH, CH32_IN_CHANNELS, 1);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_0_se_fc1_weight,   v4se23_ch32_s42_s_tcn_0_se_fc1_weight,
                        w_b0_se1, CH32_SE_HIDDEN, CH32_CH, 1);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_0_se_fc2_weight,   v4se23_ch32_s42_s_tcn_0_se_fc2_weight,
                        w_b0_se2, CH32_CH, CH32_SE_HIDDEN, 1);

    /* Block 1 (无 residual conv) */
    dequant_per_channel(v4se23_ch32_s42_w_tcn_1_conv1_weight, v4se23_ch32_s42_s_tcn_1_conv1_weight,
                        w_b1_c1, CH32_CH, CH32_CH, 3);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_1_conv2_weight, v4se23_ch32_s42_s_tcn_1_conv2_weight,
                        w_b1_c2, CH32_CH, CH32_CH, 3);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_1_se_fc1_weight, v4se23_ch32_s42_s_tcn_1_se_fc1_weight,
                        w_b1_se1, CH32_SE_HIDDEN, CH32_CH, 1);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_1_se_fc2_weight, v4se23_ch32_s42_s_tcn_1_se_fc2_weight,
                        w_b1_se2, CH32_CH, CH32_SE_HIDDEN, 1);

    /* Block 2 */
    dequant_per_channel(v4se23_ch32_s42_w_tcn_2_conv1_weight, v4se23_ch32_s42_s_tcn_2_conv1_weight,
                        w_b2_c1, CH32_CH, CH32_CH, 3);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_2_conv2_weight, v4se23_ch32_s42_s_tcn_2_conv2_weight,
                        w_b2_c2, CH32_CH, CH32_CH, 3);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_2_se_fc1_weight, v4se23_ch32_s42_s_tcn_2_se_fc1_weight,
                        w_b2_se1, CH32_SE_HIDDEN, CH32_CH, 1);
    dequant_per_channel(v4se23_ch32_s42_w_tcn_2_se_fc2_weight, v4se23_ch32_s42_s_tcn_2_se_fc2_weight,
                        w_b2_se2, CH32_CH, CH32_SE_HIDDEN, 1);

    /* Classifier (FP32, 直接 memcpy) */
    memcpy(w_fc1, v4se23_ch32_s42_fc1_weight, sizeof(w_fc1));
    memcpy(w_fc2, v4se23_ch32_s42_fc2_weight, sizeof(w_fc2));

    /* Bias 直接拷贝 (FP32) */
    memcpy(b_b0_c1,  v4se23_ch32_s42_b_tcn_0_conv1_bias,  sizeof(b_b0_c1));
    memcpy(b_b0_c2,  v4se23_ch32_s42_b_tcn_0_conv2_bias,  sizeof(b_b0_c2));
    memcpy(b_b0_rs,  v4se23_ch32_s42_b_tcn_0_residual_bias, sizeof(b_b0_rs));
    memcpy(b_b0_se1, v4se23_ch32_s42_b_tcn_0_se_fc1_bias, sizeof(b_b0_se1));
    memcpy(b_b0_se2, v4se23_ch32_s42_b_tcn_0_se_fc2_bias, sizeof(b_b0_se2));

    memcpy(b_b1_c1,  v4se23_ch32_s42_b_tcn_1_conv1_bias,  sizeof(b_b1_c1));
    memcpy(b_b1_c2,  v4se23_ch32_s42_b_tcn_1_conv2_bias,  sizeof(b_b1_c2));
    memcpy(b_b1_se1, v4se23_ch32_s42_b_tcn_1_se_fc1_bias, sizeof(b_b1_se1));
    memcpy(b_b1_se2, v4se23_ch32_s42_b_tcn_1_se_fc2_bias, sizeof(b_b1_se2));

    memcpy(b_b2_c1,  v4se23_ch32_s42_b_tcn_2_conv1_bias,  sizeof(b_b2_c1));
    memcpy(b_b2_c2,  v4se23_ch32_s42_b_tcn_2_conv2_bias,  sizeof(b_b2_c2));
    memcpy(b_b2_se1, v4se23_ch32_s42_b_tcn_2_se_fc1_bias, sizeof(b_b2_se1));
    memcpy(b_b2_se2, v4se23_ch32_s42_b_tcn_2_se_fc2_bias, sizeof(b_b2_se2));

    memcpy(b_fc1, v4se23_ch32_s42_b_fc1_bias, sizeof(b_fc1));
    memcpy(b_fc2, v4se23_ch32_s42_b_fc2_bias, sizeof(b_fc2));
}

float ch32_forward(const float* x, size_t len)
{
    (void)len;

    const float* x_in = x;
    float* cur = buf_x;        /* 当前 (32, 16) 块输入/输出 */
    float* nxt = buf_y;        /* 临时 (32, 16) */
    float* tmp = buf_z;        /* scratch (32, 16) */

    /* Block 0: 输入 23 维 -> 输出 32 维,residual = 1x1 conv(23->32) */
    memcpy(buf_x, x_in, CH32_IN_CHANNELS * CH32_WINDOW * sizeof(float));

    /* residual = 1x1 conv(23->32) -> 写到 nxt */
    conv1d_1x1(buf_x, w_b0_rs, b_b0_rs, nxt,
               CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW);

    /* main path: conv1(23->32) -> ReLU -> conv2(32->32) -> ReLU -> SE -> cur */
    conv1d_same(buf_x, w_b0_c1, b_b0_c1, tmp,
                CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu(tmp, CH32_CH * CH32_WINDOW);
    conv1d_same(tmp, w_b0_c2, b_b0_c2, cur,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu(cur, CH32_CH * CH32_WINDOW);
    se_block(cur, w_b0_se1, b_b0_se1, w_b0_se2, b_b0_se2,
             CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);

    /* add residual + ReLU -> cur */
    add_residual_relu(cur, cur, nxt, CH32_CH * CH32_WINDOW);

    /* Block 1: cur (32,16) -> cur (32,16), residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same(cur, w_b1_c1, b_b1_c1, tmp,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu(tmp, CH32_FEAT_SIZE);
    conv1d_same(tmp, w_b1_c2, b_b1_c2, cur,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu(cur, CH32_FEAT_SIZE);
    se_block(cur, w_b1_se1, b_b1_se1, w_b1_se2, b_b1_se2,
             CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Block 2: cur (32,16) -> cur (32,16), residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same(cur, w_b2_c1, b_b2_c1, tmp,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu(tmp, CH32_FEAT_SIZE);
    conv1d_same(tmp, w_b2_c2, b_b2_c2, cur,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu(cur, CH32_FEAT_SIZE);
    se_block(cur, w_b2_se1, b_b2_se1, w_b2_se2, b_b2_se2,
             CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Classifier: GAP -> FC1 -> ReLU -> FC2 -> Sigmoid */
    return classifier(cur, w_fc1, b_fc1, w_fc2, b_fc2);
}

#ifdef CH32_ENABLE_ENSEMBLE
/* 5-seed ensemble 入口(占位,实际启用需要逐个 seed 跑 ch32_forward 后求平均)
 * 完整实现需要 5 套权重,这里留接口示意 */
float ch32_forward_ensemble(const float* x, size_t len)
{
    /* TODO: 包含 5 个 seed 的 weight header,逐个调用 ch32_forward 后求平均
     * 当前默认禁用,需要 CH32_ENABLE_ENSEMBLE + 5 套 .h 才编译通过 */
    (void)x; (void)len;
    return 0.0f;
}
#endif

/* --- 4. DWT cycle counter (H743 性能基准) -------------------------------- */

/**
 * 启用 DWT cycle counter (Cortex-M7 内置,无需额外硬件)
 * 必须在 ch32_benchmark 之前调用一次
 */
void ch32_dwt_init(void)
{
    /* 启用 TRC (Trace) - 一些工程默认关闭 */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    /* 解锁 DWT 访问 (H743 因为 cache 流水线需要) */
    DWT->LAR = 0xC5ACCE55;
    /* 复位 cycle counter */
    DWT->CYCCNT = 0;
    /* 启用 cycle counter */
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/**
 * 性能基准:返回 n_runs 次推理的平均 cycle 数
 */
uint32_t ch32_benchmark(int n_runs)
{
    /* 构造 1 个固定输入 (用 LCG 伪随机,确定性) */
    float xx[CH32_IN_SIZE];
    unsigned int s = 0xDEADBEEF;
    for (int i = 0; i < CH32_IN_SIZE; i++) {
        s = s * 1103515245u + 12345u;
        xx[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
    }

    /* Warm-up (填充 cache,避免首次 cold miss 干扰) */
    volatile float p = ch32_forward(xx, CH32_IN_SIZE);
    (void)p;

    uint32_t t0 = DWT->CYCCNT;
    for (int i = 0; i < n_runs; i++) {
        volatile float pp = ch32_forward(xx, CH32_IN_SIZE);
        (void)pp;
    }
    uint32_t t1 = DWT->CYCCNT;

    return (t1 - t0) / (uint32_t)n_runs;
}

/* --- 5. 内存自检 --------------------------------------------------------- */

ch32_mem_info_t ch32_get_mem_info(void)
{
    ch32_mem_info_t m;
    size_t w = 0;
    w += sizeof(w_b0_c1) + sizeof(w_b0_c2) + sizeof(w_b0_rs)
       + sizeof(w_b0_se1) + sizeof(w_b0_se2);
    w += sizeof(w_b1_c1) + sizeof(w_b1_c2) + sizeof(w_b1_se1) + sizeof(w_b1_se2);
    w += sizeof(w_b2_c1) + sizeof(w_b2_c2) + sizeof(w_b2_se1) + sizeof(w_b2_se2);
    w += sizeof(w_fc1) + sizeof(w_fc2);
    m.weights_fp32_bytes = w;

    m.input_bytes = CH32_IN_SIZE * sizeof(float);
    m.peak_act_bytes = sizeof(buf_x) + sizeof(buf_y) + sizeof(buf_z);
    m.total_ram_bytes = m.weights_fp32_bytes + m.peak_act_bytes + m.input_bytes;

    return m;
}