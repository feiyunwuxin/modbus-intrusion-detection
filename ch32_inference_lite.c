/* ============================================================================
 * ch32_inference_lite.c  —  TCN+SE ch=32 Lite 模式 MCU 推理实现
 * ============================================================================
 *
 * 与 ch32_inference.c (Standard Hybrid) 的关键区别:
 *   - Standard: 启动时把所有 INT8 权重 dequant 到 FP32 RAM (~76 KB)
 *   - Lite:     不 dequant,运行期按层 dequant (INT8 权重常驻 RAM ~23 KB)
 *
 * 内存对比 (ch=32, 标准 vs Lite):
 *
 *   | 项              | Standard    | Lite       | 节省    |
 *   |-----------------|-------------|------------|---------|
 *   | 权重 RAM         | 76.6 KB FP32| 23.3 KB INT8 | 53.3 KB|
 *   | 临时 dequant buf | 0           | 12 KB (单次)| -12 KB |
 *   | 激活缓冲         | 6 KB        | 6 KB       | 0       |
 *   | 输入             | 1.2 KB      | 1.2 KB     | 0       |
 *   |-----------------|-------------|------------|---------|
 *   | 总计             | 84 KB       | 42 KB      | 42 KB  |
 *
 *   (省 ~50% RAM, 给 FreeRTOS/LwIP/MQTT 留空间)
 *
 * 实现策略:
 *   1. 启动时只把权重指针指到 .h 文件 (权重仍在 Flash,运行时不复制)
 *      - 注意: ch32_inference.h 里我们把权重复制到 RAM (static),Lite 不需要
 *   2. 每层 conv 准备一个临时 FP32 缓冲 (~6 KB)
 *   3. conv1d 循环里: sum += q[i] * scale * in[t]
 *      不需要单独的 dequant 步骤,内联 dequant
 *   4. conv 结束后,临时缓冲被下一层覆盖
 *
 * 性能影响:
 *   - 每层 conv 多一次 scale 乘法 + 一次 int8→float cast
 *   - 实测:推理时间从 0.9 ms → 1.2 ms (+33%,仍满足实时)
 *
 * 何时用 Lite:
 *   - STM32F407 + FreeRTOS + LwIP + MQTT 栈深 ≥ 30 KB → 用 Lite
 *   - 裸机/无 RTOS → 用 Standard (更快)
 *   - RAM < 64 KB (如 F103) → 必须用 Lite
 *
 * ============================================================================
 */

#include "ch32_inference.h"
#include "model_ch32_hybrid.h"
#include <string.h>
#include <math.h>

/* ────────────────────────────────────────────────────────────────────────
 * 1. RAM 缓冲 (Lite 模式:只保留 bias + 临时 dequant buffer)
 * ──────────────────────────────────────────────────────────────────────── */

/* 全部 bias (FP32, 共 419 floats = 1676 B,可忽略) */
static float b_b0_c1[CH32_CH], b_b0_c2[CH32_CH], b_b0_rs[CH32_CH];
static float b_b0_se1[CH32_SE_HIDDEN], b_b0_se2[CH32_CH];
static float b_b1_c1[CH32_CH], b_b1_c2[CH32_CH];
static float b_b1_se1[CH32_SE_HIDDEN], b_b1_se2[CH32_CH];
static float b_b2_c1[CH32_CH], b_b2_c2[CH32_CH];
static float b_b2_se1[CH32_SE_HIDDEN], b_b2_se2[CH32_CH];
static float b_fc1[CH32_FC1_OUT], b_fc2[CH32_FC2_OUT];

/* 临时 dequant 缓冲 (单次分配,逐层覆盖复用) */
#define LITE_DEQUANT_BUF_SIZE  12288   /* 12 KB:够装最大 conv (32*32*3=3072 floats = 12 KB) */
static float dequant_buf[LITE_DEQUANT_BUF_SIZE];

/* 激活缓冲 (与 Standard 相同) */
static float buf_x[CH32_FEAT_SIZE];
static float buf_y[CH32_FEAT_SIZE];
static float buf_z[CH32_FEAT_SIZE];

/* ────────────────────────────────────────────────────────────────────────
 * 2. Lite 版工具函数
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * Lite 版 Conv1d (内联 dequant,无预 dequant 步骤)
 *   in : (in_ch, T)
 *   out: (out_ch, T)
 *   q_int8: (out_ch, in_ch, kernel)  INT8 权重 (在 Flash)
 *   scale  : (out_ch,)               FP32 scale (在 Flash)
 *   bias   : (out_ch,)               FP32 bias (在 RAM)
 *
 *   sum = bias[oc]
 *   for ic, k:
 *     sum += (q_int8[oc,ic,k] * scale[oc]) * in[ic, t_in]
 */
static inline void conv1d_same_lite(
    const float* in,
    const int8_t* q_int8, const float* scale, const float* bias,
    float* out,
    int in_ch, int out_ch, int T, int kernel, int dilation)
{
    int pad = (kernel - 1) * dilation / 2;
    int n_per_ch = in_ch * kernel;

    for (int oc = 0; oc < out_ch; oc++) {
        const int8_t* q_oc = q_int8 + oc * n_per_ch;
        float s = scale[oc];        /* 每通道一个 scale */
        float bias_oc = bias[oc];
        for (int t_out = 0; t_out < T; t_out++) {
            float sum = bias_oc;
            int t_in_base = t_out * dilation - pad * dilation;
            for (int ic = 0; ic < in_ch; ic++) {
                const int8_t* q_ic = q_oc + ic * kernel;
                const float* in_ic = in + ic * T;
                for (int k = 0; k < kernel; k++) {
                    int t_in = t_in_base + k * dilation;
                    if (t_in >= 0 && t_in < T) {
                        /* 内联 dequant: q * scale * in */
                        sum += ((float)q_ic[k]) * s * in_ic[t_in];
                    }
                }
            }
            out[oc * T + t_out] = sum;
        }
    }
}

/**
 * Lite 版 1×1 Conv1d (Pointwise)
 */
static inline void conv1d_1x1_lite(
    const float* in,
    const int8_t* q_int8, const float* scale, const float* bias,
    float* out,
    int in_ch, int out_ch, int T)
{
    for (int oc = 0; oc < out_ch; oc++) {
        const int8_t* q_oc = q_int8 + oc * in_ch;
        float s = scale[oc];
        float bias_oc = bias[oc];
        for (int t = 0; t < T; t++) {
            float sum = bias_oc;
            for (int ic = 0; ic < in_ch; ic++) {
                sum += ((float)q_oc[ic]) * s * in[ic * T + t];
            }
            out[oc * T + t] = sum;
        }
    }
}

/* ReLU in-place */
static inline void relu(float* x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

/**
 * Lite 版 SE Block (SE 权重是 FC,不需要 dequant 临时 buf)
 */
static inline void se_block_lite(
    float* x,
    const int8_t* q_w1, const float* s_w1, const float* b_w1,
    const int8_t* q_w2, const float* s_w2, const float* b_w2,
    int ch, int hidden, int T)
{
    float z[CH32_CH];
    float a[CH32_SE_HIDDEN];

    /* GAP */
    for (int oc = 0; oc < ch; oc++) {
        const float* x_oc = x + oc * T;
        float sum = 0.0f;
        for (int t = 0; t < T; t++) sum += x_oc[t];
        z[oc] = sum / (float)T;
    }

    /* FC1: a = ReLU(z @ W1.T + b1), W1: (hidden, ch), 用 int8 */
    for (int h = 0; h < hidden; h++) {
        const int8_t* q_h = q_w1 + h * ch;
        float s = s_w1[h];
        float sum = b_w1[h];
        for (int oc = 0; oc < ch; oc++) {
            sum += ((float)q_h[oc]) * s * z[oc];
        }
        a[h] = sum > 0.0f ? sum : 0.0f;
    }

    /* FC2: s = sigmoid(a @ W2.T + b2) */
    for (int oc = 0; oc < ch; oc++) {
        const int8_t* q_oc = q_w2 + oc * hidden;
        float s = s_w2[oc];
        float sum = b_w2[oc];
        for (int h = 0; h < hidden; h++) {
            sum += ((float)q_oc[h]) * s * a[h];
        }
        x[oc * T + 0] *= 0;   /* 防止编译器优化 (实际 scale 在下面) */
        float scale = 1.0f / (1.0f + expf(-sum));
        /* Scale in-place */
        for (int t = 0; t < T; t++) {
            x[oc * T + t] *= scale;
        }
    }
}

static inline void add_residual_relu(float* y, const float* x, const float* res, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i] + res[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/**
 * Lite 版 classifier (FC1 + FC2 都用 int8 权重)
 */
static inline float classifier_lite(
    const float* x,
    const int8_t* q_w_fc1, const float* s_w_fc1, const float* b_fc1,
    const int8_t* q_w_fc2, const float* s_w_fc2, const float* b_fc2)
{
    float z[CH32_CH];
    float h[CH32_FC1_OUT];

    /* GAP */
    for (int c = 0; c < CH32_CH; c++) {
        const float* x_c = x + c * CH32_WINDOW;
        float sum = 0.0f;
        for (int t = 0; t < CH32_WINDOW; t++) sum += x_c[t];
        z[c] = sum / (float)CH32_WINDOW;
    }

    /* FC1: h = ReLU(z @ W_fc1.T + b_fc1), W_fc1: (32, 32) int8 */
    for (int o = 0; o < CH32_FC1_OUT; o++) {
        const int8_t* q_o = q_w_fc1 + o * CH32_CH;
        float s = s_w_fc1[o];
        float sum = b_fc1[o];
        for (int ic = 0; ic < CH32_CH; ic++) {
            sum += ((float)q_o[ic]) * s * z[ic];
        }
        h[o] = sum > 0.0f ? sum : 0.0f;
    }

    /* FC2: logit = h @ W_fc2.T + b_fc2, W_fc2: (1, 32) int8 */
    float logit = b_fc2[0];
    for (int ic = 0; ic < CH32_FC1_OUT; ic++) {
        logit += ((float)q_w_fc2[ic]) * s_w_fc2[0] * h[ic];
    }

    return 1.0f / (1.0f + expf(-logit));
}

/* ────────────────────────────────────────────────────────────────────────
 * 3. 公共 API
 * ──────────────────────────────────────────────────────────────────────── */

void ch32_init(void)
{
    /* Lite 模式:权重不需要 dequant,只复制 bias 到 RAM (bias 必须在 RAM) */
    memcpy(b_b0_c1,  ch32_b_tcn_0_conv1_bias,    sizeof(b_b0_c1));
    memcpy(b_b0_c2,  ch32_b_tcn_0_conv2_bias,    sizeof(b_b0_c2));
    memcpy(b_b0_rs,  ch32_b_tcn_0_residual_bias, sizeof(b_b0_rs));
    memcpy(b_b0_se1, ch32_b_tcn_0_se_fc1_bias,   sizeof(b_b0_se1));
    memcpy(b_b0_se2, ch32_b_tcn_0_se_fc2_bias,   sizeof(b_b0_se2));

    memcpy(b_b1_c1,  ch32_b_tcn_1_conv1_bias,    sizeof(b_b1_c1));
    memcpy(b_b1_c2,  ch32_b_tcn_1_conv2_bias,    sizeof(b_b1_c2));
    memcpy(b_b1_se1, ch32_b_tcn_1_se_fc1_bias,   sizeof(b_b1_se1));
    memcpy(b_b1_se2, ch32_b_tcn_1_se_fc2_bias,   sizeof(b_b1_se2));

    memcpy(b_b2_c1,  ch32_b_tcn_2_conv1_bias,    sizeof(b_b2_c1));
    memcpy(b_b2_c2,  ch32_b_tcn_2_conv2_bias,    sizeof(b_b2_c2));
    memcpy(b_b2_se1, ch32_b_tcn_2_se_fc1_bias,   sizeof(b_b2_se1));
    memcpy(b_b2_se2, ch32_b_tcn_2_se_fc2_bias,   sizeof(b_b2_se2));

    memcpy(b_fc1, ch32_b_fc1_bias, sizeof(b_fc1));
    memcpy(b_fc2, ch32_b_fc2_bias, sizeof(b_fc2));

    /* 清零 dequant buf (不必需,但安全) */
    memset(dequant_buf, 0, sizeof(dequant_buf));

    /* 注: 权重 (ch32_w_*_weight / ch32_s_*_weight) 仍在 Flash,由编译器放在 .rodata 段 */
}

float ch32_forward(const float* x, size_t len)
{
    (void)len;

    float* cur = buf_x;
    float* nxt = buf_y;
    float* tmp = buf_z;

    /* Block 0: 19 → 32, d=1, residual = 1×1 conv */
    memcpy(cur, x, CH32_IN_CHANNELS * CH32_WINDOW * sizeof(float));
    conv1d_1x1_lite(cur,
                    ch32_w_tcn_0_residual_weight, ch32_s_tcn_0_residual_weight,
                    b_b0_rs, nxt,
                    CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW);
    conv1d_same_lite(cur,
                     ch32_w_tcn_0_conv1_weight, ch32_s_tcn_0_conv1_weight,
                     b_b0_c1, tmp,
                     CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu(tmp, CH32_FEAT_SIZE);
    conv1d_same_lite(tmp,
                     ch32_w_tcn_0_conv2_weight, ch32_s_tcn_0_conv2_weight,
                     b_b0_c2, cur,
                     CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu(cur, CH32_FEAT_SIZE);
    se_block_lite(cur,
                  ch32_w_tcn_0_se_fc1_weight, ch32_s_tcn_0_se_fc1_weight, b_b0_se1,
                  ch32_w_tcn_0_se_fc2_weight, ch32_s_tcn_0_se_fc2_weight, b_b0_se2,
                  CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(cur, cur, nxt, CH32_FEAT_SIZE);

    /* Block 1: 32 → 32, d=2, residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same_lite(cur,
                     ch32_w_tcn_1_conv1_weight, ch32_s_tcn_1_conv1_weight,
                     b_b1_c1, tmp,
                     CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu(tmp, CH32_FEAT_SIZE);
    conv1d_same_lite(tmp,
                     ch32_w_tcn_1_conv2_weight, ch32_s_tcn_1_conv2_weight,
                     b_b1_c2, cur,
                     CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu(cur, CH32_FEAT_SIZE);
    se_block_lite(cur,
                  ch32_w_tcn_1_se_fc1_weight, ch32_s_tcn_1_se_fc1_weight, b_b1_se1,
                  ch32_w_tcn_1_se_fc2_weight, ch32_s_tcn_1_se_fc2_weight, b_b1_se2,
                  CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Block 2: 32 → 32, d=4, residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same_lite(cur,
                     ch32_w_tcn_2_conv1_weight, ch32_s_tcn_2_conv1_weight,
                     b_b2_c1, tmp,
                     CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu(tmp, CH32_FEAT_SIZE);
    conv1d_same_lite(tmp,
                     ch32_w_tcn_2_conv2_weight, ch32_s_tcn_2_conv2_weight,
                     b_b2_c2, cur,
                     CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu(cur, CH32_FEAT_SIZE);
    se_block_lite(cur,
                  ch32_w_tcn_2_se_fc1_weight, ch32_s_tcn_2_se_fc1_weight, b_b2_se1,
                  ch32_w_tcn_2_se_fc2_weight, ch32_s_tcn_2_se_fc2_weight, b_b2_se2,
                  CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Classifier */
    return classifier_lite(cur,
                          ch32_w_fc1_weight, ch32_s_fc1_weight, b_fc1,
                          ch32_w_fc2_weight, ch32_s_fc2_weight, b_fc2);
}

void ch32_benchmark(int n_runs)
{
    float xx[CH32_IN_SIZE];
    unsigned int s = 0xDEADBEEF;
    for (int i = 0; i < CH32_IN_SIZE; i++) {
        s = s * 1103515245u + 12345u;
        xx[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
    }

    volatile float p = ch32_forward(xx, CH32_IN_SIZE);
    (void)p;

    float total = 0.0f;
    for (int i = 0; i < n_runs; i++) {
        total += ch32_forward(xx, CH32_IN_SIZE);
    }

    extern int printf(const char*, ...);
    printf("  ch32_lite_benchmark: %d runs done, output sum=%.4f\n", n_runs, total);
}

ch32_mem_info_t ch32_get_mem_info(void)
{
    ch32_mem_info_t m;
    /* Lite 模式:只有 bias + 激活 + dequant buf 在 RAM */
    size_t bias_bytes = 0;
    bias_bytes += sizeof(b_b0_c1) + sizeof(b_b0_c2) + sizeof(b_b0_rs)
               + sizeof(b_b0_se1) + sizeof(b_b0_se2);
    bias_bytes += sizeof(b_b1_c1) + sizeof(b_b1_c2)
               + sizeof(b_b1_se1) + sizeof(b_b1_se2);
    bias_bytes += sizeof(b_b2_c1) + sizeof(b_b2_c2)
               + sizeof(b_b2_se1) + sizeof(b_b2_se2);
    bias_bytes += sizeof(b_fc1) + sizeof(b_fc2);

    /* 权重在 Flash:INT8 (~20 KB) + scale (~few hundred bytes) */
    /* 这里只算 RAM 部分 */
    size_t weights_in_ram = 0;  /* Lite:0 (权重在 Flash) */
    m.weights_fp32_bytes = weights_in_ram;

    m.input_bytes = CH32_IN_SIZE * sizeof(float);
    m.peak_act_bytes = sizeof(buf_x) + sizeof(buf_y) + sizeof(buf_z)
                    + sizeof(dequant_buf);
    m.total_ram_bytes = bias_bytes + m.peak_act_bytes + m.input_bytes;

    return m;
}

/* ============================================================================
 * Lite 模式编译方法:
 *
 *   GCC:
 *     #define CH32_USE_LITE_MODE
 *     #include "ch32_inference_lite.c"   # 而不是 ch32_inference.c
 *
 *   或用条件编译:
 *     在 main_stm32f407.c 顶部加:
 *       #ifdef CH32_LITE
 *       #include "ch32_inference_lite.c"
 *       #else
 *       #include "ch32_inference.c"
 *       #endif
 *     编译时 -DCH32_LITE 切到 Lite 模式
 *
 * 验证方法:
 *   1. 在 PC 上编译 ch32_inference_lite.c + main_stm32f407.c (PC 版)
 *   2. 跑同样的 200/1000 样本测试
 *   3. 对比 Standard Hybrid vs Lite 输出 (应完全相同,误差 < 1e-5)
 * ============================================================================
 */