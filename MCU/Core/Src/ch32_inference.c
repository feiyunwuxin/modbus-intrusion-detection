/* ============================================================================
 * ch32_inference.c  --  TCN+SE ch=32 MCU Inference  (FULL FP32, 23-dim, w=16)
 *                        STM32H743VIT6 (Cortex-M7 @ 480MHz, FPU + DSP)
 *
 * ⭐ 2026-10-18 路线 E 迁移 + ⭐⭐⭐ FP32 部署 (取代 INT8 Hybrid)
 *   路径:
 *     训练: PyTorch FP32  -> ONNX FP32 -> SavedModel
 *     PC 仿真: TFLite FP32 (rel<1e-7 跟 ONNX parity)
 *     MCU: 手写 C kernel + 全 FP32 权重 (本文件)
 *   取代原因:
 *     INT8 Hybrid + T=2.0 + thr=0.5 在 H743 4-min 实测 FPR=14.43% (vs PC sim 2.27%)
 *     FP32 完全消除 INT8 dequant drift (0.5-2% per layer 累积), 应该跟训练 thr=0.5 对齐
 *
 * 资源占用 (ch=32 FP32, H743 @ -O2 -ffast-math):
 *   - 权重 (FP32, .rodata):       ~80 KB
 *   - 权重 (dequant 后 RAM):     ~80 KB (启动时一次性 memcpy 进来, 可省)
 *   - 激活缓冲 (峰值):            ~8 KB
 *   - 总 RAM:                    ~88 KB
 *   - Flash (权重 + 代码):      ~190 KB
 *   - 推理时间 @ 480 MHz:         ~0.20 ms (比 INT8 慢 50%, 但精度优先)
 * ============================================================================
 */
#include "ch32_inference.h"
#include "model_v4_se_23dim_ch32_fp32_s42.h"   /* ⭐⭐⭐ FULL FP32, no INT8 scales */
#include <string.h>
#include <math.h>
#include <stdio.h>   /* printf for [DCACHE-DBG] (gated by MODBUS_IDS_DCACHE_INSTRUMENT) */

/* H743 HAL 提供 DWT cycle counter (main.h -> stm32h7xx_hal.h) */
#include "main.h"

/* ⭐ 2026-10-19 DCache instrumentation: 站点 3 — ch32_forward memcpy(buf_x, x_in) 后 */
#include "modbus_ids.h"   /* MODBUS_IDS_DCACHE_INSTRUMENT 宏 + g_stats */

/* --- 1. RAM 缓冲:权重 (FP32) + 激活 ---------------------------------- */

/* Block 0: conv1 (32,23,3) + conv2 (32,32,3) + residual (32,23,1) + SE(fc1:4,32 + fc2:32,4) */
float w_b0_c1[CH32_CH * CH32_IN_CHANNELS * 3];  /* 32*23*3 = 2208 */
float w_b0_c2[CH32_CH * CH32_CH       * 3];      /* 32*32*3 = 3072 */
float w_b0_rs[CH32_CH * CH32_IN_CHANNELS * 1];   /* 32*23*1 = 736 */
float w_b0_se1[CH32_SE_HIDDEN * CH32_CH];        /* 4*32 = 128 */
float w_b0_se2[CH32_CH * CH32_SE_HIDDEN];        /* 32*4 = 128 */

/* Block 1 */
float w_b1_c1[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b1_c2[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b1_se1[CH32_SE_HIDDEN * CH32_CH];        /* 128 */
float w_b1_se2[CH32_CH * CH32_SE_HIDDEN];        /* 128 */

/* Block 2 */
float w_b2_c1[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b2_c2[CH32_CH * CH32_CH * 3];            /* 3072 */
float w_b2_se1[CH32_SE_HIDDEN * CH32_CH];        /* 128 */
float w_b2_se2[CH32_CH * CH32_SE_HIDDEN];        /* 128 */

/* Classifier: fc1 (32,32) + fc2 (1,32) */
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
 * Conv1d (same padding, 带 dilation)
 *   in : (in_ch, T)
 *   out: (out_ch, T)
 *   weight: (out_ch, in_ch, kernel)
 *   dilation: d
 *   bias: (out_ch,)
 *
 * ⭐⭐⭐ 2026-10-20 BUG FIX: 对齐语义必须是 **对称 same padding**, 跟训练时的
 *   PyTorch nn.Conv1d(padding=pad, dilation=d) 完全一致:
 *
 *     PyTorch:  out[t] = sum_k w[k] * in[t + k*d - pad]      (pad = (k-1)*d/2)
 *
 *   旧代码写的是:
 *
 *     int t_in = t + (k - pad) * dilation;   // = t + k*d - pad*d  (因果!)
 *
 *   d=1 时两者巧合相同 (t-1, t, t+1), 所以 block 0 看起来是对的;
 *   d=2 时旧代码取 (t-4, t-2, t), 训练模型取 (t-2, t, t+2);
 *   d=4 时旧代码取 (t-12, t-4, t), 训练模型取 (t-4, t, t+4).
 *
 *   ⇒ 旧代码算的根本不是训练出来的那个网络! 实测 1546 个 alarm window
 *     复现 MAE=0.0005 (完美复现 H743 实测), 而跟 ONNX/PyTorch 的 logit
 *     偏差最大 25.9, 造成 Recall 只有 61.4%。
 *
 *   ⚠ 旧 bug 被 parity_ch32_vs_onnx.py 漏掉的原因: 那个脚本的 Python 复刻
 *     用的是 t - pad + k*d (正确对称), 从来没跑过真正的 C kernel。
 */
static inline void conv1d_same(
    const float* in, const float* w, const float* b, float* out,
    int in_ch, int out_ch, int T, int kernel, int dilation)
{
    int pad = (kernel - 1) * dilation / 2;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = b[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    /* ⭐ 对称 same padding, 跟 PyTorch nn.Conv1d(padding=pad, dilation=d) 一致 */
                    int t_in = t + k * dilation - pad;
                    if (t_in < 0 || t_in >= T) continue;
                    sum += w[oc * in_ch * kernel + ic * kernel + k] * in[ic * T + t_in];
                }
            }
            out[oc * T + t] = sum;
        }
    }
}

/**
 * 1x1 Conv (用于 Block 0 的 residual 旁路)
 *   in : (in_ch, T)
 *   out: (out_ch, T)
 *   weight: (out_ch, in_ch, 1)
 *   bias: (out_ch,)
 */
static inline void conv1d_1x1(
    const float* in, const float* w, const float* b, float* out,
    int in_ch, int out_ch, int T)
{
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = b[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                sum += w[oc * in_ch + ic] * in[ic * T + t];
            }
            out[oc * T + t] = sum;
        }
    }
}

/**
 * In-place ReLU
 */
static inline void relu_(float* x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/**
 * Squeeze-and-Excitation block
 *   x   : (ch, T) in-place modified
 *   w1  : (hidden, ch)  fc1 weight
 *   b1  : (hidden,)     fc1 bias
 *   w2  : (ch, hidden)  fc2 weight
 *   b2  : (ch,)         fc2 bias
 *   ch, hidden, T      通道数/隐藏维/时间步
 */
static inline void se_block(
    float* x, const float* w1, const float* b1, const float* w2, const float* b2,
    int ch, int hidden, int T)
{
    /* GAP -> (ch,) */
    float z[CH32_CH_MAX];
    for (int c = 0; c < ch; c++) {
        float s = 0.0f;
        for (int t = 0; t < T; t++) s += x[c * T + t];
        z[c] = s / T;
    }
    /* FC1: (hidden,) = ReLU(z @ W1.T + b1) */
    float h[CH32_SE_HIDDEN_MAX];
    for (int o = 0; o < hidden; o++) {
        float sum = b1[o];
        for (int i = 0; i < ch; i++) {
            sum += w1[o * ch + i] * z[i];
        }
        h[o] = sum > 0.0f ? sum : 0.0f;
    }
    /* FC2: (ch,) = Sigmoid(h @ W2.T + b2) */
    float scale[CH32_CH_MAX];
    for (int o = 0; o < ch; o++) {
        float sum = b2[o];
        for (int i = 0; i < hidden; i++) {
            sum += w2[o * hidden + i] * h[i];
        }
        scale[o] = 1.0f / (1.0f + expf(-sum));
    }
    /* x[c,t] *= scale[c] */
    for (int c = 0; c < ch; c++) {
        for (int t = 0; t < T; t++) {
            x[c * T + t] *= scale[c];
        }
    }
}

/**
 * y = ReLU(x + res), n = ch * T
 * 注:tmp 可以跟 cur 或 res 重叠 (因为是 in-place add)
 */
static inline void add_residual_relu(float* y, const float* x, const float* res, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i] + res[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/**
 * 分类器: GAP -> FC1 + ReLU -> FC2 + logit/T + Sigmoid
 *   x   : (ch, T)
 *   w_fc1, b_fc1 : (FC1_OUT, ch) + (FC1_OUT,)
 *   w_fc2, b_fc2 : (FC2_OUT, FC1_OUT) + (FC2_OUT,)
 */
static inline float classifier(
    const float* x,
    const float* w_fc1, const float* b_fc1,
    const float* w_fc2, const float* b_fc2)
{
    int ch = CH32_CH;
    int T = CH32_WINDOW;

    /* GAP */
    float z[CH32_CH_MAX];
    for (int c = 0; c < ch; c++) {
        float s = 0.0f;
        for (int t = 0; t < T; t++) s += x[c * T + t];
        z[c] = s / T;
    }

    /* FC1: h = ReLU(z @ W_fc1.T + b_fc1) */
    float h[CH32_FC1_OUT_MAX];
    for (int o = 0; o < CH32_FC1_OUT; o++) {
        const float* w_o = w_fc1 + o * ch;
        float sum = b_fc1[o];
        for (int ic = 0; ic < ch; ic++) {
            sum += w_o[ic] * z[ic];
        }
        h[o] = sum > 0.0f ? sum : 0.0f;
        /* 注:训练时 FC1 后有 Dropout(0.1),推理时关闭,无操作 */
    }

    /* FC2: logit = h @ W_fc2.T + b_fc2 */
    float logit;
    {
        const float* w_o = w_fc2;
        float sum = b_fc2[0];
        for (int ic = 0; ic < CH32_FC1_OUT; ic++) {
            sum += w_o[ic] * h[ic];
        }
        logit = sum;
    }

    /* ⭐ 2026-10-18 FP32 部署: temperature scaling 设成 1.0 (T=1.0 = no scaling)
     *   训练时 PyTorch 没做 T scaling, 部署也保持 T=1.0, 让 logit 跟 ONNX/PyTorch 1:1 一致.
     *   INT8 Hybrid 时代的 T=2.0 偏移是 dequant drift 经验值, FP32 没这个 drift. */
    logit /= CH32_TEMPERATURE;

    return 1.0f / (1.0f + expf(-logit));
}

/* --- 3. 公共 API 实现 ---------------------------------------------------- */

void ch32_init(void)
{
    /* ⭐⭐⭐ FP32 部署: 直接 memcpy const FP32 -> RAM, 跳过 INT8 dequant
     *   启动时一次性 ~80KB copy, 推理时纯 FP32 (跟 PyTorch / ONNX / TFLite PC sim bit-perfect)
     */

    /* Block 0 权重 */
    memcpy(w_b0_c1,  v4se23_ch32_s42_w_tcn_0_conv1_weight,    sizeof(w_b0_c1));
    memcpy(w_b0_c2,  v4se23_ch32_s42_w_tcn_0_conv2_weight,    sizeof(w_b0_c2));
    memcpy(w_b0_rs,  v4se23_ch32_s42_w_tcn_0_residual_weight, sizeof(w_b0_rs));
    memcpy(w_b0_se1, v4se23_ch32_s42_w_tcn_0_se_fc1_weight,   sizeof(w_b0_se1));
    memcpy(w_b0_se2, v4se23_ch32_s42_w_tcn_0_se_fc2_weight,   sizeof(w_b0_se2));

    /* Block 1 */
    memcpy(w_b1_c1,  v4se23_ch32_s42_w_tcn_1_conv1_weight,  sizeof(w_b1_c1));
    memcpy(w_b1_c2,  v4se23_ch32_s42_w_tcn_1_conv2_weight,  sizeof(w_b1_c2));
    memcpy(w_b1_se1, v4se23_ch32_s42_w_tcn_1_se_fc1_weight, sizeof(w_b1_se1));
    memcpy(w_b1_se2, v4se23_ch32_s42_w_tcn_1_se_fc2_weight, sizeof(w_b1_se2));

    /* Block 2 */
    memcpy(w_b2_c1,  v4se23_ch32_s42_w_tcn_2_conv1_weight,  sizeof(w_b2_c1));
    memcpy(w_b2_c2,  v4se23_ch32_s42_w_tcn_2_conv2_weight,  sizeof(w_b2_c2));
    memcpy(w_b2_se1, v4se23_ch32_s42_w_tcn_2_se_fc1_weight, sizeof(w_b2_se1));
    memcpy(w_b2_se2, v4se23_ch32_s42_w_tcn_2_se_fc2_weight, sizeof(w_b2_se2));

    /* Classifier */
    memcpy(w_fc1, v4se23_ch32_s42_fc1_weight, sizeof(w_fc1));
    memcpy(w_fc2, v4se23_ch32_s42_fc2_weight, sizeof(w_fc2));

    /* Bias 直接拷贝 (FP32, 跟训练 BN-folded 一致) */
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

#if MODBUS_IDS_DCACHE_INSTRUMENT
    /* ⭐ 2026-10-19 DCache instrumentation: 站点 3 — input 落地后立即 print.
     *   若 buf_x ch 12-15 出现 garbage (跟站点 1 / 2 不一致),
     *   → 真因在 ch32_forward 调用链 (memcpy buf_x / stack → buf_x 段 cache 问题).
     *   若一致 → input OK, 真因在 ch32_forward 内部算法 (算法 bug). */
    {
        printf("[DCACHE-DBG w#%u] SITE 3 ch32_forward after memcpy(buf_x): "
               "ch12_t0=%.4f ch13_t0=%.4f ch14_t0=%.4f ch15_t0=%.4f\r\n",
               (unsigned)g_stats.windows_total,
               buf_x[12*CH32_WINDOW+0], buf_x[13*CH32_WINDOW+0],
               buf_x[14*CH32_WINDOW+0], buf_x[15*CH32_WINDOW+0]);
    }
#endif

    /* residual = 1x1 conv(23->32) -> 写到 nxt */
    conv1d_1x1(buf_x, w_b0_rs, b_b0_rs, nxt,
               CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW);

    /* main path: conv1(23->32) -> ReLU -> conv2(32->32) -> ReLU -> SE -> cur */
    conv1d_same(buf_x, w_b0_c1, b_b0_c1, tmp,
                CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu_(tmp, CH32_CH * CH32_WINDOW);
    conv1d_same(tmp, w_b0_c2, b_b0_c2, cur,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu_(cur, CH32_CH * CH32_WINDOW);
    se_block(cur, w_b0_se1, b_b0_se1, w_b0_se2, b_b0_se2,
             CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);

    /* add residual + ReLU -> cur */
    add_residual_relu(cur, cur, nxt, CH32_CH * CH32_WINDOW);

    /* Block 1: cur (32,16) -> cur (32,16), residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same(cur, w_b1_c1, b_b1_c1, tmp,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu_(tmp, CH32_FEAT_SIZE);
    conv1d_same(tmp, w_b1_c2, b_b1_c2, cur,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu_(cur, CH32_FEAT_SIZE);
    se_block(cur, w_b1_se1, b_b1_se1, w_b1_se2, b_b1_se2,
             CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Block 2: cur (32,16) -> cur (32,16), residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same(cur, w_b2_c1, b_b2_c1, tmp,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu_(tmp, CH32_FEAT_SIZE);
    conv1d_same(tmp, w_b2_c2, b_b2_c2, cur,
                CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu_(cur, CH32_FEAT_SIZE);
    se_block(cur, w_b2_se1, b_b2_se1, w_b2_se2, b_b2_se2,
             CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Classifier */
    return classifier(cur, w_fc1, b_fc1, w_fc2, b_fc2);
}

/* --- 4. 性能 / 自检 ------------------------------------------------------- */

uint32_t ch32_benchmark(int n_runs)
{
    /* 用 ZEROS 输入作为参照 (避免 SE channel 输入=0 时除零) */
    float x[CH32_IN_SIZE];
    for (int i = 0; i < CH32_IN_SIZE; i++) x[i] = 0.1f;

    /* warm-up */
    ch32_forward(x, CH32_IN_SIZE);

    uint32_t t0 = DWT->CYCCNT;
    for (int i = 0; i < n_runs; i++) {
        ch32_forward(x, CH32_IN_SIZE);
    }
    uint32_t t1 = DWT->CYCCNT;

    return (t1 - t0) / n_runs;
}

void ch32_dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

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
