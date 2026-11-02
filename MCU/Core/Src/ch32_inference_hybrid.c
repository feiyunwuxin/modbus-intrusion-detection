/* ============================================================================
 * ch32_inference_hybrid.c  --  TCN+SE ch=32 MCU Inference (Hybrid: INT16 权重 / FP32 激活)
 *                              STM32H743VIT6 (Cortex-M7 @ 480MHz, FPU + DSP)
 *
 * ⭐⭐⭐ 2026-10-27 Hybrid 工程 (第三个后端, 与 FP32 / 全量 INT8 并存)
 *
 * 方案 (Lite mode): 权重 int16 **常驻 Flash**, 不做启动时 RAM 拷贝;
 *                    激活全 FP32; 运行期按输出通道 dequant.
 *
 *   权重: int16 per-output-channel 对称, scale[oc] = absmax(oc)/32767
 *   dequant: w = (float)q * scale[oc]   (逐输出通道, 只在用到时算)
 *
 * 数值语义 (跟 PC 侧 exp_hybrid_conclusion.py 的 numpy 镜像一致)
 * -----------------------------------------------------------
 *   conv1d_same 的累加顺序 **跟 Route E FP32 完全相同**:
 *       sum = b[oc]; sum += w[oc][ic][k] * in[ic][t_in];
 *   唯一的差别是 w 来自 int16 反量化 (相对误差 <= 1.5e-5).
 *   dequant 提到 per-oc 缓存 (float wq_f[]) 是 **逐位等价** 的:
 *       内联写法  (float)q * scale * x      <- C 里左结合 = ((float)q * scale) * x
 *       缓存写法  wq_f[i] = (float)q * scale;  sum += wq_f[i] * x
 *   两者第一次乘法完全相同, 所以结果逐位一致.
 *
 * 与 Route E (ch32_inference.c) 的对应关系
 * -----------------------------------------
 *   conv1d_same   -> conv1d_same_i16   (多一个 per-channel scale)
 *   conv1d_1x1    -> conv1d_1x1_i16
 *   se_block      -> se_block_i16      (fc1/fc2 都是 int16)
 *   classifier    -> 不变 (fc1/fc2 太小, 保持 FP32)
 *   ch32_init()   -> **不做任何 memcpy** (省 80.5 KB RAM)
 *
 * ⚠️ t_in 公式必须跟 ch32_inference.c 保持一致 (对称 same padding):
 *       t_in = t + k * dilation - pad
 *   PC 侧 parity 脚本直接从本文件解析这个表达式, 不手抄 (见
 *   memory: feedback-parity-test-must-bind-to-real-source).
 * ============================================================================
 */
#include "ch32_inference.h"

#if CH32_USE_HYBRID

#include "ch32_inference_hybrid.h"
#include "model_v4_se_23dim_ch32_int16_s42.h"   /* int16 权重 + per-channel scale */

#include <string.h>
#include <math.h>
#include <stdio.h>   /* printf (gated by MODBUS_IDS_DCACHE_INSTRUMENT) */

#include "main.h"    /* DWT cycle counter */
#include "modbus_ids.h"   /* MODBUS_IDS_DCACHE_INSTRUMENT 宏 + g_stats */

/* --- 维度常量 (与 ch32_inference.h 一致) --------------------------------- */
#define HY_T    CH32_WINDOW        /* 16 时间步 */
#define HY_C    CH32_CH            /* 32 通道   */
#define HY_IN   CH32_IN_CHANNELS   /* 23 输入维 */
#define HY_SEH  CH32_SE_HIDDEN     /* 4  SE 隐藏维 */
#define HY_FC1  CH32_FC1_OUT       /* 32 分类器 FC1 输出 */

/* per-oc 反量化缓存: conv 最大 in_ch*kernel = 32*3 = 96 */
#define HY_WBUF_MAX (CH32_CH_MAX * CH32_KERNEL)

/* --- 运行期缓冲 ---------------------------------------------------------- */
/* 只放激活, **没有权重 RAM 拷贝** (权重是 const, 常驻 Flash) */
/* 注: buf_x 必须是 FEAT_SIZE (512), block 1/2 里 cur 复用它当 (32,16) 缓冲 */
static float hy_buf_x[CH32_FEAT_SIZE]; /* block0 输入 / 后续 cur        2048 B */
static float hy_buf_y[CH32_FEAT_SIZE]; /* scratch (32,16)              2048 B */
static float hy_buf_z[CH32_FEAT_SIZE]; /* scratch (32,16)              2048 B */

/* ============================================================================
 * 1. 反量化 + 线性层
 * ============================================================================ */

/**
 * per-output-channel 反量化 + 线性层 (fc1 / fc2 / SE 的 fc)
 *   in     : (in_dim,)
 *   wq     : int16 (out_dim, in_dim)
 *   wsc    : float scale per output channel (out_dim,)
 *   b      : float bias (out_dim,)
 *   out    : (out_dim,)
 *
 * 语义 == Route E 的 classifier/se_block 里的手写线性层 (b 先加, 再累加 w*x).
 */
static inline void linear_i16(
    const float* in, const int16_t* wq, const float* wsc,
    const float* b, float* out, int in_dim, int out_dim)
{
    float wq_f[CH32_CH_MAX];

    for (int o = 0; o < out_dim; o++) {
        const int16_t* w_o = wq + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            wq_f[i] = (float)w_o[i] * wsc[o];
        }
        float sum = b[o];
        for (int i = 0; i < in_dim; i++) {
            sum += wq_f[i] * in[i];
        }
        out[o] = sum;
    }
}

/* ============================================================================
 * 2. 卷积
 * ============================================================================ */

/**
 * Conv1d (same padding, 带 dilation) —— INT16 权重版
 *   in : (in_ch, T)          wq : int16 (out_ch, in_ch, kernel)
 *   wsc: scale per out ch   b  : (out_ch,)     out : (out_ch, T)
 *
 * ⭐⭐⭐ 对齐语义: **对称 same padding**, 跟训练时的
 *   PyTorch nn.Conv1d(padding=pad, dilation=d) 完全一致:
 *       out[t] = sum_k w[k] * in[t + k*d - pad]      (pad = (k-1)*d/2)
 *   跟 ch32_inference.c (Route E) 的表达式逐字一致 —— 根因1 的教训:
 *   旧的因果写法 t + (k-pad)*d 在 d=2/4 时算的根本不是训练出来的网络.
 */
static inline void conv1d_same_i16(
    const float* in, const int16_t* wq, const float* wsc, const float* b, float* out,
    int in_ch, int out_ch, int T, int kernel, int dilation)
{
    int pad = (kernel - 1) * dilation / 2;
    float wq_f[HY_WBUF_MAX];

    for (int oc = 0; oc < out_ch; oc++) {
        const int16_t* w_oc = wq + (size_t)oc * in_ch * kernel;
        for (int i = 0; i < in_ch * kernel; i++) {
            wq_f[i] = (float)w_oc[i] * wsc[oc];
        }
        for (int t = 0; t < T; t++) {
            float sum = b[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    /* ⭐ 对称 same padding (跟 Route E ch32_inference.c 一致) */
                    int t_in = t + k * dilation - pad;
                    if (t_in < 0 || t_in >= T) continue;
                    sum += wq_f[ic * kernel + k] * in[ic * T + t_in];
                }
            }
            out[oc * T + t] = sum;
        }
    }
}

/**
 * 1x1 Conv (Block 0 的 residual 旁路) —— INT16 权重版
 */
static inline void conv1d_1x1_i16(
    const float* in, const int16_t* wq, const float* wsc, const float* b, float* out,
    int in_ch, int out_ch, int T)
{
    float wq_f[CH32_CH_MAX];

    for (int oc = 0; oc < out_ch; oc++) {
        const int16_t* w_oc = wq + (size_t)oc * in_ch;
        for (int i = 0; i < in_ch; i++) {
            wq_f[i] = (float)w_oc[i] * wsc[oc];
        }
        for (int t = 0; t < T; t++) {
            float sum = b[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                sum += wq_f[ic] * in[ic * T + t];
            }
            out[oc * T + t] = sum;
        }
    }
}

/* ============================================================================
 * 3. 通用算子 (跟 Route E 完全一致)
 * ============================================================================ */

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
 * Squeeze-and-Excitation block (INT16 权重版)
 *   x   : (ch, T) in-place modified
 *   w1  : int16 (hidden, ch)   s1 : scale per hidden
 *   w2  : int16 (ch, hidden)   s2 : scale per ch
 *   b1  : (hidden,)            b2 : (ch,)
 */
static inline void se_block_i16(
    float* x,
    const int16_t* w1, const float* s1, const float* b1,
    const int16_t* w2, const float* s2, const float* b2,
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
    linear_i16(z, w1, s1, b1, h, ch, hidden);
    for (int o = 0; o < hidden; o++) {
        if (h[o] < 0.0f) h[o] = 0.0f;
    }
    /* FC2: (ch,) = Sigmoid(h @ W2.T + b2) */
    float scale[CH32_CH_MAX];
    linear_i16(h, w2, s2, b2, scale, hidden, ch);
    for (int o = 0; o < ch; o++) {
        scale[o] = 1.0f / (1.0f + expf(-scale[o]));
    }
    /* x[c,t] *= scale[c] */
    for (int c = 0; c < ch; c++) {
        for (int t = 0; t < T; t++) {
            x[c * T + t] *= scale[c];
        }
    }
}

/**
 * y = ReLU(x + res)
 */
static inline void add_residual_relu(float* y, const float* x, const float* res, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i] + res[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/**
 * 分类器: GAP -> FC1 + ReLU -> FC2 -> logit/T + Sigmoid
 *   fc1/fc2 保持 FP32 (太小, 量化损失大)
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
        /* 注:训练时 FC1 后有 Dropout(0.3),推理时关闭,无操作 */
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

    logit /= CH32_TEMPERATURE;

    return 1.0f / (1.0f + expf(-logit));
}

/* ============================================================================
 * 4. 公共 API 实现
 * ============================================================================ */

void ch32_hybrid_init(void)
{
    /* ⭐⭐⭐ Hybrid 的关键优势: 权重 int16 常驻 Flash, **不做任何 memcpy**.
     *   对比 Route E FP32: 启动时把 80.5 KB FP32 权重拷到 RAM (省 80.5 KB RAM).
     *   这里只做一次常量健全性自检 (编译器会把权重地址折进去, 不会优化掉).
     */
    volatile const void* p = (const void*)v4se23_ch32_i16_s42_w_tcn_0_conv1_weight;
    (void)p;
}

float ch32_hybrid_forward(const float* x, size_t len)
{
    (void)len;

    const float* x_in = x;
    float* cur = hy_buf_x;        /* 当前块输入/输出 */
    float* nxt = hy_buf_y;        /* 临时 (32, 16) */
    float* tmp = hy_buf_z;        /* scratch (32, 16) */

    /* Block 0: 输入 23 维 -> 输出 32 维, residual = 1x1 conv(23->32) */
    memcpy(hy_buf_x, x_in, CH32_IN_CHANNELS * CH32_WINDOW * sizeof(float));

#if MODBUS_IDS_DCACHE_INSTRUMENT
    printf("[DCACHE-DBG w#%u] HYBRID SITE 3 ch32_forward after memcpy(buf_x): "
           "ch12_t0=%.4f ch13_t0=%.4f ch14_t0=%.4f ch15_t0=%.4f\r\n",
           (unsigned)g_stats.windows_total,
           hy_buf_x[12 * CH32_WINDOW + 0], hy_buf_x[13 * CH32_WINDOW + 0],
           hy_buf_x[14 * CH32_WINDOW + 0], hy_buf_x[15 * CH32_WINDOW + 0]);
#endif

    /* residual = 1x1 conv(23->32) -> 写到 nxt */
    conv1d_1x1_i16(hy_buf_x,
                   v4se23_ch32_i16_s42_w_tcn_0_residual_weight,
                   v4se23_ch32_i16_s42_s_tcn_0_residual_weight,
                   v4se23_ch32_i16_s42_b_tcn_0_residual_bias,
                   nxt, CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW);

    /* main path: conv1(23->32) -> ReLU -> conv2(32->32) -> ReLU -> SE -> cur */
    conv1d_same_i16(hy_buf_x,
                    v4se23_ch32_i16_s42_w_tcn_0_conv1_weight,
                    v4se23_ch32_i16_s42_s_tcn_0_conv1_weight,
                    v4se23_ch32_i16_s42_b_tcn_0_conv1_bias,
                    tmp, CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu_(tmp, CH32_CH * CH32_WINDOW);
    conv1d_same_i16(tmp,
                    v4se23_ch32_i16_s42_w_tcn_0_conv2_weight,
                    v4se23_ch32_i16_s42_s_tcn_0_conv2_weight,
                    v4se23_ch32_i16_s42_b_tcn_0_conv2_bias,
                    cur, CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
    relu_(cur, CH32_CH * CH32_WINDOW);
    se_block_i16(cur,
                 v4se23_ch32_i16_s42_w_tcn_0_se_fc1_weight,
                 v4se23_ch32_i16_s42_s_tcn_0_se_fc1_weight,
                 v4se23_ch32_i16_s42_b_tcn_0_se_fc1_bias,
                 v4se23_ch32_i16_s42_w_tcn_0_se_fc2_weight,
                 v4se23_ch32_i16_s42_s_tcn_0_se_fc2_weight,
                 v4se23_ch32_i16_s42_b_tcn_0_se_fc2_bias,
                 CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);

    /* add residual + ReLU -> cur */
    add_residual_relu(cur, cur, nxt, CH32_CH * CH32_WINDOW);

    /* Block 1: cur (32,16) -> cur (32,16), residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same_i16(cur,
                    v4se23_ch32_i16_s42_w_tcn_1_conv1_weight,
                    v4se23_ch32_i16_s42_s_tcn_1_conv1_weight,
                    v4se23_ch32_i16_s42_b_tcn_1_conv1_bias,
                    tmp, CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu_(tmp, CH32_FEAT_SIZE);
    conv1d_same_i16(tmp,
                    v4se23_ch32_i16_s42_w_tcn_1_conv2_weight,
                    v4se23_ch32_i16_s42_s_tcn_1_conv2_weight,
                    v4se23_ch32_i16_s42_b_tcn_1_conv2_bias,
                    cur, CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B1);
    relu_(cur, CH32_FEAT_SIZE);
    se_block_i16(cur,
                 v4se23_ch32_i16_s42_w_tcn_1_se_fc1_weight,
                 v4se23_ch32_i16_s42_s_tcn_1_se_fc1_weight,
                 v4se23_ch32_i16_s42_b_tcn_1_se_fc1_bias,
                 v4se23_ch32_i16_s42_w_tcn_1_se_fc2_weight,
                 v4se23_ch32_i16_s42_s_tcn_1_se_fc2_weight,
                 v4se23_ch32_i16_s42_b_tcn_1_se_fc2_bias,
                 CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Block 2: cur (32,16) -> cur (32,16), residual = identity */
    memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));
    conv1d_same_i16(cur,
                    v4se23_ch32_i16_s42_w_tcn_2_conv1_weight,
                    v4se23_ch32_i16_s42_s_tcn_2_conv1_weight,
                    v4se23_ch32_i16_s42_b_tcn_2_conv1_bias,
                    tmp, CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu_(tmp, CH32_FEAT_SIZE);
    conv1d_same_i16(tmp,
                    v4se23_ch32_i16_s42_w_tcn_2_conv2_weight,
                    v4se23_ch32_i16_s42_s_tcn_2_conv2_weight,
                    v4se23_ch32_i16_s42_b_tcn_2_conv2_bias,
                    cur, CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B2);
    relu_(cur, CH32_FEAT_SIZE);
    se_block_i16(cur,
                 v4se23_ch32_i16_s42_w_tcn_2_se_fc1_weight,
                 v4se23_ch32_i16_s42_s_tcn_2_se_fc1_weight,
                 v4se23_ch32_i16_s42_b_tcn_2_se_fc1_bias,
                 v4se23_ch32_i16_s42_w_tcn_2_se_fc2_weight,
                 v4se23_ch32_i16_s42_s_tcn_2_se_fc2_weight,
                 v4se23_ch32_i16_s42_b_tcn_2_se_fc2_bias,
                 CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);
    add_residual_relu(tmp, cur, nxt, CH32_FEAT_SIZE);
    memcpy(cur, tmp, CH32_FEAT_SIZE * sizeof(float));

    /* Classifier */
    return classifier(cur,
                      v4se23_ch32_i16_s42_fc1_weight, v4se23_ch32_i16_s42_b_fc1_bias,
                      v4se23_ch32_i16_s42_fc2_weight, v4se23_ch32_i16_s42_b_fc2_bias);
}

/* ============================================================================
 * 5. 性能 / 自检
 * ============================================================================ */

uint32_t ch32_hybrid_benchmark(int n_runs)
{
    /* 用 0.1 输入作为参照 (避免 SE channel 输入=0 时除零) */
    float x[CH32_IN_SIZE];
    for (int i = 0; i < CH32_IN_SIZE; i++) x[i] = 0.1f;

    /* warm-up */
    ch32_hybrid_forward(x, CH32_IN_SIZE);

    uint32_t t0 = DWT->CYCCNT;
    for (int i = 0; i < n_runs; i++) {
        ch32_hybrid_forward(x, CH32_IN_SIZE);
    }
    uint32_t t1 = DWT->CYCCNT;

    return (t1 - t0) / n_runs;
}

void ch32_hybrid_dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

ch32_mem_info_t ch32_hybrid_get_mem_info(void)
{
    ch32_mem_info_t m;
    size_t w = 0;

    /* int16 权重 (13 个 conv/SE 张量) */
    w += sizeof(v4se23_ch32_i16_s42_w_tcn_0_conv1_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_0_conv2_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_0_residual_weight);
    w += sizeof(v4se23_ch32_i16_s42_w_tcn_1_conv1_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_1_conv2_weight);
    w += sizeof(v4se23_ch32_i16_s42_w_tcn_2_conv1_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_2_conv2_weight);
    w += sizeof(v4se23_ch32_i16_s42_w_tcn_0_se_fc1_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_1_se_fc1_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_2_se_fc1_weight);
    w += sizeof(v4se23_ch32_i16_s42_w_tcn_0_se_fc2_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_1_se_fc2_weight)
       + sizeof(v4se23_ch32_i16_s42_w_tcn_2_se_fc2_weight);
    /* per-channel scale */
    w += sizeof(v4se23_ch32_i16_s42_s_tcn_0_conv1_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_0_conv2_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_0_residual_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_1_conv1_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_1_conv2_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_2_conv1_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_2_conv2_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_0_se_fc1_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_1_se_fc1_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_2_se_fc1_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_0_se_fc2_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_1_se_fc2_weight)
       + sizeof(v4se23_ch32_i16_s42_s_tcn_2_se_fc2_weight);
    /* FP32 部分 (fc1/fc2 权重 + 全部 bias) */
    w += sizeof(v4se23_ch32_i16_s42_fc1_weight)
       + sizeof(v4se23_ch32_i16_s42_fc2_weight)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_0_conv1_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_0_conv2_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_0_residual_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_1_conv1_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_1_conv2_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_2_conv1_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_2_conv2_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_0_se_fc1_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_1_se_fc1_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_2_se_fc1_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_0_se_fc2_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_1_se_fc2_bias)
       + sizeof(v4se23_ch32_i16_s42_b_tcn_2_se_fc2_bias)
       + sizeof(v4se23_ch32_i16_s42_b_fc1_bias)
       + sizeof(v4se23_ch32_i16_s42_b_fc2_bias);

    m.weights_fp32_bytes = w;     /* 字段复用: 这里报的是 .rodata 权重字节数 */

    m.input_bytes = CH32_IN_SIZE * sizeof(float);
    m.peak_act_bytes = sizeof(hy_buf_x) + sizeof(hy_buf_y) + sizeof(hy_buf_z);
    m.total_ram_bytes = m.peak_act_bytes + m.input_bytes;   /* 无权重 RAM 拷贝! */

    return m;
}

#endif /* CH32_USE_HYBRID */
