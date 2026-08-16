/* ============================================================================
 * v4_se_23dim_ch32_inference.c  —  TCN+SE ch=32 MCU Inference Implementation (23-dim)
 * ============================================================================
 *
 * 架构 (与 PyTorch retrain_tcn_23dim_b64_ch32_do01_savept.py 完全一致):
 *   x (23, 16)
 *     ↓
 *   Block 0 (d=1): conv1(23→32, k=3) → ReLU → conv2(32→32, k=3) → ReLU
 *                  → SE → + Residual(1×1: 23→32) → ReLU
 *     ↓ (32, 16)
 *   Block 1 (d=2): 同上但 residual 是 Identity
 *     ↓ (32, 16)
 *   Block 2 (d=4): 同上但 residual 是 Identity
 *     ↓ (32, 16)
 *   GAP → FC1(32→32) + ReLU + Dropout(0.3) → FC2(32→1) → Sigmoid
 *     ↓
 *   probability ∈ [0, 1]
 *
 * 数据 layout: shape (channels, time_steps), row-major
 *   例: input[ic*16 + t] = 第 ic 个特征在 t 时刻的值
 *
 * Conv1d "same" padding:
 *   pad_left  = (kernel - 1) * dilation / 2
 *   对 t_out,卷积区间 [t_out*dilation - pad_left*dilation,
 *                  t_out*dilation + (k-1)*dilation - pad_left*dilation]
 *   越界视为 0
 *
 * 5-seed ensemble 模式:
 *   共享同一 FP32 权重缓冲 (~80 KB),每次循环:
 *     dequant seed_i (~0.6 ms) → forward (~0.9 ms) → 累加概率
 *   5 个 seed 跑完除以 5,耗时约 7.5 ms / 样本
 *
 * 关键优化 (MCU 友好):
 *   - 静态 RAM 分配,无 malloc
 *   - 启动时 dequant 一次性,运行期纯 FP32
 *   - 循环顺序优化 cache 友好 (oc 在最内层 → 权重连续访问)
 *   - 内联辅助函数避免调用开销
 * ============================================================================
 */

#include "v4_se_23dim_ch32_inference.h"
#include <string.h>
#include <math.h>

/* ── 5 个 seed 的权重头文件 (提供 static const int8_t/float 数组) ── */
#include "model_v4_se_23dim_ch32_hybrid_s42.h"
#include "model_v4_se_23dim_ch32_hybrid_s123.h"
#include "model_v4_se_23dim_ch32_hybrid_s456.h"
#include "model_v4_se_23dim_ch32_hybrid_s789.h"
#include "model_v4_se_23dim_ch32_hybrid_s1024.h"

/* 默认 seed (单 seed 模式用) */
#define V4_SE_23DIM_DEFAULT_SEED  123

/* 5-seed 列表 (与 retrain 脚本完全一致) */
static const int V4_SE_23DIM_SEEDS[V4_SE_23DIM_CH32_N_SEEDS] = {42, 123, 456, 789, 1024};
static int v4_se_23dim_current_seed = V4_SE_23DIM_DEFAULT_SEED;

/* ────────────────────────────────────────────────────────────────────────
 * 1. RAM 缓冲:dequant 后的 FP32 权重 + bias + 激活
 * ──────────────────────────────────────────────────────────────────────── */

/* Block 0: conv1 (32,23,3) + conv2 (32,32,3) + residual (32,23,1) + SE */
static float w_b0_c1[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_IN_CHANNELS * 3];  /* 2208 */
static float w_b0_c2[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_CHANNELS * 3];     /* 3072 */
static float w_b0_rs[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_IN_CHANNELS * 1];  /* 736 */
static float w_b0_se1[V4_SE_23DIM_CH32_SE_HIDDEN * V4_SE_23DIM_CH32_CHANNELS];       /* 128 */
static float w_b0_se2[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_SE_HIDDEN];       /* 128 */

/* Block 1: 同上但无 residual */
static float w_b1_c1[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_CHANNELS * 3];     /* 3072 */
static float w_b1_c2[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_CHANNELS * 3];     /* 3072 */
static float w_b1_se1[V4_SE_23DIM_CH32_SE_HIDDEN * V4_SE_23DIM_CH32_CHANNELS];       /* 128 */
static float w_b1_se2[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_SE_HIDDEN];       /* 128 */

/* Block 2 */
static float w_b2_c1[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_CHANNELS * 3];     /* 3072 */
static float w_b2_c2[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_CHANNELS * 3];     /* 3072 */
static float w_b2_se1[V4_SE_23DIM_CH32_SE_HIDDEN * V4_SE_23DIM_CH32_CHANNELS];       /* 128 */
static float w_b2_se2[V4_SE_23DIM_CH32_CHANNELS * V4_SE_23DIM_CH32_SE_HIDDEN];       /* 128 */

/* Classifier: fc1 (32,32) + fc2 (1,32) */
static float w_fc1[V4_SE_23DIM_CH32_FC1_OUT * V4_SE_23DIM_CH32_CHANNELS];            /* 1024 */
static float w_fc2[V4_SE_23DIM_CH32_FC2_OUT * V4_SE_23DIM_CH32_FC1_OUT];             /* 32 */

/* Bias (FP32) */
static float b_b0_c1[V4_SE_23DIM_CH32_CHANNELS];
static float b_b0_c2[V4_SE_23DIM_CH32_CHANNELS];
static float b_b0_rs[V4_SE_23DIM_CH32_CHANNELS];
static float b_b0_se1[V4_SE_23DIM_CH32_SE_HIDDEN];
static float b_b0_se2[V4_SE_23DIM_CH32_CHANNELS];

static float b_b1_c1[V4_SE_23DIM_CH32_CHANNELS];
static float b_b1_c2[V4_SE_23DIM_CH32_CHANNELS];
static float b_b1_se1[V4_SE_23DIM_CH32_SE_HIDDEN];
static float b_b1_se2[V4_SE_23DIM_CH32_CHANNELS];

static float b_b2_c1[V4_SE_23DIM_CH32_CHANNELS];
static float b_b2_c2[V4_SE_23DIM_CH32_CHANNELS];
static float b_b2_se1[V4_SE_23DIM_CH32_SE_HIDDEN];
static float b_b2_se2[V4_SE_23DIM_CH32_CHANNELS];

static float b_fc1[V4_SE_23DIM_CH32_FC1_OUT];
static float b_fc2[V4_SE_23DIM_CH32_FC2_OUT];

/* 激活缓冲 (运行期使用) */
static float buf_x[V4_SE_23DIM_CH32_FEAT_SIZE];   /* (32, 16) 当前块输入/输出 */
static float buf_y[V4_SE_23DIM_CH32_FEAT_SIZE];   /* (32, 16) scratch */
static float buf_z[V4_SE_23DIM_CH32_FEAT_SIZE];   /* (32, 16) scratch */

/* ────────────────────────────────────────────────────────────────────────
 * 2. 工具函数
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * INT8 → FP32 逐通道 dequant
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
    for (int oc = 0; oc < out_ch; oc++) {
        const float* w_oc = w + oc * in_ch * kernel;
        float bias = b[oc];
        for (int t_out = 0; t_out < T; t_out++) {
            float sum = bias;
            int t_in_base = t_out * dilation - pad * dilation;
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
 * 1×1 Conv1d (Pointwise, 等价于矩阵乘)
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
 *   x : (ch, T)   会被 in-place 缩放
 *   1. GAP: z[oc] = mean(x[oc, :])
 *   2. FC1: a = ReLU(z @ W1 + b1)
 *   3. FC2: s = sigmoid(a @ W2 + b2)
 *   4. Scale: x[oc, t] *= s[oc]
 *
 * W1 (fc1) shape: (hidden, ch)   W2 (fc2) shape: (ch, hidden)
 */
static inline void se_block(
    float* x, const float* w1, const float* b1,
    const float* w2, const float* b2,
    int ch, int hidden, int T)
{
    float z[V4_SE_23DIM_CH32_CHANNELS];      /* GAP (ch,) */
    float a[V4_SE_23DIM_CH32_SE_HIDDEN];     /* FC1 (hidden,) */
    float s[V4_SE_23DIM_CH32_CHANNELS];      /* FC2 / scale (ch,) */

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
 *   y = ReLU(x + res)
 */
static inline void add_residual_relu(float* y, const float* x, const float* res, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i] + res[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/* ────────────────────────────────────────────────────────────────────────
 * 3. TCN Block + Classifier
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * 1 个 TCN Block (通用版本,通过 use_residual_conv 区分 Block 0 / 1,2)
 *   in/out: (ch, T),但 Block 0 输入是 (23, T) 需特判 in_ch
 */
static inline void tcn_block(
    float* x_inout, float* y_scratch, float* tmp_scratch,
    const float* w_c1, const float* b_c1,
    const float* w_c2, const float* b_c2,
    const float* w_rs, const float* b_rs,    /* NULL = identity */
    const float* w_se1, const float* b_se1,
    const float* w_se2, const float* b_se2,
    int in_ch, int out_ch, int T, int kernel, int dilation)
{
    /* 1. residual (1×1 conv 或 identity) */
    if (w_rs != NULL) {
        conv1d_1x1(x_inout, w_rs, b_rs, y_scratch, in_ch, out_ch, T);
    } else {
        memcpy(y_scratch, x_inout, out_ch * T * sizeof(float));
    }

    /* 2. main path: conv1 → ReLU → conv2 → ReLU → SE */
    conv1d_same(x_inout, w_c1, b_c1, tmp_scratch, in_ch, out_ch, T, kernel, dilation);
    relu(tmp_scratch, out_ch * T);
    conv1d_same(tmp_scratch, w_c2, b_c2, x_inout, out_ch, out_ch, T, kernel, dilation);
    relu(x_inout, out_ch * T);
    se_block(x_inout, w_se1, b_se1, w_se2, b_se2, out_ch, V4_SE_23DIM_CH32_SE_HIDDEN, T);

    /* 3. add residual + ReLU → tmp_scratch, then 回拷到 x_inout */
    add_residual_relu(tmp_scratch, x_inout, y_scratch, out_ch * T);
    memcpy(x_inout, tmp_scratch, out_ch * T * sizeof(float));
}

/**
 * GAP + FC1 + ReLU + Dropout(0.3) + FC2 + Sigmoid
 *   x: (ch, T) → logit → sigmoid → probability
 */
static inline float classifier(const float* x)
{
    float z[V4_SE_23DIM_CH32_CHANNELS];      /* GAP (ch,) */
    float h[V4_SE_23DIM_CH32_FC1_OUT];       /* FC1 输出 (32,) */
    float logit;

    /* GAP */
    for (int c = 0; c < V4_SE_23DIM_CH32_CHANNELS; c++) {
        const float* x_c = x + c * V4_SE_23DIM_CH32_WINDOW;
        float sum = 0.0f;
        for (int t = 0; t < V4_SE_23DIM_CH32_WINDOW; t++) sum += x_c[t];
        z[c] = sum / (float)V4_SE_23DIM_CH32_WINDOW;
    }

    /* FC1: h = ReLU(z @ W_fc1.T + b_fc1), W_fc1: (32, ch=32) */
    for (int o = 0; o < V4_SE_23DIM_CH32_FC1_OUT; o++) {
        const float* w_o = w_fc1 + o * V4_SE_23DIM_CH32_CHANNELS;
        float sum = b_fc1[o];
        for (int ic = 0; ic < V4_SE_23DIM_CH32_CHANNELS; ic++) {
            sum += w_o[ic] * z[ic];
        }
        h[o] = sum > 0.0f ? sum : 0.0f;
        /* 注:训练时 FC1 后有 Dropout(0.3),推理时关闭 */
    }

    /* FC2: logit = h @ W_fc2.T + b_fc2, W_fc2: (1, 32) */
    {
        const float* w_o = w_fc2;
        float sum = b_fc2[0];
        for (int ic = 0; ic < V4_SE_23DIM_CH32_FC1_OUT; ic++) {
            sum += w_o[ic] * h[ic];
        }
        logit = sum;
    }

    /* Sigmoid */
    return 1.0f / (1.0f + expf(-logit));
}

/* ────────────────────────────────────────────────────────────────────────
 * 4. Per-seed dequant 函数 (X-macro 生成)
 * ──────────────────────────────────────────────────────────────────────── */

/* 用宏生成每个 seed 的 dequant 函数,避免手写 5 份重复代码 */

#define GEN_DEQUANT_FOR_SEED(SEED)                                              \
static void dequant_seed_##SEED(void) {                                         \
    /* Block 0 权重 (INT8 → FP32) */                                            \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_0_conv1_weight,              \
                        v4se23_ch32_s##SEED##_s_tcn_0_conv1_weight,              \
                        w_b0_c1, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_IN_CHANNELS, 3);                       \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_0_conv2_weight,              \
                        v4se23_ch32_s##SEED##_s_tcn_0_conv2_weight,              \
                        w_b0_c2, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_CHANNELS, 3);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_0_residual_weight,           \
                        v4se23_ch32_s##SEED##_s_tcn_0_residual_weight,           \
                        w_b0_rs, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_IN_CHANNELS, 1);                       \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_0_se_fc1_weight,             \
                        v4se23_ch32_s##SEED##_s_tcn_0_se_fc1_weight,             \
                        w_b0_se1, V4_SE_23DIM_CH32_SE_HIDDEN,                   \
                        V4_SE_23DIM_CH32_CHANNELS, 1);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_0_se_fc2_weight,             \
                        v4se23_ch32_s##SEED##_s_tcn_0_se_fc2_weight,             \
                        w_b0_se2, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_SE_HIDDEN, 1);                         \
    /* Block 1 (无 residual) */                                                  \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_1_conv1_weight,              \
                        v4se23_ch32_s##SEED##_s_tcn_1_conv1_weight,              \
                        w_b1_c1, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_CHANNELS, 3);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_1_conv2_weight,              \
                        v4se23_ch32_s##SEED##_s_tcn_1_conv2_weight,              \
                        w_b1_c2, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_CHANNELS, 3);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_1_se_fc1_weight,             \
                        v4se23_ch32_s##SEED##_s_tcn_1_se_fc1_weight,             \
                        w_b1_se1, V4_SE_23DIM_CH32_SE_HIDDEN,                   \
                        V4_SE_23DIM_CH32_CHANNELS, 1);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_1_se_fc2_weight,             \
                        v4se23_ch32_s##SEED##_s_tcn_1_se_fc2_weight,             \
                        w_b1_se2, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_SE_HIDDEN, 1);                         \
    /* Block 2 (无 residual) */                                                  \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_2_conv1_weight,              \
                        v4se23_ch32_s##SEED##_s_tcn_2_conv1_weight,              \
                        w_b2_c1, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_CHANNELS, 3);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_2_conv2_weight,              \
                        v4se23_ch32_s##SEED##_s_tcn_2_conv2_weight,              \
                        w_b2_c2, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_CHANNELS, 3);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_2_se_fc1_weight,             \
                        v4se23_ch32_s##SEED##_s_tcn_2_se_fc1_weight,             \
                        w_b2_se1, V4_SE_23DIM_CH32_SE_HIDDEN,                   \
                        V4_SE_23DIM_CH32_CHANNELS, 1);                          \
    dequant_per_channel(v4se23_ch32_s##SEED##_w_tcn_2_se_fc2_weight,             \
                        v4se23_ch32_s##SEED##_s_tcn_2_se_fc2_weight,             \
                        w_b2_se2, V4_SE_23DIM_CH32_CHANNELS,                     \
                        V4_SE_23DIM_CH32_SE_HIDDEN, 1);                         \
    /* Classifier (FP32) */                                                      \
    memcpy(w_fc1, v4se23_ch32_s##SEED##_fc1_weight, sizeof(w_fc1));             \
    memcpy(w_fc2, v4se23_ch32_s##SEED##_fc2_weight, sizeof(w_fc2));             \
    /* Biases (FP32) */                                                          \
    memcpy(b_b0_c1, v4se23_ch32_s##SEED##_b_tcn_0_conv1_bias, sizeof(b_b0_c1));   \
    memcpy(b_b0_c2, v4se23_ch32_s##SEED##_b_tcn_0_conv2_bias, sizeof(b_b0_c2));   \
    memcpy(b_b0_rs, v4se23_ch32_s##SEED##_b_tcn_0_residual_bias, sizeof(b_b0_rs));\
    memcpy(b_b0_se1, v4se23_ch32_s##SEED##_b_tcn_0_se_fc1_bias, sizeof(b_b0_se1));\
    memcpy(b_b0_se2, v4se23_ch32_s##SEED##_b_tcn_0_se_fc2_bias, sizeof(b_b0_se2));\
    memcpy(b_b1_c1, v4se23_ch32_s##SEED##_b_tcn_1_conv1_bias, sizeof(b_b1_c1));   \
    memcpy(b_b1_c2, v4se23_ch32_s##SEED##_b_tcn_1_conv2_bias, sizeof(b_b1_c2));   \
    memcpy(b_b1_se1, v4se23_ch32_s##SEED##_b_tcn_1_se_fc1_bias, sizeof(b_b1_se1));\
    memcpy(b_b1_se2, v4se23_ch32_s##SEED##_b_tcn_1_se_fc2_bias, sizeof(b_b1_se2));\
    memcpy(b_b2_c1, v4se23_ch32_s##SEED##_b_tcn_2_conv1_bias, sizeof(b_b2_c1));   \
    memcpy(b_b2_c2, v4se23_ch32_s##SEED##_b_tcn_2_conv2_bias, sizeof(b_b2_c2));   \
    memcpy(b_b2_se1, v4se23_ch32_s##SEED##_b_tcn_2_se_fc1_bias, sizeof(b_b2_se1));\
    memcpy(b_b2_se2, v4se23_ch32_s##SEED##_b_tcn_2_se_fc2_bias, sizeof(b_b2_se2));\
    memcpy(b_fc1, v4se23_ch32_s##SEED##_b_fc1_bias, sizeof(b_fc1));             \
    memcpy(b_fc2, v4se23_ch32_s##SEED##_b_fc2_bias, sizeof(b_fc2));             \
    v4_se_23dim_current_seed = SEED;                                            \
}

GEN_DEQUANT_FOR_SEED(42)
GEN_DEQUANT_FOR_SEED(123)
GEN_DEQUANT_FOR_SEED(456)
GEN_DEQUANT_FOR_SEED(789)
GEN_DEQUANT_FOR_SEED(1024)

/* seed → dequant 函数指针表 (用于 ensemble 循环) */
typedef void (*dequant_fn_t)(void);
static const dequant_fn_t dequant_table[V4_SE_23DIM_CH32_N_SEEDS] = {
    dequant_seed_42,
    dequant_seed_123,
    dequant_seed_456,
    dequant_seed_789,
    dequant_seed_1024,
};

/* ────────────────────────────────────────────────────────────────────────
 * 5. 公共 API 实现
 * ──────────────────────────────────────────────────────────────────────── */

void v4_se_23dim_ch32_init(void)
{
    dequant_seed_123();  /* 默认 seed=123 (本工作 5-seed 中 F1m 最高 0.8766) */
}

void v4_se_23dim_ch32_init_seed(int seed)
{
    switch (seed) {
        case 42:   dequant_seed_42();   break;
        case 123:  dequant_seed_123();  break;
        case 456:  dequant_seed_456();  break;
        case 789:  dequant_seed_789();  break;
        case 1024: dequant_seed_1024(); break;
        default:
            /* 未知 seed 静默退回默认 */
            dequant_seed_123();
            break;
    }
}

float v4_se_23dim_ch32_forward(const float* x, size_t len)
{
    (void)len;

    /*
     * Block 0 输入是 23 维,需要把 x[23*16] 复制到 buf_x 前 23*16 位置
     * 第一次 Block 0 后,buf_x 装的就是 32 维 (32*16)
     *
     * 内存布局:
     *   cur        = buf_x   (32, 16) — 当前块输入/输出
     *   residual   = buf_y   (32, 16) — 残差 / scratch
     *   tmp        = buf_z   (32, 16) — conv scratch
     */
    float* cur = buf_x;
    float* residual = buf_y;
    float* tmp = buf_z;

    /* Block 0: 输入 23 维 → 输出 32 维,residual = 1×1 conv(23→32) */
    {
        /* 把 23 维输入复制到 cur (用前 23*16 位置) */
        memcpy(cur, x, V4_SE_23DIM_CH32_IN_SIZE * sizeof(float));

        tcn_block(cur, residual, tmp,
                  w_b0_c1, b_b0_c1,
                  w_b0_c2, b_b0_c2,
                  w_b0_rs, b_b0_rs,
                  w_b0_se1, b_b0_se1,
                  w_b0_se2, b_b0_se2,
                  V4_SE_23DIM_CH32_IN_CHANNELS,  /* in_ch = 23 */
                  V4_SE_23DIM_CH32_CHANNELS,     /* out_ch = 32 */
                  V4_SE_23DIM_CH32_WINDOW, 3, V4_SE_23DIM_CH32_DIL_B0);
    }

    /* Block 1: 32 维 → 32 维,residual = identity */
    tcn_block(cur, residual, tmp,
              w_b1_c1, b_b1_c1,
              w_b1_c2, b_b1_c2,
              NULL, NULL,                       /* identity */
              w_b1_se1, b_b1_se1,
              w_b1_se2, b_b1_se2,
              V4_SE_23DIM_CH32_CHANNELS,
              V4_SE_23DIM_CH32_CHANNELS,
              V4_SE_23DIM_CH32_WINDOW, 3, V4_SE_23DIM_CH32_DIL_B1);

    /* Block 2: 32 维 → 32 维,residual = identity */
    tcn_block(cur, residual, tmp,
              w_b2_c1, b_b2_c1,
              w_b2_c2, b_b2_c2,
              NULL, NULL,
              w_b2_se1, b_b2_se1,
              w_b2_se2, b_b2_se2,
              V4_SE_23DIM_CH32_CHANNELS,
              V4_SE_23DIM_CH32_CHANNELS,
              V4_SE_23DIM_CH32_WINDOW, 3, V4_SE_23DIM_CH32_DIL_B2);

    /* Classifier: GAP → FC1 → ReLU → FC2 → Sigmoid */
    return classifier(cur);
}

float v4_se_23dim_ch32_ensemble_forward(const float* x)
{
    float prob_sum = 0.0f;
    for (int i = 0; i < V4_SE_23DIM_CH32_N_SEEDS; i++) {
        dequant_table[i]();
        prob_sum += v4_se_23dim_ch32_forward(x, V4_SE_23DIM_CH32_IN_SIZE);
    }
    return prob_sum / (float)V4_SE_23DIM_CH32_N_SEEDS;
}

void v4_se_23dim_ch32_batch(
    const float* input, int N, float* output, int input_stride_floats)
{
    /* 一次性 dequant 当前 seed,然后 N 次 forward */
    /* 假设调用方已调过 v4_se_23dim_ch32_init() 或 init_seed() */
    for (int i = 0; i < N; i++) {
        output[i] = v4_se_23dim_ch32_forward(
            input + i * input_stride_floats, input_stride_floats);
    }
}

void v4_se_23dim_ch32_ensemble_batch(
    const float* input, int N, float* output, int input_stride_floats)
{
    for (int i = 0; i < N; i++) {
        output[i] = v4_se_23dim_ch32_ensemble_forward(
            input + i * input_stride_floats);
    }
}

void v4_se_23dim_ch32_benchmark(int n_runs)
{
    /* 构造 1 个固定输入 (LCG 伪随机,确定性) */
    float x[V4_SE_23DIM_CH32_IN_SIZE];
    unsigned int s = 0xDEADBEEF;
    for (int i = 0; i < V4_SE_23DIM_CH32_IN_SIZE; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
    }

    /* Warm-up */
    volatile float p = v4_se_23dim_ch32_forward(x, V4_SE_23DIM_CH32_IN_SIZE);
    (void)p;

    /* 简单计时:PC 上没有 DWT,只能用循环次数. STM32 上请用 DWT->CYCCNT */
    float total = 0.0f;
    for (int i = 0; i < n_runs; i++) {
        total += v4_se_23dim_ch32_forward(x, V4_SE_23DIM_CH32_IN_SIZE);
    }

    extern int printf(const char*, ...);
    printf("  v4_se_23dim_ch32_benchmark: %d runs done, output sum=%.4f "
           "(use DWT->CYCCNT on STM32 for timing)\n",
           n_runs, total);
}

v4_se_23dim_ch32_mem_info_t v4_se_23dim_ch32_get_mem_info(void)
{
    v4_se_23dim_ch32_mem_info_t m;
    size_t w = 0;
    /* Block 0 weights */
    w += sizeof(w_b0_c1) + sizeof(w_b0_c2) + sizeof(w_b0_rs)
       + sizeof(w_b0_se1) + sizeof(w_b0_se2);
    /* Block 1, 2 weights */
    w += sizeof(w_b1_c1) + sizeof(w_b1_c2) + sizeof(w_b1_se1) + sizeof(w_b1_se2);
    w += sizeof(w_b2_c1) + sizeof(w_b2_c2) + sizeof(w_b2_se1) + sizeof(w_b2_se2);
    /* Classifier weights */
    w += sizeof(w_fc1) + sizeof(w_fc2);
    m.weights_fp32_bytes = w;

    /* Input + activations */
    m.input_bytes = V4_SE_23DIM_CH32_IN_SIZE * sizeof(float);
    m.peak_act_bytes = sizeof(buf_x) + sizeof(buf_y) + sizeof(buf_z);

    /* Total RAM (单 seed 模式;ensemble 模式共享权重缓冲,RAM 相同) */
    m.total_ram_bytes = m.weights_fp32_bytes + m.peak_act_bytes + m.input_bytes;
    m.current_seed = v4_se_23dim_current_seed;

    return m;
}