/* ============================================================================
 * ch32_inference.c  —  TCN+SE ch=32 MCU Inference Implementation
 * ============================================================================
 *
 * 架构 (与 PyTorch train_quantize_3ch.py ch=32 完全一致):
 *   x (19, 16)
 *     ↓
 *   Block 0 (d=1): conv1(19→32, k=3) → ReLU → conv2(32→32, k=3) → ReLU
 *                  → SE → + Residual(1×1: 19→32) → ReLU
 *     ↓ (32, 16)
 *   Block 1 (d=2): 同上但 residual 是 Identity
 *     ↓ (32, 16)
 *   Block 2 (d=4): 同上但 residual 是 Identity
 *     ↓ (32, 16)
 *   GAP → FC1(32→32) + ReLU + Dropout(0.3) → FC2(32→1) → Sigmoid
 *     ↓
 *   probability ∈ [0, 1]
 *
 * 数据 layout (与 PyTorch Conv1d 一致):
 *   shape (channels, time_steps), row-major
 *   例: input[ic*16 + t]  = 第 ic 个特征在 t 时刻的值
 *
 * Conv1d "same" padding 计算:
 *   pad_left  = (kernel - 1) * dilation / 2
 *   pad_right  = kernel - 1 - pad_left  (保证左右对称填 0)
 *   对边界 t_out: 实际卷积区间 [t_out - pad_left, t_out + kernel - 1 - pad_left]
 *                  dilation 步长,左越界或右越界的输入值视为 0
 *
 * 关键优化 (MCU 友好):
 *   - 静态 RAM 分配,无 malloc
 *   - 启动时 dequant 一次性,运行期纯 FP32
 *   - 循环顺序优化 cache 友好 (oc 在最内层 → 权重连续访问)
 *   - 内联辅助函数避免调用开销
 *
 * 进一步优化选项:
 *   1. 用 CMSIS-DSP 替换 conv1d: arm_conv_f32 / arm_mat_mult_f32
 *      (代码 +10 KB Flash,推理可从 ~0.9 ms 降到 ~0.3 ms @ M4 168MHz)
 *   2. 用 SIMD intrinsics 手写 MAC 循环 (-O3 -ffast-math)
 *   3. "Lite 模式":运行期按层 dequant (省 ~52 KB RAM,见 tcn_v4_lite_mode)
 * ============================================================================
 */

#include "ch32_inference.h"
#include "model_ch32_hybrid.h"
#include <string.h>
#include <math.h>

/* ────────────────────────────────────────────────────────────────────────
 * 1. RAM 缓冲:dequant 后的 FP32 权重 (启动时由 ch32_init 填充)
 * ──────────────────────────────────────────────────────────────────────── */

/* Block 0: conv1 (32,19,3) + conv2 (32,32,3) + residual (32,19,1) + SE(fc1:4,32 + fc2:32,4) */
static float w_b0_c1[CH32_CH * CH32_IN_CHANNELS * 3];  /* 32*19*3 = 1824 */
static float w_b0_c2[CH32_CH * CH32_CH       * 3];      /* 32*32*3 = 3072 */
static float w_b0_rs[CH32_CH * CH32_IN_CHANNELS * 1];   /* 32*19*1 = 608 */
static float w_b0_se1[CH32_SE_HIDDEN * CH32_CH];        /* 4*32 = 128 */
static float w_b0_se2[CH32_CH * CH32_SE_HIDDEN];        /* 32*4 = 128 */

/* Block 1: 同上但 residual 是 identity (无需 1×1 卷积权重) */
static float w_b1_c1[CH32_CH * CH32_CH * 3];            /* 3072 */
static float w_b1_c2[CH32_CH * CH32_CH * 3];            /* 3072 */
static float w_b1_se1[CH32_SE_HIDDEN * CH32_CH];        /* 128 */
static float w_b1_se2[CH32_CH * CH32_SE_HIDDEN];        /* 128 */

/* Block 2 */
static float w_b2_c1[CH32_CH * CH32_CH * 3];            /* 3072 */
static float w_b2_c2[CH32_CH * CH32_CH * 3];            /* 3072 */
static float w_b2_se1[CH32_SE_HIDDEN * CH32_CH];        /* 128 */
static float w_b2_se2[CH32_CH * CH32_SE_HIDDEN];        /* 128 */

/* Classifier: fc1 (32,32) + fc2 (1,32) */
static float w_fc1[CH32_FC1_OUT * CH32_CH];             /* 32*32 = 1024 */
static float w_fc2[CH32_FC2_OUT * CH32_FC1_OUT];        /* 1*32  = 32 */

/* 全部 bias (FP32) */
static float b_b0_c1[CH32_CH];
static float b_b0_c2[CH32_CH];
static float b_b0_rs[CH32_CH];
static float b_b0_se1[CH32_SE_HIDDEN];
static float b_b0_se2[CH32_CH];

static float b_b1_c1[CH32_CH];
static float b_b1_c2[CH32_CH];
static float b_b1_se1[CH32_SE_HIDDEN];
static float b_b1_se2[CH32_CH];

static float b_b2_c1[CH32_CH];
static float b_b2_c2[CH32_CH];
static float b_b2_se1[CH32_SE_HIDDEN];
static float b_b2_se2[CH32_CH];

static float b_fc1[CH32_FC1_OUT];
static float b_fc2[CH32_FC2_OUT];

/* 激活缓冲 (运行期使用) */
static float buf_x[CH32_FEAT_SIZE];   /* (32, 16) 当前块输入 */
static float buf_y[CH32_FEAT_SIZE];   /* (32, 16) scratch */
static float buf_z[CH32_FEAT_SIZE];   /* (32, 16) scratch / SE GAP 输出前 */

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
 *
 *   padding: pad_left = (k-1)*d/2, pad_right = k-1 - pad_left
 *   t_out 对应的 t_in 区间: [t_out*dilation - pad_left*dilation, t_out*dilation + (k-1)*dilation - pad_left*dilation]
 *   等价于: t_in_实际 = t_out*dilation - pad_left*dilation + i*dilation, i ∈ [0, k)
 *         若 t_in_实际 < 0 或 >= T,输入视为 0
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
 *   x : (ch, T)
 *   1. Global Avg Pool: z[oc] = mean(x[oc, :])
 *   2. FC1: a = ReLU(z @ W1 + b1), shape (hidden,)
 *   3. FC2: s = sigmoid(a @ W2 + b2), shape (ch,)
 *   4. Scale: x[oc, t] *= s[oc]
 *
 * W1: (hidden, ch), W2: (ch, hidden)
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
 *   y = ReLU(x + res)
 *   x, res, y: (ch, T)
 */
static inline void add_residual_relu(float* y, const float* x, const float* res, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i] + res[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/**
 * 1 个 TCN Block (通用版本,通过 use_residual_conv 区分 Block 0 / 1,2)
 */
static inline void tcn_block(
    float* x, float* y, float* scratch,         /* 缓冲,每个 (ch, T) */
    const float* w_c1, const float* b_c1,        /* conv1: (ch, in_ch, k)  Block 0 in_ch=19, 1,2 in_ch=32 */
    const float* w_c2, const float* b_c2,        /* conv2: (ch, ch, k) */
    const float* w_rs, const float* b_rs,        /* residual 1×1: (ch, in_ch, 1),identity 时为 NULL */
    const float* w_se1, const float* b_se1,      /* SE fc1: (hidden, ch) */
    const float* w_se2, const float* b_se2,      /* SE fc2: (ch, hidden) */
    int in_ch, int out_ch, int T, int kernel, int dilation)
{
    /* 1. residual (1×1 conv 或 identity) */
    if (w_rs != NULL) {
        conv1d_1x1(x, w_rs, b_rs, y, in_ch, out_ch, T);
    } else {
        memcpy(y, x, out_ch * T * sizeof(float));
    }

    /* 2. main path: conv1 → ReLU → conv2 → ReLU → SE */
    conv1d_same(x, w_c1, b_c1, scratch, in_ch, out_ch, T, kernel, dilation);
    relu(scratch, out_ch * T);
    conv1d_same(scratch, w_c2, b_c2, x, out_ch, out_ch, T, kernel, dilation);
    relu(x, out_ch * T);
    se_block(x, w_se1, b_se1, w_se2, b_se2, out_ch, CH32_SE_HIDDEN, T);

    /* 3. add residual + ReLU (输出到 scratch,然后回拷到 x) */
    add_residual_relu(scratch, x, y, out_ch * T);
    memcpy(x, scratch, out_ch * T * sizeof(float));
}

/**
 * GAP + FC1 + ReLU + Dropout(0.3) + FC2 + Sigmoid
 *   x: (ch, T) → logit → sigmoid
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

    /* FC1: h = ReLU(z @ W_fc1.T + b_fc1), W_fc1: (32, ch=32) */
    for (int o = 0; o < CH32_FC1_OUT; o++) {
        const float* w_o = w_fc1 + o * CH32_CH;
        float sum = b_fc1[o];
        for (int ic = 0; ic < CH32_CH; ic++) {
            sum += w_o[ic] * z[ic];
        }
        h[o] = sum > 0.0f ? sum : 0.0f;
        /* 注:训练时 FC1 后有 Dropout(0.3),推理时关闭,无操作 */
    }

    /* FC2: logit = h @ W_fc2.T + b_fc2, W_fc2: (1, 32) */
    {
        const float* w_o = w_fc2;  /* shape (1, 32) */
        float sum = b_fc2[0];
        for (int ic = 0; ic < CH32_FC1_OUT; ic++) {
            sum += w_o[ic] * h[ic];
        }
        logit = sum;
    }

    /* Sigmoid */
    return 1.0f / (1.0f + expf(-logit));
}

/* ────────────────────────────────────────────────────────────────────────
 * 3. 公共 API 实现
 * ──────────────────────────────────────────────────────────────────────── */

void ch32_init(void)
{
    /* Block 0 */
    dequant_per_channel(ch32_w_tcn_0_conv1_weight,    ch32_s_tcn_0_conv1_weight,
                        w_b0_c1, CH32_CH, CH32_IN_CHANNELS, 3);
    dequant_per_channel(ch32_w_tcn_0_conv2_weight,    ch32_s_tcn_0_conv2_weight,
                        w_b0_c2, CH32_CH, CH32_CH, 3);
    dequant_per_channel(ch32_w_tcn_0_residual_weight, ch32_s_tcn_0_residual_weight,
                        w_b0_rs, CH32_CH, CH32_IN_CHANNELS, 1);
    dequant_per_channel(ch32_w_tcn_0_se_fc1_weight,   ch32_s_tcn_0_se_fc1_weight,
                        w_b0_se1, CH32_SE_HIDDEN, CH32_CH, 1);
    dequant_per_channel(ch32_w_tcn_0_se_fc2_weight,   ch32_s_tcn_0_se_fc2_weight,
                        w_b0_se2, CH32_CH, CH32_SE_HIDDEN, 1);

    /* Block 1 (无 residual conv) */
    dequant_per_channel(ch32_w_tcn_1_conv1_weight, ch32_s_tcn_1_conv1_weight,
                        w_b1_c1, CH32_CH, CH32_CH, 3);
    dequant_per_channel(ch32_w_tcn_1_conv2_weight, ch32_s_tcn_1_conv2_weight,
                        w_b1_c2, CH32_CH, CH32_CH, 3);
    dequant_per_channel(ch32_w_tcn_1_se_fc1_weight, ch32_s_tcn_1_se_fc1_weight,
                        w_b1_se1, CH32_SE_HIDDEN, CH32_CH, 1);
    dequant_per_channel(ch32_w_tcn_1_se_fc2_weight, ch32_s_tcn_1_se_fc2_weight,
                        w_b1_se2, CH32_CH, CH32_SE_HIDDEN, 1);

    /* Block 2 */
    dequant_per_channel(ch32_w_tcn_2_conv1_weight, ch32_s_tcn_2_conv1_weight,
                        w_b2_c1, CH32_CH, CH32_CH, 3);
    dequant_per_channel(ch32_w_tcn_2_conv2_weight, ch32_s_tcn_2_conv2_weight,
                        w_b2_c2, CH32_CH, CH32_CH, 3);
    dequant_per_channel(ch32_w_tcn_2_se_fc1_weight, ch32_s_tcn_2_se_fc1_weight,
                        w_b2_se1, CH32_SE_HIDDEN, CH32_CH, 1);
    dequant_per_channel(ch32_w_tcn_2_se_fc2_weight, ch32_s_tcn_2_se_fc2_weight,
                        w_b2_se2, CH32_CH, CH32_SE_HIDDEN, 1);

    /* Classifier */
    dequant_per_channel(ch32_w_fc1_weight, ch32_s_fc1_weight,
                        w_fc1, CH32_FC1_OUT, CH32_CH, 1);
    dequant_per_channel(ch32_w_fc2_weight, ch32_s_fc2_weight,
                        w_fc2, CH32_FC2_OUT, CH32_FC1_OUT, 1);

    /* Bias 直接拷贝 (FP32) */
    memcpy(b_b0_c1,  ch32_b_tcn_0_conv1_bias,  sizeof(b_b0_c1));
    memcpy(b_b0_c2,  ch32_b_tcn_0_conv2_bias,  sizeof(b_b0_c2));
    memcpy(b_b0_rs,  ch32_b_tcn_0_residual_bias, sizeof(b_b0_rs));
    memcpy(b_b0_se1, ch32_b_tcn_0_se_fc1_bias, sizeof(b_b0_se1));
    memcpy(b_b0_se2, ch32_b_tcn_0_se_fc2_bias, sizeof(b_b0_se2));

    memcpy(b_b1_c1,  ch32_b_tcn_1_conv1_bias,  sizeof(b_b1_c1));
    memcpy(b_b1_c2,  ch32_b_tcn_1_conv2_bias,  sizeof(b_b1_c2));
    memcpy(b_b1_se1, ch32_b_tcn_1_se_fc1_bias, sizeof(b_b1_se1));
    memcpy(b_b1_se2, ch32_b_tcn_1_se_fc2_bias, sizeof(b_b1_se2));

    memcpy(b_b2_c1,  ch32_b_tcn_2_conv1_bias,  sizeof(b_b2_c1));
    memcpy(b_b2_c2,  ch32_b_tcn_2_conv2_bias,  sizeof(b_b2_c2));
    memcpy(b_b2_se1, ch32_b_tcn_2_se_fc1_bias, sizeof(b_b2_se1));
    memcpy(b_b2_se2, ch32_b_tcn_2_se_fc2_bias, sizeof(b_b2_se2));

    memcpy(b_fc1, ch32_b_fc1_bias, sizeof(b_fc1));
    memcpy(b_fc2, ch32_b_fc2_bias, sizeof(b_fc2));
}

float ch32_forward(const float* x, size_t len)
{
    (void)len;

    /* buf_x 是当前 (32, 16),从 (19, 16) → (32, 16) 第一次 Block 0 后转换 */
    /* 这里直接把输入复制到 buf_x,Block 0 会做 19→32 的残差 */
    /* 注意:Block 0 输入是 19 维,不能直接存到 32 维的 buf_x.
       我们用 buf_x 存当前 (32, 16) 块输出,首次 Block 0 前用 buf_x 临时存 19 维输入 */
    /* 简单起见: 用 buf_y 暂存 19 维输入,Block 0 后写回 buf_x (32, 16) */
    const float* x_in = x;
    float* cur = buf_x;        /* 当前 (32, 16) 块输入/输出 */
    float* nxt = buf_y;        /* 临时 (32, 16) 块输出 */
    float* tmp = buf_z;        /* scratch (32, 16) */

    /* Block 0: 输入 19 维,输出 32 维
       先把 19 维输入放到一个能放下 32 维的位置. 用 buf_x 的前 19*T 部分 */
    /* 这里为了简化, 我们需要一个额外的 19 维入口. 用 nxt 作为 19 维临时区 */
    /* 但 nxt 大小是 32*16=512,够装 19*16=304,后面 Block 1,2 都用 32 维,无问题 */

    /* Block 0: cur (19,16) → nxt (32,16), residual = 1×1 conv(19→32) */
    {
        /* cur 当前不是 32 维,而是 19 维. 我们先准备一个 "19 维 cur"
           用 buf_x 的前 19*16 = 304 个 float 装输入 */
        /* 但前面已经声明 cur = buf_x,会冲突.
           解决: Block 0 用静态 scratch — 把 x_in 复制到 buf_x 前 304 个元素,
                 当作 "19 维输入". residual 用 1×1 conv 写到 nxt */
        memcpy(buf_x, x_in, CH32_IN_CHANNELS * CH32_WINDOW * sizeof(float));

        /* residual = 1×1 conv(19→32) → 写到 nxt */
        conv1d_1x1(buf_x, w_b0_rs, b_b0_rs, nxt,
                   CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW);

        /* main path: conv1(19→32) → ReLU → conv2(32→32) → ReLU → SE → 输出到 cur (32,16) */
        conv1d_same(buf_x, w_b0_c1, b_b0_c1, tmp,
                    CH32_IN_CHANNELS, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
        relu(tmp, CH32_CH * CH32_WINDOW);
        conv1d_same(tmp, w_b0_c2, b_b0_c2, cur,
                    CH32_CH, CH32_CH, CH32_WINDOW, 3, CH32_DIL_B0);
        relu(cur, CH32_CH * CH32_WINDOW);
        se_block(cur, w_b0_se1, b_b0_se1, w_b0_se2, b_b0_se2,
                 CH32_CH, CH32_SE_HIDDEN, CH32_WINDOW);

        /* add residual + ReLU → cur */
        add_residual_relu(cur, cur, nxt, CH32_CH * CH32_WINDOW);
    }

    /* Block 1: cur (32,16) → cur (32,16), residual = identity */
    {
        /* residual → nxt */
        memcpy(nxt, cur, CH32_FEAT_SIZE * sizeof(float));

        /* main path */
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
    }

    /* Block 2: cur (32,16) → cur (32,16), residual = identity */
    {
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
    }

    /* Classifier: GAP → FC1 → ReLU → FC2 → Sigmoid */
    return classifier(cur, w_fc1, b_fc1, w_fc2, b_fc2);
}

void ch32_benchmark(int n_runs)
{
    /* 构造 1 个固定输入 (用 LCG 伪随机,确定性) */
    float x[CH32_IN_SIZE];
    unsigned int s = 0xDEADBEEF;
    for (int i = 0; i < CH32_IN_SIZE; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
    }

    /* Warm-up */
    volatile float p = ch32_forward(x, CH32_IN_SIZE);
    (void)p;

    /* 简单计时:用 DWT cycle counter 需要 CMSIS,这里退化用循环次数 */
    /* 注: STM32 上请用 DWT->CYCCNT,见 main_stm32f407.c */
    float total = 0.0f;
    for (int i = 0; i < n_runs; i++) {
        total += ch32_forward(x, CH32_IN_SIZE);
    }

    /* 这里不打印时间 (无 DWT),只在 STM32 模板里打印.
       打印 sum 是为了防止编译器把循环优化掉 */
    extern int printf(const char*, ...);
    printf("  ch32_benchmark: %d runs done, output sum=%.4f (use DWT for timing)\n",
           n_runs, total);
}

ch32_mem_info_t ch32_get_mem_info(void)
{
    ch32_mem_info_t m;
    /* dequant 后 FP32 权重 */
    size_t w = 0;
    w += sizeof(w_b0_c1) + sizeof(w_b0_c2) + sizeof(w_b0_rs)
       + sizeof(w_b0_se1) + sizeof(w_b0_se2);
    w += sizeof(w_b1_c1) + sizeof(w_b1_c2) + sizeof(w_b1_se1) + sizeof(w_b1_se2);
    w += sizeof(w_b2_c1) + sizeof(w_b2_c2) + sizeof(w_b2_se1) + sizeof(w_b2_se2);
    w += sizeof(w_fc1) + sizeof(w_fc2);
    m.weights_fp32_bytes = w;

    /* 输入 */
    m.input_bytes = CH32_IN_SIZE * sizeof(float);

    /* 激活峰值: buf_x + buf_y + buf_z = 3 * 512 * 4 = 6144 字节 */
    /* 加上 classifier 临时 z/h: + 32*4 + 32*4 = 256, 实际 3 个 512 缓冲 6144 已够 */
    /* 实际峰值再加 SE 临时 a[4] 略 */
    m.peak_act_bytes = sizeof(buf_x) + sizeof(buf_y) + sizeof(buf_z);

    /* 总 RAM */
    m.total_ram_bytes = m.weights_fp32_bytes + m.peak_act_bytes + m.input_bytes;

    return m;
}