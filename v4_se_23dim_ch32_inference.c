/* ============================================================================
 * v4_se_23dim_ch32_inference.c - TCN+SE ch=32 (23-dim) MCU Inference Implementation
 *
 * Hybrid INT8 推理: 权重 INT8, 运行时 dequantize 到 FP32
 * 激活全 FP32, Conv1d 用手写 im2col + matmul (无 CMSIS-NN 依赖)
 *
 * 性能估算 (Cortex-M4 @ 168 MHz, FPU):
 *   - 加载 + dequantize: ~1 ms
 *   - 前向推理:  ~0.3 ms
 *   - 合计:      ~1.3 ms / sample
 *   - RAM:       ~30 KB (FP32 中间激活)
 *   - Flash:     ~5 KB (INT8 权重)
 *
 * 注: 本文件是参考实现, 实际部署可根据 MCU 优化 (e.g. CMSIS-NN, 定点化)
 * ============================================================================
 */
#include "v4_se_23dim_ch32_inference.h"
// 选择 seed=123 (本工作 5-seed 中 F1m 最高 0.8766):
#include "model_v4_se_23dim_ch32_hybrid_s123.h"
// 替换 seed 只需改 include 即可

// 内部 helper: 1D conv with bias (no padding assumption here)
static void conv1d_fp32(
    const float *input, int in_ch, int in_len,
    const int8_t *w_q, const float *w_scale, const float *bias,
    int out_ch, int kernel, int dilation,
    float *output
) {
    int pad = (kernel - 1) * dilation / 2;
    for (int oc = 0; oc < out_ch; ++oc) {
        // dequantize this output channel's weights
        float w[256];  // max in_ch * kernel
        const int8_t *w_oc = w_q + oc * (in_ch * kernel);
        for (int i = 0; i < in_ch * kernel; ++i) {
            w[i] = ((float)w_oc[i]) * w_scale[oc];
        }
        for (int t = 0; t < in_len; ++t) {
            float s = bias[oc];
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < kernel; ++k) {
                    int t_in = t - pad + k * dilation;
                    if (t_in >= 0 && t_in < in_len) {
                        s += input[ic * in_len + t_in] * w[ic * kernel + k];
                    }
                }
            }
            output[oc * in_len + t] = s;
        }
    }
}

static void linear_fp32(
    const float *input, int in_dim,
    const float *weight, const float *bias,
    int out_dim,
    float *output
) {
    for (int o = 0; o < out_dim; ++o) {
        float s = bias[o];
        for (int i = 0; i < in_dim; ++i) {
            s += input[i] * weight[o * in_dim + i];
        }
        output[o] = s;
    }
}

static void se_block(
    const float *input, int channels, int seq_len,
    const int8_t *fc1_w, const float *fc1_s, const float *fc1_b,
    const int8_t *fc2_w, const float *fc2_s, const float *fc2_b
) {
    // GAP
    float gap[32] = {0};
    for (int c = 0; c < channels; ++c) {
        float sum = 0;
        for (int t = 0; t < seq_len; ++t) sum += input[c * seq_len + t];
        gap[c] = sum / seq_len;
    }
    // FC1: channels -> channels/8
    int hidden = max(channels / 8, 4);
    float fc1_out[4] = {0};
    linear_fp32(gap, channels, (const float*)fc1_w, fc1_b, hidden, fc1_out);
    for (int i = 0; i < hidden; ++i) fc1_out[i] = fmaxf(0, fc1_out[i]);
    // FC2: hidden -> channels
    float fc2_out[32] = {0};
    linear_fp32(fc1_out, hidden, (const float*)fc2_w, fc2_b, channels, fc2_out);
    for (int c = 0; c < channels; ++c) {
        float sig = 1.0f / (1.0f + expf(-fc2_out[c]));
        for (int t = 0; t < seq_len; ++t) {
            input[c * seq_len + t] *= sig;
        }
    }
}

void tcn_v4_se_23dim_ch32_inference(const float input[23][16], float *output) {
    // 把 input (23,16) reshape 到一维 buffer
    static float buf[32 * 16];
    // block 0
    float *t0_in = (float*)input;
    float t0_conv1[32 * 16], t0_conv2[32 * 16];
    conv1d_fp32(t0_in, 23, 16,
                v4se23_ch32_s123_w_tcn_0_conv1_weight,
                v4se23_ch32_s123_s_tcn_0_conv1_weight,
                v4se23_ch32_s123_b_tcn_0_conv1_bias,
                32, 3, 1, t0_conv1);
    for (int i = 0; i < 32*16; ++i) t0_conv1[i] = fmaxf(0, t0_conv1[i]);
    conv1d_fp32(t0_conv1, 32, 16,
                v4se23_ch32_s123_w_tcn_0_conv2_weight,
                v4se23_ch32_s123_s_tcn_0_conv2_weight,
                v4se23_ch32_s123_b_tcn_0_conv2_bias,
                32, 3, 1, t0_conv2);
    for (int i = 0; i < 32*16; ++i) t0_conv2[i] = fmaxf(0, t0_conv2[i]);
    se_block(t0_conv2, 32, 16,
             v4se23_ch32_s123_w_tcn_0_se_fc1_weight,
             v4se23_ch32_s123_s_tcn_0_se_fc1_weight,
             v4se23_ch32_s123_b_tcn_0_se_fc1_bias,
             v4se23_ch32_s123_w_tcn_0_se_fc2_weight,
             v4se23_ch32_s123_s_tcn_0_se_fc2_weight,
             v4se23_ch32_s123_b_tcn_0_se_fc2_bias);
    // residual
    float t0_res[32 * 16];
    conv1d_fp32(t0_in, 23, 16,
                v4se23_ch32_s123_w_tcn_0_residual_weight,
                v4se23_ch32_s123_s_tcn_0_residual_weight,
                v4se23_ch32_s123_b_tcn_0_residual_bias,
                32, 1, 1, t0_res);
    for (int i = 0; i < 32*16; ++i) t0_conv2[i] = fmaxf(0, t0_conv2[i] + t0_res[i]);
    // block 1, 2 省略 (结构相同, d=2, 4)
    // ... (实际部署需补全)
    // GAP
    float gap[32] = {0};
    for (int c = 0; c < 32; ++c) {
        for (int t = 0; t < 16; ++t) gap[c] += t0_conv2[c*16+t];
        gap[c] /= 16;
    }
    // FC1 -> ReLU -> Dropout(0.3) -> FC2
    float fc1[32];
    linear_fp32(gap, 32, v4se23_ch32_s123_fc1_weight, v4se23_ch32_s123_b_fc1_bias, 32, fc1);
    for (int i = 0; i < 32; ++i) fc1[i] = fmaxf(0, fc1[i]);
    // dropout(0.3) 在 inference 时关闭
    float fc2[1];
    linear_fp32(fc1, 32, v4se23_ch32_s123_fc2_weight, v4se23_ch32_s123_b_fc2_bias, 1, fc2);
    *output = 1.0f / (1.0f + expf(-fc2[0]));
}

void tcn_v4_se_23dim_ch32_inference_batch(
    const float *input, int N, float *output, int input_stride_floats
) {
    for (int i = 0; i < N; ++i) {
        tcn_v4_se_23dim_ch32_inference(
            (const float(*)[16])(input + i * input_stride_floats),
            output + i
        );
    }
}
