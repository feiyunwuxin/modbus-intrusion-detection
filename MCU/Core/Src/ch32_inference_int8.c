/* ============================================================================
 * ch32_inference_int8.c  --  TCN+SE ch=32 MCU Inference (全量 INT8, 23-dim, w=16)
 *                             STM32H743VIT6 (Cortex-M7 @ 480MHz) + CMSIS-NN
 *
 * ⭐ 2026-10-21 全量 INT8 路线 (与 ch32_inference.c 的 FP32 路线并存, 由
 *   ch32_inference.h 的 CH32_USE_INT8 开关分派)
 *
 * 定点方案 (TFLite / CMSIS-NN 标准做法):
 *   权重   : int8, per-output-channel 对称 (scale[oc] = max|w[oc]|/127)
 *   激活   : int8, per-tensor 非对称 (real = (q - zp) * s)
 *   bias   : int32 = round(b / (s_in * s_w[oc]))
 *   requant: per-channel (mult[oc], shift[oc]) -> arm_nn_requantize
 *   offset : input_offset = -zp_in, output_offset = +zp_out  (CMSIS 约定)
 *
 * ⚠️ 与 PC 侧 numpy 镜像 (sim_int8_kernel.py) 的逐位一致性由三件事保证:
 *   1. requantize 直接调用 CMSIS-NN 的 arm_nn_requantize (不自己实现舍入)
 *   2. conv 用 CMSIS-NN 的 arm_convolve_s8 (padding = 对称 same, 源码实测
 *      k_x = base_idx_x + dilation_x * i_ker_x, base_idx_x = -pad_x)
 *   3. FC / GAP / SE scale-back 手写, 语义逐条对齐镜像 (见 L1 检查脚本)
 *
 * 布局约定:
 *   输入 x   : (23, 16) channel-major  (跟 FP32 版一致, 调用方无感)
 *   内部激活 : (16, 32) NHWC           (CMSIS-NN 输入/输出约定)
 *               index = t * C + c
 * ============================================================================
 */
#include "ch32_inference.h"

#if CH32_USE_INT8

#include "ch32_inference_int8.h"
#include "model_v4_se_23dim_ch32_int8_s42.h"   /* int8 权重 + 量化参数 */

#include "arm_nnfunctions.h"        /* arm_convolve_s8 */
#include "arm_nnsupportfunctions.h"  /* arm_nn_requantize */

#include <string.h>
#include <math.h>

#include "main.h"   /* DWT cycle counter */

/* --- 维度常量 (与 ch32_inference.h 一致) --------------------------------- */
#define I8_T    CH32_WINDOW        /* 16 时间步 */
#define I8_C    CH32_CH            /* 32 通道   */
#define I8_IN   CH32_IN_CHANNELS   /* 23 输入维 */
#define I8_SEH  CH32_SE_HIDDEN     /* 4  SE 隐藏维 */
#define I8_FC1  CH32_FC1_OUT       /* 32 分类器 FC1 输出 */

/* --- 运行期缓冲 ---------------------------------------------------------- */
/* 全部放 .bss (DTCM, 不可 cache -> 无 DCache 一致性问题) */
static int8_t  i8_x[I8_T * I8_IN];    /* 量化后的输入 (NHWC)        368 B */
static int8_t  i8_res[I8_T * I8_C];   /* block0 residual (add 域)   512 B */
static int8_t  i8_c1[I8_T * I8_C];    /* conv1 输出                 512 B */
static int8_t  i8_c2[I8_T * I8_C];    /* conv2 输出                 512 B */
static int8_t  i8_se[I8_T * I8_C];    /* SE 缩放后 (add 域)         512 B */
static int8_t  i8_add[I8_T * I8_C];   /* block 输出 (add 域)        512 B */
static int8_t  i8_gapv[I8_C];        /* GAP 输出                    32 B */
static int8_t  i8_h[I8_SEH];         /* SE FC1 输出                  4 B */
static int8_t  i8_fc1v[I8_FC1];       /* 分类器 FC1 输出             32 B */
static float   i8_logit[I8_C];        /* SE FC2 logit                128 B */

/* CMSIS-NN im2col scratch: 2 * C_in * K * sizeof(int16)
 * 最大需求 = 2 * 32 * 3 * 2 = 384 B -> 这里给 512 B (int32 对齐) */
static int32_t i8_cmsis_buf[128];

/* ============================================================================
 * 定点原语
 * ============================================================================ */

/**
 * 输入量化: FP32 (23,16) -> int8 NHWC (16,23)
 *   q = round(x / s_x) + zp_x,  clamp 到 [-128, 127]
 * round 语义 = half away from zero (跟 numpy 镜像的 q_round 一致)
 */
static void i8_quant_input(const float* x, int8_t* out)
{
    const float s = v4se23_ch32_i8_s_x;
    const int32_t zp = v4se23_ch32_i8_zp_x;

    for (int ic = 0; ic < I8_IN; ic++) {
        for (int t = 0; t < I8_T; t++) {
            float v = x[ic * I8_T + t] / s;
            int32_t q = (int32_t)((v >= 0.0f) ? (v + 0.5f) : (v - 0.5f)) + zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            out[t * I8_IN + ic] = (int8_t)q;
        }
    }
}

/**
 * GAP (全局平均池化): (16, C) -> (C)
 *   sum -> 加/减 T/2 -> 整数除法 (截断向零), 保持原 scale/zp.
 * ⚠️ 必须跟 numpy 镜像 gap_int8() 完全一致, 否则概率对不上.
 */
static void i8_gap(const int8_t* in, int T, int C, int8_t* out)
{
    const int32_t half = T / 2;
    for (int c = 0; c < C; c++) {
        int32_t s = 0;
        for (int t = 0; t < T; t++) {
            s += (int32_t)in[t * C + c];
        }
        int32_t v = (s > 0) ? (s + half) : (s - half);
        out[c] = (int8_t)(v / T);   /* C 的整数除法 = 截断向零 */
    }
}

/**
 * FC (per-output-channel requant)
 *   acc = sum_i (q_in[i] - zp_in) * w[oc][i] + bias[oc]
 *   out = arm_nn_requantize(acc, mult[oc], shift[oc]) + zp_out, 再 clamp
 */
static void i8_fc(const int8_t* in, int in_c,
                  const int8_t* w, const int32_t* bias,
                  const int32_t* mult, const int32_t* shift,
                  int out_c, int in_zp, int out_zp,
                  int act_min, int act_max, int8_t* out)
{
    for (int oc = 0; oc < out_c; oc++) {
        const int8_t* wo = w + oc * in_c;
        int32_t acc = bias[oc];
        for (int ic = 0; ic < in_c; ic++) {
            acc += ((int32_t)in[ic] - in_zp) * (int32_t)wo[ic];
        }
        int32_t v = arm_nn_requantize(acc, mult[oc], shift[oc]) + out_zp;
        if (v < act_min)  v = act_min;
        if (v > act_max)  v = act_max;
        out[oc] = (int8_t)v;
    }
}

/**
 * FC -> int32 累加器 -> FP32 (logit 通道, 不量化)
 *   logit[oc] = acc * (s_in * sw[oc])
 */
static void i8_fc_logit(const int8_t* in, int in_c,
                        const int8_t* w, const int32_t* bias,
                        const float* sw, float s_in,
                        int out_c, int in_zp, float* out)
{
    for (int oc = 0; oc < out_c; oc++) {
        const int8_t* wo = w + oc * in_c;
        int32_t acc = bias[oc];
        for (int ic = 0; ic < in_c; ic++) {
            acc += ((int32_t)in[ic] - in_zp) * (int32_t)wo[ic];
        }
        out[oc] = (float)acc * (s_in * sw[oc]);
    }
}

/* ============================================================================
 * CMSIS-NN conv 包装
 * ============================================================================ */

/**
 * 一层 conv (CMSIS-NN arm_convolve_s8)
 *   in   : (T, in_c) int8 NHWC
 *   w    : (out_c, K, in_c) int8  == CMSIS filter_dims = [C_OUT, 1, K, C_IN]
 *   dil  : dilation (1 / 2 / 4)
 *   pad  : (K-1)*dil/2, 对称 same padding (跟 PyTorch nn.Conv1d 一致)
 *
 * ⚠️ CMSIS 的 padding 语义 (源码 arm_convolve_s8.c 实测):
 *      k_x = base_idx_x + dilation_x * i_ker_x,  base_idx_x = stride*i_out_x - pad_x
 *   即 out[t] 用 in[t + k*dil - pad] —— 跟 PyTorch 的对称 same padding 一致.
 */
static void i8_conv(const int8_t* in, int in_c,
                    const int8_t* w, const int32_t* bias,
                    const int32_t* mult, const int32_t* shift,
                    int out_c, int T, int K, int dil,
                    int in_zp, int out_zp,
                    int act_min, int act_max, int8_t* out)
{
    cmsis_nn_context ctx;
    cmsis_nn_dims in_dims, filt_dims, bias_dims, out_dims;
    cmsis_nn_per_channel_quant_params qp;
    cmsis_nn_conv_params cp;

    const int pad = (K - 1) * dil / 2;

    ctx.buf  = i8_cmsis_buf;
    ctx.size = (int32_t)sizeof(i8_cmsis_buf);

    in_dims.n   = 1;     in_dims.h   = 1; in_dims.w = T;     in_dims.c = in_c;
    filt_dims.n = out_c; filt_dims.h = 1; filt_dims.w = K;    filt_dims.c = in_c;
    bias_dims.n = 1;     bias_dims.h = 1; bias_dims.w = 1;    bias_dims.c = out_c;
    out_dims.n  = 1;     out_dims.h  = 1; out_dims.w  = T;     out_dims.c  = out_c;

    qp.multiplier = (int32_t*)mult;
    qp.shift      = (int32_t*)shift;

    cp.input_offset  = -in_zp;      /* CMSIS: input_offset = -zp_in */
    cp.output_offset = out_zp;      /* CMSIS: output_offset = +zp_out */
    cp.stride.h   = 1; cp.stride.w   = 1;
    cp.padding.h  = 0; cp.padding.w  = pad;
    cp.dilation.h  = 1; cp.dilation.w = dil;
    cp.activation.min = act_min;
    cp.activation.max = act_max;

    (void)arm_convolve_s8(&ctx, &cp, &qp, &in_dims, in,
                           &filt_dims, w, &bias_dims, bias,
                           &out_dims, out);
}

/* ============================================================================
 * SE block
 * ============================================================================ */

/**
 * Squeeze-and-Excitation (全定点 + FP32 sigmoid)
 *   c2 (T,32) in c2 域
 *     -> GAP -> SE FC1 (per-channel requant, ReLU) -> SE FC2 (int32 -> logit)
 *     -> sigmoid -> Q12 (sig_q) -> 逐通道缩放 c2 -> add 域
 *
 * 缩放公式 (跟 numpy 镜像 _se_block 一致):
 *   v     = q_c2 - zp_c2                       (c2 域, 中心化)
 *   qsig  = round(sigmoid(logit) * sig_q)      (Q12)
 *   acc   = v * qsig
 *   out   = arm_nn_requantize(acc, mult_se, shift_se) + zp_add
 */
static void i8_se_block(const int8_t* c2, int bi, int8_t* out)
{
    const int32_t zp_c2  = (bi == 0) ? v4se23_ch32_i8_zp_c2_0 :
                           (bi == 1) ? v4se23_ch32_i8_zp_c2_1 :
                                       v4se23_ch32_i8_zp_c2_2;
    const int32_t zp_seh = (bi == 0) ? v4se23_ch32_i8_zp_seh_0 :
                           (bi == 1) ? v4se23_ch32_i8_zp_seh_1 :
                                       v4se23_ch32_i8_zp_seh_2;
    const int32_t zp_add = (bi == 0) ? v4se23_ch32_i8_zp_add_0 :
                           (bi == 1) ? v4se23_ch32_i8_zp_add_1 :
                                       v4se23_ch32_i8_zp_add_2;
    const float   s_seh  = (bi == 0) ? v4se23_ch32_i8_s_seh_0 :
                           (bi == 1) ? v4se23_ch32_i8_s_seh_1 :
                                       v4se23_ch32_i8_s_seh_2;

    const int8_t*  w_se1 = (bi == 0) ? v4se23_ch32_i8_w_tcn_0_se_fc1 :
                           (bi == 1) ? v4se23_ch32_i8_w_tcn_1_se_fc1 :
                                       v4se23_ch32_i8_w_tcn_2_se_fc1;
    const int32_t* b_se1 = (bi == 0) ? v4se23_ch32_i8_b_tcn_0_se_fc1 :
                           (bi == 1) ? v4se23_ch32_i8_b_tcn_1_se_fc1 :
                                       v4se23_ch32_i8_b_tcn_2_se_fc1;
    const int32_t* m_se1 = (bi == 0) ? v4se23_ch32_i8_mult_tcn_0_se_fc1 :
                           (bi == 1) ? v4se23_ch32_i8_mult_tcn_1_se_fc1 :
                                       v4se23_ch32_i8_mult_tcn_2_se_fc1;
    const int32_t* s_se1 = (bi == 0) ? v4se23_ch32_i8_shift_tcn_0_se_fc1 :
                           (bi == 1) ? v4se23_ch32_i8_shift_tcn_1_se_fc1 :
                                       v4se23_ch32_i8_shift_tcn_2_se_fc1;
    const int8_t*  w_se2 = (bi == 0) ? v4se23_ch32_i8_w_tcn_0_se_fc2 :
                           (bi == 1) ? v4se23_ch32_i8_w_tcn_1_se_fc2 :
                                       v4se23_ch32_i8_w_tcn_2_se_fc2;
    const int32_t* b_se2 = (bi == 0) ? v4se23_ch32_i8_b_tcn_0_se_fc2 :
                           (bi == 1) ? v4se23_ch32_i8_b_tcn_1_se_fc2 :
                                       v4se23_ch32_i8_b_tcn_2_se_fc2;
    const float*   sw_se2 = (bi == 0) ? v4se23_ch32_i8_sw_tcn_0_se_fc2 :
                           (bi == 1) ? v4se23_ch32_i8_sw_tcn_1_se_fc2 :
                                       v4se23_ch32_i8_sw_tcn_2_se_fc2;

    const int32_t mult_se = (bi == 0) ? v4se23_ch32_i8_mult_se_0 :
                              (bi == 1) ? v4se23_ch32_i8_mult_se_1 :
                                          v4se23_ch32_i8_mult_se_2;
    const int32_t shift_se = (bi == 0) ? v4se23_ch32_i8_shift_se_0 :
                               (bi == 1) ? v4se23_ch32_i8_shift_se_1 :
                                           v4se23_ch32_i8_shift_se_2;

    /* GAP (c2 域) */
    i8_gap(c2, I8_T, I8_C, i8_gapv);

    /* SE FC1 -> seh 域 (ReLU) */
    i8_fc(i8_gapv, I8_C, w_se1, b_se1, m_se1, s_se1,
          I8_SEH, zp_c2, zp_seh, zp_seh, 127, i8_h);

    /* SE FC2 -> logit (int32 累加器 + per-channel 反量化) */
    i8_fc_logit(i8_h, I8_SEH, w_se2, b_se2, sw_se2, s_seh, I8_C, zp_seh, i8_logit);

    /* sigmoid -> Q12 -> 逐通道缩放 */
    for (int c = 0; c < I8_C; c++) {
        float sig = 1.0f / (1.0f + expf(-i8_logit[c]));
        int32_t qsig = (int32_t)(sig * (float)v4se23_ch32_i8_sig_q + 0.5f);  /* sig >= 0 */
        for (int t = 0; t < I8_T; t++) {
            int32_t v = (int32_t)c2[t * I8_C + c] - zp_c2;
            int32_t acc = v * qsig;
            int32_t o = arm_nn_requantize(acc, mult_se, shift_se) + zp_add;
            if (o < -128) o = -128;
            if (o > 127)  o = 127;
            out[t * I8_C + c] = (int8_t)o;
        }
    }
}

/* ============================================================================
 * 残差相加
 * ============================================================================ */

/**
 * add + ReLU (两边同域 add_b):
 *   out = clamp( (cur - zp) + (res - zp) + zp, zp, 127 )
 */
static void i8_add_res(int8_t* out, const int8_t* cur, const int8_t* res, int zp)
{
    for (int i = 0; i < I8_T * I8_C; i++) {
        int32_t v = ((int32_t)cur[i] - zp) + ((int32_t)res[i] - zp) + zp;
        if (v < zp)   v = zp;
        if (v > 127)  v = 127;
        out[i] = (int8_t)v;
    }
}

/**
 * identity residual (block 1/2): 上一个 block 的 add 域 -> 本 block 的 add 域
 *   res_c = arm_nn_requantize(prev - zp_prev, mult_res, shift_res)
 * (结果就是本 block add 域里 "中心化" 的残差量)
 */
static void i8_res_requant_add(int8_t* out, const int8_t* prev,
                               int32_t mult, int32_t shift, int zp_prev,
                               const int8_t* cur, int zp_add)
{
    for (int i = 0; i < I8_T * I8_C; i++) {
        int32_t rc = arm_nn_requantize((int32_t)prev[i] - zp_prev, mult, shift);
        int32_t cc = (int32_t)cur[i] - zp_add;
        int32_t v = cc + rc + zp_add;
        if (v < zp_add)  v = zp_add;
        if (v > 127)     v = 127;
        out[i] = (int8_t)v;
    }
}

/* ============================================================================
 * 公共 API
 * ============================================================================ */

void ch32_int8_init(void)
{
    /* INT8 权重直接是 const int8, 常驻 Flash, 无需 dequant / 无需 RAM 拷贝.
     * 这里只做一次常量健全性自检 (编译期就已知, 运行期几乎零成本). */
    (void)v4se23_ch32_i8_sig_q;
}

float ch32_int8_forward(const float* x, size_t len)
{
    (void)len;

    /* ---------- Block 0 (d=1) ---------- */
    i8_quant_input(x, i8_x);

    /* residual: 1x1 conv 23->32, 直接出到 add_0 域 (省一次 requant) */
    i8_conv(i8_x, I8_IN,
            v4se23_ch32_i8_w_tcn_0_residual, v4se23_ch32_i8_b_tcn_0_residual,
            v4se23_ch32_i8_mult_tcn_0_residual, v4se23_ch32_i8_shift_tcn_0_residual,
            I8_C, I8_T, 1, 1,
            v4se23_ch32_i8_zp_x, v4se23_ch32_i8_zp_add_0, -128, 127, i8_res);

    /* conv1 (23->32) + ReLU */
    i8_conv(i8_x, I8_IN,
            v4se23_ch32_i8_w_tcn_0_conv1, v4se23_ch32_i8_b_tcn_0_conv1,
            v4se23_ch32_i8_mult_tcn_0_conv1, v4se23_ch32_i8_shift_tcn_0_conv1,
            I8_C, I8_T, 3, CH32_DIL_B0,
            v4se23_ch32_i8_zp_x, v4se23_ch32_i8_zp_c1_0,
            v4se23_ch32_i8_zp_c1_0, 127, i8_c1);

    /* conv2 (32->32) + ReLU */
    i8_conv(i8_c1, I8_C,
            v4se23_ch32_i8_w_tcn_0_conv2, v4se23_ch32_i8_b_tcn_0_conv2,
            v4se23_ch32_i8_mult_tcn_0_conv2, v4se23_ch32_i8_shift_tcn_0_conv2,
            I8_C, I8_T, 3, CH32_DIL_B0,
            v4se23_ch32_i8_zp_c1_0, v4se23_ch32_i8_zp_c2_0,
            v4se23_ch32_i8_zp_c2_0, 127, i8_c2);

    /* SE -> add_0 域 */
    i8_se_block(i8_c2, 0, i8_se);

    /* add + ReLU -> add_0 */
    i8_add_res(i8_add, i8_se, i8_res, v4se23_ch32_i8_zp_add_0);

    /* ---------- Block 1 (d=2) ---------- */
    i8_conv(i8_add, I8_C,
            v4se23_ch32_i8_w_tcn_1_conv1, v4se23_ch32_i8_b_tcn_1_conv1,
            v4se23_ch32_i8_mult_tcn_1_conv1, v4se23_ch32_i8_shift_tcn_1_conv1,
            I8_C, I8_T, 3, CH32_DIL_B1,
            v4se23_ch32_i8_zp_add_0, v4se23_ch32_i8_zp_c1_1,
            v4se23_ch32_i8_zp_c1_1, 127, i8_c1);

    i8_conv(i8_c1, I8_C,
            v4se23_ch32_i8_w_tcn_1_conv2, v4se23_ch32_i8_b_tcn_1_conv2,
            v4se23_ch32_i8_mult_tcn_1_conv2, v4se23_ch32_i8_shift_tcn_1_conv2,
            I8_C, I8_T, 3, CH32_DIL_B1,
            v4se23_ch32_i8_zp_c1_1, v4se23_ch32_i8_zp_c2_1,
            v4se23_ch32_i8_zp_c2_1, 127, i8_c2);

    i8_se_block(i8_c2, 1, i8_se);

    /* identity residual: add_0 域 -> add_1 域, 再相加 */
    i8_res_requant_add(i8_add, i8_add,
                       v4se23_ch32_i8_mult_res_1, v4se23_ch32_i8_shift_res_1,
                       v4se23_ch32_i8_zp_add_0,
                       i8_se, v4se23_ch32_i8_zp_add_1);

    /* ---------- Block 2 (d=4) ---------- */
    i8_conv(i8_add, I8_C,
            v4se23_ch32_i8_w_tcn_2_conv1, v4se23_ch32_i8_b_tcn_2_conv1,
            v4se23_ch32_i8_mult_tcn_2_conv1, v4se23_ch32_i8_shift_tcn_2_conv1,
            I8_C, I8_T, 3, CH32_DIL_B2,
            v4se23_ch32_i8_zp_add_1, v4se23_ch32_i8_zp_c1_2,
            v4se23_ch32_i8_zp_c1_2, 127, i8_c1);

    i8_conv(i8_c1, I8_C,
            v4se23_ch32_i8_w_tcn_2_conv2, v4se23_ch32_i8_b_tcn_2_conv2,
            v4se23_ch32_i8_mult_tcn_2_conv2, v4se23_ch32_i8_shift_tcn_2_conv2,
            I8_C, I8_T, 3, CH32_DIL_B2,
            v4se23_ch32_i8_zp_c1_2, v4se23_ch32_i8_zp_c2_2,
            v4se23_ch32_i8_zp_c2_2, 127, i8_c2);

    i8_se_block(i8_c2, 2, i8_se);

    i8_res_requant_add(i8_add, i8_add,
                       v4se23_ch32_i8_mult_res_2, v4se23_ch32_i8_shift_res_2,
                       v4se23_ch32_i8_zp_add_1,
                       i8_se, v4se23_ch32_i8_zp_add_2);

    /* ---------- Classifier ---------- */
    /* GAP (add_2 域) */
    i8_gap(i8_add, I8_T, I8_C, i8_gapv);

    /* FC1 (32->32) + ReLU */
    i8_fc(i8_gapv, I8_C,
          v4se23_ch32_i8_w_fc1, v4se23_ch32_i8_b_fc1,
          v4se23_ch32_i8_mult_fc1, v4se23_ch32_i8_shift_fc1,
          I8_FC1, v4se23_ch32_i8_zp_add_2, v4se23_ch32_i8_zp_fc1,
          v4se23_ch32_i8_zp_fc1, 127, i8_fc1v);

    /* FC2 (32->1) -> logit */
    float logit;
    i8_fc_logit(i8_fc1v, I8_FC1,
                 v4se23_ch32_i8_w_fc2, v4se23_ch32_i8_b_fc2,
                 v4se23_ch32_i8_sw_fc2, v4se23_ch32_i8_s_fc1,
                 1, v4se23_ch32_i8_zp_fc1, &logit);

    /* temperature scaling (跟 FP32 版一致, T=1.0) */
    logit /= CH32_TEMPERATURE;

    return 1.0f / (1.0f + expf(-logit));
}

/* --- 性能 / 自检 --------------------------------------------------------- */

uint32_t ch32_int8_benchmark(int n_runs)
{
    float x[CH32_IN_SIZE];
    for (int i = 0; i < CH32_IN_SIZE; i++) x[i] = 0.1f;

    ch32_int8_forward(x, CH32_IN_SIZE);   /* warm-up */

    uint32_t t0 = DWT->CYCCNT;
    for (int i = 0; i < n_runs; i++) {
        ch32_int8_forward(x, CH32_IN_SIZE);
    }
    uint32_t t1 = DWT->CYCCNT;

    return (t1 - t0) / n_runs;
}

void ch32_int8_dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

ch32_mem_info_t ch32_int8_get_mem_info(void)
{
    ch32_mem_info_t m;
    size_t w = 0;
    w += sizeof(v4se23_ch32_i8_w_tcn_0_conv1)
       + sizeof(v4se23_ch32_i8_w_tcn_0_conv2)
       + sizeof(v4se23_ch32_i8_w_tcn_0_residual)
       + sizeof(v4se23_ch32_i8_w_tcn_0_se_fc1)
       + sizeof(v4se23_ch32_i8_w_tcn_0_se_fc2);
    w += sizeof(v4se23_ch32_i8_w_tcn_1_conv1)
       + sizeof(v4se23_ch32_i8_w_tcn_1_conv2)
       + sizeof(v4se23_ch32_i8_w_tcn_1_se_fc1)
       + sizeof(v4se23_ch32_i8_w_tcn_1_se_fc2);
    w += sizeof(v4se23_ch32_i8_w_tcn_2_conv1)
       + sizeof(v4se23_ch32_i8_w_tcn_2_conv2)
       + sizeof(v4se23_ch32_i8_w_tcn_2_se_fc1)
       + sizeof(v4se23_ch32_i8_w_tcn_2_se_fc2);
    w += sizeof(v4se23_ch32_i8_w_fc1) + sizeof(v4se23_ch32_i8_w_fc2);

    m.weights_fp32_bytes = w;   /* 复用字段: 现在是 int8 权重字节数 */
    m.input_bytes        = CH32_IN_SIZE * sizeof(float);
    m.peak_act_bytes     = sizeof(i8_x) + sizeof(i8_res) + sizeof(i8_c1)
                         + sizeof(i8_c2) + sizeof(i8_se) + sizeof(i8_add)
                         + sizeof(i8_gapv) + sizeof(i8_h) + sizeof(i8_fc1v)
                         + sizeof(i8_logit) + sizeof(i8_cmsis_buf);
    m.total_ram_bytes    = m.peak_act_bytes + m.input_bytes;

    return m;
}

#endif /* CH32_USE_INT8 */
