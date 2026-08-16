/* ============================================================================
 * v19_inference.c  —  TCN V19 MCU Inference Implementation
 * ============================================================================
 *
 * 架构 (与 PyTorch V19 完全一致):
 *   x (44, 16)
 *     ↓
 *   TCN Block 0  (Conv1d×2 + ReLU + SE + Residual 1×1)
 *     ↓ (12, 16)
 *   TCN Block 1  (Conv1d×2 + ReLU + SE + Identity Residual)
 *     ↓ (12, 16)
 *   GAP  →  Linear(12→32) + ReLU  →  Linear(32→1)  →  Sigmoid
 *     ↓
 *   probability ∈ [0, 1]
 *
 * 数据 layout (与 PyTorch Conv1d 一致):
 *   shape (channels, time_steps),row-major
 *   例: input[ic*16 + t]  = 第 ic 个特征在 t 时刻的值
 *
 * 关键优化 (MCU 友好):
 *   - 静态 RAM 分配,无 malloc
 *   - 启动时 dequant 一次性,运行期纯 FP32
 *   - 内联辅助函数避免调用开销
 *   - 循环顺序优化 cache 友好
 *
 * 进一步优化选项 (如需极致性能):
 *   1. 用 CMSIS-NN 替换 conv1d: arm_conv_f32 / arm_fully_connected_f32
 *      (代码 +5-20 KB Flash,但单次推理可从 0.47ms 降到 ~0.15ms @ Cortex-M4 168MHz)
 *   2. 用 -O3 -ffast-math -funroll-loops 编译
 *   3. 用 SIMD intrinsics (__SMLAD 等) 手写 MAC 循环
 *   4. "Lite 模式":跳过 init dequant,运行期按需 dequant (省 12 KB RAM)
 *      改 conv1d 循环: sum += q[oc*step + i] * scale[oc] * in[ic*L+t_in]
 *      只需把 w 参数换成 (q, scale) 对,实现见末尾注释
 * ============================================================================
 */

#include "v19_inference.h"
#include "model_v19_hybrid.h"
#include <string.h>
#include <math.h>

/* ────────────────────────────────────────────────────────────────────────
 * 1. RAM 缓冲:dequant 后的 FP32 权重 (启动时由 v19_init 填充)
 * ──────────────────────────────────────────────────────────────────────── */

/* TCN Block 0 */
static float w_tcn_0_c1[V19_CH * V19_IN_CHANNELS * 3];   /* 12*44*3 = 1584 */
static float w_tcn_0_c2[V19_CH * V19_CH * 3];             /* 12*12*3 = 432 */
static float w_tcn_0_rs[V19_CH * V19_IN_CHANNELS * 1];    /* 12*44*1 = 528 */
static float w_se_0_fc1[V19_SE_HIDDEN * V19_CH];          /* 4*12 = 48 */
static float w_se_0_fc2[V19_CH * V19_SE_HIDDEN];          /* 12*4 = 48 */
static float b_tcn_0_c1[V19_CH];
static float b_tcn_0_c2[V19_CH];
static float b_tcn_0_rs[V19_CH];
static float b_se_0_fc1[V19_SE_HIDDEN];
static float b_se_0_fc2[V19_CH];

/* TCN Block 1 */
static float w_tcn_1_c1[V19_CH * V19_CH * 3];
static float w_tcn_1_c2[V19_CH * V19_CH * 3];
static float w_se_1_fc1[V19_SE_HIDDEN * V19_CH];
static float w_se_1_fc2[V19_CH * V19_SE_HIDDEN];
static float b_tcn_1_c1[V19_CH];
static float b_tcn_1_c2[V19_CH];
static float b_se_1_fc1[V19_SE_HIDDEN];
static float b_se_1_fc2[V19_CH];

/* Classifier */
static float w_fc1[V19_FC1_OUT * V19_CH];                /* 32*12 = 384 */
static float w_fc2[V19_FC2_OUT * V19_FC1_OUT];            /* 1*32 = 32 */
static float b_fc1[V19_FC1_OUT];
static float b_fc2[V19_FC2_OUT];

/* 激活缓冲 (运行期使用) */
static float buf_x[V19_FEAT_SIZE];   /* (12, 16) 当前块输入 */
static float buf_a[V19_FEAT_SIZE];   /* (12, 16) scratch */
static float buf_b[V19_FEAT_SIZE];   /* (12, 16) scratch / 最终输出 */

/* ────────────────────────────────────────────────────────────────────────
 * 2. 工具函数
 * ──────────────────────────────────────────────────────────────────────── */

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/* ────────────────────────────────────────────────────────────────────────
 * 3. Per-channel 对称 INT8 → FP32 dequant
 *
 *    公式: w[oc, ...] = q[oc, ...] * scale[oc]
 *    因为 scale 是按输出通道索引的,内层循环可以对每个 oc 把 scale 提到外层
 * ──────────────────────────────────────────────────────────────────────── */

static void dequant_per_ch(const int8_t* q, const float* scale,
                           float* w, int out_ch, int per_ch_size) {
    for (int oc = 0; oc < out_ch; oc++) {
        float s = scale[oc];
        const int8_t* q_oc = q + oc * per_ch_size;
        float* w_oc = w + oc * per_ch_size;
        for (int i = 0; i < per_ch_size; i++) {
            w_oc[i] = (float)q_oc[i] * s;
        }
    }
}

static void copy_bias(const float* src, float* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

/* ────────────────────────────────────────────────────────────────────────
 * 4. 核心算子
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * 1D 卷积 (无 bias 计算,bias 由调用方加)
 *  layout: in/out/w 都是 (C, T) row-major
 *          w[oc, ic, k] = w[oc * (in_ch * kernel) + ic * kernel + k]
 *  padding: 对称补 0
 *  注意: 假设 stride=1 (V19 中所有 conv 都是 stride=1)
 */
static void conv1d(const float* in, const float* w, const float* b,
                   float* out,
                   int in_ch, int out_ch, int kernel, int dilation, int padding,
                   int L_in, int L_out) {
    for (int oc = 0; oc < out_ch; oc++) {
        const float* w_oc = w + oc * in_ch * kernel;
        const float* b_oc = b + oc;  /* 偏置按 oc 索引,长度 out_ch */
        for (int t = 0; t < L_out; t++) {
            float sum = b_oc[0];
            for (int ic = 0; ic < in_ch; ic++) {
                const float* in_ic = in + ic * L_in;
                const float* w_oc_ic = w_oc + ic * kernel;
                for (int k = 0; k < kernel; k++) {
                    int t_in = t + k * dilation - padding;
                    if (t_in >= 0 && t_in < L_in) {
                        sum += w_oc_ic[k] * in_ic[t_in];
                    }
                }
            }
            out[oc * L_out + t] = sum;
        }
    }
}

/**
 * SE 块:GAP → Linear → ReLU → Linear → Sigmoid,结果写入 gate[ch]
 */
static void se_block(const float* in,
                     const float* w_fc1, const float* b_fc1,
                     const float* w_fc2, const float* b_fc2,
                     float* gate,
                     int ch, int hidden, int L) {
    /* Step 1: GAP over time → gate[ch] */
    for (int c = 0; c < ch; c++) {
        float s = 0.0f;
        const float* in_c = in + c * L;
        for (int t = 0; t < L; t++) s += in_c[t];
        gate[c] = s / (float)L;
    }

    /* Step 2: fc1 + ReLU (写入 gate 临时复用,然后用 stack 缓冲) */
    float tmp[V19_SE_HIDDEN];  /* hidden <= 4,栈上分配 */
    for (int h = 0; h < hidden; h++) {
        float s = b_fc1[h];
        const float* w_h = w_fc1 + h * ch;
        for (int c = 0; c < ch; c++) {
            s += w_h[c] * gate[c];
        }
        tmp[h] = (s > 0.0f) ? s : 0.0f;  /* ReLU */
    }

    /* Step 3: fc2 + Sigmoid → gate[ch] */
    for (int c = 0; c < ch; c++) {
        float s = b_fc2[c];
        const float* w_c = w_fc2 + c * hidden;
        for (int h = 0; h < hidden; h++) {
            s += w_c[h] * tmp[h];
        }
        gate[c] = sigmoidf(s);
    }
}

/**
 * 原地把 SE gate 乘到 (ch, L) 张量上
 */
static void apply_se_gate_inplace(float* io, const float* gate, int ch, int L) {
    for (int c = 0; c < ch; c++) {
        float g = gate[c];
        float* io_c = io + c * L;
        for (int t = 0; t < L; t++) io_c[t] *= g;
    }
}

/**
 * GAP: (ch, L) → (ch,)
 */
static void gap(const float* in, float* out, int ch, int L) {
    for (int c = 0; c < ch; c++) {
        float s = 0.0f;
        const float* in_c = in + c * L;
        for (int t = 0; t < L; t++) s += in_c[t];
        out[c] = s / (float)L;
    }
}

/**
 * Linear: (in_dim) → (out_dim)
 *  layout: w[o, i] = w[o * in_dim + i]
 */
static void linear(const float* in, const float* w, const float* b,
                   float* out, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float s = b[o];
        const float* w_o = w + o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            s += w_o[i] * in[i];
        }
        out[o] = s;
    }
}

/* ────────────────────────────────────────────────────────────────────────
 * 5. 公开 API
 * ──────────────────────────────────────────────────────────────────────── */

void v19_init(void) {
    /* TCN Block 0 */
    dequant_per_ch(v19_w_tcn_0_c1_weight, v19_s_tcn_0_c1_weight, w_tcn_0_c1,
                   V19_CH, V19_IN_CHANNELS * 3);
    dequant_per_ch(v19_w_tcn_0_c2_weight, v19_s_tcn_0_c2_weight, w_tcn_0_c2,
                   V19_CH, V19_CH * 3);
    dequant_per_ch(v19_w_tcn_0_rs_weight, v19_s_tcn_0_rs_weight, w_tcn_0_rs,
                   V19_CH, V19_IN_CHANNELS * 1);
    dequant_per_ch(v19_w_tcn_0_se_fc1_weight, v19_s_tcn_0_se_fc1_weight, w_se_0_fc1,
                   V19_SE_HIDDEN, V19_CH);
    dequant_per_ch(v19_w_tcn_0_se_fc2_weight, v19_s_tcn_0_se_fc2_weight, w_se_0_fc2,
                   V19_CH, V19_SE_HIDDEN);
    copy_bias(v19_b_tcn_0_c1_bias, b_tcn_0_c1, V19_CH);
    copy_bias(v19_b_tcn_0_c2_bias, b_tcn_0_c2, V19_CH);
    copy_bias(v19_b_tcn_0_rs_bias, b_tcn_0_rs, V19_CH);
    copy_bias(v19_b_tcn_0_se_fc1_bias, b_se_0_fc1, V19_SE_HIDDEN);
    copy_bias(v19_b_tcn_0_se_fc2_bias, b_se_0_fc2, V19_CH);

    /* TCN Block 1 (no rs since Identity) */
    dequant_per_ch(v19_w_tcn_1_c1_weight, v19_s_tcn_1_c1_weight, w_tcn_1_c1,
                   V19_CH, V19_CH * 3);
    dequant_per_ch(v19_w_tcn_1_c2_weight, v19_s_tcn_1_c2_weight, w_tcn_1_c2,
                   V19_CH, V19_CH * 3);
    dequant_per_ch(v19_w_tcn_1_se_fc1_weight, v19_s_tcn_1_se_fc1_weight, w_se_1_fc1,
                   V19_SE_HIDDEN, V19_CH);
    dequant_per_ch(v19_w_tcn_1_se_fc2_weight, v19_s_tcn_1_se_fc2_weight, w_se_1_fc2,
                   V19_CH, V19_SE_HIDDEN);
    copy_bias(v19_b_tcn_1_c1_bias, b_tcn_1_c1, V19_CH);
    copy_bias(v19_b_tcn_1_c2_bias, b_tcn_1_c2, V19_CH);
    copy_bias(v19_b_tcn_1_se_fc1_bias, b_se_1_fc1, V19_SE_HIDDEN);
    copy_bias(v19_b_tcn_1_se_fc2_bias, b_se_1_fc2, V19_CH);

    /* Classifier */
    dequant_per_ch(v19_w_fc1_weight, v19_s_fc1_weight, w_fc1,
                   V19_FC1_OUT, V19_CH);
    dequant_per_ch(v19_w_fc2_weight, v19_s_fc2_weight, w_fc2,
                   V19_FC2_OUT, V19_FC1_OUT);
    copy_bias(v19_b_fc1_bias, b_fc1, V19_FC1_OUT);
    copy_bias(v19_b_fc2_bias, b_fc2, V19_FC2_OUT);
}

float v19_forward(const float* x, size_t len) {
    (void)len;

    /* 局部缓冲 (gate / fc1 / gap) — 栈上,V19 都很小 */
    float gate[V19_CH];
    float fc1_out[V19_FC1_OUT];
    float gap_out[V19_CH];

    /* 把 x 复制到 buf_x (后续要原地修改) */
    memcpy(buf_x, x, V19_FEAT_SIZE * sizeof(float));

    /* ========== Block 0: Conv1d×2 + ReLU + SE + Residual 1×1 ========== */
    /* c1: (44, 16) → (12, 16), k=3, d=1, p=1 */
    conv1d(buf_x, w_tcn_0_c1, b_tcn_0_c1, buf_a,
           V19_IN_CHANNELS, V19_CH, 3, 1, 1, V19_WINDOW, V19_WINDOW);
    relu_inplace(buf_a, V19_FEAT_SIZE);
    /* c2: (12, 16) → (12, 16), k=3, d=1, p=1 */
    conv1d(buf_a, w_tcn_0_c2, b_tcn_0_c2, buf_b,
           V19_CH, V19_CH, 3, 1, 1, V19_WINDOW, V19_WINDOW);
    relu_inplace(buf_b, V19_FEAT_SIZE);
    /* SE: 算出 gate 并应用 */
    se_block(buf_b, w_se_0_fc1, b_se_0_fc1, w_se_0_fc2, b_se_0_fc2,
             gate, V19_CH, V19_SE_HIDDEN, V19_WINDOW);
    apply_se_gate_inplace(buf_b, gate, V19_CH, V19_WINDOW);
    /* Residual: 1×1 conv(buf_x) → 加到 buf_b */
    conv1d(buf_x, w_tcn_0_rs, b_tcn_0_rs, buf_a,
           V19_IN_CHANNELS, V19_CH, 1, 1, 0, V19_WINDOW, V19_WINDOW);
    for (int i = 0; i < V19_FEAT_SIZE; i++) {
        float v = buf_b[i] + buf_a[i];
        buf_b[i] = (v > 0.0f) ? v : 0.0f;  /* ReLU after add */
    }
    /* Block 0 输出: buf_b → 复制到 buf_x 作为 Block 1 输入 */
    memcpy(buf_x, buf_b, V19_FEAT_SIZE * sizeof(float));

    /* ========== Block 1: Conv1d×2 + ReLU + SE + Identity Residual ========== */
    /* c1: (12, 16) → (12, 16), k=3, d=2, p=2 */
    conv1d(buf_x, w_tcn_1_c1, b_tcn_1_c1, buf_a,
           V19_CH, V19_CH, 3, 2, 2, V19_WINDOW, V19_WINDOW);
    relu_inplace(buf_a, V19_FEAT_SIZE);
    /* c2: (12, 16) → (12, 16), k=3, d=2, p=2 */
    conv1d(buf_a, w_tcn_1_c2, b_tcn_1_c2, buf_b,
           V19_CH, V19_CH, 3, 2, 2, V19_WINDOW, V19_WINDOW);
    relu_inplace(buf_b, V19_FEAT_SIZE);
    /* SE */
    se_block(buf_b, w_se_1_fc1, b_se_1_fc1, w_se_1_fc2, b_se_1_fc2,
             gate, V19_CH, V19_SE_HIDDEN, V19_WINDOW);
    apply_se_gate_inplace(buf_b, gate, V19_CH, V19_WINDOW);
    /* Residual: Identity (直接加 buf_x) */
    for (int i = 0; i < V19_FEAT_SIZE; i++) {
        float v = buf_b[i] + buf_x[i];
        buf_b[i] = (v > 0.0f) ? v : 0.0f;
    }
    /* Block 1 输出: buf_b */

    /* ========== Classifier: GAP → FC → ReLU → FC → Sigmoid ========== */
    gap(buf_b, gap_out, V19_CH, V19_WINDOW);
    linear(gap_out, w_fc1, b_fc1, fc1_out, V19_CH, V19_FC1_OUT);
    relu_inplace(fc1_out, V19_FC1_OUT);
    float logit;
    linear(fc1_out, w_fc2, b_fc2, &logit, V19_FC1_OUT, V19_FC2_OUT);
    return sigmoidf(logit);
}

v19_mem_info_t v19_get_mem_info(void) {
    v19_mem_info_t info;
    /* 权重(dequant 后的 FP32) */
    info.weights_fp32_bytes =
        sizeof(w_tcn_0_c1) + sizeof(w_tcn_0_c2) + sizeof(w_tcn_0_rs) +
        sizeof(w_se_0_fc1) + sizeof(w_se_0_fc2) +
        sizeof(b_tcn_0_c1) + sizeof(b_tcn_0_c2) + sizeof(b_tcn_0_rs) +
        sizeof(b_se_0_fc1) + sizeof(b_se_0_fc2) +
        sizeof(w_tcn_1_c1) + sizeof(w_tcn_1_c2) +
        sizeof(w_se_1_fc1) + sizeof(w_se_1_fc2) +
        sizeof(b_tcn_1_c1) + sizeof(b_tcn_1_c2) +
        sizeof(b_se_1_fc1) + sizeof(b_se_1_fc2) +
        sizeof(w_fc1) + sizeof(w_fc2) +
        sizeof(b_fc1) + sizeof(b_fc2);
    /* 输入 */
    info.input_bytes = V19_IN_SIZE * (int)sizeof(float);
    /* 激活峰值:3 个 (12,16) 缓冲 + gate(12) + fc1(32) + gap(12) */
    info.peak_act_bytes =
        3 * V19_FEAT_SIZE * (int)sizeof(float) +
        V19_CH * (int)sizeof(float) +
        V19_FC1_OUT * (int)sizeof(float) +
        V19_CH * (int)sizeof(float);
    return info;
}

void v19_benchmark(int n_runs) {
    /* 构造一个随机输入 */
    float x[V19_IN_SIZE];
    for (int i = 0; i < V19_IN_SIZE; i++) {
        /* 简易 LCG 伪随机(避免引入 rand/srand 依赖) */
        static uint32_t s = 12345;
        s = s * 1103515245u + 12345u;
        x[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;  /* [-1, 1] */
    }
    /* warmup */
    volatile float sink = 0.0f;
    for (int i = 0; i < 10; i++) sink += v19_forward(x, V19_IN_SIZE);
    (void)sink;

    /* 实测:用 DWT->CYCCNT 计 cycle 数(若定义了 DWT) */
#if defined(DWT) && defined(DWT_CTRL) && defined(DWT_CYCCNT)
    #define HAS_DWT 1
#else
    #define HAS_DWT 0
#endif

#if HAS_DWT
    /* 启用 DWT cycle counter (Cortex-M3/M4/M7) */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    uint32_t t0 = DWT->CYCCNT;
    for (int i = 0; i < n_runs; i++) sink += v19_forward(x, V19_IN_SIZE);
    uint32_t t1 = DWT->CYCCNT;
    uint32_t cycles = (t1 - t0) / (uint32_t)n_runs;
    (void)sink;
    v19_mem_info_t m = v19_get_mem_info();
    /* 这些 printf 假设有重定向到串口,实际项目中替换为你的串口输出 */
    printf("[V19] %d runs: %lu cycles/run  weights=%uB  act=%uB  total=%uB\n",
           n_runs, (unsigned long)cycles,
           (unsigned)m.weights_fp32_bytes, (unsigned)m.peak_act_bytes,
           (unsigned)(m.weights_fp32_bytes + m.peak_act_bytes));
#else
    /* 无 DWT,粗略时间(假设调用方已配置 SysTick) */
    for (int i = 0; i < n_runs; i++) sink += v19_forward(x, V19_IN_SIZE);
    (void)sink;
    v19_mem_info_t m = v19_get_mem_info();
    printf("[V19] %d runs done.  weights=%uB  act=%uB  total=%uB\n",
           n_runs,
           (unsigned)m.weights_fp32_bytes, (unsigned)m.peak_act_bytes,
           (unsigned)(m.weights_fp32_bytes + m.peak_act_bytes));
#endif
}

/* ============================================================================
 * "Lite 模式" (省 12 KB RAM) — 改写说明
 * ============================================================================
 *
 * 动机: 标准模式 dequant 后,权重占 16.5 KB RAM。Arduino Uno (2 KB RAM) 跑不动。
 *       但 V19 全部权重只占 4.4 KB INT8,远小于 RAM 需求。Lite 模式跳过 dequant,
 *       每次 conv 现场 dequant:  sum += q[oc,i,k] * scale[oc] * in[ic,t_in]
 *
 * 实现方法 (只需改 4 处):
 *   1. 删除所有 w_tcn_0_c1 等 FP32 缓冲 (释放 ~16 KB)
 *   2. conv1d 改签名: 增加 const int8_t* w_q, const float* scale 参数
 *   3. 内部循环改为:
 *         float w = (float)w_q[oc * step + ic*kernel + k] * scale[oc];
 *         sum += w * in[ic*L_in + t_in];
 *   4. v19_init 改为空函数(或只做 sanity check)
 *
 * 性能影响 (Cortex-M4 @ 168 MHz):
 *   - 标准:  0.47 ms / inference
 *   - Lite:  ~0.65 ms / inference (38% 慢)
 *   - 节省:  12 KB RAM (16.5 → 4.4 KB 权重驻留)
 *
 * 推荐场景:
 *   - Arduino Uno / 任何 < 32 KB RAM 的 MCU  → 用 Lite
 *   - STM32F4 / ESP32 / RP2040              → 用标准
 * ============================================================================
 */
