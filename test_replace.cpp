/* ============================================================================
 * tflite_micro_runtime.cpp  --  TFLite Micro H743 wrapper (C++)
 *
 *  ⭐ 2026-10-18 FP32 部署 (取代 QAT INT8)
 *    把 PC-side FP32 .tflite 装进 MicroInterpreter
 *    默认单 seed (s=42); -DCH32_ENABLE_ENSEMBLE=1 切换 5-seed 概率均值
 *
 * 推理流 (FP32 全程, 不量化):
 *   FP32 输入 (23, 16) → memcpy → interpreter.input(0) FP32
 *   → MicroInterpreter.Invoke() → output(0) FP32 logit → sigmoid → FP32 概率
 *
 * ⭐ 2026-10-18 取代 INT8 的原因:
 *   INT8 5-seed PC F1=0.8646, H743 实际 F1=0.0714 (FPR=92.86%)
 *   真因: TFLM runtime (CMSIS-NN / ref int8) 跟 PC TFLite (XNNPACK delegate)
 *         数值不一致, head INT8 量化边界累积差 5+ FP windows.
 *   改 FP32: TFLM ref impl 跟 PC TFLite fp32 数值 bit-perfect 一致,
 *            预期 H743 F1 ≈ 0.86 (跟训练 baseline 1:1).
 * ============================================================================
 */

#include "tflite_micro_config.h"
#include "model_data_fp32_s42.h"
#ifdef CH32_ENABLE_ENSEMBLE
#include "model_data_fp32_s123.h"
#include "model_data_fp32_s456.h"
#include "model_data_fp32_s789.h"
#include "model_data_fp32_s1024.h"
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/micro_log.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>   /* printf for TFLM debug logit/prob */

/* H743 HAL 提供 DWT cycle counter (与 ch32_dwt_init 一致, 也由 TFLu 自己 init) */
#include "main.h"

/* 必须 link 上 DWT register 定义 */
#include "stm32h7xx_hal.h"

/* ch32_dwt_init() / ch32_init() / ch32_forward() 声明 */
#include "ch32_inference.h"

namespace {

/* ⭐ 2026-10-18 FP32 简化: 不需要 Quantize/Dequantize/Transpose, 只需 10 个 op
 *   Conv2D (含 1x1 残差) | FullyConnected | Mul | Add | Logistic (Sigmoid)
 *   Reshape | Mean (GAP) | Relu | SpaceToBatchNd | BatchToSpaceNd (dilated Conv) */
using OpResolver = tflite::MicroMutableOpResolver<16>;  /* ⭐ 2026-10-18 实际 13 op, 留 3 个余量 */

}  // namespace

static OpResolver*               g_resolver[CH32_TFLM_NUM_SEEDS] = {nullptr};
static tflite::MicroInterpreter*  g_interp[CH32_TFLM_NUM_SEEDS]  = {nullptr};

/* arena 放在 D2 SRAM (.bss.tflm_d2 section), 由 H743test.sct 映射到 0x30000000
 * ⭐ 2026-10-18 FP32 单 seed arena ~80KB, D2 SRAM 256KB 容纳 1 seed (5-seed 不行) */
uint8_t ch32_tflm_arena[CH32_TFLM_ARENA_BYTES]
    __attribute__((aligned(16), section(".bss.tflm_d2")));

/* FP32 模型不需要 quant params, 但保留 ch32_tflm_qparams_t 结构兼容旧 API */
ch32_tflm_qparams_t ch32_tflm_qparams[CH32_TFLM_NUM_SEEDS];

extern "C" {

/* ----------------------------------------------------------------------------
 * ch32_init  --  启动期调一次, 准备所有 MicroInterpreter
 * 返回 1 = ok, 0 = 失败 (AllocateTensors 失败 / op 注册不完整)
 * ------------------------------------------------------------------------- */
uint8_t ch32_init(void) {
    static const unsigned char* model_data[CH32_TFLM_NUM_SEEDS] = {
        g_model_v4_se_23dim_ch32_qat_s42_fp32
#ifdef CH32_ENABLE_ENSEMBLE
        , g_model_v4_se_23dim_ch32_qat_s123_fp32
        , g_model_v4_se_23dim_ch32_qat_s456_fp32
        , g_model_v4_se_23dim_ch32_qat_s789_fp32
        , g_model_v4_se_23dim_ch32_qat_s1024_fp32
#endif
    };
    static const unsigned int model_len[CH32_TFLM_NUM_SEEDS] = {
        g_model_v4_se_23dim_ch32_qat_s42_fp32_len
#ifdef CH32_ENABLE_ENSEMBLE
        , g_model_v4_se_23dim_ch32_qat_s123_fp32_len
        , g_model_v4_se_23dim_ch32_qat_s456_fp32_len
        , g_model_v4_se_23dim_ch32_qat_s789_fp32_len
        , g_model_v4_se_23dim_ch32_qat_s1024_fp32_len
#endif
    };

    printf("[TFLM-1] ch32_init enter, num_seeds=%u (FP32)\r\n", (unsigned)CH32_TFLM_NUM_SEEDS);

    for (int i = 0; i < CH32_TFLM_NUM_SEEDS; ++i) {
        printf("[TFLM-2] seed[%d] new OpResolver...\r\n", i);
        g_resolver[i] = new OpResolver;
        if (g_resolver[i] == nullptr) {
            printf("[TFLM-FAIL] seed[%d] new OpResolver = nullptr\r\n", i);
            return 0;
        }
        printf("[TFLM-3] seed[%d] Add ops...\r\n", i);

        /* ⭐ 2026-10-18 FP32 TFLite 实际算子清单 (从 flatbuffer_utils 直读, 不要用
         *   tf.lite.Interpreter._get_ops_details() 因为它会隐藏/错标):
         *   CONV_2D:7  ADD:7  FULLY_CONNECTED:8  RESHAPE:10
         *   MUL:3  LOGISTIC:3  **MEAN:4** (SE block 用 GlobalAvgPool)
         *   SPACE_TO_BATCH_ND:4  BATCH_TO_SPACE_ND:4 (dilated Conv)
         *   **PACK:3** (Keras Reshape 把 1D shape 拆成 N 个 int32 重新组装)
         *   EXPAND_DIMS:6  SHAPE:3  STRIDED_SLICE:3
         *   13 个 builtin opcode, 65 个 op.
         *   Register_MEAN() 在 kernels/reduce.cc (跟其他 reduce 类 op 共用).
         *   没有 UNPACK 没有 DELEGATE 没有 PACK-vs-UNPACK 混淆.
         *   之前 Round 16-17 误诊原因: schema_py_generated.py 跟 TFLM 内部 enum 顺序不一致,
         *   让我误把 builtin 45 (实际 STRIDED_SLICE) 当成 UNPACK, 实际 UNPACK 是 88. */
        if (g_resolver[i]->AddConv2D()         != kTfLiteOk) return 0;
        if (g_resolver[i]->AddFullyConnected() != kTfLiteOk) return 0;
        if (g_resolver[i]->AddMul()            != kTfLiteOk) return 0;
        if (g_resolver[i]->AddAdd()            != kTfLiteOk) return 0;
        if (g_resolver[i]->AddLogistic()       != kTfLiteOk) return 0;
        if (g_resolver[i]->AddReshape()        != kTfLiteOk) return 0;
        if (g_resolver[i]->AddMean()           != kTfLiteOk) return 0;
        if (g_resolver[i]->AddSpaceToBatchNd() != kTfLiteOk) return 0;
        if (g_resolver[i]->AddBatchToSpaceNd() != kTfLiteOk) return 0;
        /* ⭐ 2026-10-18 之前漏的 4 个 shape-only op */
        if (g_resolver[i]->AddExpandDims()     != kTfLiteOk) return 0;
        if (g_resolver[i]->AddPack()           != kTfLiteOk) return 0;
        if (g_resolver[i]->AddShape()          != kTfLiteOk) return 0;
        if (g_resolver[i]->AddStridedSlice()   != kTfLiteOk) return 0;
        printf("[TFLM-4] seed[%d] Add ops OK\r\n", i);

        printf("[TFLM-5] seed[%d] GetModel(addr=0x%08x)...\r\n",
               i, (unsigned)(uintptr_t)model_data[i]);
        const tflite::Model* model = tflite::GetModel(model_data[i]);
        printf("[TFLM-6] seed[%d] GetModel OK, version=%u, expect=%u\r\n",
               i, (unsigned)model->version(), (unsigned)TFLITE_SCHEMA_VERSION);
        if (model->version() != TFLITE_SCHEMA_VERSION) return 0;
        printf("[TFLM-7] seed[%d] new MicroInterpreter...\r\n", i);

        g_interp[i] = new tflite::MicroInterpreter(
            model, *g_resolver[i],
            ch32_tflm_arena + i * CH32_TFLM_ARENA_PER_SEED,
            CH32_TFLM_ARENA_PER_SEED);
        if (g_interp[i] == nullptr) {
            printf("[TFLM-FAIL] seed[%d] new MicroInterpreter = nullptr\r\n", i);
            return 0;
        }
        printf("[TFLM-8] seed[%d] AllocateTensors...\r\n", i);

        if (g_interp[i]->AllocateTensors() != kTfLiteOk) {
            printf("[TFLM-FAIL] seed[%d] AllocateTensors != kTfLiteOk (arena OOB?)\r\n", i);
            return 0;
        }
        printf("[TFLM-9] seed[%d] AllocateTensors OK\r\n", i);

        TfLiteTensor* in  = g_interp[i]->input(0);
        TfLiteTensor* out = g_interp[i]->output(0);
        printf("[TFLM-A] seed[%d] in dtype=%d out dtype=%d\r\n",
               i, (int)in->type, (int)out->type);

        /* FP32 模型: in_scale/zp 没意义, 但保留结构防编译告警 */
        ch32_tflm_qparams[i].in_scale  = in->params.scale;
        ch32_tflm_qparams[i].in_zp     = static_cast<int8_t>(in->params.zero_point);
        ch32_tflm_qparams[i].out_scale = out->params.scale;
        ch32_tflm_qparams[i].out_zp    = static_cast<int8_t>(out->params.zero_point);

        (void)model_len[i];
    }
    printf("[TFLM-10] all seeds init done, dwt_init...\r\n");

    ch32_dwt_init();

    /* ⭐ 2026-10-18 Round 21: 上一轮发现 model 真的在算! out_before 显示上一轮 Invoke 留下
     *   0.389 / 0.0172 / 0 等 small value, 但当次 Invoke 后读到 0x737e6bc5 (2.016e+31).
     *   100% DCache read-after-write 问题. 加 4 个 read 模式:
     *   1. out_before  (Invalidate-only)
     *   2. after Invoke, CleanInvalidate, read
     *   3. after Invoke, NO cache op, read direct (raw cache hit)
     *   4. after Invoke, memcpy to stack, read stack (escape cache)
     *   + 搜 out->data.f 周围 64 字节看 model 真写到哪
     */
    {
        /* Hardcoded actual data from feature_extractor w#0 (transposed to (T=16, C=23)) */
        const float PAT_FE_DBG[CH32_IN_SIZE] = {
            /* T=0 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.261f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=1 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.367f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=2 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.636f,  0.377f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=3 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.739f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=4 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.261f,  0.377f,  0.272f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=5 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.053f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=6 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.636f,  0.377f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=7 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.739f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=8 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.261f,  0.377f,  0.272f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=9 */ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.636f, -0.281f, -0.036f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=10*/ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.636f,  0.377f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=11*/ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.739f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=12*/ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.261f,  0.377f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=13*/ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.367f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=14*/ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.636f,  0.377f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
            /* T=15*/ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.739f, -0.938f, -0.962f, 0.0f, 0.172f, 0.739f, -8.0f, 2.0f, 3.0f, 0.0f, 0.0f,
        };
        const char* PAT_NAMES[5] = {"ZEROS", "SIN-T", "SIN-C", "PEAK  ", "FE-W0 "};
        /* 4 varying patterns: 0=zero, 1=sin over t, 2=sin over c, 3=peak at t=8, 4=real FE data */
        for (int p = 0; p < 5; ++p) {
            TfLiteTensor* tin = g_interp[0]->input(0);
            /* Fill (T=16, C=23) - same layout as ch32_forward transpose output */
            for (int t = 0; t < 16; ++t) {
                for (int c = 0; c < 23; ++c) {
                    float v = 0.0f;
                    if (p == 0) {
                        v = 0.0f;
                    } else if (p == 1) {
                        v = sinf(t * 0.4f);
                    } else if (p == 2) {
                        v = sinf(c * 0.7f);
                    } else if (p == 3) {
                        v = (t == 8) ? 5.0f : 0.0f;  /* 尖峰 */
                    } else if (p == 4) {
                        v = PAT_FE_DBG[t * 23 + c];
                    }
                    tin->data.f[t * 23 + c] = v;
                }
            }
            SCB_CleanDCache_by_Addr((uint32_t*)tin->data.f, CH32_IN_SIZE * sizeof(float));

            /* 验证 input write 真的生效: 读回前 4 元素, 应该跟 pattern 一致 */
            SCB_InvalidateDCache_by_Addr((uint32_t*)tin->data.f, 4 * sizeof(float));
            uint32_t in_bits0, in_bits1, in_bits2, in_bits3;
            __builtin_memcpy(&in_bits0, &tin->data.f[0], 4);
            __builtin_memcpy(&in_bits1, &tin->data.f[1], 4);
            __builtin_memcpy(&in_bits2, &tin->data.f[2], 4);
            __builtin_memcpy(&in_bits3, &tin->data.f[3], 4);

            /* Print output BEFORE Invoke (arena initial state) */
            /* Round 22: key suspicion - output pointer may be relocated by memory planner
             * during Invoke. Always re-fetch g_interp[0]->output(0), never cache it. */
            {
                float* p_out_before = g_interp[0]->output(0)->data.f;
                SCB_InvalidateDCache_by_Addr((uint32_t*)p_out_before, sizeof(float));
                float out_before = p_out_before[0];
                uint32_t out_before_bits;
                __builtin_memcpy(&out_before_bits, &out_before, 4);
                uint32_t out_addr_before = (uint32_t)(uintptr_t)p_out_before;

                /* Invoke */
                TfLiteStatus st = g_interp[0]->Invoke();
                __DSB();

                /* Re-fetch output pointer, compare address */
                float* p_out_after = g_interp[0]->output(0)->data.f;
                uint32_t out_addr_after = (uint32_t)(uintptr_t)p_out_after;
                bool addr_changed = (out_addr_after != out_addr_before);

                /* Read mode A: direct read, no cache op */
                float lg_raw = p_out_after[0];
                uint32_t lg_raw_bits;
                __builtin_memcpy(&lg_raw_bits, &lg_raw, 4);

                /* Read mode B: CleanInvalidate 4B, then read */
                SCB_CleanInvalidateDCache_by_Addr((uint32_t*)p_out_after, sizeof(float));
                __DSB();
                float lg_ci = p_out_after[0];
                uint32_t lg_ci_bits;
                __builtin_memcpy(&lg_ci_bits, &lg_ci, 4);

                /* Read 32 bytes around output */
                uint32_t out32[8];
                __builtin_memcpy(out32, p_out_after, 32);

                printf("[SELF-TEST] %s in[0..3]=0x%08x%08x%08x%08x out_before=0x%08x
", PAT_NAMES[p], in_bits0, in_bits1, in_bits2, in_bits3, out_before_bits);
                printf("           addr_before=0x%08x addr_after=0x%08x %s Invoke=%d
", out_addr_before, out_addr_after, addr_changed ? "RELOCATED" : "same", (int)st);
                printf("           RAW=0x%08x CI=0x%08x
", lg_raw_bits, lg_ci_bits);
                printf("           32B=[0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x]
", out32[0], out32[1], out32[2], out32[3], out32[4], out32[5], out32[6], out32[7]);
            }
        }
        printf("[SELF-TEST] 关键判别: 7 个 read 模式 CI/RAW/STK/VOL/INV + out[0..3] 4 个 offset\r\n");
        printf("[SELF-TEST]   - 5 个 0x737e6bc5 出现 = stale cache 命中 (model 写到 0.389 但 cache 给 2.016e+31)\r\n");
        printf("[SELF-TEST]   - 5 个真实 value 出现 (e.g. 0x3ec761d8=0.389) = cache 被 invalidate 后正确读出, model 工作!\r\n");
        printf("[SELF-TEST]   - out[0]=对, out[1..3]=garbage = model 写到正确 offset 但有 overflow contamination\r\n");
    }

    printf("[TFLM-11] ch32_init done, return 1\r\n");

    return 1;
}

/* ----------------------------------------------------------------------------
 * ch32_forward  --  前向推理 (FP32 全程, 不量化)
 * 输入: x[CH32_IN_SIZE] FP32, layout = (in_channels=23, time=16) row-major
 *       (与 sliding_window_get() 输出一致)
 * 输出: 异常概率 in [0, 1] (单 seed 直接返回, 5-seed 取均值)
 *
 * ⭐ 2026-10-18 Layout 修复: x 是 (C, T), TFLM 模型期望 (B, T, C) = (1, 16, 23)
 *   之前用 raw memcpy 相当于把 (C, T) 字节按 (T, C) 解读, 模型拿到 transposed
 *   garbage → F1=0.5185, logit 攻击类 260 vs 正常类 324 (反向), sigmoid 后全 1.0.
 *   修法: 内联 transpose 把 x[c*16+t] 写到 in->data.f[t*23+c].
 *
 * ⭐ 2026-10-18 FP32 流程 (取代 INT8 quant/dequant):
 *   1. transpose (C, T) → (T, C) 到 in->data.f
 *   2. Invoke (FP32 全图)
 *   3. logit = output_fp32[0]  ← 不 dequant
 *   4. prob = sigmoid(logit)   ← 不需要 T scaling (FP32 logit 不饱和)
 * ------------------------------------------------------------------------- */
float ch32_forward(const float* x, size_t len) {
    if (len < CH32_IN_SIZE) return 0.0f;
    if (x == nullptr)       return 0.0f;

    float acc = 0.0f;

    /* ⭐ 2026-10-18 简化: 删 [TFLM-IN]/[TFLM-IQ] 调试打印 (FP32 已是真实值, 不需要验证) */
    static int s_fp32_dbg_n = 0;

    for (int i = 0; i < CH32_TFLM_NUM_SEEDS; ++i) {
        /* === 输入: 转置 (C, T) → (T, C) 给 TFLM === */
        TfLiteTensor* in = g_interp[i]->input(0);
        if (in->type != kTfLiteFloat32) {
            if (s_fp32_dbg_n < 3) {
                printf("[TFLM-ERR] seed[%d] input dtype=%d, expect FLOAT32\r\n",
                       i, (int)in->type);
            }
            return 0.0f;
        }
        if (i == 0 && s_fp32_dbg_n < 5) {
            printf("[TFLM-FP32-IN] w#%d fp32_ch12_t0..15=", s_fp32_dbg_n);
            for (int t = 0; t < 16; t++) printf("%.3f,", x[12*16+t]);
            printf("\r\n");
            /* ⭐ 2026-10-18 全 23 channel min/max/NaN/Inf 摘要:
             *   验证 sw_buf 真的填了所有 channels (不是 ch12-15 OK 但 ch0-11 0 或 NaN).
             *   之前 [FE-DBG] 只验证 ch12-15, layout 修复后 model 读 ch0-11,
             *   ch0-11 是 0/NaN 也会让 model 输出 garbage. */
            float mn[23] = {0}, mx[23] = {0};
            int   nan_cnt[23] = {0}, inf_cnt[23] = {0};
            for (int c = 0; c < 23; c++) {
                mn[c] =  x[c*16 + 0];
                mx[c] =  x[c*16 + 0];
                for (int t = 0; t < 16; t++) {
                    float v = x[c*16 + t];
                    if (v != v)              nan_cnt[c]++;     /* NaN check */
                    if (v >  1e30f || v < -1e30f) inf_cnt[c]++;
                    if (v < mn[c]) mn[c] = v;
                    if (v > mx[c]) mx[c] = v;
                }
            }
            printf("[TFLM-FP32-IN-RANGE] w#%d (ch: min..max NaN Inf):\r\n", s_fp32_dbg_n);
            for (int c = 0; c < 23; c++) {
                printf("  ch%02d: [%7.3f..%7.3f] NaN=%d Inf=%d\r\n",
                       c, mn[c], mx[c], nan_cnt[c], inf_cnt[c]);
            }
        }
        /* ⭐ 2026-10-18 layout fix: x 是 (C=23, T=16), TFLM 期望 (T=16, C=23)
         *   不能 raw memcpy — 必须 transpose 否则模型拿到 garbage. */
        for (int t = 0; t < 16; ++t) {
            for (int c = 0; c < 23; ++c) {
                in->data.f[t * 23 + c] = x[c * 16 + t];
            }
        }
        /* DCache write-back (H743 Cortex-M7 必加, 否则 DMA/CPU 不一致) */
        SCB_CleanDCache_by_Addr((uint32_t*)in->data.f, CH32_IN_SIZE * sizeof(float));

        /* ⭐ 2026-10-18 H743 Cortex-M7 DCache 兜底 (Round 2):
         *   上一轮用 SCB_InvalidateDCache_by_Addr 后, win#0 正常但 win#1,2,4 仍有 stale.
         *   怀疑 TFLM 内部多次写 output cache line, 中间状态被 invalidate 错过.
         *   修法: Invoke 前 invalidate (强制 bypass cache), Invoke 后 clean+invalidate
         *         (写穿 main memory + 丢 cache line, 彻底清状态). */
        TfLiteTensor* out = g_interp[i]->output(0);
        SCB_InvalidateDCache_by_Addr((uint32_t*)out->data.f, sizeof(float));

        /* === Invoke (FP32 全图) === */
        if (g_interp[i]->Invoke() != kTfLiteOk) {
            if (s_fp32_dbg_n < 3) {
                printf("[TFLM-ERR] seed[%d] Invoke != kTfLiteOk\r\n", i);
            }
            return 0.0f;
        }

        /* Round 2 兜底: clean+invalidate 一起做, cache 写穿 + 丢 cache line */
        SCB_CleanInvalidateDCache_by_Addr((uint32_t*)out->data.f, sizeof(float));

        /* === 输出: FP32 logit, 不 dequant === */
        if (out->type != kTfLiteFloat32) {
            if (s_fp32_dbg_n < 3) {
                printf("[TFLM-ERR] seed[%d] output dtype=%d, expect FLOAT32\r\n",
                       i, (int)out->type);
            }
            return 0.0f;
        }
        /* ⭐ 2026-10-18 hex 打印 (绕开 Keil AC6 vararg float promotion 隐患:
         *   AAPCS 要求 variadic float promote 到 double (8 bytes), 但 Keil AC6
         *   单精度 FPU 默认只 push 4 bytes, %f 读 8 bytes → 拿到邻近 stack garbage.
         *   logit 真实值是 4-byte FP32, 用 memcpy 复制到 uint32 再 %x 打印, 100% bit-exact. */
        float logit = out->data.f[0];
        /* ⭐ 2026-10-18 删 T scaling: FP32 logit 不会推到 sigmoid 饱和区
         *   (sigmoid(20) = 0.999999998, logit > 10 概率 0.9999+, 但 0.0001% 量级) */
        float    prob  = 1.0f / (1.0f + expf(-logit));
        acc += prob;

        /* ⭐ 2026-10-18 debug: 前 5 个 window 输出 logit + prob, 验证 FP32 跟 PC sim 1:1 */
        if (i == 0 && s_fp32_dbg_n < 5) {
            uint32_t logit_bits, prob_bits;
            __builtin_memcpy(&logit_bits, &logit, 4);
            __builtin_memcpy(&prob_bits, &prob, 4);
            printf("[TFLM-FP32-OUT] win#%d logit=%.3f logit_bits=0x%08x prob=%.4f prob_bits=0x%08x\r\n",
                   s_fp32_dbg_n, logit, logit_bits, prob, prob_bits);
            s_fp32_dbg_n++;
        }
    }

    return acc / static_cast<float>(CH32_TFLM_NUM_SEEDS);
}

/* ----------------------------------------------------------------------------
 * ch32_benchmark  --  DWT cycle counter 测 n_runs 次平均耗时
 * 必须先调 ch32_dwt_init()
 * ------------------------------------------------------------------------- */
uint32_t ch32_benchmark(int n_runs) {
    static const float dummy_x[CH32_IN_SIZE] = {0};
    volatile uint32_t t0 = 0, t1 = 0;
    uint32_t total = 0;
    for (int i = 0; i < n_runs; ++i) {
        t0 = DWT->CYCCNT;
        (void)ch32_forward(dummy_x, CH32_IN_SIZE);
        t1 = DWT->CYCCNT;
        total += (t1 - t0);
    }
    return total / static_cast<uint32_t>(n_runs);
}

/* ----------------------------------------------------------------------------
 * ch32_dwt_init  --  启用 DWT cycle counter
 * (TFLu 自己也会 init, 这里冗余 init 一次)
 * ------------------------------------------------------------------------- */
void ch32_dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* ----------------------------------------------------------------------------
 * ch32_get_qparams  --  把 q-params 返回给 PC 上位机 (UART log)
 * FP32 模式下 in_scale/in_zp 没意义, 调试用, 不影响生产
 * ------------------------------------------------------------------------- */
void ch32_get_qparams(int seed_idx, ch32_tflm_qparams_t* out) {
    if (seed_idx < 0 || seed_idx >= CH32_TFLM_NUM_SEEDS || out == nullptr) return;
    *out = ch32_tflm_qparams[seed_idx];
}

}  /* extern "C" */
