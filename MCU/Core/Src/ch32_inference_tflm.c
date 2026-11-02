/* ============================================================================
 * ch32_inference_tflm.c  --  TFLite Micro 后端 (C shim)
 *
 * 对外 API 与 Route E (ch32_inference.c) **完全一致**:
 *     ch32_init / ch32_forward / ch32_benchmark / ch32_dwt_init / ch32_get_mem_info
 * 所以 app 层 (main.c / modbus_ids.c) 一行都不用改, 两个后端可以直接对拍.
 *
 * 语义对齐 (跟 Route E 逐项核对):
 *   - 输入: (23,16) channel-major 数组, 368 个 float, 跟 sw_buf 一致
 *   - 输出: 概率 [0,1] (logit -> sigmoid), 阈值用 CH32_BEST_THRESHOLD = 0.5
 *   - 温度缩放: 同样除以 CH32_TEMPERATURE (FP32 部署下 = 1.0, 不做缩放)
 *
 * 本文件只做转接, 真正的解释器在 tflm_runtime.cpp (唯一的 C++ TU).
 * ============================================================================
 */
#include "ch32_inference.h"
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* tflm_runtime.cpp 提供的接口 */
extern int     tflm_init(void);
extern int     tflm_selftest(void);
extern float   tflm_logit(const float* x);
extern float   tflm_forward(const float* x);
extern uint32_t tflm_benchmark(int n_runs);
extern void    tflm_dwt_init(void);
extern void    tflm_mem_info(uint32_t* arena_used, uint32_t* arena_total,
                             uint32_t* model_bytes, uint32_t* input_bytes);

#ifdef __cplusplus
}
#endif

/* --- 公共 API ---------------------------------------------------------- */

void ch32_init(void)
{
    const int rc = tflm_init();
    if (rc != 0) {
        /* 失败就明确报出来, 不让 app 拿着没初始化好的解释器继续跑. */
        printf("\r\n[CH32-TFLM] init FAILED rc=%d -- inference disabled\r\n", rc);
    }
}

float ch32_forward(const float* x, size_t len)
{
    if (len < (size_t)CH32_IN_SIZE) {
        printf("[CH32-TFLM] forward: len=%u < %u, rejected\r\n",
               (unsigned)len, (unsigned)CH32_IN_SIZE);
        return 0.0f;
    }
    float logit = tflm_logit(x);
    if (isnan(logit)) {
        return 0.0f;
    }
    logit /= CH32_TEMPERATURE;
    return 1.0f / (1.0f + expf(-logit));
}

uint32_t ch32_benchmark(int n_runs)
{
    return tflm_benchmark(n_runs);
}

void ch32_dwt_init(void)
{
    tflm_dwt_init();
}

ch32_mem_info_t ch32_get_mem_info(void)
{
    ch32_mem_info_t mi;
    uint32_t arena_used = 0, arena_total = 0, model_bytes = 0, input_bytes = 0;
    tflm_mem_info(&arena_used, &arena_total, &model_bytes, &input_bytes);

    mi.weights_fp32_bytes = model_bytes;   /* .tflite flatbuffer (在 Flash) */
    mi.input_bytes       = input_bytes;   /* 1 个输入窗口 */
    mi.peak_act_bytes    = arena_used;    /* TFLM arena 实际占用 */
    mi.total_ram_bytes   = arena_used + input_bytes;
    return mi;
}

/* --- 上机自检 (给 'T' 命令用) ------------------------------------------ */
int ch32_tflm_selftest(void)
{
    return tflm_selftest();
}
