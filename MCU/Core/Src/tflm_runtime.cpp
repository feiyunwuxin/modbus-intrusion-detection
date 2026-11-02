/* ============================================================================
 * tflm_runtime.cpp  --  TFLite Micro (TFLM) FP32 推理运行时 (唯一的 C++ TU)
 *
 * 定位: **论文/答辩用的 TFLM 对照**, 不是生产部署路径.
 *       生产路径是 Route E (手写 C kernel, ch32_inference.c).
 *       两条路径对外 API 完全一致 (ch32_init / ch32_forward / ...),
 *       由 ch32_inference_tflm.c 转接, app 层 (main.c / modbus_ids.c) 无需改动.
 *
 * ⭐⭐⭐ 这个实现是**从零新写**的, 设计前提直接来自手写 kernel 路线踩过的坑:
 *
 *   1. 唯一真源 = flatbuffer (model_tflm_fp32_s42_data.c), 不用手抄公式.
 *      算子集合是 9 种 (ADD/CONV_2D/EXPAND_DIMS/FULLY_CONNECTED/LOGISTIC/
 *      MEAN/MUL/RESHAPE/TRANSPOSE), 全部有 TFLM kernel, 一个不多一个不少.
 *      (旧路线 .tflite 里有 PACK/SHAPE/STRIDED_SLICE/UNPACK, OpResolver 漏注册
 *       -> AllocateTensors 在 Op[0] 直接失败. 这次在 PC 侧就用
 *       experiments/tflm_fp32/tools/tflm_compat_check.py 静态核对过.)
 *
 *   2. **零转置**: 图输入是 (1,23,16) channel-major, 跟 MCU 的 sw_buf[c*16+t]
 *      逐字节一致 -> memcpy 进去即可, 不做 (C,T)/(T,C) 转换.
 *      (历史 (C,T) vs (T,C) 陷阱: 一次 memcpy transpose 能把 F1 从 0.77 打到 0.52.)
 *
 *   3. **输出是 logit 不是概率**: ONNX 图里 3 个 Sigmoid 只在 SE 内部,
 *      图输出是 logit. sigmoid 在 MCU 端补, 与 Route E 的 ch32_forward 语义一致.
 *
 *   4. **DCache**: arena 在 D2 SRAM (可 DCache). CPU 写 input 后要 clean,
 *      Invoke 后读 output 前要 invalidate, 否则可能读到 AllocateTensors 之后的
 *      stale 行 (2026-10-18 旧路线的"5 个 controlled input 恒定"症状).
 *
 *   5. **必须检查 AllocateTensors / Invoke 的返回码**, 失败就拒绝推理,
 *      绝不拿 stale 输出继续跑 (旧路线就是忽略返回码后拿到垃圾值).
 *
 *   6. arena 放 .bss.tflm_d2 (D2 SRAM 256KB, 独立链接脚本 H743_TFLM_arena.sct),
 *      不跟 DTCM 抢 stack/heap.
 * ============================================================================
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <new>

#include "tflm_config.h"
#include "model_tflm_fp32_s42_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "stm32h7xx.h"

namespace {

/* ------------------------------------------------------------------ *
 * Arena
 * ------------------------------------------------------------------ *
 * 放在 .bss.tflm_d2 -> 链接脚本 MCU/MDK-ARM/H743_TFLM_arena.sct 把它
 * 放到 D2 SRAM (0x30000000, 256KB). 不用全局 new, 避免堆碎片.
 */
__attribute__((section(TFLM_ARENA_SECTION)))
uint8_t g_tflm_arena[TFLM_ARENA_BYTES];

/* Interpreter 自己的对象也放静态区 (placement new), 不吃堆 */
alignas(8) uint8_t g_interp_storage[sizeof(tflite::MicroInterpreter)];

tflite::MicroMutableOpResolver<TFLM_OP_RESOLVER_SIZE> g_resolver;
tflite::MicroInterpreter* g_interp = nullptr;

bool g_ready = false;
uint32_t g_arena_used = 0;

/* 5 个 controlled input (跟 PC 侧 experiments/tflm_fp32/artifacts/
 * tflm_selftest_ref.json 一一对应) */
struct ControlledCase {
  const char* name;
  float fill;
};
const ControlledCase kControlled[5] = {
    {"ZEROS", 0.0f}, {"ONES", 1.0f}, {"HALF", 0.5f}, {"NEG1", -1.0f}, {"TWOS", 2.0f},
};

/* DWT cycle counter (跟 Route E 的 ch32_dwt_init 同一套用法) */
volatile uint32_t g_dwt_available = 0;

}  // namespace

extern "C" {

/* ================================================================== *
 * 初始化
 * ================================================================== */
int tflm_init(void) {
  printf("\r\n[TFLM] init: model %d B, arena %u B\r\n",
         (int)TFLM_MODEL_S42_BYTES, (unsigned)TFLM_ARENA_BYTES);

  const tflite::Model* model = tflite::GetModel(TFLM_MODEL_S42_SYMBOL);
  if (model == nullptr) {
    printf("[TFLM] FATAL: GetModel() returned null\r\n");
    return -1;
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("[TFLM] FATAL: schema version %d != %d\r\n",
           (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
    return -2;
  }

  /* ---- 算子注册: 必须覆盖 flatbuffer 里出现的全部 op ---- *
   * 顺序无关, 但每个 Add* 的返回值都要查 (旧路线就是这里漏了). */
  if (g_resolver.AddConv2D() != kTfLiteOk) return -10;
  if (g_resolver.AddFullyConnected() != kTfLiteOk) return -11;
  if (g_resolver.AddMean() != kTfLiteOk) return -12;
  if (g_resolver.AddMul() != kTfLiteOk) return -13;
  if (g_resolver.AddAdd() != kTfLiteOk) return -14;
  if (g_resolver.AddLogistic() != kTfLiteOk) return -15;
  if (g_resolver.AddReshape() != kTfLiteOk) return -16;
  if (g_resolver.AddTranspose() != kTfLiteOk) return -17;
  if (g_resolver.AddExpandDims() != kTfLiteOk) return -18;
  printf("[TFLM] op resolver: 9 ops registered\r\n");

  g_interp = new (g_interp_storage)
      tflite::MicroInterpreter(model, g_resolver, g_tflm_arena,
                               TFLM_ARENA_BYTES);

  /* AllocateTensors 失败 = arena 不够 / op 缺失 / 图不合法.
   * **必须检查返回值**, 不能像旧路线那样忽略它然后拿 stale 输出. */
  TfLiteStatus st = g_interp->AllocateTensors();
  if (st != kTfLiteOk) {
    printf("[TFLM] FATAL: AllocateTensors failed, status=%d (arena too small "
           "or op missing?)\r\n",
           (int)st);
    g_interp = nullptr;
    return -20;
  }

  TfLiteTensor* in = g_interp->input(0);
  TfLiteTensor* out = g_interp->output(0);
  if (in == nullptr || out == nullptr) {
    printf("[TFLM] FATAL: input/output tensor null\r\n");
    g_interp = nullptr;
    return -21;
  }

  /* 形状/类型自检: 图输入必须正好是 (1,23,16) float32, 输出 (1,) float32 */
  const int in_dims = in->dims->size;
  const int out_dims = out->dims->size;
  int in_shape[4] = {0, 0, 0, 0};
  int out_shape[4] = {0, 0, 0, 0};
  for (int i = 0; i < in_dims && i < 4; i++) in_shape[i] = in->dims->data[i];
  for (int i = 0; i < out_dims && i < 4; i++) out_shape[i] = out->dims->data[i];
  printf("[TFLM] input  shape [%d,%d,%d,%d] type=%d\r\n", in_shape[0],
         in_shape[1], in_shape[2], in_shape[3], (int)in->type);
  printf("[TFLM] output shape [%d,%d,%d,%d] type=%d\r\n", out_shape[0],
         out_shape[1], out_shape[2], out_shape[3], (int)out->type);

  if (in_dims != 3 || in_shape[0] != 1 || in_shape[1] != TFLM_MODEL_S42_INPUT_DIMS ||
      in_shape[2] != TFLM_MODEL_S42_WINDOW || in->type != kTfLiteFloat32) {
    printf("[TFLM] FATAL: unexpected input tensor\r\n");
    g_interp = nullptr;
    return -22;
  }
  if (out_dims != 1 || out->type != kTfLiteFloat32) {
    printf("[TFLM] FATAL: unexpected output tensor\r\n");
    g_interp = nullptr;
    return -23;
  }

  g_arena_used = (uint32_t)g_interp->arena_used_bytes();
  g_ready = true;
  printf("[TFLM] AllocateTensors OK, arena used %u / %u B\r\n",
         (unsigned)g_arena_used, (unsigned)TFLM_ARENA_BYTES);
  return 0;
}

bool tflm_ready(void) { return g_ready; }

/* ================================================================== *
 * 前向推理
 * ================================================================== */

/* 跑一次, 返回 logit. 失败返回 NAN (调用方必须检查). */
float tflm_logit(const float* x) {
  if (!g_ready || g_interp == nullptr) return NAN;

  TfLiteTensor* in = g_interp->input(0);
  TfLiteTensor* out = g_interp->output(0);

  /* 零转置: MCU 的 sw_buf 是 (23,16) channel-major, 图输入也是 (1,23,16),
   * 内存布局完全一致 -> 直接 memcpy 368 个 float. */
  const uint32_t bytes = TFLM_MODEL_S42_IN_ELEMS * (uint32_t)sizeof(float);
  memcpy(in->data.f, x, bytes);

  /* DCache: CPU 写 input 到 (可能) 可 cache 的 arena -> 写穿, 保证
   * TFLM kernel 无论从 cache 还是 memory 读都拿到新值. */
  SCB_CleanDCache_by_Addr((uint32_t*)in->data.f, (int32_t)bytes);

  /* Invoke 前把 output 的 cache 行丢掉 (旧路线踩过: 读到 AllocateTensors
   * 之后的 stale garbage). */
  SCB_InvalidateDCache_by_Addr((uint32_t*)out->data.f, (int32_t)sizeof(float));

  TfLiteStatus st = g_interp->Invoke();
  if (st != kTfLiteOk) {
    printf("[TFLM] FATAL: Invoke failed status=%d\r\n", (int)st);
    return NAN;
  }

  /* 写穿 + 丢 cache, 双保险 (2026-10-18 Round 2 验证过的做法). */
  SCB_CleanInvalidateDCache_by_Addr((uint32_t*)out->data.f, (int32_t)sizeof(float));

  return out->data.f[0];
}

/* 返回概率 [0,1] (logit -> sigmoid), 与 Route E 的 ch32_forward 同语义. */
float tflm_forward(const float* x) {
  const float logit = tflm_logit(x);
  if (isnan(logit)) return NAN;
  return 1.0f / (1.0f + expf(-logit));
}

/* ================================================================== *
 * 上机自检: 5 个 controlled input
 * ================================================================== *
 * 跟 PC 侧 experiments/tflm_fp32/artifacts/tflm_selftest_ref.json 对拍.
 * 判读:
 *   5 个输出全恒定      -> 图/运行时在 MCU 上坏了
 *   5 个互不相同且与 PC 一致 -> 运行时 OK, 问题在数据通路
 *   5 个互不相同但与 PC 不一致 -> 数值实现差异 (算子语义)
 */
int tflm_selftest(void) {
  static float buf[TFLM_MODEL_S42_IN_ELEMS];
  printf("\r\n[TFLM-TEST] controlled input self-test (5 cases)\r\n");
  printf("[TFLM-TEST] %-8s %14s %14s\r\n", "case", "logit", "prob");

  int rc = 0;
  float first = 0.0f;
  bool have_first = false;
  bool all_same = true;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < TFLM_MODEL_S42_IN_ELEMS; j++) buf[j] = kControlled[i].fill;
    const float logit = tflm_logit(buf);
    if (isnan(logit)) {
      printf("[TFLM-TEST] %-8s Invoke FAILED\r\n", kControlled[i].name);
      rc = -1;
      continue;
    }
    const float prob = 1.0f / (1.0f + expf(-logit));
    printf("[TFLM-TEST] %-8s %14.6f %14.8f\r\n", kControlled[i].name, logit, prob);
    if (!have_first) {
      first = logit;
      have_first = true;
    } else if (logit != first) {
      all_same = false;
    }
  }

  if (all_same && have_first) {
    printf("[TFLM-TEST] WARN: all 5 outputs identical (%.6f) -> "
           "graph/runtime broken on MCU?\r\n", (double)first);
    rc = -2;
  } else {
    printf("[TFLM-TEST] OK: 5 outputs differ\r\n");
  }
  return rc;
}

/* ================================================================== *
 * 性能 / 内存信息
 * ================================================================== */
uint32_t tflm_benchmark(int n_runs) {
  if (!g_ready || n_runs <= 0) return 0;
  static float buf[TFLM_MODEL_S42_IN_ELEMS];
  for (int j = 0; j < TFLM_MODEL_S42_IN_ELEMS; j++) buf[j] = 0.01f * (float)(j % 7);

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  const uint32_t t0 = DWT->CYCCNT;
  for (int i = 0; i < n_runs; i++) {
    tflm_logit(buf);
  }
  const uint32_t t1 = DWT->CYCCNT;
  return (t1 - t0) / (uint32_t)n_runs;
}

void tflm_dwt_init(void) {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  g_dwt_available = 1;
}

void tflm_mem_info(uint32_t* arena_used, uint32_t* arena_total,
                   uint32_t* model_bytes, uint32_t* input_bytes) {
  if (arena_used) *arena_used = g_arena_used;
  if (arena_total) *arena_total = (uint32_t)TFLM_ARENA_BYTES;
  if (model_bytes) *model_bytes = (uint32_t)TFLM_MODEL_S42_BYTES;
  if (input_bytes) *input_bytes = TFLM_MODEL_S42_IN_ELEMS * (uint32_t)sizeof(float);
}

}  // extern "C"
