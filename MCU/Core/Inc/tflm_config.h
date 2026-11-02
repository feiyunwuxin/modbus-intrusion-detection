/* ============================================================================
 * tflm_config.h  --  TFLM (TFLite Micro) FP32 后端的资源/模型配置
 *
 * 这个后端是 **论文/答辩用的 TFLM 对照**, 不是生产部署路径.
 * 生产部署路径是 Route E (手写 C kernel, ch32_inference.c), 见
 * [[project-route-e-migration]].
 *
 * 数值全部来自 experiments/tflm_fp32/ 的导出 + 校验:
 *   - 模型: experiments/tflm_fp32/artifacts/model_tflm_fp32_s42.tflite
 *           (sha256 eaee5340fbefa000..., 93,596 B, 50 op)
 *   - 算子集合: ADD / CONV_2D / EXPAND_DIMS / FULLY_CONNECTED / LOGISTIC /
 *               MEAN / MUL / RESHAPE / TRANSPOSE  (9 种, 全部有 TFLM kernel)
 *   - 图输入: (1, 23, 16) float32, **channel-major** (跟 MCU sw_buf[c*16+t]
 *             逐字节一致) -> MCU 端零转置
 *   - 图输出: (1,) float32 = **logit** (不是概率!), sigmoid 由 MCU 端补
 * ============================================================================
 */
#ifndef TFLM_CONFIG_H
#define TFLM_CONFIG_H

/* --- 模型 --- */
#define TFLM_MODEL_S42_SYMBOL      g_tflm_model_s42
#define TFLM_MODEL_S42_INPUT_DIMS  23      /* 特征维度 */
#define TFLM_MODEL_S42_WINDOW      16      /* 时间窗 */
#define TFLM_MODEL_S42_IN_ELEMS    (TFLM_MODEL_S42_INPUT_DIMS * TFLM_MODEL_S42_WINDOW) /* 368 */

/* --- 算子注册 --- *
 * 注册的 op 必须**正好**覆盖 flatbuffer 里出现的 op:
 *   ADD / CONV_2D / EXPAND_DIMS / FULLY_CONNECTED / LOGISTIC /
 *   MEAN / MUL / RESHAPE / TRANSPOSE
 * 少一个 -> AllocateTensors 在那个 op 上返回 kTfLiteError (旧路线的死法).
 * 多注册几个不报错, 但会拉进没用的 kernel, 所以这里按实际 op 数 + 2 余量.
 * (断言在 tflm_runtime.cpp 里做: 见 tflm_op_count())
 */
#define TFLM_OP_RESOLVER_SIZE 11

/* --- Arena --- *
 * TFLM 的 greedy planner 实际占用比"激活张量之和"大不少 (元数据 + scratch +
 * 每个 op 的 working buffer). 2026-10-16 实测: 单 seed 32KB 会 AllocateTensors
 * 成功但 Invoke 时 OOB 写穿到 stack -> HardFault; 60KB 才稳.
 * 这里给 96KB, 放在 D2 SRAM (256KB, 目前完全空着), 宁多勿少.
 */
#define TFLM_ARENA_BYTES  (96u * 1024u)

/* Arena 所在的 section 名. 必须跟链接脚本 MCU/MDK-ARM/H743_TFLM_arena.sct
 * 里的 `* (.bss.tflm_d2)` 对上. */
#define TFLM_ARENA_SECTION ".bss.tflm_d2"

#endif /* TFLM_CONFIG_H */
