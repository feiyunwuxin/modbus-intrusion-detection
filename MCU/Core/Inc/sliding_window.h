/* ============================================================================
 * sliding_window.h  --  16 帧 × 18 维 滑窗环形 buffer (内部)
 *                      + 输出 23 维 (KEEP_23 切片 + clip ±10, 给 ch32_forward)
 *
 * ⭐ 2026-09-27 重构 (跟 PC 训练管线对齐):
 *   旧架构: sw_buf[23*16], 每帧推 23-dim, get 时只 broadcast 聚合
 *   新架构: sw_buf[18*16], 每帧推 18-dim 归一化, get 时算 6 个聚合 +
 *          组装 27-dim + KEEP_23 切片 + clip ±10
 *
 * 两套 buffer:
 *   - SW_DIM = 18 (内部 sw_buf 维度, feature_extractor 直接推 18 维归一化)
 *   - SW_OUT_DIM = 23 (输出维度, 给 ch32_forward, 跟模型 23-dim 输入对齐)
 *
 * 输出 layout (跟 ch32_forward 期望严格一致):
 *   out[c * SW_WINDOW + t] = feature[c] 在时间步 t 的值
 *   即 (in_channels=23, window=16) row-major
 *
 * 用法:
 *   sliding_window_init();
 *   sliding_window_push(feats_18);             // 每收到一帧调一次, feats_18 是 18 维
 *   if (sliding_window_ready()) {
 *       sliding_window_get(inference_input);  // 输出 23*16=368 floats 给 ch32_forward
 *   }
 * ============================================================================
 */
#ifndef SLIDING_WINDOW_H
#define SLIDING_WINDOW_H

#include <stdint.h>
#include <stddef.h>
#include "feature_extractor.h"
#include "ch32_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SW_WINDOW  CH32_WINDOW          /* 16 */

/* ⭐ 2026-09-27 重构:
 *   SW_DIM     = 内部 sw_buf 维度 (= FEAT_DIM = 18), feature_extractor 直接推 18 维
 *   SW_OUT_DIM = 模型输入维度 (= 23), ch32_forward 期望 23 维
 *   SW_OUT_SIZE = SW_OUT_DIM * SW_WINDOW = 368 (跟 ch32_forward 期望一致)
 *   旧 SW_SIZE = SW_DIM * SW_WINDOW (用于 sw_buf 静态数组, = 288)
 *
 * 这里定义 SW_OUT_DIM / SW_OUT_SIZE 给 modbus_ids.c 用, SW_SIZE 仍给 sw_buf 静态数组用 */
#define SW_DIM         FEAT_DIM         /* 18, 内部 sw_buf 维度 */
#define SW_SIZE        (SW_DIM * SW_WINDOW)         /* 288, sw_buf 静态数组大小 */
#define SW_OUT_DIM     CH32_IN_CHANNELS              /* 23, 模型输入维度 */
#define SW_OUT_SIZE    (SW_OUT_DIM * SW_WINDOW)     /* 368, ch32_forward 输入大小 */

/* KEEP_23 (23-dim model_idx → 27-dim source idx)
 * ⭐ 2026-09-27 重构: 从 feature_extractor.c 搬到 header,
 *   跟 scaler_export.h 的 KEEP_23 表注释保持完全一致 (跟 PC retrain_tcn_*.py 一致)
 *
 *   KEEP_23[:17] = 19-dim scaler idx (跳过 scaler[2]=length, scaler[3]=setpoint)
 *   KEEP_23[17..22] = 27-dim 聚合 idx
 *   [17] press_mean_w                    (27-dim[19])
 *   [18] crc_max_w                       (27-dim[21])
 *   [19] resp_count_w                    (27-dim[23], signed sum = -M ∈[-16,0])
 *   [20] cmd_resp_balance_w              (27-dim[24])
 *   [21] length_nunique_w                (27-dim[25], from sw_buf[2]=scaled length)
 *   [22] unusual_count_w                 (27-dim[26])
 */
#define SW_KEEP_23_COUNT  23
static const uint8_t sw_keep_23[SW_KEEP_23_COUNT] = {
     0,  /* [ 0] address                        (scaler[0])  */
     1,  /* [ 1] function                       (scaler[1])  */
     4,  /* [ 2] gain                           (scaler[4])  */
     5,  /* [ 3] reset_rate                     (scaler[5])  */
     6,  /* [ 4] deadband                       (scaler[6])  */
     7,  /* [ 5] cycle_time                     (scaler[7])  */
     8,  /* [ 6] rate                           (scaler[8])  */
     9,  /* [ 7] system_mode                    (scaler[9])  */
    10,  /* [ 8] control_scheme                 (scaler[10]) */
    11,  /* [ 9] pump                           (scaler[11]) */
    12,  /* [10] solenoid                       (scaler[12]) */
    13,  /* [11] pressure_measurement            (scaler[13]) */
    14,  /* [12] crc_rate                       (scaler[14]) */
    15,  /* [13] time_diff                      (scaler[15]) */
    16,  /* [14] time_since_last_same_addr_func (scaler[16]) */
    17,  /* [15] is_unusual_fc                  (scaler[17]) */
    18,  /* [16] is_response                    (scaler[18]) */
    19,  /* [17] press_mean_w                   (27-dim[19]) */
    21,  /* [18] crc_max_w                      (27-dim[21]) */
    23,  /* [19] resp_count_w                   (27-dim[23]) */
    24,  /* [20] cmd_resp_balance_w             (27-dim[24]) */
    25,  /* [21] length_nunique_w               (27-dim[25]) */
    26,  /* [22] unusual_count_w                (27-dim[26]) */
};

/* KEEP_18 (18-dim sw_buf → 19-dim source idx, skip setpoint@idx3)
 * 用于 sliding_window_get 里组装 27-dim 中间数组
 *
 *   sw_buf[ 0] ← scaler[0]   address
 *   sw_buf[ 1] ← scaler[1]   function
 *   sw_buf[ 2] ← scaler[2]   length       ← 用于 length_nunique_w
 *   sw_buf[ 3] ← scaler[4]   gain
 *   sw_buf[ 4] ← scaler[5]   reset_rate
 *   sw_buf[ 5] ← scaler[6]   deadband
 *   sw_buf[ 6] ← scaler[7]   cycle_time
 *   sw_buf[ 7] ← scaler[8]   rate
 *   sw_buf[ 8] ← scaler[9]   system_mode
 *   sw_buf[ 9] ← scaler[10]  control_scheme
 *   sw_buf[10] ← scaler[11]  pump
 *   sw_buf[11] ← scaler[12]  solenoid
 *   sw_buf[12] ← scaler[13]  pressure_measurement
 *   sw_buf[13] ← scaler[14]  crc_rate
 *   sw_buf[14] ← scaler[15]  time_diff
 *   sw_buf[15] ← scaler[16]  time_since_last_same_addr_func
 *   sw_buf[16] ← scaler[17]  is_unusual_fc
 *   sw_buf[17] ← scaler[18]  is_response
 *
 * 注意: scaler_export.h feat_keep_18[] 是反向的 (19-dim → 18-dim),
 * 这里 sw_buf_keep_18 是正向的 (18-dim → 19-dim), 用于查表.
 */
#define SW_KEEP_18_COUNT  18
static const uint8_t sw_buf_keep_18[SW_KEEP_18_COUNT] = {
     0,  /* [ 0] address     */
     1,  /* [ 1] function    */
     2,  /* [ 2] length      */
     4,  /* [ 3] gain        */
     5,  /* [ 4] reset_rate  */
     6,  /* [ 5] deadband    */
     7,  /* [ 6] cycle_time  */
     8,  /* [ 7] rate        */
     9,  /* [ 8] system_mode */
    10,  /* [ 9] control_scheme */
    11,  /* [10] pump        */
    12,  /* [11] solenoid    */
    13,  /* [12] pressure    */
    14,  /* [13] crc_rate    */
    15,  /* [14] time_diff   */
    16,  /* [15] time_since  */
    17,  /* [16] is_unusual  */
    18,  /* [17] is_response */
};

/* 初始化:清空 buffer + 计数 */
void sliding_window_init(void);

/**
 * 推入一帧 18-dim 特征
 * @param feats  长度 SW_DIM (18) 的特征数组 (已 RobustScaler 归一化)
 * @return       1=窗口已满 (可以推理), 0=未满
 */
uint8_t sliding_window_push(const float* feats);

/**
 * 检查窗口是否已满 (>= SW_WINDOW 帧)
 */
uint8_t sliding_window_ready(void);

/**
 * 取当前窗口内容,组装为 23-dim (KEEP_23 切片 + clip ±10) 给 ch32_forward
 * @param out  输出 23*16=368 floats, layout (23, 16) row-major
 * @return     1=成功 (window 已满), 0=未满
 *
 * 内部步骤:
 *   1. 拷 16 帧 × 18 维归一化到临时数组 (按时间顺序 t=0..15)
 *   2. 算 6 个聚合 (从归一化数据, 等同 PC preprocess_v2_scada.py:183-191)
 *      - press_mean_w = sw_buf[:, 12].mean() (18-dim idx 12 = pressure)
 *      - crc_max_w    = sw_buf[:, 13].max()  (18-dim idx 13 = crc_rate)
 *      - resp_signed  = sum(sw_buf[:, 17])   (signed sum = -M ∈[-16,0])
 *      - balance_w    = 1 - resp_signed / 8
 *      - length_nunique_w = unique count(sw_buf[:, 2]) (18-dim idx 2 = length)
 *      - unusual_signed = sum(sw_buf[:, 16])
 *   3. 组装 27-dim (broadcast 聚合到 16 行, 用 sw_buf_keep_18 填 0..18)
 *   4. KEEP_23 切片 → 23-dim (用 sw_keep_23 表)
 *   5. final clip ±10 (跟 PC retrain_tcn_*.py:136 一致)
 */
uint8_t sliding_window_get(float out[SW_OUT_SIZE]);

/**
 * 当前已推入帧数 (0..SW_WINDOW+)
 */
uint32_t sliding_window_count(void);

/* ⭐ 2026-09-28 DEBUG: 最近一次 sliding_window_get 的 23*16=368 floats 输出
 *   跟 PC 端 ch32_input bit-perfect 对比用. 定义在 sliding_window.c
 *   (aliased section .axi_sram, 1472B). */
extern float g_last_ch32_input[SW_OUT_SIZE];

#ifdef __cplusplus
}
#endif

#endif /* SLIDING_WINDOW_H */