/* ============================================================================
 * scaler_export.h  --  Real RobustScaler (mu, sigma) for H743 TCN+SE 23-dim
 *
 * 由 verify_h743_pipeline_v2.py 验证 + 手动生成, 基于:
 *   scaler_binary_v2_scada.joblib (19-dim RobustScaler, fit on train)
 *   preprocess_v2_scada.py UNUSUAL_FCS (8 个 command-only FC)
 *
 * 架构 (跟 PC preprocess_v2_scada.py 一一对应):
 *   1. feature_extractor_update(): 把 RAW 17-dim + length → 18 维, 应用
 *      (raw - feat_mu_18[i]) / feat_sigma_18[i] + clip ±10
 *   2. sliding_window_push(feats_18): 推 18 维到 sw_buf
 *   3. sliding_window_get(out[23*16]): 从 sw_buf 算 6 个聚合 + 切片 KEEP_23 + clip ±10
 *
 * KEEP_18 (18-dim sw_buf → 19-dim source idx, skip setpoint@idx3):
 *   [ 0] address                         (scaler[0])
 *   [ 1] function                        (scaler[1])
 *   [ 2] length                          (scaler[2])    <- 用于 length_nunique_w
 *   [ 3] gain                            (scaler[4])
 *   [ 4] reset rate                      (scaler[5])
 *   [ 5] deadband                        (scaler[6])
 *   [ 6] cycle time                      (scaler[7])
 *   [ 7] rate                            (scaler[8])
 *   [ 8] system mode                     (scaler[9])
 *   [ 9] control scheme                  (scaler[10])
 *   [10] pump                            (scaler[11])
 *   [11] solenoid                        (scaler[12])
 *   [12] pressure measurement            (scaler[13])
 *   [13] crc rate                        (scaler[14])
 *   [14] time_diff                       (scaler[15])
 *   [15] time_since_last_same_addr_func  (scaler[16])
 *   [16] is_unusual_fc                   (scaler[17])
 *   [17] is_response                     (scaler[18])
 *
 * KEEP_23 (23-dim model_idx → 19-dim / 27-dim source idx):
 *   [ 0..16] from scaler[KEEP_23[i]] for i in 0..16
 *            KEEP_23[:17] = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
 *   [17] press_mean_w                    (27-dim[19])
 *   [18] crc_max_w                       (27-dim[21])
 *   [19] resp_count_w                    (27-dim[23], signed sum = -M ∈[-16,0])
 *   [20] cmd_resp_balance_w              (27-dim[24])
 *   [21] length_nunique_w                (27-dim[25], from sw_buf[2]=scaled length)
 *   [22] unusual_count_w                 (27-dim[26])
 * ============================================================================
 */
#ifndef SCALER_EXPORT_H
#define SCALER_EXPORT_H

#include "feature_extractor.h"

#define FEAT_KEEP_18_COUNT  18

/* KEEP_18: 18-dim sw_buf idx → 19-dim scaler source idx
 * (skip scaler[3]=setpoint, which is dropped from 23-dim and unused in aggregates)
 * Use as: src = feat_keep_18[i]; v = (raw[src] - feat_mu_18[i]) / feat_sigma_18[i]; */
static const uint8_t feat_keep_18[FEAT_KEEP_18_COUNT] = {
    0,        /* [ 0] address */
    1,        /* [ 1] function */
    2,        /* [ 2] length */
    4,        /* [ 3] gain */
    5,        /* [ 4] reset rate */
    6,        /* [ 5] deadband */
    7,        /* [ 6] cycle time */
    8,        /* [ 7] rate */
    9,        /* [ 8] system mode */
    10,       /* [ 9] control scheme */
    11,       /* [10] pump */
    12,       /* [11] solenoid */
    13,       /* [12] pressure measurement */
    14,       /* [13] crc rate */
    15,       /* [14] time_diff */
    16,       /* [15] time_since_last_same_addr_func */
    17,       /* [16] is_unusual_fc */
    18,       /* [17] is_response */
};

/* RobustScaler center_ via KEEP_18 (scaler_binary_v2_scada.joblib) */
/* ⭐ 2026-10-15: 去掉 static, 让 feat_extractor.h 的 extern 声明匹配.
 *   scaler_export.h 只被 feature_extractor.c include, 多 .c 不会冲突. */
const float feat_mu_18[FEAT_KEEP_18_COUNT] = {
         4.000000f,   /* [ 0] address                        (scaler[0])  */
         3.000000f,   /* [ 1] function                       (scaler[1])  */
        16.000000f,   /* [ 2] length                         (scaler[2])  */
         0.000000f,   /* [ 3] gain                           (scaler[4])  */
         0.000000f,   /* [ 4] reset rate                     (scaler[5])  */
         0.000000f,   /* [ 5] deadband                       (scaler[6])  */
         0.000000f,   /* [ 6] cycle time                     (scaler[7])  */
         0.000000f,   /* [ 7] rate                           (scaler[8])  */
         0.000000f,   /* [ 8] system mode                    (scaler[9])  */
         0.000000f,   /* [ 9] control scheme                 (scaler[10]) */
         0.000000f,   /* [10] pump                           (scaler[11]) */
         0.000000f,   /* [11] solenoid                       (scaler[12]) */
         0.000000f,   /* [12] pressure measurement           (scaler[13]) */
     14135.000000f,   /* [13] crc rate                       (scaler[14]) */
         1.426832f,   /* [14] time_diff                      (scaler[15]) */
         3.116917f,   /* [15] time_since_last_same_addr_func (scaler[16]) */
         0.000000f,   /* [16] is_unusual_fc                  (scaler[17]) */
         1.000000f,   /* [17] is_response                    (scaler[18]) */
};

/* RobustScaler scale_ via KEEP_18 (scaler_binary_v2_scada.joblib) */
const float feat_sigma_18[FEAT_KEEP_18_COUNT] = {
         1.000000f,   /* [ 0] address                        (scaler[0])  */
        13.000000f,   /* [ 1] function                       (scaler[1])  */
        30.000000f,   /* [ 2] length                         (scaler[2])  */
         1.000000f,   /* [ 3] gain                           (scaler[4])  */
         1.000000f,   /* [ 4] reset rate                     (scaler[5])  */
         1.000000f,   /* [ 5] deadband                       (scaler[6])  */
         1.000000f,   /* [ 6] cycle time                     (scaler[7])  */
         1.000000f,   /* [ 7] rate                           (scaler[8])  */
         1.000000f,   /* [ 8] system mode                    (scaler[9])  */
         1.000000f,   /* [ 9] control scheme                 (scaler[10]) */
         1.000000f,   /* [10] pump                           (scaler[11]) */
         1.000000f,   /* [11] solenoid                       (scaler[12]) */
         1.000000f,   /* [12] pressure measurement           (scaler[13]) */
      4849.000000f,   /* [13] crc rate                       (scaler[14]) */
         1.521081f,   /* [14] time_diff                      (scaler[15]) */
         3.241358f,   /* [15] time_since_last_same_addr_func (scaler[16]) */
         1.000000f,   /* [16] is_unusual_fc                  (scaler[17]) */
         1.000000f,   /* [17] is_response                    (scaler[18]) */
};

/* UNUSUAL_FCS -- 与 preprocess_v2_scada.py UNUSUAL_FCS 一致 (7 个 command-only FC)
 * ⭐ 2026-09-27 重构: 旧版 {7,8,11,12,17,20,21,24} 跟 PC {136,171,139,133,137,138,140} 不一致
 *    旧值导致 is_unusual_fc() 永远返回 0, unusual_count_w feature 永远为 0
 *    新值与 PC preprocess_v2_scada.py:43 bit-perfect 一致 */
const uint8_t feat_unusual_fc[FEAT_UNUSUAL_FC_COUNT] = {
    133, 136, 137, 138, 139, 140, 171
};

#endif /* SCALER_EXPORT_H */