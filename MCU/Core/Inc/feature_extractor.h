/* ============================================================================
 * feature_extractor.h  --  18 维特征提取器 (MCU 端, 与 PC preprocess_v2_scada.py 一一对应)
 *
 * ⭐ 2026-09-27 重构 (跟 PC 训练管线对齐):
 *   旧架构: 算 23-dim RAW → 一次性归一化 23 维 (结构跟 PC 不一致)
 *   新架构: 算 18-dim RAW → 一次性归一化 18 维 (跳过 setpoint@idx3, 不进入聚合)
 *           → sw_buf 存 18-dim 归一化 → sliding_window_get 算 6 个聚合
 *           → 组装 27-dim → KEEP_23 切片 → clip ±10
 *   效果: H743 数据流结构跟 PC 完全一致 (scale → aggregate → slice → clip)
 *         同时修复 2 个 bug:
 *           - feat_unusual_fc[] 现在跟 PC preprocess_v2_scada.py UNUSUAL_FCS bit-perfect
 *           - length_nunique_w 现在从 length (sw_buf idx 2) 算, 不是 gain
 *
 * 18-dim 列表 (按 sw_buf 索引, 跟 scaler_export.h feat_keep_18[] 一致):
 *   [ 0] address                         per-row
 *   [ 1] function                        per-row
 *   [ 2] length                          per-row (用于 length_nunique_w)
 *   [ 3] gain                            控制参数
 *   [ 4] reset_rate                      控制参数
 *   [ 5] deadband                        控制参数
 *   [ 6] cycle_time                      控制参数
 *   [ 7] rate                            控制参数
 *   [ 8] system_mode                     控制参数
 *   [ 9] control_scheme                  控制参数
 *   [10] pump                            执行机构
 *   [11] solenoid                        执行机构
 *   [12] pressure_measurement            物理测量 (clip [0,100] 先)
 *   [13] crc_rate                        物理测量
 *   [14] time_diff                       时序 (sec, 帧间隔)
 *   [15] time_since_last_same_addr_func  时序 (sec, 同 (slave,func) 上次时间差)
 *   [16] is_unusual_fc                   SCADA (0/1, 跟 PC UNUSUAL_FCS 一致)
 *   [17] is_response                     SCADA (0/1, master=-1/slave=0 after scaling)
 *
 * 注: setpoint 不在 sw_buf 里 (也不在 KEEP_23, 也不用于任何聚合).
 *     RobustScaler 19-dim center_/scale_ 里 setpoint (idx 3) 对应的 mu/sigma 仍存在
 *     scaler_export.h 里以便未来扩展, 但当前不参与.
 *
 * 用法:
 *   1. feature_extractor_init();
 *   2. 每收到一帧:  feature_extractor_update(&row, &features);  // 输出 18 维 (已归一化)
 *   3. 把 features 送入 sliding_window_push();
 *   4. 当 sw_buf 满 16 帧后, sliding_window_get() 输出 23-dim (已 KEEP_23 切片 + clip)
 *
 * RobustScaler 归一化:训练时用 RobustScaler (median/IQR),此处用预存的 mu/sigma.
 * 需要从 scaler_binary_v2_scada.joblib 导出 center_ 和 scale_ (经 KEEP_18 索引)
 * → 写入 scaler_export.h 的 feat_mu_18[] / feat_sigma_18[] 常量数组.
 * ============================================================================
 */
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <stdint.h>
#include <stddef.h>
#include "modbus_rtu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ⭐ 2026-09-27 重构: FEAT_DIM 从 23 改成 18, sw_buf 存 18 维
 * (旧 FEAT_DIM=23 跟 sw_buf 旧维度混用, 容易引起阅读混淆, 现统一)
 *
 * 18 = KEEP_18 维数 = scaler_export.h feat_keep_18[] 长度
 * 23 = 模型输入维度 (sliding_window_get 输出维度)
 * 两者不要混用, 否则会出现 sw_buf 越界写入等严重 bug */
#define FEAT_DIM          18       /* ⭐ 新: feature_extractor_update 输出维度 (= KEEP_18 数) */
#define FEAT_OUT_DIM      18       /* 同 FEAT_DIM, 别名以便语义清晰 */
#define FEAT_WIN_ROWS     16       /* 窗口统计的历史行数 (与训练一致) */
#define FEAT_CLIP_VAL     10.0f    /* clip 上限 (训练时 np.clip(-10, 10)) */

/* KEEP_18 (18-dim sw_buf → 19-dim scaler source idx)
 * 直接在 feature_extractor.c 里 hardcode (跟 scaler_export.h feat_keep_18[] 一致),
 * 不放在这里避免循环依赖 */

/* RobustScaler 18-dim 参数 (scaler_export.h 内定义) */
extern const float feat_mu_18[FEAT_DIM];
extern const float feat_sigma_18[FEAT_DIM];

/* 已知 "unusual" FC 集合 (与 preprocess_v2_scada.py UNUSUAL_FCS 一致, 7 个值)
 * ⭐ 2026-09-27 重构: 旧 8 个 {7,8,11,12,17,20,21,24} 跟 PC {133,136,137,138,139,140,171} 不一致 */
#define FEAT_UNUSUAL_FC_COUNT  7
extern const uint8_t feat_unusual_fc[FEAT_UNUSUAL_FC_COUNT];

/* --- 原始输入 (从 Modbus 帧 + SCADA 过程值整合) ------------------------- *
 * 用户需要在收到每帧时填好此结构传给 feature_extractor_update().
 * - 帧字段来自 modbus_rtu 解析
 * - 过程值字段来自 SCADA 监控系统 (Modbus response / 其它通道)
 */
typedef struct {
    /* 来自 Modbus 帧解析 */
    uint8_t   slave;            /* address (1..247, 0=broadcast) */
    uint8_t   func;             /* function code */
    uint8_t   is_response;      /* 1=response, 0=request */
    uint8_t   payload_len;      /* Modbus data 字段长度 */
    double    timestamp_sec;    /* ⭐ 2026-09-24 第四轮 round 2: 字段名/类型都改了.
                                  *   旧: uint32_t timestamp_ms (HAL_GetTick ms, 整数 sec, f32 bits).
                                  *   新: double timestamp_sec (PC 下发 f64 sec, 精度 μs).
                                  *   PC 路径 modbus_ids.c: row.timestamp_sec = frame.raw17.time;
                                  *   feature_extractor.c: double ts = row.timestamp_sec; 直读.
                                  *   HAL_GetTick() 调用方需 /1000.0 转 sec 或自维护. */
    /* SCADA 过程值 — 由用户提供 (从 Modbus response 或外部信号源) */
    float     gain;
    float     reset_rate;
    float     deadband;
    float     cycle_time;
    float     rate;
    uint8_t   system_mode;
    uint8_t   control_scheme;
    uint8_t   pump;
    uint8_t   solenoid;
    float     pressure_measurement;
    uint16_t  crc_rate;         /* 累计 CRC 错误数 (CSV 'crc rate' 列, 范围 12336..17990).
                                  * 类型跟 PC 端 uint16 BE 编码一致 (rtu_replay_gui.py:203),
                                  * 跟 modbus_rtu.h raw17_t.crc 字段对齐. */
} scada_row_t;

/* 初始化 (清空时间戳 cache) */
void feature_extractor_init(void);

/**
 * 更新一帧, 输出 18-dim 特征 (已 RobustScaler 归一化 + clip ±10)
 * @param row        输入 SCADA 行
 * @param out_feats  输出 18 维特征 (已归一化), 顺序跟 KEEP_18 一致
 * @return           1=成功, 0=输入非法
 *
 * 内部维护 (用于时序特征):
 *   - last_ts_sec: 上一帧 timestamp
 *   - last_addr_func_ts[]: hash[(slave,func)] → 上次同 (slave,func) 的 timestamp
 *
 * 不再维护 hist[] (旧架构的窗口聚合缓存, 现改在 sliding_window_get 里算).
 */
uint8_t feature_extractor_update(const scada_row_t* row, float out_feats[FEAT_DIM]);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_EXTRACTOR_H */