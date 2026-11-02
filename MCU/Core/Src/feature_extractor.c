/* ============================================================================
 * feature_extractor.c  --  18 维特征提取器实现
 * ============================================================================
 *
 * ⭐ 2026-09-27 重构 (跟 PC preprocess_v2_scada.py 完全对齐):
 *   旧架构: 算 23-dim RAW → 一次性 normalize 23 维 + 维护 hist[16] 缓存
 *   新架构: 算 18-dim RAW → 一次性 normalize 18 维 + 删 hist[] + 在
 *           sliding_window_get 里从 sw_buf 重算聚合
 *
 * 内存影响:
 *   sw_buf 23*16*4 = 1472B → 18*16*4 = 1152B (-320B)
 *   hist[] 256B → 删除 (-256B)
 *
 * 修复 2 个真 bug:
 *   1. feat_unusual_fc[] 改为 PC 的 {133,136,137,138,139,140,171}
 *      旧版 {7,8,11,12,17,20,21,24} 永远匹配不上 wire FC, 导致 is_unusual_fc=0
 *   2. length_nunique_w 现在从 sw_buf[2]=归一化 length 算 (旧版读 feats[2]=gain)
 *
 * 数据流 (跟 PC preprocess_v2_scada.py 一一对应):
 *   row (RAW 16-dim scada_row_t)
 *     ↓ feature_extractor_update
 *   RAW 18-dim (skip setpoint, 顺序跟 KEEP_18 一致)
 *     ↓ RobustScaler (raw - feat_mu_18[i]) / feat_sigma_18[i]
 *     ↓ clip ±10
 *   out_feats[18] 归一化
 *     ↓ sliding_window_push
 *   sw_buf[18*16]  ←──── sliding_window.c 维护
 *     ↓ sliding_window_get (frame_count >= 16)
 *   sw_buf[18*16] 归一化
 *     ↓ 算 6 聚合 (在归一化空间, 跟 PC line 183-191 一致)
 *     ↓ 组装 27-dim (broadcast 聚合到 16 行, 用 sw_buf_keep_18 表填 0..18)
 *     ↓ KEEP_23 切片 (用 sw_keep_23 表)
 *     ↓ clip ±10
 *   out[23*16] → ch32_forward
 * ============================================================================
 */

#include "feature_extractor.h"
#include "scaler_export.h"
#include <string.h>
#include <math.h>
#include <stdio.h>   /* ⭐ 2026-10-17 DEBUG printf */
#include "modbus_ids.h"   /* MODBUS_IDS_RAW18_DUMP 宏 — 必须 include 否则 SITE 4 块被 #if 0 删 */

/* --- 时间戳 cache: LRU 64 entries, round-robin --------------------------- *
 * ⭐ 2026-09-28 重构: 删了 256 槽 hash 数组, 改用 LRU 64 round-robin.
 *   旧版 (slave*16+func) % 256 跟 PC 训练端 address*1000+function (dict) 不一致,
 *   当 (s1,f1) != (s2,f2) 但 hash 相同时产生碰撞 → time_since_last_same_addr_func
 *   取错值 → is_unusual_fc/balance 派生特征错位 → 模型误判.
 *   新版 LRU 64: 线性扫描 64 槽, slave/func 完整匹配, 无 hash 碰撞.
 *   SCADA 实测 unique (slave,func) ≈ 4-8 种, 64 槽 round-robin 8 轮才覆盖首槽,
 *   实际不会丢失任何活条目.
 *
 * 用于 time_diff / time_since_last_same_addr_func. 时间戳是 double (f64 sec),
 * 跟 PC 下发 float64 sec bit-perfect, μs 级精度. */

#define ADDR_FUNC_LRU_SIZE  64u

/* 4 个并行数组 (避免 struct padding), 总 64*(8+1+1+1) = 704B
 *   ts_sec   double (8B), bit-perfect 跟 PC 端 pandas float64
 *   slave    uint8_t
 *   func     uint8_t
 *   valid    uint8_t (0=空槽, 1=已写入) */
static double s_lru_ts    [ADDR_FUNC_LRU_SIZE];
static uint8_t s_lru_slave[ADDR_FUNC_LRU_SIZE];
static uint8_t s_lru_func [ADDR_FUNC_LRU_SIZE];
static uint8_t s_lru_valid[ADDR_FUNC_LRU_SIZE];
static uint8_t s_lru_rr_idx = 0;   /* round-robin 写入指针 (0..63) */

/* 上一帧 timestamp, 用于 time_diff */
static double last_ts_sec = 0.0;

/* ⭐ 2026-09-27: 用 is_first_frame flag 替代 last_ts_sec == 0.0 检查
 *   旧检查 bug: ts[0]=0 时, last_ts_sec 被设成 0, 下一帧 last_ts_sec 还是 0 又触发 first-frame
 *   新设计: 第一次 update 强制 time_diff=0, 之后用 last_ts_sec 正常算 */
static uint8_t is_first_frame = 1;

/* ⭐⭐ 2026-10-17 DEBUG: file-local frame counter for window-aligned debug printf
 *   每调 feature_extractor_update() 增 1, 用于判断是否在 window 首帧触发 print.
 *   跟 s_dbg_win 配对: 5 个窗口 = 5 raw print + 5 out_feats print (完美配对). */
static uint32_t s_dbg_frame = 0;
static int     s_dbg_win   = 0;  /* 已 print 过的窗口数 (0..5) */

/* --- 工具 ----------------------------------------------------------------- */

static inline uint8_t is_unusual_fc(uint8_t fc)
{
    /* ⭐ 2026-09-27 修复: 旧版 {7,8,11,12,17,20,21,24} 跟 PC {133,136,137,138,139,140,171} 不一致
     *   旧值导致 is_unusual_fc() 永远返回 0, unusual_count_w feature 永远为 0
     *   新值 (feat_unusual_fc[] in scaler_export.h) 跟 PC preprocess_v2_scada.py:43 bit-perfect */
    for (uint8_t i = 0; i < FEAT_UNUSUAL_FC_COUNT; i++) {
        if (feat_unusual_fc[i] == fc) return 1;
    }
    return 0;
}

/* LRU 64 查找 + 写入 (返回 time_since_last_same)
 *
 * 算法: round-robin
 *   1. 线性扫描 64 槽, 找 (slave, func) 匹配 → 计算 ts - lru_ts[i], 更新该槽
 *   2. 找不到 → 写入当前 round-robin 指针指向的槽 (可能覆盖最老条目), 返回 0.0
 *
 * 跟 PC dict (preprocess_v2_scada.py:80-92) 行为对齐:
 *   - PC dict 是 "完全无碰撞" 全历史, 64 槽在 SCADA 实际 4-8 种 unique (slave,func)
 *     场景下不会撞; 64 >> 8 充足.
 *   - PC 首次出现 key → 0.0; LRU 同理 (valid=0 槽直接当 miss).
 *
 * 复杂度: O(N) = O(64) per frame, 64 次 double 比较 ≈ 1 µs (无 cache miss, AXI SRAM).
 */
static inline float lru_addr_func_lookup(uint8_t slave, uint8_t func, double ts)
{
    /* === 1. 扫描找匹配 === */
    for (uint8_t i = 0; i < ADDR_FUNC_LRU_SIZE; i++) {
        if (s_lru_valid[i] &&
            s_lru_slave[i] == slave &&
            s_lru_func[i]  == func) {
            float diff = (float)(ts - s_lru_ts[i]);
            s_lru_ts[i] = ts;                  /* 更新为本次时间 */
            return diff;
        }
    }
    /* === 2. 未命中 → 写入 round-robin 槽, 返回 0.0 === */
    uint8_t w = s_lru_rr_idx;
    s_lru_slave[w] = slave;
    s_lru_func[w]  = func;
    s_lru_ts[w]    = ts;
    s_lru_valid[w] = 1;
    s_lru_rr_idx   = (uint8_t)((w + 1u) % ADDR_FUNC_LRU_SIZE);
    return 0.0f;
}

/* --- 初始化 -------------------------------------------------------------- */

void feature_extractor_init(void)
{
    /* LRU 全 0 清空 (valid=0 表示空槽) */
    memset(s_lru_ts,    0, sizeof(s_lru_ts));
    memset(s_lru_slave, 0, sizeof(s_lru_slave));
    memset(s_lru_func,  0, sizeof(s_lru_func));
    memset(s_lru_valid, 0, sizeof(s_lru_valid));
    s_lru_rr_idx = 0;
    last_ts_sec  = 0.0;
    is_first_frame = 1;
}

/* --- 主 API:更新一帧, 输出 18 维归一化 -------------------------------- */

uint8_t feature_extractor_update(const scada_row_t* row, float out_feats[FEAT_DIM])
{
    if (row == NULL || out_feats == NULL) return 0;

    /* === 1. 计算 time_diff / time_since_last_same_addr_func ===
     *
     * 2026-09-24 第四轮修正 (round 2): row.timestamp_sec 是 double (f64 sec),
     *   PC 端 rtu_replay_gui.py 用 struct.pack('>d', t) 编码 (f64 BE),
     *   H743 端 modbus_rtu.c 用 be_f64 解码到 frame.raw17.time (double),
     *   modbus_ids.c 直接赋值 row.timestamp_sec = frame.raw17.time; (double).
     *   精度 μs 级, 保留 ARFF 6 位小数, 跟 preprocess_v2_scada.py:76 time_diff 单位一致.
     *
     * ⭐ 2026-09-27: 用 is_first_frame flag 替代 (last_ts_sec == 0.0) 检查,
     *   修复 cascading bug (ts[0]=0 时多次触发 first-frame 强制 0).
     */
    double ts = row->timestamp_sec;
    float time_diff_sec;
    if (is_first_frame) {
        time_diff_sec = 0.0f;
        is_first_frame = 0;
    } else {
        time_diff_sec = (float)(ts - last_ts_sec);
    }
    last_ts_sec = ts;

    /* LRU 64 一次完成 "查 + 写" (见 lru_addr_func_lookup 注释).
     * 旧逻辑里 (last_addr_func_ts[key] == 0.0) 的特殊 case 不再需要:
     *   - 命中: 直接返回差值 (≥ 0)
     *   - 未命中: LRU 函数内部返回 0.0f, 跟 PC dict 行为完全一致.
     * 新的查找消除了 hash 碰撞 + 死代码. */
    float time_since_same = lru_addr_func_lookup(row->slave, row->func, ts);

    /* === 2. build RAW 18-dim (顺序跟 KEEP_18 = sw_buf_keep_18 一致) ===
     *
     * 18-dim idx  →  source
     *   [ 0]      →  slave
     *   [ 1]      →  func
     *   [ 2]      →  payload_len      (用于 length_nunique_w)
     *   [ 3]      →  gain
     *   [ 4]      →  reset_rate
     *   [ 5]      →  deadband
     *   [ 6]      →  cycle_time
     *   [ 7]      →  rate
     *   [ 8]      →  system_mode
     *   [ 9]      →  control_scheme
     *   [10]      →  pump
     *   [11]      →  solenoid
     *   [12]      →  pressure_measurement (clip [0, 100] 先)
     *   [13]      →  crc_rate
     *   [14]      →  time_diff
     *   [15]      →  time_since_same
     *   [16]      →  is_unusual_fc (跟 PC UNUSUAL_FCS 一致)
     *   [17]      →  is_response (0/1, 后归一化时变成 {0, -1})
     *
     * 注: setpoint (RAW 17 维中第 4 列) **不进 sw_buf**, 也不进 KEEP_23, 也不进任何聚合.
     *    scaler_export.h feat_keep_18[] 跳过了 scaler[3]=setpoint, 故 feat_mu_18/feat_sigma_18
     *    里没有 setpoint 的参数.
     */
    float raw[FEAT_DIM];
    raw[0]  = (float)row->slave;
    raw[1]  = (float)row->func;
    raw[2]  = (float)row->payload_len;
    raw[3]  = row->gain;
    raw[4]  = row->reset_rate;
    raw[5]  = row->deadband;
    raw[6]  = row->cycle_time;
    raw[7]  = row->rate;
    raw[8]  = (float)row->system_mode;
    raw[9]  = (float)row->control_scheme;
    raw[10] = (float)row->pump;
    raw[11] = (float)row->solenoid;

    /* ⭐ 训练时 preprocess_v2_scada.py 对 pressure clip [0, 100] (line 132),
     *    否则 NMRI/CMRI 攻击段里 pressure=10^38 这种异常值会归一化后
     *    clip 到 10σ, 偏离 RobustScaler fit 分布 (fit 时已 clip).
     *    这里必须 clip, 跟训练管线对齐. */
    raw[12] = row->pressure_measurement;
    if (raw[12] < 0.0f)   raw[12] = 0.0f;
    if (raw[12] > 100.0f) raw[12] = 100.0f;

    raw[13] = (float)row->crc_rate;
    raw[14] = time_diff_sec;
    raw[15] = time_since_same;
    raw[16] = (float)is_unusual_fc(row->func);  /* ⭐ 修 bug */
    raw[17] = (float)(row->is_response ? 1 : 0);

    /* ⭐⭐ 2026-10-17 DEBUG: 定位 ch 12-15 garbage 根因
     *   'D' dump sw_buf 显示 ch 12-15 = (crc_rate, time_diff, time_since, is_unusual_fc)
     *   是 garbage (8.7e32, 2.02e29, -7.3e19, subnormal).
     *   打印 (1) raw[12..17] = modbus_rtu 解出来的 raw 值, 看是否本身已 garbage
     *            (2) out_feats[12..17] = 归一化后, 看 (raw-mu)/sigma 是否引入 garbage
     *   限每 16 帧 1 次 (1 个 window 1 次), 跟 window 对齐, 不阻塞.
     *   打印 5 个 window 后停, 够定位.
     *   s_dbg_frame 是 file-local counter (per-call 增 1), (cnt & 0x0F)==1 → window 首帧
     *   (cnt 1, 17, 33... = 第 1 帧 + 16 倍数 = window 首帧, sw_buf 推进后立即算).
     */
    s_dbg_frame++;
    if (s_dbg_win < 5 && (s_dbg_frame & 0x0F) == 1) {
        printf("[FE-DBG] w#%d frame#%lu raw[12..17]=crc=%u td=%.3f ts=%.3f uf=%d resp=%d | "
               "mu[12..17]={%.1f,%.1f,%.1f,%.1f,%.1f,%.1f} sg={%.1f,%.1f,%.1f,%.1f,%.1f,%.1f}\r\n",
               s_dbg_win, (unsigned long)s_dbg_frame,
               row->crc_rate, time_diff_sec, time_since_same,
               is_unusual_fc(row->func), row->is_response ? 1 : 0,
               feat_mu_18[12], feat_mu_18[13], feat_mu_18[14],
               feat_mu_18[15], feat_mu_18[16], feat_mu_18[17],
               feat_sigma_18[12], feat_sigma_18[13], feat_sigma_18[14],
               feat_sigma_18[15], feat_sigma_18[16], feat_sigma_18[17]);
        s_dbg_win++;
    }

    /* === 3. RobustScaler 归一化 (no clip!)
     *
     * 2026-09-27 重构修正: 训练管线 preprocess_v2_scada.py + retrain_tcn_*.py 的实际顺序:
     *   - RobustScaler.transform() → X_train_raw (无 clip)
     *   - window + aggregates → X_train_w (无 clip)
     *   - np.save() → X_test_binary_v2_scada_window16.npy (无 clip, idx 4 gain 可达 119)
     *   - retrain_tcn_*.py: X = X[:, :, KEEP_23] + np.clip(±10) → 模型输入
     * 所以 clip ±10 只在最终 23-dim 模型输入端做, **不在 per-row 18 维归一化时做**.
     * 旧版本在 feature_extractor 里 clip 是错的 (跟训练管线不对齐), 现在删掉.
     *
     * sliding_window_get 在最后一步 (组装 27-dim → KEEP_23 → clip ±10) 仍然做,
     * 跟 PC retrain_tcn_*.py:136 bit-perfect 对齐.
     */
    for (uint8_t i = 0; i < FEAT_DIM; i++) {
        out_feats[i] = (raw[i] - feat_mu_18[i]) / feat_sigma_18[i];
    }

    /* ⭐⭐ 2026-10-17 DEBUG (续): 归一化后 ch 12-17 看是否 garbage
     *   跟 raw print 同一窗口 (counter 共享), 配对看是否 raw 正常但 out garbage
     *   → 根因 = (raw-mu)/sigma 算式错或 mu/sigma 值错
     *   → 根因 = sw_buf 写后 DCache 写回丢失
     *   s_dbg_win 已被 raw print 推到 1..5, out_feats 在归一化 loop 之后, 同一窗口触发.
     *   条件 s_dbg_win >= 1 && s_dbg_win <= 5 → 5 raw + 5 out_feats 完美配对, 0 orphan.
     */
    if (s_dbg_win >= 1 && s_dbg_win <= 5) {
        /* ⭐ 2026-10-19 SITE 4 capture: temporarily disabled to fit all 7 suspect
         *   [RAW18-DBG] dumps in single 5KB terminal buffer. Per-frame dumps use
         *   ~50 bytes each (6 floats + label). Restore = delete "0 &&". */
        printf("[FE-DBG2] out[12..17]=%.3f %.3f %.3f %.3f %.3f %.3f\r\n",
               out_feats[12], out_feats[13], out_feats[14],
               out_feats[15], out_feats[16], out_feats[17]);
    }

#if MODBUS_IDS_RAW18_DUMP
    /* ⭐ 2026-10-19 SITE 4 instrumentation: 抓 18-dim 全部 raw + out_feats 全部,
     *   锁住 6 个 parity_compare_dump_v2 标红的 suspect windows (1, 2, 3, 7, 8, 15).
     *   H743 w#N 首帧 = frame_count 16N+1 = s_dbg_frame 16N+1.
     *   win_idx = (s_dbg_frame - 1) >> 4 = N 当 s_dbg_frame 在 [16N+1, 16(N+1)].
     *   决策:
     *     - raw 离谱 (e.g. 768, 7395, 500) → feature_extractor 算式 / scaler_export.h 错
     *     - raw 正常但 out 离谱 → 归一化算式 / feat_mu_18 / feat_sigma_18 错
     *     - raw + out 都正常 → 真因在 sw_buf_push / sliding_window_get 写路径
     *   每个 window 只 dump 1 次 (s_done bitmap 标记), 不阻塞, 共 6 次. */
    {
        static const uint8_t s_targets[6] = {1, 2, 3, 7, 8, 15};
        static uint8_t s_done = 0;
        uint32_t win_idx = (s_dbg_frame - 1u) >> 4;
        for (uint8_t k = 0; k < 6; k++) {
            if (s_targets[k] == win_idx && !(s_done & (1u << k))) {
                s_done |= (1u << k);
                printf("[RAW18-DBG w#%u frame#%u] raw18:",
                       (unsigned)win_idx, (unsigned)s_dbg_frame);
                for (uint8_t i = 0; i < FEAT_DIM; i++) printf(" %.4f", raw[i]);
                printf("\r\n[RAW18-DBG w#%u] out18:", (unsigned)win_idx);
                for (uint8_t i = 0; i < FEAT_DIM; i++) printf(" %.4f", out_feats[i]);
                printf("\r\n");
                break;
            }
        }
    }
#endif

    return 1;
}