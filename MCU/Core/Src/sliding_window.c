/* ============================================================================
 * sliding_window.c  --  16×18 滑窗环形 buffer 实现 (新架构, 2026-09-27)
 * ============================================================================
 *
 * ⭐ 2026-09-27 重构 (跟 PC preprocess_v2_scada.py 完全对齐):
 *   旧架构: sw_buf[23*16], 推 23-dim, get 时只 broadcast 聚合 (有 length_nunique_w 读错列 bug)
 *   新架构:
 *     - sw_buf[18*16] = 1152 B (-320B vs 旧 1472B)
 *     - 推 18-dim (从 feature_extractor_update, 已 RobustScaler 归一化)
 *     - get 时:
 *         1. 拷 16 帧 × 18 维按时间顺序
 *         2. 算 6 个聚合 (在归一化空间, 跟 PC line 183-191 bit-perfect)
 *         3. 组装 27-dim (broadcast 聚合 + 用 sw_buf_keep_18 表填 0..18, setpoint 占位 0)
 *         4. KEEP_23 切片 → 23-dim (用 sw_keep_23 表)
 *         5. final clip ±10 (跟 PC retrain_tcn_*.py:136 一致)
 *
 * 修复 1 个真 bug:
 *   - length_nunique_w 从 sw_buf[2]=归一化 length 算 (旧版读 out[2*16+t] 但当时
 *     sw_buf 存的是 23-dim, idx 2 是 gain, 不是 length)
 *     注意: length 归一化后是 float, 用 round 后 int 比较 (跟 PC 训练数据一致,
 *     preprocess_v2_scada.py 把 length 转 float64 经过 center_/scale_ 后还是
 *     "接近整数" 的浮点, round 不会引入 1σ 误差)
 *
 * 内存:
 *   - sw_buf[SW_SIZE] = SW_SIZE * sizeof(float) = 18*16*4 = 1152 B (.axi_sram)
 *   - 临时 27-dim 数组在 sliding_window_get 栈上 (27*16*4 = 1728 B),
 *     调用频率 1 / 16 帧, 可接受; 或在 AXI SRAM 静态分配.
 *     为简单起见用栈分配 (ch32_forward 本身也在栈上分配更大的激活缓冲,
 *     1.7KB 的 27-dim 不构成栈爆风险).
 */

#include "sliding_window.h"
#include <string.h>
#include <stdio.h>   /* printf for [DCACHE-DBG] (gated by MODBUS_IDS_DCACHE_INSTRUMENT) */
#include "main.h"    /* 拉入 stm32h7xx_hal.h -> core_cm7.h -> cachel1_armv7.h,
                      *   SCB_CleanInvalidateDCache_by_Addr 是 __STATIC_FORCEINLINE,
                      *   没有 .o 符号, 必须 include 才能 inlined. */
#include "modbus_ids.h"   /* MODBUS_IDS_DCACHE_INSTRUMENT 宏 + g_stats */

/* 静态分配 + aligned(32) + section(".axi_sram") 三件套 (防栈爆 + Cache 一致) */
__attribute__((aligned(32), section(".axi_sram")))
float sw_buf[SW_SIZE];   /* 18 * 16 = 288 floats = 1152 B */

static volatile uint32_t frame_count = 0;   /* 已推入总帧数 */
static volatile uint32_t write_idx   = 0;   /* 当前写入的列 (0..SW_WINDOW-1) */

/* ⭐ 2026-09-28 DEBUG: 缓存最近一次 sliding_window_get 输出 (23*16=368 floats)
 *   供 modbus_ids.c 在 alarm 时 dump 上报, 跟 PC 端算的 ch32_input bit-perfect 对比
 *   定位 H743 vs PC 概率漂移根因. 1152B 额外 RAM (AXI SRAM). */
__attribute__((aligned(32), section(".axi_sram")))
float g_last_ch32_input[SW_OUT_SIZE];   /* 23 * 16 = 368 floats = 1472 B */


void sliding_window_init(void)
{
    memset(sw_buf, 0, sizeof(sw_buf));
    memset(g_last_ch32_input, 0, sizeof(g_last_ch32_input));
    frame_count = 0;
    write_idx   = 0;
}

uint8_t sliding_window_push(const float* s_feats)
{
    if (s_feats == NULL) return 0;

    /* 在当前 write_idx 列写入 18 维特征 (新架构, 旧是 23 维) */
    uint32_t t = write_idx;
    for (uint32_t c = 0; c < SW_DIM; c++) {
        sw_buf[c * SW_WINDOW + t] = s_feats[c];
    }

    write_idx   = (write_idx + 1u) % SW_WINDOW;
    frame_count++;

    return sliding_window_ready();
}

uint8_t sliding_window_ready(void)
{
    return (frame_count >= SW_WINDOW) ? 1 : 0;
}

/* 按时间顺序读出 16 行 (从最旧到最新). sw_buf 是按通道排列 (c * SW_WINDOW + t) */
static inline float win_at_18(uint32_t c, uint32_t t)
{
    /* t 是当前帧的写入位置, 最旧帧是 (write_idx) 即下次将被覆盖的位置 */
    uint32_t oldest = (write_idx) % SW_WINDOW;
    uint32_t idx = (oldest + t) % SW_WINDOW;
    return sw_buf[c * SW_WINDOW + idx];
}

uint32_t sliding_window_count(void)
{
    return frame_count;
}

/* 18-dim channel indices (跟 KEEP_18 = sw_buf_keep_18[] 一致):
 *   [11] solenoid (不用)
 *   [12] pressure_measurement  (clip [0,100] 先, 然后归一化)
 *   [13] crc_rate
 *   [16] is_unusual_fc          (normal:0, unusual:1)
 *   [17] is_response            (master:0/slave:1 → 归一化 master:-1, slave:0)
 *   [ 2] length                 (用于 length_nunique_w)
 */
#define SW18_PRESSURE_IDX   12
#define SW18_CRC_IDX        13
#define SW18_UNUSUAL_IDX    16
#define SW18_RESPONSE_IDX   17
#define SW18_LENGTH_IDX     2

uint8_t sliding_window_get(float out[SW_OUT_SIZE])
{
    if (!sliding_window_ready()) return 0;

#if MODBUS_IDS_SWBUF_DUMP
    /* ⭐ 2026-10-20 SITE 5 instrumentation: 在 get 开头抓 sw_buf[c=12..15] 全部 16 帧.
     *   windows_total 在 ISR 里先 ++, 所以当前 win_idx = windows_total - 1 (0-indexed).
     *   配 SITE 4 [RAW18-DBG] (raw18 bit-perfect) → 若 sw_buf 也 bit-perfect,
     *   garbage 在 frame18→frame27 组装; 否则 push 时序/索引错.
     *   6 个 suspect windows (1,2,3,7,8,15) 各 dump 一次, 总 ~4.2KB, 5KB buffer 内.
     *   ⭐ 2026-10-20 SITE 5 验证完成: sw_buf[c=12..15] 全 16 帧 100% 干净, 排除
     *   sliding_window_push 写路径. 默认 0 (prod) 不打印. */
    {
        static const uint8_t s_targets[6] = {1, 2, 3, 7, 8, 15};
        static uint8_t s_done = 0;
        uint32_t win_idx = g_stats.windows_total - 1u;
        for (uint8_t k = 0; k < 6; k++) {
            if (s_targets[k] == win_idx && !(s_done & (1u << k))) {
                s_done |= (1u << k);
                printf("  sw_buf[c=12..15] x 16:\r\n");
                for (uint8_t c = 12; c <= 15; c++) {
                    printf("  c%u:", c);
                    for (uint8_t t = 0; t < SW_WINDOW; t++) {
                        printf(" %.4f", sw_buf[c * SW_WINDOW + t]);
                    }
                    printf("\r\n");
                }
                break;
            }
        }
    }
#endif

    /* === 1. 拷出当前 16 帧 (按时间顺序 t=0..15, 18-dim) === */
    float frame18[SW_WINDOW][SW_DIM];   /* (16, 18) 栈分配 */
    for (uint32_t t = 0; t < SW_WINDOW; t++) {
        for (uint32_t c = 0; c < SW_DIM; c++) {
            frame18[t][c] = win_at_18(c, t);
        }
    }

    /* === 2. 算 6 个聚合 (在归一化空间, 跟 PC preprocess_v2_scada.py:183-191 一致)
     *
     * 训练管线 (line 183-191):
     *   press_mean_w   = window['pressure measurement'].mean()
     *   crc_max_w      = window['crc rate'].max()
     *   resp_count_w   = window['is_response'].sum()   # signed sum = -M ∈[-16,0]
     *   cmd_count_w    = (1 - window['is_response']).sum()  # = 16 + resp_signed
     *   balance_w      = (cmd - resp) / 16 = (3M + R) / 16 = 1 - resp_signed / 8
     *   unusual_count  = window['is_unusual_fc'].sum()  # = +U ∈ [0, 16]
     *   length_nunique = len(unique(window['length']))
     *
     * PC 验证:
     *   is_resp_scaled ∈ {-1 (master), 0 (slave)}  (RobustScaler center=1, scale=1)
     *   is_unusual_scaled ∈ {0 (normal), 1 (unusual)}
     */
    float press_sum = 0.0f;
    float crc_max   = -1e30f;
    int32_t resp_signed   = 0;   /* sum(is_resp_scaled) = -M ∈ [-16, 0] */
    int32_t unusual_signed = 0;  /* sum(is_unusual_scaled) = +U ∈ [0, 16] */
    uint32_t seen_lens[SW_WINDOW];
    uint8_t  seen_n = 0;

    for (uint32_t t = 0; t < SW_WINDOW; t++) {
        float press_n      = frame18[t][SW18_PRESSURE_IDX];
        float crc_n        = frame18[t][SW18_CRC_IDX];
        float is_unusual_n = frame18[t][SW18_UNUSUAL_IDX];
        float is_resp_n    = frame18[t][SW18_RESPONSE_IDX];
        float length_n     = frame18[t][SW18_LENGTH_IDX];

        press_sum += press_n;
        if (crc_n > crc_max) crc_max = crc_n;
        resp_signed   += (int32_t)is_resp_n;     /* master:-1, slave:0 → sum=-M */
        unusual_signed += (int32_t)is_unusual_n; /* normal:0, unusual:1 → sum=+U */

        /* ⭐ length unique: 训练时 length 经 (raw - 16) / 30 归一化, 归一化前 round 到 int
         * 这里 length_n 是归一化后的 float, round 回 int (跟 PC 等价), O(n²) 查重 */
        int32_t len_int = (int32_t)(length_n + 0.5f);  /* round to int */
        uint8_t dup = 0;
        for (uint8_t k = 0; k < seen_n; k++) {
            if (seen_lens[k] == (uint32_t)len_int) { dup = 1; break; }
        }
        if (!dup && seen_n < SW_WINDOW) {
            seen_lens[seen_n++] = (uint32_t)len_int;
        }
    }

    float press_mean_w = press_sum / (float)SW_WINDOW;
    /* balance_w = 1 - resp_signed / 8 (推导见 .h 注释) */
    float balance_w    = 1.0f - (float)resp_signed / 8.0f;
    float resp_count_w = (float)resp_signed;       /* signed sum = -M */
    float unusual_w    = (float)unusual_signed;   /* +U ∈ [0, 16] */
    float nunique_w    = (float)seen_n;

    /* === 3. 组装 27-dim (broadcast 聚合到 16 行) ===
     * 27-dim 列顺序 (跟 PC preprocess_v2_scada.py:247-263 一致):
     *   [ 0..18] = 19-dim row-level (我们只有 18 个, setpoint@idx3 占位 0)
     *   [19] press_mean_w
     *   [20] crc_mean_w (训练算了, 即使 KEEP_23 不取 — 计算但不导出)
     *   [21] crc_max_w
     *   [22] cmd_count_w (训练算了, 即使 KEEP_23 不取 — 计算但不导出)
     *   [23] resp_count_w
     *   [24] cmd_resp_balance_w
     *   [25] length_nunique_w
     *   [26] unusual_count_w
     */
    float frame27[SW_WINDOW][27];   /* (16, 27) 栈分配 */
    memset(frame27, 0, sizeof(frame27));

    /* 3a. 0..18 列: 用 sw_buf_keep_18 表把 18-dim 数据填到 19-dim 对应位置
     *   sw_buf_keep_18[i] = 19-dim src idx, frame18[t][i] = 18-dim 值
     *   → frame27[t][sw_buf_keep_18[i]] = frame18[t][i]
     *
     *   19-dim[3] = setpoint 不被 sw_buf 覆盖 (18-dim skip), 故保持 memset 的 0.
     */
    for (uint32_t t = 0; t < SW_WINDOW; t++) {
        for (uint32_t i = 0; i < SW_DIM; i++) {
            frame27[t][sw_buf_keep_18[i]] = frame18[t][i];
        }
    }

    /* 3b. 19..26 列: broadcast 聚合到全部 16 行
     *
     * 注意: crc_mean_w 和 cmd_count_w 训练算了 (line 185-186),
     * 但 KEEP_23 不取它们, 23-dim 输出不会包含.
     * 这里仍算以便数值上跟训练数据 bit-perfect 对齐 (调试用).
     */
    float crc_sum = 0.0f;
    for (uint32_t t = 0; t < SW_WINDOW; t++) {
        crc_sum += frame18[t][SW18_CRC_IDX];
    }
    float crc_mean_w   = crc_sum / (float)SW_WINDOW;
    float cmd_count_w  = (float)(SW_WINDOW - unusual_signed - resp_signed); /* = 2M+R ∈ [16,32] */

    for (uint32_t t = 0; t < SW_WINDOW; t++) {
        frame27[t][19] = press_mean_w;
        frame27[t][20] = crc_mean_w;
        frame27[t][21] = crc_max;
        frame27[t][22] = cmd_count_w;
        frame27[t][23] = resp_count_w;
        frame27[t][24] = balance_w;
        frame27[t][25] = nunique_w;
        frame27[t][26] = unusual_w;
    }

    /* === 4. KEEP_23 切片 + clip ±10 → 输出 out[23*16] ===
     * out layout: out[c * SW_WINDOW + t] = 第 c 个特征在时间步 t 的值
     * frame27 layout: frame27[t][c] = 第 t 行第 c 列
     * → out[sw_keep_23[c] * SW_WINDOW + t] = frame27[t][sw_keep_23[c]]
     *   (KEEP_23 从 23-dim model_idx → 27-dim source idx)
     *
     * 然后逐元素 clip ±FEAT_CLIP_VAL (10.0), 跟 PC retrain_tcn_*.py:136 一致.
     */
    for (uint32_t c = 0; c < SW_OUT_DIM; c++) {
        uint32_t src = sw_keep_23[c];
        for (uint32_t t = 0; t < SW_WINDOW; t++) {
            float v = frame27[t][src];
            if (v >  FEAT_CLIP_VAL) v =  FEAT_CLIP_VAL;
            if (v < -FEAT_CLIP_VAL) v = -FEAT_CLIP_VAL;
            out[c * SW_WINDOW + t] = v;
        }
    }

#if MODBUS_IDS_DCACHE_INSTRUMENT
    /* ⭐ 2026-10-19 DCache instrumentation: 站点 1 — 算法刚算完 out, 看 ch 12-15
     *   是否已经 garbage (8.7e32/2.02e29/-7.3e19/subnormal 是已知 stale 表现).
     *   步骤: print (cache hit) → CleanInvalidateDCache (force RAM re-read)
     *   → 再 print. 若两次值不同 → CPU 读到 stale cache. */
    {
        printf("[DCACHE-DBG w#%u] SITE 1 sliding_window_get cache-hit: "
               "ch12_t0=%.4f ch13_t0=%.4f ch14_t0=%.4f ch15_t0=%.4f "
               "(ch=12=crc_rate,13=time_diff,14=time_since,15=is_unusual_fc)\r\n",
               (unsigned)g_stats.windows_total,
               out[12*SW_WINDOW+0], out[13*SW_WINDOW+0],
               out[14*SW_WINDOW+0], out[15*SW_WINDOW+0]);
        SCB_CleanInvalidateDCache_by_Addr((uint32_t *)out,
                                          (int32_t)(SW_OUT_SIZE * sizeof(float)));
        printf("[DCACHE-DBG w#%u] SITE 1 after CleanInvalidateDCache (force RAM): "
               "ch12_t0=%.4f ch13_t0=%.4f ch14_t0=%.4f ch15_t0=%.4f\r\n",
               (unsigned)g_stats.windows_total,
               out[12*SW_WINDOW+0], out[13*SW_WINDOW+0],
               out[14*SW_WINDOW+0], out[15*SW_WINDOW+0]);
    }
#endif

    /* ⭐ 2026-09-28 DEBUG: 拷贝 out → g_last_ch32_input 供 alarm dump */
    memcpy(g_last_ch32_input, out, sizeof(g_last_ch32_input));

    return 1;
}