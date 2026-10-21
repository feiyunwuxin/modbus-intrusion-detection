/* ============================================================================
 * modbus_ids.h  --  Modbus 入侵检测 顶层状态机
 *
 * 集成关系:
 *   USART 空闲中断 → modbus_ids_on_frame(buf, len)
 *                     → modbus_rtu_parse
 *                     → feature_extractor_update (scada_row_t)
 *                     → sliding_window_push
 *                     → ch32_forward (>=16 帧时)
 *                     → alarm (LED + UART 报警)
 *
 * scada_row_t 的过程值字段 (gain/pressure 等) 由用户在 main.c 的
 * modbus_ids_fill_process_values() 钩子中填充 (从 Modbus response 或
 * 其它数据源,如 SPI ADC / CAN 总线).
 *
 * 报警阈值: CH32_BEST_THRESHOLD (定义在 ch32_inference.h, = 0.5)
 *   2026-10-17 曾临时调到 0.46 修 Recall, 2026-10-18 FP32 部署时改回训练默认 0.5;
 *   2026-10-21 三个根因全修后重扫阈值, 部署区 thr=0.5 仍是最优 (F1 0.8671 / FPR 5.91%).
 * ============================================================================
 */
#ifndef MODBUS_IDS_H
#define MODBUS_IDS_H

#include <stdint.h>
#include <stddef.h>
#include "feature_extractor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 报警 pin: 默认 PB0 (LED),可在 main.c 改 */
#define MODBUS_IDS_ALARM_PORT   GPIOB
#define MODBUS_IDS_ALARM_PIN    GPIO_PIN_0

/* 报警日志限流 (DEPRECATED 2026-10-01 Round 24):
 *   旧逻辑: 至少间隔 N 帧才输出一次 UART 报警 (避免 ML 误报刷屏).
 *   新逻辑: 节流已禁用 (modbus_ids.c 中 ids_print_alarm 改为每次内部 alarm 都调用).
 *   PC 端 Option A 逐帧回放能正确分类 delta=1 高频 alarm.
 *   常量保留仅为兼容旧代码引用, 不再生效. */
#define MODBUS_IDS_ALARM_MIN_GAP_FRAMES  1

/* ⭐ 2026-09-28: dump_feature 关闭原因 - fputc 是 HAL_UART_Transmit 阻塞 + TC 等待,
 *   每 ALARM dump 4.3KB 在 115200 baud 下阻塞 CPU ~370ms. 此期间 USART1 RX DMA
 *   buffer (256B) 累积 3-4 帧, 但 HAL_UARTEx_RxEventCallback 单帧处理 (modbus_ids_on_frame
 *   只解 1 帧), 后续帧被截断丢失. 验证: 老 firmware (无 dump) 77 alarms vs dump 版 1 alarm.
 *   调试时改 1 即可启用, 默认关闭以保证 IDS 性能. */
#ifndef MODBUS_IDS_DUMP_ENABLE
#define MODBUS_IDS_DUMP_ENABLE  0
#endif

/* ⭐ 2026-10-19 DCache instrumentation: 验证 sw_buf / g_inference_snapshot 写后
 *   cache coherency. 0=off (prod), 1=on (debug).
 *   启用后会在 3 个站点 printf "[DCACHE-DBG]" 报告 cache clean/invalidate 前后
 *   ch 12-15 (crc_rate / time_diff / time_since / is_unusual_fc) t=0 的值:
 *     - 站点 1 sliding_window.c:243 sliding_window_get 末尾 (算法算完)
 *     - 站点 2 modbus_ids.c:407 ch32_forward 调用前 (memcpy 之后)
 *     - 站点 3 ch32_inference.c:317 memcpy(buf_x, x_in) 之后 (input 落地)
 *   解读:
 *     - 同一站点两次 print (clean 前 vs 后) 数值相同 → cache coherency OK
 *     - 数值不同 → CPU 读到 cache stale 值, 真因就是 DCache write-back
 *       丢失, 修复 = ch32_forward 前显式 CleanDCache_by_Addr(local_snapshot)
 *   配套: 实测 ch 12-15 出现 8.7e32/2.02e29/-7.3e19/subnormal garbage 时,
 *   站点 1 的"clean 前"值若正常, 站点 2/3 出现 garbage → cache coherency bug. */
#ifndef MODBUS_IDS_DCACHE_INSTRUMENT
#define MODBUS_IDS_DCACHE_INSTRUMENT  0
#endif

/* ⭐ 2026-10-19 SITE 4 instrumentation: 在 feature_extractor_update 末尾
 *   抓 18-dim 原始 raw + 归一化 out_feats 全部 18 维. 配 parity_compare_dump_v2
 *   锁定的 6 个 suspect windows (H743 w#1, 2, 3, 7, 8, 15), 一次一 window.
 *   定位 768/7395/500 garbage 是 feature_extractor 算坏 (raw 已坏) 还是
 *   sw_buf_push 写坏 (raw 好但 out_feats 坏).
 *   0=off (prod), 1=on (debug).
 *   ⭐ 2026-10-20 SITE 4 验证完成: raw18 dump 全部 6 个窗口 100% bit-perfect,
 *   feature_extractor 排除. 默认 0 (prod). */
#ifndef MODBUS_IDS_RAW18_DUMP
#define MODBUS_IDS_RAW18_DUMP  0
#endif

/* ⭐ 2026-10-20 SITE 5 instrumentation: 在 sliding_window_get 开头抓 sw_buf
 *   4 个 suspect 通道 (c=12..15, pressure/crc/time_diff/time_since) 全部 16 帧.
 *   配 SITE 4 [RAW18-DBG] (raw18 干净, 算式 bit-perfect) → 若 sw_buf 也干净,
 *   真因在 frame18→frame27 组装; 若 sw_buf 出现 garbage, 真因在
 *   sliding_window_push 时序/索引错. 0=off (prod), 1=on (debug).
 *   ⭐ 2026-10-20 SITE 5 验证完成: sw_buf[c=12..15] 全 16 帧 100% 干净, 排除
 *   sliding_window_push 写路径. 默认 0 (prod). */
#ifndef MODBUS_IDS_SWBUF_DUMP
#define MODBUS_IDS_SWBUF_DUMP  0
#endif

/* 统计结构 (只读) */
/* ⭐ 2026-10-09 第五十轮 (Round 50): 改成 per-window 统计, 跟 PC 训练 1:1 对齐.
 *   旧: frames_attack (累计攻击帧数, sliding 每帧都可能推理) → 1 frame = 1 alarm
 *   新: windows_attack (累计攻击 window 数, non-overlapping 16 帧 1 次推理) → 1 window = 1 alarm
 *   跟 PC 训练 preprocess_v2_scada.py reshape(n//16, 16, -1) 完全对齐.
 */
typedef struct {
    uint32_t frames_total;              /* 总接收帧数 (含 CRC 错) */
    uint32_t frames_crc_err;            /* CRC 失败帧数 */
    uint32_t windows_total;             /* 处理的 window 数 (= frames_total / 16, 整除) */
    uint32_t windows_normal;            /* 模型判 normal 的 window 数 */
    uint32_t windows_attack;            /* 模型判 attack 的 window 数 (= 上行 [ALARM] 次数) */
    uint32_t frames_in_current_window;  /* 当前 window 已收帧数 (0..15), 凑齐 16 触发推理 */
    uint32_t last_prob_x1000;           /* 最近一次推理概率 ×1000 (0..1000) */
} modbus_ids_stats_t;

/* 初始化 (Modbus 解析 + 特征 + 窗口 + ch32 + 报警 GPIO) */
void modbus_ids_init(void);

/**
 * 主入口:每收到一帧 (USART 空闲中断回调) 调用一次
 * @param buf  USART DMA 接收 buffer
 * @param len  实际字节数
 *
 * 调用前必须先 SCB_InvalidateDCache_by_Addr(buf, len) — 由调用方负责
 */
void modbus_ids_on_frame(const uint8_t* buf, uint16_t len);

/**
 * 钩子:把 17-dim 帧中的过程值字段填到 row (默认实现从 raw17 读取).
 * 保留 weak 仅为兼容旧代码; 新项目无需重写.
 */
void modbus_ids_fill_process_values(scada_row_t* row);

/**
 * 钩子 (默认 weak 空函数):当 H743 作为 Modbus 主站轮询从站并收到
 * 响应时被调用, 让主站层清零失败计数器.
 *
 * 2026-09-19: H743 改为纯 sniff, 本函数在 default 实现里为空.
 * 保留以兼容启用主站轮询的旧部署 (IDS_SNIFF_ONLY=0).
 *
 * @param slave  收到的响应帧从站地址
 */
void modbus_ids_mark_response_received(uint8_t slave);

/* 取统计信息 (用于主循环定期打印) */
void modbus_ids_get_stats(modbus_ids_stats_t* out);

/* ⭐ 2026-10-19: sliding_window.c instrumentation 需要 windows_total 编号,
 *   暴露给其他模块用. g_stats 本身仍 modbus_ids.c private (写), 仅读权限开放. */
extern modbus_ids_stats_t g_stats;

/* 重置统计 */
void modbus_ids_reset_stats(void);

/* ⭐ 2026-10-17 新增: main loop 调, 消费 ISR 置的 inference pending flag.
 *   调 ch32_forward (TFLM 推理 ~2ms) + alarm 报告.
 *   关键: TFLM 推理从 ISR 移到 main loop, ISR 几 us 完成, USART 永远不死锁. */
void modbus_ids_run_inference_if_pending(void);

/* ⭐ 2026-10-17: ISR 凑齐 16 帧时 memcpy 23*16=368 floats 到此 snapshot,
 *   置 g_inference_pending=1, 立即返回 ISR.
 *   main loop 调 modbus_ids_run_inference_if_pending() 消费 snapshot.
 *   内存: 368 floats * 4B = 1472B, 放在 .axi_sram 段. */
extern float g_inference_snapshot[368];   /* SW_OUT_SIZE = 23*16 = 368 */
extern volatile uint8_t g_inference_pending;

/**
 * 最近 stale_threshold_ms 内是否收到过至少一帧
 * (替代旧的 process_state_is_fresh, 不依赖 FC3 response)
 */
uint8_t modbus_ids_is_fresh(uint32_t stale_threshold_ms);

/* --- 报警输出 (USART1 TX 走 printf) ------------------------------------- *
 * ⭐⭐ 第十六轮 (2026-09-19): 删除 JSON 上行, 改为最简单的 printf.
 *   唯一格式: "[ALARM] <count>\r\n"
 *     count = 累计攻击 window 数 (从 boot 起).
 *   ⭐ 第五十轮 (2026-10-09): 语义从 "累计攻击帧数" 改为 "累计攻击 window 数",
 *     跟 PC 训练 1 window = 1 binary decision 1:1 对齐.
 *   PC 端按行切分, 正则匹配 "ALARM N" 即可拿当前攻击 window 数 (delta = 当前 window 是否 attack).
 *   本地报警指示用 PB0 LED (ids_alarm_on()).
 *   不再有 hello 握手, 不再有 stats 上行, 不再有 IdsJsonParser.
 */

#ifdef __cplusplus
}
#endif

#endif /* MODBUS_IDS_H */