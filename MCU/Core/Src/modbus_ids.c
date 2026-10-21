/* ============================================================================
 * modbus_ids.c  --  Modbus 入侵检测 顶层状态机实现
 * ============================================================================
 *
 * 2026-09-19 第七轮重构:
 *   - 不再使用 process_state (FC3 response 更新).
 *   - 不再做 FC-based is_response 判定, 直接读 payload[27] command 位.
 *   - 不再做 Modbus 主站轮询, H743 永远是纯 sniff 模式.
 *   - 17 维原始数据全部来自 raw17_t (frame payload 解码), 不再依赖外部 hook
 *     提供过程值. process_state / FC3 解析路径全部废弃但保留 weak 函数名兼容.
 *
 * 2026-09-19 第十六轮:
 *   - 删除所有 ids_json_send_* (JSON 上行已废弃).
 *   - 报警输出改为最简单的 printf("[ALARM] %lu\r\n", count).
 *   - USART1 串口只发两类内容:
 *     (1) 启动 banner (modbus_ids_init 时一次)
 *     (2) 攻击事件 printf (ML 推理路径, 无 length 硬规则)
 *   - 本地报警指示 = PB0 LED (ids_alarm_on).
 *   - 没有 PC 端 GUI 面板, 没有 JSON 解析器, 没有 handshake.
 *
 * 2026-10-01 第二十一轮:
 *   - 删掉 prob=0.xxx 字段, 输出简化为 "[ALARM] <count>\r\n" (与第十六轮注释对齐).
 *   - 改 HAL_UART_Transmit_DMA 绕过 fputc 阻塞 (115200 baud 下 ~1.7ms/alarm 阻塞丢 RX).
 *   - 启动 banner 仍走 printf (一次性, 阻塞 OK); alarm 高频路径走 DMA (非阻塞).
 *   - TX 完成由 usart.c HAL_UART_TxCpltCallback 设 Uart1TxCplt=1 通知.
 *
 * 2026-10-01 第二十二轮 (本轮):
 *   - 不用 snprintf (ARM CC microlib 的 snprintf 走 _vfprintf → stdout → fputc,
 *     实际不写我们传的 buffer, 导致 DMA 读到栈帧残留.
 *     用户实测收到 0x03 0x00 0x00 0x00 0x04 0x00 0x00 0x00 0xa0 0x0a 0x00).
 *   - 改用 static __align(4) tx_buf + 手动 itoa, 完全确定性字节输出.
 *   - 加 SCB_CleanDCache_by_Addr: H7 write-back D-Cache, 不清 DMA 读到旧值.
 *
 * 2026-10-09 第五十轮 (Round 50):
 *   - 触发逻辑从 sliding (每帧推理) 改为 non-overlapping (16 帧 1 次推理).
 *   - 跟 PC 训练 preprocess_v2_scada.py reshape(n//16, 16, -1) 1:1 对齐.
 *   - 1 个 16 帧 window = 1 次推理 = 1 个 binary decision.
 *   - 1 个 attack window = 1 条 [ALARM] 上行.
 *   - 评估粒度从 per-alarm vs per-frame 改为 per-alarm vs per-WINDOW.
 *   - 推理次数从 274k → ~17k (16 倍节省), 响应延迟不变 (都要等 16 帧凑齐).
 */

#include "modbus_ids.h"
#include "modbus_rtu.h"
#include "sliding_window.h"
#include "ch32_inference.h"
#include "main.h"          /* HAL_GPIO_WritePin / huart1 / Uart1TxCplt */

/* 2026-09-21 DEBUG: 启用 1 → 在 USART1 打印 scada_row_t 全部字段 + 23 维 feats (前 16 帧).
 * 关闭 0 → 静默. 默认 0, 在 Keil 工程 Options → C/C++ → Define 加 DEBUG_DUMP_FEATS=1 开启. */
#ifndef DEBUG_DUMP_FEATS
#define DEBUG_DUMP_FEATS 0
#endif
#include <stdio.h>
#include <string.h>

/* --- 统计 ----------------------------------------------------------------- */
modbus_ids_stats_t g_stats = {0};
scada_row_t row;
/* ⭐ 2026-09-27 重构: feats 现在是 18 维 (FEAT_DIM=18), 归一化后的特征, 推入 sliding_window.
 *   sliding_window_get 输出 23 维 (SW_OUT_DIM=23), 装入 win[SW_OUT_SIZE] 给 ch32_forward. */
float feats[FEAT_DIM];
/* 报警限流 */
/* ⭐ 2026-10-01 第二十四轮 (Round 24): 节流禁用
 *   旧: MIN_GAP_FRAMES=4 让 H743 每 4 帧才发 1 次 [ALARM], 导致周期性短脉冲 (每 4 帧
 *       一次 attack run) 完全漏报 — 17 个 missed runs 全是这种模式.
 *   新: 每个内部 alarm 都通过 [ALARM] 上行. PC 端 Option A 已能正确处理 delta=1
 *       高频 alarm (逐帧回放 binary_history).
 *   串口带宽: 25fps × 50B = 1.2 KB/s << 115200 baud, 无压力.
 *   变量保留 (仅赋值, 不读取) 以兼容外部 weak 引用.
 */
uint32_t g_alarm_gap_counter = 0;

/* 最近解析出的 17-dim 原始数据 (供 weak hook 读取) */
raw17_t g_last_raw17 = {0};

/* ⭐ 2026-10-17 DEBUG: 验证 sliding_window_get 算 23-dim 是否跟 [FE-DBG2] 单帧归一化 bit-perfect.
 *   跟 [[project-2026-10-17-fe-dbg-results]] 实测结果配对 (算式 + mu/sigma 已 100% bit-perfect,
 *   但 'D' dump ch 12-15 仍 garbage, 根因在 sw_buf 写后环节).
 *   [SW-DBG] 在 main loop 消费 g_inference_snapshot 时打, 不能在 ISR 内打 (printf 阻塞 ~ms 级
 *   会破坏 USART DMA/IDLE 时序, 跟 [[project-2026-10-17-inference-moved-to-mainloop]] 同理).
 *   t=0 = window 最旧帧, 配对 [FE-DBG2] frame_1/17/33/49/65 (5 windows, s_dbg_frame & 0x0F == 1).
 *   预期: 跟 [FE-DBG2] out[12..15] bit-perfect (归一化空间 + clip ±10 但 out_feats 范围 [-1,6] 不会 clip). */
static int s_sw_dbg_call = 0;

/* ⭐ 2026-10-18 DEBUG: H743 0 FP + 19% Recall gap 根因定位
 *   PC sim 同 918 windows: 387 alarms, H743: 298 alarms, 89 PC-only (H743 silent)
 *   89 PC-only 全部 attack windows, PC prob 0.6-1.0. H743 跟 PC sim 输入数据流有 drift.
 *   离线 parity_h743_vs_pccsv.py 显示 0 diff (但脚本里 H743 view 把 length 放 18-dim[2] = payload_len
 *   跟实际 H743 一致, PC sim view 用 CSV 实际值, KEEP_23 不取 length 所以 23-dim 相同
 *   — 离线 parity 实际上只验证了 FN windows 的 length 都正常 = 16/46/90, 没验证
 *   异常 pressure/CRC 帧 (W=2 那种 subtle 攻击, 唯一 attack 在 F46, len=46, pressure=1.25)
 *   真正需要对比的: H743 sw_buf 实际值 vs CSV 行号对应的值
 *   [SW-DBG2] dump g_last_ch32_input (23-dim, 368 floats) 给指定 window index 列表
 *   配 [FE-DBG2] (已存在) 配对: 验证 sw_buf 输入跟 RTU 解码一致
 *   预期: 跟 PC sim Xtr_w (前 918 windows) bit-perfect. 如果不 match → H743 数据流有 drift
 *   限 8 个 windows, 避免撑爆 115200 串口带宽. */
static const int s_swdbg2_windows[] = {2, 3, 7, 15, 51, 0, 1, 8};  /* 5 FN + 3 TP */
static const int SWDBG2_N = (int)(sizeof(s_swdbg2_windows) / sizeof(s_swdbg2_windows[0]));
static int s_swdbg2_call = 0;

/* 最近一次成功解析帧的 tick */
uint32_t g_last_rx_ms = 0;

/* ⭐ 2026-10-17: TFLM 推理从 ISR 移到 main loop (修复 USART 死锁)
 *   现象: ISR 内 ch32_forward (TFLM) 耗时 ~2ms, 期间 DMA/IDLE 中断错乱,
 *          第 16 帧后 USART RX 永远失活 (g_idle_count 停在 19, abort HAL_TIMEOUT).
 *   修复: ISR 内只做滑动窗口 get + memcpy 到 snapshot, 置 g_inference_pending=1.
 *         main loop 检测 flag → 调 ch32_forward + 处理 alarm.
 *   为什么不会丢帧: ISR 仍可嵌套, main loop 的 ch32_forward 被打断后,
 *         TFLM 推理是纯函数, 输入相同输出相同, 继续执行无副作用. */
__attribute__((aligned(32), section(".axi_sram")))
float g_inference_snapshot[SW_OUT_SIZE];   /* 23 * 16 = 368 floats = 1472 B */
volatile uint8_t g_inference_pending = 0;

/* ⭐⭐ 2026-10-21: 窗口末帧快照 (修 GUI per-window 归窗错位).
 *
 *   问题: 之前 ids_print_alarm() 报的是 g_stats.frames_total —— 那是"当前已收帧总数".
 *         而推理被推迟到 main loop (2026-10-17 USART 死锁修复), ISR 快照完窗口后,
 *         若 main loop 还没跑到, 又来了 1 帧, frames_total 就变成 窗口末帧+1.
 *         实测 274628 帧回放里 3241/7287 (44.5%) 的 alarm 报成 16k+1 而不是 16k.
 *         PC 端 WindowEvaluator 用 W = (frame-1)//16 归窗 → 这 44.5% 全部落到
 *         下一个窗口 → 真 TP 被记成 FP (202) 或直接丢失 (614),
 *         GUI summary F1 被低报 0.07 / FPR 被高报 0.024.
 *
 *   修法: 在 ISR 触发推理的那一刻 (frames_total 恰好 == 窗口末帧号) 把它存进
 *         g_pending_frame, main loop 报 alarm 时用 g_pending_frame 而不是 frames_total.
 *         这样报出去的 frame 恒等于窗口末帧 (16 的倍数), PC 端公式无需改动.
 */
volatile uint32_t g_pending_frame = 0;

/* --- 默认钩子:从 raw17_t 提取过程值 ------------------------------------ *
 * 2026-09-19 第七轮重构:
 *   不再依赖 process_state. raw17_t 已经包含全部 17 维原始数据,
 *   此函数只是从 raw17 字段填到 scada_row_t (并保持 csv ×100 缩放规则).
 *   训练 csv 里的 setpoint/gain/reset/deadband/cycle/rate/pressure 都是 ×100
 *   后取整 (int16). raw17 解码时也按 ×100, 直接透传即可.
 *
 *   main.c 不再需要重写本函数. 保留 weak 是为兼容旧项目代码引用.
 */
__attribute__((weak)) void modbus_ids_fill_process_values(scada_row_t* row)
{
    /* scada_row_t 字段对齐说明 (feature_extractor.h):
     *   gain                 → csv col 4
     *   reset_rate           → csv col 5
     *   deadband             → csv col 6
     *   cycle_time           → csv col 7
     *   rate                 → csv col 8
     *   system_mode          → csv col 9
     *   control_scheme       → csv col 10
     *   pump                 → csv col 11
     *   solenoid             → csv col 12
     *   pressure_measurement → csv col 13
     *   crc_rate             → csv col 14 (由 modbus_ids_on_frame 覆盖)
     *
     * 注: setpoint (csv col 3) 不在 23-dim 模型特征中, 故 scada_row_t 没有此字段.
     */
    row->gain                 = g_last_raw17.gain;
    row->reset_rate           = g_last_raw17.reset;
    row->deadband             = g_last_raw17.deadband;
    row->cycle_time           = g_last_raw17.cycle;
    row->rate                 = g_last_raw17.rate;
    row->system_mode          = g_last_raw17.system_mode;
    row->control_scheme       = g_last_raw17.control_scheme;
    row->pump                 = g_last_raw17.pump;
    row->solenoid             = g_last_raw17.solenoid;
    row->pressure_measurement = g_last_raw17.pressure;
    /* crc_rate 由 modbus_ids_on_frame 从帧末 CRC 直接赋值, 此处占位 (uint16) */
    row->crc_rate             = 0u;
}

/* --- 默认钩子:空实现 (main.c 提供强符号覆盖) --------------------------- */
__attribute__((weak)) void modbus_ids_mark_response_received(uint8_t slave)
{
    (void)slave;
    /* 默认什么都不做 — H743 只 sniff, 不发指令, 无响应轮询 */
}

/* --- 报警 GPIO ------------------------------------------------------------ */

static void ids_alarm_on(void)
{
    HAL_GPIO_WritePin(MODBUS_IDS_ALARM_PORT, MODBUS_IDS_ALARM_PIN, GPIO_PIN_SET);
}

static void ids_alarm_off(void)
{
    HAL_GPIO_WritePin(MODBUS_IDS_ALARM_PORT, MODBUS_IDS_ALARM_PIN, GPIO_PIN_RESET);
}

/* --- 报警上行 (USART1 TX DMA) -------------------------------------------- *
 * ⭐⭐ 第二十二轮 (2026-10-01): 简化格式 + 改 DMA 发送 + 手动 itoa.
 *
 *   ⭐ 第六十一轮 (2026-10-11): 格式扩展为 "[ALARM] <count> prob=0.NNN frame=<K>\r\n"
 *     count = g_stats.windows_attack (累计攻击 window 数, 从 boot 起).
 *     prob  = sigmoid * 1000, 3 位 zero-pad (e.g., prob=0.873)
 *     frame = g_stats.frames_total (= 当前处理的帧序号, 1-indexed)
 *
 *   为什么加 frame_idx:
 *     2026-10-11 auto-40ms 全量测试发现 1-3 帧 pipeline lag, 导致 IdsLineParser
 *     把本属于 window W 的 alarm 归属给 W+1/W+2, 把 ~1200 个攻击 window 误标 FP.
 *     现在 H743 主动告知 PC frame_idx, PC 端 IdsLineParser 用 frame_idx 精确归属,
 *     消除 lag-induced FPR inflation. PC 端 _PROB_RE 已存在但 H743 没发, 现补发.
 *     PC 端将新增 _FRAME_RE 解析.
 *
 *   格式: "[ALARM] <count> prob=0.NNN frame=<K>\r\n"
 *     count   = g_stats.windows_attack
 *     prob    = sigmoid(model(win)) × 1000, 3 位 zero-pad (e.g., prob=0.873)
 *     frame   = g_stats.frames_total (= 当前处理的帧序号 1-indexed)
 *
 *   走 HAL_UART_Transmit_DMA 绕过 fputc 的阻塞路径:
 *     - fputc (usart.c) 内部是 HAL_UART_Transmit + TC 等待 → 阻塞 ~ms 级
 *     - 在 115200 baud 下, 一次 ALARM (~38B 含 frame) 阻塞 ~3.3ms, 而 H743 IDS 处理
 *       单帧仅 ~0.13ms. 阻塞期间 USART1 RX DMA buffer (256B) 累积 3-4 帧,
 *       HAL_UARTEx_RxEventCallback 单帧处理 (modbus_ids_on_frame 只解 1 帧),
 *       后续帧被截断丢失.
 *     - 改 DMA 后, CPU 仅配置 buffer + 启动传输, 立刻返回处理下一帧.
 *       TX 完成由 HAL_UART_TxCpltCallback 设 Uart1TxCplt=1 通知.
 *
 *   ⭐ 第二十二轮新坑: 不要用 snprintf.
 *     ARM CC microlib 的 snprintf 内部走 _vfprintf → stdout → fputc,
 *     实际上没把数据写入我们的 tx_buf 而是 stdout. tx_buf 留下栈帧残留
 *     (用户实测收到 0x03 0x00 0x00 0x00 0x04 0x00 0x00 0x00 0xa0 0x0a 0x00,
 *      11B 是栈帧垃圾不是 ASCII [ALARM] N).
 *     改用手动 itoa: 完全确定性字节, 不依赖 stdio 实现.
 *
 *   ⭐ DCache coherency (H7 必踩坑):
 *     H7 默认 write-back D-Cache. static tx_buf 在 .bss (AXI SRAM, cacheable).
 *     CPU 写 tx_buf 后 cache 有最新数据, 但 DMA 绕过 cache 直接读物理内存,
 *     读到旧值. 必须在 HAL_UART_Transmit_DMA 前 SCB_CleanDCache_by_Addr.
 *
 *   带宽: ~38B / 攻击事件 (含 frame=274628 6 位); DMA 非阻塞, 无 IDS 主循环影响.
 */
static char tx_buf[48] __attribute__((aligned(4)));  /* ⭐ 24→48 容纳 prob+frame (Round 61 + 36B). __align() 是 ARMCC 5 扩展, AC6 改 __attribute__((aligned)). */

/* 手动 uint32 → ASCII, 返回写入字节数 (不含 '\0') */
static uint32_t ids_u32_to_str(uint32_t v, char *out)
{
    char tmp[10];
    uint32_t n = 0;
    if (v == 0) {
        tmp[n++] = '0';
    } else {
        while (v > 0) {
            tmp[n++] = (char)('0' + (v % 10u));
            v /= 10u;
        }
    }
    /* 反转写入 out */
    char *orig = out;
    while (n > 0) {
        *out++ = tmp[--n];
    }
    return (uint32_t)(out - orig);
}

/* ⭐ 第六十一轮新增: 3 位 zero-pad uint32 -> ASCII (用于 prob=0.NNN)
 *   例: 0   -> "000"
 *       7   -> "007"
 *       123 -> "123"
 *
 * ⭐⭐⭐ 第六十二轮 (2026-10-12) BUG FIX: prob_x1000=1000 时输出 "000" 与 prob=0.0 冲突.
 *   现象: alarm 9/11/12 raw=0x3F800000 (即 prob=1.0 精确值), (uint32_t)(1.0*1000.0f)=1000,
 *         padded3(1000) = "000" → 打印 "prob=0.000" 而真实概率是 1.0.
 *   修复: 把 v clamp 到 999. 所有 prob >= 0.999 都打 "0.999" (或更精细的 "0.998/0.999"),
 *   IDS 阈值 (2026-10-13 已调到 0.9) 完全不变, PC 端 IdsLineParser 也无需改 (仍按 3 位解析).
 *
 *   不取 4 位 padded 输出: "prob=1.000" 长度 +1B, 改格式可能让 PC GUI 解析坏.
 *   不取 raw="1000" 表示: PC 端需要改 _PROB_RE.
 *   选 clamp-999: 最少改动, 损失精度仅在 prob=1.0 边界 (< 0.1% 区间, 不影响 IDS 决策).
 */
static uint32_t ids_u32_to_str_padded3(uint32_t v, char *out)
{
    /* Round 62 fix: prob_x1000 范围 [0, 1000], v=1000 时 3 位会塌成 "000". */
    if (v > 999u) v = 999u;
    out[0] = (char)('0' + (v / 100u) % 10u);
    out[1] = (char)('0' + (v /  10u) % 10u);
    out[2] = (char)('0' + (v       ) % 10u);
    return 3;
}

static void ids_print_alarm(uint32_t attack_count, uint32_t frame_total, uint32_t prob_x1000)
{
    char *p = tx_buf;
    /* "[ALARM] " (8 字节) */
    static const char prefix[] = "[ALARM] ";
    for (const char *q = prefix; *q; ++q) {
        *p++ = *q;
    }
    /* count (1-6 字节, 实际最多 5 位因 8971 ALARMS 上限) */
    p += ids_u32_to_str(attack_count, p);
    /* " " (1 字节) */
    *p++ = ' ';
    /* "prob=0." (7 字节) */
    static const char prob_str[] = "prob=0.";
    for (const char *q = prob_str; *q; ++q) {
        *p++ = *q;
    }
    /* 3 位 zero-pad prob (3 字节) */
    p += ids_u32_to_str_padded3(prob_x1000, p);
    /* " " (1 字节) */
    *p++ = ' ';
    /* "frame=" (6 字节) */
    static const char frame_str[] = "frame=";
    for (const char *q = frame_str; *q; ++q) {
        *p++ = *q;
    }
    /* frame_total (1-6 字节, 最大 274628 = 6 位) */
    p += ids_u32_to_str(frame_total, p);
    /* "\r\n" (2 字节) */
    *p++ = '\r';
    *p++ = '\n';
    uint32_t len = (uint32_t)(p - tx_buf);
    if (len >= sizeof(tx_buf)) return;  /* 防御性: 不应触发 (max 38B < 48B) */

    /* ⭐ H7 D-Cache coherency: 清 cache 让 DMA 读到最新数据
     *   (DCache 是 write-back, 不清 DMA 读到旧值). */
    SCB_CleanDCache_by_Addr((uint32_t *)tx_buf, (int32_t)len);

    /* DMA 发送 (非阻塞). HAL_UART_Transmit_DMA 内部:
     *   1. 配置 hdma_usart1_tx (DMA1 Stream1)
     *   2. 启动传输
     *   3. 完成后 HAL_UART_TxCpltCallback 设 Uart1TxCplt=1
     * 若上一次 DMA 尚未完成 (极端节流失效), HAL_BUSY 直接返回, 丢本条 ALARM
     * 不阻塞 (优于 fputc 的硬等). */
    HAL_StatusTypeDef rc = HAL_UART_Transmit_DMA(&huart1, (uint8_t *)tx_buf, len);
    if (rc != HAL_OK) {
        /* 静默丢消息, 不阻塞 IDS 主循环 */
    }
}

/* --- 初始化 --------------------------------------------------------------- */

void modbus_ids_init(void)
{
    modbus_rtu_init();
    feature_extractor_init();
    sliding_window_init();
    ids_alarm_off();
    /* ch32_init 已由 main.c 在 ch32_dwt_init() 之后调用 */
    memset(&g_stats, 0, sizeof(g_stats));
    memset(&g_last_raw17, 0, sizeof(g_last_raw17));
}

/* --- 主入口:每收到一帧调用 ----------------------------------------------- */

void modbus_ids_on_frame(const uint8_t* buf, uint16_t len)
{
    g_stats.frames_total++;

    /* 1. raw17 帧解析 (length 驱动, 6 种格式)
     * 命名 frame: 本函数局部 modbus_frame_t, 作用域仅在 modbus_ids_on_frame 内.
     * 下文 line 232 引入 float feats[FEAT_DIM] (feature_extractor 的输出),
     * 命名不冲突 (frame 是 struct, feats 是 float[], 类型也不同). */
    modbus_frame_t frame;
    modbus_rtu_parse(buf, len, &frame);
    /* 3. 缓存最新 raw17 (供 weak hook 读取) */
    g_last_raw17 = frame.raw17;
    /* 4. 组装 scada_row_t
     *
     * 2026-09-19 第九轮: length 驱动解码. features 4-14 已按 length switch
     * 解出为 float, 直接喂给 feature_extractor. is_response 取自
     * frame[length+3] (= raw17.is_response), 不做 FC+payload_len 组合判定.
     */

    row.slave        = frame.slave;
    row.func         = frame.func;
    row.is_response  = frame.raw17.is_response;
    row.payload_len  = frame.raw17.length;
    row.timestamp_sec = frame.raw17.time;   /* double→double 直传 */
    modbus_ids_fill_process_values(&row);
    row.crc_rate = (uint16_t)frame.raw17.crc;
    /* 5. 18 维特征提取 (旧架构是 23 维, 现对齐 PC 拆分: 18-dim 归一化 →
     *    sliding_window_get 时算 6 聚合 + KEEP_23 切片 + clip ±10 输出 23-dim) */
    feature_extractor_update(&row, feats);
    /* 6. 入滑窗 (仍每帧推入, 维持 sw_buf 环形 buffer 状态)
     *    ⭐ 第五十轮 (Round 50, 2026-10-09): 推理触发改为 16 帧 1 次 (non-overlapping)
     *    跟 PC 训练 preprocess_v2_scada.py reshape(n//16, 16, -1) 1:1 对齐.
     *    - 每帧 sliding_window_push 维持 sw_buf (16 帧环形 buffer)
     *    - frames_in_current_window 计数 (0..15)
     *    - 凑齐 16 帧才触发 1 次 ch32_forward
     *    - 1 个 window = 1 个 binary decision → 1 条 [ALARM] (if attack)
     */
    sliding_window_push(feats);
    g_stats.frames_in_current_window++;
    if (g_stats.frames_in_current_window >= SW_WINDOW) {
        /* 凑齐 1 个完整 window */
        g_stats.frames_in_current_window = 0;
        g_stats.windows_total++;

        if (!sliding_window_ready()) {
            /* 防御性: 不应发生, 因 sw_buf frame_count 已 >= 16 */
            return;
        }
        /* ⭐ 2026-10-17: ISR 内只做 get + memcpy snapshot + 置 flag.
         *   ch32_forward (TFLM 推理 ~2ms) 移到 main loop 的
         *   modbus_ids_run_inference_if_pending() 里, 避免 ISR 内
         *   阻塞 ~2ms 导致 USART DMA/IDLE 中断错乱 + RxState 卡死. */
        if (g_inference_pending == 0) {  /* 避免重复 trigger: main loop 还来不及消费 */
            if (sliding_window_get(g_inference_snapshot)) {
                g_inference_pending = 1;
                /* ⭐ 2026-10-21: 此刻 frames_total 恰好 == 本窗口末帧号 (16 的倍数).
                 *   快照下来, main loop 报 alarm 时用它, 避免迟 1 帧导致 PC 归窗错位. */
                g_pending_frame = g_stats.frames_total;
            }
        }
        /* 注意: 统计计数 (windows_total / frames_in_current_window reset) 在 ISR 完成.
         * 后续 ch32_forward + alarm 报告在 main loop 完成, 跟 windows_total 计数解耦. */
    }
}

/* ⭐ 2026-10-17 新增: main loop 调用, 消费 ISR 置的 inference pending flag.
 *   调 ch32_forward (TFLM 推理 ~2ms) + alarm 报告.
 *   这是修复 USART 死锁的关键: TFLM 推理从 ISR 移到 main loop, ISR 几 us 完成,
 *   USART DMA/IDLE 永远不死锁.
 *
 *   关键: 先 memcpy snapshot 到 local buffer 再清 flag, 避免 ch32_forward 期间
 *         ISR 写 g_inference_snapshot 覆盖正在用的数据.  */
void modbus_ids_run_inference_if_pending(void)
{
    if (g_inference_pending == 0) return;

    /* 1. memcpy snapshot 到 local buffer (栈上, 1472B, 不会跟 ISR 共享).
     *    memcpy 期间不关中断: 即使 ISR 进来写 g_inference_snapshot, 也是写 cache
     *    (D-Cache write-back), 不会破坏 memcpy 读源. memcpy 完成后 local 已经是
     *    一致 snapshot. */
    float local_snapshot[SW_OUT_SIZE];
    memcpy(local_snapshot, g_inference_snapshot, sizeof(local_snapshot));

#if MODBUS_IDS_DCACHE_INSTRUMENT
    /* ⭐ 2026-10-19 DCache instrumentation: 站点 2 — ch32_forward 调用前.
     *   步骤: print (cache hit) → CleanInvalidateDCache (force RAM re-read)
     *   → 再 print. 若两次值不同 → CPU 读到 stale cache → 真因在 cache.
     *   跟站点 1 / 3 配对诊断 ch 12-15 garbage 根因. */
    {
        printf("[DCACHE-DBG w#%u] SITE 2 ch32_forward before cache-hit: "
               "local[12*16+0]=%.4f local[13*16+0]=%.4f "
               "local[14*16+0]=%.4f local[15*16+0]=%.4f\r\n",
               (unsigned)g_stats.windows_total,
               local_snapshot[12*SW_WINDOW+0], local_snapshot[13*SW_WINDOW+0],
               local_snapshot[14*SW_WINDOW+0], local_snapshot[15*SW_WINDOW+0]);
        SCB_CleanInvalidateDCache_by_Addr((uint32_t *)local_snapshot,
                                          (int32_t)sizeof(local_snapshot));
        printf("[DCACHE-DBG w#%u] SITE 2 after CleanInvalidateDCache (force RAM): "
               "local[12*16+0]=%.4f local[13*16+0]=%.4f "
               "local[14*16+0]=%.4f local[15*16+0]=%.4f\r\n",
               (unsigned)g_stats.windows_total,
               local_snapshot[12*SW_WINDOW+0], local_snapshot[13*SW_WINDOW+0],
               local_snapshot[14*SW_WINDOW+0], local_snapshot[15*SW_WINDOW+0]);
    }
#endif

    /* 2. 清 flag, 让 ISR 可以 trigger 下一个 window.
     *    此时 local snapshot 已 memcpy 完, ISR 写 g_inference_snapshot 不会影响. */
    g_inference_pending = 0;

    /* ⭐ 2026-10-17 DEBUG: 打 23-dim out23[12..15] t=0 跟 [FE-DBG2] 配对定位 ch 12-15 garbage 根因
     *   KEEP_23[12..15] = {14,15,16,17} 对应 27-dim[14..17] = scaler[14..17] = {crc, td, ts, uf}
     *   t=0 是 window 最旧帧 = sw_buf[c*16 + (write_idx+0)%16]
     *   window 0..4 触发时 write_idx = 0 (16*N % 16 = 0), 所以 t=0 = sw_buf idx 0
     *   而 [FE-DBG] 触发 frame_1/17/33/49/65 写入 sw_buf 位置 (16N+1) % 16 = 1
     *   但 sliding_window_get 在 frame_16N+16 触发 (window N 凑齐), 此时 sw_buf[idx] = frame_16N+1
     *   即 [FE-DBG2] 打印的那一帧, 配对 bit-perfect.
     *   仅前 5 次触发, 0 orphan. */
    if (s_sw_dbg_call < 5) {
        printf("[SW-DBG] w#%d out23[12..15]_t0=%.3f %.3f %.3f %.3f "
               "(KEEP_23[12..15]={14,15,16,17}=crc/td/ts/uf; t=0=window 最旧帧=[FE-DBG2] 同一帧)\r\n",
               s_sw_dbg_call,
               local_snapshot[12*16+0], local_snapshot[13*16+0],
               local_snapshot[14*16+0], local_snapshot[15*16+0]);
        s_sw_dbg_call++;
    }

    /* 3. 调 ch32_forward (用 local snapshot, 2ms) */
    float prob = ch32_forward(local_snapshot, SW_OUT_SIZE);

    /* ⭐ 2026-10-19 SITE 4 capture: temporarily disabled [SW-DBG2] to fit 7
     *   [RAW18-DBG] dumps in single 5KB terminal buffer. SW-DBG2 dumps 23*16=368
     *   floats × ~7 chars = 2.5KB per window × 8 windows = 20KB total — far
     *   exceeds 5KB. Already have SW-DBG2 dumps from earlier (parity_compare_dump_v2
     *   found 768/7395/500 garbage), so this disable is safe for SITE 4 purpose.
     *   Restore = delete "#if 0 ... #endif".
     *   ⭐ 2026-10-20 [SW-DBG2] 验证完成: SITE 4 (raw18) + SITE 5 (sw_buf) + SW-DBG2 三层
     *   dump 全干净, 768/7395/500 garbage 在当前代码不存在 (老 firmware 残留).
     *   2026-10-20 重新 #if 0 关闭. */
#if 0
    {
        static int s_swdbg2_done[8] = {0};  /* 大小 = SWDBG2_N */
        uint32_t cur_w = g_stats.windows_total;
        for (int i = 0; i < SWDBG2_N; i++) {
            if (!s_swdbg2_done[i] && (int)cur_w == s_swdbg2_windows[i]) {
                printf("[SW-DBG2] w#%d prob=%.4f | out23 (23x16, ch-major):\r\n",
                       cur_w, prob);
                for (int c = 0; c < SW_OUT_DIM; c++) {
                    printf("  ch%02d:", c);
                    for (int t = 0; t < SW_WINDOW; t++) {
                        printf(" %.4f", local_snapshot[c * SW_WINDOW + t]);
                    }
                    printf("\r\n");
                }
                printf("[SW-DBG2] w#%d end (FN=%d, expected PC sim prob~%.2f)\r\n",
                       cur_w, (i < 5) ? 1 : 0,  /* first 5 are FN (label=1) */
                       (i == 0) ? 0.78f : (i == 1) ? 1.00f : (i == 2) ? 1.00f :
                       (i == 3) ? 0.91f : (i == 4) ? 0.63f : 0.50f);
                s_swdbg2_done[i] = 1;
                break;  /* 每个 window 调一次, 一次只 dump 一个 */
            }
        }
    }
#endif
    /* clamp prob 到 [0, 1] (跟原 on_frame 逻辑一致) */
    if (!(prob >= 0.0f)) {  /* NaN 检查 */
        prob = 0.0f;
    } else if (prob > 1.0f) {
        prob = 1.0f;
    } else if (prob < 0.0f) {
        prob = 0.0f;
    }
    g_stats.last_prob_x1000 = (uint32_t)(prob * 1000.0f);

    if (prob >= CH32_BEST_THRESHOLD) {
        /* === 攻击 window! === */
        g_stats.windows_attack++;
			#ifdef LED
        HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_1);           //red
			#endif
        ids_alarm_on();
        ids_print_alarm(g_stats.windows_attack, g_pending_frame, g_stats.last_prob_x1000);
    } else {
        /* 正常 window */
        g_stats.windows_normal++;
        g_alarm_gap_counter = 0;
    }
}

/* --- 统计 API ------------------------------------------------------------- */

void modbus_ids_get_stats(modbus_ids_stats_t* out)
{
    if (out) memcpy(out, &g_stats, sizeof(g_stats));
}

void modbus_ids_reset_stats(void)
{
    memset(&g_stats, 0, sizeof(g_stats));
}

/**
 * 返回是否在 stale_threshold_ms 内收到过至少一帧
 * (替代旧的 process_state_is_fresh, 不依赖 FC3 response).
 */
uint8_t modbus_ids_is_fresh(uint32_t stale_threshold_ms)
{
    if (g_last_rx_ms == 0) return 0;
    uint32_t now = HAL_GetTick();
    uint32_t age = now - g_last_rx_ms;
    return (age <= stale_threshold_ms) ? 1 : 0;
}