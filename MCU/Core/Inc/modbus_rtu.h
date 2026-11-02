/* ============================================================================
 * modbus_rtu.h  --  Length-driven 17-dim 帧解析器 (MCU 端, 无动态分配)
 *                   适配 STM32H743 + HAL USART + DMA + 空闲中断接收
 * ============================================================================
 *
 * 2026-09-19 第九轮重构: 帧长度由 length 字段决定, 6 种格式.
 * PC 端 rtu_replay_gui.py 与 H743 端 modbus_rtu.c 同步.
 *
 * 帧格式 (变长, 共 = length + 12 字节, 2026-09-24 第四轮 time 4B→8B):
 *   [slave(1)] [func(1)] [length(1)] [features 4-15 (length B)] [is_resp(1)] [time(4)]
 *
 * features 4-15 (length B) 按 length 字段分 3 种编码 + 攻击格式:
 *   length = 16   → 10×1B uint8 BE (f4..f13) + 1×4B float32 BE (f14) + 1×2B uint16 BE (f15)
 *   length = 46   → 11×4B float32 BE (f4..f14)                       + 1×2B uint16 BE (f15)
 *   length = 90   → 11×8B float64 BE (f4..f14)                       + 1×2B uint16 BE (f15)
 *   length < 16   → (length-2)B 全 0x00 + 1×2B uint16 BE (CRC)         ⭐ 攻击信号
 *   其它          → (length-2)B 全 0x00 + 1×2B uint16 BE (CRC)         ⭐ 未知 length
 *
 * 数据集 IanArffDataset.csv length 分布:
 *   16 = 134k (全部 0 特征)      ← 攻击 (NMRI 类型)
 *   46 = 69k  (压力注入 overflow) ← 攻击 (CMRI 类型)
 *   90 = 64k  (真实浮点)         ← 正常 SCADA
 *   10 = 4k   (全部 0 + CRC)      ← 攻击 (截断注入)
 *   12 = 2k   (全部 0 + CRC)      ← 攻击 (截断注入)
 *   14 = 1k   (全部 0 + CRC)      ← 攻击 (截断注入)
 *
 * 帧末关键定位 (用户口头约定 2026-09-19):
 *   - 倒数第 5 字节 = frame[length+3] = is_response
 *   - 最后  4 字节 = frame[length+4..length+7] = time (uint32 BE)
 *
 * H743 端只做 sniff: 解析 17-dim, 不解释 FC, 不发指令. is_response 直接
 * 取自 frame[length+3] command 位, 不做 FC+payload_len 组合判定.
 *
 * 用法 (3 步):
 *   modbus_rtu_init();                               // 一次
 *   uint16_t n = modbus_rtu_parse(buf, len, &frame); // 每收到一帧
 *   if (n > 0) {
 *       ... 访问 frame.raw17 提取 17 维原始 ...
 *       // 攻击判定交给 modbus_ids_on_frame (TCN+SE 模型推理)
 *       // 不再有 length 硬规则 (2026-10-01 第二十二轮起)
 *   }
 * ============================================================================
 */
#ifndef MODBUS_RTU_H
#define MODBUS_RTU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- 帧常量 -------------------------------------------------------------- */
#define RAW17_HEADER_LEN    3      /* slave + func + length */
#define RAW17_ISRESP_LEN    1      /* is_response 字节 */
#define RAW17_TIME_LEN      8      /* time (float64 BE, 2026-09-24 第四轮 4B→8B) */

/* 整帧最大长度 = length=90 + 12 = 102 (预留 buffer, 2026-09-24 +4B for f64) */
#define RAW17_FRAME_MAX     102
/* 攻击格式最小长度 = length=10 + 12 = 22 (再小的截断视为无效, 2026-09-24 +4B) */
#define RAW17_FRAME_MIN     22

/* 兼容旧 API (供外部可能还在引用 MODBUS_RTU_*) */
#define MODBUS_RTU_MIN_FRAME   RAW17_FRAME_MIN
#define MODBUS_RTU_MAX_FRAME   RAW17_FRAME_MAX
#define MODBUS_RTU_BROADCAST   0
#define MODBUS_RTU_SLAVE_MIN   1
#define MODBUS_RTU_SLAVE_MAX   247

/* 攻击 length 阈值 (用户口头约定 2026-09-19) */
#define RAW17_ATTACK_LEN_THRESHOLD  16

/* --- 17-dim 原始数据 (解码后) -------------------------------------------- */
typedef struct {
    /* f1..f3: header */
    uint8_t  address;        /* f1 ← CSV col 0 */
    uint8_t  function;       /* f2 ← CSV col 1 */
    uint8_t  length;         /* f3 ← CSV col 2 (决定编码) */

    /* f4..f14: 11 个原始特征 (统一 float, 解码时按 length 分支 cast) */
    float    setpoint;       /* f4  ← CSV col 3 */
    float    gain;           /* f5  ← CSV col 4 */
    float    reset;          /* f6  ← CSV col 5 (reset rate) */
    float    deadband;       /* f7  ← CSV col 6 */
    float    cycle;          /* f8  ← CSV col 7 (cycle time) */
    float    rate;           /* f9  ← CSV col 8 */
    float    system_mode;    /* f10 ← CSV col 9 */
    float    control_scheme; /* f11 ← CSV col 10 */
    float    pump;           /* f12 ← CSV col 11 */
    float    solenoid;       /* f13 ← CSV col 12 */
    float    pressure;       /* f14 ← CSV col 13 (pressure_measurement) */

    /* f15: CRC rate (uint16 BE, 每种 length 末尾 2B) */
    uint16_t crc;

    /* f16: is_response (frame[length+3]) */
    uint8_t  is_response;    /* ⭐ CSV col 15 (command response) */

    /* f17: time (float64 BE, frame[length+4..length+11])
     * 2026-09-24 第四轮: uint32 (整数 sec) → float32 → float64.
     * 原因: float32 在 Unix 2026 时间戳 (~1.4e9) 精度只有 ~50sec, 完全无法
     * 保留 ARFF 6 位小数. float64 精度 μs 级, 训练管线 preprocess_v2_scada.py:76
     * time_diff 单位是秒 + 6 位小数, 与此对齐.
     * uint32 整数秒会丢精度 → H743 f[13] 归一化范围错位 → 模型饱和输出 attack. */
    double time;             /* unix time (sec, f64) */

    /* 已废弃 (2026-10-01 第二十二轮起不再 length 硬规则), 永远为 0, 仅保留字段兼容 ABI */
    uint8_t  is_attack;
} raw17_t;

/* --- 解析结果结构 --------------------------------------------------------- */
typedef struct {
    uint8_t  slave;          /* 1..247 */
    uint8_t  func;           /* 直接来自字节, H743 不做有效性判断 */
    uint8_t  length;         /* = raw17.length */
    uint16_t total_len;      /* = length + 12 (2026-09-24 第四轮, 旧: length + 8) */
    raw17_t  raw17;          /* 17-dim 解码结果 */
} modbus_frame_t;

/* --- 公共 API ------------------------------------------------------------ */

/**
 * 初始化 (目前无需初始化,留接口兼容后续扩展)
 */
void modbus_rtu_init(void);

/**
 * 校验 function code 是否在已知范围 (H743 不做判断, 永远返回 1)
 */
uint8_t modbus_is_known_func(uint8_t fc);

/**
 * 解析一帧 (本模块核心)
 * @param buf   接收 buffer (USART 空闲中断回调传入)
 * @param len   接收字节数 (必须 ≥ RAW17_FRAME_MIN)
 * @param out   输出解析结果
 * @return      成功 1, 失败 0 (长度非法或 length 字段越界)
 *
 * 调用方约定:
 *   - 在 USART 空闲中断中: SCB_InvalidateDCache_by_Addr(buf, len) 后再调用
 *   - buf 至少 RAW17_FRAME_MAX 字节
 */
uint8_t modbus_rtu_parse(const uint8_t* buf, uint16_t len, modbus_frame_t* out);

/**
 * 从已知 func/reg_addr/reg_count 中预解析 (用于回放 / 仿真)
 * @return 1=成功, 0=fail
 */
uint8_t modbus_extract_addr_count(uint8_t func, const uint8_t* data,
                                  uint16_t data_len,
                                  uint16_t* reg_addr, uint16_t* reg_count);

#ifdef __cplusplus
}
#endif

#endif /* MODBUS_RTU_H */
