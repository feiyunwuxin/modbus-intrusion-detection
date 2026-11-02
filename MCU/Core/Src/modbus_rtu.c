/* ============================================================================
 * modbus_rtu.c  --  Length-driven 17-dim 帧解析器实现
 * ============================================================================
 *
 * 2026-10-01 第二十二轮 (本轮, 用户决策):
 *   - 移除 length 硬规则 (length<16 / 未知 length 都不再置 is_attack=1).
 *   - 所有攻击判定统一交给 TCN+SE 模型 (modbus_ids_on_frame 推理).
 *   - H743 这里只负责"长度驱动正确解码 raw17", 判决权完全交给 modbus_ids.c.
 *
 * 2026-09-29 第十一轮 (修订, mirror rtu_replay_gui.py):
 *   - length<16 帧新约定: PC 保留 length_field 原值 (10/12/14), body 固定 13 字节
 *     (= 11×0x00 + 2B uint16 CRC). body_size 与 length_field 解耦!
 *     H743 解析:
 *       - features 4-14 全 0
 *       - CRC 从 body[11..12] 提取 (固定位置, 不再依赖 length_field)
 *       - is_response 从 buf[16] 读, time 从 buf[17..24] 读 (固定偏移)
 *       - out->length = length_field (原值, 保留给 feature_extractor 当 length 特征)
 *   - 总帧长恒 25 字节 (1+1+1+13+1+8), 与 length_field 无关.
 *   - 旧逻辑 (length-2)B 0x00 + body[len-2..len-1] 已废弃.
 *
 * 2026-09-19 第九轮重构:
 *   - 帧长度由 length 字段决定, 4 种格式:
 *     length=16/46/90 → 3 种编码 (uint8/float32/float64 + uint16 CRC)
 *     length<16       → 11×0x00 + uint16 CRC = 13 字节 body (与 length_field 解耦)
 *   - raw17_decode 按 length switch case 解码.
 *   - 删 modbus_crc16 + CRC 表 (不再使用, 第八轮已不再校验).
 */

#include "modbus_rtu.h"
#include <string.h>

/* length<16 帧 body 大小 (固定 13 字节 = 11×0x00 + 2B CRC)
 * 2026-09-29 第十一轮修订: body_size 与 length_field 解耦, 用这个常量代替.
 * 总帧长 = RAW17_ATTACK_BODY_SIZE + 12 = 25 字节 (1+1+1+13+1+8). */
#define RAW17_ATTACK_BODY_SIZE  13u

/* --- BE 解码辅助 (无对齐, 裸位读) -------------------------------------- */
static inline uint16_t be16(const uint8_t* p) {
    return ((uint16_t)p[0] << 8) | (uint16_t)p[1];
}
static inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

/* BE float32 解码: 输入 4 字节, 返回 IEEE 754 float.
 * 用 union 避免 (float*) 未对齐访问 (Cortex-M7 默认对齐, 否则 HardFault). */
typedef union { uint32_t u; float f; } float32_cv_t;
static inline float be_f32(const uint8_t* p) {
    float32_cv_t cv;
    cv.u = be32(p);
    return cv.f;
}

/* BE float64 解码: 输入 8 字节.
 * Cortex-M7 double = 8B IEEE 754 双精度, 直接 union 转换. */
typedef union { uint64_t u; double d; } float64_cv_t;
static inline double be_f64(const uint8_t* p) {
    float64_cv_t cv;
    /* 8B BE 拼字, 分两次写避免 unaligned uint64 警告 */
    cv.u = ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] <<  8) |  (uint64_t)p[7];
    return cv.d;
}

/* --- 公共 API ------------------------------------------------------------ */
void modbus_rtu_init(void)
{
    /* 当前无状态, 留接口 */
}

uint8_t modbus_is_known_func(uint8_t fc)
{
    (void)fc;
    return 1;  /* H743 不判断 FC 内容, 永远当作已知 */
}

uint8_t modbus_extract_addr_count(uint8_t func, const uint8_t* data,
                                  uint16_t data_len,
                                  uint16_t* reg_addr, uint16_t* reg_count)
{
    (void)func; (void)data; (void)data_len;
    *reg_addr = 0;
    *reg_count = 0;
    return 0;
}

/* --- 17-dim payload 解码 (length 驱动) --------------------------------- */
static void raw17_decode(const uint8_t* p, raw17_t* out)
{
    out->address  = p[0];
    out->function = p[1];
    out->length   = p[2];   /* 保留原值, 供 feature_extractor 当 length 特征 */

    uint8_t  len     = out->length;
    /* 2026-09-29 第十一轮: body_size 与 length_field 解耦!
     *   length<16 时 body_size 恒为 RAW17_ATTACK_BODY_SIZE=13 (不读 length_field).
     *   length=16/46/90 时 body_size = length_field. */
    uint16_t body_size = (len < RAW17_ATTACK_LEN_THRESHOLD)
                         ? (uint16_t)RAW17_ATTACK_BODY_SIZE
                         : (uint16_t)len;
    const uint8_t* body = &p[3];                /* features 4-15 共 body_size 字节 */
    const uint8_t* tail = &p[3 + body_size];    /* is_response + time 起始 */

    /* 默认: 非攻击 */
    out->is_attack = 0;

    if (len < RAW17_ATTACK_LEN_THRESHOLD) {
        /* ⭐ NMRI 攻击信号 (length<16) 解码 (2026-09-29 第十一轮修订):
         *   PC 端 length_field 保留原值 (10/12/14), body_size 固定 13 字节
         *   (= 11×0x00 + 2B uint16 CRC). body_size 与 length_field 解耦!
         *
         *   H743 解析:
         *     - features 4-14 全 0 (NMRI 攻击未携带有效 payload)
         *     - CRC 固定从 body[11..12] 提取 (不依赖 length_field)
         *     - out->length = 原 length_field (供 feature_extractor 当 length 特征,
         *       训练时归一化到 -0.20/-0.13/-0.067, 模型可识别)
         *     - is_response 从 tail[0] = buf[16], time 从 tail[1..8] = buf[17..24]
         *       (固定偏移, 不依赖 length_field)
         *
         *   ⭐⭐ 2026-10-01 第二十二轮: 不再置 out->is_attack = 1.
         *     所有攻击判定统一交给 TCN+SE 模型 (modbus_ids_on_frame).
         *     length_field 作为 raw17.length 透传给模型当特征, 模型自己识别 NMRI.
         */
        out->setpoint = 0.0f;
        out->gain     = 0.0f;
        out->reset    = 0.0f;
        out->deadband = 0.0f;
        out->cycle    = 0.0f;
        out->rate     = 0.0f;
        out->system_mode     = 0.0f;
        out->control_scheme  = 0.0f;
        out->pump     = 0.0f;
        out->solenoid = 0.0f;
        out->pressure = 0.0f;
        /* CRC 固定在 body[11..12] (新格式 body 长度恒为 13, 等同 body[body_size-2..body_size-1]) */
        out->crc = be16(&body[RAW17_ATTACK_BODY_SIZE - 2]);
    } else  if (len == 16) {
        /* 10×uint8 BE (f4..f13) + float32 BE (f14=pressure) + uint16 BE (f15=CRC)
         *
         * f4..f13 在 length=16 时是 1 字节整数, 直接读即可.
         * struct 字段是 float, 编译器做隐式转换; 不会做 float 数学运算,
         * 也不调用 be_f32/be_f64 (那是给 length=46/90 用的). */
        out->setpoint = body[0];
        out->gain     = body[1];
        out->reset    = body[2];
        out->deadband = body[3];
        out->cycle    = body[4];
        out->rate     = body[5];
        out->system_mode     = body[6];
        out->control_scheme  = body[7];
        out->pump     = body[8];
        out->solenoid = body[9];
        out->pressure = be_f32(&body[10]);   /* ⭐ pressure 仍是 float32 */
        out->crc      = be16(&body[14]);
    } else if (len == 46) {
        /* 11×float32 BE (f4..f14) + uint16 BE (f15=CRC) */
        out->setpoint = be_f32(&body[0]);
        out->gain     = be_f32(&body[4]);
        out->reset    = be_f32(&body[8]);
        out->deadband = be_f32(&body[12]);
        out->cycle    = be_f32(&body[16]);
        out->rate     = be_f32(&body[20]);
        out->system_mode     = be_f32(&body[24]);
        out->control_scheme  = be_f32(&body[28]);
        out->pump     = be_f32(&body[32]);
        out->solenoid = be_f32(&body[36]);
        out->pressure = be_f32(&body[40]);
        out->crc      = be16(&body[44]);
    } else if (len == 90) {
        /* 11×float64 BE (f4..f14) + uint16 BE (f15=CRC) */
        out->setpoint = (float)be_f64(&body[0]);
        out->gain     = (float)be_f64(&body[8]);
        out->reset    = (float)be_f64(&body[16]);
        out->deadband = (float)be_f64(&body[24]);
        out->cycle    = (float)be_f64(&body[32]);
        out->rate     = (float)be_f64(&body[40]);
        out->system_mode     = (float)be_f64(&body[48]);
        out->control_scheme  = (float)be_f64(&body[56]);
        out->pump     = (float)be_f64(&body[64]);
        out->solenoid = (float)be_f64(&body[72]);
        out->pressure = (float)be_f64(&body[80]);
        out->crc      = be16(&body[88]);
    } else {
        /* 未知 length (>16 但不在 {46, 90}):
         *   2026-09-29 第十一轮曾设为防御性攻击 (is_attack=1).
         *   2026-10-01 第二十二轮用户决策: 不再 length 硬规则判 attack, 所有判决走模型.
         *   这里 features 全 0, CRC 末尾 2B, is_attack 保持 0 由模型判. */
        out->is_attack = 0;                                          /* ⭐ 第二十二轮: 改回 0, 不再 length 硬规则 */
        out->setpoint = 0.0f;
        out->gain     = 0.0f;
        out->reset    = 0.0f;
        out->deadband = 0.0f;
        out->cycle    = 0.0f;
        out->rate     = 0.0f;
        out->system_mode     = 0.0f;
        out->control_scheme  = 0.0f;
        out->pump     = 0.0f;
        out->solenoid = 0.0f;
        out->pressure = 0.0f;
        out->crc      = be16(&body[len - 2]);
    }

    /* 末尾固定: is_response (1B) + time (8B float64 BE)
     * 2026-09-24 第四轮: time 从 4B uint32 → 4B f32 → 8B f64 (精度 μs 级).
     * 总帧长: length_field + 12 (length=16/46/90)
     *        或 RAW17_ATTACK_BODY_SIZE + 12 = 25 (length<16, 第十一轮修订)
     * 帧尾偏移: body_size + 4 (is_response 1B + time 8B = 9B). */
    out->is_response = tail[0];
    out->time        = be_f64(&tail[1]);
}

/**
 * 解析一帧 length-driven raw17 帧.
 *
 * @param buf  USART RX DMA buffer
 * @param len  实际接收字节数
 * @param out  输出解析结果 (slave/func/length/raw17)
 * @return     1=成功, 0=长度非法
 *
 * 注意:
 *   - 调用方需先 SCB_InvalidateDCache_by_Addr(buf, len)
 *   - 总长度计算 (2026-09-29 第十一轮修订):
 *       length_field >= 16 → 总长 = length_field + 12
 *       length_field <  16 → 总长 = RAW17_ATTACK_BODY_SIZE + 12 = 25 (body_size 与 length_field 解耦)
 *     (2026-09-24 第四轮: time 4B → 8B f64, 尾偏移 +4 → +8, 帧长 +8 → +12)
 *   - length 不再触发 is_attack=1 硬规则 (2026-10-01 第二十二轮起, 判决权全交模型)
 *   - out->length 保留 length_field 原值 (供 feature_extractor 当 length 特征, 模型识别)
 */
uint8_t modbus_rtu_parse(const uint8_t* buf, uint16_t len, modbus_frame_t* out)
{
    /* 最小长度校验: 必须能容纳 header(3) + 至少 1B body + is_resp(1) + time(8) = 13 */
//    if (len < 13) {
//        return 0;
//    }

    /* length 字段越界检查 + body_size 计算
     * 2026-09-29 第十一轮: length<16 时 body_size 固定 13, 与 length_field 解耦 */
    uint8_t declared_len = buf[2];
    uint16_t body_size = (declared_len < RAW17_ATTACK_LEN_THRESHOLD)
                         ? (uint16_t)RAW17_ATTACK_BODY_SIZE
                         : (uint16_t)declared_len;
    uint16_t expected_total = body_size + 12;  /* header(3) + body + is_resp(1) + time(8) */
//    if (expected_total > RAW17_FRAME_MAX || expected_total < RAW17_FRAME_MIN) {
//        return 0;
//    }
//    /* 接收长度必须 ≥ 声明总长 */
//    if (len < expected_total) {
//        return 0;
//    }

    out->slave     = buf[0];
    out->func      = buf[1];
    out->length    = declared_len;
    out->total_len = expected_total;

    raw17_decode(buf, &out->raw17);
    return 1;
}
