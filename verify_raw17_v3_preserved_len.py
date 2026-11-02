#!/usr/bin/env python3
"""验证 v3 格式 (2026-09-29 第十一轮修订): length_field 保留原值 + body_size 固定 13.

新约定:
  - length<16 时 PC 不覆盖 length_field, 保留 CSV 原值 (10/12/14)
  - body 固定 13 字节 = 11×0x00 + 2B uint16 CRC
  - 总帧长 = 1+1+1+13+1+8 = 25 字节 (与 length_field 无关)
  - H743 modbus_rtu.c 用 RAW17_ATTACK_BODY_SIZE=13 解析, body_size 与 length_field 解耦
  - is_attack=1 硬规则恢复
"""
import sys, io, struct
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import csv
import pandas as pd

CSV_PATH = r"C:/work/Claude/Issue/IanArffDatasetF32.csv"

# ── PC 端 (第十一轮修订版, 镜像 rtu_replay_gui.py) ─────────────────────
RAW17_FEAT_COLS = (
    'setpoint', 'gain', 'reset rate', 'deadband',
    'cycle time', 'rate', 'system mode', 'control scheme',
    'pump', 'solenoid', 'pressure measurement',
)


def _safe_uint(v, default=0, vmin=0, vmax=255):
    try:
        x = int(float(v))
        return max(vmin, min(vmax, x))
    except (ValueError, TypeError):
        return default


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def build_raw17_frame_pc_v3(row: dict) -> bytes:
    """PC 端 raw17 帧编码 v3 (length 保留, body 固定 13 字节)."""
    slave = _safe_uint(row.get('address'), default=1, vmin=1, vmax=247)
    func  = _safe_uint(row.get('function'), default=3, vmin=0, vmax=255)
    length = _safe_uint(row.get('length'), default=16, vmin=0, vmax=255)  # ⭐ 保留原值
    is_resp = 1 if _safe_uint(row.get('command response'),
                              default=0, vmin=0, vmax=1) == 1 else 0
    t = _safe_float(row.get('time'), default=0.0)
    time_bytes = struct.pack('>d', t)
    crc_raw = _safe_float(row.get('crc rate'))
    crc_u16 = max(0, min(65535, int(round(crc_raw))))
    crc_bytes = struct.pack('>H', crc_u16)
    feat_vals = [_safe_float(row.get(c)) for c in RAW17_FEAT_COLS]

    if length == 16:
        body = bytearray()
        for v in feat_vals[:10]:
            body.append(max(0, min(255, int(round(v)))))
        body += struct.pack('>f', float(feat_vals[10]))
        body += crc_bytes
    elif length == 46:
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>f', float(v))
        body += crc_bytes
    elif length == 90:
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>d', float(v))
        body += crc_bytes
    else:
        # ⭐ v3: length 字段保留原值, body 固定 13 字节
        body = bytearray(11)
        body += crc_bytes

    return bytes([slave, func, length]) + bytes(body) + bytes([is_resp]) + time_bytes


# ── H743 端 v3 解析 (镜像 modbus_rtu.c 第十一轮修订版) ─────────────────
RAW17_ATTACK_BODY_SIZE = 13
RAW17_ATTACK_LEN_THRESHOLD = 16


def be_f32(p): return struct.unpack('>f', p)[0]
def be_f64(p): return struct.unpack('>d', p)[0]
def be16(p):   return struct.unpack('>H', p)[0]


def raw17_decode_h743_v3(buf: bytes) -> dict:
    """H743 端 v3 解析: body_size=13 当 length<16 (与 length_field 解耦)."""
    p = buf
    out = {}
    out['address']  = p[0]
    out['function'] = p[1]
    out['length']   = p[2]  # ⭐ 保留原值

    len_ = out['length']
    # ⭐ v3 关键: body_size 与 length_field 解耦
    body_size = RAW17_ATTACK_BODY_SIZE if len_ < RAW17_ATTACK_LEN_THRESHOLD else len_
    body = p[3:3+body_size]
    tail = p[3+body_size:3+body_size+9]
    out['is_attack'] = 0

    if len_ < RAW17_ATTACK_LEN_THRESHOLD:
        out['is_attack'] = 1
        for k in ['setpoint','gain','reset','deadband','cycle','rate',
                  'system_mode','control_scheme','pump','solenoid','pressure']:
            out[k] = 0.0
        # CRC 从 body[11..12] 固定位置
        out['crc'] = be16(body[RAW17_ATTACK_BODY_SIZE - 2:RAW17_ATTACK_BODY_SIZE])
    elif len_ == 16:
        out['setpoint'] = float(body[0])
        out['gain']     = float(body[1])
        out['reset']    = float(body[2])
        out['deadband'] = float(body[3])
        out['cycle']    = float(body[4])
        out['rate']     = float(body[5])
        out['system_mode']     = float(body[6])
        out['control_scheme']  = float(body[7])
        out['pump']     = float(body[8])
        out['solenoid'] = float(body[9])
        out['pressure'] = be_f32(body[10:14])
        out['crc']      = be16(body[14:16])
        out['is_attack'] = 0
    elif len_ == 46:
        out['setpoint'] = be_f32(body[0:4])
        out['gain']     = be_f32(body[4:8])
        out['reset']    = be_f32(body[8:12])
        out['deadband'] = be_f32(body[12:16])
        out['cycle']    = be_f32(body[16:20])
        out['rate']     = be_f32(body[20:24])
        out['system_mode']     = be_f32(body[24:28])
        out['control_scheme']  = be_f32(body[28:32])
        out['pump']     = be_f32(body[32:36])
        out['solenoid'] = be_f32(body[36:40])
        out['pressure'] = be_f32(body[40:44])
        out['crc']      = be16(body[44:46])
        out['is_attack'] = 0
    elif len_ == 90:
        out['setpoint'] = float(be_f64(body[0:8]))
        out['gain']     = float(be_f64(body[8:16]))
        out['reset']    = float(be_f64(body[16:24]))
        out['deadband'] = float(be_f64(body[24:32]))
        out['cycle']    = float(be_f64(body[32:40]))
        out['rate']     = float(be_f64(body[40:48]))
        out['system_mode']     = float(be_f64(body[48:56]))
        out['control_scheme']  = float(be_f64(body[56:64]))
        out['pump']     = float(be_f64(body[64:72]))
        out['solenoid'] = float(be_f64(body[72:80]))
        out['pressure'] = float(be_f64(body[80:88]))
        out['crc']      = be16(body[88:90])
        out['is_attack'] = 0
    else:
        out['is_attack'] = 1
        for k in ['setpoint','gain','reset','deadband','cycle','rate',
                  'system_mode','control_scheme','pump','solenoid','pressure']:
            out[k] = 0.0
        out['crc'] = be16(body[len_-2:len_]) if len_ >= 2 else 0

    # ⭐ v3: tail 偏移基于 body_size, 不依赖 length_field
    out['is_response'] = tail[0]
    out['time']        = struct.unpack('>d', tail[1:9])[0]
    return out


# ── 测试 ─────────────────────────────────────────────────────────────
print("=" * 70)
print("v3 格式验证 (length 保留 + body_size=13 固定)")
print("=" * 70)

df = pd.read_csv(CSV_PATH, nrows=300)
print(f"测试样本数: {len(df)}")
print()

length_dist = df['length'].value_counts().sort_index()
print(f"Length 分布 (前300行):")
for k, v in length_dist.items():
    print(f"  length={int(k):>3}  count={v:>4}")
print()

errors = []
stats = {
    'total_lt16': 0,
    'length_preserved': 0,    # length_field 保留原值
    'frame_size_25': 0,        # 总帧长 25
    'body_size_13': 0,         # body 13 字节
    'body_zeros_ok': 0,        # body[0..10] 全 0
    'crc_correct': 0,          # CRC 正确
    'is_attack_1': 0,          # 硬规则置 1
    'is_resp_correct': 0,      # is_response 正确
    'time_correct': 0,          # time 正确
    'features_zero': 0,         # features 全 0
    'length_eq16': 0,
    'length_eq46': 0,
    'length_eq90': 0,
}

for idx, row in df.iterrows():
    row_dict = row.to_dict()
    csv_length = int(row_dict['length'])
    csv_crc = int(round(float(row_dict['crc rate'])))
    csv_is_resp = int(row_dict['command response'])
    csv_time = float(row_dict['time'])

    frame = build_raw17_frame_pc_v3(row_dict)

    issues = []
    if csv_length < 16:
        # 验证 length<16 帧
        stats['total_lt16'] += 1
        encoded_length = frame[2]
        body = frame[3:16]
        zeros = body[:11]

        if encoded_length == csv_length:
            stats['length_preserved'] += 1
        else:
            issues.append(f"length_field 应保留原值={csv_length}, 实际={encoded_length}")

        if len(frame) == 25:
            stats['frame_size_25'] += 1
        else:
            issues.append(f"总帧长 应=25, 实际={len(frame)}")

        if len(body) == 13:
            stats['body_size_13'] += 1
        else:
            issues.append(f"body 大小 应=13, 实际={len(body)}")

        if all(b == 0 for b in zeros):
            stats['body_zeros_ok'] += 1
        else:
            issues.append(f"body[0..10] 不是全 0")

        # H743 decode
        decoded = raw17_decode_h743_v3(frame)

        if decoded['crc'] == csv_crc:
            stats['crc_correct'] += 1
        else:
            issues.append(f"CRC 解码 {decoded['crc']} != CSV {csv_crc}")

        if decoded['is_attack'] == 1:
            stats['is_attack_1'] += 1
        else:
            issues.append(f"is_attack 应=1, 实际={decoded['is_attack']}")

        if decoded['is_response'] == csv_is_resp:
            stats['is_resp_correct'] += 1
        else:
            issues.append(f"is_response {decoded['is_response']} != CSV {csv_is_resp}")

        if abs(decoded['time'] - csv_time) < 1e-9:
            stats['time_correct'] += 1
        else:
            issues.append(f"time 解码 {decoded['time']} != CSV {csv_time}")

        all_zero = all(decoded[k] == 0.0 for k in
                      ['setpoint','gain','reset','deadband','cycle','rate',
                       'system_mode','control_scheme','pump','solenoid','pressure'])
        if all_zero:
            stats['features_zero'] += 1
        else:
            issues.append(f"features 不全为 0")

    elif csv_length == 16:
        stats['length_eq16'] += 1
        decoded = raw17_decode_h743_v3(frame)
        if decoded['length'] != 16:
            issues.append(f"length_field 应=16, 实际={decoded['length']}")
    elif csv_length == 46:
        stats['length_eq46'] += 1
    elif csv_length == 90:
        stats['length_eq90'] += 1

    if issues:
        errors.append(f"row {idx} (CSV length={csv_length}): {'; '.join(issues[:3])}")

print(f"错误数: {len(errors)} / {len(df)}")
if errors:
    print("\n前 5 个错误:")
    for e in errors[:5]:
        print(f"  {e}")
else:
    print("\n[OK] 全部样本编码/解码一致!")
print()

# 专项统计
print("=" * 70)
print("v3 专项验证 (length<16 → length 保留 + body_size=13)")
print("=" * 70)
for k, v in stats.items():
    if stats['total_lt16'] and 'lt16' in k or v > 0:
        expected = stats['total_lt16'] if 'lt16' not in k and k != 'total_lt16' else stats['total_lt16']
        denom = stats['total_lt16'] if k != 'total_lt16' else stats['total_lt16']
        pct = v / denom * 100 if denom else 0
        print(f"  {k:<20}: {v:>4} / {denom:<4}  ({pct:.1f}%)")
print()

# ── 实际 rtu_replay_gui.build_raw17_frame 测试 ──
print("=" * 70)
print("实际 rtu_replay_gui.build_raw17_frame 输出")
print("=" * 70)
sys.path.insert(0, '.')
try:
    from rtu_replay_gui import build_raw17_frame as actual_func
    test_row = {'address':'1','function':'136','length':'10','command response':'0',
                'time':'1418682163.170388','crc rate':'13616',
                'setpoint':'0','gain':'0','reset rate':'0','deadband':'0','cycle time':'0',
                'rate':'0','system mode':'0','control scheme':'0','pump':'0','solenoid':'0',
                'pressure measurement':'0'}
    frame = actual_func(test_row)
    print(f"  输入 CSV length=10:")
    print(f"  实际 length_field = {frame[2]}  (期望 10, 原值)")
    print(f"  实际 body 大小 = {len(frame)-12}  (期望 13)")
    print(f"  body[0..10] = {list(frame[3:14])}  (期望全 0)")
    print(f"  CRC bytes = {frame[14]:02x} {frame[15]:02x} = {(frame[14]<<8)|frame[15]}  (期望 13616)")
    print(f"  is_response = {frame[16]}  (期望 0)")
    print(f"  总帧长 = {len(frame)}  (期望 25)")
except Exception as e:
    print(f"  ⚠ 导入失败: {e}")

# 总结
print()
print("=" * 70)
print("v3 总结")
print("=" * 70)
print("  PC 端: length_field 保留 CSV 原值 (10/12/14), body 固定 13 字节")
print("  H743 端: body_size=13 (不读 length_field), CRC 从 body[11..12]")
print("  优点: length 特征保留 (-0.20/-0.13/-0.067), 总帧长固定 25 字节,")
print("         body_size 与 length_field 解耦, 硬规则 is_attack=1 立即报警")
print("  Trade-off: H743 modbus_rtu.c 需要新增 RAW17_ATTACK_BODY_SIZE 常量")
print("             并在 raw17_decode + modbus_rtu_parse 中解耦 body_size")