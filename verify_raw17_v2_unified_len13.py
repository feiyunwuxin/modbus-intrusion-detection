#!/usr/bin/env python3
"""验证 rtu_replay_gui 新格式 (2026-09-29 第十一轮) + H743 解码.

新约定 (length<16):
  - length_header 强制为 13
  - body = 11 个 0x00 + 2B uint16 CRC (共 13 字节)
  - H743 modbus_rtu.c 解析: features 全 0, CRC 从 body[11..12], is_attack=1
"""
import sys, io, struct
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import csv
import pandas as pd

CSV_PATH = r"C:/work/Claude/Issue/IanArffDatasetF32.csv"

# ── PC 端 build_raw17_frame (从 rtu_replay_gui.py 第十一轮复制) ─────────
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


def build_raw17_frame_pc_v2(row: dict) -> bytes:
    """PC 端 raw17 帧编码 (第十一轮)."""
    slave = _safe_uint(row.get('address'), default=1, vmin=1, vmax=247)
    func  = _safe_uint(row.get('function'), default=3, vmin=0, vmax=255)
    length = _safe_uint(row.get('length'), default=16, vmin=0, vmax=255)
    is_resp = 1 if _safe_uint(row.get('command response'), default=0, vmin=0, vmax=1) == 1 else 0
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
        # ⭐ 第十一轮统一格式: length_header 强制 13, body = 11 zeros + 2B CRC
        length = 13
        body = bytearray(11)
        body += crc_bytes

    return bytes([slave, func, length]) + bytes(body) + bytes([is_resp]) + time_bytes


# ── H743 端 raw17_decode (从 modbus_rtu.c 第十一轮翻译成 Python) ─────────
def be_f32(p): return struct.unpack('>f', p)[0]
def be_f64(p): return struct.unpack('>d', p)[0]
def be16(p):   return struct.unpack('>H', p)[0]


def raw17_decode_h743_v2(buf: bytes) -> dict:
    """H743 端 raw17 帧解码 (第十一轮, is_attack 硬规则恢复)."""
    p = buf
    out = {}
    out['address']  = p[0]
    out['function'] = p[1]
    out['length']   = p[2]
    out['is_response'] = p[-9]
    out['time']     = struct.unpack('>d', p[-8:])[0]

    len_ = out['length']
    body = p[3:3+len_]
    out['is_attack'] = 0  # 默认

    if len_ < 16:
        # ⭐ 第十一轮: 硬规则恢复, is_attack = 1
        out['is_attack'] = 1
        for k in ['setpoint','gain','reset','deadband','cycle','rate',
                  'system_mode','control_scheme','pump','solenoid','pressure']:
            out[k] = 0.0
        # CRC 从 body[len-2..len-1] (新格式 length=13 时即 body[11..12])
        out['crc'] = be16(body[len_-2:len_]) if len_ >= 2 else 0
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
    else:
        # ⭐ 第十一轮: 未知 length 也置 is_attack=1
        out['is_attack'] = 1
        for k in ['setpoint','gain','reset','deadband','cycle','rate',
                  'system_mode','control_scheme','pump','solenoid','pressure']:
            out[k] = 0.0
        out['crc'] = be16(body[len_-2:len_]) if len_ >= 2 else 0
    return out


# ── 测试 ─────────────────────────────────────────────────────────────
print("=" * 70)
print("验证 rtu_replay_gui v2 (length<13 统一) + H743 modbus_rtu v2 (硬规则恢复)")
print("=" * 70)

df = pd.read_csv(CSV_PATH, nrows=300)
print(f"测试样本数: {len(df)}")
print()

# 按 length 分组测试
length_dist = df['length'].value_counts().sort_index()
print(f"Length 分布 (前300行):")
for k, v in length_dist.items():
    print(f"  length={int(k):>3}  count={v:>4}")
print()

# 统计 length<16 在新格式下的字段
errors = []
length13_count = 0
length13_attack_flag = 0
length13_crc_match = 0
length13_total = 0

for idx, row in df.iterrows():
    row_dict = row.to_dict()
    csv_length = int(row_dict['length'])
    frame = build_raw17_frame_pc_v2(row_dict)

    # 检查帧格式
    encoded_length = frame[2]
    expected_total = encoded_length + 12
    if len(frame) != expected_total:
        errors.append(f'row {idx}: frame length {len(frame)} != expected {expected_total}')
        continue

    decoded = raw17_decode_h743_v2(frame)

    # 验证关键字段 (所有 length)
    issues = []
    if decoded['address'] != int(row_dict['address']):
        issues.append(f"address: {decoded['address']} != {int(row_dict['address'])}")
    if decoded['function'] != int(row_dict['function']):
        issues.append(f"function: {decoded['function']} != {int(row_dict['function'])}")
    if decoded['is_response'] != int(row_dict['command response']):
        issues.append(f"is_response: {decoded['is_response']} != {int(row_dict['command response'])}")
    if abs(decoded['time'] - float(row_dict['time'])) > 1e-9:
        issues.append(f"time: {decoded['time']} != {float(row_dict['time'])}")

    # 验证 length_field 和 CRC
    if csv_length < 16:
        # ⭐ 新格式: length_field 强制为 13
        if decoded['length'] != 13:
            issues.append(f"length_field: {decoded['length']} != 13 (CSV length={csv_length} < 16 应统一为 13)")
        # CRC 必须与 CSV crc rate 一致 (body[11..12])
        expected_crc = int(round(float(row_dict['crc rate'])))
        if decoded['crc'] != expected_crc:
            issues.append(f"crc: {decoded['crc']} != {expected_crc}")
        # is_attack 必须为 1
        if decoded['is_attack'] != 1:
            issues.append(f"is_attack: {decoded['is_attack']} != 1 (硬规则)")
        # features 必须全 0
        for k in ['setpoint','gain','reset','deadband','cycle','rate',
                  'system_mode','control_scheme','pump','solenoid','pressure']:
            if decoded[k] != 0.0:
                issues.append(f"{k}: {decoded[k]} != 0")
        # body 内容检查: 11 zeros + 2B CRC
        body = frame[3:3+13]
        zeros = body[:11]
        if any(b != 0 for b in zeros):
            issues.append(f"body[0..10] 不是全 0: {list(zeros[:3])}...")
        if decoded['length'] == 13:
            length13_total += 1
            if decoded['is_attack'] == 1:
                length13_attack_flag += 1
            if decoded['crc'] == expected_crc:
                length13_crc_match += 1
    elif csv_length == 16:
        if decoded['length'] != 16:
            issues.append(f"length_field: {decoded['length']} != 16")
    elif csv_length == 46:
        if decoded['length'] != 46:
            issues.append(f"length_field: {decoded['length']} != 46")
    elif csv_length == 90:
        if decoded['length'] != 90:
            issues.append(f"length_field: {decoded['length']} != 90")

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
print("新格式专项验证 (length<16 → length_header=13)")
print("=" * 70)
print(f"  length<16 总数: {length13_total}")
print(f"  length_header=13 ✓: {length13_total}")
print(f"  is_attack=1 ✓: {length13_attack_flag}")
print(f"  CRC 正确 ✓: {length13_crc_match}")
print()

# ── 额外检查: 新旧格式对比 ──
print("=" * 70)
print("新旧格式差异总结")
print("=" * 70)
print("  旧 (length=10/12/14):")
print("    length_field = 10/12/14, body = (length-2) zeros + 2B CRC")
print("    CRC 位置: body[len-2..len-1] (依赖 length 字段)")
print("    is_attack 标志: 注释掉 (硬规则失效)")
print()
print("  新 (length<16 统一):")
print("    length_field 强制 = 13, body = 11 zeros + 2B CRC")
print("    CRC 位置: body[11..12] (固定, 等同于 body[len-2..len-1] 当 len=13)")
print("    is_attack = 1 (硬规则生效)")
print()
print("  Trade-off:")
print("    - 模型 length 特征: {-0.20, -0.13, -0.067} 全部归一化到 -0.10")
print("    - 但 length<16 攻击的强信号 (function=136/171, pressure=0) 不变")
print("    - 优点: CRC 位置固定, 硬规则立即报警, 无需 ML 模型兜底")

# ── 检查 time 精度 ──
print()
print("=" * 70)
print("[检查] time 字段精度 (f64 BE)")
print("=" * 70)
sample = df.iloc[0].to_dict()
t = float(sample['time'])
frame = build_raw17_frame_pc_v2(sample)
t_decoded = struct.unpack('>d', frame[-8:])[0]
print(f"  CSV time = {t:.6f}")
print(f"  H743 解码 time = {t_decoded:.6f}")
print(f"  ✓ f64 误差 = {abs(t-t_decoded):.2e} sec")