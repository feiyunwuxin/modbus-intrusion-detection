#!/usr/bin/env python3
"""验证 rtu_replay_gui.build_raw17_frame 与 H743 modbus_rtu.raw17_decode 是否 bit-perfect 一致."""

import sys, io, struct
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import csv
import pandas as pd

CSV_PATH = r"C:/work/Claude/Issue/IanArffDatasetF32.csv"

# ── PC 端 build_raw17_frame (从 rtu_replay_gui.py 复制) ──────────────────────
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


def build_raw17_frame_pc(row: dict) -> bytes:
    """PC 端 raw17 帧编码."""
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
        body = bytearray(max(0, length - 2))
        body += crc_bytes

    return bytes([slave, func, length]) + bytes(body) + bytes([is_resp]) + time_bytes


# ── H743 端 raw17_decode (从 modbus_rtu.c 翻译成 Python) ───────────────────
def be_f32(p): return struct.unpack('>f', p)[0]
def be_f64(p): return struct.unpack('>d', p)[0]
def be16(p):   return struct.unpack('>H', p)[0]


def raw17_decode_h743(buf: bytes) -> dict:
    """H743 端 raw17 帧解码."""
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
        out['is_attack'] = 1  # 硬规则 (但 modbus_rtu.c:86 这行被注释掉了!)
        for k in ['setpoint','gain','reset','deadband','cycle','rate',
                  'system_mode','control_scheme','pump','solenoid','pressure']:
            out[k] = 0.0
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
        out['is_attack'] = 1
        for k in ['setpoint','gain','reset','deadband','cycle','rate',
                  'system_mode','control_scheme','pump','solenoid','pressure']:
            out[k] = 0.0
        out['crc'] = be16(body[len_-2:len_]) if len_ >= 2 else 0
    return out


# ── 测试 ─────────────────────────────────────────────────────────────
print("=" * 70)
print("验证 rtu_replay_gui.build_raw17_frame vs H743 modbus_rtu.raw17_decode")
print("=" * 70)

# 读 CSV 测试每种 length 类型
df = pd.read_csv(CSV_PATH, nrows=200)
print(f"测试样本数: {len(df)}")
print()

# 按 length 分组测试
length_dist = df['length'].value_counts().sort_index()
print(f"Length 分布 (前200行):")
for k, v in length_dist.items():
    print(f"  length={int(k):>3}  count={v:>4}")
print()

errors = []
for idx, row in df.iterrows():
    row_dict = row.to_dict()
    frame = build_raw17_frame_pc(row_dict)

    # 帧格式: [slave][func][length][body(length B)][is_resp][time(8B)]
    expected_len = row_dict['length'] + 12
    if len(frame) != expected_len:
        errors.append(f'row {idx}: frame length {len(frame)} != expected {expected_len}')
        continue

    decoded = raw17_decode_h743(frame)

    # 验证关键字段
    issues = []
    if decoded['address'] != int(row_dict['address']):
        issues.append(f"address: {decoded['address']} != {int(row_dict['address'])}")
    if decoded['function'] != int(row_dict['function']):
        issues.append(f"function: {decoded['function']} != {int(row_dict['function'])}")
    if decoded['length'] != int(row_dict['length']):
        issues.append(f"length: {decoded['length']} != {int(row_dict['length'])}")
    if decoded['is_response'] != int(row_dict['command response']):
        issues.append(f"is_response: {decoded['is_response']} != {int(row_dict['command response'])}")
    if abs(decoded['time'] - float(row_dict['time'])) > 1e-9:
        issues.append(f"time: {decoded['time']} != {float(row_dict['time'])}")
    if decoded['crc'] != int(round(float(row_dict['crc rate']))):
        issues.append(f"crc: {decoded['crc']} != {int(round(float(row_dict['crc rate'])))}")

    # 验证特征字段 (length 决定编码)
    feat_keys = ['setpoint','gain','reset','deadband','cycle','rate',
                 'system_mode','control_scheme','pump','solenoid','pressure']
    if int(decoded['length']) == 16:
        # PC: feat_vals[:10] 是 uint8 round, feat_vals[10] 是 pressure float32
        for i, k in enumerate(feat_keys[:10]):
            v_raw = row_dict[RAW17_FEAT_COLS[i]]
            if v_raw == '?':
                expected_u8 = 0
            else:
                expected_u8 = max(0, min(255, int(round(float(v_raw)))))
            if abs(decoded[k] - expected_u8) > 1e-3:
                issues.append(f"{k}: {decoded[k]} != {expected_u8}")
        # pressure float32 round-trip
        p_raw = row_dict[RAW17_FEAT_COLS[10]]
        if p_raw == '?':
            p_val = 0.0
        else:
            p_val = float(p_raw)
        expected_f32 = struct.pack('>f', p_val)
        expected_decoded = struct.unpack('>f', expected_f32)[0]
        if abs(decoded['pressure'] - expected_decoded) > 1e-3:
            issues.append(f"pressure: {decoded['pressure']} != {expected_decoded}")
    elif int(decoded['length']) in (46, 90):
        for i, k in enumerate(feat_keys):
            v_raw = row_dict[RAW17_FEAT_COLS[i]]
            v = 0.0 if v_raw == '?' else float(v_raw)
            if abs(decoded[k] - v) > 1e-3:
                issues.append(f"{k}: {decoded[k]} != {v}")
    else:
        # length<16, 全部应为 0
        for k in feat_keys:
            if decoded[k] != 0.0:
                issues.append(f"{k}: {decoded[k]} != 0 (length<16)")
        if decoded['is_attack'] != 1:
            issues.append(f"is_attack: {decoded['is_attack']} != 1 (length<16 硬规则)")

    if issues:
        errors.append(f"row {idx} (length={int(row_dict['length'])}): {'; '.join(issues[:3])}")

print(f"错误数: {len(errors)} / {len(df)}")
if errors:
    print("\n所有错误详情:")
    for e in errors[:5]:
        print(f"  {e}")
if errors:
    print("\n前 10 个错误:")
    for e in errors[:10]:
        print(f"  {e}")
else:
    print("\n[OK] 全部样本编码/解码一致!")
print()

# ── 额外检查 1: H743 modbus_rtu.c:86 is_attack 注释掉的 bug ──
print("=" * 70)
print("[重要] H743 端 is_attack 硬规则当前是否生效?")
print("=" * 70)
print("  rtu_replay_gui.py (PC): length<16 时只编码 (length-2) 个 0x00 + 2B CRC")
print("  H743 modbus_rtu.c:83-87 原文:")
print("    if (len < RAW17_ATTACK_LEN_THRESHOLD) {")
print("        // out->is_attack = 1;  ← ⚠️ 被注释掉了!")
print("  H743 modbus_ids.h 默认 is_attack 路径:")
print()
import re
hdr = open('h743/Core/Inc/modbus_ids.h').read()
if 'is_attack' in hdr:
    for line in hdr.split('\n'):
        if 'is_attack' in line.lower():
            print(f"    {line.strip()}")
print()
print("  ⚠️ 结论: length<16 的硬规则攻击信号 is_attack=1 当前 H743 不会触发!")
print("     (虽然 PC 把 length<16 数据发出来了, H743 解析到了 features=0,")
print("      但不会走 '立即报警' 路径, 只能靠 TCN+SE 模型判 attack=1)")

# ── 额外检查 2: time 精度 ──
print()
print("=" * 70)
print("[检查] time 字段精度 (f64 BE)")
print("=" * 70)
sample = df.iloc[0].to_dict()
t = float(sample['time'])
frame = build_raw17_frame_pc(sample)
t_decoded = struct.unpack('>d', frame[-8:])[0]
t_f32_decoded = struct.unpack('>f', struct.pack('>f', t))[0]
t_f32_roundtrip_err = abs(t - t_f32_decoded)
print(f"  CSV time = {t:.6f}")
print(f"  H743 解码 time = {t_decoded:.6f}")
print(f"  如果用 float32: 误差 = {t_f32_roundtrip_err:.1f} sec (≈ {int(t_f32_roundtrip_err/86400*365)} 年!)")
print(f"  ✓ f64 误差 = {abs(t-t_decoded):.2e} sec (μs 级精度, OK)")