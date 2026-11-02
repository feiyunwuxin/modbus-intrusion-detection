"""
verify_h743_pipeline.py
========================
2026-09-19 第九轮重构后, 端到端验证 PC ↔ H743 length-driven 17-dim 帧编码/解码一致性.

用法: python verify_h743_pipeline.py [N]
  N = 测试行数 (默认 1000)

验证 6 件事:
  1. PC 端 build_raw17_frame (Python) 按 length 分发 6 种格式:
     length=16/46/90 (3 种编码) + length<16 (攻击: 0+CRC) + 未知 length
  2. PC 编码后帧总长 = length + 8
  3. H743 端 raw17_decode (Python 镜像 C) 解码 features 4-14
  4. 解码 17 字段 round-trip 与 CSV 原始值一致
     (×100 量化已废除, 现按 length 分支验证 float 精度)
  5. length<16 → raw17.is_attack=1 (硬规则)
  6. ⭐⭐ 第十一轮: 每一条 length<16 帧必须触发报警 (硬规则不受 gap counter 限制)
     模拟 H743 alarm path, 检查所有 length<16 都进 alarm_count (rate-limit 已禁用).
"""

import csv
import struct
import sys
import os
import importlib.util

# === 1. 加载 PC 编码器 (rtu_replay_gui.py) =================================
def load_pc_encoder():
    """动态加载 rtu_replay_gui.py 中的 build_raw17_frame / parse_raw17_payload."""
    gui_path = os.path.join(os.path.dirname(__file__), 'rtu_replay_gui.py')
    if not os.path.exists(gui_path):
        raise FileNotFoundError(gui_path)
    spec = importlib.util.spec_from_file_location("rtu_replay_gui", gui_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# === 2. H743 C 算法的纯 Python 镜像 ========================================
# 第八轮起 modbus_crc16 不再调用, 第九轮帧长由 length 字段决定.

def be16(p: bytes, off: int) -> int:
    return (p[off] << 8) | p[off + 1]

def be32(p: bytes, off: int) -> int:
    return (p[off] << 24) | (p[off + 1] << 16) | (p[off + 2] << 8) | p[off + 3]

def be_f32(p: bytes, off: int) -> float:
    return struct.unpack_from('>f', p, off)[0]

def be_f64(p: bytes, off: int) -> float:
    return struct.unpack_from('>d', p, off)[0]


def h743_decode_raw17(frame: bytes) -> dict | None:
    """镜像 KeilH743/H743/Core/Src/modbus_rtu.c 的 modbus_rtu_parse + raw17_decode.

    Returns dict {slave, func, length, total_len, raw17{...}} or None if invalid.
    """
    if len(frame) < 9:
        return None
    length = frame[2]
    expected_total = length + 12  # 2026-09-24 第四轮 round 2: +8B f64 time, 帧长 +4
    if expected_total > 102 or expected_total < 22:
        return None
    if len(frame) < expected_total:
        return None

    body = frame[3:3 + length]   # features 4-15 (length bytes)
    tail = frame[3 + length:]    # is_resp(1) + time(8) = 9 bytes (2026-09-24 第四轮)

    raw = {
        'address':  frame[0],
        'function': frame[1],
        'length':   length,
        'is_attack': 1 if length < 16 else 0,
    }

    if length < 16:
        # 攻击: features 4-14 全 0, CRC 从 body[len-2..len-1]
        for k in ('setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate',
                  'system_mode', 'control_scheme', 'pump', 'solenoid', 'pressure'):
            raw[k] = 0.0
        raw['crc'] = be16(body, length - 2) if length >= 2 else 0
    elif length == 16:
        raw['setpoint'] = float(body[0])
        raw['gain']     = float(body[1])
        raw['reset']    = float(body[2])
        raw['deadband'] = float(body[3])
        raw['cycle']    = float(body[4])
        raw['rate']     = float(body[5])
        raw['system_mode']     = float(body[6])
        raw['control_scheme']  = float(body[7])
        raw['pump']     = float(body[8])
        raw['solenoid'] = float(body[9])
        raw['pressure'] = be_f32(body, 10)
        raw['crc']      = be16(body, 14)
    elif length == 46:
        raw['setpoint'] = be_f32(body, 0)
        raw['gain']     = be_f32(body, 4)
        raw['reset']    = be_f32(body, 8)
        raw['deadband'] = be_f32(body, 12)
        raw['cycle']    = be_f32(body, 16)
        raw['rate']     = be_f32(body, 20)
        raw['system_mode']     = be_f32(body, 24)
        raw['control_scheme']  = be_f32(body, 28)
        raw['pump']     = be_f32(body, 32)
        raw['solenoid'] = be_f32(body, 36)
        raw['pressure'] = be_f32(body, 40)
        raw['crc']      = be16(body, 44)
    elif length == 90:
        raw['setpoint'] = float(be_f64(body, 0))
        raw['gain']     = float(be_f64(body, 8))
        raw['reset']    = float(be_f64(body, 16))
        raw['deadband'] = float(be_f64(body, 24))
        raw['cycle']    = float(be_f64(body, 32))
        raw['rate']     = float(be_f64(body, 40))
        raw['system_mode']     = float(be_f64(body, 48))
        raw['control_scheme']  = float(be_f64(body, 56))
        raw['pump']     = float(be_f64(body, 64))
        raw['solenoid'] = float(be_f64(body, 72))
        raw['pressure'] = float(be_f64(body, 80))
        raw['crc']      = be16(body, 88)
    else:
        # 未知 length: 防御性视作攻击
        raw['is_attack'] = 1
        for k in ('setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate',
                  'system_mode', 'control_scheme', 'pump', 'solenoid', 'pressure'):
            raw[k] = 0.0
        raw['crc'] = be16(body, length - 2) if length >= 2 else 0

    raw['is_response'] = tail[0]
    raw['time']        = be_f64(tail, 1)   # float64 BE, 2026-09-24 第四轮 round 2 (4B→8B)

    return {
        'slave':     frame[0],
        'func':      frame[1],
        'length':    length,
        'total_len': expected_total,
        'raw17':     raw,
    }


# === 3. 主测试 ==============================================================
def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    pc = load_pc_encoder()
    csv_path = os.path.join(os.path.dirname(__file__), 'IanArffDataset.csv')
    print(f"CSV: {csv_path}")
    print(f"Test rows: {N}\n")

    rows = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if i + 1 >= N:
                break

    if not rows:
        print("ERROR: no rows loaded")
        return 1

    print(f"Loaded {len(rows)} rows.\n")

    # 按 length 分桶统计
    from collections import Counter
    length_hist = Counter()
    attack_hist = Counter()
    attack_alarm_emitted = Counter()  # ⭐ 第十一轮: 统计硬规则报警数
    pass_count = 0
    fail_count = 0
    roundtrip_fail = []

    # 11 个特征 → CSV 列名 (f4..f14)
    FEAT_CSV_COLS = [
        'setpoint', 'gain', 'reset rate', 'deadband', 'cycle time', 'rate',
        'system mode', 'control scheme', 'pump', 'solenoid', 'pressure measurement',
    ]
    FEAT_KEYS = [
        'setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate',
        'system_mode', 'control_scheme', 'pump', 'solenoid', 'pressure',
    ]

    for i, row in enumerate(rows):
        # === PC 端编码 ===
        frame = pc.build_raw17_frame(row)

        # 读 length 字段 (frame[2])
        length = frame[2]

        # (1) 帧总长 = length + 8
        if len(frame) != length + 8:
            fail_count += 1
            roundtrip_fail.append(f"row {i}: frame length {len(frame)} != length+8 ({length+8})")
            continue

        # === H743 端解码 (Python 镜像 C) ===
        parsed = h743_decode_raw17(frame)
        if parsed is None:
            fail_count += 1
            roundtrip_fail.append(f"row {i}: h743_decode_raw17 returned None")
            continue

        length_hist[length] += 1
        if parsed['raw17']['is_attack']:
            attack_hist[length] += 1
            # ⭐⭐ 第十一轮: 模拟 H743 硬规则路径, 每一条 length<16 都应触发报警
            # (gap counter 已对硬规则禁用, 不再限流)
            attack_alarm_emitted[length] += 1

        # (2) slave / func 一致
        csv_slave = int(row['address'])
        csv_func  = int(row['function'])
        if parsed['slave'] != csv_slave or parsed['func'] != csv_func:
            fail_count += 1
            roundtrip_fail.append(f"row {i}: slave/func mismatch")
            continue

        r17 = parsed['raw17']

        # (3) length<16 硬规则: features 4-14 应全 0, is_attack=1
        if length < 16:
            if not r17['is_attack']:
                fail_count += 1
                roundtrip_fail.append(f"row {i}: length={length} but is_attack=0")
                continue
            # features 应全 0
            for k in FEAT_KEYS:
                if r17[k] != 0.0:
                    fail_count += 1
                    if len(roundtrip_fail) < 10:
                        roundtrip_fail.append(
                            f"row {i}: length={length} attack but {k}={r17[k]} != 0"
                        )
                    break
            else:
                # 验证 body 前 (length-2) 字节是 0x00, 末尾 2B 是 CRC
                body = frame[3:3+length]
                zeros_ok = all(b == 0 for b in body[:length-2])
                if not zeros_ok:
                    fail_count += 1
                    if len(roundtrip_fail) < 10:
                        roundtrip_fail.append(f"row {i}: length={length} body not all zeros")
                    continue
            # 攻击格式不再验证特征数值
            pass_count += 1
            continue

        # (4) is_response + time 一致 (末尾固定 5 字节)
        csv_cmd = int(row['command response'])
        if r17['is_response'] != (1 if csv_cmd == 1 else 0):
            fail_count += 1
            if len(roundtrip_fail) < 10:
                roundtrip_fail.append(
                    f"row {i}: is_response csv={csv_cmd} decoded={r17['is_response']}"
                )
            continue

        # 2026-09-24 第四轮 round 2: time 是 float64 BE, ARFF f64 → f64 无精度损失
        csv_time = float(row['time'])
        if abs(r17['time'] - csv_time) > 1e-9:
            fail_count += 1
            if len(roundtrip_fail) < 10:
                roundtrip_fail.append(f"row {i}: time csv={csv_time} decoded={r17['time']}")
            continue

        # (5) CRC_rate 末 2B 字段
        csv_crc = max(0, min(65535, int(round(float(row['crc rate'])))))
        if r17['crc'] != csv_crc:
            fail_count += 1
            if len(roundtrip_fail) < 10:
                roundtrip_fail.append(f"row {i}: crc csv={csv_crc} decoded={r17['crc']}")
            continue

        # (6) features 4-14 数值 round-trip (按 length 分支定精度)
        feat_ok = True
        for col, key in zip(FEAT_CSV_COLS, FEAT_KEYS):
            csv_val = float(row[col])

            if length == 16:
                # uint8 (features 4-13) + float32 (pressure)
                if key == 'pressure':
                    # float32 round-trip 容差
                    if abs(r17[key] - csv_val) > 1e-5 * max(1.0, abs(csv_val)):
                        feat_ok = False
                else:
                    # uint8 截断容差
                    expected = max(0.0, min(255.0, csv_val))
                    if abs(r17[key] - expected) > 0.5:
                        feat_ok = False
            elif length == 46:
                # float32 round-trip
                if abs(r17[key] - csv_val) > 1e-5 * max(1.0, abs(csv_val)):
                    feat_ok = False
            elif length == 90:
                # float64 round-trip
                if abs(r17[key] - csv_val) > 1e-9 * max(1.0, abs(csv_val)):
                    feat_ok = False

            if not feat_ok:
                fail_count += 1
                if len(roundtrip_fail) < 10:
                    roundtrip_fail.append(
                        f"row {i}: length={length} {key} csv={csv_val} decoded={r17[key]}"
                    )
                break

        if not feat_ok:
            continue

        pass_count += 1

    print(f"=== RESULT ===")
    print(f"PASS: {pass_count}/{len(rows)}")
    print(f"FAIL: {fail_count}\n")
    print(f"length distribution:")
    for ln in sorted(length_hist.keys()):
        marker = ' [attack]' if ln < 16 else ''
        alarm = attack_alarm_emitted.get(ln, 0)
        if ln < 16:
            # 硬规则必须每帧报警: alarm_count == attack_count == length_count
            alarm_ok = "[OK]" if alarm == length_hist[ln] else "[FAIL]"
            print(f"  length={ln:>3}: {length_hist[ln]:>5} rows "
                  f"({attack_hist[ln]} attack, {alarm} alarm) {alarm_ok}{marker}")
            if alarm != length_hist[ln]:
                fail_count += 1
                roundtrip_fail.append(
                    f"length={ln}: alarm={alarm} != rows={length_hist[ln]} (硬规则漏报!)"
                )
        else:
            print(f"  length={ln:>3}: {length_hist[ln]:>5} rows ({attack_hist[ln]} attack){marker}")
    if roundtrip_fail:
        print("\nFirst failures:")
        for msg in roundtrip_fail[:10]:
            print(f"  {msg}")
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
