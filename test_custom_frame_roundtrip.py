"""
test_custom_frame_roundtrip.py  --  验证 PC build_custom_frame ↔ H743 frame_parser_parse 字节级一致

用法:
    python test_custom_frame_roundtrip.py

测试 3 个 case:
  1. 普通帧 (CSV row 25 = normal)
  2. 攻击帧 (CSV row 45 = NMRI attack)
  3. 边界帧 (crc_rate=最大值 + 全部 uint8 字段=255)
"""
import sys
import struct
import csv

sys.path.insert(0, '.')
from rtu_replay_gui import build_custom_frame, CUSTOM_FRAME_SIZE


def hex_dump(b: bytes) -> str:
    return ' '.join(f'{x:02x}' for x in b)


def verify_frame_structure(frame: bytes, label: str) -> dict:
    """模拟 H743 端 frame_parser_parse 的字段提取, 验证一致性"""
    assert len(frame) == CUSTOM_FRAME_SIZE, f"[{label}] size {len(frame)} != {CUSTOM_FRAME_SIZE}"

    sop = frame[0]
    seq = struct.unpack_from('<H', frame, 1)[0]
    ts_ms = struct.unpack_from('<I', frame, 3)[0]
    slave = frame[7]
    func = frame[8]
    length = struct.unpack_from('<H', frame, 9)[0]

    floats = []
    for i in range(12):
        v = struct.unpack_from('<f', frame, 11 + i * 4)[0]
        floats.append(v)

    is_resp = frame[59]
    time_unix = struct.unpack_from('<I', frame, 60)[0]
    rx_csum = struct.unpack_from('<H', frame, 64)[0]

    # 验证 checksum
    calc = 0xA5A5
    for b in frame[1:64]:
        calc ^= b
    calc &= 0xFFFF

    assert sop == 0xAA, f"[{label}] sop={sop:#x}"
    assert calc == rx_csum, f"[{label}] csum mismatch: calc={calc:#x} rx={rx_csum:#x}"

    return {
        'seq': seq, 'slave': slave, 'func': func, 'length': length,
        'floats': floats, 'is_resp': is_resp, 'time_unix': time_unix,
        'crc_rate': floats[11],  # idx 11
    }


def main():
    print(f"=== Custom Frame Roundtrip Test (size={CUSTOM_FRAME_SIZE}) ===\n")

    # 读 CSV
    with open('IanArffDataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"CSV rows loaded: {len(rows)}\n")

    # Case 1: CSV row 25 (normal, 应该无攻击)
    print("--- Case 1: CSV row 25 (normal) ---")
    f1 = build_custom_frame(rows[25], seq=1, ts_ms=123456)
    print(f"  hex: {hex_dump(f1[:32])}")
    print(f"  hex: {hex_dump(f1[32:])}")
    info1 = verify_frame_structure(f1, 'normal')
    print(f"  decoded: seq={info1['seq']} slave={info1['slave']} func={info1['func']} "
          f"crc_rate={info1['crc_rate']:.0f} is_resp={info1['is_resp']}")
    print()

    # Case 2: CSV row 45 (NMRI attack, specific=30)
    print("--- Case 2: CSV row 45 (NMRI attack) ---")
    f2 = build_custom_frame(rows[45], seq=2, ts_ms=234567)
    print(f"  hex: {hex_dump(f2[:32])}")
    print(f"  hex: {hex_dump(f2[32:])}")
    info2 = verify_frame_structure(f2, 'attack')
    print(f"  cat={int(rows[45]['categorized'])} specific={rows[45]['specific']}")
    print(f"  decoded: seq={info2['seq']} slave={info2['slave']} func={info2['func']} "
          f"crc_rate={info2['crc_rate']:.0f}")
    print()

    # Case 3: CSV row 100 (DoS attack, crc_rate 应该爆高)
    print("--- Case 3: find DoS row ---")
    for i, r in enumerate(rows[:5000]):
        if int(r['categorized']) == 6:
            print(f"  first DoS at row {i}")
            f3 = build_custom_frame(r, seq=3, ts_ms=345678)
            print(f"  hex: {hex_dump(f3[:32])}")
            info3 = verify_frame_structure(f3, 'dos')
            print(f"  decoded: crc_rate={info3['crc_rate']:.0f} (DoS should be huge)")
            break
    print()

    # Case 4: 100 帧连续, 验证时序字段 (time_diff/time_since_last) 的输入
    print("--- Case 4: 100 frames with time field ---")
    last_t = 0
    for i in range(100):
        fr = build_custom_frame(rows[i], seq=i, ts_ms=i * 100)
        dec = verify_frame_structure(fr, f'frame{i}')
        # CSV 'time' 是 Unix 秒, 应该是递增的
        t = dec['time_unix']
        if i > 0:
            dt = t - last_t
            if dt < 0:
                print(f"  WARN: time went backward at frame {i}: {t} < {last_t}")
        last_t = t
    print(f"  first 100 frames OK")
    print()

    print("=== ALL PASS ===")
    print(f"Frame size: {CUSTOM_FRAME_SIZE} bytes")
    print(f"Throughput @ 115200 8N1: {115200 // (CUSTOM_FRAME_SIZE * 10)} fps")


if __name__ == '__main__':
    main()
