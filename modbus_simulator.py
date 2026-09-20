"""Modbus 模拟器 - 主程序入口。

CLI 模式:
    python modbus_simulator.py [data_file]   # 启动 GUI
    python modbus_simulator.py --self-test   # 自检模式
"""
import struct
import sys
from typing import List


# 帧字段顺序定义：每个元组 (字段名, 字节数, struct 格式)
_FRAME_FIELDS = [
    ("address", 1, "B"), ("function", 1, "B"), ("length", 1, "B"),
    ("setpoint", 1, "B"), ("gain", 1, "B"), ("reset", 1, "B"),
    ("deadband", 1, "B"), ("cycle", 1, "B"), ("rate", 1, "B"),
    ("system", 1, "B"), ("control", 1, "B"), ("pump", 1, "B"),
    ("solenoid", 1, "B"), ("pressure", 1, "B"),
    ("crc", 2, ">H"),      # big-endian uint16
    ("command", 1, "B"),
    ("time", 4, ">I"),     # big-endian uint32
]


def _coerce_byte(value, size: int) -> int:
    """将任意值转换为整数，按 size 字节截断。"""
    if value is None or value == "":
        return 0
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return 0
    mask = (1 << (size * 8)) - 1
    return v & mask


def build_frame(record: dict) -> List[int]:
    """根据 CSV/XLSX 记录构建 21 字节 Modbus 风格帧。

    返回整数列表（每个元素 0-255）。
    """
    frame: List[int] = []
    for name, size, _fmt in _FRAME_FIELDS:
        raw = _coerce_byte(record.get(name), size)
        if size == 1:
            frame.append(raw)
        else:
            # 大端拆分
            for i in range(size - 1, -1, -1):
                frame.append((raw >> (i * 8)) & 0xFF)
    return frame


def frame_to_hex(frame: List[int]) -> str:
    """将字节列表格式化为 'XX XX ... XX'。"""
    return " ".join(f"{b:02X}" for b in frame)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        print("Self-test mode not yet implemented")
        return 1
    print("GUI mode not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
