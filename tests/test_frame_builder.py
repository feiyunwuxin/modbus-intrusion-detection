"""帧构建器单元测试。"""
import unittest
import sys
from pathlib import Path

# 允许从项目根导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modbus_simulator import build_frame, frame_to_hex


class TestBuildFrame(unittest.TestCase):
    def test_basic_frame_from_csv_row(self):
        """CSV 第 1 行: 4,3,16,0,0,0,0,0,0,0,0,0,0,0,12869,1,1418682163"""
        record = {
            "address": 4, "function": 3, "length": 16,
            "setpoint": 0, "gain": 0, "reset": 0.0, "deadband": 0.0,
            "cycle": 0, "rate": 0, "system": 0, "control": 0,
            "pump": 0, "solenoid": 0, "pressure": 0.0,
            "crc": 12869, "command": 1, "time": 1418682163,
        }
        frame = build_frame(record)
        self.assertEqual(len(frame), 21)
        # 前 14 字节：address=4, function=3, length=16, 其它全 0
        self.assertEqual(frame[:14], [4, 3, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # CRC: 12869 = 0x3245, big-endian
        self.assertEqual(frame[14], 0x32)
        self.assertEqual(frame[15], 0x45)
        # command=1
        self.assertEqual(frame[16], 1)
        # time: 1418682163 = 0x548F5F33, big-endian
        # NOTE: brief docstring said 0x5487F4B3, but 1418682163 actually = 0x548F5F33
        self.assertEqual(frame[17:21], [0x54, 0x8F, 0x5F, 0x33])

    def test_float_fields_truncated(self):
        """浮点字段取整。"""
        record = {
            "address": 4, "function": 16, "length": 90,
            "setpoint": 10, "gain": 115, "reset": 0.2, "deadband": 0.5,
            "cycle": 1, "rate": 0, "system": 0, "control": 1,
            "pump": 0, "solenoid": 0, "pressure": 0.689655,
            "crc": 17219, "command": 1, "time": 1418682165,
        }
        frame = build_frame(record)
        # setpoint=10, gain=115, reset=0→0, deadband=0→0, cycle=1
        self.assertEqual(frame[3], 10)
        self.assertEqual(frame[4], 115)
        self.assertEqual(frame[5], 0)   # 0.2 截断
        self.assertEqual(frame[6], 0)   # 0.5 截断
        self.assertEqual(frame[7], 1)
        # system=0, control=1, pump=0, solenoid=0, pressure=0
        self.assertEqual(frame[9:14], [0, 1, 0, 0, 0])

    def test_overflow_truncation(self):
        """超出字段宽度的值截断处理。"""
        record = {
            "address": 4, "function": 3, "length": 16,
            "setpoint": 0, "gain": 0, "reset": 0, "deadband": 0,
            "cycle": 0, "rate": 0, "system": 0, "control": 0,
            "pump": 0, "solenoid": 0, "pressure": 0,
            "crc": 0x1FFFF,  # 17-bit > uint16
            "command": 1, "time": 0x1FFFFFFFF,  # 33-bit > uint32
        }
        frame = build_frame(record)
        self.assertEqual(frame[14], 0xFF)
        self.assertEqual(frame[15], 0xFF)
        self.assertEqual(frame[17:21], [0xFF, 0xFF, 0xFF, 0xFF])

    def test_missing_fields_default_zero(self):
        """缺失字段视为 0。"""
        record = {"address": 4, "function": 3, "length": 16,
                  "crc": 12869, "command": 1, "time": 1418682163}
        frame = build_frame(record)
        self.assertEqual(len(frame), 21)
        self.assertEqual(frame[0], 4)
        self.assertEqual(frame[14:17], [0x32, 0x45, 1])


class TestFrameToHex(unittest.TestCase):
    def test_hex_format(self):
        frame = [0x04, 0x03, 0x10, 0x00, 0x32, 0x45, 0x01]
        result = frame_to_hex(frame)
        self.assertEqual(result, "04 03 10 00 32 45 01")

    def test_hex_uppercase(self):
        frame = [0xab, 0xcd]
        result = frame_to_hex(frame)
        self.assertEqual(result, "AB CD")

    def test_empty_frame(self):
        self.assertEqual(frame_to_hex([]), "")


if __name__ == "__main__":
    unittest.main()
