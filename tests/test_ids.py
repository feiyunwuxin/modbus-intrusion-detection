"""IDS 单元测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ids import FEATURE_COLUMNS, extract_features


_FULL_ROW = {
    "address": 4, "function": 3, "length": 16,
    "setpoint": 0, "gain": 0, "reset": 0.0, "deadband": 0.0,
    "cycle": 0, "rate": 0, "system": 0, "control": 0,
    "pump": 0, "solenoid": 0, "pressure": 0.0,
    "crc": 12869, "command": 1, "time": 1418682163,
    "binary": 0, "categorized": 0, "specific": 0,
}


class TestFeatureColumns(unittest.TestCase):
    def test_columns_count(self):
        self.assertEqual(len(FEATURE_COLUMNS), 17)

    def test_columns_order(self):
        expected = (
            "address", "function", "length",
            "setpoint", "gain", "reset", "deadband", "cycle", "rate",
            "system", "control", "pump", "solenoid", "pressure",
            "crc", "command", "time",
        )
        self.assertEqual(FEATURE_COLUMNS, expected)


class TestExtractFeatures(unittest.TestCase):
    def test_shape_and_dtype(self):
        import numpy as np
        features = extract_features(_FULL_ROW)
        self.assertEqual(features.shape, (17,))
        self.assertEqual(features.dtype, np.float32)

    def test_values_first_row(self):
        """CSV 第 1 行: 4,3,16,0,...,0,12869,1,1418682163"""
        features = extract_features(_FULL_ROW)
        self.assertEqual(int(features[0]), 4)    # address
        self.assertEqual(int(features[1]), 3)    # function
        self.assertEqual(int(features[2]), 16)   # length
        # 索引 3-13: setpoint..pressure 全 0
        for i in range(3, 14):
            self.assertEqual(int(features[i]), 0)
        # crc=12869, command=1, time=1418682163
        self.assertEqual(int(features[14]), 12869)
        self.assertEqual(int(features[15]), 1)
        self.assertEqual(int(features[16]), 1418682163)

    def test_float_truncation(self):
        row = dict(_FULL_ROW, reset=0.7, deadband=0.9, pressure=0.689)
        features = extract_features(row)
        # reset=0.7→0, deadband=0.9→0, pressure=0.689→0
        self.assertEqual(int(features[5]), 0)
        self.assertEqual(int(features[6]), 0)
        self.assertEqual(int(features[13]), 0)

    def test_missing_field_defaults_zero(self):
        row = {"address": 4, "function": 3, "length": 16,
               "crc": 12869, "command": 1, "time": 1418682163}
        features = extract_features(row)
        self.assertEqual(features.shape, (17,))
        # 缺失字段视为 0
        for i in range(3, 14):
            self.assertEqual(int(features[i]), 0)


if __name__ == "__main__":
    unittest.main()