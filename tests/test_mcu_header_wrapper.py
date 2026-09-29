"""MCU header wrapper + 23-dim feature tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ids.inference import FEATURE_COLUMNS_23, extract_features_23


_FULL_RECORD = {
    "address": 4, "function": 3, "length": 16,
    "setpoint": 0, "gain": 5, "reset": 0.0, "deadband": 0.0,
    "cycle": 0, "rate": 0, "system": 0, "control": 0,
    "pump": 0, "solenoid": 0, "pressure": 0.0, "crc": 0,
    "command": 0, "time_diff": 0.1,
    "time_since_last_same_addr_func": 0.0,
    "is_unusual_fc": 0, "is_response": 1,
    "press_mean_w": 0.0, "crc_mean_w": 0.0, "cmd_count_w": 0,
    "crc_max_w": 0.0, "resp_count_w": 0,
    "cmd_resp_balance_w": 0.0, "length_nunique_w": 0,
    "unusual_count_w": 0,
}


class TestExtractFeatures23(unittest.TestCase):
    def test_columns_length_23(self):
        self.assertEqual(len(FEATURE_COLUMNS_23), 23)

    def test_columns_first_three(self):
        self.assertEqual(
            FEATURE_COLUMNS_23[:3],
            ("address", "function", "gain"),
        )

    def test_columns_skips_length_setpoint(self):
        # 23-dim must skip length (idx 2) and setpoint (idx 3) from source
        self.assertNotIn("length", FEATURE_COLUMNS_23)
        self.assertNotIn("setpoint", FEATURE_COLUMNS_23)

    def test_extract_returns_shape_23(self):
        feats = extract_features_23(_FULL_RECORD)
        self.assertEqual(feats.shape, (23,))
        self.assertEqual(feats.dtype, np.float32)

    def test_extract_address_value(self):
        feats = extract_features_23(_FULL_RECORD)
        self.assertEqual(float(feats[0]), 4.0)

    def test_extract_function_value(self):
        feats = extract_features_23(_FULL_RECORD)
        self.assertEqual(float(feats[1]), 3.0)

    def test_extract_skips_length_and_setpoint(self):
        # Indices 2 and 3 in source are length(16) and setpoint(0); 23-dim skips both
        feats = extract_features_23(_FULL_RECORD)
        # The 3rd value (index 2 in 23-dim) should be "gain" = 5, not length
        self.assertEqual(float(feats[2]), 5.0)


if __name__ == "__main__":
    unittest.main()
