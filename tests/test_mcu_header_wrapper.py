"""MCU header wrapper + 23-dim feature tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ids.inference import FEATURE_COLUMNS_23, extract_features_23
from ids.model_loader import _parse_mcu_header

# Path to the actual MCU header file in repo
_MCU_HEADER_PATH = (
    Path(__file__).resolve().parent.parent
    / "KeilH743"
    / "H743"
    / "Core"
    / "Inc"
    / "model_v4_se_23dim_ch32_hybrid_s42.h"
)


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


class TestParseMcuHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _MCU_HEADER_PATH.exists():
            raise unittest.SkipTest(f"MCU header not found: {_MCU_HEADER_PATH}")
        cls.parsed = _parse_mcu_header(_MCU_HEADER_PATH)

    def test_returns_dict(self):
        self.assertIsInstance(self.parsed, dict)

    def test_has_tcn_0_conv1_weight(self):
        w = self.parsed["tcn.0.conv1.weight"]
        self.assertEqual(w.dtype, np.int8)
        # Shape from ch32_inference.h: tcn.0.conv1 is (32, 23, 3) = 2208 elements
        self.assertEqual(w.shape, (32, 23, 3))

    def test_has_tcn_0_conv1_scale(self):
        s = self.parsed["tcn.0.conv1.scale"]
        self.assertEqual(s.dtype, np.float32)
        self.assertEqual(s.shape, (32,))

    def test_has_tcn_0_conv1_bias(self):
        b = self.parsed["tcn.0.conv1.bias"]
        self.assertEqual(b.dtype, np.float32)
        self.assertEqual(b.shape, (32,))

    def test_has_all_three_tcn_blocks(self):
        for bi in range(3):
            for ck in ("conv1.weight", "conv2.weight", "se.fc1.weight", "se.fc2.weight"):
                key = f"tcn.{bi}.{ck}"
                self.assertIn(key, self.parsed, f"missing {key}")

    def test_has_tcn_0_residual(self):
        # Only block 0 has residual
        self.assertIn("tcn.0.residual.weight", self.parsed)
        self.assertEqual(self.parsed["tcn.0.residual.weight"].dtype, np.int8)
        self.assertEqual(self.parsed["tcn.0.residual.weight"].shape, (32, 23, 1))

    def test_block_1_2_no_residual(self):
        self.assertNotIn("tcn.1.residual.weight", self.parsed)
        self.assertNotIn("tcn.2.residual.weight", self.parsed)

    def test_has_fc1_fc2_weights_fp32(self):
        # fc1, fc2 weights are FP32 (per quantize script comment)
        self.assertEqual(self.parsed["fc1.weight"].dtype, np.float32)
        self.assertEqual(self.parsed["fc1.weight"].shape, (32, 32))
        self.assertEqual(self.parsed["fc2.weight"].dtype, np.float32)
        self.assertEqual(self.parsed["fc2.weight"].shape, (1, 32))

    def test_has_all_biases(self):
        for bi in range(3):
            for ck in ("conv1.bias", "conv2.bias", "se.fc1.bias", "se.fc2.bias"):
                key = f"tcn.{bi}.{ck}"
                self.assertIn(key, self.parsed, f"missing {key}")
        self.assertIn("fc1.bias", self.parsed)
        self.assertIn("fc2.bias", self.parsed)


if __name__ == "__main__":
    unittest.main()
