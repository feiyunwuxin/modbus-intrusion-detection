"""IDS 单元测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ids import FEATURE_COLUMNS, extract_features, load_model, list_available_models, ModelWrapper


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
        # crc=12869, command=1, time=1418682163 (CSV int; float32 → 1418682112 due to mantissa rounding)
        self.assertEqual(int(features[14]), 12869)
        self.assertEqual(int(features[15]), 1)
        self.assertEqual(int(features[16]), 1418682112)

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


import tempfile
import os


class TestLoadSklearn(unittest.TestCase):
    def _make_dummy_sklearn(self, n_features: int = 17) -> str:
        """训练一个 LogisticRegression 并保存，返回路径。"""
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            import joblib
        except ImportError:
            self.skipTest("sklearn/joblib not installed")
        X = np.random.RandomState(42).randn(20, n_features)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(model, f.name)
            return f.name

    def test_load_sklearn_returns_wrapper(self):
        path = self._make_dummy_sklearn(17)
        try:
            from ids import load_model
            wrapper = load_model(path)
            self.assertEqual(wrapper.input_features, 17)
        finally:
            os.unlink(path)

    def test_load_sklearn_infer_returns_tuple(self):
        import numpy as np
        path = self._make_dummy_sklearn(17)
        try:
            from ids import load_model
            wrapper = load_model(path)
            features = np.zeros(17, dtype=np.float32)
            label, prob = wrapper.infer(features)
            self.assertIn(label, (0, 1))
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        finally:
            os.unlink(path)

    def test_load_sklearn_wrong_features_raises(self):
        path = self._make_dummy_sklearn(5)  # 仅 5 特征
        try:
            from ids import load_model
            with self.assertRaises(ValueError) as ctx:
                load_model(path)
            self.assertIn("17", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_list_available_models(self):
        """扫描目录只返回 model_*.{pt,joblib} 文件。"""
        from ids import list_available_models
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "model_a.joblib").write_bytes(b"")
            (td / "model_b.pt").write_bytes(b"")
            (td / "other.txt").write_text("ignore me")
            (td / "model_no_ext").write_bytes(b"")
            result = list_available_models(str(td))
            names = [Path(p).name for p in result]
            self.assertIn("model_a.joblib", names)
            self.assertIn("model_b.pt", names)
            self.assertNotIn("other.txt", names)
            self.assertNotIn("model_no_ext", names)


if __name__ == "__main__":
    unittest.main()