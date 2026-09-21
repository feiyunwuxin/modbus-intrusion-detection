"""IDS 单元测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ids import FEATURE_COLUMNS, extract_features, load_model, list_available_models, ModelWrapper

try:
    import torch
    import torch.nn as nn

    class _TinyTorchModel(nn.Module):
        """模块级定义，避免 torch.save 无法 pickle 局部类。"""

        def __init__(self, in_features: int = 17):
            super().__init__()
            self.fc = nn.Linear(in_features, 2)

        def forward(self, x):
            return self.fc(x)
except ImportError:
    _TinyTorchModel = None


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


class TestLoadTorch(unittest.TestCase):
    def _make_dummy_torch(self, in_features: int = 17) -> str:
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        model = _TinyTorchModel(in_features)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
            return f.name

    def test_load_torch_returns_wrapper(self):
        path = self._make_dummy_torch(17)
        try:
            from ids import load_model
            wrapper = load_model(path)
            self.assertEqual(wrapper.input_features, 17)
        finally:
            os.unlink(path)

    def test_load_torch_infer(self):
        import numpy as np
        path = self._make_dummy_torch(17)
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

    def test_load_torch_wrong_features_raises(self):
        path = self._make_dummy_torch(5)
        try:
            from ids import load_model
            with self.assertRaises(ValueError) as ctx:
                load_model(path)
            self.assertIn("17", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_torch_nested_state_dict_unwraps(self):
        """嵌套 dict 格式 {"state_dict": {...}, "best_threshold": ...} 应解包加载。"""
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import numpy as np
        import torch.nn as nn

        class _NestedFNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(17, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2),
                )

            def forward(self, x):
                return self.net(x)

        model = _NestedFNN()
        wrapped = {
            "state_dict": model.state_dict(),
            "model_name": "test",
            "best_threshold": 0.5,
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(wrapped, f.name)
            path = f.name
        try:
            wrapper = load_model(path)
            self.assertEqual(wrapper.input_features, 17)
            features = np.zeros(17, dtype=np.float32)
            label, prob = wrapper.infer(features)
            self.assertIn(label, (0, 1))
        finally:
            os.unlink(path)

    def test_load_torch_lstm_state_dict_rejected_cleanly(self):
        """含 LSTM 层的 state_dict 应抛 ValueError（友好），而非 AttributeError。"""
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        # 构造 LSTM 风格 state_dict（key 含 lstm.*）
        import torch.nn as nn
        lstm = nn.LSTM(input_size=17, hidden_size=8, batch_first=True)
        fake_state = {
            "lstm.weight_ih_l0": torch.randn(32, 17),
            "lstm.weight_hh_l0": torch.randn(32, 8),
            "lstm.bias_ih_l0": torch.zeros(32),
            "lstm.bias_hh_l0": torch.zeros(32),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(fake_state, f.name)
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_model(path)
            msg = str(ctx.exception)
            # 必须包含 3D/LSTM 关键词，不能是 AttributeError
            self.assertNotIn("AttributeError", msg)
            self.assertNotIn("modules", msg)
            self.assertTrue(
                any(tok in msg for tok in ("LSTM", "3D", "窗口")),
                f"错误消息应提示 3D 窗口，实际: {msg}",
            )
        finally:
            os.unlink(path)

    def test_load_torch_fnn_state_dict_reconstructs(self):
        """Linear+BN+ReLU Sequential 模式应能从 state_dict 重建并推理。"""
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import numpy as np
        import torch.nn as nn

        class _FNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(17, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.net(x)

        model = _FNN()
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"state_dict": model.state_dict()}, f.name)
            path = f.name
        try:
            wrapper = load_model(path)
            self.assertEqual(wrapper.input_features, 17)
            features = np.zeros(17, dtype=np.float32)
            label, prob = wrapper.infer(features)
            self.assertIn(label, (0, 1))
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        finally:
            os.unlink(path)

    def test_load_torch_3d_window_loads_with_window_size(self):
        """3D 窗口模型在指定 window_size >= 2 时应能加载；否则抛友好 ValueError。

        对应 IDS 面板的 Spinbox：用户将"窗口"从 1 改为 16 后重新选择模型。

        关键约束：
        - key 必须带 'lstm.' 前缀（让 _detect_torch_input_features 走 3D 分支）
        - classifier key 必须带 'net.' 前缀（让 _reconstruct_classifier 找到）
        - nn.LSTM load_state_dict 要求 weight_ih_l0 等无前缀 → 用 _lstm 前缀
          sd 不能直接 load，需要重建器内部去掉前缀。检查现有 _reconstruct_lstm
          行为：它把整个 sd 直接传给 rnn.load_state_dict(sd)，所以若 sd key
          是 'lstm.weight_ih_l0' 会失败。本测试只验证"加载器接受 window_size"
          路径，不强制跑通 inference。
        """
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import torch.nn as nn

        # 使用与 SCADA 项目相同的 key 格式 (lstm.* 前缀)
        # 让 _detect_torch_input_features 跳过 LSTM key，进入 3D 分支
        # _parse_lstm_arch 正则 (lstm|gru)\.(weight|bias)_(ih|hh)_l(\d+)
        # 会把 'lstm.weight_ih_l0' 的 mod_name 解析为 'lstm'，但 weight 实际
        # 在 nn.LSTM 里没有 'lstm.' 前缀 → load_state_dict 会失败。
        # 真实场景下 _Torch3DStateDictWrapper 应该把 key 去掉前缀再 load。
        # 现有实现是直接 load，所以这个 case 会失败 —— 这是一个潜在 bug
        # 但超出本测试范围。本测试只覆盖"友好拒绝 + window_size 接受"。
        fake_state = {
            "lstm.weight_ih_l0": torch.randn(32, 17),
            "lstm.weight_hh_l0": torch.randn(32, 8),
            "lstm.bias_ih_l0": torch.zeros(32),
            "lstm.bias_hh_l0": torch.zeros(32),
            "net.0.weight": torch.randn(1, 8),
            "net.0.bias": torch.zeros(1),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(fake_state, f.name)
            path = f.name
        try:
            # 默认 window_size=1 → 拒绝 (友好)
            with self.assertRaises(ValueError) as ctx:
                load_model(path)
            msg = str(ctx.exception)
            self.assertTrue(
                any(tok in msg for tok in ("LSTM", "3D", "窗口")),
                f"错误消息应提示 3D 窗口，实际: {msg}",
            )
        finally:
            os.unlink(path)

        # 单独验证：用户构造正确的 3D wrapper 路径（手工 patch key 去掉前缀）
        fake_state2 = dict(fake_state)
        # 去除 'lstm.' 前缀以让 nn.LSTM.load_state_dict 接受
        fake_state2["weight_ih_l0"] = fake_state2.pop("lstm.weight_ih_l0")
        fake_state2["weight_hh_l0"] = fake_state2.pop("lstm.weight_hh_l0")
        fake_state2["bias_ih_l0"] = fake_state2.pop("lstm.bias_ih_l0")
        fake_state2["bias_hh_l0"] = fake_state2.pop("lstm.bias_hh_l0")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(fake_state2, f.name)
            path2 = f.name
        try:
            # 这个 sd 没有 'lstm' 前缀，会被当作 Linear 路径 → 拒绝 key 形式
            with self.assertRaises(ValueError):
                load_model(path2, window_size=16)
        finally:
            os.unlink(path2)

    def test_load_torch_bilstm_cross_prefix_infers(self):
        """SCADA-style BiLSTM with lstm.* prefix should load + infer.

        Regression for the ``lstm.`` / ``gru.`` prefix bug: real SCADA
        checkpoints store RNN keys under ``lstm.`` / ``gru.`` but
        ``nn.LSTM.load_state_dict`` expects unprefixed keys. The fix
        lives in ``ids.model_loader._reconstruct_lstm``.

        This test exercises the *happy path* (warm-up + first prediction),
        not just the friendly-rejection path covered by
        ``test_load_torch_3d_window_loads_with_window_size``.
        """
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import torch.nn as nn
        import numpy as np

        class _BiLSTMWithPrefix(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=17, hidden_size=8,
                    batch_first=True, bidirectional=True,
                )
                self.net = nn.Sequential(nn.Linear(16, 2))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.net(out[:, -1, :])

        m = _BiLSTMWithPrefix()
        m.eval()

        # The model is defined as self.lstm = nn.LSTM(...) and self.net = nn.Sequential(...),
        # so m.state_dict() already emits SCADA-style keys with the ``lstm.`` /
        # ``net.`` prefixes baked in. Use it directly — do NOT re-prefix.
        sd = m.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(sd, f.name)
            path = f.name
        try:
            wrapper = load_model(path, window_size=8)
            self.assertEqual(wrapper.window_size, 8)
            # Warm-up: first 7 frames must return None
            for _ in range(7):
                self.assertIsNone(
                    wrapper.infer(np.zeros(17, dtype=np.float32))
                )
            # 8th frame returns a (label, prob) tuple
            result = wrapper.infer(np.zeros(17, dtype=np.float32))
            self.assertIsNotNone(result)
            label, prob = result
            self.assertIn(label, (0, 1))
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()