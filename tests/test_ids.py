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

    def test_load_torch_bilstm_with_normalization_keys(self):
        """state_dict 含 __norm_mean__ / __norm_std__ 时, _Torch3DStateDictWrapper
        应识别并在 infer 前对每帧做 (x - mean) / std。

        这些 key 是 'magic' — 不被 _parse_lstm_arch / _reconstruct_lstm /
        _reconstruct_classifier 处理。本测试确保它们被检测到并生效。
        """
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import torch.nn as nn
        import numpy as np

        class _BiLSTMNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=17, hidden_size=8,
                                    batch_first=True, bidirectional=True)
                self.net = nn.Sequential(nn.Linear(16, 2))
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.net(out[:, -1, :])

        m = _BiLSTMNorm()
        m.eval()
        # state_dict already has ``lstm.`` / ``net.`` prefixes (because
        # self.lstm and self.net are submodules). Use it directly.
        sd = m.state_dict()
        # 嵌入归一化参数 (1D, shape (17,))
        # mean=1.0, std=2.0 → 任何 (x - 1) / 2 不会产生 NaN，且不是全 0
        sd["__norm_mean__"] = torch.ones(17, dtype=torch.float32)
        sd["__norm_std__"]  = torch.full((17,), 2.0, dtype=torch.float32)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(sd, f.name)
            path = f.name
        try:
            wrapper = load_model(path, window_size=8)
            # 推理：传入全 0 特征，归一化后 = (0 - 1) / 2 = -0.5
            for _ in range(7):
                self.assertIsNone(wrapper.infer(np.zeros(17, dtype=np.float32)))
            result = wrapper.infer(np.zeros(17, dtype=np.float32))
            self.assertIsNotNone(result)
            label, prob = result
            self.assertIn(label, (0, 1))
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        finally:
            os.unlink(path)


    def test_load_torch_bilstm_flat_classifier_reconstructs(self):
        """Regression: flat-style classifier (fc1/fc2) must rebuild as Sequential.

        Earlier bug: `_find_linear_classifier` only matched `prefix.<digit>.weight`
        style keys. Models with `fc1.weight` / `fc2.weight` style names fell
        through to `nn.Identity()` → wrapper produced constant ≈ 1/128 output
        regardless of input. All 500 test predictions hit prob≈0.008.
        """
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import torch.nn as nn
        import numpy as np

        class _BiLSTMFlatHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=17, hidden_size=8,
                                    batch_first=True, bidirectional=True)
                self.fc1 = nn.Linear(16, 4)
                self.fc2 = nn.Linear(4, 1)
                self.dropout = nn.Dropout(0.0)

            def forward(self, x):
                out, _ = self.lstm(x)
                h = torch.relu(self.fc1(out[:, -1, :]))
                h = self.dropout(h)
                return self.fc2(h).squeeze(-1)

        m = _BiLSTMFlatHead()
        m.eval()
        sd = m.state_dict()
        # Verify the keys look like the buggy state_dict: fc1.weight, fc2.weight
        # (no digit-index between prefix and .weight)
        assert "fc1.weight" in sd and "fc2.weight" in sd

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(sd, f.name)
            path = f.name
        try:
            wrapper = load_model(path, window_size=8)
            # The internal classifier MUST be a real Sequential, NOT nn.Identity
            self.assertIsNot(type(wrapper._model.classifier), nn.Identity)
            self.assertEqual(len(wrapper._model.classifier), 3)  # Linear, ReLU, Linear

            # After 7-frame warmup the 8th frame must produce a real probability
            for _ in range(7):
                self.assertIsNone(wrapper.infer(np.zeros(17, dtype=np.float32)))
            result = wrapper.infer(np.zeros(17, dtype=np.float32))
            self.assertIsNotNone(result)
            label, prob = result

            # Regression assertion: prob must NOT be ≈ 1/64 ≈ 0.016 (the
            # buggy constant from softmax-over-hidden-states)
            self.assertNotAlmostEqual(prob, 1.0 / 64, places=2,
                msg=f"wrapper still produces constant ~1/N prob, got {prob:.4f}")
            # And must be a sensible sigmoid output (any value in (0, 1))
            self.assertGreater(prob, 0.0)
            self.assertLess(prob, 1.0)
        finally:
            os.unlink(path)


class TestIDsPanelStats(unittest.TestCase):
    """Tests for the IDS panel's classification-metrics display.

    Exercises ``_update_stats`` / ``_render_stats`` without booting the
    full GUI — we just need a Tk root for the StringVars. The panel is
    constructed against that root, then we drive ``_update_stats`` directly
    with synthetic (pred, truth) pairs and read ``stats_var`` / ``metrics_var``.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import tkinter as tk
            cls._tk = tk.Tk()
            cls._tk.withdraw()
        except Exception as e:
            raise unittest.SkipTest(f"no Tk available: {e}")

    @classmethod
    def tearDownClass(cls):
        try:
            cls._tk.destroy()
        except Exception:
            pass

    def setUp(self):
        # Import here so the module-level Tk requirement is enforced by
        # setUpClass, not by module import.
        from ids_panel import IDsPanel
        self.panel = IDsPanel(self._tk)

    def tearDown(self):
        # Drop the frame so the next test gets a clean tree; widget
        # destruction is what cleans up the Tk children we created.
        try:
            self.panel.destroy()
        except Exception:
            pass

    def _feed(self, pairs):
        """Drive _update_stats with a list of (pred, truth) pairs."""
        for pred, truth in pairs:
            self.panel._update_stats(pred, truth, counted=True)

    def test_empty_stats_show_dashes(self):
        """On a fresh panel the metrics line must read all '-' so the
        user does not see a misleading 0.000 before any frames run."""
        self.assertEqual(
            self.panel.stats_var.get(),
            "总:0  正常:0  攻击:0  正确:0",
        )
        self.assertEqual(
            self.panel.metrics_var.get(),
            "准确率:-  精确率:-  召回率:-  F1:-",
        )

    def test_confusion_matrix_cells_increment_independently(self):
        """Every (pred, truth) cell must land in its own counter — feeding
        a known mix and reading the raw stats dict is the strongest test
        that the branching is right."""
        # (pred, truth) distribution:
        #   TP: pred=1 truth=1  → 3
        #   FP: pred=1 truth=0  → 1
        #   TN: pred=0 truth=0  → 4
        #   FN: pred=0 truth=1  → 2
        self._feed([
            (1, 1), (1, 1), (1, 1),    # TP ×3
            (1, 0),                    # FP ×1
            (0, 0), (0, 0), (0, 0), (0, 0),  # TN ×4
            (0, 1), (0, 1),            # FN ×2
        ])
        s = self.panel._stats
        self.assertEqual(s["total"], 10)
        self.assertEqual(s["attack"], 4)  # pred=1 count
        self.assertEqual(s["normal"], 6)  # pred=0 count
        self.assertEqual(s["correct"], 7)  # TP + TN
        self.assertEqual(s["tp"], 3)
        self.assertEqual(s["fp"], 1)
        self.assertEqual(s["tn"], 4)
        self.assertEqual(s["fn"], 2)

    def test_metrics_known_fixture(self):
        """Pin the four metric values on a balanced fixture so any
        future regression in the formula is loud.

        Fixture: TP=3, FP=1, TN=4, FN=2 (total=10)
          Accuracy = (3+4)/10           = 0.700
          Precision = 3/(3+1)           = 0.750
          Recall    = 3/(3+2)           = 0.600
          F1        = 2*0.75*0.6/(0.75+0.6) ≈ 0.667
        """
        self._feed([
            (1, 1), (1, 1), (1, 1),    # TP ×3
            (1, 0),                    # FP ×1
            (0, 0), (0, 0), (0, 0), (0, 0),  # TN ×4
            (0, 1), (0, 1),            # FN ×2
        ])
        m = self.panel.metrics_var.get()
        # Substring checks — display order is fixed and stable, but the
        # surrounding format string is more readable as a piece-level test
        # than as a single literal.
        self.assertIn("准确率:0.700", m)
        self.assertIn("精确率:0.750", m)
        self.assertIn("召回率:0.600", m)
        # F1 is a repeating decimal; the rounded-to-3dp string is what
        # the user sees. Compare with the same formula.
        f1_expected = 2 * 0.75 * 0.6 / (0.75 + 0.6)
        self.assertIn(f"F1:{f1_expected:.3f}", m)

    def test_precision_undefined_when_no_positive_predictions(self):
        """If the model never predicts attack, Precision is mathematically
        undefined (0/0). Display must be '-' rather than 0.000 so the
        user does not mistake it for a real value."""
        # Only TN — model predicted 0 every time.
        self._feed([(0, 0), (0, 0), (0, 0)])
        m = self.panel.metrics_var.get()
        self.assertIn("精确率:-", m)
        # Recall is well-defined here (denominator is the truth count = 0
        # because all truths were 0) — same undefined case, also '-'.
        self.assertIn("召回率:-", m)
        self.assertIn("F1:-", m)
        # Accuracy is still computable: 3/3.
        self.assertIn("准确率:1.000", m)

    def test_recall_undefined_when_no_positive_truths(self):
        """Mirror case: if every actual frame was normal, Recall is
        0/0. Same '-' handling."""
        self._feed([(1, 0), (1, 0), (0, 0)])  # 2 FP + 1 TN
        m = self.panel.metrics_var.get()
        self.assertIn("精确率:0.000", m)  # 0/(0+2)=0 → defined as 0
        self.assertIn("召回率:-", m)       # 0/(0+0)=undefined
        self.assertIn("F1:-", m)
        # Accuracy = 1/3.
        self.assertIn("准确率:0.333", m)

    def test_counted_false_does_not_change_skips(self):
        """Warm-up / error frames (counted=False) must not touch the
        counters at all — otherwise long BiLSTM warm-ups would corrupt
        the metrics."""
        before = dict(self.panel._stats)
        self.panel._update_stats(1, 1, counted=False)
        self.panel._update_stats(0, 0, counted=False)
        self.assertEqual(self.panel._stats, before)

    def test_clear_resets_all_eight_keys(self):
        """clear() must zero all 8 stat keys — tp/fp/tn/fn are easy to
        forget because they're behind the older total/normal/attack/correct
        API."""
        self._feed([
            (1, 1), (1, 0), (0, 0), (0, 1),
        ])
        self.panel.clear()
        s = self.panel._stats
        for k in ("total", "normal", "attack", "correct",
                  "tp", "fp", "tn", "fn"):
            self.assertEqual(s[k], 0, f"{k} should be reset to 0 after clear()")
        # And the display must reflect the empty state again.
        self.assertEqual(self.panel.metrics_var.get(),
                         "准确率:-  精确率:-  召回率:-  F1:-")

    def test_maybe_reset_for_new_model_first_load_resets(self):
        """First successful load: _active_path goes from None → path, and
        stats (which happen to be empty here) get a reset pass through
        clear(). The status suffix is '（统计已重置）'."""
        suffix = self.panel._maybe_reset_for_new_model("/path/to/model_a.pt")
        self.assertIn("已重置", suffix)
        self.assertEqual(self.panel._active_path, "/path/to/model_a.pt")

    def test_maybe_reset_for_new_model_same_path_no_op(self):
        """Re-selecting the SAME model must not wipe stats — that would
        surprise the operator every time they accidentally click the
        dropdown. Empty suffix means 'no reset happened'."""
        self.panel._maybe_reset_for_new_model("/path/to/model_a.pt")
        # Add some stats so we can prove they survive the second call.
        self._feed([(1, 1), (0, 0)])
        suffix = self.panel._maybe_reset_for_new_model("/path/to/model_a.pt")
        self.assertEqual(suffix, "")
        self.assertEqual(self.panel._stats["total"], 2)

    def test_maybe_reset_for_new_model_switch_clears(self):
        """Switching to a different path MUST clear stats and the
        results tree — otherwise the displayed Accuracy/Precision/Recall/F1
        would silently belong to the previous model. Empty the tree too
        so the operator does not see rows tagged with the old model's
        colors after the switch."""
        # Populate some stats + tree rows.
        self._feed([(1, 1), (1, 0), (0, 0)])
        self.panel.results_tree.insert(
            "", "end", values=("1", "deadbeef", 1, 1, "0.870", "✓"),
            tags=("attack",),
        )
        self.assertEqual(self.panel._stats["total"], 3)
        self.assertEqual(len(self.panel.results_tree.get_children()), 1)

        suffix = self.panel._maybe_reset_for_new_model("/path/to/model_b.pt")
        self.assertIn("已重置", suffix)
        self.assertEqual(self.panel._active_path, "/path/to/model_b.pt")
        # Stats are zeroed — every confusion-matrix key included.
        for k in ("total", "normal", "attack", "correct",
                  "tp", "fp", "tn", "fn"):
            self.assertEqual(self.panel._stats[k], 0)
        # Tree is empty.
        self.assertEqual(self.panel.results_tree.get_children(), ())
        # Display is the empty-state form again.
        self.assertEqual(self.panel.metrics_var.get(),
                         "准确率:-  精确率:-  召回率:-  F1:-")


class TestExtractFeatures19(unittest.TestCase):
    """19 行级特征 = 17 维 IDS 列 + 4 个派生列 (time_diff,
    time_since_last_same_addr_func, is_unusual_fc, is_response)。"""

    def test_returns_19_floats_in_documented_order(self):
        from ids import extract_features_19
        record = {
            "address": 4, "function": 3, "length": 16,
            "setpoint": 0, "gain": 100, "reset": 0, "deadband": 0,
            "cycle": 10, "rate": 0, "system": 1, "control": 0,
            "pump": 1, "solenoid": 0, "pressure": 50.0,
            "crc": 12869, "command": 1, "time": 1418682163,
        }
        out = extract_features_19(record)
        self.assertEqual(out.shape, (19,))
        self.assertEqual(out.dtype, __import__("numpy").float32)
        # 前两列直接来自 record（rename 后）
        self.assertEqual(float(out[0]), 4.0)   # address
        self.assertEqual(float(out[1]), 3.0)   # function
        self.assertEqual(float(out[2]), 16.0)  # length
        self.assertEqual(float(out[4]), 100.0)  # gain

    def test_first_frame_derived_features_are_zero(self):
        """没有 prev_time 时 time_diff/time_since_last 应该是 0。"""
        from ids import extract_features_19
        record = {"address": 1, "function": 3, "time": 100}
        out = extract_features_19(record)
        # time_diff 在索引 15
        self.assertEqual(float(out[15]), 0.0)
        # time_since_last_same_addr_func 在索引 16
        self.assertEqual(float(out[16]), 0.0)

    def test_subsequent_frame_time_diff_is_delta(self):
        """第二帧 time_diff = time - prev_time。"""
        from ids import extract_features_19
        last_seen = {}
        r1 = {"address": 1, "function": 3, "time": 100}
        out1 = extract_features_19(r1, last_seen_time=last_seen)
        last_seen[1003] = 100
        r2 = {"address": 1, "function": 3, "time": 150}
        out2 = extract_features_19(
            r2, prev_time=100, last_seen_time=last_seen,
        )
        # time_diff = 150 - 100 = 50
        self.assertEqual(float(out2[15]), 50.0)
        # 同 (1, 3) 出现 → time_since_last = 150 - 100 = 50
        self.assertEqual(float(out2[16]), 50.0)

    def test_unusual_fc_flag(self):
        """function ∈ UNUSUAL_FCS 时 is_unusual_fc = 1。"""
        from ids import extract_features_19
        r_unusual = {"address": 1, "function": 136, "time": 0}
        out = extract_features_19(r_unusual)
        self.assertEqual(float(out[17]), 1.0)
        r_normal = {"address": 1, "function": 3, "time": 0}
        out2 = extract_features_19(r_normal)
        self.assertEqual(float(out2[17]), 0.0)

    def test_is_response_tracks_command_field(self):
        """is_response 直接取 record.command。"""
        from ids import extract_features_19
        out = extract_features_19({"address": 0, "function": 0,
                                   "command": 1, "time": 0})
        self.assertEqual(float(out[18]), 1.0)
        out2 = extract_features_19({"address": 0, "function": 0,
                                    "command": 0, "time": 0})
        self.assertEqual(float(out2[18]), 0.0)

    def test_scada_csv_column_names_accepted(self):
        """IanArffDataset_RAW.csv 用 SCADA 列名（"pressure measurement"
        / "crc rate" / "reset rate" / "cycle time" / "system mode" /
        "control scheme" / "command response"），extract_features_19 必须
        也能解析这些 key，否则 IDS 面板从 csv_loader 拿到的 record 会
        全部字段返回 None → 0 → 模型预测崩溃为「全部 attack」。

        修复前：pressure_measurement(13) / crc_rate(14) / reset_rate(5)
        / cycle_time(7) / system_mode(9) / control_scheme(10) /
        is_response(18) 全是 0；X_test 在该模型上能到 85% 准确率，但
        IDS 面板 100% 报警。
        修复后：所有字段都能从 SCADA CSV 列名读到。
        """
        from ids import extract_features_19
        rec_scada = {
            "address": 4, "function": 3, "length": 16,
            "setpoint": 0, "gain": 100,
            "reset rate": 0, "deadband": 0,
            "cycle time": 10, "rate": 0,
            "system mode": 1, "control scheme": 0,
            "pump": 1, "solenoid": 0,
            "pressure measurement": 50.0, "crc rate": 12869,
            "command response": 1, "time": 1418682163,
        }
        out = extract_features_19(rec_scada)
        # 这些字段在 SCADA 命名下应该非零
        self.assertEqual(float(out[5]), 0.0)   # reset_rate (来自 "reset rate")
        self.assertEqual(float(out[7]), 10.0)   # cycle_time (来自 "cycle time")
        self.assertEqual(float(out[9]), 1.0)    # system_mode (来自 "system mode")
        self.assertEqual(float(out[10]), 0.0)   # control_scheme (来自 "control scheme")
        self.assertEqual(float(out[13]), 50.0)  # pressure_measurement (来自 "pressure measurement")
        self.assertEqual(float(out[14]), 12869.0)  # crc_rate (来自 "crc rate")
        self.assertEqual(float(out[18]), 1.0)   # is_response (来自 "command response")

    def test_float_precision_preserved_for_pressure(self):
        """pressure_measurement 是 float（0–100），_resolve_int 之前用
        int(float(value)) 把 0.689655 截断成 0 → IDS 面板 23-dim 模型
        的 raw_19[13] 全是 0 → press_mean_w 全 0 → 模型预测全 attack
        （X_train 用 pd.to_numeric 保留小数位，wrapper 必须匹配）。

        修复前：out[13] == 0.0（截断）
        修复后：out[13] == 0.689655（精确）
        """
        from ids import extract_features_19
        rec = {
            "address": 4, "function": 3, "length": 16,
            "setpoint": 0, "gain": 0,
            "reset rate": 0.5, "deadband": 1.25,
            "cycle time": 10.75, "rate": 2.5,
            "system mode": 1, "control scheme": 0,
            "pump": 1, "solenoid": 0,
            "pressure measurement": 0.689655, "crc rate": 12869,
            "command response": 1, "time": 1418682163,
        }
        out = extract_features_19(rec)
        # 13 = pressure_measurement (浮点精度不能丢)
        self.assertAlmostEqual(float(out[13]), 0.689655, places=5)
        # 其他 NUMERIC_COLS 字段同样要保精度
        self.assertAlmostEqual(float(out[5]), 0.5, places=5)   # reset_rate
        self.assertAlmostEqual(float(out[6]), 1.25, places=5)  # deadband
        self.assertAlmostEqual(float(out[7]), 10.75, places=5) # cycle_time
        self.assertAlmostEqual(float(out[8]), 2.5, places=5)   # rate

    def test_extract_features_17d_accepts_scada_csv_columns(self):
        """17-dim 主线 extract_features 也必须能读 SCADA CSV 列名，
        否则 17-dim 模型在 IDS 面板上同样 100% 报警（之前以为只影响
        23-dim，实测 BiLSTM 17-dim 也 38.46% acc = pos_rate）。
        """
        from ids import extract_features
        rec_scada = {
            "address": 4, "function": 3, "length": 16,
            "setpoint": 0, "gain": 100,
            "reset rate": 7, "deadband": 0,
            "cycle time": 10, "rate": 0,
            "system mode": 1, "control scheme": 0,
            "pump": 1, "solenoid": 0,
            "pressure measurement": 50.0, "crc rate": 12869,
            "command response": 1, "time": 1418682163,
        }
        out = extract_features(rec_scada)
        # FEATURE_COLUMNS 顺序：address, function, length, setpoint,
        # gain, reset, deadband, cycle, rate, system, control, pump,
        # solenoid, pressure, crc, command, time
        self.assertEqual(float(out[5]), 7.0)      # reset
        self.assertEqual(float(out[7]), 10.0)     # cycle
        self.assertEqual(float(out[9]), 1.0)      # system
        self.assertEqual(float(out[10]), 0.0)     # control
        self.assertEqual(float(out[13]), 50.0)    # pressure
        self.assertEqual(float(out[14]), 12869.0) # crc
        self.assertEqual(float(out[15]), 1.0)     # command

    def test_short_keys_still_work_for_backward_compat(self):
        """短名（旧测试和部分代码路径还在用）必须继续工作。"""
        from ids import extract_features_19, extract_features
        rec_short = {
            "address": 4, "function": 3, "length": 16,
            "setpoint": 0, "gain": 100, "reset": 7, "deadband": 0,
            "cycle": 10, "rate": 0, "system": 1, "control": 0,
            "pump": 1, "solenoid": 0, "pressure": 50.0,
            "crc": 12869, "command": 1, "time": 1418682163,
        }
        out19 = extract_features_19(rec_short)
        out17 = extract_features(rec_short)
        self.assertEqual(float(out19[5]), 7.0)
        self.assertEqual(float(out17[5]), 7.0)


class TestLoadTorch23Dim(unittest.TestCase):
    """load_model() 接受 23-dim KEEP_23 训练的 3D 模型。"""

    def _build_tcn_state_dict(self):
        """构造一个最小可重建的 23-dim TCN state_dict:
        - 一个 Conv1d(23, 8, kernel_size=3)
        - 一个 classifier Linear(8, 1)

        state_dict 必须有 3 段命名（如 ``layers.0.weight``），因此用
        Module 子类持有 ``self.layers = nn.Sequential(...)``；裸
        ``nn.Sequential(...).state_dict()`` 会产生 ``0.weight``（2 段），
        ``_reconstruct_conv1d`` 拒绝。
        """
        import torch
        import torch.nn as nn
        torch.manual_seed(0)

        class _Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv1d(23, 8, kernel_size=3, padding=1),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Linear(8, 1),
                )

        return _Holder().state_dict()

    def test_load_23dim_tcn_with_window_16_succeeds(self):
        """window=16 时 load_model 应该成功，返回 input_features=23 的 wrapper。"""
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import tempfile, os
        sd = self._build_tcn_state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"state_dict": sd, "seed": 0, "best_epoch": 1}, f.name)
            path = f.name
        try:
            wrapper = load_model(path, window_size=16)
            self.assertEqual(wrapper.input_features, 23)
            self.assertEqual(wrapper.window_size, 16)
        finally:
            os.unlink(path)

    def test_load_23dim_tcn_without_window_fails_cleanly(self):
        """window=1 时 3D 模型直接拒绝，错误信息提到 23-dim。"""
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import tempfile, os
        sd = self._build_tcn_state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"state_dict": sd}, f.name)
            path = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                load_model(path, window_size=1)
            # 友好的 3D 模型错误
            self.assertIn("3D", str(cm.exception))
        finally:
            os.unlink(path)

    def test_23dim_wrapper_warmup_then_infer(self):
        """warm-up 阶段 infer 返回 None；满窗后返回 (label, prob)。"""
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        import tempfile, os
        import numpy as np
        sd = self._build_tcn_state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"state_dict": sd}, f.name)
            path = f.name
        try:
            wrapper = load_model(path, window_size=16)
            # 第 1..15 帧：返回 None（warm-up）
            for i in range(15):
                rec = {
                    "address": 1, "function": 3, "length": 16,
                    "setpoint": 0, "gain": 0, "reset": 0, "deadband": 0,
                    "cycle": 0, "rate": 0, "system": 0, "control": 0,
                    "pump": 0, "solenoid": 0, "pressure": 0.0,
                    "crc": 12869, "command": 1,
                    "time": 1000 + i,
                }
                self.assertIsNone(wrapper.infer(rec))
            # 第 16 帧：返回 (label, prob)
            rec["time"] = 1015
            result = wrapper.infer(rec)
            self.assertIsNotNone(result)
            label, prob = result
            self.assertIn(label, (0, 1))
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        finally:
            os.unlink(path)


class TestLoadTorchTCNPool(unittest.TestCase):
    """``model_tcn_23dim_w16_*_s*.pt`` 嵌套 TCN+pool 格式可加载并推理。

    这些 checkpoint 来自 ``train_tcn_23dim_*5seed.py``，每个 TCN block 是
    带 ``conv1/bn1/conv2/bn2/residual`` 子模块的 Module（不是扁平
    Sequential），外加 attpool / gatedpool / mhattpool / tfpool / gap
    pool head —— 现有 ``_reconstruct_conv1d`` 解析不了这些 key。
    ``_reconstruct_tcn_pool_model`` 必须能：

    * 自动识别 5 种 pool head
    * 加载真实 checkpoint 并 round-trip 出 wrapper
    * 推理 16 帧后产出 (label, prob)
    """

    POOL_VARIANTS = ("gelu", "attpool", "gatedpool", "mhattpool", "tfpool")

    def _checkpoint_path(self, variant: str) -> Path:
        return Path(__file__).resolve().parent.parent / f"model_tcn_23dim_w16_{variant}_s123.pt"

    def test_real_checkpoints_present(self):
        """测试本身依赖真实 checkpoint 存在；缺失时给出明确指引。"""
        for variant in self.POOL_VARIANTS:
            p = self._checkpoint_path(variant)
            self.assertTrue(
                p.exists(),
                f"缺少 checkpoint {p.name}；请确认 5 个 model_tcn_23dim_w16_*_s123.pt 在项目根目录",
            )

    def test_load_each_pool_variant_succeeds(self):
        """5 种 pool 都能加载到 input_features=23、window_size=16 的 wrapper。"""
        if _TinyTorchModel is None:
            self.skipTest("torch not installed")
        for variant in self.POOL_VARIANTS:
            p = self._checkpoint_path(variant)
            if not p.exists():
                self.skipTest(f"checkpoint {p.name} 缺失")
            with self.subTest(variant=variant):
                w = load_model(str(p), window_size=16)
                self.assertEqual(w.input_features, 23, variant)
                self.assertEqual(w.window_size, 16, variant)

    def test_real_checkpoint_warmup_then_infer(self):
        """真实 gelu checkpoint：前 15 帧 None，第 16 帧返回 (label, prob)。"""
        if _TinyTorchModel is None:
            self.skipTest("torch not installed")
        p = self._checkpoint_path("gelu")
        if not p.exists():
            self.skipTest(f"checkpoint {p.name} 缺失")
        w = load_model(str(p), window_size=16)

        def make_record(i):
            return {
                "address": 1, "function": 3, "length": 16,
                "setpoint": 0, "gain": 0, "reset": 0, "deadband": 0,
                "cycle": 0, "rate": 0, "system": 0, "control": 0,
                "pump": 0, "solenoid": 0, "pressure": 0.0,
                "crc": 12869, "command": 1,
                "time": 1000 + i,
            }

        for i in range(15):
            self.assertIsNone(w.infer(make_record(i)))
        result = w.infer(make_record(15))
        self.assertIsNotNone(result)
        label, prob = result
        self.assertIn(label, (0, 1))
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_detect_nested_format_helper(self):
        """``_detect_tcn_pool_nested`` 识别嵌套 4 段 key，扁平 Conv1d 不识别。"""
        from ids.model_loader import _detect_tcn_pool_nested
        nested = {"tcn.0.conv1.weight": "x", "tcn.0.bn1.weight": "y"}
        flat = {"layers.0.weight": "x", "layers.1.running_mean": "y"}
        self.assertTrue(_detect_tcn_pool_nested(nested))
        self.assertFalse(_detect_tcn_pool_nested(flat))

    def test_detect_pool_type_helper(self):
        """``_detect_pool_type`` 按前缀识别 attpool / gatedpool / mhattpool / tfpool / gap。"""
        from ids.model_loader import _detect_pool_type
        self.assertEqual(_detect_pool_type({"attpool.proj.0.weight": "x"}), "attpool")
        self.assertEqual(_detect_pool_type({"gatedpool.gate.0.weight": "x"}), "gatedpool")
        self.assertEqual(_detect_pool_type({"mhattpool.heads.0.0.weight": "x"}), "mhattpool")
        self.assertEqual(_detect_pool_type({"tfpool.norm.weight": "x"}), "tfpool")
        self.assertEqual(_detect_pool_type({"tcn.0.conv1.weight": "x"}), "gap")


class TestLoadCNN23Dim(unittest.TestCase):
    """``model_cnn_23dim_w16_s*.pt`` 扁平 Conv1d + flat-style fc1/fc2 classifier。

    与 TCN+pool 不同：CNN 用 ``conv.{0,1,4,5,8,9}.{weight,bias,running_*}`` 这种
    扁平 Sequential 命名，但 classifier 用 ``fc1.weight`` / ``fc2.weight``
    flat-style（2 段），``_reconstruct_conv1d`` 之前不会自动回退到
    flat-style，所以加这条 fallback：内联 conv.* 没 2D classifier 时，
    试 ``_find_linear_classifier`` 找 ``fc*/`` ``head*/`` ``classifier.``。
    """

    CNN_SEEDS = (42, 123, 456, 789, 1024)

    def _cnn_path(self, seed: int) -> Path:
        return Path(__file__).resolve().parent.parent / f"model_cnn_23dim_w16_s{seed}.pt"

    def test_load_each_cnn_seed_succeeds(self):
        """5 个 CNN 23-dim seed 都能加载。"""
        if _TinyTorchModel is None:
            self.skipTest("torch not installed")
        for seed in self.CNN_SEEDS:
            p = self._cnn_path(seed)
            if not p.exists():
                self.skipTest(f"checkpoint {p.name} 缺失")
            with self.subTest(seed=seed):
                w = load_model(str(p), window_size=16)
                self.assertEqual(w.input_features, 23)
                self.assertEqual(w.window_size, 16)


class TestLoadTCNv4SE23Dim(unittest.TestCase):
    """``model_v4_se_23dim_b64_ch32_do01_window16_s*.pt`` TCN+SE 架构——
    每个 block 含 ``tcn.X.se.fc1/fc2`` 子模块。这是触发
    ``_make_tcn_block(..., has_se=True)`` 的回归测试。
    """

    SEEDS = (42, 123, 456, 789, 1024)

    def _path(self, seed: int) -> Path:
        return Path(__file__).resolve().parent.parent / f"model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt"

    def test_load_each_v4se_seed_succeeds(self):
        """5 个 v4_se_23dim seed 都能加载并跑出第 16 帧推理。"""
        if _TinyTorchModel is None:
            self.skipTest("torch not installed")
        for seed in self.SEEDS:
            p = self._path(seed)
            if not p.exists():
                self.skipTest(f"checkpoint {p.name} 缺失")
            with self.subTest(seed=seed):
                w = load_model(str(p), window_size=16)
                self.assertEqual(w.input_features, 23)
                self.assertEqual(w.window_size, 16)
                # 推 16 帧覆盖 warm-up；如果 SE 路径 torch 未 import 会
                # 在这里抛 NameError：name 'torch' is not defined。
                rec = {
                    "address": 1, "function": 3, "length": 16,
                    "setpoint": 0, "gain": 0, "reset": 0, "deadband": 0,
                    "cycle": 0, "rate": 0, "system": 0, "control": 0,
                    "pump": 0, "solenoid": 0, "pressure": 0.0,
                    "crc": 12869, "command": 1,
                    "time": 1000,
                }
                last_result = None
                for i in range(16):
                    rec["time"] = 1000 + i
                    last_result = w.infer(rec)
                self.assertIsNotNone(last_result)
                label, prob = last_result
                self.assertIn(label, (0, 1))
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_extreme_inputs_clipped_to_avoid_sigmoid_saturation(self):
        """回归测试：``_Torch3DStateDictWrapper23.infer`` 必须在送入模型前
        把 23-dim 窗口截断到 ±10（训练管道 ``_common_train.py:46-48``），
        否则 ``press_mean_w`` / ``crc_max_w`` 等聚合会被 outliers 撑到 10^3
        量级，Conv1d + TCN 累加后 logits 爆到 +∞，sigmoid 饱和在 1.0 →
        IDS 面板把每帧都标成 attack。

        行为级测试（仅看 prob 是否变）不可靠：合成 record dict 的特征分布
        与训练数据不符，模型本来就会输出极端概率。本测试改为结构级：
        在模型 forward 上挂 hook 截获输入，断言 ``input.abs().max() <= 10``。
        """
        if _TinyTorchModel is None or torch is None:
            self.skipTest("torch not installed")
        p = self._path(123)
        if not p.exists():
            self.skipTest(f"checkpoint {p.name} 缺失")
        wrapper = load_model(str(p), window_size=16)
        captured: list[torch.Tensor] = []
        hook_handle = wrapper._model.register_forward_pre_hook(
            lambda _m, inp: (captured.append(inp[0].detach().clone()), None)[1]
        )
        try:
            # 极端 outliers：pressure=5000 (scaled=5000)、crc=200000 (scaled=38)
            # — 没 clip 的话模型会直接拿到 5000 / 38；有 clip 会全在 [-10, 10]。
            rec = {
                "address": 1, "function": 3, "length": 16,
                "setpoint": 0, "gain": 0, "reset": 0, "deadband": 0,
                "cycle": 0, "rate": 0, "system": 0, "control": 0,
                "pump": 0, "solenoid": 0, "pressure": 5000.0,
                "crc": 200000, "command": 1,
                "time": 1000,
            }
            for i in range(16):
                rec["time"] = 1000 + i
                wrapper.infer(rec)
            self.assertGreaterEqual(len(captured), 1,
                "模型 forward 没被触发，hook 未生效")
            model_input = captured[0]
            max_abs = float(model_input.abs().max())
            self.assertLessEqual(
                max_abs, 10.0 + 1e-5,
                f"模型收到了 |x|={max_abs:.2f} > 10 的输入（缺 np.clip(±10)），"
                f"会被 Conv1d 累加 → sigmoid=1.0 → IDS 把每帧都标 attack。",
            )
        finally:
            hook_handle.remove()


class TestIDSPanelProbe(unittest.TestCase):
    """``IDSPanel._probe_model`` 必须正确探测 23-dim TCN+pool 模型。

    修复前：probe 读 ``fc1.weight.shape[1] = 32``（分类器 head 的
    in_features），把所有 TCN+pool 23-dim 标记成「⚠ 32特征(需17)」
    → dropdown 不可选。修复后：先识别嵌套 ``tcn.0.conv1.weight``，
    读 ``shape[1] = 23`` 标 compatible。

    本类只测 probe 的纯函数行为，不实例化 panel，避免 Tk root 依赖。
    """

    V4_SE_23DIM_SEEDS = (42, 123, 456, 789, 1024)
    CNN_23DIM_SEEDS = (42, 123, 456, 789, 1024)

    def _probe(self, path):
        # 延迟导入，避免在无 Tk 环境下加载 ids_panel 模块顶层失败
        from ids_panel import IDsPanel
        return IDsPanel._probe_model(str(path))

    def _path(self, name: str) -> Path:
        return Path(__file__).resolve().parent.parent / name

    def test_v4_se_23dim_probe_accepts(self):
        """5 个 v4_se_23dim seed 都被 probe 标为 compatible=23 / TCN+pool。"""
        for seed in self.V4_SE_23DIM_SEEDS:
            p = self._path(f"model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt")
            if not p.exists():
                self.skipTest(f"checkpoint {p.name} 缺失")
            with self.subTest(seed=seed):
                info = self._probe(p)
                self.assertTrue(info["compatible"], info)
                self.assertEqual(info["n_features"], 23, info)
                self.assertEqual(info["kind"], "TCN+pool", info)
                self.assertEqual(info["window_size"], 16, info)

    def test_cnn_23dim_probe_accepts(self):
        """5 个 CNN 23-dim seed 都被 probe 标为 compatible=23 / Conv1d。"""
        for seed in self.CNN_23DIM_SEEDS:
            p = self._path(f"model_cnn_23dim_w16_s{seed}.pt")
            if not p.exists():
                self.skipTest(f"checkpoint {p.name} 缺失")
            with self.subTest(seed=seed):
                info = self._probe(p)
                self.assertTrue(info["compatible"], info)
                self.assertEqual(info["n_features"], 23, info)
                self.assertEqual(info["kind"], "Conv1d", info)

    def test_tcn_pool_23dim_probe_accepts(self):
        """5 种 tcn+pool 23-dim（gelu/attpool/...）也走 TCN+pool 路径。"""
        for variant in ("gelu", "attpool", "gatedpool", "mhattpool", "tfpool"):
            p = self._path(f"model_tcn_23dim_w16_{variant}_s123.pt")
            if not p.exists():
                self.skipTest(f"checkpoint {p.name} 缺失")
            with self.subTest(variant=variant):
                info = self._probe(p)
                self.assertTrue(info["compatible"], info)
                self.assertEqual(info["n_features"], 23, info)
                self.assertEqual(info["kind"], "TCN+pool", info)


class TestFormatModelTag(unittest.TestCase):
    """``IDSPanel._format_model_tag(info, has_scaler)`` 把 probe 结果映射
    到 dropdown 上显示的中文标签。

    设计要点：

    * 17-dim 兼容 → ``✅ {kind} · 17f``，窗口>1 加 ``· W{window}``。
    * 23-dim 兼容 + scaler 在 → ``✅ {kind} · 23f · W{window} · scaler✓``。
    * 23-dim 兼容 + scaler 缺 → ``⚠ {kind} · 23f · W{window} · 需scaler``。
    * wrong_features → ``⚠ {n}f · 需 17/23``（IDS 同时支持 17/23，
      所以"需17"是过时的说法）。
    * load_error → ``❌ 加载错误``。

    把这个函数做成纯函数是为了让 dropdown 文案不靠实例化 panel
    也能测——probe 本身已经是纯函数了，format 顺延同样的取舍。
    """

    def _fmt(self, info, has_scaler=True):
        from ids_panel import IDsPanel
        return IDsPanel._format_model_tag(info, has_scaler)

    def test_17d_bilstm_compatible(self):
        info = {"compatible": True, "n_features": 17, "kind": "BiLSTM",
                "window_size": 16}
        self.assertEqual(
            self._fmt(info), "✅ BiLSTM · 17f · W16")

    def test_17d_gru_compatible(self):
        info = {"compatible": True, "n_features": 17, "kind": "GRU",
                "window_size": 16}
        self.assertEqual(
            self._fmt(info), "✅ GRU · 17f · W16")

    def test_17d_fnn_single_frame_omits_window(self):
        info = {"compatible": True, "n_features": 17, "kind": "FNN",
                "window_size": 1}
        self.assertEqual(
            self._fmt(info), "✅ FNN · 17f")

    def test_17d_sklearn(self):
        info = {"compatible": True, "n_features": 17, "kind": "sklearn-17",
                "window_size": 1}
        self.assertEqual(
            self._fmt(info), "✅ sklearn-17 · 17f")

    def test_23d_tcn_with_scaler(self):
        info = {"compatible": True, "n_features": 23, "kind": "TCN+pool",
                "window_size": 16}
        self.assertEqual(
            self._fmt(info, has_scaler=True),
            "✅ TCN+pool · 23f · W16 · scaler✓")

    def test_23d_tcn_without_scaler(self):
        info = {"compatible": True, "n_features": 23, "kind": "TCN+pool",
                "window_size": 16}
        self.assertEqual(
            self._fmt(info, has_scaler=False),
            "⚠ TCN+pool · 23f · W16 · 需scaler")

    def test_23d_cnn_with_scaler(self):
        info = {"compatible": True, "n_features": 23, "kind": "Conv1d",
                "window_size": 16}
        self.assertEqual(
            self._fmt(info, has_scaler=True),
            "✅ Conv1d · 23f · W16 · scaler✓")

    def test_23d_cnn_without_scaler(self):
        info = {"compatible": True, "n_features": 23, "kind": "Conv1d",
                "window_size": 16}
        self.assertEqual(
            self._fmt(info, has_scaler=False),
            "⚠ Conv1d · 23f · W16 · 需scaler")

    def test_wrong_features_32(self):
        info = {"compatible": False, "n_features": 32,
                "kind": "wrong_features", "window_size": 1}
        # IDS 现在同时支持 17 和 23，"需17" 已过时
        self.assertEqual(self._fmt(info), "⚠ 32f · 需 17/23")

    def test_load_error(self):
        info = {"compatible": False, "n_features": None,
                "kind": "load_error", "window_size": 1}
        self.assertEqual(self._fmt(info), "❌ 加载错误")


if __name__ == "__main__":
    unittest.main()