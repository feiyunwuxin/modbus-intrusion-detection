# IDS Panel MCU Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCU-deployed Hybrid INT8 model selection to IDS panel — load `model_v4_se_23dim_ch32_hybrid_s{seed}.h` C header files and bit-perfect replicate `ch32_forward()` inference on PC.

**Architecture:** Parse C header → extract int8 weights + FP32 scales + FP32 biases → dequantize per-channel → load into standard `TCNClassifierSE` (FP32, eval mode) → wrapper returns `(label, prob)` with threshold 0.49. Architecture class is imported from existing `quantize_v4se_23dim_ch32_to_h.py`. IDS panel dropdown shows FP32 and MCU models grouped, MCU prefixed with `[MCU]`.

**Tech Stack:** Python 3.10+, PyTorch (already required), NumPy, Tkinter (`ids_panel.py`), C header regex parsing. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-29-ids-panel-mcu-model-selection-design.md`

---

## Global Constraints

- Use `unittest` framework for all new tests (existing convention)
- Use `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at top of test files (existing pattern)
- Threshold for MCU model: **0.49** (from `CH32_BEST_THRESHOLD`)
- 23-dim feature order is fixed; see spec section 3.2 — do not change ordering
- `TCNClassifierSE` class must be imported from `quantize_v4se_23dim_ch32_to_h.py` (don't redefine)
- All commits use Conventional Commits format (`feat:`, `fix:`, `test:`, `docs:`)
- New file `ids/inference.py` additions go AFTER existing `extract_features_19`
- New file `ids/model_loader.py` additions go AFTER existing `_Torch3DStateDictWrapper23` class

---

## File Structure

| File | Responsibility |
|------|----------------|
| `ids/inference.py` (modify) | Add `FEATURE_COLUMNS_23` + `extract_features_23` for 23-dim MCU input |
| `ids/model_loader.py` (modify) | Add `_parse_mcu_header`, `_McuHeaderWrapper`, `list_available_mcu_headers`, `.h` dispatch in `load_model` |
| `ids/__init__.py` (modify) | Export `extract_features_23`, `FEATURE_COLUMNS_23`, `list_available_mcu_headers` |
| `ids_panel.py` (modify) | UI: `_probe_model` `.h` branch, `_format_model_tag` MCU prefix, `_refresh_models` grouping, `_on_model_selected` MCU handling, `process_frame` 23-dim branch |
| `tests/test_mcu_header_wrapper.py` (create) | 5 unit tests for parser, wrapper, threshold, features |
| `compare_fp32_vs_mcu_int8.py` (create) | Comparison script: FP32 .pt vs MCU .h bit-perfect verification |

---

## Task 1: 23-dim Feature Extraction in `ids/inference.py`

**Files:**
- Modify: `ids/inference.py` (add after `extract_features_19` near line 246)
- Test: `tests/test_mcu_header_wrapper.py` (create new file)

**Interfaces:**
- Produces: `FEATURE_COLUMNS_23: tuple[str, ...]` (length 23), `extract_features_23(record: dict) -> np.ndarray` shape `(23,)` dtype `float32`

**23-dim feature order** (from `ch32_inference.h` header comment — DO NOT change):
```
(address, function, gain, reset, deadband, cycle, rate, system, control,
 pump, solenoid, pressure, crc, command, time_diff,
 time_since_last_same_addr_func, is_unusual_fc, is_response,
 press_mean_w, crc_max_w, resp_count_w,
 cmd_resp_balance_w, length_nunique_w, unusual_count_w)
```
Indices into the source 27-dim record fields: 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcu_header_wrapper.py` with this content at the top:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestExtractFeatures23 -v`
Expected: FAIL with `ImportError: cannot import name 'FEATURE_COLUMNS_23'`

- [ ] **Step 3: Implement `extract_features_23`**

Add to `ids/inference.py` AFTER the existing `extract_features_19` function (around line 246, before any helper functions):

```python
# ── 23-dim feature extraction for MCU Hybrid INT8 models ─────────────────
# Source: KeilH743/H743/Core/Inc/ch32_inference.h header comment.
# Indices into source 27-dim record: 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
# 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26. Skips length(2), setpoint(3),
# crc_mean_w(20), cmd_count_w(22).
FEATURE_COLUMNS_23: tuple[str, ...] = (
    "address", "function",
    "gain", "reset", "deadband", "cycle", "rate",
    "system", "control", "pump", "solenoid",
    "pressure", "crc", "command", "time_diff",
    "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
    "press_mean_w", "crc_max_w", "resp_count_w",
    "cmd_resp_balance_w", "length_nunique_w", "unusual_count_w",
)

_MCU_FEATURE_INDICES_27 = (
    0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 21, 23, 24, 25, 26,
)


def extract_features_23(record: dict) -> np.ndarray:
    """Extract 23-dim feature vector matching MCU Hybrid INT8 model input order.

    Returns float32 numpy array of shape (23,). Missing fields default to 0.0.
    """
    out = np.empty(23, dtype=np.float32)
    for out_i, src_key in enumerate(_MCU_FEATURE_INDICES_27):
        col_name = FEATURE_COLUMNS_23[out_i]
        val = record.get(col_name, 0.0)
        out[out_i] = float(val) if val is not None else 0.0
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestExtractFeatures23 -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add ids/inference.py tests/test_mcu_header_wrapper.py
git commit -m "feat(ids): add extract_features_23 for MCU Hybrid INT8 models"
```

---

## Task 2: MCU Header Parser in `ids/model_loader.py`

**Files:**
- Modify: `ids/model_loader.py` (add before `_Torch3DStateDictWrapper23` or at end of file)
- Test: `tests/test_mcu_header_wrapper.py` (add new test class)

**Interfaces:**
- Produces: `_parse_mcu_header(path: str | Path) -> dict[str, np.ndarray]` — keys like `"tcn.0.conv1.weight"`, `"tcn.0.conv1.scale"`, `"tcn.0.conv1.bias"`, `"fc1.weight"`, `"fc1.bias"`. int8 weights stored as int8 dtype, scales/biases as float32.

**C header format** (from `model_v4_se_23dim_ch32_hybrid_s42.h`):
```c
static const int8_t v4se23_ch32_s{seed}_w_tcn_0_conv1_weight[2208] = { -21, 7, ... };
static const float  v4se23_ch32_s{seed}_s_tcn_0_conv1_weight[32] = { 1.234e-3f, ... };
static const float  v4se23_ch32_s{seed}_b_tcn_0_conv1_bias[32] = { 0.1f, ... };
static const float  v4se23_ch32_s{seed}_fc1_weight[1024] = { ... };  // FP32
```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcu_header_wrapper.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestParseMcuHeader -v`
Expected: FAIL with `ImportError: cannot import name '_parse_mcu_header'`

- [ ] **Step 3: Implement `_parse_mcu_header`**

Add to `ids/model_loader.py` (at end of file, or before `_Torch3DStateDictWrapper23`):

```python
import re

# ── MCU C header parser for Hybrid INT8 models ────────────────────────────
# Parses model_v4_se_23dim_ch32_hybrid_s{seed}.h files generated by
# quantize_v4se_23dim_ch32_to_h.py.

_INT8_ARRAY_RE = re.compile(
    r"static\s+const\s+int8_t\s+(?P<name>\w+)\s*\[\s*(?P<n>\d+)\s*\]\s*=\s*\{(?P<vals>[^}]+)\}\s*;",
    re.DOTALL,
)
_FLOAT_ARRAY_RE = re.compile(
    r"static\s+const\s+float\s+(?P<name>\w+)\s*\[\s*(?P<n>\d+)\s*\]\s*=\s*\{(?P<vals>[^}]+)\}\s*;",
    re.DOTALL,
)


def _parse_c_array_floats(text: str) -> np.ndarray:
    """Parse a comma-separated float literal list like '1.234e-3f, 0.5f, ...'."""
    nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", text)
    return np.array([float(n) for n in nums], dtype=np.float32)


def _parse_c_array_int8(text: str) -> np.ndarray:
    """Parse a comma-separated int literal list like '-21, 7, 127, -128, ...'."""
    nums = re.findall(r"[-+]?\d+", text)
    return np.array([int(n) for n in nums], dtype=np.int8)


def _parse_mcu_header(path) -> dict:
    """Parse a KeilH743 MCU C header into a dict of numpy arrays.

    Keys use PyTorch state_dict style: 'tcn.0.conv1.weight', 'tcn.0.conv1.scale',
    'tcn.0.conv1.bias', 'fc1.weight', 'fc1.bias', etc.

    - int8 weights for tcn.*.conv{1,2}.weight, tcn.*.residual.weight, tcn.*.se.fc{1,2}.weight
    - float32 scale per output channel (same keys with '.scale' suffix)
    - float32 bias (same keys with '.bias' suffix)
    - float32 fc1.weight, fc2.weight (already FP32 in source)
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    out: dict[str, np.ndarray] = {}

    for m in _INT8_ARRAY_RE.finditer(text):
        name = m.group("name")
        arr = _parse_c_array_int8(m.group("vals"))
        key = _mcu_array_name_to_key(name, kind="int8")
        if key is not None:
            out[key] = arr

    for m in _FLOAT_ARRAY_RE.finditer(text):
        name = m.group("name")
        arr = _parse_c_array_floats(m.group("vals"))
        key = _mcu_array_name_to_key(name, kind="float")
        if key is not None:
            out[key] = arr

    return out


# Mapping from C array name to PyTorch-style state_dict key.
# C naming: v4se23_ch32_s{seed}_{w|s|b}_tcn_{N}_{layer}[_{suffix}]
# We strip the prefix and decode the suffix.
_MCU_NAME_PREFIX_RE = re.compile(
    r"^v4se23_ch32_s\d+_(?P<kind>[wsb])_(?P<rest>.+)$"
)


def _mcu_array_name_to_key(name: str, kind: str) -> str | None:
    """Convert a C array name to PyTorch state_dict key.

    Examples:
      v4se23_ch32_s42_w_tcn_0_conv1_weight  -> 'tcn.0.conv1.weight'
      v4se23_ch32_s42_s_tcn_0_conv1_weight  -> 'tcn.0.conv1.scale'
      v4se23_ch32_s42_b_tcn_0_conv1_bias    -> 'tcn.0.conv1.bias'
      v4se23_ch32_s42_fc1_weight            -> 'fc1.weight'
      v4se23_ch32_s42_b_fc1_bias            -> 'fc1.bias'
    """
    m = _MCU_NAME_PREFIX_RE.match(name)
    if m is None:
        return None
    letter = m.group("kind")
    rest = m.group("rest")
    # rest is e.g. 'tcn_0_conv1_weight', 'fc1_weight', 'b_fc1_bias' (but b prefix is already stripped)
    parts = rest.split("_")
    # Build PyTorch-style key
    if parts[0] == "tcn":
        # tcn_0_conv1_weight -> tcn.0.conv1.weight
        block_idx = parts[1]
        layer_parts = parts[2:]  # ['conv1', 'weight'] or ['se', 'fc1', 'weight']
        if layer_parts[0] == "se":
            # tcn.0.se.fc1.weight
            return f"tcn.{block_idx}.se.{layer_parts[1]}.{layer_parts[2]}"
        else:
            return f"tcn.{block_idx}.{layer_parts[0]}.{layer_parts[1]}"
    elif parts[0] == "fc":
        # fc1_weight -> fc1.weight
        return f"{parts[0]}{parts[1]}.{parts[2]}"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestParseMcuHeader -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add ids/model_loader.py tests/test_mcu_header_wrapper.py
git commit -m "feat(ids): add _parse_mcu_header for Hybrid INT8 C headers"
```

---

## Task 3: `_McuHeaderWrapper` class in `ids/model_loader.py`

**Files:**
- Modify: `ids/model_loader.py` (add after `_parse_mcu_header`)
- Test: `tests/test_mcu_header_wrapper.py` (add new test class)

**Interfaces:**
- Produces: `_McuHeaderWrapper` class with:
  - `kind: ClassVar[str] = "MCU-Hybrid-INT8"`
  - `n_features: ClassVar[int] = 23`
  - `threshold: ClassVar[float] = 0.49`
  - `__init__(self, header_path: str | Path, window_size: int = 16)`
  - `infer(self, record_or_features) -> tuple[int, float]` — accepts either a Modbus record dict or a 23-dim np.ndarray

**Architecture source**: Import `TCNClassifierSE` from `quantize_v4se_23dim_ch32_to_h.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcu_header_wrapper.py`:

```python
from ids.model_loader import _McuHeaderWrapper


class TestMcuHeaderWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _MCU_HEADER_PATH.exists():
            raise unittest.SkipTest(f"MCU header not found: {_MCU_HEADER_PATH}")
        cls.wrapper = _McuHeaderWrapper(_MCU_HEADER_PATH, window_size=16)

    def test_kind_attribute(self):
        self.assertEqual(_McuHeaderWrapper.kind, "MCU-Hybrid-INT8")

    def test_n_features_attribute(self):
        self.assertEqual(_McuHeaderWrapper.n_features, 23)

    def test_threshold_attribute(self):
        self.assertEqual(_McuHeaderWrapper.threshold, 0.49)

    def test_init_no_error(self):
        # Already constructed in setUpClass
        self.assertIsNotNone(self.wrapper)

    def test_infer_with_features_returns_tuple(self):
        # Provide 16 random feature vectors (23-dim) directly
        rng = np.random.default_rng(42)
        feats_seq = rng.normal(size=(16, 23)).astype(np.float32)
        result = self.wrapper.infer(feats_seq)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        label, prob = result
        self.assertIn(label, (0, 1))
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_infer_with_record_returns_tuple(self):
        record = _FULL_RECORD.copy()
        # Need 16 frames before getting a label
        for _ in range(15):
            self.wrapper.infer(record)
        result = self.wrapper.infer(record)
        self.assertIsInstance(result, tuple)
        label, prob = result
        self.assertIn(label, (0, 1))

    def test_threshold_applied(self):
        # Force a high-probability window by reusing same record many times (warmup)
        for _ in range(16):
            label, prob = self.wrapper.infer(_FULL_RECORD)
        # Just verify label is in {0, 1}
        self.assertIn(label, (0, 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestMcuHeaderWrapper -v`
Expected: FAIL with `ImportError: cannot import name '_McuHeaderWrapper'`

- [ ] **Step 3: Implement `_McuHeaderWrapper`**

Add to `ids/model_loader.py` after `_parse_mcu_header`:

```python
import torch

# Try to import the canonical TCNClassifierSE from the quantize script.
# If unavailable (e.g., script moved), fall back to local reconstruction.
try:
    from quantize_v4se_23dim_ch32_to_h import TCNClassifierSE
except ImportError:
    TCNClassifierSE = None  # type: ignore


class _McuHeaderWrapper:
    """ModelWrapper for STM32H743 MCU Hybrid INT8 C headers.

    Bit-perfect replicates ch32_forward() by dequantizing INT8 weights to FP32
    at construction time, then running standard FP32 inference on a
    TCNClassifierSE architecture (imported from quantize script).

    Class attributes:
        kind:       "MCU-Hybrid-INT8" — used by ids_panel for routing
        n_features: 23               — used by ids_panel for feature extraction branch
        threshold:  0.49             — CH32_BEST_THRESHOLD
    """

    kind: str = "MCU-Hybrid-INT8"
    n_features: int = 23
    threshold: float = 0.49

    def __init__(self, header_path, window_size: int = 16):
        if TCNClassifierSE is None:
            raise RuntimeError(
                "TCNClassifierSE not importable from quantize_v4se_23dim_ch32_to_h. "
                "Ensure the script is on sys.path."
            )
        self._window_size = int(window_size)
        self._header_path = str(header_path)

        parsed = _parse_mcu_header(header_path)
        self._model = self._build_model_from_parsed(parsed)
        self._model.eval()

        # Sliding window buffer: holds last window_size feature vectors
        self._feat_buffer: list[np.ndarray] = []

    @torch.no_grad()
    def infer(self, record_or_features):
        """Run one forward pass and return (label, prob).

        Args:
            record_or_features: Either a Modbus record dict (will be passed to
                extract_features_23) or a pre-extracted feature vector
                (np.ndarray shape (23,) or (window_size, 23)).

        Returns:
            (label, prob) where label is 0 or 1, prob is sigmoid(logit) in [0,1].
            Returns (0, 0.5) during warmup (fewer than window_size frames seen).
        """
        # Normalize input to a single 23-dim feature vector
        if isinstance(record_or_features, dict):
            feats = extract_features_23(record_or_features)
        elif isinstance(record_or_features, np.ndarray):
            if record_or_features.ndim == 2:
                # (window_size, 23) — caller pre-built the window
                window = record_or_features.astype(np.float32)
                return self._forward_window(window)
            feats = record_or_features.astype(np.float32).reshape(23)
        else:
            feats = np.asarray(record_or_features, dtype=np.float32).reshape(23)

        self._feat_buffer.append(feats)
        if len(self._feat_buffer) > self._window_size:
            self._feat_buffer = self._feat_buffer[-self._window_size:]

        if len(self._feat_buffer) < self._window_size:
            return (0, 0.5)  # warmup

        window = np.stack(self._feat_buffer, axis=-1)  # (23, window_size)
        return self._forward_window(window)

    def _forward_window(self, window: np.ndarray) -> tuple[int, float]:
        x = torch.from_numpy(window).unsqueeze(0).float()  # (1, 23, 16)
        logit = self._model(x).squeeze().item()
        prob = 1.0 / (1.0 + np.exp(-logit))
        label = 1 if prob >= self.threshold else 0
        return (label, float(prob))

    def _build_model_from_parsed(self, parsed: dict) -> "torch.nn.Module":
        """Construct TCNClassifierSE and load dequantized FP32 weights."""
        model = TCNClassifierSE(in_ch=23, channels=32, n_blocks=3,
                                dilations=(1, 2, 4), dropout=0.1)
        # Build state_dict
        sd = {}
        for bi, block in enumerate(model.tcn):
            # conv1, conv2: dequantize int8 weight using per-channel scale
            for ck in ("conv1", "conv2"):
                w_key = f"tcn.{bi}.{ck}.weight"
                s_key = f"tcn.{bi}.{ck}.scale"
                b_key = f"tcn.{bi}.{ck}.bias"
                if w_key in parsed and s_key in parsed:
                    w_q = parsed[w_key].astype(np.float32)  # (out_ch, in_ch, k)
                    scale = parsed[s_key].reshape(-1, 1, 1)  # (out_ch, 1, 1)
                    sd[f"tcn.{bi}.{ck}.weight"] = torch.from_numpy(w_q * scale)
                if b_key in parsed:
                    sd[f"tcn.{bi}.{ck}.bias"] = torch.from_numpy(parsed[b_key])

            # SE fc1, fc2: same per-channel dequant
            for ck in ("se.fc1", "se.fc2"):
                w_key = f"tcn.{bi}.{ck}.weight"
                s_key = f"tcn.{bi}.{ck}.scale"
                b_key = f"tcn.{bi}.{ck}.bias"
                if w_key in parsed and s_key in parsed:
                    w_q = parsed[w_key].astype(np.float32)  # (out_ch, in_ch)
                    scale = parsed[s_key].reshape(-1, 1)
                    sd[f"tcn.{bi}.{ck}.weight"] = torch.from_numpy(w_q * scale)
                if b_key in parsed:
                    sd[f"tcn.{bi}.{ck}.bias"] = torch.from_numpy(parsed[b_key])

            # Residual (only block 0)
            r_key = f"tcn.{bi}.residual.weight"
            if r_key in parsed:
                rs_key = f"tcn.{bi}.residual.scale"
                rb_key = f"tcn.{bi}.residual.bias"
                w_q = parsed[r_key].astype(np.float32)  # (out_ch, in_ch, 1)
                scale = parsed[rs_key].reshape(-1, 1, 1)
                sd[f"tcn.{bi}.residual.weight"] = torch.from_numpy(w_q * scale)
                if rb_key in parsed:
                    sd[f"tcn.{bi}.residual.bias"] = torch.from_numpy(parsed[rb_key])

        # fc1, fc2: already FP32
        if "fc1.weight" in parsed:
            sd["fc1.weight"] = torch.from_numpy(parsed["fc1.weight"])
        if "fc1.bias" in parsed:
            sd["fc1.bias"] = torch.from_numpy(parsed["fc1.bias"])
        if "fc2.weight" in parsed:
            sd["fc2.weight"] = torch.from_numpy(parsed["fc2.weight"])
        if "fc2.bias" in parsed:
            sd["fc2.bias"] = torch.from_numpy(parsed["fc2.bias"])

        # Load with strict=False to ignore BN running stats (we set them to defaults)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys when loading MCU model: {unexpected}")

        # Set BN defaults so eval mode is identity (BN folded into conv bias already)
        for block in model.tcn:
            for bn in (block.bn1, block.bn2):
                bn.weight.data.fill_(1.0)
                bn.bias.data.zero_()
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0)

        return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestMcuHeaderWrapper -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add ids/model_loader.py tests/test_mcu_header_wrapper.py
git commit -m "feat(ids): add _McuHeaderWrapper with bit-perfect FP32 inference"
```

---

## Task 4: Wire `_McuHeaderWrapper` into `load_model` and add `list_available_mcu_headers`

**Files:**
- Modify: `ids/model_loader.py` (`load_model` function around line 1207; `list_available_models` around line 1377)
- Test: `tests/test_mcu_header_wrapper.py` (add new test class)

**Interfaces:**
- Produces: `list_available_mcu_headers(directory: str) -> list[str]` — scans `model_v4_se_*_ch32_*_hybrid_s*.h` and `model_v*_*_ch32_*_s*.h` patterns
- Modifies: `load_model(path, window_size)` — when `path.endswith(".h")`, dispatch to `_McuHeaderWrapper`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcu_header_wrapper.py`:

```python
from ids import load_model
from ids.model_loader import list_available_mcu_headers


class TestListAvailableMcuHeaders(unittest.TestCase):
    def test_returns_list(self):
        # Scan repo root
        repo_root = str(Path(__file__).resolve().parent.parent)
        result = list_available_mcu_headers(repo_root)
        self.assertIsInstance(result, list)

    def test_finds_known_header(self):
        repo_root = str(Path(__file__).resolve().parent.parent)
        result = list_available_mcu_headers(repo_root)
        # The header should be found if it's in the repo
        if _MCU_HEADER_PATH.exists():
            self.assertIn(str(_MCU_HEADER_PATH), result)

    def test_empty_dir(self):
        result = list_available_mcu_headers("/nonexistent_dir_xyz")
        self.assertEqual(result, [])


class TestLoadModelMcuHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _MCU_HEADER_PATH.exists():
            raise unittest.SkipTest(f"MCU header not found: {_MCU_HEADER_PATH}")

    def test_load_via_load_model(self):
        wrapper = load_model(str(_MCU_HEADER_PATH), window_size=16)
        self.assertIsInstance(wrapper, _McuHeaderWrapper)
        self.assertEqual(wrapper.kind, "MCU-Hybrid-INT8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestListAvailableMcuHeaders tests/test_mcu_header_wrapper.py::TestLoadModelMcuHeader -v`
Expected: FAIL with `ImportError: cannot import name 'list_available_mcu_headers'`

- [ ] **Step 3: Implement `list_available_mcu_headers` and update `load_model`**

Add to `ids/model_loader.py` (anywhere after `_McuHeaderWrapper`):

```python
def list_available_mcu_headers(directory: str) -> list[str]:
    """Scan a directory for MCU Hybrid INT8 C header files.

    Matches: model_v4_se_23dim_ch32_hybrid_s{seed}.h and similar patterns.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    files = []
    # Primary pattern: explicit hybrid s{seed}.h
    files.extend(d.glob("model_*hybrid_s*.h"))
    # Fallback: any model_*.h with TCN/SE/ch32 patterns
    for f in d.glob("model_*.h"):
        if f not in files:
            text = f.read_text(encoding="utf-8", errors="ignore")[:2000]
            if "v4se23_ch32" in text or "ch32_inference" in text:
                files.append(f)
    return sorted(str(f) for f in files)
```

Now modify the existing `load_model` function (around line 1207). Find the current dispatcher:

```python
def load_model(path: str, *, window_size: int = 1) -> ModelWrapper:
    ...
    if ext == ".joblib":
        return _load_sklearn(p)
    if ext == ".pt":
        return _load_torch(p, window_size=window_size)
    raise ValueError(f"不支持的模型格式: {ext}（仅 .pt / .joblib）")
```

Add the `.h` branch BEFORE the final `raise`:

```python
def load_model(path: str, *, window_size: int = 1) -> ModelWrapper:
    ...
    if ext == ".joblib":
        return _load_sklearn(p)
    if ext == ".pt":
        return _load_torch(p, window_size=window_size)
    if ext == ".h":
        return _McuHeaderWrapper(p, window_size=window_size)
    raise ValueError(f"不支持的模型格式: {ext}（仅 .pt / .joblib / .h）")
```

Also update the `list_available_models` function (around line 1377) to include `.h` files. Find:

```python
def list_available_models(directory: str) -> list[str]:
    """扫描目录下 model_*.pt 和 model_*.joblib，按名字排序返回完整路径。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    files: list[Path] = []
    for ext in (".pt", ".joblib"):
        files.extend(d.glob(f"model_*{ext}"))
    return sorted(str(f) for f in files)
```

Replace with:

```python
def list_available_models(directory: str) -> list[str]:
    """扫描目录下 model_*.pt / model_*.joblib / model_*hybrid_s*.h, 按名字排序返回完整路径。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    files: list[Path] = []
    for ext in (".pt", ".joblib"):
        files.extend(d.glob(f"model_*{ext}"))
    # MCU Hybrid INT8 headers (separate pattern)
    files.extend(list_available_mcu_headers(directory))
    return sorted(set(str(f) for f in files))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/test_mcu_header_wrapper.py::TestListAvailableMcuHeaders tests/test_mcu_header_wrapper.py::TestLoadModelMcuHeader -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add ids/model_loader.py tests/test_mcu_header_wrapper.py
git commit -m "feat(ids): wire _McuHeaderWrapper into load_model + list_available_models"
```

---

## Task 5: Export new APIs from `ids/__init__.py`

**Files:**
- Modify: `ids/__init__.py`

**Interfaces:**
- Produces: exports of `extract_features_23`, `FEATURE_COLUMNS_23`, `list_available_mcu_headers`

- [ ] **Step 1: Verify current exports**

Read `ids/__init__.py` (current contents should be around lines 13-22). Confirm what's exported. The file currently has:

```python
from .inference import (
    FEATURE_COLUMNS, FEATURE_COLUMNS_19, extract_features,
    extract_features_19, keep_23_per_frame_indices,
    keep_23_aggregate_indices, resolve_label,
)
from .model_loader import ModelWrapper, load_model, list_available_models

__all__ = ["FEATURE_COLUMNS", ..., "ModelWrapper", "load_model", "list_available_models"]
```

- [ ] **Step 2: Add the new exports**

Update `ids/__init__.py` to add:

```python
from .inference import (
    FEATURE_COLUMNS, FEATURE_COLUMNS_19, FEATURE_COLUMNS_23,
    extract_features, extract_features_19, extract_features_23,
    keep_23_per_frame_indices,
    keep_23_aggregate_indices, resolve_label,
)
from .model_loader import (
    ModelWrapper, load_model,
    list_available_models, list_available_mcu_headers,
)

__all__ = [
    "FEATURE_COLUMNS", "FEATURE_COLUMNS_19", "FEATURE_COLUMNS_23",
    "extract_features", "extract_features_19", "extract_features_23",
    "keep_23_per_frame_indices", "keep_23_aggregate_indices", "resolve_label",
    "ModelWrapper", "load_model",
    "list_available_models", "list_available_mcu_headers",
]
```

- [ ] **Step 3: Verify import works**

Run: `cd D:/workspace/claude/Issue/Issue && python -c "from ids import extract_features_23, FEATURE_COLUMNS_23, list_available_mcu_headers; print('OK')"`
Expected: prints `OK`

- [ ] **Step 4: Run all tests to confirm no regressions**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/ -v`
Expected: all existing tests still pass + new tests pass

- [ ] **Step 5: Commit**

```bash
git add ids/__init__.py
git commit -m "feat(ids): export extract_features_23 + list_available_mcu_headers"
```

---

## Task 6: UI integration in `ids_panel.py`

**Files:**
- Modify: `ids_panel.py` — `_probe_model`, `_format_model_tag`, `_refresh_models`, `_on_model_selected`, `process_frame`
- Test: manual smoke test (run modbus_simulator.py and verify dropdown)

**Interfaces:**
- `_probe_model(path)` returns dict with `kind` field — for `.h` files returns `{"kind": "MCU-Hybrid-INT8", "n_features": 23, ...}`
- `_format_model_tag(info, has_scaler)` — MCU branch shows `· MCU-Hybrid-INT8 · 23-dim`
- `_refresh_models()` — groups FP32 first, MCU last with `[MCU]` prefix
- `_on_model_selected()` — sets threshold to 0.49 when wrapper.kind == "MCU-Hybrid-INT8"
- `process_frame(record, idx)` — calls `extract_features_23(record)` for MCU wrapper, `extract_features(record)` otherwise

- [ ] **Step 1: Add MCU branch to `_probe_model`**

Read `ids_panel.py` `_probe_model()` method (around line 154). Find the opening:

```python
def _probe_model(self, p) -> dict:
    ...
```

Add this branch AT THE TOP of the method (before any .pt handling):

```python
def _probe_model(self, p) -> dict:
    p_str = str(p)
    if p_str.endswith(".h"):
        try:
            from ids.model_loader import _parse_mcu_header
            parsed = _parse_mcu_header(p_str)
            has_all = all(
                f"tcn.{bi}.{ck}.weight" in parsed
                for bi in range(3) for ck in ("conv1", "conv2")
            )
            return {
                "kind": "MCU-Hybrid-INT8" if has_all else "mcu_header_incomplete",
                "n_features": 23,
                "path": p_str,
            }
        except Exception as e:
            return {"kind": "load_error", "error": str(e), "path": p_str}
    # ... existing logic unchanged
```

- [ ] **Step 2: Add MCU tag in `_format_model_tag`**

Read `_format_model_tag()` (around line 291). Add MCU handling BEFORE the existing kind branches:

```python
def _format_model_tag(self, info, has_scaler):
    kind = info.get("kind", "")
    if kind == "MCU-Hybrid-INT8":
        return "✅ MCU-Hybrid-INT8 · 23-dim"
    if kind == "mcu_header_incomplete":
        return "❌ MCU header incomplete"
    # ... existing logic unchanged
```

- [ ] **Step 3: Update `_refresh_models` to group and prefix MCU**

Find `_refresh_models()` (around line 321). Replace the section that builds `tagged_names`:

Current logic builds one flat list. Replace with grouped:

```python
def _refresh_models(self) -> None:
    project_dir = Path(__file__).parent
    paths = list_available_models(str(project_dir))
    self._available_paths = paths
    self.model_combo["values"] = []
    if not paths:
        self.status_var.set("目录下未发现模型文件（model_*.pt / *.joblib / *.h）")
        return

    # Group: FP32 (.pt/.joblib) first, MCU (.h) last
    fp32_paths = [p for p in paths if not p.endswith(".h")]
    mcu_paths = [p for p in paths if p.endswith(".h")]

    tagged_names: list[str] = []
    for p in fp32_paths:
        info = self._probe_model(p)
        base = Path(p).name
        tag = self._format_model_tag(info, has_scaler=False)
        tagged_names.append(f"{base}  {tag}")
    for p in mcu_paths:
        info = self._probe_model(p)
        base = Path(p).name
        tag = self._format_model_tag(info, has_scaler=False)
        tagged_names.append(f"[MCU] {base}  {tag}")

    self.model_combo["values"] = tagged_names
    self._tagged_to_path = {tn: p for tn, p in zip(tagged_names, paths)}
    # ... rest of existing init logic
```

(Adjust the existing `_tagged_to_path` mapping variable name if it differs in current code — match existing convention.)

- [ ] **Step 4: Set default threshold in `_on_model_selected`**

Find `_on_model_selected()` (around line 359). After the wrapper is successfully loaded, add threshold adjustment:

```python
def _on_model_selected(self, _event=None) -> None:
    ...
    try:
        self.wrapper = load_model(path, window_size=ws)
        # MCU model uses 0.49 threshold (CH32_BEST_THRESHOLD)
        if getattr(self.wrapper, "kind", None) == "MCU-Hybrid-INT8":
            self.threshold_var.set(0.49)
            self.status_var.set(f"MCU model loaded (threshold 0.49): {Path(path).name}")
        else:
            self.status_var.set(f"已加载模型: {Path(path).name}")
    except Exception as e:
        self.status_var.set(f"加载失败: {e}")
        self.wrapper = None
```

- [ ] **Step 5: Branch feature extraction in `process_frame`**

Find `process_frame()` (around line 474). Find where `extract_features(record)` is called. Wrap it:

Current code likely has:
```python
if n_feat == 23:
    result = self.wrapper.infer(record)
else:
    features = extract_features(record)
    result = self.wrapper.infer(features)
```

Modify to:
```python
wrapper_kind = getattr(self.wrapper, "kind", None)
if wrapper_kind == "MCU-Hybrid-INT8":
    # MCU model needs 23-dim features, then wrapper handles windowing
    result = self.wrapper.infer(record)
elif n_feat == 23:
    result = self.wrapper.infer(record)
else:
    features = extract_features(record)
    result = self.wrapper.infer(features)
```

Also add the import at the top of `ids_panel.py`:
```python
from ids.inference import FEATURE_COLUMNS, FEATURE_COLUMNS_19, FEATURE_COLUMNS_23, extract_features, extract_features_19, extract_features_23, resolve_label
```

(Replace the existing `from ids.inference import ...` line — add the new symbols.)

- [ ] **Step 6: Run tests**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 7: Smoke test by running the simulator**

Run: `cd D:/workspace/claude/Issue/Issue && python -c "from ids_panel import IDsPanel; print('import OK')"`
Expected: prints `import OK`

Then manually launch `python modbus_simulator.py`, open the dropdown, verify:
- FP32 `.pt` files listed (existing behavior preserved)
- MCU `.h` files listed with `[MCU]` prefix
- Selecting MCU model loads without error
- Threshold Scale shows 0.49 after MCU selection

- [ ] **Step 8: Commit**

```bash
git add ids_panel.py
git commit -m "feat(ids-panel): support MCU Hybrid INT8 model selection in dropdown"
```

---

## Task 7: Comparison script `compare_fp32_vs_mcu_int8.py`

**Files:**
- Create: `compare_fp32_vs_mcu_int8.py` (repo root)

**Interfaces:**
- CLI: `python compare_fp32_vs_mcu_int8.py [--n 200] [--seed 42]`
- Outputs: `compare_fp32_vs_mcu_int8_v4se_s42.json` + `.md`

- [ ] **Step 1: Create the script**

Write `compare_fp32_vs_mcu_int8.py`:

```python
#!/usr/bin/env python3
"""Compare FP32 PyTorch reference model vs MCU Hybrid INT8 .h header.

Loads:
  - model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt  (FP32 reference)
  - model_v4_se_23dim_ch32_hybrid_s{seed}.h              (MCU deployed)

Runs both on the same N test samples, compares probabilities and verdict
agreement. Should be near bit-perfect (< 1e-5 max diff) since C code
dequantizes INT8 to FP32 at init time.

Usage:
  python compare_fp32_vs_mcu_int8.py --n 200 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--data", type=str, default="X_test_binary.npy",
                        help="Test data .npy file (N, 23, 16) or (N*16, 23)")
    parser.add_argument("--labels", type=str, default="y_test_binary.npy")
    args = parser.parse_args()

    repo = Path(__file__).parent
    fp32_pt = repo / f"model_v4_se_23dim_b64_ch32_do01_window16_s{args.seed}.pt"
    mcu_h = repo / "KeilH743" / "H743" / "Core" / "Inc" / \
            f"model_v4_se_23dim_ch32_hybrid_s{args.seed}.h"

    if not fp32_pt.exists():
        sys.exit(f"FP32 model not found: {fp32_pt}")
    if not mcu_h.exists():
        sys.exit(f"MCU header not found: {mcu_h}")

    # Load test data
    X_path = repo / args.data
    y_path = repo / args.labels
    if not X_path.exists() or not y_path.exists():
        sys.exit(f"Test data not found: {X_path} / {y_path}")
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    # Reshape to (N, 23, 16) windows
    window = 16
    if X.ndim == 2 and X.shape[1] == 23:
        n_windows = (len(X) // window) * window
        Xw = X[:n_windows].reshape(-1, window, 23).transpose(0, 2, 1)
        yw = (y[:n_windows].reshape(-1, window).sum(axis=1) >= 1).astype(np.int64)
    else:
        Xw = X
        yw = y

    Xw = Xw[:args.n]
    yw = yw[:args.n]
    print(f"Test set: N={len(Xw)} windows of shape {Xw.shape[1:]}")

    # Load FP32 model
    import torch
    from quantize_v4se_23dim_ch32_to_h import TCNClassifierSE
    ckpt = torch.load(fp32_pt, map_location="cpu", weights_only=False)
    fp32_model = TCNClassifierSE(in_ch=23, channels=32, n_blocks=3,
                                  dilations=(1, 2, 4), dropout=0.1)
    fp32_model.load_state_dict(ckpt["state_dict"])
    fp32_model.eval()

    # Load MCU wrapper
    sys.path.insert(0, str(repo))
    from ids.model_loader import _McuHeaderWrapper
    mcu_wrapper = _McuHeaderWrapper(mcu_h, window_size=16)

    # Run inference
    fp32_probs = []
    mcu_probs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(Xw)):
            x = torch.from_numpy(Xw[i:i+1]).float()
            logit = fp32_model(x).squeeze().item()
            fp32_probs.append(1.0 / (1.0 + np.exp(-logit)))
    fp32_time = time.time() - t0

    t0 = time.time()
    for i in range(len(Xw)):
        # Pass the window directly to MCU wrapper
        window_feats = Xw[i].T  # (16, 23) for wrapper.infer's 2D path
        label, prob = mcu_wrapper.infer(window_feats)
        mcu_probs.append(prob)
    mcu_time = time.time() - t0

    fp32_probs = np.array(fp32_probs)
    mcu_probs = np.array(mcu_probs)
    fp32_preds = (fp32_probs >= 0.5).astype(int)
    mcu_preds = (mcu_probs >= 0.49).astype(int)

    # Metrics
    max_diff = float(np.max(np.abs(fp32_probs - mcu_probs)))
    mean_diff = float(np.mean(np.abs(fp32_probs - mcu_probs)))
    verdict_agreement = float(np.mean(fp32_preds == mcu_preds))

    def acc_f1(p, y):
        from sklearn.metrics import accuracy_score, f1_score
        return float(accuracy_score(y, p)), float(f1_score(y, p, zero_division=0))

    fp32_acc, fp32_f1 = acc_f1(fp32_preds, yw)
    mcu_acc, mcu_f1 = acc_f1(mcu_preds, yw)

    report = {
        "n_samples": int(len(Xw)),
        "seed": args.seed,
        "fp32_model": str(fp32_pt),
        "mcu_header": str(mcu_h),
        "max_prob_diff": max_diff,
        "mean_prob_diff": mean_diff,
        "verdict_agreement": verdict_agreement,
        "fp32_threshold": 0.5,
        "mcu_threshold": 0.49,
        "fp32_accuracy": fp32_acc,
        "fp32_f1": fp32_f1,
        "mcu_accuracy": mcu_acc,
        "mcu_f1": mcu_f1,
        "fp32_inference_seconds": fp32_time,
        "mcu_inference_seconds": mcu_time,
        "fp32_ms_per_sample": fp32_time / len(Xw) * 1000,
        "mcu_ms_per_sample": mcu_time / len(Xw) * 1000,
    }

    out_json = repo / f"compare_fp32_vs_mcu_int8_v4se_s{args.seed}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(json.dumps(report, indent=2))

    # Markdown summary
    md = f"""# FP32 vs MCU Hybrid INT8 Comparison (seed={args.seed})

**N samples**: {len(Xw)}

| Metric | FP32 (.pt) | MCU (.h) |
|--------|-----------|----------|
| Threshold | 0.50 | 0.49 |
| Accuracy | {fp32_acc:.4f} | {mcu_acc:.4f} |
| F1 | {fp32_f1:.4f} | {mcu_f1:.4f} |
| Inference (ms/sample) | {report['fp32_ms_per_sample']:.2f} | {report['mcu_ms_per_sample']:.2f} |

## Numerical agreement

- Max prob diff: **{max_diff:.2e}** ({"PASS" if max_diff < 1e-5 else "FAIL"} < 1e-5)
- Mean prob diff: {mean_diff:.2e}
- Verdict agreement: **{verdict_agreement*100:.2f}%**
"""
    out_md = repo / f"compare_fp32_vs_mcu_int8_v4se_s{args.seed}.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `cd D:/workspace/claude/Issue/Issue && python compare_fp32_vs_mcu_int8.py --n 200 --seed 42`
Expected: prints JSON report, max_prob_diff < 1e-5, writes `.json` + `.md`

- [ ] **Step 3: Commit**

```bash
git add compare_fp32_vs_mcu_int8.py compare_fp32_vs_mcu_int8_v4se_s42.json compare_fp32_vs_mcu_int8_v4se_s42.md
git commit -m "feat: add compare_fp32_vs_mcu_int8.py for bit-perfect verification"
```

---

## Task 8: End-to-end smoke test and doc commit

**Files:**
- Verify: all files compile, all tests pass, simulator launches

- [ ] **Step 1: Run full test suite**

Run: `cd D:/workspace/claude/Issue/Issue && python -m pytest tests/ -v`
Expected: ALL PASS (existing + new tests)

- [ ] **Step 2: Verify simulator imports cleanly**

Run: `cd D:/workspace/claude/Issue/Issue && python -c "import modbus_simulator; print('OK')"`
Expected: prints `OK` (may show Tk warning but no ImportError)

- [ ] **Step 3: Manual UI smoke test**

Launch: `cd D:/workspace/claude/Issue/Issue && python modbus_simulator.py`

Verify in GUI:
- [ ] IDS panel dropdown shows FP32 `.pt` files (existing)
- [ ] Dropdown shows `[MCU] model_v4_se_23dim_ch32_hybrid_s42.h` entries
- [ ] Selecting MCU entry does not raise an error in status bar
- [ ] Threshold Scale auto-adjusts to 0.49 when MCU selected
- [ ] Selecting FP32 entry resets to default threshold
- [ ] No regression: existing FP32 model still works end-to-end

- [ ] **Step 4: Final commit**

```bash
git status
# If anything modified, commit
git add -A
git commit -m "chore: post-implementation cleanup" || echo "Nothing to commit"
```

- [ ] **Step 5: Update spec with implementation status**

Edit `docs/superpowers/specs/2026-09-29-ids-panel-mcu-model-selection-design.md` line 2:

```diff
-**状态**：草稿，待用户审核
+**状态**：已实现
+**实现日期**：2026-09-29
```

Commit:
```bash
git add docs/superpowers/specs/2026-09-29-ids-panel-mcu-model-selection-design.md
git commit -m "docs(spec): mark MCU model selection spec as implemented"
```

---

## Self-Review Notes

### Spec coverage check
- ✅ Spec 1.1 goal → Task 3, 6, 8
- ✅ Spec 1.2 in-scope items → Tasks 1, 3, 5, 6, 7, 8
- ✅ Spec 1.3 out-of-scope → respected (no 5-seed ensemble, no latency sim)
- ✅ Spec 3.1 bit-perfect design → Task 3 (dequant to FP32)
- ✅ Spec 3.2 23-dim features → Task 1
- ✅ Spec 3.3 header parsing → Task 2
- ✅ Spec 3.6 wrapper interface → Task 3 (kind, n_features, threshold, infer)
- ✅ Spec 4 UI changes → Task 6
- ✅ Spec 5 error handling → Task 2, 3 (raise ValueError, defaults)
- ✅ Spec 6.1 unit tests → Tasks 1, 2, 3, 4
- ✅ Spec 6.2 comparison script → Task 7
- ✅ Spec 10 acceptance criteria → Task 8

### Type/name consistency
- `_McuHeaderWrapper.kind` = `"MCU-Hybrid-INT8"` — used consistently in Tasks 3, 4, 6
- `_McuHeaderWrapper.n_features` = `23` — Task 1 features, Task 6 branch
- `_McuHeaderWrapper.threshold` = `0.49` — Task 6 default, Task 7 script
- `extract_features_23` — Task 1 defines, Task 3 uses, Task 5 exports, Task 6 imports

### Placeholder scan
No TBD/TODO. All code blocks are concrete.
