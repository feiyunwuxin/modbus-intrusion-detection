# IDS 3D Window Model `lstm.`/`gru.` Prefix Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real SCADA BiLSTM / GRU `.pt` checkpoints (with `lstm.` / `gru.` prefixed RNN keys) loadable and inferable through `load_model(path, window_size=N)`.

**Architecture:** Strip the matched RNN prefix (`arch["module"] + "."`) from state_dict keys before delegating to `rnn.load_state_dict`. Single-function change in `ids/model_loader.py::_reconstruct_lstm`. New test verifies the actual happy-path inference (not just the friendly-rejection path).

**Tech Stack:** Python 3.11+, PyTorch (already a runtime dependency of the IDS subsystem), `unittest` (stdlib, matches existing test layout).

## Global Constraints

- Spec location: `docs/superpowers/specs/2026-09-21-ids-3d-prefix-fix-design.md`
- Test command: `python -m unittest discover -v` from repo root (`D:\workspace\claude\Issue\Issue`)
- All commits use the project's existing conventional-commit style with a scope tag: `fix(ids):` or `test(ids):`
- Public API surface (`load_model`, `ModelWrapper` protocol, `IDsPanel`) is **not** changed.
- Only `_reconstruct_lstm` in `ids/model_loader.py` is modified. The companion `_reconstruct_gru` path goes through the same function (it is the `else` branch), so the fix covers both.
- Classifier keys (e.g. `net.0.weight`) are NOT prefixed with `lstm.` / `gru.` and must pass through untouched.

---

### Task 1: Add the failing cross-prefix BiLSTM e2e test

**Files:**
- Modify: `tests/test_ids.py:304` (append a new test method to `class TestLoadTorch`)
- No other files touched in this task.

**Interfaces:**
- Consumes: existing `load_model(path, window_size=...)` signature (added in the prior 3D-window commit), `wrapper.window_size` property, `wrapper.infer(features)` returning `tuple[int, float] | None`.
- Produces: a green test `test_load_torch_bilstm_cross_prefix_infers` that exercises real SCADA-style `lstm.*`-prefixed state_dict end-to-end.

- [ ] **Step 1: Open `tests/test_ids.py` and locate `class TestLoadTorch`**

Use the Read tool on `D:\workspace\claude\Issue\Issue\tests\test_ids.py`. Find the `class TestLoadTorch` block and the last method inside it (currently `test_load_torch_3d_window_loads_with_window_size` ending around line 372). Append the new method immediately after.

- [ ] **Step 2: Add the new test method**

Insert the following method as the **last** method of `class TestLoadTorch`. Do not modify any existing test.

```python
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

        # Re-key with the SCADA-style lstm.* prefix for RNN; classifier keys
        # under net.* pass through unchanged.
        sd = {
            f"lstm.{k}": v for k, v in m.state_dict().items()
            if k.startswith("lstm.")
        }
        sd.update({
            k: v for k, v in m.state_dict().items()
            if k.startswith("net.")
        })

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
```

- [ ] **Step 3: Run the new test in isolation to verify it FAILS**

Run from repo root:

```bash
python -m unittest tests.test_ids.TestLoadTorch.test_load_torch_bilstm_cross_prefix_infers -v
```

Expected: FAIL with an error mentioning `RNN state_dict 加载不完整` (because `_reconstruct_lstm` currently passes the full prefixed state_dict to `rnn.load_state_dict`, which reports all expected keys as missing). The exact wording may vary; the key signal is that the test fails in `_reconstruct_lstm`, not in test setup.

- [ ] **Step 4: Verify the other 29 tests still pass**

Run:

```bash
python -m unittest discover -v
```

Expected: 28 PASS, 1 FAIL (the new test). All pre-existing tests must still pass — the new test does not change any other code path.

- [ ] **Step 5: Commit the failing test**

```bash
git add tests/test_ids.py
git commit -m "test(ids): add BiLSTM cross-prefix e2e test for 3D window loader

Verifies the *happy path* of _Torch3DStateDictWrapper: a SCADA-style
state_dict with lstm.* prefixed RNN keys must produce real (label, prob)
output after warm-up. Currently fails because _reconstruct_lstm does
not strip the prefix before rnn.load_state_dict."
```

---

### Task 2: Fix `_reconstruct_lstm` to strip the matched prefix

**Files:**
- Modify: `ids/model_loader.py` — single function `_reconstruct_lstm` (locate via Read; the function body is roughly 25 lines starting with `def _reconstruct_lstm(sd: dict, arch: dict):`)

**Interfaces:**
- Consumes: `sd: dict` (full state_dict with prefixed and unprefixed keys), `arch: dict` returned by `_parse_lstm_arch(sd)` containing key `"module"` ∈ `{"lstm", "gru"}`.
- Produces: a `nn.LSTM` or `nn.GRU` module with weights loaded and `eval()` mode set. Same signature as before.

- [ ] **Step 1: Read `_reconstruct_lstm` to confirm current shape**

Use Read on `D:\workspace\claude\Issue\Issue\ids\model_loader.py`. Find `_reconstruct_lstm`. Confirm:
- Function starts with `def _reconstruct_lstm(sd: dict, arch: dict):`
- Body builds an `nn.LSTM` or `nn.GRU` (the `if arch["module"] == "lstm":` branch)
- Then calls `missing, unexpected = rnn.load_state_dict(sd, strict=False)` directly with the raw `sd`
- Then raises `ValueError` if missing/unexpected is non-empty, then `rnn.eval()` and `return rnn`

- [ ] **Step 2: Replace the body of `_reconstruct_lstm`**

Replace the entire function body (everything from `def _reconstruct_lstm(...)` through the final `return rnn`) with:

```python
def _reconstruct_lstm(sd: dict, arch: dict):
    """从 state_dict 构建 nn.LSTM / nn.GRU 并加载权重。

    Real SCADA checkpoints store RNN weights under ``lstm.`` / ``gru.``
    prefixes (e.g. ``lstm.weight_ih_l0``); strip that prefix before
    delegating to ``rnn.load_state_dict``. Classifier keys (e.g.
    ``net.0.weight``) are untouched here — they are handled by
    ``_reconstruct_classifier_from_state_dict`` in the caller.
    """
    import torch.nn as nn

    if arch["module"] == "lstm":
        rnn = nn.LSTM(
            input_size=arch["input_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            batch_first=True,
            bidirectional=arch["bidirectional"],
        )
    else:
        rnn = nn.GRU(
            input_size=arch["input_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            batch_first=True,
            bidirectional=arch["bidirectional"],
        )

    prefix = arch["module"] + "."
    stripped = {
        k.removeprefix(prefix): v
        for k, v in sd.items()
        if k.startswith(prefix)
    }
    missing, unexpected = rnn.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"RNN state_dict 加载不完整: missing={missing[:3]}..., "
            f"unexpected={unexpected[:3]}..."
        )
    rnn.eval()
    return rnn
```

Do not change:
- The function signature `def _reconstruct_lstm(sd: dict, arch: dict):`
- The `_parse_lstm_arch` function (upstream caller)
- The `_reconstruct_classifier_from_state_dict` function (downstream caller)
- Anything outside this function.

- [ ] **Step 3: Run the previously-failing test to verify it now PASSES**

Run:

```bash
python -m unittest tests.test_ids.TestLoadTorch.test_load_torch_bilstm_cross_prefix_infers -v
```

Expected: PASS (`ok`). The test should now produce a `(label, prob)` tuple after 8 warm-up frames.

- [ ] **Step 4: Run the full suite to verify nothing else regressed**

Run:

```bash
python -m unittest discover -v
```

Expected: 30 PASS, 0 FAIL. The previously-passing 29 tests must still pass; the new test from Task 1 must pass.

- [ ] **Step 5: Sanity-check the existing friendly-rejection test still passes**

The test `test_load_torch_3d_window_loads_with_window_size` exercises the window_size=1 rejection path. The fix in this task does not touch that branch, but run it explicitly to be sure:

```bash
python -m unittest tests.test_ids.TestLoadTorch.test_load_torch_3d_window_loads_with_window_size -v
```

Expected: PASS.

- [ ] **Step 6: Commit the fix**

```bash
git add ids/model_loader.py
git commit -m "fix(ids): strip lstm./gru. prefix before rnn.load_state_dict

Real SCADA BiLSTM/GRU checkpoints store RNN weights under lstm./gru.
prefixes (e.g. lstm.weight_ih_l0) that nn.LSTM/nn.GRU do not accept.
Strip the matched prefix inside _reconstruct_lstm so the actual
inference path now works for prefixed state_dicts.

Classifier keys (net.*) pass through untouched.

Closes the latent bug acknowledged in the docstring of
test_load_torch_3d_window_loads_with_window_size."
```

---

### Task 3: Manual smoke test against a real SCADA BiLSTM file

**Files:** No code changes. Spot-check only.

**Interfaces:** Uses `load_model(path, window_size=...)` with a real `.pt` file from the SCADA model set (one of the 183 in the repo).

- [ ] **Step 1: Pick a real SCADA BiLSTM file**

Run from repo root:

```bash
python -c "from ids.model_loader import list_available_models; [print(p) for p in list_available_models('.') if 'bilstm' in p.lower()][:5]"
```

Expected: a few `.pt` paths printed (e.g. `model_bilstm_19dim_v3_19dim.pt`). Pick the first one.

If none are printed (model files not present in this checkout), skip the rest of this task — the unit-test coverage from Tasks 1–2 is sufficient.

- [ ] **Step 2: Run a one-shot inference check**

Run from repo root (substitute the path from Step 1):

```bash
python -c "
import numpy as np
from ids.model_loader import load_model
from ids.inference import extract_features, FEATURE_COLUMNS

# window_size must match the model's training-time window.
# BiLSTM SCADA models were trained at window=16 per the daily log.
w = load_model('<PATH_FROM_STEP_1>', window_size=16)
print('window_size =', w.window_size)
print('warmup_remaining =', w.warmup_remaining)
# Drive warmup with dummy single-frame features
for _ in range(w.warmup_remaining):
    out = w.infer(np.zeros(17, dtype=np.float32))
    assert out is None, f'expected None during warmup, got {out}'
# First real prediction
out = w.infer(np.zeros(17, dtype=np.float32))
print('first prediction =', out)
assert out is not None
"
```

Expected: `window_size` prints 16, `warmup_remaining` decreases to 0, and the first `infer` call returns a `(label, prob)` tuple (not `None`). This proves end-to-end that real prefixed BiLSTM checkpoints now load and infer.

If you see `ValueError: RNN 期望 N 特征，但 IDS 提供 17`, that means the chosen BiLSTM was trained on a different feature count — try another file, or report the issue. The unit-test path is what matters; this manual step is a bonus confirmation.

- [ ] **Step 3: No commit (smoke test only)**

This task produces no source changes. Do not commit.

---

## Self-Review

**1. Spec coverage:**
- Spec goal: real SCADA BiLSTM/GRU loadable and inferable through `load_model(path, window_size=N)` → Task 2 (fix) + Task 1 (test proves it).
- Spec non-goal "Conv1d reconstruction" — not changed → confirmed, only `_reconstruct_lstm` is touched.
- Spec non-goal "Bidirectional key handling" — passes through unchanged after prefix strip → confirmed, `_removeprefix` strips exactly the outer `lstm.` / `gru.`; `_reverse` suffix keys are part of `nn.LSTM.load_state_dict`'s standard accepted naming.
- Spec non-goal "Public API changes" → confirmed, no signature or protocol changes.
- Spec error-handling paragraph (empty stripped → missing keys → existing ValueError) → covered: `rnn.load_state_dict(stripped, strict=False)` raises the existing ValueError with the same wording.
- Spec test paragraph (build BiLSTM with `lstm.` prefix, save, load, warmup, predict) → Task 1 implements it exactly; minor adjustments (added bidirectional True and `net.Sequential` for realistic classifier shape, used `_BiLSTMWithPrefix` helper class instead of `m.state_dict()` re-keying into the same module — semantically equivalent).

**2. Placeholder scan:** No TBD/TODO/fill-in markers. Each step has concrete commands or code. No "similar to Task N" cross-references that omit content.

**3. Type consistency:**
- `load_model(path, window_size=8)` — matches existing signature in `ids/model_loader.py`.
- `wrapper.window_size` — matches the property added in the prior 3D-window commit (verified in the unstaged diff).
- `wrapper.infer(features)` returning `tuple[int, float] | None` — matches the same commit.
- `_reconstruct_lstm(sd, arch)` — signature unchanged; `arch["module"]` ∈ `{"lstm", "gru"}` matches the docstring of `_parse_lstm_arch` (also verified in the unstaged diff).
- `_removeprefix` is a Python 3.9+ stdlib `str` method. Project requires Python 3.11+ per global constraints → safe.
