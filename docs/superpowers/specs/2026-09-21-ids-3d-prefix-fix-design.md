# 2026-09-21 IDS 3D Window Model `lstm.`/`gru.` Prefix Fix — Design

## Problem

`_Torch3DStateDictWrapper._reconstruct_lstm()` in `ids/model_loader.py` calls
`rnn.load_state_dict(sd)` with the raw state_dict. Real SCADA `BiLSTM` /
`GRU` checkpoints store RNN weights under a `lstm.` / `gru.` prefix
(e.g. `lstm.weight_ih_l0`, `lstm.weight_hh_l0_reverse`), but `nn.LSTM` /
`nn.GRU` expect keys without the prefix (`weight_ih_l0`,
`weight_hh_l0_reverse`). Result: real BiLSTM models cannot be loaded for
inference; only the friendly-rejection path is currently exercised by the
test suite (see test docstring at
`tests/test_ids.py::TestLoadTorch::test_load_torch_3d_window_loads_with_window_size`).

## Goal

Make real SCADA BiLSTM / GRU `.pt` checkpoints (with `lstm.` / `gru.`
prefixed RNN keys) loadable and inferable through
`load_model(path, window_size=N)`.

Non-goals:

- Conv1d reconstruction (already covered by `_reconstruct_conv1d`;
  this change does not touch it)
- Bidirectional key handling beyond what `nn.LSTM.load_state_dict` already
  accepts (`weight_hh_l0_reverse` keys are already in PyTorch's standard
  naming, so they pass through unchanged once the outer prefix is stripped)
- Changes to the public API (`load_model`, `ModelWrapper`, `IDsPanel`)

## Approach

Strip the matched RNN prefix from state_dict keys before delegating to
`rnn.load_state_dict`. The prefix to strip is exactly
`arch["module"] + "."`, where `arch` is the dict returned by
`_parse_lstm_arch(sd)` (already computed one frame up).

The classifier keys (e.g. `net.0.weight`) are NOT prefixed with `lstm.`
or `gru.`, so they pass through `_reconstruct_lstm` untouched and reach
`_reconstruct_classifier_from_state_dict` in the caller as before.

## Code change

Single function: `ids/model_loader.py::_reconstruct_lstm`.

Before:

```python
def _reconstruct_lstm(sd: dict, arch: dict):
    """从 state_dict 构建 nn.LSTM / nn.GRU 并加载权重。"""
    import torch.nn as nn

    if arch["module"] == "lstm":
        rnn = nn.LSTM(...)
    else:
        rnn = nn.GRU(...)

    missing, unexpected = rnn.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(...)
    rnn.eval()
    return rnn
```

After:

```python
def _reconstruct_lstm(sd: dict, arch: dict):
    """从 state_dict 构建 nn.LSTM / nn.GRU 并加载权重。

    Real SCADA checkpoints store RNN weights under ``lstm.`` / ``gru.``
    prefixes (e.g. ``lstm.weight_ih_l0``); strip that prefix before
    delegating to ``rnn.load_state_dict``. Classifier keys (e.g.
    ``net.0.weight``) are untouched.
    """
    import torch.nn as nn

    if arch["module"] == "lstm":
        rnn = nn.LSTM(...)
    else:
        rnn = nn.GRU(...)

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

## Error handling

- Empty `stripped` after filtering → `rnn.load_state_dict` returns all
  expected keys as missing → existing `ValueError("RNN state_dict 加载不完整")`
  fires. (Reachable only via a malformed state_dict that `_parse_lstm_arch`
  mis-classifies; current parse logic prevents this.)
- Keys that do not match the prefix (e.g. classifier `net.0.weight`)
  pass through unchanged; `_reconstruct_classifier_from_state_dict` in the
  caller handles them separately.
- A different prefix (e.g. `rnn.weight_ih_l0`) is not stripped; the
  resulting `missing` set will be non-empty and the existing error
  message surfaces.

## Test

New test in `tests/test_ids.py`, class `TestLoadTorch`:

```python
def test_load_torch_bilstm_cross_prefix_infers(self):
    """SCADA-style BiLSTM with lstm.* prefix should load + infer."""
    if torch is None:
        self.skipTest("torch not installed")
    import torch.nn as nn

    class _TinyBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=17, hidden_size=8,
                                batch_first=True, bidirectional=True)
            self.net = nn.Sequential(nn.Linear(16, 2))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.net(out[:, -1, :])

    m = _TinyBiLSTM()
    m.eval()
    # Add the SCADA-style prefix to LSTM keys
    sd = {f"lstm.{k}": v for k, v in m.state_dict().items()
          if k.startswith("lstm.")}
    # Classifier keys stay as net.0.weight / net.0.bias
    sd.update({k: v for k, v in m.state_dict().items()
               if k.startswith("net.")})

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(sd, f.name)
        path = f.name
    try:
        wrapper = load_model(path, window_size=8)
        self.assertEqual(wrapper.window_size, 8)
        # warmup: first 7 frames return None
        for _ in range(7):
            self.assertIsNone(wrapper.infer(np.zeros(17, dtype=np.float32)))
        # 8th frame returns a tuple
        result = wrapper.infer(np.zeros(17, dtype=np.float32))
        self.assertIsNotNone(result)
        label, prob = result
        self.assertIn(label, (0, 1))
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
    finally:
        os.unlink(path)
```

This test fails before the fix (prefix-stripping path is what makes it
pass) and passes after.

## Files touched

- `ids/model_loader.py` — modify `_reconstruct_lstm` only.
- `tests/test_ids.py` — add `test_load_torch_bilstm_cross_prefix_infers`.

No changes to: `ids_panel.py`, `ids/__init__.py`, public `load_model`,
`ModelWrapper` protocol, GUI, or anything in `modbus_replay_simulator/`.

## Verification

1. `python -m unittest discover` → still 30/30 (was 29, +1 new test).
2. Spot-check with a real SCADA BiLSTM file (one of the 183) using a
   1-line REPL snippet to confirm the previously-broken case now
   returns `(label, prob)` after warmup.
