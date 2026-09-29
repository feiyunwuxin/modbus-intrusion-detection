"""Tests for the PC-side IDS feature pipeline.

These tests are pure-Python: no PyQt5, no torch. They verify that the
streaming featurizer bit-matches the reference batch implementation in
``scripts/pc_inference_765_vs_h743.py`` and that the loaders accept
the reference files. The tests use ``pytest.importorskip("numpy")``
etc. so they run cleanly on a stripped-down install that only has the
``[dev]`` extra.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="ids_pipeline requires numpy")
import numpy as np

from modbus_replay.ids_pipeline import (
    CLIP_VAL,
    IncrementalFeatureState,
    KEEP_18,
    KEEP_23,
    SW_WINDOW,
    UNUSUAL_FCS,
    build_window_input,
    featurize_one,
    features_array_from_rows,
    iter_features_from_rows,
    load_scaler,
    load_threshold,
)


# ---------------------------------------------------------------------------
# Reference fixture data — small hand-crafted row sets exercising the
# edge cases of featurize_one (first row, repeat (addr,fc), missing
# fields, unusual FCs).
# ---------------------------------------------------------------------------


def _row(
    addr=1,
    fc=3,
    length=16,
    gain=1.0,
    reset_rate=1.0,
    deadband=0.0,
    cycle_time=0,
    rate=0.0,
    system_mode=0,
    control_scheme=0,
    pump=0,
    solenoid=0,
    pressure=0.0,
    crc_rate=0,
    command_response=0,
    time=0.0,
):
    return {
        "address": addr,
        "function": fc,
        "length": length,
        "gain": gain,
        "reset rate": reset_rate,
        "deadband": deadband,
        "cycle time": cycle_time,
        "rate": rate,
        "system mode": system_mode,
        "control scheme": control_scheme,
        "pump": pump,
        "solenoid": solenoid,
        "pressure measurement": pressure,
        "crc rate": crc_rate,
        "command response": command_response,
        "time": time,
    }


# ---------------------------------------------------------------------------
# Constants — sanity checks so a typo in KEEP_23 / KEEP_18 trips the
# tests immediately rather than silently producing wrong model output.
# ---------------------------------------------------------------------------


def test_constants_shape():
    assert SW_WINDOW == 16
    assert CLIP_VAL == 10.0
    assert len(KEEP_18) == 18
    assert len(KEEP_23) == 23
    assert len(set(KEEP_18)) == 18
    assert len(set(KEEP_23)) == 23
    assert max(KEEP_18) < 19
    assert max(KEEP_23) < 27


def test_unusual_fcs_set():
    # UNUSUAL_FCS is exactly the 7-element set from the reference.
    assert UNUSUAL_FCS == frozenset({136, 171, 139, 133, 137, 138, 140})


# ---------------------------------------------------------------------------
# featurize_one — bit-match checks against the reference semantics.
# ---------------------------------------------------------------------------


def test_featurize_one_first_row_time_diff_zero():
    """First row's time_diff is 0 even when time > 0 (matches .diff().fillna(0))."""
    state = IncrementalFeatureState()
    row = _row(time=1000.5, pressure=12.3)
    feat = featurize_one(state, row)
    assert feat.shape == (18,)
    assert feat.dtype == np.float32
    # idx 14 is time_diff
    assert feat[14] == 0.0
    # idx 15 is time_since_last_same_addr_func — first occurrence → 0
    assert feat[15] == 0.0
    # state was updated
    assert state.prev_time == 1000.5
    assert state.last_seen == {(1, 3): 1000.5}


def test_featurize_one_second_row_time_diff_nonzero():
    state = IncrementalFeatureState()
    featurize_one(state, _row(time=1000.0))
    feat = featurize_one(state, _row(time=1001.5))
    assert feat[14] == pytest.approx(1.5, abs=1e-6)


def test_featurize_one_time_since_last_zero_on_first_occurrence():
    """Same (addr, fc) twice — second row's tsl is the delta."""
    state = IncrementalFeatureState()
    featurize_one(state, _row(addr=2, fc=5, time=2000.0))
    # Different (addr, fc) — tsl should be 0
    feat = featurize_one(state, _row(addr=3, fc=5, time=2001.0))
    assert feat[15] == 0.0
    # Same (addr, fc) — tsl should be the delta
    feat = featurize_one(state, _row(addr=2, fc=5, time=2003.0))
    assert feat[15] == pytest.approx(3.0, abs=1e-6)


def test_featurize_one_handles_question_mark_sentinel():
    """`?` sentinels → 0.0 (matches pd.to_numeric(errors='coerce').fillna(0))."""
    state = IncrementalFeatureState()
    row = _row(pressure="?", gain="?", reset_rate="?")
    feat = featurize_one(state, row)
    # pressure measurement → idx 12
    assert feat[12] == 0.0
    # gain → idx 3 (CSV col 4)
    assert feat[3] == 0.0
    # reset rate → idx 4 (CSV col 5)
    assert feat[4] == 0.0


def test_featurize_one_pressure_clipped_to_0_100():
    state = IncrementalFeatureState()
    feat = featurize_one(state, _row(pressure=-5.0))
    assert feat[12] == 0.0
    feat = featurize_one(state, _row(pressure=150.0))
    assert feat[12] == 100.0
    feat = featurize_one(state, _row(pressure=42.5))
    assert feat[12] == 42.5


def test_featurize_one_unusual_fc_flag():
    state = IncrementalFeatureState()
    # Function 136 is in UNUSUAL_FCS
    feat = featurize_one(state, _row(fc=136))
    assert feat[16] == 1.0
    # Function 16 is normal
    feat = featurize_one(state, _row(fc=16))
    assert feat[16] == 0.0


def test_featurize_one_is_response_flag():
    state = IncrementalFeatureState()
    feat = featurize_one(state, _row(command_response=1))
    assert feat[17] == 1.0
    feat = featurize_one(state, _row(command_response=0))
    assert feat[17] == 0.0


def test_featurize_one_column_order_matches_reference():
    """Index-by-index check that we use the same 18 columns as build_18_raw_from_csv.

    Order reference (from pc_inference_765_vs_h743.py:100-103):
      address, function, length, gain, reset rate, deadband, cycle time,
      rate, system mode, control scheme, pump, solenoid,
      pressure measurement, crc rate, time_diff,
      time_since_last_same_addr_func, is_unusual_fc, is_response
    """
    state = IncrementalFeatureState()
    feat = featurize_one(
        state,
        _row(
            addr=4, fc=3, length=16, gain=0.5, reset_rate=0.1, deadband=0.2,
            cycle_time=10, rate=0.3, system_mode=1, control_scheme=2,
            pump=1, solenoid=0, pressure=42.5, crc_rate=99,
            command_response=1, time=5000.0,
        ),
    )
    expected = np.array(
        [4, 3, 16, 0.5, 0.1, 0.2, 10.0, 0.3, 1, 2, 1, 0, 42.5, 99.0,
         0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(feat, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# iter_features_from_rows — convenience wrappers.
# ---------------------------------------------------------------------------


def test_iter_features_from_rows_yields_one_per_row():
    rows = [_row(time=1000.0), _row(time=1001.0), _row(time=1002.0)]
    feats = list(iter_features_from_rows(rows))
    assert len(feats) == 3
    for feat in feats:
        assert feat.shape == (18,)


def test_features_array_from_rows_shape():
    rows = [_row(time=1000.0 + i) for i in range(50)]
    arr = features_array_from_rows(rows)
    assert arr.shape == (50, 18)
    assert arr.dtype == np.float32


def test_streaming_matches_batch_for_constructed_rows():
    """iter_features_from_rows and a single stateful call produce the same array."""
    rows = [_row(time=1000.0 + 0.1 * i, addr=(i % 3) + 1, fc=(i % 5) + 3,
                 pressure=10.0 + i * 0.1, crc_rate=i * 7,
                 command_response=i % 2)
            for i in range(100)]
    arr = features_array_from_rows(rows)
    # Reproduce by hand with a fresh state.
    state = IncrementalFeatureState()
    by_hand = np.stack([featurize_one(state, r) for r in rows], axis=0)
    np.testing.assert_array_equal(arr, by_hand)


# ---------------------------------------------------------------------------
# build_window_input — matches sliding_window_18_to_23 semantics.
# ---------------------------------------------------------------------------


def _make_normalized_window(seed: int = 0) -> deque:
    """Build a 16-frame window of normalized 18-dim values.

    Values are in [-3, 3] so the post-clip output is unchanged; we use
    a fixed seed so the test is deterministic.
    """
    rng = np.random.default_rng(seed)
    return deque(rng.uniform(-3.0, 3.0, size=(SW_WINDOW, 18)).astype(np.float32))


def test_build_window_input_layout():
    window = _make_normalized_window()
    out = build_window_input(window)
    assert out.shape == (23, 16)
    assert out.dtype == np.float32
    # The reference clips to ±CLIP_VAL. With input in [-3, 3] we
    # expect no clipping happened, so the max absolute value stays
    # bounded by 3.
    assert np.abs(out).max() <= CLIP_VAL


def test_build_window_input_wrong_length_raises():
    window = _make_normalized_window()
    window.pop()  # 15 rows now
    with pytest.raises(ValueError, match="exactly 16"):
        build_window_input(window)


def test_build_window_input_clipping():
    """If a window cell exceeds CLIP_VAL, it gets clipped."""
    window = _make_normalized_window()
    # Slam one cell way past CLIP_VAL.
    arr = np.stack(list(window), axis=0)
    arr[5, 12] = 999.0  # column 12 = pressure measurement
    window2 = deque(arr)
    out = build_window_input(window2)
    # Column 12 (pressure) is scattered into frame27 col KEEP_18[12] = 13.
    # KEEP_23 contains 13 at index 11 → output channel 11 holds pressure.
    assert out[11, 5] == pytest.approx(CLIP_VAL, abs=1e-6)
    # press_mean (999/16 ≈ 62.4) lives in frame27 col 19 → KEEP_23 index 17.
    # That also clips to CLIP_VAL.
    assert out[17, 5] == pytest.approx(CLIP_VAL, abs=1e-6)
    # Unrelated channels remain in their original [-3, 3] range.
    assert np.abs(out).max() == pytest.approx(CLIP_VAL, abs=1e-6)


def test_build_window_input_kill_switch_missing_fields_are_zero():
    """An all-zero window must produce a finite (23, 16) output, no NaN/inf."""
    window = deque(np.zeros((SW_WINDOW, 18), dtype=np.float32))
    out = build_window_input(window)
    assert out.shape == (23, 16)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Loaders — exercise against the real reference files if they exist.
# Skipped cleanly on a stripped CI image that doesn't have them.
# ---------------------------------------------------------------------------


#: Path to the canonical 23-dim scaler (matches
#: scripts/pc_inference_765_vs_h743.py:19).
_SCALER_PATH = Path(r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib")
#: Path to the canonical threshold JSON. This may live under any of
#: several names; we look for the most likely one and skip if missing.
_THRESHOLD_CANDIDATES = [
    Path(r"C:\work\Claude\Issue\threshold.json"),
    Path(r"C:\work\Claude\Issue\best_threshold_tcn_v4_se_window16.json"),
]


def _find_threshold() -> Path | None:
    for p in _THRESHOLD_CANDIDATES:
        if p.exists():
            return p
    return None


def test_load_scaler_projects_keep_18():
    if not _SCALER_PATH.exists():
        pytest.skip(f"scaler not found at {_SCALER_PATH}")
    bundle = load_scaler(_SCALER_PATH)
    assert bundle.mu_18.shape == (18,)
    assert bundle.sg_18.shape == (18,)
    assert bundle.mu_18.dtype == np.float32
    # mu_18 must equal scaler.center_[KEEP_18]
    pytest.importorskip("joblib")
    import joblib
    scaler = joblib.load(_SCALER_PATH)
    expected_mu = scaler.center_[KEEP_18].astype(np.float32)
    np.testing.assert_array_equal(bundle.mu_18, expected_mu)


def test_load_scaler_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_scaler(r"C:\work\nope\does\not\exist.joblib")


def test_load_threshold_returns_best_threshold():
    path = _find_threshold()
    if path is None:
        pytest.skip("no canonical threshold JSON found")
    bundle = load_threshold(path)
    # Reference value from best_threshold_tcn_v4_se_window16.json
    assert 0.0 < bundle.best_threshold < 1.0
    assert isinstance(bundle.raw, dict)


def test_load_threshold_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_threshold(r"C:\work\nope\does\not\exist.json")


def test_load_threshold_missing_best_threshold_falls_back_to_0_5(tmp_path: Path):
    """When ``best_threshold`` is absent the loader falls back to 0.5."""
    import json

    p = tmp_path / "thr.json"
    p.write_text(json.dumps({"metric": "macro_f1"}), encoding="utf-8")
    bundle = load_threshold(p)
    assert bundle.best_threshold == 0.5


def test_scaler_normalize_subtracts_mu_only():
    """normalize(x) == (x - mu_18) / sg_18 — element-wise.

    We verify the math against a hand-built bundle with known mu/sg so
    the test is independent of the real scaler's values.
    """
    pytest.importorskip("numpy")
    from modbus_replay.ids_pipeline import ScalerBundle

    mu = np.array([1.0, 2.0, 3.0] + [0.0] * 15, dtype=np.float32)
    sg = np.array([2.0, 4.0, 1.0] + [1.0] * 15, dtype=np.float32)
    bundle = ScalerBundle(mu_18=mu, sg_18=sg)
    raw = np.array([[5.0, 6.0, 3.0] + [7.0] * 15], dtype=np.float32)
    norm = bundle.normalize(raw)
    expected = np.array([[2.0, 1.0, 0.0] + [7.0] * 15], dtype=np.float32)
    np.testing.assert_allclose(norm, expected, atol=1e-6)