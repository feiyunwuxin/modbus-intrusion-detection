"""Pure-Python PC-side IDS feature pipeline.

This module mirrors the reference implementation in
``scripts/pc_inference_765_vs_h743.py`` so that the 23-dim TCN+SE model
can score the Modbus traffic that ``SerialWorker`` writes to the serial
port in real time.

Two responsibilities:

1. ``featurize_one`` + ``IncrementalFeatureState`` — convert one CSV row
   dict into the 18-dim raw feature vector used by the model, bit-matching
   the batch reference ``build_18_raw_from_csv``. Stateful so the
   caller can stream rows one at a time without re-loading the dataset.

2. ``build_window_input`` — convert a 16-frame deque of (already
   normalized) 18-dim vectors into the (23, 16) model input tensor,
   bit-matching the reference ``sliding_window_18_to_23``.

Loaders for the ``RobustScaler`` (joblib) and threshold JSON live here
too so the IDS worker doesn't have to import joblib/sklearn at module
top.

Importability contract: importing constants and dataclasses works
without numpy/joblib installed — only the loader functions and
``featurize_one``/``build_window_input`` need numpy. This keeps the
``[ids]`` extra opt-in (the rest of the GUI + the IDS panel can be
constructed without torch/numpy/joblib installed).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Constants — must match pc_inference_765_vs_h743.py:23-25 + 109-112
# ---------------------------------------------------------------------------

#: Sliding window length. The TCN+SE model expects the last 16 frames.
SW_WINDOW: int = 16

#: Hard clip applied to the (23, 16) model input. The reference clips to
#: ±10 because the per-batch ``RobustScaler`` can produce arbitrarily
#: large values on outliers and the TCN was trained on the clipped range.
CLIP_VAL: float = 10.0

#: Indices into the 19-feature scaler to select the 18-dim subset used
#: for the 16-frame sliding window. Dropped: column 3 (``setpoint``,
#: which is a control target rather than a measurement).
KEEP_18: list[int] = [
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
]

#: Indices into the 27-dim augmented frame to select the final 23-dim
#: model input. The 27-dim frame is the 18 raw columns plus 9 derived
#: aggregates (pressure_mean, crc_mean, crc_max, cmd_count,
#: resp_signed, balance, n_distinct_lengths, unusual_sign) — this
#: KEEP_23 drops 4 of those aggregates that the LOO ablation proved
#: unhelpful.
KEEP_23: list[int] = [
    0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 21, 23, 24, 25, 26,
]

#: Function codes that are statistically rare in normal Modbus traffic
#: — flagging them gives the model a strong attack signal (DoS/NMRI
#: variants occasionally inject these). Same set as the reference.
UNUSUAL_FCS: frozenset[int] = frozenset({136, 171, 139, 133, 137, 138, 140})


# ---------------------------------------------------------------------------
# Loaders — joblib/sklearn/numpy are imported lazily so the module itself
# imports cleanly on a stripped-down install.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalerBundle:
    """A loaded ``RobustScaler`` projected onto the 18-dim KEEP_18 subset.

    ``mu_18`` / ``sg_18`` are float32 1-D arrays of length 18. Use
    :meth:`normalize` to convert a (N, 18) raw feature matrix into the
    scaler-normalized space the model was trained on.
    """

    mu_18: Any  # np.ndarray, but type is Any so module imports without numpy
    sg_18: Any

    def normalize(self, raw18: Any) -> Any:
        """``(N, 18) → (N, 18)`` element-wise (x - mu) / sg."""
        return (raw18 - self.mu_18) / self.sg_18


@dataclass(frozen=True)
class ThresholdBundle:
    """A loaded ``best_threshold_*.json`` file.

    The numeric fields mirror the reference JSON keys. ``raw`` keeps the
    full payload so the GUI can surface extra metrics (e.g. as tooltips).
    """

    best_threshold: float
    val_macro_f1_at_best: float | None
    val_binary_f1_at_best: float | None
    val_macro_f1_at_0_5: float | None
    raw: dict


def load_scaler(path: str | Path) -> ScalerBundle:
    """Load a ``joblib``-dumped ``RobustScaler`` and project to KEEP_18.

    The reference scaler was fit on 19 features; we slice it down to the
    18-dim subset via :data:`KEEP_18`. Raises ``ValueError`` if the
    loaded scaler's ``center_`` / ``scale_`` are not length 19.

    Imports ``joblib`` and ``numpy`` lazily so the rest of the module
    imports cleanly without them.
    """
    import joblib
    import numpy as np

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scaler not found: {p}")
    scaler = joblib.load(p)
    center = np.asarray(scaler.center_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    if center.shape != (19,) or scale.shape != (19,):
        raise ValueError(
            f"scaler has unexpected shape: center={center.shape}, "
            f"scale={scale.shape} (expected (19,))"
        )
    keep = np.asarray(KEEP_18, dtype=np.int64)
    return ScalerBundle(
        mu_18=center[keep].astype(np.float32),
        sg_18=scale[keep].astype(np.float32),
    )


def load_threshold(path: str | Path) -> ThresholdBundle:
    """Load a threshold JSON file.

    Required key: ``best_threshold``. Optional keys (stored as-is for
    display in the GUI):

    * ``val_macro_f1_at_best``
    * ``val_binary_f1_at_best``
    * ``val_macro_f1_at_0_5``
    * ``metric``

    Falls back to 0.5 if ``best_threshold`` is missing — the GUI's
    threshold spinbox will then show 0.5 and the user is expected to
    fix it manually.

    Imports ``json`` (stdlib only — no numpy/joblib needed here).
    """
    import json

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"threshold not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"threshold JSON must be an object: got {type(data)}")
    best = data.get("best_threshold")
    if best is None:
        best = 0.5
    return ThresholdBundle(
        best_threshold=float(best),
        val_macro_f1_at_best=_maybe_float(data.get("val_macro_f1_at_best")),
        val_binary_f1_at_best=_maybe_float(data.get("val_binary_f1_at_best")),
        val_macro_f1_at_0_5=_maybe_float(data.get("val_macro_f1_at_0_5")),
        raw=data,
    )


def _maybe_float(v: Any) -> float | None:
    """Best-effort float conversion; returns None for None/non-numeric."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Incremental 18-dim featurizer — bit-matches build_18_raw_from_csv.
# ---------------------------------------------------------------------------


@dataclass
class IncrementalFeatureState:
    """Per-run state for the streaming 18-dim featurizer.

    Reset between replays. ``last_seen`` carries the wall-clock time of
    the most recent row seen for each ``(address, function)`` key, so
    ``time_since_last_same_addr_func`` can be computed incrementally.
    ``prev_time`` is the wall-clock time of the most recent row overall
    (for ``time_diff``). On the very first row both values are treated
    as "no previous row" → 0.0 — matching the reference's
    ``df["time"].diff().fillna(0)`` semantics.
    """

    last_seen: dict[tuple[int, int], float] = field(default_factory=dict)
    prev_time: float | None = None


def featurize_one(state: IncrementalFeatureState, row: dict) -> Any:
    """Compute the 18-dim raw feature vector for one row.

    Bit-matches ``build_18_raw_from_csv`` in
    ``scripts/pc_inference_765_vs_h743.py:85-106``. The output for row N
    equals ``build_18_raw_from_csv(df)[N]`` when ``state`` starts fresh
    and ``row`` is the Nth dict from ``csv_loader.load_rows``.

    ``row`` uses the original CSV column names — the same dicts that
    ``csv_loader`` produces and that ``SerialWorker._rows`` already
    holds. ``"?"`` sentinels (and any value that fails the numeric
    coerce) are mapped to 0.0 — matching the reference's
    ``pd.to_numeric(errors="coerce").fillna(0)``.

    Returns an ``np.ndarray`` shape ``(18,)`` dtype float32.
    """
    import numpy as np

    def _num(v: Any, default: float = 0.0) -> float:
        """Match pd.to_numeric(errors="coerce").fillna(default)."""
        if v is None or v == "?":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _int(v: Any, default: int = 0) -> int:
        """Integer-cast (used for columns the reference treats as int)."""
        if v is None or v == "?":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return default

    t = _num(row.get("time"))
    dt = 0.0 if state.prev_time is None else t - state.prev_time

    addr_i = _int(row.get("address"))
    fc_i = _int(row.get("function"))
    k = (addr_i, fc_i)
    tsl = 0.0 if k not in state.last_seen else t - state.last_seen[k]

    # Update state AFTER reading it — mirrors the reference which reads
    # ``df["time"].iat[i] - last.get(k)`` before assigning ``last[k]``.
    state.prev_time = t
    state.last_seen[k] = t

    is_unusual_fc = 1.0 if fc_i in UNUSUAL_FCS else 0.0
    is_response = float(_int(row.get("command response")))
    # Reference clips pressure measurement to [0, 100] after fillna(0).
    pressure = max(0.0, min(100.0, _num(row.get("pressure measurement"))))

    # Order MUST match build_18_raw_from_csv: ``keep`` list in the
    # reference is exactly these 18 columns in this order.
    vec = np.array(
        [
            addr_i,                                              # 0  address
            fc_i,                                                # 1  function
            _int(row.get("length")),                              # 2  length
            _num(row.get("gain")),                                # 4  gain
            _num(row.get("reset rate")),                          # 5  reset rate
            _num(row.get("deadband")),                            # 6  deadband
            _num(row.get("cycle time")),                          # 7  cycle time
            _num(row.get("rate")),                                # 8  rate
            _int(row.get("system mode")),                         # 9  system mode
            _int(row.get("control scheme")),                      # 10 control scheme
            _int(row.get("pump")),                                # 11 pump
            _int(row.get("solenoid")),                            # 12 solenoid
            pressure,                                            # 13 pressure measurement
            _num(row.get("crc rate")),                            # 14 crc rate
            dt,                                                  # 15 time_diff
            tsl,                                                 # 16 time_since_last_same_addr_func
            is_unusual_fc,                                       # 17 is_unusual_fc
            is_response,                                         # 18 is_response
        ],
        dtype=np.float32,
    )
    return vec


def iter_features_from_rows(
    rows: list[dict],
) -> Iterator[Any]:
    """Yield one 18-dim vector per row using a fresh state.

    Convenience wrapper used by ``features_array_from_rows`` and the
    batch-featurizer test. Not on the streaming hot path; for that, the
    IDS worker holds its own :class:`IncrementalFeatureState` and calls
    :func:`featurize_one` directly.
    """
    state = IncrementalFeatureState()
    for row in rows:
        yield featurize_one(state, row)


def features_array_from_rows(rows: list[dict]) -> Any:
    """Stack ``iter_features_from_rows(rows)`` into an ``(N, 18)`` array."""
    import numpy as np

    return np.stack(list(iter_features_from_rows(rows)), axis=0)


# ---------------------------------------------------------------------------
# Sliding window → (23, 16) model input — bit-matches sliding_window_18_to_23.
# ---------------------------------------------------------------------------


def build_window_input(window_deque: deque) -> Any:
    """Convert a 16-frame deque of (already normalized) 18-dim vectors
    into the model's ``(23, 16)`` input tensor.

    Bit-matches ``sliding_window_18_to_23`` in
    ``scripts/pc_inference_765_vs_h743.py:109-145``. ``window_deque``
    must already contain exactly ``SW_WINDOW`` rows of length-18
    vectors; the caller decides whether to normalize before or after
    windowing (the IDS pipeline normalizes *before* windowing, so the
    values here are on the scaler's scale).

    Output: ``np.ndarray`` shape ``(23, 16)`` dtype float32, clipped to
    ±:data:`CLIP_VAL`. The (16, 23) intermediate is transposed at the
    very end to match the channel-first layout the model expects.
    """
    import numpy as np

    if len(window_deque) != SW_WINDOW:
        raise ValueError(
            f"window_deque must have exactly {SW_WINDOW} rows, "
            f"got {len(window_deque)}"
        )
    win = np.stack(list(window_deque), axis=0)  # (16, 18)

    press_sum = float(win[:, 12].sum())
    crc_max = float(win[:, 13].max())
    resp_signed = float(win[:, 17].sum())
    unusual_sign = float(win[:, 16].sum())

    # Distinct lengths in the window (rounded to int to absorb the
    # RobustScaler rounding noise on column 2 = ``length``).
    seen: set[int] = set()
    n_distinct_lengths = 0
    for v in win[:, 2]:
        k = int(round(float(v)))
        if k not in seen:
            seen.add(k)
            n_distinct_lengths += 1

    press_mean = press_sum / SW_WINDOW
    balance = 1.0 - resp_signed / 8.0
    crc_mean = float(win[:, 13].mean())
    cmd_count = float(SW_WINDOW - unusual_sign - resp_signed)

    # 27-dim augmented frame, then slice by KEEP_23 to get 23-dim.
    frame27 = np.zeros((SW_WINDOW, 27), dtype=np.float32)
    # Scatter the 18 raw columns into the 27-dim frame at KEEP_18.
    # Note: KEEP_18 and KEEP_23 are NOT the same — we put the raw
    # 18 columns into the same positions the reference uses (the
    # KEEP_18 indices match the 27-dim frame's left 19 slots minus the
    # dropped setpoint column). This mirrors lines 132-141 of the
    # reference verbatim.
    keep_18 = np.asarray(KEEP_18, dtype=np.int64)
    for c18 in range(18):
        frame27[:, keep_18[c18]] = win[:, c18]
    frame27[:, 19] = press_mean
    frame27[:, 20] = crc_mean
    frame27[:, 21] = crc_max
    frame27[:, 22] = cmd_count
    frame27[:, 23] = resp_signed
    frame27[:, 24] = balance
    frame27[:, 25] = n_distinct_lengths
    frame27[:, 26] = unusual_sign

    keep_23 = np.asarray(KEEP_23, dtype=np.int64)
    sliced = frame27[:, keep_23]  # (16, 23)
    clipped = np.clip(sliced, -CLIP_VAL, CLIP_VAL)
    return clipped.T.astype(np.float32)  # (23, 16) — channel-first


__all__ = [
    "SW_WINDOW",
    "CLIP_VAL",
    "KEEP_18",
    "KEEP_23",
    "UNUSUAL_FCS",
    "ScalerBundle",
    "ThresholdBundle",
    "IncrementalFeatureState",
    "load_scaler",
    "load_threshold",
    "featurize_one",
    "build_window_input",
    "iter_features_from_rows",
    "features_array_from_rows",
]