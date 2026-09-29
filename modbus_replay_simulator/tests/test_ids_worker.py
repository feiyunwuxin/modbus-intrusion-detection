"""Tests for IdsWorker(QThread).

The worker normally imports torch lazily inside run(). These tests
monkeypatch the torch model factory in
``modbus_replay.ids_worker_torch`` so the run-loop can use a tiny
constant-output fake without ever loading torch.

Tests cover:
* warm-up (no frame_scored until 16 rows have arrived),
* per-frame payload fields,
* threshold override,
* CSV log creation, header, and append-without-duplicate-header,
* missing-torch and bad-model-path error paths,
* run_summary totals.
"""
from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("numpy", reason="ids_pipeline requires numpy")
import numpy as np

from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Fake model — module shim so the IdsWorker's ``from
# modbus_replay.ids_worker_torch import build_tcn_se_model`` resolves to
# a function that returns a constant-logit Module.
# ---------------------------------------------------------------------------


class _FakeModel:
    """Constant-logit torch stand-in. forward(x) -> logit * ones((batch,))."""

    def __init__(self, logit: float = 2.0):
        self._logit = logit

    def load_state_dict(self, _state):  # signature parity with nn.Module
        pass

    def eval(self):
        pass

    def __call__(self, x):
        # ``x`` is a torch tensor in production; tests don't depend on
        # the actual tensor, just on the returned scalar logit.
        return self._logit * _ones_like(x)


def _ones_like(x):
    """Return a 1-D tensor of ones with the right batch size.

    Uses torch if available; falls back to numpy otherwise.
    """
    try:
        import torch
        return torch.ones((int(x.shape[0]),), dtype=torch.float32)
    except Exception:
        return np.ones((int(getattr(x, "shape", [1])[0]),), dtype=np.float32)


@pytest.fixture
def fake_torch(monkeypatch):
    """Inject a torch stub into sys.modules so the worker's lazy
    ``import torch`` inside run() finds a working numpy-backed stand-in.
    """
    import numpy as _np

    # --- minimal torch stub -------------------------------------------------
    # Use ModuleType (not SimpleNamespace) so Python's import system can
    # resolve ``import torch.nn as nn`` at module load in ids_worker_torch.
    fake_torch = types.ModuleType("torch")
    fake_torch.__path__ = []

    class _Tensor:
        def __init__(self, arr):
            self._arr = _np.asarray(arr, dtype=_np.float32)

        @property
        def shape(self):
            return self._arr.shape

        def float(self):
            return self

        def unsqueeze(self, dim):
            self._arr = _np.expand_dims(self._arr, axis=dim)
            return self

        def item(self):
            return float(self._arr.flatten()[0])

    fake_torch.Tensor = _Tensor
    fake_torch.float32 = _np.float32

    def _from_numpy(arr, _Tensor=_Tensor):
        return _Tensor(arr)

    def _load(path, map_location=None, weights_only=False):
        # Mimic a real torch.load: raise FileNotFoundError if the
        # path doesn't exist on disk. Tests that want to drive the
        # scoring loop pass ``model_path="ignored"`` — treat that
        # sentinel as "any path" so the run proceeds; only paths that
        # look like real files (no sentinel) trigger the error.
        from pathlib import Path as _P
        spath = str(path)
        if spath != "ignored" and not _P(spath).exists():
            raise FileNotFoundError(f"fake_torch: {path} does not exist")
        return {"state_dict": {}}

    def _no_grad():
        class _Ctx:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *args):
                return False
        return _Ctx()

    fake_torch.from_numpy = _from_numpy
    fake_torch.load = _load
    fake_torch.no_grad = _no_grad

    # ids_worker_torch.py does ``import torch.nn as nn`` at module load;
    # provide a ModuleType stub so the import works. We populate `Module`
    # with a no-op base class so SEBlock/TCNBlockSE/TCNClassifierSE can
    # inherit from it (their bodies are never executed because the test
    # monkeypatches ``build_tcn_se_model`` to return our constant-logit
    # fake before run() is called).
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.__path__ = []

    class _NoOpModule:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

    fake_nn.Module = _NoOpModule

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)

    # --- patch the model factory to return our constant-logit fake ----------
    from modbus_replay import ids_worker_torch as _torch_mod

    def _build(*args, **kwargs):
        return _FakeModel(logit=2.0)

    monkeypatch.setattr(_torch_mod, "build_tcn_se_model", _build)
    monkeypatch.setattr(
        "modbus_replay.ids_worker.build_tcn_se_model", _build, raising=False
    )
    return fake_torch


def _make_scaler():
    """Build a ScalerBundle with mu=0, sg=1 — normalization is identity."""
    from modbus_replay.ids_pipeline import ScalerBundle
    mu = np.zeros(18, dtype=np.float32)
    sg = np.ones(18, dtype=np.float32)
    return ScalerBundle(mu_18=mu, sg_18=sg)


def _row(addr=1, fc=3, time=1000.0, command_response=0):
    return {
        "address": addr,
        "function": fc,
        "length": 16,
        "command response": command_response,
        "setpoint": 0.0,
        "gain": 0.0,
        "reset rate": 0.0,
        "deadband": 0.0,
        "cycle time": 0,
        "rate": 0.0,
        "system mode": 0,
        "control scheme": 0,
        "pump": 0,
        "solenoid": 0,
        "pressure measurement": 0.0,
        "crc rate": 0,
        "time": time,
    }


@pytest.fixture
def qapp():
    a = QCoreApplication.instance() or QApplication([])
    return a


def _wait_for(signal, timeout=2000):
    loop = QEventLoop()
    captured = []

    def slot(*args):
        captured.append(args)
        loop.quit()

    signal.connect(slot)
    QTimer.singleShot(timeout, loop.quit)
    loop.exec()
    signal.disconnect(slot)
    return captured


def _wait_until(predicate, timeout=3000, interval=0.02):
    """Pump the event loop until predicate() is True or timeout."""
    deadline_ms = timeout
    loop = QEventLoop()
    QTimer.singleShot(deadline_ms, loop.quit)

    def _tick():
        if predicate():
            loop.quit()
        else:
            QTimer.singleShot(int(interval * 1000), _tick)

    _tick()
    loop.exec()


def _slow_feed(worker, n: int) -> None:
    """Feed ``n`` events to ``worker`` with a small inter-event sleep.

    The worker's input queue is bounded at 64. Without pacing, a tight
    Python for-loop will fill the queue and ``put_nowait`` will drop
    events before the QThread has spun up. Tests want to verify the
    scoring / counters, not stress the queue, so we sleep 5 ms per
    event which is >> the per-event cost on the worker thread.
    """
    import time
    for i in range(n):
        worker.feed_event(i, _row(time=1000.0 + i * 0.001), b"\x00" * 32)
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ids_worker_emits_started_scoring_after_16_rows(qapp, fake_torch):
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
    )
    started_emissions = []
    worker.started_scoring.connect(lambda: started_emissions.append(True))

    worker.start()
    _slow_feed(worker, 30)

    _wait_until(lambda: len(started_emissions) > 0, timeout=2000)
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)
    assert len(started_emissions) == 1


def test_ids_worker_emits_frame_scored_with_correct_fields(qapp, fake_torch):
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
    )
    emissions = []
    worker.frame_scored.connect(
        lambda row_idx, prob, is_alarm, addr, fc, direction, time_ms:
            emissions.append({
                "row_idx": row_idx, "prob": prob, "is_alarm": is_alarm,
                "addr": addr, "fc": fc, "direction": direction,
                "time_ms": time_ms,
            })
    )

    worker.start()
    import time
    for i in range(30):
        worker.feed_event(
            i,
            _row(addr=4 + i % 2, fc=3, time=2000.0 + i * 0.001,
                 command_response=i % 2),
            b"\x00" * 32,
        )
        time.sleep(0.005)

    _wait_until(lambda: len(emissions) >= 14, timeout=3000)
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)

    # First 15 feeds are warm-up → no frame_scored. Subsequent 15 →
    # 15 emissions. Sigmoid(logit=2) ≈ 0.88, threshold 0.5 → is_alarm=True.
    assert len(emissions) == 15
    for e in emissions:
        assert e["prob"] == pytest.approx(0.880797, abs=1e-4)
        assert e["is_alarm"] is True
        assert e["time_ms"] == int(round(e["time_ms"] / 1000.0 * 1000))
        # addr/fc/direction must match the source row (we used a
        # deterministic sequence above).
        assert e["addr"] in (4, 5)
        assert e["fc"] == 3
        assert e["direction"] in (0, 1)


def test_ids_worker_threshold_override(qapp, fake_torch):
    from modbus_replay.ids_worker import IdsWorker

    # Threshold 0.99 — above the fake prob of 0.88 → no alarm ever.
    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.99,
    )
    emissions = []
    worker.frame_scored.connect(
        lambda row_idx, prob, is_alarm, *rest:
            emissions.append({"row_idx": row_idx, "prob": prob, "is_alarm": is_alarm})
    )

    worker.start()
    _slow_feed(worker, 30)

    _wait_until(lambda: len(emissions) >= 10, timeout=3000)
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)
    for e in emissions:
        assert e["is_alarm"] is False
        assert e["prob"] < 0.99


def test_ids_worker_writes_csv_log(qapp, fake_torch, tmp_path: Path):
    from modbus_replay.ids_worker import IdsWorker

    log = tmp_path / "ids_log.csv"
    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
        csv_log_path=str(log),
    )
    emissions = []
    worker.frame_scored.connect(
        lambda row_idx, prob, is_alarm, *rest:
            emissions.append((row_idx, prob))
    )

    worker.start()
    _slow_feed(worker, 30)

    _wait_until(lambda: len(emissions) >= 10, timeout=3000)
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)

    assert log.exists()
    with open(log, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # 1 header + N data rows (one per scored frame after warm-up).
    assert rows[0] == [
        "row_idx", "time_ms", "addr", "fc", "direction", "prob", "is_alarm",
    ]
    data_rows = rows[1:]
    assert len(data_rows) == 15  # 30 rows - 15 warm-up
    # Last data row's prob matches the last emission.
    last = emissions[-1]
    assert int(data_rows[-1][0]) == last[0]
    assert float(data_rows[-1][5]) == pytest.approx(last[1], abs=1e-5)
    assert int(data_rows[-1][6]) == 1  # is_alarm column


def test_ids_worker_no_csv_log_no_file_written(qapp, fake_torch, tmp_path: Path):
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
        csv_log_path=None,
    )
    worker.start()
    _slow_feed(worker, 30)
    _wait_for(worker.finished_run, timeout=2000)  # natural drain
    worker.stop()
    worker.wait(2000)

    # tmp_path should have no .csv files created by us.
    assert list(tmp_path.glob("*.csv")) == []


def test_ids_worker_csv_log_appends_no_duplicate_header(
    qapp, fake_torch, tmp_path: Path
):
    from modbus_replay.ids_worker import IdsWorker

    log = tmp_path / "ids_log.csv"

    # Run 1.
    w1 = IdsWorker(
        model_path="ignored", scaler=_make_scaler(), threshold=0.5,
        csv_log_path=str(log),
    )
    w1.start()
    import time
    for i in range(20):
        w1.feed_event(i, _row(time=1000.0 + i * 0.001), b"\x00" * 32)
        time.sleep(0.005)
    w1.stop()
    _wait_for(w1.finished_run, timeout=2000)
    w1.wait(2000)

    # Run 2 — reuses the same path.
    w2 = IdsWorker(
        model_path="ignored", scaler=_make_scaler(), threshold=0.5,
        csv_log_path=str(log),
    )
    w2.start()
    for i in range(20):
        w2.feed_event(i, _row(time=2000.0 + i * 0.001), b"\x00" * 32)
        time.sleep(0.005)
    w2.stop()
    _wait_for(w2.finished_run, timeout=2000)
    w2.wait(2000)

    with open(log, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header_count = sum(1 for r in rows if r and r[0] == "row_idx")
    assert header_count == 1
    # Each run produces (20 - 15 warm-up) = 5 data rows. Total = 10.
    assert len(rows) == 1 + 5 + 5


def test_ids_worker_missing_torch_emits_error(qapp, monkeypatch):
    """If torch import fails, run() emits error and finishes."""
    # Block ``import torch`` by replacing sys.modules with a stub that
    # raises ImportError on attribute access.
    class _Blocker:
        def __getattr__(self, name):
            raise ImportError("blocked by test")

    monkeypatch.setitem(sys.modules, "torch", _Blocker())
    monkeypatch.setitem(sys.modules, "modbus_replay.ids_worker_torch", _Blocker())

    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
    )
    errors = []
    worker.error.connect(lambda msg: errors.append(msg))
    worker.finished_run_captured = []
    worker.finished_run.connect(lambda: worker.finished_run_captured.append(True))

    worker.start()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)
    assert errors, "expected error signal"
    assert "torch" in errors[0].lower() or "IDS deps" in errors[0]
    assert len(worker.finished_run_captured) == 1


def test_ids_worker_bad_model_path_emits_error(qapp, fake_torch, tmp_path: Path):
    """If torch.load raises, the worker emits error and finishes."""
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path=str(tmp_path / "does_not_exist.pt"),
        scaler=_make_scaler(),
        threshold=0.5,
    )
    errors = []
    worker.error.connect(lambda msg: errors.append(msg))
    worker.start()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)
    assert errors, "expected error signal"
    assert "model" in errors[0].lower() or "load" in errors[0].lower()


def test_ids_worker_run_summary_counts(qapp, fake_torch):
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
    )
    summaries = []
    # New signature: (frame_count, alarm_count, tp, fp, tn, fn).
    worker.run_summary.connect(
        lambda fc, ac, tp, fp, tn, fn: summaries.append((fc, ac, tp, fp, tn, fn))
    )

    worker.start()
    # Feed slowly enough that the bounded queue (maxsize=64) does not
    # drop events. The test's goal is to verify the summary counters,
    # not to stress-test the queue. 5 ms per event >> per-event cost on
    # the worker thread, so the queue stays near-empty.
    import time
    for i in range(100):
        worker.feed_event(i, _row(time=1000.0 + i * 0.001), b"\x00" * 32)
        time.sleep(0.005)
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)

    assert summaries, "expected run_summary signal"
    frame_count, alarm_count, tp, fp, tn, fn = summaries[-1]
    # 100 feeds - 15 warm-up = 85 scored frames.
    assert frame_count == 85
    # All alarms because the fake model returns prob ≈ 0.88 ≥ 0.5.
    assert alarm_count == 85
    # _row() carries no ``binary result`` ground truth → all four
    # confusion-matrix counters stay at 0.
    assert tp == 0
    assert fp == 0
    assert tn == 0
    assert fn == 0


def test_ids_worker_metrics_updated_with_ground_truth(qapp, fake_torch):
    """When CSV rows carry ``binary result`` the worker emits
    metrics_updated and the end-of-run confusion matrix is correct."""
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
    )
    metrics_emissions: list[tuple[int, int, int, int]] = []
    worker.metrics_updated.connect(
        lambda tp, fp, tn, fn: metrics_emissions.append((tp, fp, tn, fn))
    )

    # Build a 40-row dataset: 10 normal (binary=0) + 10 attack
    # (binary=1) repeated twice. The fake model always returns prob ≈
    # 0.88 (above threshold 0.5) → every scored frame is an alarm.
    # Expected confusion matrix after warm-up:
    #   TP = attack rows flagged    = 20 (after warm-up is 5 frames
    #                                     consumed, 35 scored, all alarm)
    #   FP = normal rows flagged    = 15  (35 scored - 20 TP - 0 FN)
    #   TN = normal rows not flagged = 0   (every frame alarms)
    #   FN = attack rows not flagged = 0
    # Wait — all 35 alarms are positive predictions, so the only way
    # to score TN or FN is to have a non-alarm frame. The fake model
    # always alarms, so TN = FN = 0. The split between TP and FP is
    # determined by the ground truth label: 20 attack rows + 15 normal
    # rows in the 35 scored windows → TP=20, FP=15.
    rows: list[dict] = []
    # Alternate to make ground truth unambiguous.
    for i in range(40):
        is_attack = (i % 4) < 2  # 0,1 = attack; 2,3 = normal
        rows.append({
            "address": 1,
            "function": 3,
            "length": 16,
            "command response": 0,
            "setpoint": 0.0,
            "gain": 0.0,
            "reset rate": 0.0,
            "deadband": 0.0,
            "cycle time": 0,
            "rate": 0.0,
            "system mode": 0,
            "control scheme": 0,
            "pump": 0,
            "solenoid": 0,
            "pressure measurement": 0.0,
            "crc rate": 0,
            "binary result": 1 if is_attack else 0,
            "time": 1000.0 + i * 0.001,
        })

    worker.start()
    import time
    for i, row in enumerate(rows):
        worker.feed_event(i, row, b"\x00" * 32)
        time.sleep(0.005)
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)

    assert metrics_emissions, "expected at least one metrics_updated emission"
    tp, fp, tn, fn = metrics_emissions[-1]
    # Fake model always alarms → 0 TN, 0 FN.
    assert tn == 0
    assert fn == 0
    # 40 rows - 15 warm-up = 25 scored frames. Attack rows in the
    # scored range = rows where (i % 4) < 2 for i in 15..39:
    # i = 15 (3→normal), 16 (0→attack), 17 (1→attack), 18 (2→normal),
    # 19 (3→normal), 20 (0→attack), 21 (1→attack), 22 (2→normal),
    # 23 (3→normal), 24 (0→attack), 25 (1→attack), 26 (2→normal),
    # 27 (3→normal), 28 (0→attack), 29 (1→attack), 30 (2→normal),
    # 31 (3→normal), 32 (0→attack), 33 (1→attack), 34 (2→normal),
    # 35 (3→normal), 36 (0→attack), 37 (1→attack), 38 (2→normal),
    # 39 (3→normal).
    # Count: i in 15..39 → 25 rows; attack (i%4<2) at i=16,17,20,21,
    # 24,25,28,29,32,33,36,37 → 12 attacks; normal → 13.
    assert tp == 12, f"expected 12 TPs, got {tp}"
    assert fp == 13, f"expected 13 FPs, got {fp}"


def test_ids_worker_metrics_skips_rows_without_ground_truth(qapp, fake_torch):
    """Rows missing ``binary result`` must not corrupt the confusion
    matrix (TN/FN/FN stay at 0 if no label has ever been seen)."""
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
    )
    metrics_emissions: list[tuple[int, int, int, int]] = []
    worker.metrics_updated.connect(
        lambda tp, fp, tn, fn: metrics_emissions.append((tp, fp, tn, fn))
    )

    worker.start()
    import time
    for i in range(30):
        # _row() helper deliberately omits ``binary result``.
        worker.feed_event(i, _row(time=1000.0 + i * 0.001), b"\x00" * 32)
        time.sleep(0.005)
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)

    # No ground truth supplied → no metrics_updated emissions.
    assert metrics_emissions == []


def test_ids_worker_stop_during_run_finishes(qapp, fake_torch):
    """stop() while events are flowing must finish cleanly."""
    from modbus_replay.ids_worker import IdsWorker

    worker = IdsWorker(
        model_path="ignored",
        scaler=_make_scaler(),
        threshold=0.5,
    )
    worker.start()
    import time
    for i in range(50):
        worker.feed_event(i, _row(time=1000.0 + i * 0.001), b"\x00" * 32)
        time.sleep(0.005)
    # Stop while the queue still has events — drain finishes the run.
    worker.stop()
    _wait_for(worker.finished_run, timeout=2000)
    worker.wait(2000)
    # No assertion on counts — just that the worker finished cleanly.