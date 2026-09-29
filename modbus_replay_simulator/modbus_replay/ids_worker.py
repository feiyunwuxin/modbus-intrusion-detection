"""``IdsWorker(QThread)`` — runs the PC-side TCN+SE IDS on every TX frame.

Taps the ``tx_event`` signal emitted by :class:`SerialWorker`. Each
emission carries ``(row_idx, csv_row_dict, frame_bytes)`` — we feed
the CSV-row dict into the 18-dim featurizer
(:mod:`modbus_replay.ids_pipeline`), maintain a 16-frame sliding
window, and after warm-up run the (23, 16) window through the torch
model to produce a per-frame probability.

Threading model:
    * The TX pacing loop lives on :class:`SerialWorker` (its own thread).
    * This worker lives on its own thread; ``import torch`` happens
      here, **not** at module top, so the rest of the GUI imports
      cleanly without the ``[ids]`` extra.
    * ``feed_event`` is connected across threads via Qt's queued
      connection; it just pushes onto a bounded ``queue.Queue`` so a
      slow forward pass never blocks the TX pacing loop.

CSV logging: when ``csv_log_path`` is set, every scored frame
(``row_idx, time_ms, addr, fc, direction, prob, is_alarm``) is
appended. The header is only written if the file is empty at open
time, so re-runs append cleanly.
"""
from __future__ import annotations

import csv
import math
import os
import queue
from collections import deque
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from modbus_replay.ids_pipeline import (
    IncrementalFeatureState,
    ScalerBundle,
    SW_WINDOW,
    build_window_input,
    featurize_one,
)


#: Maximum number of unprocessed ``feed_event`` items allowed to queue
#: up before new emissions are dropped. Prevents a stalled IDS from
#: eating RAM or — worse — blocking the TX pacing loop.
_QUEUE_MAXSIZE: int = 64


class IdsWorker(QThread):
    """Streams ``tx_event`` payloads through the TCN+SE model.

    Signals
    -------
    started_scoring()
        Emitted exactly once, the first time the sliding window fills
        (after 16 frames). The GUI uses this to flip its status label
        from "warming up" to "scoring".
    frame_scored(int, float, bool, int, int, int, int)
        Per-frame payload ``(row_idx, prob, is_alarm, addr, fc,
        direction, time_ms)``. ``prob`` is in ``[0, 1]``; ``is_alarm``
        is ``prob >= self._threshold`` (the threshold the worker was
        constructed with).
    run_summary(int, int, int, int, int, int)
        End-of-run totals ``(frame_count, alarm_count, tp, fp, tn,
        fn)``. The four confusion-matrix counters are 0 when the run
        never saw a row with a ``binary result`` ground-truth label.
    error(str)
        Last-resort error report — model load failure, CSV open
        failure, queue overflow, etc. Goes to ``LogPanel`` as ``ERROR``.
    finished_run()
        Always fires at the end of ``run()``, even on error.
    """

    started_scoring = pyqtSignal()
    frame_scored = pyqtSignal(int, float, bool, int, int, int, int)
    # Confusion-matrix update — fires whenever a new scored frame can
    # be classified against a ground-truth label (binary result). The
    # four values are ``(tp, fp, tn, fn)`` for the running session.
    # Frames without a ``binary result`` in the CSV row do not
    # contribute (the confusion matrix is unchanged).
    metrics_updated = pyqtSignal(int, int, int, int)
    run_summary = pyqtSignal(int, int, int, int, int, int)
    error = pyqtSignal(str)
    finished_run = pyqtSignal()

    def __init__(
        self,
        model_path: str,
        scaler: ScalerBundle,
        threshold: float,
        *,
        csv_log_path: Optional[str | Path] = None,
        parent: Any = None,
        in_ch: int = 23,
        n_blocks: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        se_reduction: int = 8,
    ) -> None:
        super().__init__(parent)
        self._model_path = str(model_path)
        self._scaler = scaler
        self._threshold = float(threshold)
        self._csv_log_path: Optional[Path] = (
            Path(csv_log_path) if csv_log_path is not None else None
        )
        # Architecture kwargs — only used when the .pt is being built
        # from scratch (i.e. when the checkpoint is a bare state_dict).
        # Production checkpoints already carry the full module shape,
        # so these kwargs are mostly documentation + tests' convenience.
        self._arch_kwargs = dict(
            in_ch=in_ch,
            n_blocks=n_blocks,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            se_reduction=se_reduction,
        )

        # Cross-thread inbox. ``feed_event`` is called from
        # SerialWorker's thread; ``run()`` drains it on this thread.
        # Maxsize=64 keeps memory bounded if torch stalls.
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stopping: bool = False
        self._frame_count: int = 0
        self._alarm_count: int = 0
        self._first_window_done: bool = False
        # Confusion-matrix counters — only advance when the CSV row
        # carries a ground-truth ``binary result`` (1 = attack, 0 =
        # normal). Frames without it are skipped silently so unlabeled
        # data (e.g. live capture) doesn't corrupt the running totals.
        self._tp: int = 0
        self._fp: int = 0
        self._tn: int = 0
        self._fn: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """The active alarm threshold (read-only — set at construction)."""
        return self._threshold

    @pyqtSlot(int, dict, bytes)
    def feed_event(
        self,
        row_idx: int,
        csv_row: dict,
        frame_bytes: bytes,
    ) -> None:
        """Receive one TX event from :class:`SerialWorker` via queued conn.

        Pushes ``(row_idx, csv_row)`` onto an in-memory queue. The
        ``frame_bytes`` payload is accepted for signal-shape parity
        but is currently unused — the featurizer only consumes
        ``csv_row``.

        If the queue is full (torch forward pass is falling behind),
        emits ``error`` and **drops** the event — never blocks the
        TX pacing loop.
        """
        # Coerce csv_row to a plain dict in case a Qt-wrapped type came
        # across the thread boundary (rare in practice but cheap).
        row = dict(csv_row) if csv_row is not None else {}
        try:
            self._queue.put_nowait((int(row_idx), row))
        except queue.Full:
            self.error.emit("IDS queue full — dropping frame (TX pacing continues)")

    def stop(self) -> None:
        """Request graceful shutdown.

        Sets ``_stopping`` and pokes the queue with a sentinel so the
        run-loop wakes from its 100 ms ``queue.get(timeout=...)`` poll
        even if no more TX events arrive.
        """
        self._stopping = True
        try:
            self._queue.put_nowait((None, None))
        except queue.Full:
            # Already full — the run-loop will pick up _stopping on its
            # next 100 ms tick anyway.
            pass

    # ------------------------------------------------------------------
    # QThread.run
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901  (single-threaded orchestrator)
        """Build model + drain the queue.

        Returns early with an ``error.emit`` if torch is not installed
        or the model file fails to load. Always emits ``finished_run``
        on the way out so the GUI can release its reference.
        """
        try:
            import torch  # noqa: F401  (presence check)
            from modbus_replay.ids_worker_torch import build_tcn_se_model
        except ImportError as e:
            self.error.emit(
                f"torch / IDS deps not installed — run `pip install -e "
                f"\".[ids]\"` to enable PC-side IDS ({e})"
            )
            self.finished_run.emit()
            return

        # 1. Build + load the model on this thread so the GUI thread
        #    never imports torch.
        try:
            model = build_tcn_se_model(**self._arch_kwargs)
            ckpt = torch.load(self._model_path, map_location="cpu",
                              weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
            model.load_state_dict(sd)
            model.eval()
        except Exception as e:
            self.error.emit(f"failed to load model '{self._model_path}': {e}")
            self.finished_run.emit()
            return

        # 2. Open the CSV log (append mode). Header is only written if
        #    the file is currently empty so re-runs append cleanly.
        csv_fp = None
        csv_writer = None
        if self._csv_log_path is not None:
            try:
                self._csv_log_path.parent.mkdir(parents=True, exist_ok=True)
                is_new = (
                    not self._csv_log_path.exists()
                    or os.stat(self._csv_log_path).st_size == 0
                )
                csv_fp = open(
                    self._csv_log_path, "a", newline="", encoding="utf-8"
                )
                csv_writer = csv.writer(csv_fp)
                if is_new:
                    csv_writer.writerow(
                        [
                            "row_idx",
                            "time_ms",
                            "addr",
                            "fc",
                            "direction",
                            "prob",
                            "is_alarm",
                        ]
                    )
                    csv_fp.flush()
            except Exception as e:
                self.error.emit(f"failed to open CSV log: {e}")
                if csv_fp is not None:
                    try:
                        csv_fp.close()
                    except Exception:
                        pass
                csv_fp = None
                csv_writer = None

        # 3. Per-run state.
        feat_state = IncrementalFeatureState()
        window: deque = deque(maxlen=SW_WINDOW)

        # 4. Drain loop. We poll with a short timeout so ``stop()`` can
        #    interrupt even when no more events are arriving.
        try:
            import numpy as np
            import torch as _torch

            while True:
                try:
                    row_idx, csv_row = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._stopping:
                        break
                    continue
                if row_idx is None:
                    # Sentinel from stop().
                    break

                self._score_one(
                    row_idx=row_idx,
                    csv_row=csv_row,
                    window=window,
                    feat_state=feat_state,
                    model=model,
                    np=np,
                    torch=_torch,
                    csv_writer=csv_writer,
                )

        except Exception as e:
            # Catch-all so a bug in the per-row path can't crash the
            # whole worker silently. ``error.emit`` is the only way the
            # GUI sees the failure short of a hang.
            self.error.emit(f"IDS run failed: {e}")
        finally:
            self.run_summary.emit(
                self._frame_count, self._alarm_count,
                self._tp, self._fp, self._tn, self._fn,
            )
            if csv_fp is not None:
                try:
                    csv_fp.close()
                except Exception:
                    pass
            self.finished_run.emit()

    # ------------------------------------------------------------------
    # Per-row scoring — factored out so tests can drive it directly.
    # ------------------------------------------------------------------

    def _score_one(
        self,
        *,
        row_idx: int,
        csv_row: dict,
        window: deque,
        feat_state: IncrementalFeatureState,
        model: Any,
        np: Any,
        torch: Any,
        csv_writer: Any,
    ) -> None:
        """Score one row. Emits signals / writes CSV rows as appropriate.

        Raises no exceptions — wraps the torch forward in a try so a
        single bad row surfaces as ``error.emit`` instead of killing
        the worker.
        """
        try:
            feat18 = featurize_one(feat_state, csv_row)              # (18,)
            norm18 = self._scaler.normalize(
                feat18.reshape(1, 18)
            )                                                         # (1, 18)
            window.append(norm18[0])

            # Header fields straight from the CSV row so the GUI
            # doesn't need access to SerialWorker's row buffer.
            addr = _safe_int(csv_row.get("address"))
            fc = _safe_int(csv_row.get("function"))
            direction = _safe_int(csv_row.get("command response"))
            time_ms = int(round(_safe_float(csv_row.get("time")) * 1000))

            if len(window) < SW_WINDOW:
                # Warm-up — no score yet.
                return

            if not self._first_window_done:
                self._first_window_done = True
                self.started_scoring.emit()

            x = build_window_input(window)                            # (23, 16)
            xt = torch.from_numpy(np.asarray(x)).float().unsqueeze(0) # (1, 23, 16)
            with torch.no_grad():
                logit = float(model(xt).item())
            prob = 1.0 / (1.0 + math.exp(-logit))
            is_alarm = bool(prob >= self._threshold)

            self._frame_count += 1
            if is_alarm:
                self._alarm_count += 1

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        row_idx,
                        time_ms,
                        addr,
                        fc,
                        direction,
                        f"{prob:.6f}",
                        int(is_alarm),
                    ]
                )

            self.frame_scored.emit(
                row_idx, prob, is_alarm, addr, fc, direction, time_ms
            )

            # Confusion-matrix update — only when the ground-truth
            # ``binary result`` column is present and parses cleanly.
            # The reference convention (SCADA/IDS Binary metrics)
            # treats ``binary result == 1`` as "attack".
            gt_raw = csv_row.get("binary result")
            if gt_raw is not None and gt_raw != "?":
                try:
                    is_attack = int(float(gt_raw)) == 1
                except (TypeError, ValueError):
                    is_attack = None
                if is_attack is not None:
                    if is_alarm and is_attack:
                        self._tp += 1
                    elif is_alarm and not is_attack:
                        self._fp += 1
                    elif (not is_alarm) and (not is_attack):
                        self._tn += 1
                    else:
                        self._fn += 1
                    self.metrics_updated.emit(self._tp, self._fp, self._tn, self._fn)
        except Exception as e:
            self.error.emit(f"IDS row {row_idx} failed: {e}")


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None or v == "?":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "?":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


__all__ = ["IdsWorker"]