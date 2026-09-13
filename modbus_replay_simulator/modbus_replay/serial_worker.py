"""SerialWorker: QThread that owns the pyserial port and emits
progress/state/log signals.

Serial I/O is blocking; keeping it on a worker thread lets the GUI
stay responsive. The worker consumes a list of dataset rows
(loaded by csv_loader) and a ReplayEngine (built internally), pacing
each frame to the dataset's `time` column.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from modbus_replay.frame_format import FRAME_SIZE
from modbus_replay.replay_engine import ReplayEngine


class SerialWorker(QThread):
    # (row_idx, total_rows, address, function_code, direction)
    progress = pyqtSignal(int, int, int, int, int)
    # ("idle"|"running"|"paused"|"stopped"|"finished"|"error",)
    state_changed = pyqtSignal(str)
    # (level, message)
    log_message = pyqtSignal(str, str)
    finished_run = pyqtSignal()

    def __init__(self,
                 rows: list[dict],
                 port: str,
                 baudrate: int,
                 databits: int,
                 parity: str,
                 stopbits: int,
                 loop_mode: bool,
                 verbose_tx: bool = False,
                 serial_factory: Optional[Callable[[], object]] = None,
                 parent=None):
        super().__init__(parent)
        self._rows = rows
        self._port = port
        self._baudrate = baudrate
        self._databits = databits
        self._parity = parity
        self._stopbits = stopbits
        self._loop_mode = loop_mode
        # When True, emit a log_message per successfully written frame so
        # the GUI can show what each TX actually contained. Off by default
        # because 274k lines per run is too noisy for the log panel.
        self._verbose_tx = verbose_tx
        self._serial_factory = serial_factory or self._default_serial_factory

        self._paused = False
        self._stopping = False
        self._pause_cond = threading.Condition()
        self._serial = None

    # ------------------------------------------------------------------
    def _default_serial_factory(self):
        import serial  # local import so tests can run without pyserial
        return serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=self._databits,
            parity=self._parity,
            stopbits=self._stopbits,
            timeout=1,
        )

    # ------------------------------------------------------------------
    def pause(self):
        with self._pause_cond:
            self._paused = True
        self.state_changed.emit("paused")
        self.log_message.emit("INFO", "paused")

    def resume(self):
        with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()
        self.state_changed.emit("running")
        self.log_message.emit("INFO", "resumed")

    def stop(self):
        with self._pause_cond:
            self._stopping = True
            self._paused = False
            self._pause_cond.notify_all()
        self.log_message.emit("INFO", "stop requested")

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_stopping(self) -> bool:
        return self._stopping

    # ------------------------------------------------------------------
    def run(self):
        try:
            self._serial = self._serial_factory()
            self.log_message.emit(
                "INFO", f"opened {self._port} @ {self._baudrate} "
                        f"{self._databits}{self._parity}{self._stopbits}")
        except Exception as e:
            self.log_message.emit("ERROR", f"open failed: {e}")
            self.state_changed.emit("error")
            self.finished_run.emit()
            return

        self.state_changed.emit("running")
        engine = ReplayEngine(self._rows)
        total_rows = len(self._rows)

        try:
            while True:
                t_run_start = time.monotonic()
                base_time = float(self._rows[0]["time"])

                for i, frame in enumerate(
                    engine.encode_all(
                        time_offset_fn=lambda idx: int(
                            round((float(self._rows[idx]["time"])
                                   - base_time) * 1000))
                    )
                ):
                    if self._stopping:
                        break

                    with self._pause_cond:
                        while self._paused and not self._stopping:
                            self._pause_cond.wait()

                    if self._stopping:
                        break

                    # Pacing: sleep until the planned emit time
                    target_s = t_run_start + (
                        float(self._rows[i]["time"]) - base_time)
                    now_s = time.monotonic()
                    if target_s > now_s:
                        # msleep is non-blocking to Qt event loop
                        self.msleep(int((target_s - now_s) * 1000))

                    try:
                        self._serial.write(frame)
                    except Exception as e:
                        self.log_message.emit(
                            "WARN", f"write error at row {i}: {e}")
                        self.msleep(100)
                        try:
                            self._serial.write(frame)
                        except Exception as e2:
                            self.log_message.emit(
                                "ERROR", f"retry failed: {e2}")
                            self._stopping = True
                            break

                    addr = frame[0]
                    fc = frame[1]
                    direction = frame[3]
                    self.progress.emit(i, total_rows, addr, fc, direction)

                    # Optional verbose TX log — one line per frame with
                    # row index, protocol header, ms offset, and the first
                    # 8 bytes as hex. Useful for hardware debugging; off
                    # by default because it floods the log panel on full
                    # 274k-row runs.
                    if self._verbose_tx:
                        t_ms = int(round(
                            (float(self._rows[i]["time"]) - base_time) * 1000
                        ))
                        hex_head = " ".join(f"{b:02x}" for b in frame[:8])
                        d = "resp" if direction else "cmd"
                        self.log_message.emit(
                            "TX",
                            f"row={i} t={t_ms}ms addr={addr} fc={fc} "
                            f"{d} hex={hex_head}",
                        )

                if self._stopping:
                    break
                if not self._loop_mode:
                    break
                # else: loop

        except Exception as e:
            self.log_message.emit("ERROR", f"run failed: {e}")
            self.state_changed.emit("error")
        else:
            if self._stopping:
                self.log_message.emit("INFO", "run stopped")
                self.state_changed.emit("stopped")
            else:
                self.log_message.emit("INFO", "run finished")
                self.state_changed.emit("finished")
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self.finished_run.emit()
