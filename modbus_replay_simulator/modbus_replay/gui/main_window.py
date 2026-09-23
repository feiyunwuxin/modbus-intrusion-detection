"""MainWindow — assembles widgets, owns SerialWorker, wires signals.

This is the composition layer: it imports the leaf widgets from
:mod:`modbus_replay.gui.widgets`, the CSV loader, and the SerialWorker
thread, then connects them so the GUI drives the replay loop end to
end. There is no business logic here — only signal/slot wiring and
input validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from modbus_replay.csv_loader import load_rows
from modbus_replay.gui.widgets import (
    LogPanel,
    PortSelector,
    ProgressPanel,
    RxPanel,
)
from modbus_replay.serial_worker import SerialReader, SerialWorker

DEFAULT_DATASET = r"C:\work\Claude\Issue\IanArffDataset.csv"


class MainWindow(QMainWindow):
    """Top-level window: PortSelector + CSV picker + controls + log."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Modbus Replay Simulator")
        self.resize(720, 560)
        self._worker: SerialWorker | None = None
        # Diagnostic probe handle — independent from SerialWorker so the
        # user can verify wiring + serial parameters before kicking off
        # a long replay. None when the port is not currently held by
        # this window.
        self._probe_serial = None
        # Background reader — polls the probe handle while it's open and
        # forwards received bytes to the RxPanel. None when no probe is
        # active.
        self._rx_reader = None
        self._build_ui()
        self._wire()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Serial selector (COM + baud + parity + ...)
        self.port_selector = PortSelector()
        layout.addWidget(self.port_selector)

        # CSV picker
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("CSV:"))
        self.csv_edit = QLineEdit(DEFAULT_DATASET)
        self.csv_btn = QPushButton("\U0001f4c2 选择 CSV…")
        csv_row.addWidget(self.csv_edit, stretch=1)
        csv_row.addWidget(self.csv_btn)
        layout.addLayout(csv_row)

        # Run controls
        ctrl_row = QHBoxLayout()
        self.open_btn = QPushButton("\U0001f4e1 打开串口")
        self.close_btn = QPushButton("\U0001f50c 关闭串口")
        self.start_btn = QPushButton("▶ 开始")
        self.pause_btn = QPushButton("⏸ 暂停")
        self.stop_btn = QPushButton("⏹ 停止")
        self.loop_chk = QCheckBox("循环播放")
        # Verbose TX log toggle — when on, every successfully written
        # frame shows up in the LogPanel as a one-line summary (row,
        # t-offset, addr, fc, direction, first 8 bytes hex). Defaults
        # to ON so the user immediately sees what the simulator is
        # actually transmitting; uncheck for a clean log on long 274k
        # runs.
        self.verbose_chk = QCheckBox("📋 显示发送日志")
        self.verbose_chk.setChecked(True)
        ctrl_row.addWidget(self.open_btn)
        ctrl_row.addWidget(self.close_btn)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.pause_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.loop_chk)
        ctrl_row.addWidget(self.verbose_chk)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        # Progress + log + rx
        self.progress_panel = ProgressPanel()
        layout.addWidget(self.progress_panel)

        self.log_panel = LogPanel()
        self.rx_panel = RxPanel()
        layout.addWidget(self.log_panel, stretch=1)
        layout.addWidget(self.rx_panel, stretch=1)

        # Initial control state.
        # - Open:    enabled (user can probe the port immediately)
        # - Close:   enabled only after a successful open
        # - Start:   enabled only after a successful probe-open, so the
        #            user is forced to verify wiring + serial params
        #            before kicking off a long replay. The check is
        #            also re-enforced inside _on_start (defense in depth)
        #            in case a button is enabled programmatically.
        # - Pause/Stop: enabled only while a replay is running
        self.close_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _wire(self) -> None:
        self.csv_btn.clicked.connect(self._pick_csv)
        self.open_btn.clicked.connect(self._on_open_port)
        self.close_btn.clicked.connect(self._on_close_port)
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn.clicked.connect(self._on_stop)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _pick_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV", "", "CSV files (*.csv)"
        )
        if path:
            self.csv_edit.setText(path)

    def _on_open_port(self) -> None:
        """Probe-open the selected serial port.

        Uses a dedicated pyserial.Serial handle that is independent of
        the SerialWorker used for replay. Lets the user verify wiring +
        serial parameters before kicking off a long replay.

        Idempotent: clicking Open while the probe is already open logs
        INFO and returns without re-opening.
        """
        if self._probe_serial is not None:
            self.log_panel.append("INFO", "port already open (probe)")
            return

        params = self.port_selector.current_params()
        if not params["port"]:
            self.log_panel.append("ERROR", "no COM port selected")
            return

        try:
            import serial
            self._probe_serial = serial.Serial(
                port=params["port"],
                baudrate=params["baudrate"],
                bytesize=params["databits"],
                parity=params["parity"],
                stopbits=params["stopbits"],
                timeout=1,
            )
        except Exception as e:
            self.log_panel.append(
                "ERROR",
                f"open failed: {e}",
            )
            self._probe_serial = None
            return

        self.log_panel.append(
            "INFO",
            f"opened {params['port']} @ {params['baudrate']} "
            f"{params['databits']}{params['parity']}{params['stopbits']} (probe)",
        )
        self.open_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        # The probe is live — the user can now kick off a replay.
        # Pause/Stop stay disabled; they only enable once _on_start
        # starts the worker.
        self.start_btn.setEnabled(True)

        # Start the background reader so the RxPanel updates as data
        # arrives from the MCU.
        self._rx_reader = SerialReader(self._probe_serial)
        self._rx_reader.data_received.connect(self.rx_panel.append_bytes)
        self._rx_reader.start()

    def _on_close_port(self) -> None:
        """close the probe handle (no-op if not currently open)."""
        if self._probe_serial is None:
            self.log_panel.append("INFO", "port already closed (probe)")
            return
        # Stop the reader first so it does not try to read from a closed
        # handle and emit an error.
        if self._rx_reader is not None:
            self._rx_reader.stop()
            self._rx_reader.wait(1000)
            self._rx_reader = None
        try:
            self._probe_serial.close()
        except Exception as e:
            self.log_panel.append("WARN", f"close failed: {e}")
        finally:
            self._probe_serial = None
            self.open_btn.setEnabled(True)
            self.close_btn.setEnabled(False)
            # Probe handle is gone — Start must wait for the next open.
            # _on_start() re-enables it via the auto-close-then-reopen
            # path inside _on_start itself, so this is safe.
            self.start_btn.setEnabled(False)
            self.log_panel.append("INFO", "port closed (probe)")

    def _on_start(self) -> None:
        # Defensive: the UI normally disables Start when the probe is
        # not open, but programmatic invocation should still hit this
        # guard so we never silently kick off a replay against a port
        # the user has not verified.
        if self._probe_serial is None:
            self.log_panel.append(
                "ERROR",
                "serial port not open — click 📡 打开串口 first",
            )
            return

        csv_path = self.csv_edit.text().strip()
        if not csv_path or not Path(csv_path).exists():
            self.log_panel.append("ERROR", f"CSV not found: {csv_path}")
            return

        # Modal progress dialog while the (potentially slow) CSV load
        # runs. The row-by-row type coercion is the bottleneck on the
        # 274k-row dataset, hence the per-N-row callback.
        progress = QProgressDialog(
            "加载 CSV…", "", 0, 0, self
        )
        progress.setWindowTitle("加载 CSV")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # show immediately
        progress.show()
        QApplication.processEvents()

        def _on_progress(cur: int, total: int) -> None:
            # The (0, total) call from load_rows lands here first —
            # use it to size the bar. Subsequent calls advance the bar.
            progress.setMaximum(total)
            progress.setValue(cur)
            progress.setLabelText(
                f"加载 CSV: {cur}/{total} 行 "
                f"({100 * cur / total:.1f}%)"
                if total else "加载 CSV…"
            )
            QApplication.processEvents()

        try:
            rows = load_rows(csv_path, progress_callback=_on_progress)
        except Exception as e:
            progress.close()
            self.log_panel.append("ERROR", f"load failed: {e}")
            return

        progress.close()

        params = self.port_selector.current_params()
        if not params["port"]:
            self.log_panel.append("ERROR", "no COM port selected")
            return

        self.log_panel.append("INFO", f"CSV loaded: {len(rows)} rows")
        self.progress_panel.set_total(len(rows))
        self.progress_panel.reset()

        # Windows serial ports are exclusive: if the probe handle is
        # still open the worker can't acquire the same COM port and
        # silently fails with PermissionError. Close the probe first
        # so the worker can open it cleanly.
        if self._probe_serial is not None:
            self.log_panel.append(
                "INFO",
                "closing probe before replay so the worker can open the port",
            )
            self._on_close_port()

        self._worker = SerialWorker(
            rows=rows,
            port=params["port"],
            baudrate=params["baudrate"],
            databits=params["databits"],
            parity=params["parity"],
            stopbits=params["stopbits"],
            loop_mode=self.loop_chk.isChecked(),
            verbose_tx=self.verbose_chk.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.state_changed.connect(self._on_state)
        self._worker.log_message.connect(self.log_panel.append)
        # Forward MCU replies from the worker's serial handle straight
        # into the RxPanel — without this the GUI would only see RX
        # data while the probe handle is open (it never is during
        # replay because we just closed it).
        self._worker.rx_received.connect(self.rx_panel.append_bytes)
        self._worker.finished_run.connect(self._on_finished)

        self._set_running_ui(True)
        self._worker.start()

    def _on_pause(self) -> None:
        if self._worker is None:
            return
        # Toggle between paused and running via the button label.
        if self.pause_btn.text().startswith("⏸"):
            self._worker.pause()
            self.pause_btn.setText("▶ 继续")
        else:
            self._worker.resume()
            self.pause_btn.setText("⏸ 暂停")

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def _on_progress(
        self,
        row_idx: int,
        total: int,
        addr: int,
        fc: int,
        direction: int,
    ) -> None:
        self.progress_panel.set_progress(row_idx, total)
        self.progress_panel.set_current_row_info(
            row_idx, addr, fc, direction
        )

    def _on_state(self, state: str) -> None:
        self.log_panel.append("INFO", f"state → {state}")
        if state in ("finished", "stopped", "error"):
            self._set_running_ui(False)

    def _on_finished(self) -> None:
        # Reset pause button label after a finished/stopped run.
        if self.pause_btn.text() != "⏸ 暂停":
            self.pause_btn.setText("⏸ 暂停")

    # ------------------------------------------------------------------
    # UI state
    # ------------------------------------------------------------------
    def _set_running_ui(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        self.port_selector.setEnabled(not running)
        self.csv_edit.setEnabled(not running)
        self.csv_btn.setEnabled(not running)
        self.loop_chk.setEnabled(not running)
        self.verbose_chk.setEnabled(not running)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        # Stop the worker cleanly so the serial port is not orphaned.
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        # Stop the diagnostic RX reader if running.
        if self._rx_reader is not None:
            self._rx_reader.stop()
            self._rx_reader.wait(1000)
            self._rx_reader = None
        # Also release the diagnostic probe handle if the user opened
        # one and forgot to close it.
        if self._probe_serial is not None:
            try:
                self._probe_serial.close()
            except Exception:
                pass
            self._probe_serial = None
        super().closeEvent(ev)


def main() -> int:
    """Entry point used by ``__main__.py`` and the console script."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()
