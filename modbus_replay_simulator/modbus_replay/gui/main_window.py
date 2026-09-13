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

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from modbus_replay.csv_loader import load_rows
from modbus_replay.gui.widgets import (
    LogPanel,
    PortSelector,
    ProgressPanel,
)
from modbus_replay.serial_worker import SerialWorker

DEFAULT_DATASET = r"C:\work\Claude\Issue\IanArffDataset.csv"


class MainWindow(QMainWindow):
    """Top-level window: PortSelector + CSV picker + controls + log."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Modbus Replay Simulator")
        self.resize(720, 560)
        self._worker: SerialWorker | None = None
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
        self.csv_btn = QPushButton("\U0001f4c2 选择 CSV\u2026")
        csv_row.addWidget(self.csv_edit, stretch=1)
        csv_row.addWidget(self.csv_btn)
        layout.addLayout(csv_row)

        # Run controls
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("\u25b6 开始")
        self.pause_btn = QPushButton("\u23f8 暂停")
        self.stop_btn = QPushButton("\u23f9 停止")
        self.loop_chk = QCheckBox("循环播放")
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.pause_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.loop_chk)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        # Progress + log
        self.progress_panel = ProgressPanel()
        layout.addWidget(self.progress_panel)

        self.log_panel = LogPanel()
        layout.addWidget(self.log_panel, stretch=1)

        # Initial control state — only Start is enabled until a run starts.
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _wire(self) -> None:
        self.csv_btn.clicked.connect(self._pick_csv)
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

    def _on_start(self) -> None:
        csv_path = self.csv_edit.text().strip()
        if not csv_path or not Path(csv_path).exists():
            self.log_panel.append("ERROR", f"CSV not found: {csv_path}")
            return

        try:
            rows = load_rows(csv_path)
        except Exception as e:
            self.log_panel.append("ERROR", f"load failed: {e}")
            return

        params = self.port_selector.current_params()
        if not params["port"]:
            self.log_panel.append("ERROR", "no COM port selected")
            return

        self.log_panel.append("INFO", f"CSV loaded: {len(rows)} rows")
        self.progress_panel.set_total(len(rows))
        self.progress_panel.reset()

        self._worker = SerialWorker(
            rows=rows,
            port=params["port"],
            baudrate=params["baudrate"],
            databits=params["databits"],
            parity=params["parity"],
            stopbits=params["stopbits"],
            loop_mode=self.loop_chk.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.state_changed.connect(self._on_state)
        self._worker.log_message.connect(self.log_panel.append)
        self._worker.finished_run.connect(self._on_finished)

        self._set_running_ui(True)
        self._worker.start()

    def _on_pause(self) -> None:
        if self._worker is None:
            return
        # Toggle between paused and running via the button label.
        if self.pause_btn.text().startswith("\u23f8"):
            self._worker.pause()
            self.pause_btn.setText("\u25b6 继续")
        else:
            self._worker.resume()
            self.pause_btn.setText("\u23f8 暂停")

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
        self.log_panel.append("INFO", f"state \u2192 {state}")
        if state in ("finished", "stopped", "error"):
            self._set_running_ui(False)

    def _on_finished(self) -> None:
        # Reset pause button label after a finished/stopped run.
        if self.pause_btn.text() != "\u23f8 暂停":
            self.pause_btn.setText("\u23f8 暂停")

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        # Stop the worker cleanly so the serial port is not orphaned.
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(ev)


def main() -> int:
    """Entry point used by ``__main__.py`` and the console script."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()
