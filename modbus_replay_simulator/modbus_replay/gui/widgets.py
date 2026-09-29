"""Reusable Qt widgets for the Modbus Replay Simulator GUI.

This module exposes three independent widgets designed to be composed
into the MainWindow by Task 7:

* :class:`PortSelector` — port + 4 serial parameters (baud/databits/
  parity/stopbits) with a refresh button; emits ``port_changed`` /
  ``params_changed`` whenever the user mutates the selection.

* :class:`ProgressPanel` — progress bar + current-row label +
  elapsed/ETA label driven by an internal QTimer.

* :class:`LogPanel` — read-only QTextEdit that prefixes each entry
  with a timestamp and a level tag.

PyQt5 is imported at module top. Callers that want graceful failure
on systems without the GUI stack should guard with
``pytest.importorskip("PyQt5")`` (as the test modules do) before
importing anything from this file.
"""
from __future__ import annotations

import datetime
import time

from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
DATA_BITS = [7, 8]
PARITIES = ["N", "E", "O"]
STOP_BITS = [1, 2]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_serial_ports() -> list[str]:
    """Return available serial port device names.

    Falls back to an empty list if pyserial's listing utility is not
    importable (e.g. running on a stripped CI image).
    """
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    try:
        return [p.device for p in list_ports.comports()]
    except Exception:
        return []


def _fmt_hms(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` (clamped to non-negative)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# PortSelector
# ---------------------------------------------------------------------------


class PortSelector(QWidget):
    """Five-field serial-port configuration widget.

    Signals
    -------
    port_changed(str)
        Emitted whenever the user picks a different COM port (or after
        a refresh that drops the previous selection).
    params_changed(dict)
        Emitted whenever *any* of baud/databits/parity/stopbits
        changes. The dict matches :meth:`current_params`.
    """

    port_changed = pyqtSignal(str)
    params_changed = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._refresh_ports()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Combos
        self.port_combo = QComboBox()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems([str(b) for b in BAUD_RATES])
        self.baud_combo.setCurrentText("115200")

        self.databits_combo = QComboBox()
        self.databits_combo.addItems([str(b) for b in DATA_BITS])
        self.databits_combo.setCurrentText("8")

        self.parity_combo = QComboBox()
        self.parity_combo.addItems(PARITIES)
        self.parity_combo.setCurrentText("N")

        self.stopbits_combo = QComboBox()
        self.stopbits_combo.addItems([str(b) for b in STOP_BITS])
        self.stopbits_combo.setCurrentText("1")

        refresh_btn = QPushButton("\U0001f504 刷新")
        refresh_btn.clicked.connect(self._refresh_ports)

        # Lay out
        layout.addWidget(QLabel("COM:"))
        layout.addWidget(self.port_combo)
        layout.addWidget(QLabel("Baud:"))
        layout.addWidget(self.baud_combo)
        layout.addWidget(QLabel("Data:"))
        layout.addWidget(self.databits_combo)
        layout.addWidget(QLabel("Parity:"))
        layout.addWidget(self.parity_combo)
        layout.addWidget(QLabel("Stop:"))
        layout.addWidget(self.stopbits_combo)
        layout.addWidget(refresh_btn)

        # Signal wiring
        self.port_combo.currentTextChanged.connect(self.port_changed.emit)
        for w in (
            self.baud_combo,
            self.databits_combo,
            self.parity_combo,
            self.stopbits_combo,
        ):
            w.currentTextChanged.connect(self._emit_params)

    # -- Public API -------------------------------------------------------

    def current_params(self) -> dict:
        """Snapshot of the five serial parameters as a dict."""
        return {
            "port": self.port_combo.currentText(),
            "baudrate": int(self.baud_combo.currentText()),
            "databits": int(self.databits_combo.currentText()),
            "parity": self.parity_combo.currentText(),
            "stopbits": int(self.stopbits_combo.currentText()),
        }

    def set_baudrate(self, value: int) -> None:
        """Programmatically change the baud selection.

        ``params_changed`` is emitted at most once (zero times if the
        value is unchanged; one time otherwise, via the connected
        ``currentTextChanged`` → ``_emit_params`` wiring).
        """
        text = str(value)
        if self.baud_combo.currentText() == text:
            return
        # setCurrentText emits currentTextChanged → _emit_params on a
        # real change; do NOT also call _emit_params here (double emit).
        self.baud_combo.setCurrentText(text)

    def _emit_params(self, *_args) -> None:
        self.params_changed.emit(self.current_params())

    def _refresh_ports(self) -> None:
        """Re-scan the OS for serial ports, preserving the current pick."""
        prev = self.port_combo.currentText()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        self.port_combo.addItems(_list_serial_ports())
        if prev and prev in [
            self.port_combo.itemText(i) for i in range(self.port_combo.count())
        ]:
            self.port_combo.setCurrentText(prev)
        self.port_combo.blockSignals(False)
        # Always re-emit so listeners (e.g. MainWindow) can sync state.
        self.port_changed.emit(self.port_combo.currentText())


# ---------------------------------------------------------------------------
# ProgressPanel
# ---------------------------------------------------------------------------


class ProgressPanel(QWidget):
    """Current-row label + elapsed/ETA label.

    The horizontal progress bar that used to live above these labels
    was removed — CSV import now uses its own modal QProgressDialog,
    and the persistent bar was visual clutter that crept along over
    multi-hour replays. The two labels remain because they show
    operator-useful info (most recent protocol header, wall-clock pace)
    that the verbose TX log line per frame already prints in detail.

    Driven by :class:`PyQt5.QtCore.QTimer` (500 ms tick) — the timer is
    started lazily on the first ``set_progress`` call so the ETA
    remains stable while the user configures the port.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total = 0
        self._run_start: float | None = None
        self._last_cur = 0

        layout = QVBoxLayout(self)

        self.current_row_label = QLabel("-")
        self.elapsed_label = QLabel("00:00:00 / 00:00:00")

        layout.addWidget(self.current_row_label)
        layout.addWidget(self.elapsed_label)

        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._refresh_elapsed)

    # -- Public API -------------------------------------------------------

    def set_total(self, total: int) -> None:
        """Set the upper bound used to compute ETA.

        Kept on the API surface because ``MainWindow._on_start`` still
        calls it — the bound is needed for the elapsed/ETA math even
        though no progress bar reads it any more.
        """
        self._total = max(0, int(total))
        self._refresh_elapsed()

    def set_progress(self, cur: int, total: int = -1) -> None:
        """Update the current row position (used for ETA + tick start).

        Starts the elapsed/ETA timer on the first non-zero update. The
        ``total`` arg is accepted for parity with the old interface but
        the bound is governed by ``set_total``.
        """
        self._last_cur = int(cur)
        if self._run_start is None and self._last_cur > 0:
            self._run_start = time.monotonic()
            self._tick.start()

    def set_current_row_info(
        self, row_idx: int, addr: int, fc: int, direction: int
    ) -> None:
        """Display the row index + the protocol header fields.

        ``direction=1`` is rendered as ``resp`` (response from slave);
        ``direction=0`` as ``cmd`` (command from master).
        """
        d = "resp" if direction else "cmd"
        self.current_row_label.setText(
            f"row={row_idx} addr={addr} fc={fc} {d}"
        )

    def reset(self) -> None:
        """Stop the timer and zero out the panel (for the next run)."""
        self._run_start = None
        self._tick.stop()
        self._last_cur = 0
        self.elapsed_label.setText("00:00:00 / 00:00:00")
        self.current_row_label.setText("-")

    # -- Internal ---------------------------------------------------------

    def _refresh_elapsed(self) -> None:
        if self._run_start is None:
            return
        elapsed = time.monotonic() - self._run_start
        cur = self._last_cur
        if cur > 0 and elapsed > 0:
            rate = cur / elapsed
            remaining = (self._total - cur) / rate if rate > 0 else 0
        else:
            remaining = 0
        self.elapsed_label.setText(
            f"{_fmt_hms(elapsed)} / {_fmt_hms(remaining)} (ETA)"
        )


# ---------------------------------------------------------------------------
# LogPanel
# ---------------------------------------------------------------------------


class LogPanel(QWidget):
    """Read-only, timestamped, level-tagged text view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def append(self, level: str, message: str) -> None:
        """Append one log line, e.g. ``[12:34:56] [INFO] hello``.

        Auto-scrolls to the bottom so the most recent entry is always
        visible without manual scrolling during long runs.
        """
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.text_edit.append(f"[{ts}] [{level}] {message}")
        # Move cursor to end and ensure it's visible — without this the
        # QTextEdit stays at the top during long multi-hour replays.
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()


# ---------------------------------------------------------------------------
# RxPanel — receives bytes from the serial port and renders them as
# hex+ASCII with timestamps. Used to inspect what the MCU sends back.
# ---------------------------------------------------------------------------


def _format_hex_ascii(data: bytes) -> str:
    """Render bytes as ``HH HH HH | ccc`` (hex on left, ASCII on right).

    Each 16-byte row is laid out as a fixed-width block. Non-printable
    ASCII bytes are rendered as ``.``.
    """
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset:offset + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        # Pad hex part to 16*3-1 = 47 chars for aligned ASCII column.
        hex_part = hex_part.ljust(16 * 3 - 1)
        ascii_part = "".join(
            chr(b) if 32 <= b < 127 else "." for b in chunk
        )
        lines.append(f"{offset:04x}  {hex_part}  |{ascii_part}|")
    return "\n".join(lines)


def _format_hex_only(data: bytes) -> str:
    """Render bytes as space-separated hex pairs, no offsets or ASCII."""
    return " ".join(f"{b:02x}" for b in data)


def _format_ascii_only(data: bytes) -> str:
    """Render bytes as ASCII (non-printable as ``.``)."""
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data)


class RxPanel(QWidget):
    """Hex+ASCII receive viewer with a clear button.

    Use :meth:`append_bytes` from any thread (safe across thread
    boundaries because the underlying QTextEdit accepts append() via
    Qt's queued connection when wired through a signal).
    """

    # Emitted when the user clicks the 🗑 clear button.
    cleared = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        # Toolbar row: clear button + format combo + byte counter
        toolbar = QHBoxLayout()
        self.clear_btn = QPushButton("\U0001f5d1 清空")
        self.clear_btn.clicked.connect(self._on_clear)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["hex+ASCII", "hex", "ASCII"])
        self.byte_count_label = QLabel("0 bytes")
        toolbar.addWidget(self.clear_btn)
        toolbar.addWidget(QLabel("格式:"))
        toolbar.addWidget(self.format_combo)
        toolbar.addStretch(1)
        toolbar.addWidget(self.byte_count_label)
        outer.addLayout(toolbar)

        # Read-only, monospace text view
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        font = self.text_edit.font()
        font.setFamily("Consolas, Courier New, monospace")
        self.text_edit.setFont(font)
        outer.addWidget(self.text_edit, stretch=1)

        self._byte_count = 0

    def _on_clear(self) -> None:
        self.text_edit.clear()
        self._byte_count = 0
        self.byte_count_label.setText("0 bytes")
        self.cleared.emit()

    def append_bytes(self, data: bytes) -> None:
        """Append one chunk of received bytes.

        Each call renders as a single block, prefixed by a timestamp and
        bracketed by blank lines so bursts from the MCU stay grouped.
        """
        if not data:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # ms
        fmt = self.format_combo.currentText()
        if fmt == "hex":
            body = _format_hex_only(data)
        elif fmt == "ASCII":
            body = _format_ascii_only(data)
        else:
            body = _format_hex_ascii(data)
        # ``append`` inserts a new paragraph — keeps each chunk visually
        # grouped without manually inserting newlines.
        self.text_edit.append(f"[{ts}] RX ({len(data)} B):\n{body}")
        self._byte_count += len(data)
        self.byte_count_label.setText(f"{self._byte_count} bytes")

        # Auto-scroll to the latest block.
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()


# ---------------------------------------------------------------------------
# IdsPanel — PC-side intrusion-detection monitor.
# ---------------------------------------------------------------------------
#
# Taps the SerialWorker.tx_event signal (via MainWindow wiring), runs
# each emitted frame through the 23-dim TCN+SE model on a dedicated
# worker thread, and surfaces the per-frame probability as KPIs +
# alarm-log rows. The user picks the .pt / .joblib / .json files each
# session — paths are deliberately NOT persisted via QSettings so the
# panel can be re-purposed against a different model without rebooting
# the GUI.
#
# No torch / numpy import here: the panel only consumes signals from
# the worker and renders state. The model itself lives in
# ``modbus_replay.ids_worker`` (lazy import inside its run()).


#: Maximum number of alarm rows retained in the on-screen table.
#: Older rows are dropped from the top. The CSV log (if enabled)
#: captures every scored frame for offline analysis.
_ALARM_LOG_MAX_ROWS: int = 200


class IdsPanel(QWidget):
    """File-pickers + KPIs + alarm-log table for the PC-side IDS.

    Signals
    -------
    enabled_changed(bool)
        User toggled the "启用 IDS" checkbox.
    model_path_changed(str), scaler_path_changed(str),
    threshold_path_changed(str), csv_log_path_changed(str)
        User picked one of the four files. ``MainWindow`` stores the
        path so the next ``_on_start`` has the latest picks even if
        the panel is rebuilt.

    Public slots
    ------------
    on_frame_scored(row_idx, prob, is_alarm, addr, fc, direction, time_ms)
        Updates KPI labels and (if ``is_alarm``) appends a row to the
        alarm-log table.
    on_started_scoring()
        Flips the status label from "warming up" to "scoring".
    on_run_summary(frame_count, alarm_count)
        End-of-run totals (no-op on the panel — the log panel surfaces
        the same numbers).
    on_error(message)
        Shows the error in the status label.

    The panel must be enabled (``checkbox`` ticked + ``[ids]`` extra
    installed + all three required files picked) **and** the
    MainWindow must have spawned an :class:`IdsWorker` for the data
    to start flowing. The checkbox is the user's opt-in; the actual
    "is the IDS running?" state is decided by :attr:`isEnabled`.
    """

    enabled_changed = pyqtSignal(bool)
    model_path_changed = pyqtSignal(str)
    scaler_path_changed = pyqtSignal(str)
    threshold_path_changed = pyqtSignal(str)
    csv_log_path_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Track the user's selections so file pickers can be cleared
        # on demand and ``current_*_path`` returns None before pick.
        self._model_path: str | None = None
        self._scaler_path: str | None = None
        self._threshold_path: str | None = None
        self._csv_log_path: str | None = None

        # KPI counters — incremented by on_frame_scored, zeroed by reset().
        self._scored: int = 0
        self._alarms: int = 0
        self._window_filled: int = 0  # for the "window: N/16" label
        # Confusion-matrix counters — advanced by on_metrics_updated.
        # Stay at 0 until the worker sees a row with a ``binary result``
        # ground-truth label.
        self._tp: int = 0
        self._fp: int = 0
        self._tn: int = 0
        self._fn: int = 0

        self._build_ui()
        self._wire()
        self._refresh_status_label()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        # Row 1 — enable + file pickers
        row1 = QHBoxLayout()
        self.enable_chk = QCheckBox("🛡 启用 IDS")
        row1.addWidget(self.enable_chk)
        row1.addWidget(QLabel("文件:"))
        self.model_btn = QPushButton("📁 模型 (.pt)…")
        self.scaler_btn = QPushButton("📁 Scaler (.joblib)…")
        self.threshold_btn = QPushButton("📁 Threshold (.json)…")
        row1.addWidget(self.model_btn)
        row1.addWidget(self.scaler_btn)
        row1.addWidget(self.threshold_btn)
        row1.addStretch(1)
        outer.addLayout(row1)

        # Path display row (read-only line edits so the user can see the
        # current picks at a glance; copyable for debugging).
        row1b = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        self.model_edit.setPlaceholderText("(no model)")
        self.scaler_edit = QLineEdit()
        self.scaler_edit.setReadOnly(True)
        self.scaler_edit.setPlaceholderText("(no scaler)")
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setReadOnly(True)
        self.threshold_edit.setPlaceholderText("(no threshold)")
        row1b.addWidget(self.model_edit, stretch=1)
        row1b.addWidget(self.scaler_edit, stretch=1)
        row1b.addWidget(self.threshold_edit, stretch=1)
        outer.addLayout(row1b)

        # Row 2 — threshold override + CSV log
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("报警阈值:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setValue(0.5)  # overwritten by _on_threshold_path_set
        self.threshold_spin.setToolTip(
            "Read once when 开始 is clicked. Mid-run changes are ignored."
        )
        row2.addWidget(self.threshold_spin)
        row2.addSpacing(20)
        self.csv_chk = QCheckBox("📝 记录每帧 prob 到 CSV")
        row2.addWidget(self.csv_chk)
        self.csv_btn = QPushButton("📁…")
        self.csv_btn.setMaximumWidth(40)
        row2.addWidget(self.csv_btn)
        self.csv_edit = QLineEdit()
        self.csv_edit.setReadOnly(True)
        self.csv_edit.setPlaceholderText("(no CSV log)")
        row2.addWidget(self.csv_edit, stretch=1)
        outer.addLayout(row2)

        # Row 3 — count KPIs (scored / alarm / alarm-rate)
        row3 = QHBoxLayout()
        self.scored_label = QLabel("已评分: 0")
        self.alarm_label = QLabel("报警数: 0")
        self.rate_label = QLabel("报警率: 0.00%")
        for lbl in (self.scored_label, self.alarm_label, self.rate_label):
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
        row3.addWidget(self.scored_label)
        row3.addWidget(self.alarm_label)
        row3.addWidget(self.rate_label)
        row3.addStretch(1)
        outer.addLayout(row3)

        # Row 3b — Binary classification metrics. Per the IDS Binary
        # convention (see feedback-ids-binary-precision-convention.md),
        # an IDS evaluation table is incomplete without these four.
        # They are computed live from the running confusion matrix
        # maintained by IdsWorker; absent ground truth the labels stay
        # at "—" so the operator can tell the difference between "no
        # data yet" and "all zeros".
        row3b = QHBoxLayout()
        self.accuracy_label = QLabel("准确率 Acc: —")
        self.precision_label = QLabel("精确率 P: —")
        self.recall_label = QLabel("召回率 R: —")
        self.f1_label = QLabel("F1: —")
        for lbl in (self.accuracy_label, self.precision_label,
                    self.recall_label, self.f1_label):
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
        row3b.addWidget(self.accuracy_label)
        row3b.addWidget(self.precision_label)
        row3b.addWidget(self.recall_label)
        row3b.addWidget(self.f1_label)
        row3b.addStretch(1)
        outer.addLayout(row3b)

        # Row 4 — alarm log table
        self.alarm_table = QTableWidget(0, 6)
        self.alarm_table.setHorizontalHeaderLabels(
            ["row_idx", "time_ms", "addr", "fc", "direction", "prob"]
        )
        self.alarm_table.verticalHeader().setVisible(False)
        self.alarm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.alarm_table.setSelectionBehavior(QTableWidget.SelectRows)
        hdr = self.alarm_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        outer.addWidget(self.alarm_table, stretch=1)

        # Row 5 — status label
        self.status_label = QLabel("model: not loaded  /  window: 0/16")
        outer.addWidget(self.status_label)

        # File pickers start disabled — they only enable when the
        # user ticks the IDS enable checkbox. The enable checkbox
        # itself starts unchecked (the user must opt in each session).
        self._set_pickers_enabled(False)

    def _wire(self) -> None:
        self.enable_chk.toggled.connect(self._on_enable_toggled)
        self.model_btn.clicked.connect(self._pick_model)
        self.scaler_btn.clicked.connect(self._pick_scaler)
        self.threshold_btn.clicked.connect(self._pick_threshold)
        self.csv_btn.clicked.connect(self._pick_csv_log)
        self.threshold_path_changed.connect(self._on_threshold_path_set)

    # ------------------------------------------------------------------
    # File picker handlers
    # ------------------------------------------------------------------

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 IDS 模型 (.pt)",
            "",
            "PyTorch checkpoints (*.pt *.pth);;All files (*.*)",
        )
        if path:
            self._model_path = path
            self.model_edit.setText(path)
            self.model_path_changed.emit(path)

    def _pick_scaler(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Scaler (.joblib)",
            "",
            "Joblib files (*.joblib);;All files (*.*)",
        )
        if path:
            self._scaler_path = path
            self.scaler_edit.setText(path)
            self.scaler_path_changed.emit(path)

    def _pick_threshold(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Threshold (.json)",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if path:
            self._threshold_path = path
            self.threshold_edit.setText(path)
            self.threshold_path_changed.emit(path)

    def _pick_csv_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择 IDS CSV 日志",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self._csv_log_path = path
            self.csv_edit.setText(path)
            self.csv_log_path_changed.emit(path)

    def _on_threshold_path_set(self, _path: str) -> None:
        """Pre-fill the spinbox from the threshold JSON.

        Failures are silent — the user can type a value manually.
        The actual ``load_threshold`` call lives in ``MainWindow`` so
        the panel does not need numpy/joblib to render.
        """
        try:
            from modbus_replay.ids_pipeline import load_threshold
            bundle = load_threshold(self._threshold_path)
            self.threshold_spin.setValue(bundle.best_threshold)
        except Exception:
            # Leave the spinbox at its current value; the user can
            # override. The next reset() will not touch it.
            pass

    def _on_enable_toggled(self, checked: bool) -> None:
        self._set_pickers_enabled(checked)
        self.enabled_changed.emit(checked)

    def _set_pickers_enabled(self, on: bool) -> None:
        """Enable/disable the file pickers + threshold/CSV widgets.

        Does NOT touch the enable checkbox itself — that stays live so
        the user can toggle the IDS off mid-run if they want.
        """
        for w in (
            self.model_btn,
            self.scaler_btn,
            self.threshold_btn,
            self.threshold_spin,
            self.csv_chk,
            self.csv_btn,
        ):
            w.setEnabled(on)

    def set_file_pickers_enabled(self, on: bool) -> None:
        """MainWindow calls this around ``_on_start`` / ``_on_finished``.

        We lock the file/threshold/CSV widgets during a run so the user
        can't swap a model mid-flight, but leave the enable checkbox
        alone so they can still toggle the IDS off.
        """
        # Only meaningful when the IDS is enabled; when disabled the
        # pickers are already off via _set_pickers_enabled.
        if self.enable_chk.isChecked():
            for w in (
                self.model_btn,
                self.scaler_btn,
                self.threshold_btn,
                self.threshold_spin,
                self.csv_chk,
                self.csv_btn,
            ):
                w.setEnabled(on)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return bool(self.enable_chk.isChecked())

    def current_threshold(self) -> float:
        return float(self.threshold_spin.value())

    def current_model_path(self) -> str | None:
        return self._model_path

    def current_scaler_path(self) -> str | None:
        return self._scaler_path

    def current_threshold_path(self) -> str | None:
        return self._threshold_path

    def current_csv_log_path(self) -> str | None:
        return self._csv_log_path

    def csv_log_enabled(self) -> bool:
        return bool(self.csv_chk.isChecked()) and self._csv_log_path is not None

    def reset(self) -> None:
        """Zero KPIs, clear alarm table, reset window counter.

        Called by MainWindow between runs. File paths / threshold
        spinbox are deliberately preserved — the user picked them
        deliberately and we don't want them to re-pick between runs.
        """
        self._scored = 0
        self._alarms = 0
        self._window_filled = 0
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        self.scored_label.setText("已评分: 0")
        self.alarm_label.setText("报警数: 0")
        self.rate_label.setText("报警率: 0.00%")
        self.accuracy_label.setText("准确率 Acc: —")
        self.precision_label.setText("精确率 P: —")
        self.recall_label.setText("召回率 R: —")
        self.f1_label.setText("F1: —")
        self.alarm_table.setRowCount(0)
        self._refresh_status_label()

    # ------------------------------------------------------------------
    # Slots — called from IdsWorker signals.
    # ------------------------------------------------------------------

    @pyqtSlot(int, float, bool, int, int, int, int)
    def on_frame_scored(
        self,
        row_idx: int,
        prob: float,
        is_alarm: bool,
        addr: int,
        fc: int,
        direction: int,
        time_ms: int,
    ) -> None:
        """Per-frame score from :class:`IdsWorker`.

        Always increments the scored counter; only inserts a row in
        the alarm-log table when ``is_alarm`` is True. The table is
        capped at :data:`_ALARM_LOG_MAX_ROWS` rows — older rows are
        dropped from the top so the most recent alarms stay visible
        during a long 274k-row replay.
        """
        self._scored += 1
        if is_alarm:
            self._alarms += 1
            self._append_alarm_row(row_idx, time_ms, addr, fc, direction, prob)
        rate = (self._alarms / self._scored) * 100.0 if self._scored else 0.0
        self.scored_label.setText(f"已评分: {self._scored}")
        self.alarm_label.setText(f"报警数: {self._alarms}")
        self.rate_label.setText(f"报警率: {rate:.2f}%")

    @pyqtSlot()
    def on_started_scoring(self) -> None:
        self._window_filled = 16
        self._refresh_status_label()

    @pyqtSlot(int, int, int, int, int, int)
    def on_run_summary(
        self,
        frame_count: int,
        alarm_count: int,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ) -> None:
        """End-of-run totals.

        The live KPIs already reflect the last ``on_frame_scored`` /
        ``on_metrics_updated`` call; here we just stamp the final
        confusion matrix onto the panel so the labels stay correct
        even if a metrics update was dropped mid-shutdown.
        """
        self._tp = tp
        self._fp = fp
        self._tn = tn
        self._fn = fn
        self._refresh_metrics_labels()
        self._refresh_status_label(
            extra=(
                f" / run finished: scored={frame_count}, alarm={alarm_count}, "
                f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
            )
        )

    @pyqtSlot(int, int, int, int)
    def on_metrics_updated(
        self, tp: int, fp: int, tn: int, fn: int
    ) -> None:
        """Recompute Accuracy / Precision / Recall / F1 from the
        running confusion matrix. Fires once per scored frame whose
        CSV row carries a ``binary result`` ground-truth label."""
        self._tp = tp
        self._fp = fp
        self._tn = tn
        self._fn = fn
        self._refresh_metrics_labels()

    def _refresh_metrics_labels(self) -> None:
        """Format the 4 Binary-convention metric labels from the
        current confusion matrix. ``—`` indicates no ground truth yet
        (the worker has not seen a ``binary result`` column)."""
        total = self._tp + self._fp + self._tn + self._fn
        if total == 0:
            self.accuracy_label.setText("准确率 Acc: —")
            self.precision_label.setText("精确率 P: —")
            self.recall_label.setText("召回率 R: —")
            self.f1_label.setText("F1: —")
            return
        acc = (self._tp + self._tn) / total
        precision = (
            self._tp / (self._tp + self._fp) if (self._tp + self._fp) else 0.0
        )
        recall = (
            self._tp / (self._tp + self._fn) if (self._tp + self._fn) else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        self.accuracy_label.setText(f"准确率 Acc: {acc:.4f}")
        self.precision_label.setText(f"精确率 P: {precision:.4f}")
        self.recall_label.setText(f"召回率 R: {recall:.4f}")
        self.f1_label.setText(f"F1: {f1:.4f}")

    @pyqtSlot(str)
    def on_error(self, message: str) -> None:
        self._refresh_status_label(extra=f" / ERROR: {message}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_alarm_row(
        self,
        row_idx: int,
        time_ms: int,
        addr: int,
        fc: int,
        direction: int,
        prob: float,
    ) -> None:
        table = self.alarm_table
        # Cap: drop oldest row(s) from the top.
        if table.rowCount() >= _ALARM_LOG_MAX_ROWS:
            table.removeRow(0)
        table.insertRow(table.rowCount())
        items = [
            QTableWidgetItem(str(row_idx)),
            QTableWidgetItem(str(time_ms)),
            QTableWidgetItem(str(addr)),
            QTableWidgetItem(str(fc)),
            QTableWidgetItem("resp" if direction else "cmd"),
            QTableWidgetItem(f"{prob:.4f}"),
        ]
        for col, item in enumerate(items):
            table.setItem(table.rowCount() - 1, col, item)
        # Auto-scroll to the most-recent alarm.
        table.scrollToBottom()

    def _refresh_status_label(self, extra: str = "") -> None:
        model_str = (
            "model: loaded" if self._model_path else "model: not loaded"
        )
        window_str = (
            f"window: {self._window_filled}/16"
            if self._window_filled > 0
            else "window: 0/16 (warming up)"
        )
        self.status_label.setText(f"{model_str}  /  {window_str}{extra}")
