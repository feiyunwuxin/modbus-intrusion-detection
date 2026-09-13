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

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
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
    """Progress bar + current-row label + elapsed/ETA label.

    Driven by :class:`PyQt5.QtCore.QTimer` (500 ms tick) — the timer is
    started lazily on the first ``set_progress`` call so the ETA
    remains stable while the user configures the port.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total = 0
        self._run_start: float | None = None

        layout = QVBoxLayout(self)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.current_row_label = QLabel("-")
        self.elapsed_label = QLabel("00:00:00 / 00:00:00")

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.current_row_label)
        layout.addWidget(self.elapsed_label)

        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._refresh_elapsed)

    # -- Public API -------------------------------------------------------

    def set_total(self, total: int) -> None:
        """Set the upper bound for the progress bar (total row count)."""
        self._total = max(0, int(total))
        self.progress_bar.setMaximum(self._total)
        self.progress_bar.setValue(0)
        self._refresh_elapsed()

    def set_progress(self, cur: int, total: int = -1) -> None:
        """Update the current row position.

        Starts the elapsed/ETA timer on the first non-zero update. The
        ``total`` arg is accepted for parity with the documented
        interface but the upper bound is governed by ``set_total``.
        """
        self.progress_bar.setValue(int(cur))
        if self._run_start is None and int(cur) > 0:
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
        self.progress_bar.setValue(0)
        self.elapsed_label.setText("00:00:00 / 00:00:00")
        self.current_row_label.setText("-")

    # -- Internal ---------------------------------------------------------

    def _refresh_elapsed(self) -> None:
        if self._run_start is None:
            return
        elapsed = time.monotonic() - self._run_start
        cur = self.progress_bar.value()
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
