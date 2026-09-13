"""Tests for RxPanel widget + SerialReader thread.

RxPanel tests cover the three display formats and clear-button behavior.
SerialReader tests use a fake serial handle that returns queued bytes
so we can verify the polling/emit loop deterministically.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QCoreApplication, QEventLoop, QThread, QTimer
from PyQt5.QtWidgets import QApplication

from modbus_replay.gui.main_window import _SerialReader
from modbus_replay.gui.widgets import (
    RxPanel,
    _format_ascii_only,
    _format_hex_ascii,
    _format_hex_only,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    a = QApplication.instance() or QApplication([])
    return a


def _wait_for(signal, timeout=2000):
    """Block until signal emitted, or timeout (ms). Returns args list."""
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


# ---------------------------------------------------------------------------
# Format helpers (pure)
# ---------------------------------------------------------------------------


def test_format_hex_ascii_alignment():
    out = _format_hex_ascii(b"ABCDEFGHIJKLMNOP")
    assert "41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50" in out
    assert "|ABCDEFGHIJKLMNOP|" in out
    # Offset prefix is hex 0000
    assert out.startswith("0000  ")


def test_format_hex_ascii_handles_short_chunk():
    out = _format_hex_ascii(b"\x01\x02\xff")
    assert "01 02 ff" in out
    # Non-printable bytes become '.' in ASCII column
    assert "|..\\x" not in out  # no literal escape sequences
    assert "|" in out


def test_format_hex_ascii_handles_multi_row():
    """Chunk spanning > 16 bytes yields multiple lines."""
    data = bytes(range(32))  # 32 bytes
    out = _format_hex_ascii(data)
    # Two lines (16 + 16)
    assert "\n" in out
    # Second line starts with offset 0x0010
    assert "0010  " in out


def test_format_hex_only():
    assert _format_hex_only(b"\x00\x01\xff") == "00 01 ff"


def test_format_ascii_only():
    assert _format_ascii_only(b"AB\x00\xffCD") == "AB..CD"


# ---------------------------------------------------------------------------
# RxPanel widget
# ---------------------------------------------------------------------------


def test_rx_panel_constructs(qapp):
    panel = RxPanel()
    assert panel.text_edit.isReadOnly() is True
    assert panel.byte_count_label.text() == "0 bytes"


def test_rx_panel_append_bytes_hex_ascii(qapp):
    panel = RxPanel()
    panel.append_bytes(b"\x01\x02hello")
    text = panel.text_edit.toPlainText()
    assert "RX (7 B)" in text
    assert "01 02 68 65 6c 6c 6f" in text  # hex bytes
    assert "|..hello|" in text  # ASCII column
    assert "7 bytes" in panel.byte_count_label.text()


def test_rx_panel_append_bytes_hex_only(qapp):
    panel = RxPanel()
    panel.format_combo.setCurrentText("hex")
    panel.append_bytes(b"\xde\xad\xbe\xef")
    text = panel.text_edit.toPlainText()
    assert "de ad be ef" in text
    # ASCII column should NOT be present in hex-only mode
    assert "|" not in text
    assert "4 bytes" in panel.byte_count_label.text()


def test_rx_panel_append_bytes_ascii_only(qapp):
    panel = RxPanel()
    panel.format_combo.setCurrentText("ASCII")
    panel.append_bytes(b"hi\xff")
    text = panel.text_edit.toPlainText()
    assert "hi." in text
    # No hex in ASCII-only mode
    assert "68 69" not in text


def test_rx_panel_byte_count_accumulates(qapp):
    panel = RxPanel()
    panel.append_bytes(b"abc")
    panel.append_bytes(b"defgh")
    assert panel.byte_count_label.text() == "8 bytes"


def test_rx_panel_empty_bytes_no_op(qapp):
    panel = RxPanel()
    panel.append_bytes(b"")
    assert panel.byte_count_label.text() == "0 bytes"
    assert panel.text_edit.toPlainText() == ""


def test_rx_panel_clear_button_resets(qapp):
    panel = RxPanel()
    panel.append_bytes(b"data")
    cleared = []
    panel.cleared.connect(lambda: cleared.append(True))
    panel.clear_btn.click()
    assert panel.text_edit.toPlainText() == ""
    assert panel.byte_count_label.text() == "0 bytes"
    assert cleared == [True]


# ---------------------------------------------------------------------------
# _SerialReader thread
# ---------------------------------------------------------------------------


class _FakeSerial:
    """Minimal stub of pyserial.Serial for SerialReader tests."""

    def __init__(self):
        self._buf = bytearray()
        self._raise_on_read = False

    def feed(self, data: bytes) -> None:
        """Queue bytes to be returned by the next read() call."""
        self._buf.extend(data)

    @property
    def in_waiting(self) -> int:
        return len(self._buf)

    def read(self, n: int):
        if self._raise_on_read:
            raise RuntimeError("read failed")
        chunk = bytes(self._buf[:n])
        del self._buf[:n]
        return chunk


def test_serial_reader_emits_received_bytes(qapp):
    fake = _FakeSerial()
    fake.feed(b"\x01\x02\x03")
    reader = _SerialReader(fake)
    captured = []
    reader.data_received.connect(lambda b: captured.append(b))
    reader.start()
    # Poll interval is 100ms; give the reader time to wake + read.
    QTimer.singleShot(300, reader.stop)
    # Drive a local event loop that quits when the reader thread emits
    # ``finished``. Without this loop the queued ``data_received`` signal
    # has nowhere to land and ``captured`` stays empty (QThread.wait()
    # blocks the GUI thread so Qt cannot deliver the queued slot).
    loop = QEventLoop()
    reader.finished.connect(loop.quit)
    QTimer.singleShot(2000, loop.quit)  # hard timeout fallback
    loop.exec()
    assert captured and captured[0] == b"\x01\x02\x03"


def test_serial_reader_emits_multiple_chunks(qapp):
    fake = _FakeSerial()
    reader = _SerialReader(fake)
    captured = []
    reader.data_received.connect(lambda b: captured.append(b))
    reader.start()
    # Feed bytes while the thread is alive; the reader should emit each.
    QTimer.singleShot(50, lambda: fake.feed(b"AAA"))
    QTimer.singleShot(150, lambda: fake.feed(b"BBBB"))
    QTimer.singleShot(400, reader.stop)
    loop = QEventLoop()
    reader.finished.connect(loop.quit)
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    assert b"AAA" in captured
    assert b"BBBB" in captured


def test_serial_reader_stop_exits_promptly(qapp):
    fake = _FakeSerial()
    reader = _SerialReader(fake)
    reader.start()
    time.sleep(0.05)
    reader.stop()
    # join within 1s — the polling loop checks _stopping each tick.
    assert reader.wait(1000) is True
