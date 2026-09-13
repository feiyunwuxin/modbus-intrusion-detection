"""Tests for GUI widgets (PortSelector, ProgressPanel, LogPanel).

Uses pytest-qt's `qtbot` fixture for QApplication lifecycle where
available; falls back to a manual QApplication instance otherwise.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication

from modbus_replay.gui.widgets import (
    LogPanel,
    PortSelector,
    ProgressPanel,
)


@pytest.fixture
def qapp():
    """Provide a single QApplication for the whole test session.

    Mirrors the pytest-qt convention (fixture named `qapp`) so this
    module coexists cleanly with the SerialWorker tests.
    """
    a = QApplication.instance() or QApplication([])
    return a


def test_port_selector_has_default_baud(qapp):
    sel = PortSelector()
    params = sel.current_params()
    assert params["baudrate"] == 115200
    assert params["databits"] == 8
    assert params["parity"] == "N"
    assert params["stopbits"] == 1


def test_port_selector_emits_params_changed(qapp):
    sel = PortSelector()
    captured = []
    sel.params_changed.connect(lambda p: captured.append(p))
    sel.set_baudrate(9600)
    assert captured, "params_changed signal was not emitted"
    assert captured[-1]["baudrate"] == 9600
    # Exactly one emission on a real change (not two — guards against a
    # previous double-emit bug where both setCurrentText's signal and an
    # explicit _emit_params call fired).
    assert len(captured) == 1


def test_port_selector_set_baudrate_no_change_emits_nothing(qapp):
    sel = PortSelector()
    captured = []
    sel.params_changed.connect(lambda p: captured.append(p))
    sel.set_baudrate(115200)  # already the default
    assert captured == []


def test_progress_panel_renders_values(qapp):
    panel = ProgressPanel()
    panel.set_total(100)
    panel.set_progress(50, 100)
    assert panel.progress_bar.value() == 50
    assert panel.progress_bar.maximum() == 100


def test_progress_panel_shows_current_row_info(qapp):
    panel = ProgressPanel()
    panel.set_current_row_info(123, addr=4, fc=3, direction=1)
    assert "123" in panel.current_row_label.text()
    assert "4" in panel.current_row_label.text()
    assert "3" in panel.current_row_label.text()


def test_log_panel_appends_messages(qapp):
    panel = LogPanel()
    panel.append("INFO", "hello")
    text = panel.text_edit.toPlainText()
    assert "hello" in text
    panel.append("ERROR", "boom")
    text = panel.text_edit.toPlainText()
    assert "ERROR" in text
    assert "boom" in text
