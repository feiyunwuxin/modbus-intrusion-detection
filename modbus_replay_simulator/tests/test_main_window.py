"""Smoke tests for MainWindow assembly.

The plan defers deep MainWindow coverage to the integration test in
Task 9 and the manual checklist in Task 10; this module only asserts
that the window composes cleanly under Qt and exposes the documented
controls.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from modbus_replay.gui.main_window import MainWindow
from modbus_replay.gui.widgets import (
    LogPanel,
    PortSelector,
    ProgressPanel,
)


@pytest.fixture
def qapp():
    a = QApplication.instance() or QApplication([])
    return a


def test_main_window_constructs(qapp):
    win = MainWindow()
    assert win.windowTitle() == "Modbus Replay Simulator"
    # The three core widgets must be present and parented to the window.
    assert isinstance(win.port_selector, PortSelector)
    assert isinstance(win.progress_panel, ProgressPanel)
    assert isinstance(win.log_panel, LogPanel)


def test_main_window_initial_button_state(qapp):
    win = MainWindow()
    assert win.start_btn.isEnabled() is True
    assert win.pause_btn.isEnabled() is False
    assert win.stop_btn.isEnabled() is False


def test_main_window_csv_missing_emits_error(qapp, tmp_path, monkeypatch):
    """When Start is clicked with a non-existent CSV, the log gets ERROR."""
    win = MainWindow()
    win.csv_edit.setText(str(tmp_path / "does_not_exist.csv"))
    # No port selected yet, but the missing-CSV branch fires first.
    win._on_start()
    text = win.log_panel.text_edit.toPlainText()
    assert "ERROR" in text
    assert "not found" in text.lower()


def test_main_window_csv_present_no_port(qapp, tmp_path):
    """CSV exists but no COM port -> second error branch."""
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "time,address,function,length,setpoint,gain,reset,deadband,cycle,"
        "rate,system_mode,control_scheme,pump,solenoid,pressure,response,"
        "crc_rate\n"
        "1000,1,3,8,0,0,0,0,0,0.0,0,0,0,0,0.0,0,0\n",
        encoding="utf-8",
    )
    win = MainWindow()
    win.csv_edit.setText(str(csv))
    # No port selected on an empty system: PortSelector currentText() == "".
    win._on_start()
    text = win.log_panel.text_edit.toPlainText()
    assert "ERROR" in text
