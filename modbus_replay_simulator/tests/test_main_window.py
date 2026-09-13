"""Smoke tests for MainWindow assembly.

The plan defers deep MainWindow coverage to the integration test in
Task 9 and the manual checklist in Task 10; this module only asserts
that the window composes cleanly under Qt and exposes the documented
controls.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
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
    """CSV exists but no COM port -> second error branch.

    The CSV header must use the same column names as ``REQUIRED_COLUMNS``
    in ``csv_loader.py`` (space-separated), otherwise ``load_rows`` will
    raise ValueError and route us into the wrong branch. We also
    explicitly clear the PortSelector combo so the test does not depend
    on whether the host happens to have any serial devices.
    """
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "address,function,length,setpoint,gain,reset rate,deadband,"
        "cycle time,rate,system mode,control scheme,pump,solenoid,"
        "pressure measurement,crc rate,command response,time\n"
        "1,3,8,0,0,0,0,0,0.0,0,0,0,0,0.0,0,0,1000\n",
        encoding="utf-8",
    )
    win = MainWindow()
    win.csv_edit.setText(str(csv))
    # Force the "no port selected" branch regardless of host hardware.
    win.port_selector.port_combo.clear()
    win._on_start()
    text = win.log_panel.text_edit.toPlainText()
    assert "ERROR" in text
    # Must specifically be the no-port branch, not the load-failed branch.
    assert "no COM port" in text
