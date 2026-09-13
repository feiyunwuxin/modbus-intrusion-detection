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


def test_main_window_open_port_no_port_emits_error(qapp):
    """📡 打开串口 with no COM port selected emits ERROR."""
    win = MainWindow()
    win.port_selector.port_combo.clear()
    assert win.open_btn.isEnabled() is True
    assert win.close_btn.isEnabled() is False
    win._on_open_port()
    text = win.log_panel.text_edit.toPlainText()
    assert "ERROR" in text
    assert "no COM port" in text
    # Probe handle stays None; buttons stay in initial state.
    assert win._probe_serial is None
    assert win.open_btn.isEnabled() is True
    assert win.close_btn.isEnabled() is False


def test_main_window_close_port_when_not_open_emits_info(qapp):
    """🔌 关闭串口 when no probe is open logs INFO and stays no-op."""
    win = MainWindow()
    win._on_close_port()
    text = win.log_panel.text_edit.toPlainText()
    assert "INFO" in text
    assert "already closed" in text
    assert win._probe_serial is None


def test_main_window_open_close_probe_handle_lifecycle(qapp, monkeypatch):
    """Open assigns a probe handle; close releases it; both log INFO.

    We do not exercise a real pyserial.Serial against hardware; instead
    we stub the import inside ``_on_open_port`` with a fake object.
    """
    fake_handle = object()  # stand-in for serial.Serial()

    def fake_serial_factory(*args, **kwargs):
        return fake_handle

    # Patch the `import serial` lookup so ``serial.Serial`` is the fake.
    import sys
    import types
    fake_serial_mod = types.SimpleNamespace(Serial=fake_serial_factory)
    monkeypatch.setitem(sys.modules, "serial", fake_serial_mod)

    win = MainWindow()
    # Pick a port so the open() path proceeds past the no-port guard.
    # Clear first so the test is deterministic regardless of host ports.
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")
    win._on_open_port()

    assert win._probe_serial is fake_handle
    assert win.open_btn.isEnabled() is False
    assert win.close_btn.isEnabled() is True
    text = win.log_panel.text_edit.toPlainText()
    assert "opened COM_FAKE" in text

    # Now close — verifies the close() is called on the handle.
    closed = []
    class FakeClosable:
        def close(self_inner):
            closed.append(True)
    win._probe_serial = FakeClosable()
    win._on_close_port()
    assert closed == [True]
    assert win._probe_serial is None
    assert win.open_btn.isEnabled() is True
    assert win.close_btn.isEnabled() is False
    text = win.log_panel.text_edit.toPlainText()
    assert "port closed" in text


def test_main_window_open_idempotent_when_already_open(qapp, monkeypatch):
    """Clicking Open twice logs INFO on the second click and does not
    re-open the port (the existing handle is preserved)."""
    fake_handle = object()
    call_count = []
    def fake_factory(*a, **kw):
        call_count.append(1)
        return fake_handle
    import sys, types
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=fake_factory))

    win = MainWindow()
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")
    win._on_open_port()
    win._on_open_port()  # second click — should be a no-op
    assert len(call_count) == 1, "Serial() must only be called once"
    assert win._probe_serial is fake_handle


def test_main_window_verbose_chk_passed_to_worker(qapp, tmp_path, monkeypatch):
    """📋 详细日志 checkbox state is forwarded to SerialWorker.verbose_tx.

    Captures the kwargs passed to SerialWorker so we don't actually
    start a thread.
    """
    captured = {}
    class FakeWorker:
        # Stub signals/properties/methods that MainWindow touches after
        # constructing the worker. Signals must be connect()-able.
        class _Signal:
            def connect(self, *_a, **_kw):
                pass
        progress = _Signal()
        state_changed = _Signal()
        log_message = _Signal()
        finished_run = _Signal()

        def __init__(self, **kwargs):
            captured.update(kwargs)
        def start(self):
            pass

    # Patch the SerialWorker symbol imported into main_window's namespace.
    monkeypatch.setattr(
        "modbus_replay.gui.main_window.SerialWorker",
        FakeWorker,
    )

    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "address,function,length,setpoint,gain,reset rate,deadband,"
        "cycle time,rate,system mode,control scheme,pump,solenoid,"
        "pressure measurement,crc rate,command response,time\n"
        "1,3,8,0,0,0,0,0,0.0,0,0,0,0,0.0,0,0,1000\n",
        encoding="utf-8",
    )

    # 1. verbose checkbox OFF → verbose_tx=False
    win = MainWindow()
    win.csv_edit.setText(str(csv))
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")
    win.verbose_chk.setChecked(False)
    win._on_start()
    assert captured.get("verbose_tx") is False

    # 2. verbose checkbox ON → verbose_tx=True
    win.verbose_chk.setChecked(True)
    win._on_start()
    assert captured.get("verbose_tx") is True
