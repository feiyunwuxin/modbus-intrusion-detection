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


class _FakeSerialHandle:
    """Stand-in for a pyserial.Serial handle.

    Just enough for tests that bypass the Open button and want to feed
    MainWindow a non-None ``_probe_serial`` so the Start guard does not
    short-circuit them. ``close()`` is a no-op so the auto-close path
    in ``_on_start`` (when a probe happens to be set up) does not raise.
    """
    def close(self):
        pass


def test_main_window_constructs(qapp):
    win = MainWindow()
    assert win.windowTitle() == "Modbus Replay Simulator"
    # The three core widgets must be present and parented to the window.
    assert isinstance(win.port_selector, PortSelector)
    assert isinstance(win.progress_panel, ProgressPanel)
    assert isinstance(win.log_panel, LogPanel)


def test_main_window_initial_button_state(qapp):
    """Initial button state — Start is **disabled** until the user has
    verified the port via Open. Forcing the Open-then-Start sequence is
    a deliberate UX gate; see ``test_main_window_start_requires_open``.
    """
    win = MainWindow()
    assert win.start_btn.isEnabled() is False
    assert win.pause_btn.isEnabled() is False
    assert win.stop_btn.isEnabled() is False


def test_main_window_csv_missing_emits_error(qapp, tmp_path, monkeypatch):
    """When Start is clicked with a non-existent CSV, the log gets ERROR.

    Probe is pre-opened via a stub so the new ``_probe_serial is None``
    guard does not fire first — this test is specifically about the
    CSV branch.
    """
    win = MainWindow()
    win._probe_serial = _FakeSerialHandle()  # bypass port-not-open guard
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
    win._probe_serial = _FakeSerialHandle()  # bypass port-not-open guard
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
    # Close so the background SerialReader QThread stops before the test
    # returns. Without this pytest-qt's qapp teardown waits on the
    # still-running reader and the suite hangs.
    win._on_close_port()


def test_main_window_verbose_chk_default_on(qapp):
    """📋 显示发送日志 defaults to checked so TX logs stream into the
    LogPanel immediately without the user having to tick the box."""
    win = MainWindow()
    assert win.verbose_chk.isChecked() is True


def test_main_window_start_requires_open(qapp, monkeypatch):
    """Clicking ▶ 开始 with no probe handle open must be a no-op
    (the button is disabled in the UI; this also covers programmatic
    invocation paths). The log gets an ERROR so the user knows why
    nothing happened."""
    # Stub out load_rows / QProgressDialog / SerialWorker so the test
    # doesn't go past the early guard even if the guard is removed.
    import modbus_replay.gui.main_window as mw
    class FakeWorker:
        class _Signal:
            def connect(self, *_a, **_kw):
                pass
        progress = _Signal(); state_changed = _Signal()
        log_message = _Signal(); rx_received = _Signal()
        finished_run = _Signal()
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def start(self):
            pass
    class FakeProgress:
        def __init__(self, *a, **kw):
            pass
        def setWindowTitle(self, *_a): pass
        def setCancelButton(self, *_a): pass
        def setWindowModality(self, *_a): pass
        def setMinimumDuration(self, *_a): pass
        def show(self): pass
        def close(self): pass
        def setMaximum(self, *_a): pass
        def setValue(self, *_a): pass
        def setLabelText(self, *_a): pass
    monkeypatch.setattr(mw, "QProgressDialog", FakeProgress)
    monkeypatch.setattr(mw, "SerialWorker", FakeWorker)

    win = MainWindow()
    # UI gate: Start button must be disabled before any open.
    assert win.start_btn.isEnabled() is False

    # Programmatic invocation must also be a no-op when probe is unset.
    win._on_start()
    text = win.log_panel.text_edit.toPlainText()
    assert "ERROR" in text
    assert "serial port not open" in text.lower()
    # Worker was never constructed.
    assert win._worker is None


def test_main_window_start_unlocks_after_open(qapp, monkeypatch):
    """After the user opens the probe, Start must enable. After the
    probe is closed again (auto-close-before-replay, manual close, or
    window close), Start must re-disable."""
    import modbus_replay.gui.main_window as mw

    # Stub SerialReader so _on_open_port can construct one without
    # touching real pyserial. The fake reader is a no-op thread.
    class FakeReader:
        def __init__(self, *a, **kw):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            pass
        def wait(self, *_a):
            pass
        class _Sig:
            def connect(self, *_a, **_kw):
                pass
        data_received = _Sig()

    # Stub the Serial(...) factory inside _on_open_port to a fake
    # handle so we don't need a real COM port.
    fake_handle = object()
    def fake_serial_factory(*a, **kw):
        return fake_handle
    import sys, types
    monkeypatch.setitem(
        sys.modules, "serial",
        types.SimpleNamespace(Serial=fake_serial_factory),
    )
    monkeypatch.setattr(mw, "SerialReader", FakeReader)

    win = MainWindow()
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")

    assert win.start_btn.isEnabled() is False
    win._on_open_port()
    assert win._probe_serial is fake_handle
    assert win.start_btn.isEnabled() is True

    # Closing the probe must re-disable Start (so the user can't kick
    # off a replay against a freshly-closed handle).
    win._on_close_port()
    assert win._probe_serial is None
    assert win.start_btn.isEnabled() is False


def test_main_window_csv_load_runs_progress_callback(qapp, tmp_path, monkeypatch):
    """_on_start forwards a progress callback into csv_loader.load_rows so
    the modal QProgressDialog actually moves during load.

    The dialog is replaced with a fake that records (cur, total) calls
    instead of opening a real modal. The default progress_every=1000
    means a 2500-row CSV produces 3 mid-loop updates (1000/2000/2500)
    plus the boundary (0, total) call.
    """
    csv = tmp_path / "tiny.csv"
    rows = ["1,3,8,?,?,?,?,?,?,?,?,?,?,?,?,0,100,0,0,0"] * 2500
    csv.write_text(
        "address,function,length,setpoint,gain,reset rate,deadband,"
        "cycle time,rate,system mode,control scheme,pump,solenoid,"
        "pressure measurement,crc rate,command response,time,"
        "binary result,categorized result,specific result\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    captured = {}
    class FakeWorker:
        class _Signal:
            def __init__(self):
                self.connections = []
            def connect(self, slot, *_a, **_kw):
                self.connections.append(slot)
        progress = _Signal()
        state_changed = _Signal()
        log_message = _Signal()
        rx_received = _Signal()
        finished_run = _Signal()

        def __init__(self, **kwargs):
            captured.update(kwargs)
        def start(self):
            pass

    # Fake QProgressDialog that just records the (cur, total) updates
    # the real dialog would render. No window is ever shown.
    calls = []
    class FakeProgress:
        def __init__(self, *a, **kw):
            pass
        def setWindowTitle(self, *_a): pass
        def setCancelButton(self, *_a): pass
        def setWindowModality(self, *_a): pass
        def setMinimumDuration(self, *_a): pass
        def show(self): pass
        def close(self): pass
        def setMaximum(self, v): calls.append(("max", v))
        def setValue(self, v): calls.append(("val", v))
        def setLabelText(self, v): calls.append(("label", v))

    import modbus_replay.gui.main_window as mw
    monkeypatch.setattr(mw, "QProgressDialog", FakeProgress)
    monkeypatch.setattr(mw, "SerialWorker", FakeWorker)

    win = MainWindow()
    win._probe_serial = _FakeSerialHandle()  # bypass port-not-open guard
    win.csv_edit.setText(str(csv))
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")
    win._on_start()

    # (0,total) sizes the bar, then 1000/2000 advance, then 2500 closes it.
    vals = [v for kind, v in calls if kind == "val"]
    assert 0 in vals
    assert 1000 in vals
    assert 2000 in vals
    assert 2500 in vals
    assert max(v for kind, v in calls if kind == "max") == 2500
    # SerialWorker received the loaded rows so the rest of the flow
    # proceeded past the load step.
    assert len(captured.get("rows", [])) == 2500
    # The worker.rx_received signal must be wired to rx_panel.append_bytes
    # — otherwise MCU replies during replay never reach the RxPanel.
    assert win.rx_panel.append_bytes in win._worker.rx_received.connections


def test_main_window_verbose_chk_passed_to_worker(qapp, tmp_path, monkeypatch):
    """📋 显示发送日志 checkbox state is forwarded to SerialWorker.verbose_tx.

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
        rx_received = _Signal()
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
    win._probe_serial = _FakeSerialHandle()  # bypass port-not-open guard
    win.csv_edit.setText(str(csv))
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")
    win.verbose_chk.setChecked(False)
    win._on_start()
    assert captured.get("verbose_tx") is False

    # 2. verbose checkbox ON → verbose_tx=True. ``_on_start`` auto-closes
    # the probe at the top of the flow so the worker can take the port;
    # re-arm the probe handle so the second call still passes the new
    # ``_probe_serial is None`` guard.
    win._probe_serial = _FakeSerialHandle()
    win.verbose_chk.setChecked(True)
    win._on_start()
    assert captured.get("verbose_tx") is True


def test_main_window_start_closes_probe_handle(qapp, tmp_path, monkeypatch):
    """Clicking ▶ 开始 with the probe port open must close the probe
    first — otherwise the SerialWorker can't acquire the same COM
    port on Windows (exclusive handles). The probe's reader is
    stopped and the handle released before the worker is constructed.
    """
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "address,function,length,setpoint,gain,reset rate,deadband,"
        "cycle time,rate,system mode,control scheme,pump,solenoid,"
        "pressure measurement,crc rate,command response,time\n"
        "1,3,8,0,0,0,0,0,0.0,0,0,0,0,0.0,0,0,1000\n",
        encoding="utf-8",
    )

    class FakeWorker:
        class _Signal:
            def connect(self, *_a, **_kw):
                pass
        progress = _Signal(); state_changed = _Signal()
        log_message = _Signal(); rx_received = _Signal()
        finished_run = _Signal()
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def start(self):
            pass

    # Stub the probe handle so we don't need real pyserial / hardware.
    reader_stopped = []
    handle_closed = []
    class FakeReader:
        def stop(self):
            reader_stopped.append(True)
        def wait(self, *_a):
            pass
    class FakeSerialHandle:
        def close(self):
            handle_closed.append(True)

    # Stub QProgressDialog so the load dialog is a no-op.
    class FakeProgress:
        def __init__(self, *a, **kw):
            pass
        def setWindowTitle(self, *_a): pass
        def setCancelButton(self, *_a): pass
        def setWindowModality(self, *_a): pass
        def setMinimumDuration(self, *_a): pass
        def show(self): pass
        def close(self): pass
        def setMaximum(self, *_a): pass
        def setValue(self, *_a): pass
        def setLabelText(self, *_a): pass

    import modbus_replay.gui.main_window as mw
    monkeypatch.setattr(mw, "QProgressDialog", FakeProgress)
    monkeypatch.setattr(mw, "SerialWorker", FakeWorker)
    monkeypatch.setattr(mw, "SerialReader", lambda *_a, **_kw: FakeReader())

    win = MainWindow()
    win.csv_edit.setText(str(csv))
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")

    # Simulate the probe already being open — that is exactly the
    # state that previously caused PermissionError on Start.
    win._probe_serial = FakeSerialHandle()
    win._rx_reader = FakeReader()

    win._on_start()

    assert reader_stopped == [True], "probe reader must be stopped"
    assert handle_closed == [True], "probe handle must be closed"
    assert win._probe_serial is None
    assert win._rx_reader is None
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
        rx_received = _Signal()
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
    win._probe_serial = _FakeSerialHandle()  # bypass port-not-open guard
    win.csv_edit.setText(str(csv))
    win.port_selector.port_combo.clear()
    win.port_selector.port_combo.addItem("COM_FAKE")
    win.verbose_chk.setChecked(False)
    win._on_start()
    assert captured.get("verbose_tx") is False

    # 2. verbose checkbox ON → verbose_tx=True. ``_on_start`` auto-closes
    # the probe at the top of the flow so the worker can take the port;
    # re-arm the probe handle so the second call still passes the new
    # ``_probe_serial is None`` guard.
    win._probe_serial = _FakeSerialHandle()
    win.verbose_chk.setChecked(True)
    win._on_start()
    assert captured.get("verbose_tx") is True
