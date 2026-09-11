import threading
import time
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer

from modbus_replay.serial_worker import SerialWorker
from modbus_replay.frame_format import FRAME_SIZE


class FakeSerial:
    """Mimics pyserial.Serial just enough for SerialWorker."""
    def __init__(self, *args, **kwargs):
        self.written = bytearray()
        self._raise_on_write = kwargs.pop("raise_on_write", None)
        self._closed = False

    def write(self, data):
        if self._closed:
            raise RuntimeError("port closed")
        if self._raise_on_write:
            raise self._raise_on_write
        self.written.extend(data)
        return len(data)

    def close(self):
        self._closed = True


def _row(t=1000):
    return {
        "address": 4, "function": 3, "length": 16,
        "command response": 1,
        "setpoint": "?", "gain": "?", "reset rate": "?",
        "deadband": "?", "cycle time": "?", "rate": "?",
        "system mode": "?", "control scheme": "?",
        "pump": "?", "solenoid": "?",
        "pressure measurement": "?",
        "crc rate": "?", "time": t,
    }


@pytest.fixture
def qapp():
    a = QCoreApplication.instance() or QCoreApplication([])
    return a


def _wait_for(signal, timeout=2000):
    """Block until signal emitted, or timeout (ms). Returns the args."""
    loop = QEventLoop()
    args_holder = []

    def slot(*args):
        args_holder.append(args)
        loop.quit()

    signal.connect(slot)
    QTimer.singleShot(timeout, loop.quit)
    loop.exec()
    signal.disconnect(slot)
    return args_holder[-1] if args_holder else None


def test_serial_worker_writes_expected_bytes(qapp):
    rows = [_row(1000 + i * 0.001) for i in range(5)]
    fake = FakeSerial()
    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=lambda: fake,
    )
    worker.start()
    assert _wait_for(worker.finished_run, timeout=3000) is not None
    worker.wait(2000)
    # 5 rows × 32 bytes = 160 bytes
    assert len(fake.written) == 5 * FRAME_SIZE


def test_serial_worker_emits_progress(qapp):
    rows = [_row(1000 + i) for i in range(3)]
    fake = FakeSerial()
    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=lambda: fake,
    )
    worker.start()
    # last progress should reach row=3
    last = None
    deadline = time.time() + 3
    while time.time() < deadline:
        last = _wait_for(worker.progress, timeout=500)
        if last and last[0] == 2:
            break
    assert last is not None and last[0] == 2
    worker.wait(2000)


def test_serial_worker_stop_halts_quickly(qapp):
    rows = [_row(1000 + i * 0.001) for i in range(1000)]  # tight spacing
    fake = FakeSerial()
    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=lambda: fake,
    )
    worker.start()
    QTimer.singleShot(200, worker.stop)
    finished_args = _wait_for(worker.finished_run, timeout=3000)
    assert finished_args is not None
    # We should have stopped before writing all 1000 frames
    assert len(fake.written) < 1000 * FRAME_SIZE
    worker.wait(2000)


def test_serial_worker_serial_open_failure_emits_error(qapp, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("port not found")
    worker = SerialWorker(
        rows=[_row()], port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=boom,
    )
    states = []
    worker.state_changed.connect(lambda s: states.append(s))
    worker.finished_run.connect(lambda: _loop.quit())
    _loop = QEventLoop()
    worker.start()
    QTimer.singleShot(2000, _loop.quit)
    _loop.exec()
    assert "error" in states
