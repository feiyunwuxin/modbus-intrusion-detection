import threading
import time
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from modbus_replay.serial_worker import SerialReader, SerialWorker
from modbus_replay.frame_format import FRAME_SIZE


class FakeSerial:
    """Mimics pyserial.Serial just enough for SerialWorker."""
    def __init__(self, *args, **kwargs):
        self.written = bytearray()
        self._raise_on_write = kwargs.pop("raise_on_write", None)
        self._closed = False
        # RX queue so SerialReader (which polls in_waiting + read) has
        # something to deliver when the test wants to exercise the RX
        # path. Tests that don't care about RX simply leave it empty.
        self._rx_queue = bytearray()

    def write(self, data):
        if self._closed:
            raise RuntimeError("port closed")
        if self._raise_on_write:
            raise self._raise_on_write
        self.written.extend(data)
        return len(data)

    def close(self):
        self._closed = True

    # SerialReader interface — both attributes are read on every poll
    # iteration, so make them cheap properties.
    @property
    def in_waiting(self):
        return len(self._rx_queue)

    def read(self, n):
        chunk = bytes(self._rx_queue[:n])
        del self._rx_queue[:n]
        return chunk


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


def test_serial_worker_emits_stopped_state_on_user_stop(qapp):
    rows = [_row(1000 + i * 0.001) for i in range(100)]  # tight spacing
    fake = FakeSerial()
    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=lambda: fake,
    )
    states = []
    worker.state_changed.connect(lambda s: states.append(s))
    QTimer.singleShot(50, worker.stop)
    worker.start()
    _wait_for(worker.finished_run, timeout=3000)
    assert "stopped" in states


def test_serial_worker_verbose_tx_emits_per_frame_log(qapp):
    """verbose_tx=True emits one TX log_message per successfully written frame."""
    rows = [_row(1000 + i * 0.001) for i in range(5)]
    fake = FakeSerial()
    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, verbose_tx=True,
        serial_factory=lambda: fake,
    )
    tx_logs = []
    worker.log_message.connect(lambda level, msg: tx_logs.append((level, msg)))
    worker.start()
    _wait_for(worker.finished_run, timeout=3000)
    worker.wait(2000)

    # All 5 frames should produce a TX log
    tx_only = [(lvl, msg) for lvl, msg in tx_logs if lvl == "TX"]
    assert len(tx_only) == 5
    # Each line includes row index, time offset, addr, fc, dir, and 8-byte hex head
    for i, (_, msg) in enumerate(tx_only):
        assert f"row={i}" in msg
        assert "addr=" in msg and "fc=" in msg
        assert "hex=" in msg
        # 8 bytes joined by spaces -> 8 hex pairs separated by 7 spaces
        hex_part = msg.split("hex=")[1]
        assert len(hex_part.split()) == 8


def test_serial_worker_verbose_tx_default_is_off(qapp):
    """Default verbose_tx=False must NOT emit per-frame TX logs (avoids
    flooding the log panel on a 274k-row run)."""
    rows = [_row(1000 + i * 0.001) for i in range(5)]
    fake = FakeSerial()
    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=lambda: fake,
    )
    # explicit default check
    assert worker._verbose_tx is False
    tx_logs = []
    worker.log_message.connect(lambda level, msg: tx_logs.append((level, msg)))
    worker.start()
    _wait_for(worker.finished_run, timeout=3000)
    worker.wait(2000)
    tx_only = [(lvl, msg) for lvl, msg in tx_logs if lvl == "TX"]
    assert tx_only == []


def test_serial_worker_emits_rx_received_for_inbound_bytes(qapp):
    """Bytes queued on the serial handle while the worker is running
    must be forwarded via ``rx_received`` so the GUI's RxPanel can show
    them. This is the path that was broken — previously RxPanel was
    wired to the probe handle and never saw MCU replies during replay.
    """
    rows = [_row(1000 + i * 0.001) for i in range(3)]
    fake = FakeSerial()
    # Queue 16 bytes BEFORE the worker starts so the very first poll
    # iteration delivers both chunks atomically (the reader's first
    # ``read(in_waiting)`` returns everything available at that
    # instant). Splitting the queue across two iterations would race
    # against the fast replay loop closing the port.
    fake._rx_queue.extend(b"\x01\x02\x03\x04\x05\x06\x07\x08")
    fake._rx_queue.extend(b"\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10")

    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=lambda: fake,
    )
    rx_chunks = []
    worker.rx_received.connect(lambda b: rx_chunks.append(bytes(b)))
    worker.start()
    _wait_for(worker.finished_run, timeout=3000)
    # The reader emits on its own QThread; queued signals must be
    # drained by processing events before we assert on the captured
    # list. Without this the test sees an empty list because Qt's
    # queued-connection delivery is gated on the event loop.
    deadline = time.time() + 1.0
    while time.time() < deadline and not rx_chunks:
        QApplication.processEvents()
        time.sleep(0.02)
    worker.wait(2000)

    # At least one chunk must arrive (the reader picks up whatever was
    # waiting on the first poll). The exact split between two chunks
    # vs one combined chunk is timing-dependent and not worth pinning.
    assert rx_chunks, "rx_received must fire for bytes waiting on the port"
    # And the union of all chunks must equal what we put in the queue.
    assert b"".join(rx_chunks) == (
        b"\x01\x02\x03\x04\x05\x06\x07\x08"
        b"\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
    )


def test_serial_worker_reader_exits_cleanly_on_close(qapp):
    """After the worker finishes, its internal SerialReader must be
    stopped (not left polling a closed handle). ``_rx_reader`` is
    cleared by the run() finally block."""
    rows = [_row()]
    fake = FakeSerial()
    worker = SerialWorker(
        rows=rows, port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=lambda: fake,
    )
    worker.start()
    _wait_for(worker.finished_run, timeout=3000)
    worker.wait(2000)
    # The reader was instantiated and then cleared on shutdown.
    assert worker._rx_reader is None


def test_serial_reader_emits_received_bytes(qapp):
    """SerialReader used standalone (e.g. by the probe handle) forwards
    bytes from the in_waiting queue via ``data_received``."""
    class Handle:
        def __init__(self):
            self._q = bytearray(b"\xaa\xbb")
        @property
        def in_waiting(self):
            return len(self._q)
        def read(self, n):
            chunk = bytes(self._q[:n])
            del self._q[:n]
            return chunk

    handle = Handle()
    reader = SerialReader(handle)
    captured = []
    reader.data_received.connect(lambda b: captured.append(bytes(b)))
    reader.start()
    # Poll interval is 100ms; wait up to 1s for the first chunk.
    deadline = time.time() + 1.0
    while time.time() < deadline and not captured:
        QApplication.processEvents()
        time.sleep(0.02)
    reader.stop()
    reader.wait(1000)

    assert captured == [b"\xaa\xbb"]
