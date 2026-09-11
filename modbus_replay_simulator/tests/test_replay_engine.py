import pytest
from modbus_replay.replay_engine import ReplayEngine


def _row(time_s, address=4, fc=3, resp=1, pressure="?"):
    return {
        "address": address, "function": fc, "length": 16,
        "command response": resp,
        "setpoint": "?", "gain": "?", "reset rate": "?",
        "deadband": "?", "cycle time": "?", "rate": "?",
        "system mode": "?", "control scheme": "?",
        "pump": "?", "solenoid": "?",
        "pressure measurement": pressure,
        "crc rate": "?", "time": time_s,
    }


def test_compute_delays_first_row_has_no_delay():
    eng = ReplayEngine([_row(1000), _row(1001), _row(1005)])
    delays = eng.compute_delays()
    # delays describe time from row N to row N+1 → length = N-1
    assert delays == [1000, 4000]


def test_compute_delays_handles_constant_timestamps():
    rows = [_row(1000), _row(1000), _row(1000), _row(1000)]
    eng = ReplayEngine(rows)
    assert eng.compute_delays() == [0, 0, 0]


def test_compute_delays_empty_input():
    assert ReplayEngine([]).compute_delays() == []


def test_compute_delays_single_row():
    assert ReplayEngine([_row(1000)]).compute_delays() == []


def test_encode_all_first_packet_direction_forced_one():
    rows = [_row(1000, resp=0), _row(1001, resp=1)]
    eng = ReplayEngine(rows)
    frames = list(eng.encode_all(time_offset_fn=lambda i: i * 10))
    assert len(frames) == 2
    # first packet direction forced to 1
    assert frames[0][3] == 1
    # subsequent packets: direction follows dataset
    assert frames[1][3] == 1


def test_encode_all_offsets_match_callback():
    rows = [_row(1000), _row(1001), _row(1002)]
    eng = ReplayEngine(rows)
    frames = list(eng.encode_all(time_offset_fn=lambda i: i * 250))
    import struct
    # Each frame's last 4 bytes = uint32 time_offset_ms
    for i, f in enumerate(frames):
        offset = struct.unpack_from("<I", f, 0x1C)[0]
        assert offset == i * 250


def test_encode_all_yields_32_byte_frames():
    rows = [_row(1000 + i) for i in range(10)]
    eng = ReplayEngine(rows)
    frames = list(eng.encode_all(time_offset_fn=lambda i: i))
    assert len(frames) == 10
    assert all(len(f) == 32 for f in frames)
