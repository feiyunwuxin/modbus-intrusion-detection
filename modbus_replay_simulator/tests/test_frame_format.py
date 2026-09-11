import struct
import pytest
from modbus_replay.frame_format import (
    FRAME_SIZE, FRAME_LAYOUT, encode, decode,
)


def test_frame_size_is_32():
    assert FRAME_SIZE == 32


def test_frame_layout_has_17_fields():
    assert len(FRAME_LAYOUT) == 17
    # sum of sizes must equal 32
    total = sum(size for _, _, size in FRAME_LAYOUT)
    assert total == 32


def test_encode_known_row():
    row = {
        "address": 4,
        "function": 3,
        "length": 16,
        "command response": 1,
        "setpoint": 1.5,
        "gain": 0.25,
        "reset rate": 1.0,
        "deadband": 0.5,
        "cycle time": 100,
        "rate": 0.1,
        "system mode": 2,
        "control scheme": 1,
        "pump": 1,
        "solenoid": 0,
        "pressure measurement": 50.5,
        "crc rate": 12345,
        "time": 1418682163,
    }
    frame = encode(row, time_offset_ms=250)
    assert len(frame) == 32
    # byte 0 = slave address
    assert frame[0] == 4
    # byte 1 = function code
    assert frame[1] == 3
    # byte 2 = length
    assert frame[2] == 16
    # byte 3 = direction
    assert frame[3] == 1
    # setpoint at offset 0x04, int16, ×100 → 150
    assert struct.unpack_from("<h", frame, 0x04)[0] == 150
    # gain at offset 0x06, int16, ×100 → 25
    assert struct.unpack_from("<h", frame, 0x06)[0] == 25
    # pressure at offset 0x14, float32 → 50.5
    assert struct.unpack_from("<f", frame, 0x14)[0] == pytest.approx(50.5)
    # crc_rate at offset 0x18, uint32 → 12345
    assert struct.unpack_from("<I", frame, 0x18)[0] == 12345
    # time_offset_ms at offset 0x1C, uint32 → 250
    assert struct.unpack_from("<I", frame, 0x1C)[0] == 250


def test_encode_nan_fields_become_quiet_nan_for_float():
    row = {
        "address": 0, "function": 0, "length": 0,
        "command response": 0,
        "setpoint": "?", "gain": "?", "reset rate": "?",
        "deadband": "?", "cycle time": "?", "rate": "?",
        "system mode": "?", "control scheme": "?",
        "pump": "?", "solenoid": "?",
        "pressure measurement": "?",
        "crc rate": "?",
        "time": 0,
    }
    frame = encode(row, time_offset_ms=0)
    # float fields: pressure at 0x14 = quiet NaN
    pressure_bits = struct.unpack_from("<I", frame, 0x14)[0]
    assert pressure_bits == 0x7FC00000
    # int fields: zero
    assert struct.unpack_from("<h", frame, 0x04)[0] == 0
    # uint fields: zero
    assert struct.unpack_from("<I", frame, 0x18)[0] == 0


def test_first_packet_direction_forced_to_one():
    row = {
        "address": 1, "function": 5, "length": 8,
        "command response": 0,  # even though dataset says command
        "setpoint": "?", "gain": "?", "reset rate": "?",
        "deadband": "?", "cycle time": "?", "rate": "?",
        "system mode": "?", "control scheme": "?",
        "pump": "?", "solenoid": "?",
        "pressure measurement": "?",
        "crc rate": "?",
        "time": 100,
    }
    frame = encode(row, time_offset_ms=0, is_first_packet=True)
    assert frame[3] == 1  # direction overridden


def test_round_trip():
    row = {
        "address": 7, "function": 6, "length": 32,
        "command response": 1,
        "setpoint": 2.75, "gain": 0.5, "reset rate": 1.2,
        "deadband": 0.1, "cycle time": 250, "rate": 0.05,
        "system mode": 5, "control scheme": 3,
        "pump": 1, "solenoid": 1,
        "pressure measurement": 75.25,
        "crc rate": 99999,
        "time": 1418682300,
    }
    frame = encode(row, time_offset_ms=1000)
    decoded = decode(frame)
    assert decoded["address"] == 7
    assert decoded["function"] == 6
    assert decoded["command response"] == 1
    assert decoded["setpoint"] == pytest.approx(2.75, abs=0.005)
    assert decoded["gain"] == pytest.approx(0.5, abs=0.005)
    assert decoded["pressure measurement"] == pytest.approx(75.25, abs=1e-5)
    assert decoded["crc rate"] == 99999


def test_decode_rejects_short_buffer():
    with pytest.raises(ValueError):
        decode(b"\x00" * 10)


def test_all_int16_scaled_fields_fit_in_range():
    # setpoint=-327.68, gain=-327.68, ..., rate=-327.68 — all safe
    row = {
        "address": 0, "function": 0, "length": 0,
        "command response": 0,
        "setpoint": -327.68, "gain": -327.68, "reset rate": -327.68,
        "deadband": -327.68, "cycle time": 0, "rate": -327.68,
        "system mode": 0, "control scheme": 0,
        "pump": 0, "solenoid": 0,
        "pressure measurement": 0.0,
        "crc rate": 0,
        "time": 0,
    }
    frame = encode(row, time_offset_ms=0)
    # No exception, frame still 32 bytes
    assert len(frame) == 32
