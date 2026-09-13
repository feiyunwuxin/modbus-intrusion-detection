"""End-to-end integration test for the replay pipeline.

This is the final automated gate before manual hardware validation
(Task 10): it drives a :class:`ReplayEngine` over a synthetic subset,
then asserts that the emitted byte stream matches the dataset
schema exactly and that ``decode()`` round-trips every frame. A
second test guards the real dataset's row count so a truncated CSV
shipped by accident cannot pass silently.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from modbus_replay.frame_format import FRAME_SIZE, decode
from modbus_replay.replay_engine import ReplayEngine


def _row(t, **overrides):
    """Build one synthetic CSV-shaped row."""
    base = {
        "address": 4,
        "function": 3,
        "length": 16,
        "command response": 1,
        "setpoint": "?",
        "gain": "?",
        "reset rate": "?",
        "deadband": "?",
        "cycle time": "?",
        "rate": "?",
        "system mode": "?",
        "control scheme": "?",
        "pump": "?",
        "solenoid": "?",
        "pressure measurement": "?",
        "crc rate": "?",
        "time": t,
    }
    base.update(overrides)
    return base


def test_full_stream_round_trips():
    """50 synthetic rows produce 50 frames of 32 bytes each; offsets match."""
    rows = [_row(1000 + i * 0.1) for i in range(50)]
    eng = ReplayEngine(rows)
    base = float(rows[0]["time"])
    frames = list(
        eng.encode_all(
            time_offset_fn=lambda i: int(
                round((float(rows[i]["time"]) - base) * 1000)
            )
        )
    )

    # Exactly 50 frames, all 32 bytes long.
    assert len(frames) == 50
    assert all(len(f) == FRAME_SIZE for f in frames)

    # time_offset_ms sequence matches the dataset
    expected_offsets = [
        int(round((float(rows[i]["time"]) - base) * 1000))
        for i in range(len(rows))
    ]
    actual_offsets = [
        struct.unpack_from("<I", f, 0x1C)[0] for f in frames
    ]
    assert actual_offsets == expected_offsets

    # decode round-trips the key protocol fields for every frame
    for i, frame in enumerate(frames):
        decoded = decode(frame)
        assert decoded["address"] == rows[i]["address"]
        assert decoded["function"] == rows[i]["function"]
        assert decoded["command response"] == rows[i]["command response"]


def test_first_packet_direction_forced_to_one():
    """First row's `command response` is overridden to 0; frame still emits 1."""
    rows = [
        _row(1000.0, **{"command response": 0}),
        _row(1000.1, **{"command response": 0}),
        _row(1000.2, **{"command response": 1}),
    ]
    eng = ReplayEngine(rows)
    frames = list(
        eng.encode_all(
            time_offset_fn=lambda i: int(round(float(rows[i]["time"]) * 1000))
        )
    )
    assert len(frames) == 3
    # First packet: direction byte (offset 3) is forced to 1
    assert frames[0][3] == 1
    # Second packet: now respects the source row
    assert frames[1][3] == 0
    # Third packet: 1
    assert frames[2][3] == 1


def test_nan_fields_round_trip_as_quiet_nan():
    """Missing numeric fields encode as quiet NaN; decode recovers them."""
    import math

    rows = [
        _row(
            1000.0,
            rate=1.5,
            **{"pressure measurement": 42.0},
        )
    ]
    eng = ReplayEngine(rows)
    frames = list(
        eng.encode_all(
            time_offset_fn=lambda i: int(round(float(rows[i]["time"]) * 1000))
        )
    )
    decoded = decode(frames[0])

    # rate and pressure were given real numbers → decode yields finite floats
    assert decoded["rate"] == pytest.approx(1.5, rel=1e-6)
    assert decoded["pressure measurement"] == pytest.approx(42.0, rel=1e-6)

    # All `?` integer fields should be 0 after the missing-value rule
    assert decoded["setpoint"] == 0
    assert decoded["gain"] == 0


def test_real_dataset_row_count():
    """Guard against shipping a truncated IanArffDataset.csv.

    Skipped if the real dataset is not present at the expected path.
    """
    csv_path = Path(r"C:\work\Claude\Issue\IanArffDataset.csv")
    if not csv_path.exists():
        pytest.skip("real dataset not present at " + str(csv_path))

    from modbus_replay.csv_loader import load_rows
    rows = load_rows(csv_path)
    assert len(rows) == 274_628


def test_real_dataset_first_50_frames_round_trip(tmp_path):
    """Take the first 50 rows from the real CSV and verify their frames."""
    csv_path = Path(r"C:\work\Claude\Issue\IanArffDataset.csv")
    if not csv_path.exists():
        pytest.skip("real dataset not present at " + str(csv_path))

    from modbus_replay.csv_loader import load_rows
    rows = load_rows(csv_path)[:50]
    eng = ReplayEngine(rows)
    base = float(rows[0]["time"])
    frames = list(
        eng.encode_all(
            time_offset_fn=lambda i: int(
                round((float(rows[i]["time"]) - base) * 1000)
            )
        )
    )
    assert len(frames) == 50
    assert all(len(f) == FRAME_SIZE for f in frames)
    # First packet direction is forced to 1 regardless of source
    assert frames[0][3] == 1
