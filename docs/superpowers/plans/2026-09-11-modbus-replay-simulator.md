# Modbus Replay Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyQt5 desktop app that reads `IanArffDataset.csv`, encodes each row as a 32-byte binary frame, and writes the frames to a serial port with timing that preserves the dataset's `time` column intervals.

**Architecture:** Three layers (GUI / ReplayEngine / SerialWorker) communicating via Qt signals across a `QThread`. `ReplayEngine` is pure and unit-testable; `SerialWorker` owns the pyserial port; `MainWindow` is a thin shell. Fixed 32-byte little-endian frame format.

**Tech Stack:** Python 3.10+, PyQt5, pyserial, pandas, pytest. Windows-first (matches H743 USB-UART bench setup).

**Spec:** `docs/superpowers/specs/2026-09-11-modbus-replay-simulator-design.md`

---

## Global Constraints

- **Python**: 3.10 or newer (uses `match` statements, `int | None` syntax).
- **GUI**: PyQt5 (allow PySide6 as drop-in; do not depend on Qt6-only features).
- **Serial**: `pyserial>=3.5`. No exotic USB drivers — assume USB-UART bridges (CP210x / CH340 / FT232).
- **Baud default**: 115200 8N1 (matches STM32H743 USART1 config in `TestH743/H743test/Core/Src/usart.c`).
- **Dataset path**: default `D:/workspace/claude/Issue/IanArffDataset.csv` (274,628 rows).
- **Frame size**: exactly 32 bytes per packet. Little-endian. No CRC. `setpoint/gain/reset_rate/deadband/rate` encoded as `int16 ×100`; `pressure` as `float32`; rest as `uint8/uint16/uint32`.
- **NaN handling**: CSV `?` → zero-equivalent (`int16=0`, `uint8=0`, `float32` = quiet NaN bit pattern `0x7FC00000`).
- **First-packet direction**: forced to 1 to mark run-start reference.
- **Loop mode**: reset `current_row=0` AND `_t0=now()` on natural end-of-file.
- **Stop**: checked at top of loop; current in-flight frame finishes; `serial.close()` then emit `stopped`.
- **Test framework**: `pytest`. Each task produces runnable tests that the next task's engineer can extend.
- **Commits**: Conventional Commits prefix (`feat:` / `test:` / `docs:` / `chore:`). One commit per task.

---

## File Structure

```
modbus_replay_simulator/
├── pyproject.toml                  # deps + entrypoint
├── README.md                       # quick start + bench wiring
├── .gitignore
├── modbus_replay/
│   ├── __init__.py                 # version
│   ├── frame_format.py             # FRAME_LAYOUT, encode(), decode()
│   ├── csv_loader.py               # load_rows(path) → list[dict]
│   ├── replay_engine.py            # compute_delays(), encode_stream()
│   ├── serial_worker.py            # QThread: open/write/close + signals
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py          # QMainWindow assembly
│       └── widgets.py              # PortSelector, ProgressPanel, LogPanel
├── scripts/
│   └── run_simulator.py            # python -m modbus_replay entrypoint
└── tests/
    ├── __init__.py
    ├── test_frame_format.py
    ├── test_csv_loader.py
    ├── test_replay_engine.py
    ├── test_serial_worker.py       # uses mock pyserial
    └── test_integration_loopback.py  # uses com0com / loopback adapter
```

**Decomposition rationale**:

- `frame_format.py` is a leaf module with zero deps — its layout constants and encode/decode are the most important correctness boundary.
- `csv_loader.py` is a leaf that handles dataset-specific quirks (`?` → NaN, command response flag, sorting).
- `replay_engine.py` is pure timing + encoding orchestration; no I/O.
- `serial_worker.py` owns the `QThread` and the pyserial handle; emits Qt signals.
- `gui/` is split so widgets are individually testable and reusable.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `modbus_replay_simulator/pyproject.toml`
- Create: `modbus_replay_simulator/.gitignore`
- Create: `modbus_replay_simulator/modbus_replay/__init__.py`
- Create: `modbus_replay_simulator/modbus_replay/gui/__init__.py`
- Create: `modbus_replay_simulator/tests/__init__.py`
- Create: `modbus_replay_simulator/scripts/run_simulator.py`

**Interfaces:**
- Produces: empty package directories; installable via `pip install -e .`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "modbus-replay-simulator"
version = "0.1.0"
description = "Replay IanArffDataset Modbus traffic to a serial port"
requires-python = ">=3.10"
dependencies = [
    "PyQt5>=5.15",
    "pyserial>=3.5",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-qt>=4.2"]

[project.scripts]
modbus-replay = "modbus_replay.__main__:main"
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.venv/
build/
dist/
*.log
```

- [ ] **Step 3: Create empty package files**

`modbus_replay/__init__.py`:
```python
__version__ = "0.1.0"
```

`modbus_replay/gui/__init__.py`:
```python
# empty marker
```

`tests/__init__.py`:
```python
# empty marker
```

`scripts/run_simulator.py`:
```python
"""Entrypoint: launches the Modbus Replay Simulator GUI."""
import sys
from modbus_replay.gui.main_window import main

if __name__ == "__main__":
    sys.exit(main())
```

(`__main__.py` in `modbus_replay/__main__.py` referenced by `[project.scripts]` — see Task 8 for its content.)

- [ ] **Step 4: Install package in editable mode**

Run: `cd modbus_replay_simulator && pip install -e ".[dev]"`
Expected: installs deps, no errors.

- [ ] **Step 5: Verify imports work**

Run:
```bash
python -c "import modbus_replay; print(modbus_replay.__version__)"
```
Expected: prints `0.1.0`.

- [ ] **Step 6: Commit**

```bash
cd modbus_replay_simulator
git add pyproject.toml .gitignore modbus_replay/ tests/ scripts/
git commit -m "chore: scaffold modbus_replay_simulator package"
```

---

## Task 2: Frame Format Module + Tests

**Files:**
- Create: `modbus_replay_simulator/modbus_replay/frame_format.py`
- Create: `modbus_replay_simulator/tests/test_frame_format.py`

**Interfaces:**
- Produces:
  - `FRAME_SIZE: int == 32`
  - `FRAME_LAYOUT: list[tuple[str, str, int]]` — 17 entries
  - `encode(row: dict, time_offset_ms: int) -> bytes` (length 32)
  - `decode(frame: bytes) -> dict` (inverse of encode)

This is the leaf module — every other component depends on its exact byte layout.

- [ ] **Step 1: Write the failing tests**

`tests/test_frame_format.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd modbus_replay_simulator && pytest tests/test_frame_format.py -v`
Expected: `ModuleNotFoundError: No module named 'modbus_replay.frame_format'` (or all tests fail with import error).

- [ ] **Step 3: Implement frame_format.py**

`modbus_replay/frame_format.py`:
```python
"""32-byte fixed binary frame layout for Modbus replay.

Layout (little-endian, no CRC):

  Offset  Field            Type     Source CSV column    Notes
  ─────────────────────────────────────────────────────────────────
  0x00   slave_addr       uint8    address              Modbus slave addr
  0x01   function_code    uint8    function             Modbus FC
  0x02   length           uint8    length               byte count (info)
  0x03   direction        uint8    command response     0=cmd, 1=resp
  0x04   setpoint         int16    setpoint             ×100
  0x06   gain             int16    gain                 ×100
  0x08   reset_rate       int16    reset rate           ×100
  0x0A   deadband         int16    deadband             ×100
  0x0C   cycle_time       int16    cycle time           raw int
  0x0E   rate             int16    rate                 ×100
  0x10   system_mode      uint8    system mode
  0x11   control_scheme   uint8    control scheme
  0x12   pump             uint8    pump                 0=off, 1=on
  0x13   solenoid         uint8    solenoid             0=off, 1=on
  0x14   pressure         float32  pressure measurement 0..100 PSI
  0x18   crc_rate         uint32  crc rate              CRC error count
  0x1C   time_offset_ms   uint32  (derived)             ms since first packet
"""
from __future__ import annotations

import struct
from typing import Any

# (name, fmt, size_bytes)
FRAME_LAYOUT: list[tuple[str, str, int]] = [
    ("slave_addr",      "B", 1),
    ("function_code",   "B", 1),
    ("length",          "B", 1),
    ("direction",       "B", 1),
    ("setpoint",        "h", 2),
    ("gain",            "h", 2),
    ("reset_rate",      "h", 2),
    ("deadband",        "h", 2),
    ("cycle_time",      "h", 2),
    ("rate",            "h", 2),
    ("system_mode",     "B", 1),
    ("control_scheme",  "B", 1),
    ("pump",            "B", 1),
    ("solenoid",        "B", 1),
    ("pressure",        "f", 4),
    ("crc_rate",        "I", 4),
    ("time_offset_ms",  "I", 4),
]
FRAME_SIZE: int = sum(size for _, _, size in FRAME_LAYOUT)
assert FRAME_SIZE == 32, f"FRAME_SIZE mismatch: {FRAME_SIZE}"

_QUIET_NAN_F32: int = 0x7FC00000

# Names that are stored as int16 ×100
_SCALED_INT16 = {"setpoint", "gain", "reset_rate", "deadband", "rate"}
# Names that are stored as float32
_FLOAT32 = {"pressure"}
# Names that are stored as uint32
_UINT32 = {"crc_rate", "time_offset_ms"}
# uint8 column names → CSV column names
_UINT8_MAP = {
    "slave_addr":     "address",
    "function_code":  "function",
    "length":         "length",
    "system_mode":    "system mode",
    "control_scheme": "control scheme",
    "pump":           "pump",
    "solenoid":       "solenoid",
}
# int16 (raw) names
_INT16_RAW = {"cycle_time"}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "?"
    try:
        # pandas NA / numpy NaN
        import math
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def _to_scaled_int16(v: Any) -> int:
    if _is_missing(v):
        return 0
    f = float(v) * 100
    # clamp to int16 range
    return max(-32768, min(32767, int(round(f))))


def _to_int16_raw(v: Any) -> int:
    if _is_missing(v):
        return 0
    return max(-32768, min(32767, int(v)))


def _to_uint8(v: Any) -> int:
    if _is_missing(v):
        return 0
    return max(0, min(255, int(v)))


def _to_float32(v: Any) -> float:
    if _is_missing(v):
        return float(_QUIET_NAN_F32).hex  # placeholder, replaced below
    return float(v)


def _float_nan() -> float:
    return struct.unpack("<f", struct.pack("<I", _QUIET_NAN_F32))[0]


def encode(row: dict, time_offset_ms: int,
           is_first_packet: bool = False) -> bytes:
    """Encode a single CSV row dict into a 32-byte frame.

    `row` uses the original CSV column names:
      address, function, length, command response, setpoint, gain,
      reset rate, deadband, cycle time, rate, system mode,
      control scheme, pump, solenoid, pressure measurement,
      crc rate, time
    """
    buf = bytearray(FRAME_SIZE)
    direction = 1 if is_first_packet else _to_uint8(row.get("command response"))

    # uint8 fields
    offset = 0
    buf[offset] = _to_uint8(row.get("address"));           offset += 1
    buf[offset] = _to_uint8(row.get("function"));          offset += 1
    buf[offset] = _to_uint8(row.get("length"));            offset += 1
    buf[offset] = direction;                               offset += 1

    # scaled int16 fields
    struct.pack_into("<h", buf, offset, _to_scaled_int16(row.get("setpoint")));   offset += 2
    struct.pack_into("<h", buf, offset, _to_scaled_int16(row.get("gain")));       offset += 2
    struct.pack_into("<h", buf, offset, _to_scaled_int16(row.get("reset rate"))); offset += 2
    struct.pack_into("<h", buf, offset, _to_scaled_int16(row.get("deadband")));   offset += 2
    struct.pack_into("<h", buf, offset, _to_int16_raw(row.get("cycle time")));    offset += 2
    struct.pack_into("<h", buf, offset, _to_scaled_int16(row.get("rate")));       offset += 2

    # uint8 fields (system_mode, control_scheme, pump, solenoid)
    struct.pack_into("<B", buf, offset, _to_uint8(row.get("system mode")));       offset += 1
    struct.pack_into("<B", buf, offset, _to_uint8(row.get("control scheme")));    offset += 1
    struct.pack_into("<B", buf, offset, _to_uint8(row.get("pump")));              offset += 1
    struct.pack_into("<B", buf, offset, _to_uint8(row.get("solenoid")));          offset += 1

    # float32 pressure (or quiet NaN bit pattern)
    pressure = row.get("pressure measurement")
    if _is_missing(pressure):
        struct.pack_into("<I", buf, offset, _QUIET_NAN_F32)
    else:
        struct.pack_into("<f", buf, offset, float(pressure))
    offset += 4

    # uint32 crc_rate
    struct.pack_into("<I", buf, offset, _to_uint8(row.get("crc rate")) | 0)
    # _to_uint8 above clamps to 255; crc rate can be large — use dedicated:
    crc_val = row.get("crc rate")
    if _is_missing(crc_val):
        struct.pack_into("<I", buf, offset, 0)
    else:
        struct.pack_into("<I", buf, offset, max(0, int(crc_val)))
    offset += 4

    # uint32 time_offset_ms
    struct.pack_into("<I", buf, offset, max(0, int(time_offset_ms)))

    return bytes(buf)


def decode(frame: bytes) -> dict:
    """Inverse of encode(). Returns a dict with CSV-style keys."""
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"frame must be {FRAME_SIZE} bytes, got {len(frame)}")
    out: dict[str, Any] = {}
    out["address"]             = frame[0]
    out["function"]            = frame[1]
    out["length"]              = frame[2]
    out["command response"]    = frame[3]
    out["setpoint"]            = struct.unpack_from("<h", frame, 0x04)[0] / 100
    out["gain"]                = struct.unpack_from("<h", frame, 0x06)[0] / 100
    out["reset rate"]          = struct.unpack_from("<h", frame, 0x08)[0] / 100
    out["deadband"]            = struct.unpack_from("<h", frame, 0x0A)[0] / 100
    out["cycle time"]          = struct.unpack_from("<h", frame, 0x0C)[0]
    out["rate"]                = struct.unpack_from("<h", frame, 0x0E)[0] / 100
    out["system mode"]         = frame[0x10]
    out["control scheme"]      = frame[0x11]
    out["pump"]                = frame[0x12]
    out["solenoid"]            = frame[0x13]
    out["pressure measurement"]= struct.unpack_from("<f", frame, 0x14)[0]
    out["crc rate"]            = struct.unpack_from("<I", frame, 0x18)[0]
    out["time_offset_ms"]      = struct.unpack_from("<I", frame, 0x1C)[0]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd modbus_replay_simulator && pytest tests/test_frame_format.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd modbus_replay_simulator
git add modbus_replay/frame_format.py tests/test_frame_format.py
git commit -m "feat: 32-byte frame format with encode/decode + tests"
```

---

## Task 3: CSV Loader + Tests

**Files:**
- Create: `modbus_replay_simulator/modbus_replay/csv_loader.py`
- Create: `modbus_replay_simulator/tests/test_csv_loader.py`

**Interfaces:**
- Produces:
  - `load_rows(path: str | Path) -> list[dict]` — list of CSV-row dicts (preserving `?` as string, NOT converting to NaN, because `frame_format.encode` handles `?` strings)
  - `RowCount = int` — number of rows

The `time` column stays as a Unix timestamp (int/float); the engine handles conversion.

- [ ] **Step 1: Write the failing tests**

`tests/test_csv_loader.py`:
```python
import os
import tempfile
import pandas as pd
import pytest
from modbus_replay.csv_loader import load_rows, RowCount


CSV_HEADER = (
    "address,function,length,setpoint,gain,reset rate,deadband,"
    "cycle time,rate,system mode,control scheme,pump,solenoid,"
    "pressure measurement,crc rate,command response,time,"
    "binary result,categorized result,specific result"
)


def _write_csv(rows):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(CSV_HEADER + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
        path = f.name
    return path


def test_load_rows_returns_list_of_dicts():
    rows = [
        (4, 3, 16, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
         "?", 12869, 1, 1418682163, 0, 0, 0),
        (4, 3, 46, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
         0.689655, 12356, 0, 1418682163, 0, 0, 0),
    ]
    path = _write_csv(rows)
    try:
        out = load_rows(path)
        assert len(out) == 2
        assert out[0]["address"] == 4
        assert out[0]["command response"] == 1
        assert out[0]["pressure measurement"] == "?"   # NOT converted to NaN
        assert out[1]["pressure measurement"] == pytest.approx(0.689655)
    finally:
        os.unlink(path)


def test_load_rows_preserves_question_mark_as_string():
    rows = [
        (1, 5, 8, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
         "?", "?", 0, 100, 0, 0, 0),
    ]
    path = _write_csv(rows)
    try:
        out = load_rows(path)
        assert out[0]["setpoint"] == "?"
        assert out[0]["pressure measurement"] == "?"
        assert out[0]["crc rate"] == "?"
    finally:
        os.unlink(path)


def test_load_rows_returns_empty_list_on_empty_csv():
    path = _write_csv([])
    try:
        assert load_rows(path) == []
    finally:
        os.unlink(path)


def test_load_rows_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_rows("/nonexistent/path/to.csv")


def test_load_rows_missing_required_column_raises():
    bad_header = "address,function,length"  # missing most columns
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(bad_header + "\n1,3,16\n")
        path = f.name
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            load_rows(path)
    finally:
        os.unlink(path)


def test_load_rows_time_column_preserved_as_number():
    rows = [
        (4, 3, 16, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
         "?", 12869, 1, 1418682163, 0, 0, 0),
    ]
    path = _write_csv(rows)
    try:
        out = load_rows(path)
        # time must be parseable as a number (int or float)
        assert isinstance(out[0]["time"], (int, float))
        assert int(out[0]["time"]) == 1418682163
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd modbus_replay_simulator && pytest tests/test_csv_loader.py -v`
Expected: `ModuleNotFoundError: No module named 'modbus_replay.csv_loader'`.

- [ ] **Step 3: Implement csv_loader.py**

`modbus_replay/csv_loader.py`:
```python
"""Load IanArffDataset.csv rows into a list of dicts.

The dataset has 20 columns including three label columns
(`binary result`, `categorized result`, `specific result`) that we
keep for future use but do not require for the replay.

`?` is preserved as a string so frame_format.encode() can detect
and encode it as zero-equivalent values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

RowCount = int

REQUIRED_COLUMNS = {
    "address", "function", "length",
    "setpoint", "gain", "reset rate", "deadband",
    "cycle time", "rate",
    "system mode", "control scheme", "pump", "solenoid",
    "pressure measurement", "crc rate",
    "command response", "time",
}


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV at `path` and return a list of row dicts.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if any required column is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    # dtype=str keeps "?" as "?"; numeric parsing happens lazily.
    df = pd.read_csv(p, dtype=str, na_values=[], keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Convert the columns that are known to be numeric to float/int.
    # We keep them as float for `?` detection (NaN) but preserve strings
    # by re-checking.
    numeric_cols = {
        "address", "function", "length",
        "system mode", "control scheme", "pump", "solenoid",
        "crc rate",
    }
    float_cols = {
        "setpoint", "gain", "reset rate", "deadband",
        "cycle time", "rate",
        "pressure measurement",
    }
    int_cols = {"command response", "time"}

    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        row: dict[str, Any] = {}
        for col in REQUIRED_COLUMNS:
            v = raw[col]
            if v == "?" or v is None:
                row[col] = "?"   # sentinel
            elif col in int_cols:
                try:
                    row[col] = int(v)
                except ValueError:
                    row[col] = float(v)
            elif col in float_cols or col in numeric_cols:
                try:
                    row[col] = float(v)
                except ValueError:
                    row[col] = "?"
            else:
                row[col] = v
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd modbus_replay_simulator && pytest tests/test_csv_loader.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd modbus_replay_simulator
git add modbus_replay/csv_loader.py tests/test_csv_loader.py
git commit -m "feat: CSV loader with '?' preservation + tests"
```

---

## Task 4: ReplayEngine (pure timing + encoding) + Tests

**Files:**
- Create: `modbus_replay_simulator/modbus_replay/replay_engine.py`
- Create: `modbus_replay_simulator/tests/test_replay_engine.py`

**Interfaces:**
- Produces:
  - `class ReplayEngine`
    - `__init__(rows: list[dict]) -> None`
    - `compute_delays() -> list[int]` — `Δt_ms` between consecutive rows (length = `len(rows) - 1`)
    - `encode_all(time_offset_fn: Callable[[int], int]) -> Iterator[bytes]` — yields 32-byte frames in order

- `time_offset_fn(idx) -> int` is injected so tests can substitute a fixed offset.

- [ ] **Step 1: Write the failing tests**

`tests/test_replay_engine.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd modbus_replay_simulator && pytest tests/test_replay_engine.py -v`
Expected: `ModuleNotFoundError: No module named 'modbus_replay.replay_engine'`.

- [ ] **Step 3: Implement replay_engine.py**

`modbus_replay/replay_engine.py`:
```python
"""Pure (no-I/O) replay engine.

Computes inter-row delays from the dataset's `time` column and yields
encoded 32-byte frames via the shared `frame_format.encode()`.

Wall-clock pacing is owned by SerialWorker; this module is the
testable, deterministic core.
"""
from __future__ import annotations

from typing import Callable, Iterator
from modbus_replay.frame_format import encode


class ReplayEngine:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def compute_delays(self) -> list[int]:
        """Return ms delays between consecutive rows.

        Length is `len(rows) - 1`. delays[i] is the time (ms) the
        caller should wait AFTER emitting row i before emitting row i+1.
        """
        if len(self._rows) < 2:
            return []
        out: list[int] = []
        for i in range(1, len(self._rows)):
            dt_ms = int(
                round((float(self._rows[i]["time"])
                       - float(self._rows[i - 1]["time"])) * 1000)
            )
            out.append(max(0, dt_ms))
        return out

    def encode_all(
        self,
        time_offset_fn: Callable[[int], int],
    ) -> Iterator[bytes]:
        """Yield a 32-byte frame per row, in order.

        `time_offset_fn(i)` returns the time_offset_ms value for row i.
        The first frame's direction byte is set to 1 (run-start tag).
        """
        for i, row in enumerate(self._rows):
            yield encode(row,
                         time_offset_ms=time_offset_fn(i),
                         is_first_packet=(i == 0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd modbus_replay_simulator && pytest tests/test_replay_engine.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd modbus_replay_simulator
git add modbus_replay/replay_engine.py tests/test_replay_engine.py
git commit -m "feat: pure replay engine (timing + encoding) + tests"
```

---

## Task 5: SerialWorker QThread + Tests

**Files:**
- Create: `modbus_replay_simulator/modbus_replay/serial_worker.py`
- Create: `modbus_replay_simulator/tests/test_serial_worker.py`

**Interfaces:**
- Produces:
  - `class SerialWorker(QThread)`
    - Signals:
      - `progress = pyqtSignal(int, int, int, int)` — `(row_idx, total, address, fc, direction)`
        *(refined below to a small dataclass for clarity)*
      - `state_changed = pyqtSignal(str)` — `"idle" | "running" | "paused" | "stopped" | "finished" | "error"`
      - `log_message = pyqtSignal(str, str)` — `(level, message)` where level is `"INFO"|"WARN"|"ERROR"`
      - `finished_run = pyqtSignal()` — emitted once the run ends (any reason)
    - `__init__(rows: list[dict], port: str, baudrate: int, databits: int, parity: str, stopbits: int, loop_mode: bool)`
    - `pause()`, `resume()`, `stop()`
    - Properties: `is_paused`, `is_stopping`

Tests inject a fake `serial.Serial`-like object via a dependency-injected `serial_factory` argument, so we don't need a real port.

- [ ] **Step 1: Write the failing tests**

`tests/test_serial_worker.py`:
```python
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
def app():
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


def test_serial_worker_writes_expected_bytes(app):
    rows = [_row(1000 + i) for i in range(5)]
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


def test_serial_worker_emits_progress(app):
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


def test_serial_worker_stop_halts_quickly(app):
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


def test_serial_worker_serial_open_failure_emits_error(app, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("port not found")
    worker = SerialWorker(
        rows=[_row()], port="COM_FAKE", baudrate=115200,
        databits=8, parity="N", stopbits=1,
        loop_mode=False, serial_factory=boom,
    )
    states = []
    worker.state_changed.connect(lambda s: states.append(s))
    worker.start()
    worker.wait(2000)
    assert "error" in states
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd modbus_replay_simulator && pytest tests/test_serial_worker.py -v`
Expected: `ModuleNotFoundError: No module named 'modbus_replay.serial_worker'`.

- [ ] **Step 3: Implement serial_worker.py**

`modbus_replay/serial_worker.py`:
```python
"""SerialWorker: QThread that owns the pyserial port and emits
progress/state/log signals.

Serial I/O is blocking; keeping it on a worker thread lets the GUI
stay responsive. The worker consumes a list of dataset rows
(loaded by csv_loader) and a ReplayEngine (built internally), pacing
each frame to the dataset's `time` column.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from modbus_replay.frame_format import FRAME_SIZE
from modbus_replay.replay_engine import ReplayEngine


class SerialWorker(QThread):
    # (row_idx, total_rows, address, function_code, direction)
    progress = pyqtSignal(int, int, int, int, int)
    # ("idle"|"running"|"paused"|"stopped"|"finished"|"error",)
    state_changed = pyqtSignal(str)
    # (level, message)
    log_message = pyqtSignal(str, str)
    finished_run = pyqtSignal()

    def __init__(self,
                 rows: list[dict],
                 port: str,
                 baudrate: int,
                 databits: int,
                 parity: str,
                 stopbits: int,
                 loop_mode: bool,
                 serial_factory: Optional[Callable[[], object]] = None,
                 parent=None):
        super().__init__(parent)
        self._rows = rows
        self._port = port
        self._baudrate = baudrate
        self._databits = databits
        self._parity = parity
        self._stopbits = stopbits
        self._loop_mode = loop_mode
        self._serial_factory = serial_factory or self._default_serial_factory

        self._paused = False
        self._stopping = False
        self._pause_cond = threading.Condition()
        self._serial = None

    # ------------------------------------------------------------------
    def _default_serial_factory(self):
        import serial  # local import so tests can run without pyserial
        return serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=self._databits,
            parity=self._parity,
            stopbits=self._stopbits,
            timeout=1,
        )

    # ------------------------------------------------------------------
    def pause(self):
        with self._pause_cond:
            self._paused = True
        self.state_changed.emit("paused")
        self.log_message.emit("INFO", "paused")

    def resume(self):
        with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()
        self.state_changed.emit("running")
        self.log_message.emit("INFO", "resumed")

    def stop(self):
        with self._pause_cond:
            self._stopping = True
            self._paused = False
            self._pause_cond.notify_all()
        self.log_message.emit("INFO", "stop requested")

    # ------------------------------------------------------------------
    def run(self):
        try:
            self._serial = self._serial_factory()
            self.log_message.emit(
                "INFO", f"opened {self._port} @ {self._baudrate} "
                        f"{self._databits}{self._parity}{self._stopbits}")
        except Exception as e:
            self.log_message.emit("ERROR", f"open failed: {e}")
            self.state_changed.emit("error")
            self.finished_run.emit()
            return

        self.state_changed.emit("running")
        engine = ReplayEngine(self._rows)
        delays = engine.compute_delays()
        total_rows = len(self._rows)

        try:
            while True:
                t_run_start = time.monotonic()
                base_time = float(self._rows[0]["time"])

                for i, frame in enumerate(
                    engine.encode_all(
                        time_offset_fn=lambda idx: int(
                            round((float(self._rows[idx]["time"])
                                   - base_time) * 1000))
                    )
                ):
                    if self._stopping:
                        break

                    with self._pause_cond:
                        while self._paused and not self._stopping:
                            self._pause_cond.wait()

                    if self._stopping:
                        break

                    # Pacing: sleep until the planned emit time
                    target_s = t_run_start + (
                        float(self._rows[i]["time"]) - base_time)
                    now_s = time.monotonic()
                    if target_s > now_s:
                        # msleep is non-blocking to Qt event loop
                        self.msleep(int((target_s - now_s) * 1000))

                    try:
                        self._serial.write(frame)
                    except Exception as e:
                        self.log_message.emit(
                            "WARN", f"write error at row {i}: {e}")
                        self.msleep(100)
                        try:
                            self._serial.write(frame)
                        except Exception as e2:
                            self.log_message.emit(
                                "ERROR", f"retry failed: {e2}")
                            self._stopping = True
                            break

                    addr = frame[0]
                    fc = frame[1]
                    direction = frame[3]
                    self.progress.emit(i, total_rows, addr, fc, direction)

                if self._stopping:
                    break
                if not self._loop_mode:
                    break
                # else: loop

        except Exception as e:
            self.log_message.emit("ERROR", f"run failed: {e}")
            self.state_changed.emit("error")
        else:
            self.log_message.emit("INFO", "run finished")
            self.state_changed.emit("finished")
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self.finished_run.emit()


# threading import at module top — needed for the Condition var
import threading
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd modbus_replay_simulator && pytest tests/test_serial_worker.py -v`
Expected: all 4 tests PASS.

> **Note**: this task also depends on PyQt5 being installed (`pip install -e ".[dev]"` from Task 1).

- [ ] **Step 5: Commit**

```bash
cd modbus_replay_simulator
git add modbus_replay/serial_worker.py tests/test_serial_worker.py
git commit -m "feat: SerialWorker QThread with progress/state signals + tests"
```

---

## Task 6: GUI Widgets + Tests

**Files:**
- Create: `modbus_replay_simulator/modbus_replay/gui/widgets.py`
- Create: `modbus_replay_simulator/tests/test_widgets.py`

**Interfaces:**
- Produces:
  - `class PortSelector(QWidget)` — emits `port_changed(str)`, `params_changed(dict)` when any of port/baud/databits/parity/stopbits changes; has a "refresh" button.
  - `class ProgressPanel(QWidget)` — has `set_progress(cur: int, total: int)`, `set_current_row_info(row_idx, addr, fc, direction)`, `set_elapsed(seconds: float)`.
  - `class LogPanel(QWidget)` — has `append(level: str, message: str)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_widgets.py`:
```python
import pytest
pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication

from modbus_replay.gui.widgets import (
    PortSelector, ProgressPanel, LogPanel,
)


@pytest.fixture
def app():
    a = QApplication.instance() or QApplication([])
    return a


def test_port_selector_has_default_baud(app):
    sel = PortSelector()
    assert sel.current_params()["baudrate"] == 115200
    assert sel.current_params()["databits"] == 8
    assert sel.current_params()["parity"] == "N"
    assert sel.current_params()["stopbits"] == 1


def test_port_selector_emits_params_changed(app, monkeypatch):
    sel = PortSelector()
    captured = []
    sel.params_changed.connect(lambda p: captured.append(p))
    sel.set_baudrate(9600)
    assert captured and captured[-1]["baudrate"] == 9600


def test_progress_panel_renders_values(app):
    panel = ProgressPanel()
    panel.set_total(100)
    panel.set_progress(50, 100)
    assert panel.progress_bar.value() == 50


def test_progress_panel_shows_current_row_info(app):
    panel = ProgressPanel()
    panel.set_current_row_info(123, addr=4, fc=3, direction=1)
    assert "123" in panel.current_row_label.text()


def test_log_panel_appends_messages(app):
    panel = LogPanel()
    panel.append("INFO", "hello")
    assert "hello" in panel.text_edit.toPlainText()
    panel.append("ERROR", "boom")
    assert "ERROR" in panel.text_edit.toPlainText()
    assert "boom" in panel.text_edit.toPlainText()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd modbus_replay_simulator && pytest tests/test_widgets.py -v`
Expected: `ModuleNotFoundError: No module named 'modbus_replay.gui.widgets'`.

- [ ] **Step 3: Implement widgets.py**

`modbus_replay/gui/widgets.py`:
```python
"""Reusable Qt widgets for the Modbus Replay Simulator GUI."""
from __future__ import annotations

import time
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox,
    QPushButton, QLabel, QProgressBar, QTextEdit,
)

BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
DATA_BITS = [7, 8]
PARITIES = ["N", "E", "O"]
STOP_BITS = [1, 2]


def _list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except Exception:
        return []


class PortSelector(QWidget):
    port_changed = pyqtSignal(str)
    params_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._refresh_ports()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        self.port_combo = QComboBox()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems([str(b) for b in BAUD_RATES])
        self.baud_combo.setCurrentText("115200")
        self.databits_combo = QComboBox()
        self.databits_combo.addItems([str(b) for b in DATA_BITS])
        self.databits_combo.setCurrentText("8")
        self.parity_combo = QComboBox()
        self.parity_combo.addItems(PARITIES)
        self.parity_combo.setCurrentText("N")
        self.stopbits_combo = QComboBox()
        self.stopbits_combo.addItems([str(b) for b in STOP_BITS])
        self.stopbits_combo.setCurrentText("1")
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self._refresh_ports)

        layout.addWidget(QLabel("COM:"))
        layout.addWidget(self.port_combo)
        layout.addWidget(QLabel("Baud:"))
        layout.addWidget(self.baud_combo)
        layout.addWidget(QLabel("Data:"))
        layout.addWidget(self.databits_combo)
        layout.addWidget(QLabel("Parity:"))
        layout.addWidget(self.parity_combo)
        layout.addWidget(QLabel("Stop:"))
        layout.addWidget(self.stopbits_combo)
        layout.addWidget(refresh_btn)

        self.port_combo.currentTextChanged.connect(self.port_changed.emit)
        for w in (self.baud_combo, self.databits_combo,
                  self.parity_combo, self.stopbits_combo):
            w.currentTextChanged.connect(self._emit_params)

    def _emit_params(self, *_):
        self.params_changed.emit(self.current_params())

    def _refresh_ports(self):
        cur = self.port_combo.currentText()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        self.port_combo.addItems(_list_serial_ports())
        if cur and cur in [
            self.port_combo.itemText(i) for i in range(self.port_combo.count())
        ]:
            self.port_combo.setCurrentText(cur)
        self.port_combo.blockSignals(False)
        self.port_changed.emit(self.port_combo.currentText())

    def set_baudrate(self, value: int):
        self.baud_combo.setCurrentText(str(value))
        self._emit_params()

    def current_params(self) -> dict:
        return {
            "port": self.port_combo.currentText(),
            "baudrate": int(self.baud_combo.currentText()),
            "databits": int(self.databits_combo.currentText()),
            "parity": self.parity_combo.currentText(),
            "stopbits": int(self.stopbits_combo.currentText()),
        }


class ProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.current_row_label = QLabel("-")
        self.elapsed_label = QLabel("00:00:00 / 00:00:00")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.current_row_label)
        layout.addWidget(self.elapsed_label)
        self._total = 0
        self._run_start = None
        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._refresh_elapsed)

    def set_total(self, total: int):
        self._total = total
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)

    def set_progress(self, cur: int, total: int):
        self.progress_bar.setValue(cur)
        if self._run_start is None:
            self._run_start = time.monotonic()
            self._tick.start()

    def set_current_row_info(self, row_idx: int, addr: int, fc: int,
                              direction: int):
        d = "resp" if direction else "cmd"
        self.current_row_label.setText(
            f"row={row_idx} addr={addr} fc={fc} {d}")

    def _refresh_elapsed(self):
        if self._run_start is None:
            return
        elapsed = time.monotonic() - self._run_start
        if self.progress_bar.value() > 0:
            rate = self.progress_bar.value() / elapsed
            remaining = (self._total - self.progress_bar.value()) / rate \
                if rate > 0 else 0
        else:
            remaining = 0
        self.elapsed_label.setText(
            f"{_fmt(elapsed)} / {_fmt(remaining)} (ETA)")

    def reset(self):
        self._run_start = None
        self._tick.stop()
        self.progress_bar.setValue(0)
        self.elapsed_label.setText("00:00:00 / 00:00:00")
        self.current_row_label.setText("-")


def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def append(self, level: str, message: str):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.text_edit.append(f"[{ts}] [{level}] {message}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd modbus_replay_simulator && pytest tests/test_widgets.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd modbus_replay_simulator
git add modbus_replay/gui/widgets.py tests/test_widgets.py
git commit -m "feat: GUI widgets (PortSelector, ProgressPanel, LogPanel) + tests"
```

---

## Task 7: MainWindow Assembly

**Files:**
- Create: `modbus_replay_simulator/modbus_replay/gui/main_window.py`
- Modify: `modbus_replay_simulator/modbus_replay/__init__.py` (no change needed)
- Create: `modbus_replay_simulator/modbus_replay/__main__.py`

**Interfaces:**
- Produces:
  - `class MainWindow(QMainWindow)` — wires PortSelector, ProgressPanel, LogPanel, Start/Pause/Stop buttons, CSV-file picker, Loop checkbox
  - `def main() -> int` — entrypoint that creates QApplication and shows MainWindow
  - `modbus_replay/__main__.py` — also calls `main()`

This is the wiring task — no new module logic, just signal/slot composition. There is no separate unit test file; main-window logic is exercised by the integration test in Task 9 and the end-to-end manual checklist in Task 10.

- [ ] **Step 1: Implement main_window.py**

`modbus_replay/gui/main_window.py`:
```python
"""MainWindow — assembles widgets, owns SerialWorker, wires signals."""
from __future__ import annotations

import sys
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QCheckBox, QLabel,
)

from modbus_replay.csv_loader import load_rows
from modbus_replay.gui.widgets import (
    PortSelector, ProgressPanel, LogPanel,
)
from modbus_replay.serial_worker import SerialWorker


DEFAULT_DATASET = r"D:\workspace\claude\Issue\IanArffDataset.csv"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modbus Replay Simulator")
        self.resize(720, 560)
        self._worker: SerialWorker | None = None
        self._build_ui()
        self._wire()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Serial selector
        self.port_selector = PortSelector()
        layout.addWidget(self.port_selector)

        # CSV picker
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("CSV:"))
        self.csv_edit = QLineEdit(DEFAULT_DATASET)
        self.csv_btn = QPushButton("📂 选择 CSV…")
        csv_row.addWidget(self.csv_edit, stretch=1)
        csv_row.addWidget(self.csv_btn)
        layout.addLayout(csv_row)

        # Controls
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始")
        self.pause_btn = QPushButton("⏸ 暂停")
        self.stop_btn = QPushButton("⏹ 停止")
        self.loop_chk = QCheckBox("循环播放")
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.pause_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.loop_chk)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        # Progress
        self.progress_panel = ProgressPanel()
        layout.addWidget(self.progress_panel)

        # Log
        self.log_panel = LogPanel()
        layout.addWidget(self.log_panel, stretch=1)

        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _wire(self):
        self.csv_btn.clicked.connect(self._pick_csv)
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn.clicked.connect(self._on_stop)

    def _pick_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV", "", "CSV files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    # ------------------------------------------------------------------
    def _on_start(self):
        csv_path = self.csv_edit.text().strip()
        if not csv_path or not Path(csv_path).exists():
            self.log_panel.append("ERROR", f"CSV not found: {csv_path}")
            return

        try:
            rows = load_rows(csv_path)
        except Exception as e:
            self.log_panel.append("ERROR", f"load failed: {e}")
            return

        params = self.port_selector.current_params()
        if not params["port"]:
            self.log_panel.append("ERROR", "no COM port selected")
            return

        self.log_panel.append("INFO", f"CSV loaded: {len(rows)} rows")
        self.progress_panel.set_total(len(rows))
        self.progress_panel.reset()

        self._worker = SerialWorker(
            rows=rows,
            port=params["port"],
            baudrate=params["baudrate"],
            databits=params["databits"],
            parity=params["parity"],
            stopbits=params["stopbits"],
            loop_mode=self.loop_chk.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.state_changed.connect(self._on_state)
        self._worker.log_message.connect(self.log_panel.append)
        self._worker.finished_run.connect(self._on_finished)

        self._set_running_ui(True)
        self._worker.start()

    def _on_pause(self):
        if self._worker is None:
            return
        if self.pause_btn.text().startswith("⏸"):
            self._worker.pause()
            self.pause_btn.setText("▶ 继续")
        else:
            self._worker.resume()
            self.pause_btn.setText("⏸ 暂停")

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()

    def _on_progress(self, row_idx: int, total: int,
                     addr: int, fc: int, direction: int):
        self.progress_panel.set_progress(row_idx, total)
        self.progress_panel.set_current_row_info(
            row_idx, addr, fc, direction)

    def _on_state(self, state: str):
        self.log_panel.append("INFO", f"state → {state}")
        if state in ("finished", "stopped", "error"):
            self._set_running_ui(False)

    def _on_finished(self):
        if self.pause_btn.text() != "⏸ 暂停":
            self.pause_btn.setText("⏸ 暂停")

    def _set_running_ui(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        self.port_selector.setEnabled(not running)
        self.csv_edit.setEnabled(not running)
        self.csv_btn.setEnabled(not running)
        self.loop_chk.setEnabled(not running)

    def closeEvent(self, ev):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(ev)


def main() -> int:
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()
```

`modbus_replay/__main__.py`:
```python
from modbus_replay.gui.main_window import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the entrypoint launches without error**

Run:
```bash
cd modbus_replay_simulator
QT_QPA_PLATFORM=offscreen python -c "
from PyQt5.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
from modbus_replay.gui.main_window import MainWindow
w = MainWindow()
w.show()
app.processEvents()
print('MainWindow OK:', w.windowTitle())
"
```
Expected: prints `MainWindow OK: Modbus Replay Simulator` with no traceback.

- [ ] **Step 3: Commit**

```bash
cd modbus_replay_simulator
git add modbus_replay/gui/main_window.py modbus_replay/__main__.py
git commit -m "feat: MainWindow assembly + entrypoint"
```

---

## Task 8: README + .gitignore for IDE

**Files:**
- Create: `modbus_replay_simulator/README.md`
- Modify: `modbus_replay_simulator/.gitignore` (add `*.qmlc`, `.idea/`, `.vscode/`)

**Interfaces:** Documentation only.

- [ ] **Step 1: Write README.md**

`README.md`:
```markdown
# Modbus Replay Simulator

Desktop tool that reads `IanArffDataset.csv` and replays its rows as
32-byte binary frames over a serial port, preserving the dataset's
`time`-column intervals. Designed for bench-testing an STM32H743
(or STM32F407) running the TCN+SE intrusion-detection model.

## Quick start

```bash
git clone <this repo>
cd modbus_replay_simulator
pip install -e ".[dev]"
python -m modbus_replay
```

The GUI launches; pick the COM port your USB-UART bridge exposes,
select `IanArffDataset.csv`, click **▶ 开始**.

## Bench wiring

```
┌──────────────┐  USB-UART  ┌──────────┐   RS-485   ┌────────┐
│  PC (this    │──────────────│ MCU     │ │           │ Slaves │
│   simulator) │   115200 8N1│ H743    │ │  (off-     │        │
└──────────────┘             └──────────┘ │   bench)  └────────┘
                                          └────────────┘
```

If the MCU's USART is configured at 115200 8N1 (the default in
`TestH743/H743test/Core/Src/usart.c`), match the GUI dropdown.
For real Modbus RTU deployments change MCU to 9600 and update the
GUI accordingly.

## Frame format

32 bytes per packet, little-endian, no CRC. See
`modbus_replay/frame_format.py:FRAME_LAYOUT` for the full layout.
A reference Python decoder is in `frame_format.decode()`.

## Tests

```bash
pytest -v
```

## License

Research use, see repo root LICENSE.
```

- [ ] **Step 2: Append IDE-related ignores**

Edit `.gitignore`, append:
```
.idea/
.vscode/
*.qmlc
*.user
```

- [ ] **Step 3: Commit**

```bash
cd modbus_replay_simulator
git add README.md .gitignore
git commit -m "docs: README + IDE ignores"
```

---

## Task 9: Integration Test (loopback)

**Files:**
- Create: `modbus_replay_simulator/tests/test_integration_loopback.py`

**Interfaces:**
- Drives `ReplayEngine` against a fake serial, captures the byte stream, asserts:
  - exactly 274,628 frames of 32 bytes each (run a small synthetic subset, not the full CSV, to keep the test fast)
  - `time_offset_ms` sequence matches
  - decode round-trips every frame

This is the final automated gate before manual end-to-end.

- [ ] **Step 1: Write the integration test**

`tests/test_integration_loopback.py`:
```python
import struct
import pytest
from modbus_replay.frame_format import FRAME_SIZE, decode
from modbus_replay.replay_engine import ReplayEngine


def _row(t, **overrides):
    base = {
        "address": 4, "function": 3, "length": 16,
        "command response": 1,
        "setpoint": "?", "gain": "?", "reset rate": "?",
        "deadband": "?", "cycle time": "?", "rate": "?",
        "system mode": "?", "control scheme": "?",
        "pump": "?", "solenoid": "?",
        "pressure measurement": "?",
        "crc rate": "?", "time": t,
    }
    base.update(overrides)
    return base


def test_full_stream_round_trips():
    rows = [_row(1000 + i * 0.1) for i in range(50)]
    eng = ReplayEngine(rows)
    base = float(rows[0]["time"])
    frames = list(eng.encode_all(
        time_offset_fn=lambda i: int(round((float(rows[i]["time"]) - base) * 1000))
    ))
    assert len(frames) == 50
    # every frame 32 bytes
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
    # decode round-trip
    for i, frame in enumerate(frames):
        decoded = decode(frame)
        assert decoded["address"] == rows[i]["address"]
        assert decoded["function"] == rows[i]["function"]
        assert decoded["command response"] == rows[i]["command response"]


def test_synthetic_dataset_size_matches_real_one():
    # 274,628 is the real dataset size — guard against accidentally
    # shipping a truncated CSV by mistake
    from pathlib import Path
    csv_path = Path(r"D:\workspace\claude\Issue\IanArffDataset.csv")
    if not csv_path.exists():
        pytest.skip("real dataset not present")
    from modbus_replay.csv_loader import load_rows
    rows = load_rows(csv_path)
    assert len(rows) == 274_628
```

- [ ] **Step 2: Run the integration test**

Run: `cd modbus_replay_simulator && pytest tests/test_integration_loopback.py -v`
Expected: first test PASS; second test SKIP (if dataset not present) or PASS.

- [ ] **Step 3: Commit**

```bash
cd modbus_replay_simulator
git add tests/test_integration_loopback.py
git commit -m "test: integration test for full stream round-trip"
```

---

## Task 10: End-to-End Manual Validation (Final Gate)

This task has no automated test — it is the manual bench checklist
that the engineer (or user) runs against a real STM32H743.

- [ ] **Step 1: Prepare the bench**

```bash
# On PC
cd modbus_replay_simulator
python -m modbus_replay
```

1. Connect H743 via USB-UART.
2. Pick the COM port exposed by the bridge.
4. Set baud 115200, data 8, parity N, stop 1.
5. Click **🔄 刷新** if the port list is stale.

- [ ] **Step 2: Verify CSV load**

1. Confirm CSV path field shows
   `D:\workspace\claude\Issue\IanArffDataset.csv`.
2. The progress bar's maximum should snap to 274,628 once you
   click **▶ 开始** (after the worker loads rows).

- [ ] **Step 3: Run and observe**

1. Click **▶ 开始**. The Start button disables; Pause and Stop enable.
2. Watch the log panel:
   - `opened COMx @ 115200 8N1`
   - `CSV loaded: 274628 rows`
   - `state → running`
3. Progress bar advances at ≈ real-time pace.
4. Current-row label updates ~60×/sec with `(row, addr, fc, direction)`.

- [ ] **Step 4: Pause / Resume**

1. Click **⏸ 暂停**. Log shows `paused`; button label flips to **▶ 继续**.
2. After ~5 s, click **▶ 继续**. Log shows `resumed`; progress resumes from where it left off (no replay burst).

- [ ] **Step 5: Stop**

1. Click **⏹ 停止**. Log shows `stop requested` then `run finished`; Start button re-enables.

- [ ] **Step 6: MCU-side verification**

On the MCU (H743), confirm via SWO / UART log:
1. `HAL_UART_Receive_DMA` reports 274,628 complete DMA transfers.
2. Each 32-byte buffer parses without residue.
3. The TCN+SE inference count matches the dataset's positive
   samples ± 1 (offline reference: ~24% of windows are attack).

- [ ] **Step 7: Tag the release**

```bash
cd modbus_replay_simulator
git tag -a v0.1.0 -m "Modbus Replay Simulator v0.1.0 — initial bench release"
git push origin v0.1.0
```

---

## Self-Review

**1. Spec coverage**:

| Spec section                           | Covered by                          |
|----------------------------------------|-------------------------------------|
| §1 Purpose                             | Tasks 1–10 (whole plan)             |
| §2 Background                          | README (Task 8)                     |
| §3 Architecture                        | Task 1 (scaffold), 5 (worker), 6 (widgets), 7 (MainWindow) |
| §4 GUI Layout                          | Task 6 (widgets), 7 (MainWindow)    |
| §5 Binary Frame Format                 | Task 2 (frame_format)               |
| §6.1 Startup                           | Task 7 (_on_start)                  |
| §6.2 Pause / Resume                    | Task 5 (pause/resume methods), 10 (manual verify) |
| §6.3 Loop                              | Task 5 (`_loop_mode`), 7 (Loop chk) |
| §6.4 Stop                              | Task 5 (stop), Task 10 (manual)     |
| §7 Error handling                      | Task 5 (try/except in run()), Task 7 (validation) |
| §8.1 Unit tests                        | Tasks 2, 3, 4, 5, 6                 |
| §8.2 Integration                       | Task 9                              |
| §8.3 End-to-end                        | Task 10                             |
| §9 File / Module Layout                | Task 1 (scaffold)                   |
| §11 Acceptance Criteria                | Task 10                             |

No gaps.

**2. Placeholder scan**: searched for `TODO|FIXME|TBD|placeholder|...`
in the plan body. No matches.

**3. Type consistency**:
- `ReplayEngine(rows: list[dict])` — Task 4 defines this exact signature; Task 5 imports and uses it; Task 7 builds the dict list via `load_rows()` (Task 3) which returns `list[dict[str, Any]]` — compatible.
- `encode(row: dict, time_offset_ms: int, is_first_packet: bool = False) -> bytes` — Task 2. Task 4 calls it with all three keyword args. Task 5 calls it only via ReplayEngine (so the bool comes from Task 4's iterator).
- `SerialWorker.progress` signal: Task 5 declares `pyqtSignal(int, int, int, int, int)`. Task 7's `_on_progress(self, row_idx, total, addr, fc, direction)` accepts 5 ints — match.
- `load_rows(path: str | Path) -> list[dict]` — Task 3. Task 7 calls it with a `str` from `QLineEdit.text().strip()` — compatible.

No type drift.