"""32-byte fixed binary frame layout for Modbus replay.

Layout (little-endian, no CRC):

  Offset  Field            Type     Source CSV column    Notes
  -----------------------------------------------------------------
  0x00   slave_addr       uint8    address              Modbus slave addr
  0x01   function_code    uint8    function             Modbus FC
  0x02   length           uint8    length               byte count (info)
  0x03   direction        uint8    command response     0=cmd, 1=resp
  0x04   setpoint         int16    setpoint             x100
  0x06   gain             int16    gain                 x100
  0x08   reset_rate       int16    reset rate           x100
  0x0A   deadband         int16    deadband             x100
  0x0C   cycle_time       int16    cycle time           raw int
  0x0E   rate             int16    rate                 x100
  0x10   system_mode      uint8    system mode
  0x11   control_scheme   uint8    control scheme
  0x12   pump             uint8    pump                 0=off, 1=on
  0x13   solenoid         uint8    solenoid             0=off, 1=on
  0x14   pressure         float32  pressure measurement 0..100 PSI
  0x18   crc_rate         uint32  crc rate              CRC error count
  0x1C   time_offset_ms   uint32  (derived)             ms since first packet
"""
from __future__ import annotations

import math
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


def _is_missing(v: Any) -> bool:
    """Treat '', '?', None, and NaN floats as missing data."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "?"
    if isinstance(v, float) and math.isnan(v):
        return True
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
    # NB: crc rate can exceed uint8 range; clamp via dedicated logic.
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
