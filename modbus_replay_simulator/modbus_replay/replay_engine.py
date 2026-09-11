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
