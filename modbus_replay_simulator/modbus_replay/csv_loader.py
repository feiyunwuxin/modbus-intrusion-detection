"""Load IanArffDataset.csv rows into a list of dicts.

The dataset has 20 columns including three label columns
(`binary result`, `categorized result`, `specific result`) that we
keep for future use but do not require for the replay.

`?` is preserved as a string so frame_format.encode() can detect
and encode it as zero-equivalent values.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional
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

ProgressCallback = Callable[[int, int], None]


def load_rows(
    path: str | Path,
    progress_callback: Optional[ProgressCallback] = None,
    progress_every: int = 1000,
) -> list[dict[str, Any]]:
    """Load CSV at `path` and return a list of row dicts.

    Optional ``progress_callback(current, total)`` is invoked three times:

    1. ``(0, total)`` right after pandas has read the file, so the caller
       can size its progress bar to the real row count.
    2. ``(current, total)`` every ``progress_every`` rows (default 1000)
       during the per-row type coercion. The slow part of CSV load is the
       row-by-row loop, so this is where the bar actually moves.
    3. ``(total, total)`` once at the end, so the caller can close any
       modal dialog.

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

    total = len(df)
    if progress_callback is not None:
        progress_callback(0, total)

    # Convert the columns that are known to be numeric to float/int.
    # `numeric_cols` and `float_cols` both route to float() because int()
    # would reject values like "4.0" that appear in some rows.
    float_cols = {
        "address", "function", "length",
        "system mode", "control scheme", "pump", "solenoid",
        "crc rate",
        "setpoint", "gain", "reset rate", "deadband",
        "cycle time", "rate",
        "pressure measurement",
    }
    int_cols = {"command response", "time"}

    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(df.iterrows()):
        # ``raw`` is a (idx, Series) tuple; the row payload is raw[1].
        series = raw[1]
        row: dict[str, Any] = {}
        for col in REQUIRED_COLUMNS:
            v = series[col]
            if v is None or (isinstance(v, float) and math.isnan(v)) or v == "?":
                row[col] = "?"   # sentinel
            elif col in int_cols:
                try:
                    row[col] = int(v)
                except ValueError:
                    row[col] = float(v)
            elif col in float_cols:
                try:
                    row[col] = float(v)
                except ValueError:
                    row[col] = "?"
            else:
                row[col] = v
        rows.append(row)
        if progress_callback is not None and (i + 1) % progress_every == 0:
            progress_callback(i + 1, total)

    if progress_callback is not None:
        progress_callback(total, total)
    return rows
