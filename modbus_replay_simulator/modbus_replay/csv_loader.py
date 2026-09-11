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
    for _, raw in df.iterrows():
        row: dict[str, Any] = {}
        for col in REQUIRED_COLUMNS:
            v = raw[col]
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
    return rows
