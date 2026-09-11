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