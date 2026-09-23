import os
import tempfile
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
        # typing policy: numeric cols (numeric_cols set) land as float
        assert isinstance(out[0]["address"], float)
        assert out[0]["command response"] == 1
        assert isinstance(out[0]["command response"], int)
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


def test_load_rows_progress_callback_fires_three_times():
    """progress_callback receives (0,total), intermediate steps, then (total,total)."""
    # 2500 rows so the default progress_every=1000 fires twice mid-loop
    rows = [
        (1, 3, 8, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
         "?", "?", 0, 100, 0, 0, 0)
    ] * 2500
    path = _write_csv(rows)
    try:
        calls = []
        out = load_rows(path, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(out) == 2500
        # First call sizes the bar: (0, 2500)
        assert calls[0] == (0, 2500)
        # Last call is the terminal (total, total)
        assert calls[-1] == (2500, 2500)
        # Intermediate calls are multiples of progress_every (default 1000):
        # (1000, 2500) and (2000, 2500) — at least these two must appear.
        intermediate = calls[1:-1]
        assert (1000, 2500) in intermediate
        assert (2000, 2500) in intermediate
        # And every intermediate (current, total) has current < total.
        for c, t in intermediate:
            assert 0 < c < t
            assert c % 1000 == 0
    finally:
        os.unlink(path)


def test_load_rows_progress_callback_progress_every_override():
    """progress_every controls how often the callback fires mid-loop."""
    rows = [
        (1, 3, 8, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
         "?", "?", 0, 100, 0, 0, 0)
    ] * 2500
    path = _write_csv(rows)
    try:
        calls = []
        load_rows(
            path,
            progress_callback=lambda c, t: calls.append((c, t)),
            progress_every=500,
        )
        intermediate = calls[1:-1]
        # Every intermediate call should be a multiple of 500 (≤ total).
        # ``c == total`` is allowed: when the row count is itself a
        # multiple of progress_every the last mid-loop fire coincides
        # with the row-count boundary, which is fine.
        for c, t in intermediate:
            assert c % 500 == 0
            assert c <= t
    finally:
        os.unlink(path)


def test_load_rows_no_callback_still_returns_rows():
    """Backwards compat: omitting progress_callback behaves exactly as before."""
    rows = [
        (4, 3, 16, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
         "?", 12869, 1, 1418682163, 0, 0, 0),
    ]
    path = _write_csv(rows)
    try:
        out = load_rows(path)
        assert len(out) == 1
        assert out[0]["address"] == 4
    finally:
        os.unlink(path)
