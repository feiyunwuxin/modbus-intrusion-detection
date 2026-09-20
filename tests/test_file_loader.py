"""文件加载器单元测试。"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modbus_simulator import load_records, REQUIRED_COLUMNS


def _write_csv(path: Path, rows: list[dict]) -> None:
    """辅助：写入 CSV。"""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


_FULL_ROW = {
    "address": 4, "function": 3, "length": 16,
    "setpoint": 0, "gain": 0, "reset": 0, "deadband": 0,
    "cycle": 0, "rate": 0, "system": 0, "control": 0,
    "pump": 0, "solenoid": 0, "pressure": 0,
    "crc": 12869, "command": 1, "time": 1418682163,
}


class TestLoadCSV(unittest.TestCase):
    def test_load_valid_csv(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.csv"
            _write_csv(p, [_FULL_ROW, dict(_FULL_ROW, command=0)])
            records = load_records(str(p))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["address"], 4)
            self.assertEqual(records[1]["command"], 0)

    def test_csv_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            _write_csv(p, [{"address": 4, "function": 3}])
            with self.assertRaises(ValueError) as ctx:
                load_records(str(p))
            self.assertIn("crc", str(ctx.exception))

    def test_csv_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_records("/nonexistent/path/file.csv")

    def test_csv_numeric_coercion(self):
        """CSV 中数字可能是字符串，需转为数字。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(_FULL_ROW.keys()))
                writer.writeheader()
                # 所有字段写为字符串
                writer.writerow({k: str(v) for k, v in _FULL_ROW.items()})
            records = load_records(str(p))
            self.assertEqual(records[0]["address"], 4)
            self.assertEqual(records[0]["reset"], 0)


class TestRequiredColumns(unittest.TestCase):
    def test_required_columns_complete(self):
        for col in ("address", "function", "length", "crc", "command", "time"):
            self.assertIn(col, REQUIRED_COLUMNS)


if __name__ == "__main__":
    unittest.main()
