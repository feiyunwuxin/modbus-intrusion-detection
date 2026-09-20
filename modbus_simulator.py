"""Modbus 模拟器 - 主程序入口。

CLI 模式:
    python modbus_simulator.py [data_file]   # 启动 GUI
    python modbus_simulator.py --self-test   # 自检模式
"""
import csv
import struct
import sys
from pathlib import Path
from typing import List


# 帧字段顺序定义：每个元组 (字段名, 字节数, struct 格式)
_FRAME_FIELDS = [
    ("address", 1, "B"), ("function", 1, "B"), ("length", 1, "B"),
    ("setpoint", 1, "B"), ("gain", 1, "B"), ("reset", 1, "B"),
    ("deadband", 1, "B"), ("cycle", 1, "B"), ("rate", 1, "B"),
    ("system", 1, "B"), ("control", 1, "B"), ("pump", 1, "B"),
    ("solenoid", 1, "B"), ("pressure", 1, "B"),
    ("crc", 2, ">H"),      # big-endian uint16
    ("command", 1, "B"),
    ("time", 4, ">I"),     # big-endian uint32
]


def _coerce_byte(value, size: int) -> int:
    """将任意值转换为整数，按 size 字节截断。"""
    if value is None or value == "":
        return 0
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return 0
    mask = (1 << (size * 8)) - 1
    return v & mask


def build_frame(record: dict) -> List[int]:
    """根据 CSV/XLSX 记录构建 21 字节 Modbus 风格帧。

    返回整数列表（每个元素 0-255）。
    """
    frame: List[int] = []
    for name, size, _fmt in _FRAME_FIELDS:
        raw = _coerce_byte(record.get(name), size)
        if size == 1:
            frame.append(raw)
        else:
            # 大端拆分
            for i in range(size - 1, -1, -1):
                frame.append((raw >> (i * 8)) & 0xFF)
    return frame


def frame_to_hex(frame: List[int]) -> str:
    """将字节列表格式化为 'XX XX ... XX'。"""
    return " ".join(f"{b:02X}" for b in frame)


REQUIRED_COLUMNS = ("address", "function", "length", "crc", "command", "time")


def _coerce_number(value):
    """将字符串转为数字（保留浮点精度给原值）；空值返回 0。"""
    if value is None or value == "":
        return 0
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0


def _normalize_row(row: dict) -> dict:
    """标准化一行记录：所有数值字段转为数字（必需列强制转，其它列若可转也转）。"""
    out = {}
    for k, v in row.items():
        if k in REQUIRED_COLUMNS:
            out[k] = _coerce_number(v)
        else:
            # 其它列：尝试转数字；不能转则保留原值
            try:
                if v is None or v == "":
                    out[k] = v
                elif "." in str(v):
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return [_normalize_row(r) for r in rows]


def _load_xlsx(path: Path) -> list[dict]:
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise RuntimeError(
            "openpyxl 未安装，无法读取 .xlsx 文件。请运行: pip install openpyxl"
        ) from e
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        return []
    header = [str(h) for h in header]
    records = []
    for row in rows_iter:
        if row is None or all(c is None for c in row):
            continue
        record = {header[i]: row[i] for i in range(len(header)) if i < len(row)}
        records.append(_normalize_row(record))
    wb.close()
    return records


def load_records(path: str) -> list[dict]:
    """加载 CSV 或 XLSX 数据文件，返回标准化记录列表。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 缺少必需列
        RuntimeError: openpyxl 未安装（仅 xlsx 时）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = p.suffix.lower()
    if ext == ".csv":
        records = _load_csv(p)
    elif ext == ".xlsx":
        records = _load_xlsx(p)
    else:
        raise ValueError(f"不支持的文件格式: {ext}（仅支持 .csv / .xlsx）")
    # 校验必需列
    if records:
        missing = [c for c in REQUIRED_COLUMNS if c not in records[0]]
        if missing:
            raise ValueError(f"文件缺少必需列: {', '.join(missing)}")
    return records


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        print("Self-test mode not yet implemented")
        return 1
    print("GUI mode not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
