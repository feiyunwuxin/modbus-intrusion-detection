"""特征提取：CSV 记录 → (17,) float32 数组。"""
from __future__ import annotations

import numpy as np


FEATURE_COLUMNS: tuple[str, ...] = (
    "address", "function", "length",
    "setpoint", "gain", "reset", "deadband", "cycle", "rate",
    "system", "control", "pump", "solenoid", "pressure",
    "crc", "command", "time",
)


def _to_int(value) -> int:
    """与 build_frame 一致的浮点截断。None/空值 → 0。"""
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def extract_features(record: dict) -> np.ndarray:
    """从 CSV/XLSX 记录提取 17 个特征，返回 (17,) float32 数组。

    顺序固定为 FEATURE_COLUMNS；缺失字段视为 0。
    """
    return np.array(
        [_to_int(record.get(col)) for col in FEATURE_COLUMNS],
        dtype=np.float32,
    )