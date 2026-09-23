"""特征提取：CSV 记录 → float32 数组。

两套提取路径，互不耦合：

* :data:`FEATURE_COLUMNS` + :func:`extract_features` —— 17 维 IDS
  主线（与 2026-09-20 起 IDS 面板一直使用的 SCADA 17 特征顺序一致）。
* :data:`FEATURE_COLUMNS_19` + :func:`extract_features_19` —— 19
  维 SCADA v2 行级特征（preprocess_v2_scada.py 第 1–5 步的产出），
  用于 23 维 KEEP_23 模型的窗口推理。需 ``prev_time`` /
  ``last_seen_time`` 状态来算 ``time_diff`` / ``time_since_last`` /
  ``is_unusual_fc`` / ``is_response`` 四个派生列。
"""
from __future__ import annotations

from typing import Iterable

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


# ──────────────────────────────────────────────────────────────────────
# 23-dim 路径：SCADA v2 行级特征 (19 维)
# ──────────────────────────────────────────────────────────────────────

# 19 行级特征顺序，与 preprocess_v2_scada.py 的 feature_order_row_level
# 完全一致（preprocess 第 152–171 行注释可核对）。
FEATURE_COLUMNS_19: tuple[str, ...] = (
    "address", "function", "length",
    "setpoint", "gain", "reset_rate", "deadband", "cycle_time", "rate",
    "system_mode", "control_scheme", "pump", "solenoid",
    "pressure_measurement", "crc_rate",
    "time_diff", "time_since_last_same_addr_func",
    "is_unusual_fc", "is_response",
)

# KEEP_23 中位于 [0, 18] 的 17 个 per-frame 列索引 → 它们在 19-dim
# 中的真实位置。前 13 个直接来自 record（重命名后），后 4 个是派生列。
# 完整定义见 _common_train.py:22。
_KEEP_23_PER_FRAME_IN_19: tuple[int, ...] = (
    0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  # 13 protocol fields
    15, 16, 17, 18,                                # 4 derived
)

# UNUSUAL_FCS 与 preprocess_v2_scada.py:43 完全一致。这些 FC 是
# command-only（master → slave 单向），出现 = 可疑信号。
UNUSUAL_FCS: frozenset[int] = frozenset({136, 171, 139, 133, 137, 138, 140})


def extract_features_19(
    record: dict,
    *,
    prev_time: int | None = None,
    last_seen_time: dict | None = None,
) -> np.ndarray:
    """提取 19 行级 SCADA 特征 (raw, pre-scale)，返回 (19,) float32。

    派生列计算（与 preprocess_v2_scada.py 第 73–105 行同语义）：

    * ``time_diff`` = ``time - prev_time``；``prev_time is None`` 时为 0
      （首帧没有前驱）。
    * ``time_since_last_same_addr_func`` = ``time - 上次同 (address,
      function) 对的时间戳``；首次出现 = 0。
    * ``is_unusual_fc`` = ``1`` iff ``function ∈ UNUSUAL_FCS``。
    * ``is_response`` = record.command 字段（IDS 记录中 ``command=1``
      表示 master，与 SCADA 原始 ``command response`` 1=response 的语义
      在 KEEP_23 这组模型训练里按 ``is_response = command`` 处理；
      见 DAILY_LOG_2026-09-23 备注）。

    调用方（23-dim wrapper）负责更新 ``prev_time`` /
    ``last_seen_time`` 并维护 16 帧的 buffer，wrapper 自己再过
    RobustScaler + 加窗后聚合得到 (23, 16) 模型输入。
    """
    t = _to_int(record.get("time"))
    fn = _to_int(record.get("function"))
    addr = _to_int(record.get("address"))

    time_diff = (t - prev_time) if prev_time is not None else 0

    key = addr * 1000 + fn
    if last_seen_time is not None and key in last_seen_time:
        time_since_last = t - last_seen_time[key]
    else:
        time_since_last = 0

    is_unusual_fc = 1 if fn in UNUSUAL_FCS else 0
    is_response = _to_int(record.get("command"))

    # 13 个协议字段 + length 直接从 record 取（key 改名映射在下面）
    raw_19 = [
        float(addr),
        float(fn),
        float(_to_int(record.get("length"))),
        float(_to_int(record.get("setpoint"))),
        float(_to_int(record.get("gain"))),
        float(_to_int(record.get("reset"))),    # reset rate
        float(_to_int(record.get("deadband"))),
        float(_to_int(record.get("cycle"))),    # cycle time
        float(_to_int(record.get("rate"))),
        float(_to_int(record.get("system"))),   # system mode
        float(_to_int(record.get("control"))),  # control scheme
        float(_to_int(record.get("pump"))),
        float(_to_int(record.get("solenoid"))),
        float(_to_int(record.get("pressure"))), # pressure measurement
        float(_to_int(record.get("crc"))),      # crc rate
        float(time_diff),
        float(time_since_last),
        float(is_unusual_fc),
        float(is_response),
    ]
    return np.array(raw_19, dtype=np.float32)


def keep_23_per_frame_indices() -> tuple[int, ...]:
    """KEEP_23 中 [0, 18] 的 17 个 per-frame 列在 19-dim 中的位置。

    用于 wrapper 在 buffer 里切片，得到模型 per-frame 输入。
    """
    return _KEEP_23_PER_FRAME_IN_19


def keep_23_aggregate_indices() -> tuple[int, ...]:
    """KEEP_23 中 [19, 26] 的 6 个窗口聚合列名（按 27-dim 位置命名）。

    顺序与 _common_train.py:46 一致：press_mean_w, crc_max_w,
    resp_count_w, cmd_resp_balance_w, length_nunique_w,
    unusual_count_w。每个值都是 (kind, source_index) 二元组，
    kind ∈ {"mean", "max", "sum", "balance", "nunique"}。
    """
    return (
        ("mean",    11),  # pressure_measurement → press_mean_w
        ("max",     12),  # crc_rate → crc_max_w
        ("sum",     18),  # is_response → resp_count_w
        ("balance", 18),  # is_response → cmd_resp_balance_w = (1-resp).sum - resp.sum / W
        ("nunique", 2),   # length → length_nunique_w
        ("sum",     17),  # is_unusual_fc → unusual_count_w
    )