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

# SCADA CSV 列名 → 17-dim 短 key 的别名映射。IanArffDataset_RAW.csv 用
# SCADA 原始列名（"pressure measurement" / "crc rate" / "reset rate" /
# "cycle time" / "system mode" / "control scheme" / "command response"），
# 但 FEATURE_COLUMNS 用短名（"pressure" / "crc" / "reset" / ...）。
# 不做这个映射的话，csv_loader 输出的 record 用 SCADA 名 → extract_features
# 按短名 lookup 全部 None → 0 → IDS 面板实时推理所有字段静默丢失
# （X_test 直接喂模型 85%+ 准确率，但 IDS 面板 100% 报警）。
#
# 别名按优先级排列：先查 SCADA CSV 列名（生产路径），再查短名
# （历史测试 / 内部代码用）。
_FEATURE_ALIASES_17: dict[str, tuple[str, ...]] = {
    "address":  ("address",),
    "function": ("function",),
    "length":   ("length",),
    "setpoint": ("setpoint",),
    "gain":     ("gain",),
    "reset":    ("reset rate", "reset"),
    "deadband": ("deadband",),
    "cycle":    ("cycle time", "cycle"),
    "rate":     ("rate",),
    "system":   ("system mode", "system"),
    "control":  ("control scheme", "control"),
    "pump":     ("pump",),
    "solenoid": ("solenoid",),
    "pressure": ("pressure measurement", "pressure"),
    "crc":      ("crc rate", "crc"),
    "command":  ("command response", "command"),
    "time":     ("time",),
}


def _to_int(value) -> int:
    """与 build_frame 一致的浮点截断。None/空值 → 0。"""
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value) -> float:
    """保留小数位的浮点解析；与 preprocess_v2_scada.py 的 ``pd.to_numeric``
    语义一致（``pressure_measurement`` / ``setpoint`` / ``gain`` /
    ``deadband`` / ``cycle time`` / ``rate`` / ``system mode`` /
    ``control scheme`` / ``pump`` / ``solenoid`` / ``reset rate``
    等 SCADA 数值列都用这个语义）。None/空值/"?" → 0.0。
    """
    if value is None or value == "" or value == "?":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_int(record: dict, short_key: str) -> int:
    """按 ``_FEATURE_ALIASES_17[short_key]`` 顺序在 record 里查找字段值。

    第一个找到的非 ``None`` 值转 int 返回；全部 miss 或值为 ``"?"`` /
    空字符串则返回 0。SCADA CSV 路径下用 ``"pressure measurement"``
    等长列名也能查到。

    注意：int() 会截断小数 → 只用于模型按离散 code 处理的字段
    （address / function / length / time / command）。
    """
    for alias in _FEATURE_ALIASES_17.get(short_key, (short_key,)):
        v = record.get(alias)
        if v is None or v == "" or v == "?":
            continue
        return _to_int(v)
    return 0


def _resolve_float(record: dict, short_key: str) -> float:
    """保留小数位的版本；用于 SCADA 数值列（pressure / cumulative /
    setpoint / gain / deadband / cycle / rate / system / control /
    pump / solenoid / reset / crc_rate）。

    23-dim KEEP_23 模型训练时 ``pressure_measurement`` /
    ``reset_rate`` / ``deadband`` / ``cycle_time`` / ``rate`` 都用
    ``pd.to_numeric`` 保留小数位；wrapper 必须同样保留，否则
    ``press_mean_w`` 聚合特征错位 → 模型把所有帧预测成 attack。
    """
    for alias in _FEATURE_ALIASES_17.get(short_key, (short_key,)):
        v = record.get(alias)
        if v is None or v == "" or v == "?":
            continue
        return _to_float(v)
    return 0.0


def extract_features(record: dict) -> np.ndarray:
    """从 CSV/XLSX 记录提取 17 个特征，返回 (17,) float32 数组。

    顺序固定为 FEATURE_COLUMNS；缺失字段视为 0。

    通过 :func:`_resolve_int` 同时支持 SCADA CSV 列名（"pressure
    measurement" 等）和短名（"pressure" 等）——前者是 csv_loader
    输出的真实 record，后者保留向后兼容（历史测试 / 内部代码）。
    """
    return np.array(
        [_resolve_int(record, col) for col in FEATURE_COLUMNS],
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
    t = _resolve_int(record, "time")
    fn = _resolve_int(record, "function")
    addr = _resolve_int(record, "address")

    time_diff = (t - prev_time) if prev_time is not None else 0

    key = addr * 1000 + fn
    if last_seen_time is not None and key in last_seen_time:
        time_since_last = t - last_seen_time[key]
    else:
        time_since_last = 0

    is_unusual_fc = 1 if fn in UNUSUAL_FCS else 0
    is_response = _resolve_int(record, "command")

    # 13 个协议字段 + length 通过 _resolve_* 同时支持 SCADA CSV
    # 列名（"pressure measurement" / "reset rate" / ...）和短名
    # （"pressure" / "reset" / ...）。csv_loader 输出的是 SCADA 名，
    # 短名是历史路径。
    #
    # 数值列（setpoint / gain / reset_rate / deadband / cycle_time /
    # rate / system_mode / control_scheme / pump / solenoid /
    # pressure_measurement / crc_rate）必须用 _resolve_float 保留
    # 小数位，否则 23-dim KEEP_23 wrapper 的 press_mean_w 聚合特征
    # 跟 X_train 不匹配 → 模型全部判 attack。
    # address / function / length / command / time 是离散 code，用 int。
    raw_19 = [
        float(addr),
        float(fn),
        float(_resolve_int(record, "length")),
        _resolve_float(record, "setpoint"),
        _resolve_float(record, "gain"),
        _resolve_float(record, "reset"),        # reset rate
        _resolve_float(record, "deadband"),
        _resolve_float(record, "cycle"),        # cycle time
        _resolve_float(record, "rate"),
        _resolve_float(record, "system"),       # system mode
        _resolve_float(record, "control"),      # control scheme
        _resolve_float(record, "pump"),
        _resolve_float(record, "solenoid"),
        _resolve_float(record, "pressure"),     # pressure measurement
        _resolve_float(record, "crc"),          # crc rate
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