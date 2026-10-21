"""
rtu_replay_gui.py  --  Pure Modbus RTU Replay GUI (PC = Master)

读取 IanArffDataset.csv,按 ARFF 的 time 列时间间隔构**纯 Modbus RTU 帧**
(无自定义 0xAA SOP, 无 66 字节包装),经串口发出,用于联调 STM32H743
TCN+SE IDS (KeilH743/modbus_rtu.c 路径, process_state.c FC3 轮询拿过程值).

帧格式 (Modbus RTU, 标准):
  [slave(1)] [func(1)] [pdu(N)] [CRC_lo(1)] [CRC_hi(1)]

CSV → 帧字段映射 (17 列原始数据 → Modbus RTU):
  - address           → slave (1 byte)
  - function          → func  (1 byte)
  - length            → PDU 字节数 (决定 request/response 形状)
  - command response  → 方向 (1=响应, 0=请求). **PC 是主站, 响应行跳过**
  - setpoint/gain/reset rate/deadband/cycle time/rate
  - system mode/control scheme/pump/solenoid
  - pressure measurement/crc rate        → 12 个 int16 寄存器 (x100, clamp)
  - time                                 → 不下发 (H743 自己用 HAL_GetTick 算 time_diff)

帧模板 (Modbus RTU, CRC16 在末尾):
  FC 1/2 request:  slave(1) + func(1) + start_addr(2) + count(2)              + CRC(2) = 8
  FC 3/4 request:  slave(1) + func(1) + start_addr(2) + count(2)              + CRC(2) = 8
  FC 3/4 response: slave(1) + func(1) + byte_count(1) + regs(2*count)         + CRC(2)
  FC 5/6 request:  slave(1) + func(1) + addr(2) + value(2)                    + CRC(2) = 8
  FC 15 request:   slave(1) + func(1) + start_addr(2) + count(2) + byte_count(1) + coils + CRC(2)
  FC 16 request:   slave(1) + func(1) + start_addr(2) + count(2) + byte_count(1) + regs(2*count) + CRC(2)
  其他 FC:         slave(1) + func(1) + data(length)                          + CRC(2)

H743 端 modbus_rtu_parse → modbus_frame_t → feature_extractor_update
会按现有逻辑算 f[0..22] = 23 维 (time_diff / time_since_last_same /
is_unusual_fc / is_response / 6 窗口统计) → sliding_window → TCN+SE.
过程值 (pressure / gain 等) 由 H743 自己的 process_state FC3 轮询拿,
不需要 PC 在帧里下发.
"""
from __future__ import annotations

import csv
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import serial
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QTextCursor
# ⭐ 第三十二轮 (2026-10-02): QFont 改为按需 import — 测试 dummy 不必都提供,
#   仅 RxPanel 真正实例化时才需要 QFont (原: 顶部 import 强制要求所有 dummy 提供).
try:
    from PyQt5.QtGui import QFont
except ImportError:
    QFont = None  # type: ignore[assignment]
# ⭐ 第四十轮 (2026-10-02): matplotlib 按需 import — 仅 MetricsChartPanel 实例化时才需要.
#   缺失时降级为 QLabel 占位 (与 QFont lazy import 模式一致).
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as _MplCanvas,
    )
    _MPL_AVAILABLE = True
except ImportError:
    Figure = None  # type: ignore[assignment]
    _MplCanvas = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QScrollArea, QTextEdit,
    QVBoxLayout, QWidget,
)
from PyQt5.QtGui import QTextCharFormat, QColor
import datetime

# --- 攻击标签常量 (与 arff 行的 binary label 对应) ----------------------
ATTACK_LABELS = ['normal', 'nmri', 'cmri', 'msci', 'mpci', 'mfci', 'dos', 'recon']
# ⭐⭐⭐ 2026-10-21 修正第三十五轮 (2026-10-02) 的数据集选择:
#   第三十五轮把默认从 IanArffDatasetF32.csv 改回 IanArffDataset.csv, 理由是
#   "用户要求原始 ARFF CSV, 时间字段是整数秒, H743 算 time_diff 用整秒差也够".
#   ⚠ 这个前提是错的, 而且改错了对象:
#     · IanArffDatasetF32.csv == IanArffDatasetRaw1.csv (md5 相同), 才是**原始 ARFF 的
#       CSV 形态**: 20 列 / "?" 保留 / time 带 6 位小数 (1418682163.170388).
#     · IanArffDataset.csv 是**退化版**: 22 列 (多 2 个空列) / "?" 被填 0 /
#       time 被四舍五入成整数秒.
#
#   实测代价 (同一 test 网格 3432 window, 部署 C kernel, thr=0.5):
#     官方训练 npy (亚秒 time)      : FP= 72  FPR= 4.43%  F1=86.86%
#     换成整数秒 time                : FP=213  FPR=13.09%  F1=84.88%   ← +141 误报
#   端到端 (H743 网格 200001..274628):
#     整数秒 IanArffDataset.csv      : FP=377  FPR=16.76%  F1=83.88%
#     亚秒   IanArffDatasetF32.csv   : FP=133  FPR= 5.91%  F1=86.71%   ← 追平训练管线
#
#   根因: 模型的 col15 (time_diff) / col16 (time_since_last_same_addr_func) 都在
#   KEEP_23 里, 训练时是亚秒精度 (1.5213 / 0.0997 / 1.6062 ...), 部署时喂整数秒
#   (1 / 0 / 2 / 0 / ...) → train/serve skew, 模型把大量正常帧判成攻击.
#   固件端无需改动: raw17_decode 用 be_f64 存 double, feature_extractor 也全程
#   double 算 time_diff (μs 精度), build_raw17_frame 本来就打包 f64 BE 时间戳.
DEFAULT_CSV = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"

# ⭐ 第二十九轮 (2026-10-02): Windows PyQt5 默认 QGroupBox::title 用 8pt Tahoma,
#   中文字符顶端/底端常被裁切 ("📋 日志" 中"日"被切, "📥 接收区" 中"接"被切).
#   修复: 给所有含中文的 QGroupBox 应用统一 QSS, 增大 title subcontrol padding + 上下空间.
_GROUPBOX_TITLE_QSS = (
    "QGroupBox { font-weight: bold; margin-top: 12px; }"
    "QGroupBox::title {"
    " subcontrol-origin: margin;"
    " subcontrol-position: top left;"
    " padding: 2px 8px;"
    " left: 8px;"
    "}"
)


def _style_chinese_groupbox(box: 'QGroupBox') -> None:
    """Apply _GROUPBOX_TITLE_QSS to a QGroupBox whose title contains Chinese chars.
    Windows PyQt5 default renders the title with 8pt Tahoma, which clips Chinese glyphs.
    """
    box.setStyleSheet(_GROUPBOX_TITLE_QSS)


# ⭐ 第四十一轮 (2026-10-02): 滚动条 QSS — Windows 默认滚动条太窄, 加宽到 12px,
#   hover 时变蓝 (与 GUI 整体配色一致), 滚动时显示深色 thumb.
_SCROLLBAR_QSS = (
    "QScrollBar:vertical {"
    "  background: #f0f0f0; width: 12px; margin: 0px;"
    "  border: 1px solid #d0d0d0;"
    "}"
    "QScrollBar::handle:vertical {"
    "  background: #b0b0b0; border-radius: 4px; min-height: 30px;"
    "  margin: 2px;"
    "}"
    "QScrollBar::handle:vertical:hover {"
    "  background: #0066cc;"
    "}"
    "QScrollBar::handle:vertical:pressed {"
    "  background: #003366;"
    "}"
    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
    "  height: 0; background: none;"
    "}"
    "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
    "  background: none;"
    "}"
)


def _apply_scrollbar_style(scroll: 'QScrollArea') -> None:
    """应用统一的滚动条样式到 QScrollArea 及其 viewport."""
    scroll.setStyleSheet(_SCROLLBAR_QSS)


# === Modbus RTU Frame (pure, CRC16, standard) =========================
# 不再有 CUSTOM_FRAME_SOP/HEADER/FOOTER,帧长由 PDU 决定 (4 - 256 字节)


def _safe_float(value, default=0.0) -> float:
    """安全把 CSV 字段转 float (失败 → default)。"""
    try:
        if value is None or str(value).strip() == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_uint(value, default=0, vmin=0, vmax=0xFFFF) -> int:
    """安全把 CSV 字段转 int 并 clamp 到 [vmin, vmax]。"""
    try:
        if value is None or str(value).strip() == '':
            return default
        x = int(float(value))
        if x < vmin:
            return vmin
        if x > vmax:
            return vmax
        return x
    except (ValueError, TypeError):
        return default


def modbus_crc16(data: bytes) -> int:
    """Modbus RTU CRC16 (poly 0xA001 reflected, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


# 12 个 CSV 字段 → Modbus 16-bit 寄存器 (x100 scaled, clamp int16)
REG_COLUMNS = (
    'setpoint', 'gain', 'reset rate', 'deadband',
    'cycle time', 'rate', 'system mode', 'control scheme',
    'pump', 'solenoid', 'pressure measurement', 'crc rate',
)


def _csv_row_to_regs(row: dict, n: int) -> bytes:
    """Encode up to ``n`` Modbus 16-bit registers from CSV float columns.

    Each float is scaled x100 and clamped to int16 range. Missing values
    become 0 (matching the H743 process_state fallback).
    """
    regs = []
    for col in REG_COLUMNS:
        v = _safe_float(row.get(col))
        regs.append(max(-32768, min(32767, int(round(v * 100)))))
        if len(regs) >= n:
            break
    # pad with zeros if fewer columns than n
    while len(regs) < n:
        regs.append(0)
    return struct.pack('>' + 'h' * n, *regs)


# === 17-dim 原始数据帧 (length 驱动, 6 种格式) ===========================
# 2026-09-19 第九轮: 改为 length 字段驱动编码. PC/H743 同步.
# 数据集 IanArffDataset.csv length 分布: 16 (134k) / 46 (69k) / 90 (64k)
#                                       / 10 (4k) / 12 (2k) / 14 (1k)
# length<16 全部 0 + 2B CRC, 是攻击信号; length=90 有真实浮点; length=16/46
# 是另一种攻击 (全部 0 特征 + 压力注入).
#
# 帧格式 (变长, 共 = length + 12 字节, 2026-09-24 第四轮 time 4B→8B f64):
#   [slave(1)] [func(1)] [features 4-15 (length B)] [is_response(1)] [time(8)]
#
# features 4-15 (length B) 按 length 字段分 3 种编码 + 攻击格式:
#   length = 16   → 10×1B uint8 (f4..13) + 1×4B float32 BE (f14) + 1×2B uint16 BE (f15)
#   length = 46   → 11×4B float32 BE (f4..14) + 1×2B uint16 BE (f15)
#   length = 90   → 11×8B float64 BE (f4..14) + 1×2B uint16 BE (f15)
#   length < 16   → body = 11×0x00 + 2B CRC = 13 字节, length 字段保留原值 ⭐ 攻击信号
#                   (2026-09-29 第十一轮修订: body_size 固定 13, 不依赖 length_field,
#                    总帧长恒 25 字节 = 1+1+1+13+1+8. H743 modbus_rtu.c 同步按
#                    RAW17_ATTACK_BODY_SIZE=13 解析, CRC 固定 body[11..12].)
#   其它 length  → 同 length<16 (防御性)
#
# CSV 列 → 特征号 (12 个原始特征 = f4..15):
#   col 3  setpoint         f4
#   col 4  gain             f5
#   col 5  reset rate       f6
#   col 6  deadband         f7
#   col 7  cycle time       f8
#   col 8  rate             f9
#   col 9  system mode      f10
#   col 10 control scheme   f11
#   col 11 pump             f12
#   col 12 solenoid         f13
#   col 13 pressure_meas.   f14
#   col 14 crc rate         f15
#
# 末尾:
#   col 15 command response → is_response (1B)        ⭐ H743 f[16]
#   col 16 time            → time (uint32 BE unix sec)

# 12 个原始特征对应的 CSV 列名 (按 f4..f15 顺序)
RAW17_FEAT_COLS = (
    'setpoint', 'gain', 'reset rate', 'deadband',
    'cycle time', 'rate', 'system mode', 'control scheme',
    'pump', 'solenoid', 'pressure measurement',
)


def build_raw17_frame(row: dict) -> bytes:
    """从 CSV 行构建 length 驱动的 17-dim 原始数据帧.

    第九轮 (2026-09-19): 改为 length 字段驱动, 5 种格式:
      - length=16:  10×uint8 + float32 + uint16(CRC)
      - length=46:  11×float32 + uint16(CRC)
      - length=90:  11×float64 + uint16(CRC)
      - length<16:  body 固定 11×0x00 + uint16(CRC) = 13 字节
                    length 字段保留 CSV 原值 (10/12/14)
                    ⭐ 攻击信号 (2026-09-29 第十一轮修订)
      - 其它:       body = 11×0x00 + uint16(CRC) = 13 字节 (同 length<16)
                    length 字段保留 CSV 原值

    Args:
        row: DictReader 行 (IanArffDataset.csv 的 17 列).

    Returns:
        (length + 12) 字节 raw17 帧. 末尾 1B is_response + 8B time (f64 BE).
    """
    # 1. header: slave + func + length (3B)
    slave = _safe_uint(row.get('address'), default=1, vmin=1, vmax=247)
    func  = _safe_uint(row.get('function'), default=3, vmin=0, vmax=255)
    length = _safe_uint(row.get('length'), default=16, vmin=0, vmax=255)

    # 2. is_response (1B)
    is_resp = 1 if _safe_uint(row.get('command response'),
                              default=0, vmin=0, vmax=1) == 1 else 0

    # 3. time (8B float64 BE) — 保留 ARFF 子秒精度 (如 .170388),
    #    否则 H743 f[13] (time_diff_sec) 整数差会让模型饱和输出 attack.
    #    ⚠️ float32 在 Unix 2026 时间戳下精度只有 ~50s, 必须用 float64.
    t = _safe_float(row.get('time'), default=0.0)
    time_bytes = struct.pack('>d', t)

    # 4. CRC_rate (uint16 BE) — features 4-15 body 末尾 2B
    crc_raw = _safe_float(row.get('crc rate'))
    crc_u16 = max(0, min(65535, int(round(crc_raw))))
    crc_bytes = struct.pack('>H', crc_u16)

    # 5. features 4-14 共 11 个 CSV 列 (col 3..13) 的浮点值
    feat_vals = [_safe_float(row.get(c)) for c in RAW17_FEAT_COLS]

    # 6. 按 length 编码 features 4-15 (length 字节 body)
    if length == 16:
        # 10×uint8 (f4..f13 = cols 3..12) + float32 BE (f14 = pressure) + uint16 BE (f15)
        body = bytearray()
        for v in feat_vals[:10]:
            body.append(max(0, min(255, int(round(v)))))
        body += struct.pack('>f', float(feat_vals[10]))   # pressure
        body += crc_bytes                                  # crc_rate
    elif length == 46:
        # 11×float32 BE (f4..f14) + uint16 BE (f15)
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>f', float(v))
        body += crc_bytes
    elif length == 90:
        # 11×float64 BE (f4..f14) + uint16 BE (f15)
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>d', float(v))
        body += crc_bytes
    else:
        # length<16 或未知: 攻击格式 = 11×0x00 + 2B uint16 CRC = 13 字节 body
        # 2026-09-29 第十一轮 (修订):
        #   length 字段保留 CSV 原值 (10/12/14/未知), 不要再覆盖成 13.
        #   body 大小固定为 13 字节 (11 个 0x00 + 末尾 2B CRC), 不依赖 length 字段.
        #   帧总长 = 1+1+1+13+1+8 = 25 字节.
        #   H743 modbus_rtu.c 已知: 对 length<16 帧, body_size=13 (不读 length_field).
        # 副作用:
        #   - 模型 length 特征保留原值 (-0.20/-0.13/-0.067), 不再被覆盖
        #   - H743 需要同步改 modbus_rtu.c 用 RAW17_ATTACK_BODY_SIZE=13 替代 length_field
        #   - 总帧长固定 25 字节, 简化 PC/H743 协议处理
        body = bytearray(11)
        body += crc_bytes

    # 7. 组帧: [slave(1)][func(1)][length(1)][body(length B)][is_resp(1)][time(8)]
    #    总长 = 3 + length + 1 + 8 = length + 12 (2026-09-24 第四轮 time 4B→8B f64)
    return bytes([slave, func, length]) + bytes(body) + bytes([is_resp]) + time_bytes


def parse_raw17_payload(payload: bytes) -> dict:
    """从 length 驱动的 payload 解码 (供 H743 仿真 / 调试用).

    payload 长度 ≥ 3 + length + 1 + 8 = length + 12 (2026-09-24 第四轮).
    返回 dict, features 4-14 统一为 float (可能因编码不同有精度差异).
    """
    if len(payload) < 8:
        return {}
    length = payload[2]
    is_attack = 1 if length < 16 else 0

    body = payload[3:3 + length]  # features 4-15
    if len(body) < length:
        body = body + b'\x00' * (length - len(body))  # 短截断填 0

    # 末尾 2B 是 CRC
    crc = struct.unpack_from('>H', body, max(0, length - 2))[0]

    # features 4-14 按 length 解码
    if length == 16:
        feats = [float(body[i]) for i in range(10)]   # 10 uint8
        if length >= 14:
            feats.append(struct.unpack_from('>f', body, 10)[0])  # float32
        else:
            feats.append(0.0)
    elif length == 46:
        feats = [struct.unpack_from('>f', body, i * 4)[0] for i in range(11)]
    elif length == 90:
        feats = [struct.unpack_from('>d', body, i * 8)[0] for i in range(11)]
    else:
        # length<16: features 4-14 全部 0
        feats = [0.0] * 11

    # is_response + time
    is_resp = payload[3 + length] if len(payload) > 3 + length else 0
    time_off = 3 + length + 1
    t = struct.unpack_from('>I', payload, time_off)[0] if len(payload) >= time_off + 4 else 0

    return {
        'address':     payload[0],
        'function':    payload[1],
        'length':      length,
        'is_attack':   is_attack,
        'setpoint':    feats[0] if len(feats) > 0 else 0.0,
        'gain':        feats[1] if len(feats) > 1 else 0.0,
        'reset':       feats[2] if len(feats) > 2 else 0.0,
        'deadband':    feats[3] if len(feats) > 3 else 0.0,
        'cycle':       feats[4] if len(feats) > 4 else 0.0,
        'rate':        feats[5] if len(feats) > 5 else 0.0,
        'system_mode': feats[6] if len(feats) > 6 else 0.0,
        'control':     feats[7] if len(feats) > 7 else 0.0,
        'pump':        feats[8] if len(feats) > 8 else 0.0,
        'solenoid':    feats[9] if len(feats) > 9 else 0.0,
        'pressure':    feats[10] if len(feats) > 10 else 0.0,
        'crc':         crc,
        'command':     is_resp,
        'time':        t,
    }


def build_modbus_rtu_frame(row: dict) -> bytes:
    """已废弃 (2026-09-19 第九轮): 改为 length 驱动 raw17_frame.

    保留此函数签名仅为向后兼容, 实际转发到 build_raw17_frame.
    """
    return build_raw17_frame(row)


# === 后台回放线程 ====================================================
class ReplayWorker(QThread):
    """后台线程:读 CSV → 按 time 间隔构 RTU 帧 → 经串口发出。"""
    progress = pyqtSignal(int, int, int, int)     # current, total, frames_sent, bytes_sent
    log = pyqtSignal(str)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)
    # ⭐⭐ 第十七轮: 每次发送攻击帧 (cat != 0) emit 当前累计攻击发送计数.
    #   IdsReportPanel 监听此信号显示 "Sent: N times".
    #   与 H743 上行的 [ALARM] N (Detected) 形成对比, 看检测率.
    attack_sent = pyqtSignal(int)                # 累计已发送攻击帧数
    # ⭐⭐ 第十九轮 (2026-09-19): 每帧发送后 emit (总发送, 攻击次数, 正常数),
    #   让 IdsReportPanel 实时显示 4 个 KPI:
    #     - 总发送 = total_sent_count   (binary=0 + binary=1 之和)
    #     - 攻击次数 = attack_sent_count (binary==1 时 +1)
    #     - 正常数 = normal_sent_count   (binary==0 时 +1)
    #   识别攻击次数由 IdsLineParser 从 H743 上行 [ALARM] N 解析, 不在此处.
    sent_stats = pyqtSignal(int, int, int)       # (total, attack, normal)
    # ⭐ 第五十一轮 (2026-10-09): per-window 统计信号 (16 帧 1 window max-aggregation)
    #   跟 H743 部署 1:1: 274k 帧 → 17164 windows, 8821 attack windows (max-aggregation 真值)
    window_stats = pyqtSignal(int, int, int)     # (total_w, attack_w, normal_w)
    # ⭐ 第二十轮 (2026-09-20): 每发一帧 emit 本帧类型 (binary==1?),
    #   让 RTUReplayGUI 通知 IdsLineParser, 用于下次 [ALARM] 时判 TP/FP.
    frame_sent_type = pyqtSignal(bool)
    # ⭐ 第二十一轮 (2026-09-20): 每发一帧 emit (n_bytes, is_attack),
    #   让 RTUReplayGUI 写 TestLogger. 之前只 emit bool, TestLogger 拿不到 n_bytes.
    frame_sent_log = pyqtSignal(int, bool)

    def __init__(self, csv_path: str, port: str, baud: int,
                 attack_filter: Optional[str],
                 start_frame: int = 0,
                 verbose: bool = False,
                 interval_ms: int = 0,
                 serial_obj: Optional['serial.Serial'] = None):
        super().__init__()
        self.csv_path = csv_path
        self.port = port
        self.baud = baud
        self.attack_filter = attack_filter  # None=all, 'normal', 'mfci', ...
        # ⭐ 2026-10-01 第二十三轮: 删除 speed/loop/synth_timing/fixed_10ms 参数.
        #   新增 start_frame: 从过滤后 play_rows[N] 开始发送 (默认 0 = 从头).
        self.start_frame = max(0, int(start_frame))
        self.verbose = verbose
        # interval_ms > 0 强制固定间隔 (1-1000ms), =0 走 CSV time 自然间隔
        self.interval_ms = max(0, int(interval_ms))
        # ⭐ 第二十一轮 (2026-09-20): 共享外部已开 serial (避免双开 COM 口失败)
        #   None = 自己开 (老路径),  非 None = 用传入的 (新路径, GUI 已开)
        self._shared_ser = serial_obj
        self._stopping = False
        self._paused = False
        self.frames_sent = 0
        self.bytes_sent = 0
        self.attack_sent_count = 0   # ⭐ 累计已发送攻击帧 (cat != 0)
        self.total_sent_count = 0    # ⭐ 第十九轮: 累计总发送帧数 (attack + normal)
        self.normal_sent_count = 0   # ⭐ 第十九轮: 累计正常帧数 (binary == 0)
        # ⭐ 第五十一轮 (2026-10-09): 模拟器按 16 帧 1 window 统计, 跟 H743 部署 1:1 对齐.
        #   训练: 16 帧 max-aggregation 决定 1 window label
        #   部署: H743 16 帧 1 window 1 alarm
        #   发送: 模拟器 16 帧 max-aggregation 决定 1 window attack (跟训练/部署 1:1)
        #   Attack KPI 从 60048 frames (per-frame) 变成 8821 windows (per-window, max-aggregation 真值)
        #   ⭐ 2026-10-09 修正: 60048 attack 帧散在 274628 帧里 (21.86%), max-aggregation 16 帧后
        #     实际算出 8821 attack windows (51.4% of 17164). 不是之前估算的 3753 (=60048/16 整除,
        #     假设 attack 帧聚在一起, 实际不是).
        self.windows_sent_count = 0         # 累计已发送 window 数 (frames_total // 16)
        self.windows_attack_sent_count = 0  # 累计已发送 attack window 数 (16 帧任一 attack)
        self.windows_normal_sent_count = 0  # 累计已发送 normal window 数 (16 帧全 normal)
        self._frames_in_current_window = 0  # 当前 window 已收帧数 (0..15)
        self._current_window_had_attack = False  # 当前 window 是否有任一 attack 帧
        self.ser: Optional[serial.Serial] = None

    def stop(self):
        self._stopping = True

    def toggle_pause(self):
        self._paused = not self._paused

    def run(self):
        # ⭐ 第二十一轮 (2026-09-20): 优先用外部传入的 serial,避免双开 COM 口
        if self._shared_ser is not None:
            self.ser = self._shared_ser
            self.log.emit(f"✓ 串口共享 (worker 用 GUI 句柄): {self.port} @ {self.baud} 8N1")
        else:
            try:
                self.ser = serial.Serial(
                    port=self.port, baudrate=self.baud,
                    bytesize=8, parity='N', stopbits=1,
                    timeout=0.1, write_timeout=0.5,
                )
                self.log.emit(f"✓ 串口打开: {self.port} @ {self.baud} 8N1")
            except Exception as e:
                self.error.emit(f"打开串口失败: {e}")
                return

        try:
            # 一次性读完整 CSV (16MB,~274k 行,放内存 OK)
            self.log.emit(f"读取 CSV: {self.csv_path}")
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            total = len(rows)
            self.log.emit(f"✓ CSV 读取完成: {total} 行")

            attack_filter = self.attack_filter
            if attack_filter and attack_filter != 'all':
                self.log.emit(f"过滤攻击标签: {attack_filter}")

            # 预过滤: 先按 attack 过滤掉不需要的行,只对通过的行回放
            # 这样循环内不再做 if/continue,主循环纯净
            CAT_TO_NAME = {0: 'normal', 1: 'nmri', 2: 'cmri', 3: 'msci',
                           4: 'mpci', 5: 'mfci', 6: 'dos', 7: 'recon'}

            def passes_filter(row):
                if not attack_filter or attack_filter == 'all':
                    return True
                try:
                    cat_int = int((row.get('categorized result') or '').strip())
                except (ValueError, TypeError):
                    return False
                if attack_filter == 'normal':
                    return cat_int == 0
                return CAT_TO_NAME.get(cat_int, '') == attack_filter

            play_rows = [r for r in rows if passes_filter(r)]
            self.log.emit(f"过滤后保留 {len(play_rows)}/{len(rows)} 行 "
                          f"(filter={attack_filter or 'all'})")

            # ⭐ 2026-10-01 第二十三轮: 删除循环 (while not self._stopping 外层)
            #   和 loop 收尾分支, 单次跑完即结束.
            # ⭐ 起始帧: 从 play_rows[start_frame] 开始, 默认 0 = 从头.
            actual_start = min(self.start_frame, max(0, len(play_rows) - 1))
            if actual_start != self.start_frame:
                self.log.emit(f"⚠ 起始帧 {self.start_frame} 超出范围, 修正为 {actual_start}")
            self.log.emit(f"▶ 起始帧: {actual_start} (共 {len(play_rows)} 行)")

            # 关键: 用过滤后起始行的时间当 baseline (CSV time 是 Unix 时间戳)
            baseline_t = 0.0
            if play_rows:
                try:
                    baseline_t = float(play_rows[actual_start].get('time', 0) or 0)
                except (ValueError, TypeError):
                    baseline_t = 0.0
            last_t = baseline_t

            for i in range(actual_start, len(play_rows)):
                row = play_rows[i]
                if self._stopping:
                    break
                while self._paused and not self._stopping:
                    time.sleep(0.05)

                # 时间间隔: ⭐⭐ 第二十三轮 (2026-10-01): 删除 CSV time × speed 倍率分支,
                #   只保留 2 种模式:
                #     • interval_ms > 0  → 强制固定 N ms 间隔 (覆盖 CSV time)
                #     • interval_ms == 0 → 走 CSV time 自然间隔 (1s 精度整数, ≈ 1帧/秒)
                if self.interval_ms > 0:
                    if self.frames_sent > 0:  # 第一帧不睡
                        time.sleep(self.interval_ms / 1000.0)
                else:
                    try:
                        t = float(row.get('time', 0) or 0)
                        dt_ms = max(0.0, (t - last_t) * 1000.0)
                        last_t = t
                        if dt_ms > 1:
                            time.sleep(dt_ms / 1000.0)
                    except (ValueError, TypeError):
                        pass

                # 构纯 Modbus RTU 帧 + 发 (PC 双向模拟, H743 被动监听)
                try:
                    frame = build_modbus_rtu_frame(row)
                    if frame is None:
                        continue  # 过滤掉的行
                    self.ser.write(frame)
                    self.frames_sent += 1
                    self.bytes_sent += len(frame)
                    # ⭐⭐ 第十八轮: 攻击判定改用 binary result 列 (Binary=1 = 攻击).
                    #   CSV header: binary result (0=normal, 1=attack), categorized result (0-7).
                    #   Binary 是论文标准的二分类标签, 与 H743 二分类 IDS 输出一致.
                    try:
                        binary_int = int((row.get('binary result') or '0').strip())
                    except (ValueError, TypeError):
                        binary_int = 0
                    # ⭐⭐ 第十九轮: 累加 3 个 IDS KPI 计数器.
                    #   攻击次数 increment 规则: binary==1 时 +1 (与论文 Binary 标签一致).
                    self.total_sent_count += 1
                    if binary_int == 1:
                        self.attack_sent_count += 1
                        self.attack_sent.emit(self.attack_sent_count)
                    else:
                        self.normal_sent_count += 1
                    self.sent_stats.emit(
                        self.total_sent_count,
                        self.attack_sent_count,
                        self.normal_sent_count,
                    )
                    # ⭐ 第五十一轮 (2026-10-09): per-window 累加 (16 帧 max-aggregation).
                    #   当前帧喂给当前 window, 凑齐 16 帧时评估 1 个 window:
                    #     - 任一帧 attack → windows_attack_sent_count += 1
                    #     - 全 normal      → windows_normal_sent_count += 1
                    #   跟 H743 部署 (Round 50 16 帧 1 window 1 alarm) 1:1 对齐.
                    #   Attack KPI 从 60048 frames → 8821 windows (max-aggregation 真值, 51.4% of 17164).
                    #   ⭐ 2026-10-09 修正: 不是简单 60048/16=3753, 实际数据散在 21.86%, 16 帧 max
                    #     后大多数 window 都有 attack 帧, 实际 8821.
                    if binary_int == 1:
                        self._current_window_had_attack = True
                    self._frames_in_current_window += 1
                    if self._frames_in_current_window >= 16:
                        self._frames_in_current_window = 0
                        self.windows_sent_count += 1
                        if self._current_window_had_attack:
                            self.windows_attack_sent_count += 1
                            win_tag = 'attack'
                        else:
                            self.windows_normal_sent_count += 1
                            win_tag = 'normal'
                        self._current_window_had_attack = False
                        # ⭐ 第五十一轮: 凑齐 1 window 时 emit window_stats
                        self.window_stats.emit(
                            self.windows_sent_count,
                            self.windows_attack_sent_count,
                            self.windows_normal_sent_count,
                        )
                        # ⭐⭐ 第五十二轮 b (2026-10-10): 默认 per-window 进度日志, 解决
                        #   "auto 模式看不到发送日志" 问题. 原 verbose 模式 (每 100 帧 1 条
                        #   [TX N] hex) 默认不开启, auto 跑 274k 帧时 log_text 完全没实时进度.
                        #   现在每凑齐 1 window (16 帧) emit 1 条, 跟 per-window 1:1,
                        #   既不刷屏 (274k 帧 → 17164 条), 又能让用户实时看到 attack/normal
                        #   分布. verbose 模式 (每 100 帧 [TX N] hex) 仍保留, 供调试用.
                        self.log.emit(
                            f"[TX 帧{self.frames_sent:>6}] "
                            f"window {self.windows_sent_count:>5} ({win_tag})  "
                            f"累计 {self.windows_attack_sent_count} attack + "
                            f"{self.windows_normal_sent_count} normal"
                        )
                    # ⭐ 第二十轮: emit 本帧类型, GUI 通知 parser 准备 TP/FP 判定
                    self.frame_sent_type.emit(binary_int == 1)
                    # ⭐ 第二十一轮: emit (n_bytes, is_attack) 给 TestLogger 写 SEND 事件
                    self.frame_sent_log.emit(len(frame), binary_int == 1)
                    if self.verbose and self.frames_sent % 100 == 1:
                        self.log.emit(
                            f"[TX {self.frames_sent}] {len(frame)}B: "
                            f"{frame.hex(' ')}"
                        )
                except serial.SerialTimeoutException:
                    self.error.emit(f"写入超时 (row {i})")
                except Exception as e:
                    self.error.emit(f"写入失败: {e}")

                if (i - actual_start) % 50 == 0:
                    self.progress.emit(i + 1, total, self.frames_sent,
                                        self.bytes_sent)

            self.progress.emit(total, total, self.frames_sent, self.bytes_sent)
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(f"运行错误: {e}")
        finally:
            # ⭐ 第二十一轮: 共享的 serial 由 GUI 关, worker 不动
            if self.ser and self._shared_ser is None:
                try:
                    self.ser.close()
                except Exception:
                    pass
            avg = self.bytes_sent / max(1, self.frames_sent)
            self.log.emit(f"✓ 完成: {self.frames_sent} 帧 / {self.bytes_sent} 字节 "
                          f"(平均 {avg:.1f} B/帧)")


# === RX 区辅助 ========================================================
def _format_hex_ascii(data: bytes) -> str:
    """Render bytes as ``HH HH HH | ccc`` (hex on left, ASCII on right)."""
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset:offset + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        hex_part = hex_part.ljust(16 * 3 - 1)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{offset:04x}  {hex_part}  |{ascii_part}|")
    return "\n".join(lines)


def _format_hex_only(data: bytes) -> str:
    return " ".join(f"{b:02x}" for b in data)


def _format_ascii_only(data: bytes) -> str:
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data)


class RxPanel(QWidget):
    """Hex+ASCII 接收查看器 (清空按钮 + 格式下拉 + 字节计数).

    ⭐ 2026-10-01 第二十三轮+round24: 加回 HEX 显示 (用户撤回 ASCII-only 决定).
    ⭐ 第三十一轮 (2026-10-02): 默认改为 ASCII (用户偏好), 可切 hex+ASCII / hex 单独查看.
    """

    cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._byte_count = 0
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.clear_btn = QPushButton("\U0001f5d1 清空")
        self.clear_btn.clicked.connect(self._on_clear)
        self.format_combo = QComboBox()
        # ⭐ 第三十一轮: 默认 ASCII (index 0), 可切 hex / hex+ASCII
        self.format_combo.addItems(["ASCII", "hex", "hex+ASCII"])
        self.byte_count_label = QLabel("0 bytes")
        toolbar.addWidget(self.clear_btn)
        toolbar.addWidget(QLabel("格式:"))
        toolbar.addWidget(self.format_combo)
        toolbar.addStretch(1)
        toolbar.addWidget(self.byte_count_label)
        outer.addLayout(toolbar)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        font = self.text_edit.font()
        font.setFamily("Consolas, Courier New, monospace")
        font.setPointSize(10)  # ⭐ 第三十二轮: 显式设 10pt, 避免默认 8pt 字符裁切
        if QFont is not None:
            font.setStyleHint(QFont.Monospace)  # ⭐ 确保无 Consolas 时降级等宽
        self.text_edit.setFont(font)
        # ⭐ 第三十二轮 (2026-10-02): 文字显示不全 — 加 padding + 行高, 防字符底端裁切.
        #   根因: QTextEdit 默认 stylesheet 无 padding, 默认 line-height 太紧,
        #   Consolas 字符 (g/p/q 下伸部) 底端被裁切, 长行 wrap 后第二行顶部也裁.
        self.text_edit.setStyleSheet(
            "QTextEdit { padding: 4px 6px; line-height: 1.4em; }"
            "QTextEdit:focus { border-color: #66afe9; }"
        )
        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)  # 默认值, 但显式声明
        # 不强制 NoWrap, hex+ASCII 长行按 widget 宽度 wrap
        outer.addWidget(self.text_edit, stretch=1)

    def _on_clear(self):
        self.text_edit.clear()
        self._byte_count = 0
        self.byte_count_label.setText("0 bytes")
        self.cleared.emit()

    def append_bytes(self, data: bytes) -> None:
        if not data:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # ⭐ 第二十三轮+round24: 按 format_combo 选择显示格式
        fmt = self.format_combo.currentText()
        if fmt == "hex":
            body = _format_hex_only(data)
        elif fmt == "ASCII":
            body = _format_ascii_only(data)
        else:
            body = _format_hex_ascii(data)
        # ⭐ 第三十一轮 (2026-10-02): 时间和内容放同一行 (用空格分隔, 不换行)
        #   旧格式 "[时间 + 字节数]\n[内容]" → 新格式 "[时间 + 字节数]  [内容]"
        self.text_edit.append(f"[{ts}] RX ({len(data)} B): {body}")
        self._byte_count += len(data)
        self.byte_count_label.setText(f"{self._byte_count} bytes")

        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()


# === IDS 上报解析器 (第十六轮) ============================================
# H743 端已删除 JSON 上行, 改为最简单的 printf("[ALARM] %lu\r\n", count).
# PC 端也对应简化: 按行切分, 正则匹配 "[ALARM] N", emit count_changed(int).
#
# 与之前 JSON 解析对比:
#   - 旧: IdsJsonParser 解析 {"src":"ids","ver":1,"type":"alarm","count":N}
#   - 新: IdsLineParser 解析 [ALARM] N (单行, 16B, 无 JSON 协议开销)
#
# 不再有 hello 握手检测 (H743 不再发 hello), 连通性靠用户观察串口
# 是否还在输出 [ALARM] / 启动 banner 来判断.

import re as _re

class IdsLineParser(QObject):
    """Buffer 串口 RX 字节, 按 \\n 切行, 正则匹配 '[ALARM] <count>' 行.

    第二十轮 (2026-09-20): 增加 fp_changed 信号,emit 时同时给出当前 alarm 是
    TP 还是 FP (看上一帧 PC 是不是 attack). 这样 IdsReportPanel 能直接显示
    误报率 FPR,不需要事后比对串口日志.

    第二十二轮 (2026-09-30): 修复 is_fp 标签逻辑 — 改用滑动窗口 (size=16).
    H743 IDS 推理基于 sliding window=16 累积特征, ALARM 反映整窗口 (640ms).
    旧逻辑只看上一帧 binary 会导致 attack burst 内只有首帧标 TP, 其余 15 个
    误标 FP. 例如 2026-09-29 auto-40ms 测试 F1=25.13%, 修正后真实 F1≈51%.
    """
    count_changed = pyqtSignal(int)            # [ALARM] N 当前累计 count
    fp_changed    = pyqtSignal(int, bool)      # (count, is_fp) — True=误报 (FP)
    # ⭐ 第二十一轮: 增加 prob 字段信号 (含 H743 上行的 prob=0.NNN 1000 倍值)
    alarm_with_prob = pyqtSignal(int, int, bool, int)  # (count, prob_x1000, is_fp, h743_frame) — ⭐ 第六十一轮 + frame
    # ⭐ 第六十五轮 (2026-10-13): frame→binary 标注信号. GUI 监听此信号, 在 RX 区域
    #   追加 "[ALARM] frame=K → 实际是 attack/normal" 备注, 让用户实时看到 H743
    #   判到的帧是 attack 还是 normal, 不必事后查 CSV 对照. (count, prob_x1000,
    #   label, h743_frame). label ∈ {"attack", "normal", "?"}.
    alarm_annotated = pyqtSignal(int, int, str, int)  # (count, prob_x1000, label, h743_frame)
    # ⭐ 2026-10-13 Round 69: per-window 路由修复.
    #   IdsLineParser 自身有 _window_eval (line 840) 但只接收 on_alarm_with_frame, 不接收
    #   per-frame attack ground truth (那条信号连到 ids_report._window_eval).
    #   ids_report._window_eval 跟训练 1:1, 是 summary 真正读的对象, 但它收不到 alarm.
    #   结果: summary TP/FP 永远是 0. 解法: IdsLineParser 解析到 h743_frame 时 emit 此信号,
    #   main_window 把信号喂到 ids_report._window_eval.on_alarm_with_frame, 让 IdsReportPanel
    #   的 WindowEvaluator 同时看到 attack 状态 + alarm 状态, 正确算 TP/FP.
    alarm_frame_index = pyqtSignal(int)  # h743_frame, 每个内部 alarm 发一次 (1 个 [ALARM] 可能 emit delta 次)

    # ⭐ 第二十三轮 (2026-09-30): delta-driven TP/FP 信号
    #   H743 固件 modbus_ids.c:271 节流 (MIN_GAP_FRAMES=4): 每次判 attack frames_attack++,
    #   但只发 ~40% 的 [ALARM] 上行消息. 旧逻辑按 [ALARM] 消息数算 TP/FP 会丢失 60%.
    #   新逻辑按 frames_attack 增量算, 每个 H743 内部 alarm 都正确标签.
    #   信号携带 (delta, cur_count, prob_x1000) — delta 给计数用, cur_count/prob 给日志用.
    tp_delta = pyqtSignal(int, int, int)  # (delta, cur_count, prob_x1000)
    fp_delta = pyqtSignal(int, int, int)  # (delta, cur_count, prob_x1000)

    # ⭐ 第二十二轮: 滑动窗口大小 (H743 TCN+SE 训练时 window=16)
    _SW_SIZE = 16
    # ⭐ 第五十轮 (2026-10-09): 移除 _CLASSIFY_WINDOW, 改用 WindowEvaluator per-window 评估.
    #   H743 部署改成 non-overlapping (16 帧 1 次推理, 1 window 1 alarm),
    #   跟 PC 训练 preprocess_v2_scada.py reshape(n//16, 16, -1) 1:1 对齐.
    #   评估单元 = 1 个 16 帧 window, 不再是 1 帧或 1 alarm vs 1 帧.
    #   历史 (per-alarm 评估窗口演变, 已废弃):
    #     Round 27 (window=5): 工业 IDS 常见做法, F1=0.90, Recall 94%
    #     Round 46 (window=2): 用户"晚 1 周期也记 TP" → 当年为收紧窗口
    #     Round 47 (window=16): "部署跟训练一致" max-aggregation 1:1
    #     Round 48 (window=1): "窗口缩小到 1" 严格 per-frame
    #     Round 49 (window=2): "窗口改为 2" 容忍 lag 0-1
    #     Round 50 (本轮, per-window 替换 per-alarm): "在 H743 上按 PC 方式部署"

    _ALARM_RE = _re.compile(rb'\[ALARM\]\s+(\d+)')
    _PROB_RE  = _re.compile(rb'prob=0\.(\d{1,3})')
    # ⭐ 第六十一轮 (2026-10-11): H743 firmware 现在发 frame=N, PC 端用其精确归属
    #   alarm 到 window (消除 37ms pipeline lag 把 alarm 推到下个 window 的错位).
    #   PC 端 _PROB_RE 已存在但 H743 之前不发 prob, 现 Round 61 一起补发.
    _FRAME_RE = _re.compile(rb'frame=(\d+)')

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buf = bytearray()
        # ⭐ 第五十轮 (2026-10-09): 改用 WindowEvaluator per-window 评估.
        #   H743 部署改成 16 帧 1 window 1 次推理, 1 window 1 alarm (跟 PC 训练 1:1).
        #   评估粒度 = 1 个 16 帧 window, 而不是 1 帧 vs 1 alarm.
        #   WindowEvaluator 内部维护:
        #     - _frame_count: 累计 PC 发送帧数
        #     - _current_window_attack: 当前 16 帧窗是否有 attack (ground truth)
        #     - _recent_alarm_frames: 最近 16 帧内 H743 上行的 alarm 位置
        #   凑齐 16 帧时 WindowEvaluator 内部评估 (TP/FP/TN/FN), 自动重置窗.
        self._window_eval = WindowEvaluator()
        # 移除 _tp_budget: per-window 评估不需要 budget cap
        #   (H743 1 window 1 alarm, 不存在 over-fire, 因为 non-overlapping 1:1 对齐训练)
        # 保留 _last_alarm_count: 仍用于 [ALARM] 消息的 delta 计算
        self._last_alarm_count: Optional[int] = None
        # ⭐ 第五十轮: 初始化 _last_emit_tp/fp, 避免 dummy QObject 走 getattr 报 _DummyMethod
        self._last_emit_tp: int = 0
        self._last_emit_fp: int = 0
        # ⭐ 第六十三轮 (2026-10-12): lag 补偿. 记录每 PC 发送帧的 unix-ms 时间戳,
        #   [ALARM] 上行时算 arrival_t - send_t = lag_ms, 传给 on_alarm_with_frame
        #   反向 shift window 归属. 只保留最近 _FRAME_TS_CAP 个 frame 的 timestamp
        #   (FIFO 内存上限), 避免长时间跑测试后字典无限增长.
        self._pc_frame_count: int = 0
        self._frame_send_ts: "dict[int, int]" = {}
        self._FRAME_TS_CAP = 256
        # ⭐ 第六十五轮 (2026-10-13): frame → binary 映射. 记录每 PC 发送帧的 ground
        #   truth (0=normal, 1=attack), [ALARM] 上行时查 h743_frame 对应帧的 binary,
        #   走 alarm_annotated 信号在 RX 区域追加 "attack"/"normal" 备注. 用户能直接
        #   看到 H743 实际判到的是哪一帧, 不必事后查 CSV 对照.
        self._pc_frame_binary: "dict[int, int]" = {}

    def set_last_was_attack(self, val: Optional[bool]) -> None:
        """PC GUI 每次发完一帧后调用, 推送该帧 binary 给 WindowEvaluator.

        Args:
            val: True=attack (binary==1), False=normal (binary==0),
                 None=未初始化 (清空窗口, 不再 emit TP/FP 计数)
        """
        if val is None:
            self.reset_last_was_attack()
            return
        # 兼容旧 API: val=True/False 都喂给 WindowEvaluator
        # 旧名 set_last_was_attack 仍保留以兼容 frame_sent_type.connect(...)
        self._window_eval.on_frame(bool(val))

    def reset_last_was_attack(self) -> None:
        """重置 PC 评估状态 (测试开始前/手动清零时调用)."""
        self._window_eval.reset()
        # ⭐ 第二十三轮: 同时重置 alarm_count 跟踪
        self._last_alarm_count = None
        # ⭐ 第六十三轮 (2026-10-12): lag 补偿 timestamp 状态也一并重置
        self._pc_frame_count = 0
        self._frame_send_ts.clear()
        # ⭐ 第六十五轮 (2026-10-13): frame→binary 映射同步重置, 避免 stale label
        self._pc_frame_binary.clear()

    def _on_pc_frame_sent(self, is_attack: bool) -> None:
        """⭐ 第六十三轮 (2026-10-12): 镜像 worker.frame_sent_type, 记每 PC 发送帧的时间戳.

        用于 [ALARM] 上行时计算 lag_ms = arrival_t - send_t, 传给 on_alarm_with_frame
        反向 shift window 归属. 跟 set_last_was_attack 共用 frame_sent_type 信号 (RTUReplayGUI
        多订阅 fan-out), 不需要扩展现有信号签名.

        ⭐ 第六十五轮 (2026-10-13): 同时记 frame_idx → binary 映射, 供 alarm_annotated
        信号查 h743_frame 的 ground truth 标签 (attack / normal). 同样 FIFO 截断
        _pc_frame_binary, 跟 _frame_send_ts 共享 _FRAME_TS_CAP 上限.

        Args:
            is_attack: True=attack frame (binary=1), False=normal (binary=0). 参数本身不
                       用于 lag 补偿逻辑, 仅作为存在性哨兵 (signal-driven 调用).
        """
        self._pc_frame_count += 1
        self._frame_send_ts[self._pc_frame_count] = int(time.time() * 1000)
        # ⭐ 第六十五轮: 同步记 binary 标签
        self._pc_frame_binary[self._pc_frame_count] = 1 if is_attack else 0
        # FIFO 截断, 避免长时间跑测试后字典无限增长 (only need recent frames for lag lookup)
        if len(self._frame_send_ts) > self._FRAME_TS_CAP:
            excess = len(self._frame_send_ts) - self._FRAME_TS_CAP
            oldest = sorted(self._frame_send_ts.keys())[:excess]
            for k in oldest:
                del self._frame_send_ts[k]
                self._pc_frame_binary.pop(k, None)  # ⭐ 第六十五轮: 同步清掉 binary

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self._buf.extend(data)
        # 按 \n 切 (容忍 \r\n 或 \n)
        while True:
            idx = self._buf.find(b'\n')
            if idx < 0:
                # 限制 buffer 长度, 防止恶意/异常把内存撑爆
                if len(self._buf) > 8192:
                    del self._buf[:len(self._buf) - 4096]
                return
            line = bytes(self._buf[:idx]).strip(b'\r').strip()
            del self._buf[:idx + 1]
            if not line:
                continue
            self._parse_line(line)

    def _parse_line(self, line: bytes) -> None:
        m = self._ALARM_RE.search(line)
        if not m:
            return  # 非 [ALARM] 行 (启动 banner / Modbus echo) 忽略
        try:
            count = int(m.group(1))
        except ValueError:
            return
        # ⭐ 第二十一轮: 解析 prob=0.NNN (格式: H743 printf("[ALARM] %lu prob=0.%03u")
        #   → 截取 3 位整数作为 prob_x1000, 范围 [0, 1000]. 未匹配则 -1.
        prob_match = self._PROB_RE.search(line)
        prob_x1000 = int(prob_match.group(1)) if prob_match else -1
        # ⭐ 第六十一轮 (2026-10-11): 解析 frame=K (H743 firmware 现在的 printf 包含
        #   frames_total). 这是 H743 端的权威 frame 编号, 直接用来 attribute window,
        #   消除 PC 端 _frame_count-1 近似路径引入的 0.9-frame lag 错位.
        frame_match = self._FRAME_RE.search(line)
        if frame_match:
            try:
                h743_frame: int = int(frame_match.group(1))
            except ValueError:
                h743_frame = 0
        else:
            h743_frame = 0  # legacy 旧 firmware, 不发 frame
        # ⭐ 第六十五轮 (2026-10-13): 查 h743_frame 对应 ground truth 标签.
        #   H743 训练时用 window-level max-aggregation: 16 帧里任一 attack → 整窗 attack.
        #   部署时 H743 报 alarm 表示 "当前 16 帧 window 是 attack". 所以标注应该按
        #   window-level 判断, 不是单帧.
        #
        #   ⚠ 帧号说明: 实测 logs/test_20261013-001719 显示 h743_frame 与 PC
        #   running_total 几乎 1:1, H743 略 lag 1-2 帧 (alarm h743=48 时 PC=49,
        #   alarm h743=64 时 PC=66, alarm h743=80 时 PC=81). 用 h743_frame 直接
        #   当 PC frame index, 窗口扫描 1-2 帧 lag 在 window=16 范围内可忽略.
        if h743_frame > 0:
            # 直接 1:1 映射, 容忍 H743 1-2 帧 lag
            # ⭐⭐ 2026-10-21: 跟 WindowEvaluator.on_alarm_with_frame 同步修正 +1 抖动.
            #   H743 的窗口末帧恒为 16 的倍数, frame%16 != 0 说明是迟报 1 帧, 窗口号退 1.
            W = (h743_frame - 1) // 16
            if h743_frame % 16 != 0:
                W -= 1
            # 扫 window W 的 16 帧, 看有没有 attack
            window_has_attack = False
            window_has_normal = False
            for offset in range(16):
                pc_idx = W * 16 + offset + 1
                bin_ = self._pc_frame_binary.get(pc_idx, -1)
                if bin_ == 1:
                    window_has_attack = True
                elif bin_ == 0:
                    window_has_normal = True
            if window_has_attack:
                label = "attack"  # 整个 window 是 attack (max-aggregation 1:1)
            elif window_has_normal:
                label = "normal"  # window 全 normal
            else:
                label = "?"
            self.alarm_annotated.emit(count, prob_x1000, label, h743_frame)
        # count_changed 始终 emit (IdsReportPanel._on_detected 用此信号更新 Detected KPI
        #   = H743 frames_attack delta).
        self.count_changed.emit(count)
        # ⭐ 第二十三轮 (2026-09-30): delta-driven TP/FP 计数
        #   H743 固件 modbus_ids.c:271 节流 (MIN_GAP_FRAMES=4): 每次判 attack frames_attack++,
        #   但只发 ~40% 的 [ALARM] 上行. 旧逻辑按 [ALARM] 消息算 TP/FP 丢失 60% 真实检测.
        #   delta = cur_count - last_count 是 H743 内部真实判 attack 次数.
        if self._last_alarm_count is None:
            # 首次 [ALARM]: H743 从 0 累计到 count, delta = count (不是 1!)
            # 例如: H743 boot 后已判 5 次 attack, 第一次上行 alarm_count=5 → delta=5
            delta = count
        else:
            delta = count - self._last_alarm_count
            if delta < 1:
                delta = 1  # 异常 (H743 重启/计数回退), 保守按 1 处理
        self._last_alarm_count = count
        # ⭐ 第二十二轮 (2026-09-30): 滑动窗口评估 TP/FP (H743 推理基于 SW=16)
        # ⭐ 第二十四轮 (2026-10-01): 改为逐帧回放 — 用每个内部 alarm 对应的预测帧
        #   binary 分类, 而不是窗内是否有 attack. 修复 Round 22 错标 bug:
        #   H743 attack 流尾部接下来 15 个 normal 帧仍报警 (窗内还有 attack),
        #   这些 delayed alarms 被错标 TP, 实际预测帧是 normal, 应算 FP.
        #   Round 23 delta-driven 放大 bug: burst 内所有内部 alarm 共用一个 is_fp,
        #   错标的 TP 乘以 delta 倍数累加. Option A 真正逐帧分类, 完美修复.
        #
        #   索引规则: H743 frames_attack 序号 [last+1, count] 对应 PC 最近 delta 个发送帧
        #   _binary_history[-delta..-1] 是最近 delta 帧的 binary 序列.
        #   第 i 个内部 alarm (i=0..delta-1) 对应 _binary_history[-(delta-i)].
        #   如果 binary_history 不够 (H743 远超 PC 发送速度, 极端情况), 保守按
        #   "current frame" (window 最后一个) 分类.
        if self._window_eval._frame_count > 0:
            # ⭐ 第五十轮 (2026-10-09): 改用 WindowEvaluator per-window 路径.
            #   每个 H743 内部 alarm (delta 个) 调 on_alarm, 凑齐 16 帧时
            #   WindowEvaluator 自动评估 TP/FP/TN/FN (跟训练 max-aggregation 1:1).
            #   emit tp_delta/fp_delta 仍保留以兼容 IdsReportPanel 旧 UI 路径.
            #
            # ⭐ 第六十一轮 (2026-10-11): H743 firmware 现在发 frame=K, 用 K 作 H743 权威
            #   frame 编号, 走 _direct_alarms 精确路径, 替代 _recent_alarm_frames 近似.
            #   delta 个内部 alarm 均匀映射到 [K-delta+1, K], 避免 legacy 路径的
            #   0.9-frame lag 错位导致的 FP inflation (实测 2001 → 期望 ~600).
            if h743_frame > 0:
                # 新 firmware 路径: 用 H743 frame 编号精确 tag window (Round 61).
                # ⭐ 第六十三轮 (2026-10-12): 取消 lag 补偿. 95.3% alarms lag>=40ms 时目标
                #   window 已 evaluate, _direct_alarms.add(W) 永远 stale. lag_ms tracking
                #   (_frame_send_ts / _on_pc_frame_sent) 保留仅供未来诊断.
                # ⭐ 2026-10-13 Round 69: 同时 emit alarm_frame_index 信号, 让 main_window
                #   转发到 ids_report._window_eval.on_alarm_with_frame, 修复 per-window
                #   summary TP/FP=0 bug (IdsReportPanel 的 WindowEvaluator 之前收不到 alarm).
                for i in range(delta):
                    # 第 i 个 (0-indexed) 内部 alarm → frame (K - (delta-1) + i)
                    fr = h743_frame - (delta - 1) + i
                    if fr < 1:
                        fr = 1  # 防御: 不允许 frame<=0
                    self._window_eval.on_alarm_with_frame(fr)
                    self.alarm_frame_index.emit(fr)
            else:
                # Legacy 旧 firmware 路径: 仍用 _frame_count-1 近似 (不优化, 因为旧固件)
                for _ in range(delta):
                    self._window_eval.on_alarm()
            # 从 WindowEvaluator 拿当前累计 TP/FP, 算 delta 后 emit
            cur_metrics = self._window_eval.get_metrics()
            cur_tp = cur_metrics['tp']
            cur_fp = cur_metrics['fp']
            prev_tp = getattr(self, '_last_emit_tp', 0)
            prev_fp = getattr(self, '_last_emit_fp', 0)
            tp_delta_count = cur_tp - prev_tp
            fp_delta_count = cur_fp - prev_fp
            for _ in range(tp_delta_count):
                self.tp_delta.emit(1, count, prob_x1000)
            for _ in range(fp_delta_count):
                self.fp_delta.emit(1, count, prob_x1000)
            self._last_emit_tp = cur_tp
            self._last_emit_fp = cur_fp
            # Legacy 信号 (单消息级, deprecated, 保留以兼容旧连接).
            # 第五十轮: per-window 评估下 alarm 本身没有 "FP" 标签 (由 WindowEvaluator
            # 在 16 帧边界判), 这里全归 False.
            self.fp_changed.emit(count, False)
            # ⭐ 第六十一轮 (2026-10-11): 多带 h743_frame 给 log_alarm 写 CSV 第 3 列,
            #   诊断时可直接看到 H743 实际发了哪个 frame, 不必依赖 GUI 解析逻辑.
            self.alarm_with_prob.emit(count, prob_x1000, False, h743_frame)


# === 第四十八轮 (2026-10-08) Per-Window 评估器 (跟训练 1:1 对齐) =================
# 训练 (preprocess_v2_scada.py:154-158):
#   yw = max(y_bin in window=16)  # 16 帧任一 attack → 整窗 label=1
# 部署 (本类): 16 帧非滑动窗口评估
#   - Ground truth: max(CSV binary in window) — 来自 PC set_last_was_attack
#   - H743 prediction: any alarm in past 16 frames (lag-based) — 来自 count_changed
#   - 单元: 16 帧 window (274628 帧 → 17164 windows) vs per-alarm (65276 alarms)
#   - 评估粒度更粗, 但与训练验证 1:1, 指标可直接对比训练 F1m/Recall
#   - 不需要 budget cap, 因为 H743 prediction 不会"过度覆盖"
class WindowEvaluator:
    """Per-window confusion matrix consistent with training (max-aggregation).

    ⭐ Round 48 (2026-10-08): 部署评估.
      训练: yw = max(y_bin in window=16), 每个 16 帧窗 = 1 个 label (0/1)
      部署: 每 16 帧窗 = 1 个 sample, label = max(binary), prediction = any H743 alarm in past 16 帧

    ⚠️ Round 48 末轮调整: IdsLineParser._CLASSIFY_WINDOW = 1 (per-alarm 严格同帧才算 TP).
      跟 per-alarm 评估 (Round 47, window=16) 区别:
      - 单元: 16 帧 window (17164 个 for 274k) vs 1 alarm (65276 个)
      - 不需要 budget cap, 因为 H743 prediction 不会过度覆盖 (1 alarm 最多覆盖 16 帧)
      - 指标可与训练 F1m/Recall 直接对比, 监测模型部署性能
    """

    _WINDOW_SIZE = 16

    def __init__(self) -> None:
        from collections import deque
        self._frame_count: int = 0
        self._current_window_attack: bool = False
        self._current_window_alarm: bool = False
        self._recent_alarm_frames: "deque[int]" = deque(maxlen=16)
        # ⭐ 第六十一轮 (2026-10-11): H743 firmware 现在发 frame=N, PC 端 IdsLineParser
        #   解析后调 on_alarm_with_frame(N) 直接 tag window=(N-1)//16, 消除 37ms pipeline
        #   lag 引起的 1-3 帧错位 (旧 firmware 不发 frame, 走 on_alarm() legacy deque 路径).
        self._direct_alarms: "set[int]" = set()
        self._tp: int = 0
        self._fp: int = 0
        self._tn: int = 0
        self._fn: int = 0
        self._n_windows: int = 0
        # ⭐ Round 64 (2026-10-13): re-evaluate past window when late alarm arrives.
        #   95.3% alarms lag ≥40ms (mean 80ms, max 185ms). _evaluate_window 在 frame 16K
        #   同步 fire, 目标 window K-1 评估时 alarm 还没到 → 错标 TN/FN. Alarm 80ms 后到
        #   PC, _direct_alarms.add(K-1) 但 K-1 已 evaluate, 永远 stale. 解决: 把最近 4 个
        #   window 状态存进 _recent_evals buffer, alarm 到达时检查是否命中 (W matches) 且
        #   当时 alarm=False, 是的话回滚 TN→FP 或 FN→TP. Buffer size=4 覆盖 lag ≤160ms
        #   (4 frames × 40ms), 99.5% alarms 在这个范围内.
        self._recent_evals: "deque[dict]" = deque(maxlen=4)

    def on_frame(self, is_attack: bool) -> None:
        """每 PC 发完一帧调用. 累积到 16 帧时评估并开新窗.

        Args:
            is_attack: True=本帧 binary=1, False=本帧 binary=0
        """
        # Ground truth: 本帧 attack → 当前窗有 attack
        if is_attack:
            self._current_window_attack = True
        # H743 prediction: 过去 16 帧有 alarm → 当前窗有 alarm
        if self._recent_alarm_frames:
            self._current_window_alarm = True
        self._frame_count += 1
        # 满 16 帧就评估
        if self._frame_count % self._WINDOW_SIZE == 0:
            self._evaluate_window()

    def on_alarm(self) -> None:
        """Legacy: H743 上行 alarm 时调用. 把"最近一帧"加入 recent_alarm_frames.

        假设: H743 alarm 在 PC 端接收时, 它对应的 PC 帧是"最近刚发的帧" (auto-40ms 测试
        串口延迟 < 帧间隔). 这个近似对 lag≤1 帧的 alarm 准确, lag 更大的 alarm 也算
        在最近 16 帧范围内, 不影响窗口评估.

        ⚠️ 第六十一轮 (2026-10-11): 实测 lag mean=37ms median=39ms (>1 帧), 导致
        ~1200 个本属 window W 的 alarm 被推到 W+1 误标 FP. 新 firmware 发 frame_idx,
        IdsLineParser 优先调 on_alarm_with_frame() 走 _direct_alarms 精确路径.

        ⭐ 第六十三轮 (2026-10-12): 双路径并存.
        Direct path (on_alarm_with_frame, Round 61) 精确归 W=(K_H-1)//16. 仅 4.7% alarms
        (lag<40ms) 真正有效, 其余 95.3% 因 _evaluate_window 已 fire 而 stale.
        Legacy deque path 错归到 NEXT 窗口, 但对 attack bursts 跨 window 的场景仍然
        TP (下一个 window 仍 attack). 实测 legacy 路径贡献 ~6969 TPs 是 F1=78.34% 的
        主源. 取消第六十三轮 lag 补偿 (反向 shift=W-(K_H-1)//16 - shift) 因为
        shift=1 时落到 K_H-2 (更早的已 evaluate 窗口) 同样 stale. 保留双路径.
        """
        if self._frame_count > 0:
            self._recent_alarm_frames.append(self._frame_count - 1)

    def on_alarm_with_frame(self, frame_idx: int) -> None:
        """⭐ 第六十一轮 (2026-10-11): H743 发 frame=K_H, PC 直接 tag window (K_H-1)//16.

        不依赖 lag 估算, 也不需要等 deque 自然归并.

        ⭐ 第六十三轮 (2026-10-12): 双路径并存, 取消 lag 补偿. 原因: lag_ms 在 live
        code 里 95.3% 的 alarm 都 >= 40ms, 此时目标 window K=(K_H-1)//16 已 _evaluate
        过了, _direct_alarms.add(K) 永远 stale. 反向 shift 到 K-1 同样 stale (更早 evaluate).
        唯一有 live 效力的路径是 legacy deque 的 +1 forward shift, 保留. Direct path 仍
        给 4.7% 低 lag alarm 提供精确归 W 的 TP.

        ⭐ Round 64 (2026-10-13): re-evaluate past window. 检查 _recent_evals buffer
        (最近 4 个已 evaluate window) 是否有 W matches, 且当时 alarm=False (说明 alarm
        晚了没赶上 eval). 是的话回滚 metric: TN→FP (attack=False, alarm 到了说明
        模型判断 attack=True, 但 ARFF 标 normal) 或 FN→TP (attack=True, alarm 到了
        正确归 W, 应是 TP 不是 FN). 解决 lag-induced 错位.

        ⭐ 第六十六轮 (2026-10-13): 修正 Round 64 错算 — 只 rescue FN→TP, 不再制造 TN→FP.
        legacy 路径已断开 (rtu_replay_gui.py:1353), 95.3% alarms lag>=40ms 时 re-eval
        命中 W 必 _state['alarm']=False, 批量触发 TN→FP, 实测错算 582 spurious FP.
        GUI summary 报 1111 FP 实际是 529 真 FP + 582 spurious. 修复后 summary 跟
        CSV frame=K 重算 1:1 对齐 (F1 78.47%→85.59%, FPR 13.32%→6.34%).

        ⭐ Round 69 (2026-10-13): 重新启动 TN→FP rescue. 原因: Round 66 删除 TN→FP 时
        legacy 路径是活跃的, 同一 alarm 双路径叠加. 现在 legacy 完全断开, alarm_frame_index
        单路径 (Round 69 main_window 路由) 每 alarm 只进 _recent_evals 一次, 不会双算.
        实测 logs/test_20261013-225832: 关闭 TN→FP 时 simulation 给 FP=0 vs ground truth
        FP=33, 33 个晚到 alarm (lag > 40ms) 落在 normal window 救不回来 → 错算 TN.
        重启 TN→FP 后 summary 跟 ground truth 1:1 (FP=33 vs 33).
        """
        # ⭐⭐ 2026-10-21: 修正 H743 上报 frame 的 +1 抖动导致的归窗错位.
        #
        #   背景: H743 的推理被推迟到 main loop (2026-10-17 USART 死锁修复),
        #   ISR 快照完窗口后如果又来了 1 帧, 报上来的 frame 就成了 窗口末帧+1.
        #   实测 logs/test_20261021-003650 (274628 帧全量回放): 7287 个 alarm 里
        #   3241 个 (44.5%) 报的是 16k+1 而不是 16k.
        #
        #   原公式 W = (frame_idx-1)//16 对 16k 正确 (→k-1), 对 16k+1 错 (→k, 下一个窗).
        #   后果: 那 44.5% 的 alarm 全部落到下一个窗口 —— 真 TP 被记成 FP (202 个)
        #   或者跟别的 alarm 撞进同一窗口被合并 (614 个), GUI summary 的
        #   F1 被低报 0.070 / FPR 被高报 0.024 (F1 0.811 vs 真实 0.881).
        #
        #   修法: H743 的窗口网格末帧恒为 16 的倍数, 所以 frame_idx%16 != 0
        #   一定是迟报, 窗口号要再退 1.
        #   ⚠ 固件侧已同步修 (modbus_ids.c: g_pending_frame 快照窗口末帧),
        #     修好后 frame%16 恒 == 0, 本分支自动不生效 —— 两侧兼容.
        W = (frame_idx - 1) // 16
        if frame_idx % 16 != 0:
            W -= 1
        self._direct_alarms.add(W)
        # ⭐ Round 64 → 第六十六轮 (2026-10-13) 修正: 只 rescue FN→TP, 不再制造 TN→FP.
        #   之前 Round 64 看到 late alarm 命中 past window (alarm=False) 时一律回滚:
        #     - attack=True: FN→TP (helpful, 救回真 attack)
        #     - attack=False: TN→FP (harmful, 批量制造 spurious FP)
        #   实测 logs/test_20261013-003345: 95.3% alarms lag>=40ms, 触发 re-eval 时
        #   总是 _state['alarm']=False (因为 legacy 路径已断开, _direct_alarms.add(W)
        #   在 _evaluate_window 之后, W 已 evaluate). 制造了 582 个 spurious FP.
        #   真 FP (H743 帧级精度 49.59% 的 random 误报) 已在 on_alarm_with_frame 直接
        #   归 W 的 *未来* window 评估里算上 (legacy 路径下), 不需要 retroactive.
        for _state in self._recent_evals:
            if _state['W'] == W and not _state['alarm']:
                if _state['attack']:
                    # 当时 FN, 实际应是 TP (late alarm 确认 H743 判 attack)
                    self._fn -= 1
                    self._tp += 1
                else:
                    # ⭐ Round 69 (2026-10-13) 重启 TN→FP rescue.
                    #   Round 66 删除 TN→FP branch 是因为旧 legacy on_alarm 路径并存,
                    #   双路径叠加把同一 alarm 算两遍 → 582 spurious FP. 现在 legacy 路径
                    #   已彻底断开 (rtu_replay_gui.py:1383 注释), alarm_frame_index 单路径
                    #   转发 (Round 69 main_window 路由), 每个 alarm 只进 _recent_evals 一次.
                    #   实测 logs/test_20261013-225832: 关闭 TN→FP 时 simulation 给 FP=0
                    #   vs ground truth FP=33, 33 个晚到 alarm (lag > 40ms) 落在 normal window
                    #   救不回来. 重启 TN→FP 后 summary 跟 ground truth 1:1 对齐.
                    self._tn -= 1
                    self._fp += 1
                _state['alarm'] = True
                break  # 每个 alarm 只回滚一个 window (W 唯一)

    def _evaluate_window(self) -> None:
        """评估当前窗, 累加 TP/FP/TN/FN, 重置窗状态.

        ⭐ 关键: 必须同时清空 _recent_alarm_frames. 因为窗是非滑动 (跟训练
        max-aggregation 一致), 每个窗独立. 若不清空, 前一窗的 alarm 会通过
        deque 残留污染下一窗 → 整窗被错标 alarm=True.

        ⭐ 第六十一轮 (2026-10-11): 还要检查 _direct_alarms (H743 通过 frame_idx
        直接标记的 window). 优先信任 _direct_alarms, 兼容 legacy deque.

        ⭐ Round 64 (2026-10-13): 把当前 window 状态 (W, attack, alarm) 存进
        _recent_evals buffer. 当 on_alarm_with_frame 之后到达的 alarm 命中 buffer
        里某个 past window (lag-induced late arrival), 回滚 metric: TN→FP 或 FN→TP.
        这是 Round 63 lag compensation 失败后的替代方案, 利用了"alarm 晚到但
        我们能识别它属于哪个 window"的信息.
        """
        W_now = self._n_windows  # 当前要评估的 window 索引 (0-based)
        attack = self._current_window_attack
        # ⭐ Round 61: union legacy deque + direct alarms
        alarm = self._current_window_alarm or (W_now in self._direct_alarms)
        # ⭐ Round 64: save state for potential retroactive update
        self._recent_evals.append({
            'W': W_now,
            'attack': attack,
            'alarm': alarm,
        })
        if attack and alarm:
            self._tp += 1
        elif not attack and alarm:
            self._fp += 1
        elif attack and not alarm:
            self._fn += 1
        else:
            self._tn += 1
        self._n_windows += 1
        self._current_window_attack = False
        self._current_window_alarm = False
        # ⭐ Round 61: 清掉本次 window 的 direct alarm 标记
        self._direct_alarms.discard(W_now)
        # ⭐ 清空 alarm deque — 每个窗独立评估, 不允许跨窗污染.
        self._recent_alarm_frames.clear()

    def get_metrics(self) -> dict:
        """返回当前 per-window 指标 (供 IdsReportPanel 显示)."""
        n_attack_w = self._tp + self._fn
        n_alarm_w = self._tp + self._fp
        total = self._tp + self._fp + self._tn + self._fn
        return {
            'tp': self._tp,
            'fp': self._fp,
            'tn': self._tn,
            'fn': self._fn,
            'n_windows': self._n_windows,
            'n_attack_windows': n_attack_w,
            'n_alarm_windows': n_alarm_w,
            'recall': self._tp / n_attack_w if n_attack_w > 0 else 0.0,
            'precision': self._tp / n_alarm_w if n_alarm_w > 0 else 0.0,
            'f1': (2 * self._tp / (2 * self._tp + self._fp + self._fn)
                   if (2 * self._tp + self._fp + self._fn) > 0 else 0.0),
            'accuracy': (self._tp + self._tn) / total if total > 0 else 0.0,
        }

    def reset(self) -> None:
        """清零所有状态 (供新测试开始 / 用户手动重置)."""
        self._frame_count = 0
        self._current_window_attack = False
        self._current_window_alarm = False
        self._recent_alarm_frames.clear()
        # ⭐ 第六十一轮 (2026-10-11): 同时清空 _direct_alarms set
        self._direct_alarms.clear()
        # ⭐ Round 64 (2026-10-13): 清空 re-eval buffer (避免 stale state)
        self._recent_evals.clear()
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        self._n_windows = 0


# === IDS Report 面板 (第十七轮 双计数 → 第十九轮 4 KPI → 第二十轮 FP/TP/FPR) ==
# 显示 6 个 KPI + 2 个派生率:
#   - Total:    PC 已发送的总帧数 (binary=0 + binary=1, 即总发送)
#   - Sent:     PC 已发送的攻击帧数 (binary==1 时 +1, 即攻击次数)
#   - Normal:   PC 已发送的正常帧数 (binary==0 时 +1, 即正常数)
#   - Detected: H743 上行 [ALARM] N 的累计计数 (累计 frames_attack)
#   - TP:       真阳性 — [ALARM] 到达时最近 16 帧滑动窗口内有 attack (binary==1)
#                ⭐ 第二十八轮 (2026-10-02): 受 TP budget 限制 — 每 PC 攻击帧 = 1 个 TP 配额,
#                  保证 TP_count ≤ Attack_count, Recall ≤ 100%
#   - FP:       假阳性 — [ALARM] 到达时最近 16 帧滑动窗口内全 normal (binary==0)
#                或 budget 耗尽后的多余 [ALARM] (H743 SW=16 over-fire)
#   - Recall:   TP / Sent × 100% (召回率, H743 命中攻击的能力)
#   - FPR:      FP / Normal × 100% (误报率, 正常帧被误报的比例)
# 调试详情请看串口 raw 终端.

class IdsReportPanel(QWidget):
    """IDS 面板: 单行 13 个 per-window KPI (跟训练 max-aggregation 1:1 对齐, window=16).

    布局 (⭐⭐ 第六十一轮 2026-10-11: 删 per-frame 行 + 分隔线, 只剩 per-window 1 行):
      Total → Normal → Attack → Detected → TP → FP → TN → FN →
      Precision → Recall → F1 → Accuracy → FPR

    评估粒度 = window=16 max-aggregation (跟训练 preprocess_v2_scada.py
    reshape(n//16, 16, -1) 1:1). 4 指标 (Precision/Recall/F1/Accuracy) 可直接对比
    训练 baseline (F1m), 监测 H743 部署性能 vs 训练性能.

    ⭐ 2026-10-19: 删 per-frame state (per-frame 计数已全部移除).

    数据来源:
      - Total / Normal / Attack ← ReplayWorker.window_stats (16 帧 1 window, max-aggregation)
      - Detected                ← IdsLineParser.count_changed (H743 windows_attack delta)
      - TP / FP / TN / FN       ← _window_eval (per-window max-aggregation 评估)
      - Precision / Recall / F1 / Accuracy ← _window_eval (训练 1:1 直接对比)
      - FPR                     ← 派生: fp / _window_normal_count

    ⭐ 第四十轮 (2026-10-02): 两个新信号给 MetricsChartPanel 用. **必须在类级别声明**
      pyqtSignal 才能 connect/emit. 若在 __init__ 里 `self.X = pyqtSignal(...)` 创建的是
      未绑定的 pyqtSignal 描述符, 不会有 .connect / .emit 方法 (PyQt5 metaclass 行为).
      修复: 把声明移到类级 (__init__ 不再赋 self.metrics_updated).
    """
    # ⭐ 第四十轮: pyqtSignal 必须在类级别声明 (PyQt5 metaclass 绑定).
    metrics_updated = pyqtSignal(dict)   # {'p','r','f1','acc','fpr'}, None=未计算
    metrics_reset   = pyqtSignal()       # 测试开始/用户清零时通知曲线面板

    def __init__(self, parser: IdsLineParser, parent=None):
        super().__init__(parent)
        self._parser = parser
        # ⭐ 2026-10-19: 删 per-frame 内部 state (_total_count / _sent_count / _normal_count /
        #   _tp_count / _fp_count / _tn_count / _fn_count). 评估只用 per-window 状态,
        #   跟训练 / H743 部署 1:1 对齐.
        # ⭐⭐ 第五十二轮 (2026-10-09): per-window 累计 state, 跟训练 / H743 1:1.
        #   4 个主 KPI (Total / Normal / Attack / Detected) 全部按 16 帧 1 window 算:
        #     - Total:   累计 windows 数 (sent window 凑齐 1 个 +1, 跟训练 reshape(n//16) 一致)
        #     - Normal:  累计 normal windows (16 帧全 0, max-aggregation)
        #     - Attack:  累计 attack windows (16 帧任一帧 1, max-aggregation, 跟训练 1:1)
        #     - Detected:H743 windows_attack 累计 (Round 50 已是 per-window, count_changed 携带的 N
        #                就是 windows 累计, 1 window 1 alarm 1:1 — 因此 Detected 天然是 per-window).
        self._window_total_count = 0     # ⭐ 第五十二轮: per-window 总发送 windows
        self._window_normal_count = 0    # ⭐ 第五十二轮: per-window normal windows
        self._window_attack_count = 0    # ⭐ 第五十二轮: per-window attack windows
        # ⭐ 第二十八轮 (2026-10-02): Detected 显示 = H743 frames_attack delta (=TP+FP, 含 over-fire).
        #   ⭐ 第五十轮 (2026-10-09) 修订: H743 frames_attack → windows_attack (1 window 1 alarm 1:1),
        #   因此 Detected 现在是 per-window 累计 (= H743 累计 windows_attack delta), 天然 1:1.
        #   用 baseline 锚点确保从 0 开始递增, 即使 H743 未重启, 首次 [ALARM] N 也按 0 处理.
        self._detected_count = 0     # H743 windows_attack 总数 (本次测试 delta, = 当前 count - baseline)
        self._detected_baseline: Optional[int] = None  # 首条 [ALARM] N-1 作锚点 (Round 44, 避免 off-by-one)
        # ⭐ 第四十八轮 (2026-10-08): per-window 评估器 (跟训练 1:1 对齐)
        self._window_eval = WindowEvaluator()
        self._build_ui()
        # 接 parser 信号 (H743 上行的 [ALARM] N)
        # count_changed → _on_detected 持续订阅 (Round 28 baseline 锚点保留):
        #   Detected KPI = H743 windows_attack delta (=TP+FP), 含 over-fire 部分.
        # ⭐ 第四十八轮: count_changed 也喂给 WindowEvaluator.on_alarm (per-window 评估)
        # ⭐ 第六十六轮 (2026-10-13): 弃用 legacy on_alarm 路径.
        #   H743 现固件发 frame=K, IdsLineParser._parse_line 走 on_alarm_with_frame(K) 精确归 W.
        #   同一 alarm 之前既走 legacy (count_changed→on_alarm, 错归 W+1) 又走 direct (W=K 精确).
        #   双路径叠加制造 spurious FP/FN. 断开 legacy 路径, 只留 direct 精确归因.
        #   WindowEvaluator.on_alarm 方法仍存在 (旧固件回退), 仅此处不再 connect.
        self._parser.count_changed.connect(self._on_detected)
        # self._parser.count_changed.connect(self._window_eval.on_alarm)  # ⭐ 第六十六轮: 弃用
        # ⭐ 2026-10-11: 当前窗 alarm 状态触发 detected_label 变绿
        self._parser.count_changed.connect(self._refresh_attack_alarm_color)

    def _build_ui(self):
        # ⭐⭐ 第六十一轮 (2026-10-11): 删 per-frame (per-alarm) KPI 行 + 分隔线, 只保留
        #   per-window KPI 行 (13 项 Total/Normal/Attack/Detected/TP/FP/TN/FN/Precision/
        #   Recall/F1/Accuracy/FPR). 评估粒度与训练 max-aggregation 1:1 对齐 (window=16),
        #   避免此前 per-frame / per-window 双行造成的口径混淆 (Per-Frame Recall 算 5%
        #   而 Per-Window Recall 是 36% — 用户无法判断哪个是对的).
        #
        # ⭐ 2026-10-19: 删 per-frame state (per-frame 计数已全部移除).
        #
        #   attr 名保留 w- 前缀 (避免破坏 test_zround48_per_window_evaluator 等现有测试
        #   对 wtp_label / wfp_label / wrecall_label 的断言); 显示文案去 w- 前缀.
        outer = QVBoxLayout(self)

        # === Per-Window KPI 行 (跟训练 max-aggregation 1:1 对齐, window=16) =======
        # ⭐ 第四十八轮 (2026-10-08) 首版, 第五十二轮 (2026-10-09) 4 主 KPI 接入,
        #   第六十一轮 (2026-10-11) 补全 13 项 + 删 per-frame 行.
        row_window = QHBoxLayout()

        # 1. Total (per-window 总数, 16 帧/窗)
        total_cap = QLabel("Total:")
        total_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wn_label = QLabel("0")
        self.wn_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #333333; padding: 4px;"
        )
        row_window.addWidget(total_cap)
        row_window.addWidget(self.wn_label)

        row_window.addSpacing(12)

        # 2. Normal (per-window 正常数, 16 帧全 0 → max-aggregation → normal)
        #   ⭐ 第六十一轮 (2026-10-11): 新增, 取代 per-frame normal_label (已删).
        normal_cap = QLabel("Normal:")
        normal_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wnormal_label = QLabel("0")
        self.wnormal_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #008800; padding: 4px;"
        )
        row_window.addWidget(normal_cap)
        row_window.addWidget(self.wnormal_label)

        row_window.addSpacing(12)

        # 3. Attack (per-window 攻击数, 16 帧任 1 → max-aggregation → attack)
        #   ⭐ 第六十一轮 (2026-10-11): 新增, 取代 per-frame sent_label (已删).
        attack_cap = QLabel("Attack:")
        attack_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wattack_label = QLabel("0")
        self.wattack_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #0055aa; padding: 4px;"
        )
        row_window.addWidget(attack_cap)
        row_window.addWidget(self.wattack_label)

        row_window.addSpacing(12)

        # 4. Detected (H743 上行 [ALARM] windows_attack 累计, per-window 1:1)
        #   ⭐ 第六十一轮: 新增 (per-frame detected_label 已删); Detected ≠ Attack
        #   (H743 可能漏报 attack window → Attack-Detected = FN, 用户直接看漏报比例).
        det_cap = QLabel("Detected:")
        det_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wdetected_label = QLabel("0")
        self.wdetected_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #cc0000; padding: 4px;"
        )
        row_window.addWidget(det_cap)
        row_window.addWidget(self.wdetected_label)

        row_window.addSpacing(12)

        # 5. TP (per-window 真阳性, 窗内 attack + 过去 16 帧 alarm)
        tp_cap = QLabel("TP:")
        tp_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wtp_label = QLabel("0")
        self.wtp_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #cc0000; padding: 4px;"
        )
        row_window.addWidget(tp_cap)
        row_window.addWidget(self.wtp_label)

        row_window.addSpacing(12)

        # 6. FP (per-window 假阳性, 窗内全 normal + 过去 16 帧 alarm)
        fp_cap = QLabel("FP:")
        fp_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wfp_label = QLabel("0")
        self.wfp_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #ff6600; padding: 4px;"
        )
        row_window.addWidget(fp_cap)
        row_window.addWidget(self.wfp_label)

        row_window.addSpacing(12)

        # 7. TN (per-window 真阴性, 窗内 normal + 过去 16 帧无 alarm)
        tn_cap = QLabel("TN:")
        tn_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wtn_label = QLabel("0")
        self.wtn_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #006666; padding: 4px;"
        )
        row_window.addWidget(tn_cap)
        row_window.addWidget(self.wtn_label)

        row_window.addSpacing(12)

        # 8. FN (per-window 假阴性, 漏报)
        fn_cap = QLabel("FN:")
        fn_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wfn_label = QLabel("0")
        self.wfn_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #cc4400; padding: 4px;"
        )
        row_window.addWidget(fn_cap)
        row_window.addWidget(self.wfn_label)

        row_window.addSpacing(16)

        # 9. Precision (per-window 精确率, 跟训练 Precision 直接对比)
        prec_cap = QLabel("Precision:")
        prec_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wprecision_label = QLabel("—")
        self.wprecision_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #0055aa; padding: 4px;"
        )
        row_window.addWidget(prec_cap)
        row_window.addWidget(self.wprecision_label)

        row_window.addSpacing(12)

        # 10. Recall (per-window 召回率, 跟训练 Recall 直接对比)
        rec_cap = QLabel("Recall:")
        rec_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wrecall_label = QLabel("—")
        self.wrecall_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #0055aa; padding: 4px;"
        )
        row_window.addWidget(rec_cap)
        row_window.addWidget(self.wrecall_label)

        row_window.addSpacing(12)

        # 11. F1 (per-window F1, ⭐ 跟训练 F1m 直接对比的核心指标)
        f1_cap = QLabel("F1:")
        f1_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wf1_label = QLabel("—")
        self.wf1_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 20px; color: #0055aa; padding: 4px;"
        )
        row_window.addWidget(f1_cap)
        row_window.addWidget(self.wf1_label)

        row_window.addSpacing(12)

        # 12. Accuracy (per-window Accuracy, (TP+TN)/Total)
        #   ⭐ 第六十一轮: 新增.
        acc_cap = QLabel("Accuracy:")
        acc_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.waccuracy_label = QLabel("—")
        self.waccuracy_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #0055aa; padding: 4px;"
        )
        row_window.addWidget(acc_cap)
        row_window.addWidget(self.waccuracy_label)

        row_window.addSpacing(12)

        # 13. FPR (per-window FPR, FP / Normal windows)
        #   ⭐ 第六十一轮: 新增 (per-frame fpr_label 已删).
        fpr_cap = QLabel("FPR:")
        fpr_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.wfpr_label = QLabel("—")
        self.wfpr_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #d4006a; padding: 4px;"  # 与 metrics_chart 同色
        )
        row_window.addWidget(fpr_cap)
        row_window.addWidget(self.wfpr_label)

        row_window.addStretch(1)
        outer.addLayout(row_window)

    def _on_sent_stats(self, total: int, attack: int, normal: int) -> None:
        """ReplayWorker.sent_stats 信号回调 (每帧一次).

        ⭐ 2026-10-19: 删 per-frame 内部 state 维护. sent_stats 仅触发 _update_rates
        保持 GUI 同步 (虽然只 _on_window_stats 改变 label 数字). _on_window_stats 单独
        喂数据到主 KPI.

        Args:
            total:  累计总发送帧数 (= attack + normal)
            attack: 累计攻击帧数 (binary==1 时 +1)
            normal: 累计正常帧数 (binary==0 时 +1)
        """
        # ⭐ 2026-10-19: 不再 setText per-frame state, 删 _total_count/_sent_count/_normal_count
        # 维护. label 由 _on_window_stats 维护 (per-window 1:1 跟训练对齐).
        # 保留 _update_rates() 调用以保持 GUI 同步窗口.
        del total, attack, normal  # unused
        self._update_rates()

    def _on_window_stats(self, total_w: int, attack_w: int, normal_w: int) -> None:
        """ReplayWorker.window_stats 信号回调 (凑齐 1 window 16 帧一次).

        ⭐⭐ 第五十二轮 (2026-10-09, 用户要求: "total、normal、attack、detected
        都是按照每16帧来计算一次"):

          4 个主 KPI label 全部按 16 帧 1 window 算, 跟训练 preprocess_v2_scada.py
          reshape(n//16, 16, -1) max-aggregation 1:1, 跟 H743 部署 1 window 1 alarm
          1:1. 整个链路:
            训练:  274k 帧 → 17164 windows × 1 label (max-aggregation)
            PC 端: 274k 帧 → 17164 windows × 1 attack 决策 (Round 51 max-aggregation)
            H743:  274k 帧 → 17164 windows × 1 binary decision (Round 50 non-overlap)
            评估:  1 H743 alarm vs 1 PC window → TP/FP, 4 指标 1:1 训练 macro_f1

        Args:
            total_w:  累计已发送 windows 数 (= frames_total // 16, 尾部不足 16 帧不计)
            attack_w: 累计已发送 attack windows 数 (16 帧任一帧 attack → max-aggregation)
            normal_w: 累计已发送 normal windows 数 (16 帧全 normal)
        """
        self._window_total_count = total_w
        self._window_normal_count = normal_w
        self._window_attack_count = attack_w
        # ⭐⭐ 第六十一轮 (2026-10-11): 不再 setText per-frame label (total_label /
        #   sent_label / normal_label 已删). 3 个 per-window 主 KPI 由 _update_rates
        #   统一从 _window_total/normal/attack_count 写 wn_label / wnormal_label /
        #   wattack_label.
        self._update_rates()

    def _on_detected(self, count: int) -> None:
        """IdsLineParser.count_changed 信号回调 (H743 上行 [ALARM] N 累计).

        ⭐ 第二十八轮 (2026-10-02): H743 frames_attack 是 boot 起全局累计, 不重启 H743
        跑新测试时首条 [ALARM] N 会是旧值 (例如 20000). 首次回调捕获 baseline,
        后续显示 delta. 保证 Detected KPI 从 0 开始递增.

        ⭐ 第四十四轮 (2026-10-08): 修复 off-by-one — baseline = max(count-1, 0)
        (不是 count). 旧逻辑首条 N 作锚点 → 首条 emit 后 _detected_count=0, 但 N
        本身已经代表 1 次 alarm 工作量, 被锚点吃掉, 末条 _detected_count 永远比
        上行 [ALARM] 消息数少 1. 新逻辑 baseline=N-1 → 首条后 count=1, 末条 count=
        m 等于消息数.

        ⭐⭐ 第五十轮 (2026-10-09): H743 语义从 frames_attack (累计攻击帧) 改为
          windows_attack (累计攻击 window), 1 window 1 alarm 1:1. count_changed
          携带的 N 已经是 per-window 累计.

        ⭐⭐ 第五十二轮 (2026-10-09, 用户要求: "total、normal、attack、detected 都
          是按照每 16 帧来计算一次"): Detected label 显示 = H743 windows_attack
          delta = per-window 累计. 跟 _window_attack_count (PC 端 attack window 数)
          一一对应, 都按 16 帧 1 window 算, 1:1 跟训练 / H743 1:1 对齐.

        注: Detected 不直接等于 Attack KPI (PC 已发 attack window), 因为 H743
        推理存在漏报 (FN): 一个 attack window H743 没报 → 算漏报. 评估 Recall
        应该用 TP / Attack, 不是 Detected / Attack. Detected 只是给用户看
        "H743 报了多少 attack window 上来", 跟 _window_attack_count 配合看漏报比例.
        """
        if self._detected_baseline is None:
            self._detected_baseline = max(count - 1, 0)
        self._detected_count = count - self._detected_baseline
        # ⭐⭐ 第六十一轮: 不再 setText self.detected_label (已删); 改由 _update_rates
        #   统一写 self.wdetected_label (per-window Detected).
        # ⭐ 2026-10-19: TP/FP 计数由 _window_eval 直接维护 (per-window 1:1 跟训练对齐),
        #   不再用 _on_tp_delta / _on_fp_delta 派生.
        self._update_rates()

    def _refresh_attack_alarm_color(self, *_args) -> None:
        """⭐ 2026-10-11: 实时颜色反馈当前 16 帧窗状态.

        - 当前窗任一帧 attack (frame_sent_type=True) → wattack_label 红色背景
        - 当前窗内 H743 alarm 触发 (count_changed) → wdetected_label 绿色背景
        - 下个窗开始自动还原 (WindowEvaluator._evaluate_window 重置状态).

        数据源: WindowEvaluator._current_window_attack / _current_window_alarm
        (跟 WindowEvaluator.on_frame / on_alarm 同源, 不重复计算).

        ⭐ 第六十三轮 (2026-10-12): alarm_now 用 OR 公式 (跟 _evaluate_window 一致),
        支持 direct 路径 (_direct_alarms) + legacy 路径 (_current_window_alarm). 双路径
        并存: 4.7% 低 lag alarm 走 direct 精确归 W, 95.3% 高 lag 走 legacy +1 forward
        shift (后续 attack burst 跨 window 时仍 TP). 视觉上也用 OR 反映任一路径触发.
        """
        attack_now = bool(self._window_eval._current_window_attack)
        alarm_now  = bool(
            self._window_eval._current_window_alarm
            or (self._window_eval._n_windows in self._window_eval._direct_alarms)
        )

        # Attack: 发送攻击 → 红
        self.wattack_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #0055aa; padding: 4px; "
            "background-color: #ff5555;" if attack_now else
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #0055aa; padding: 4px;"
        )
        # Detected: 监测到 alarm → 绿
        self.wdetected_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #cc0000; padding: 4px; "
            "background-color: #55ff55;" if alarm_now else
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #cc0000; padding: 4px;"
        )

    def _update_rates(self) -> None:
        """计算并显示 13 个 per-window KPI (跟训练 max-aggregation 1:1 对齐).

        ⭐⭐ 第六十一轮 (2026-10-11): 重写 — 删 per-frame 公式, 只算 per-window 13 项.
          评估粒度跟训练 preprocess_v2_scada.py reshape(n//16, 16, -1) 完全 1:1,
          F1m/Recall 可直接对比训练 baseline.

        ⭐ 2026-10-19: 删 per-frame state 维护 (per-frame 已删除).
        """
        # ⭐⭐ 第六十一轮 (2026-10-11): 双源数据架构.
        #   - 4 个主 KPI (Total / Normal / Attack / Detected) ← 内部 state (_window_*_count
        #     + _detected_count). 这些是 PC 实际发送的 windows 累计 (per-window 累计).
        #   - TP / FP / TN / FN + 4 派生率 ← _window_eval.get_metrics() (WindowEvaluator).
        #   这样 _on_window_stats 喂 state 即时驱动主 KPI label (test 友好), WindowEvaluator
        #   喂 wm 驱动评估指标 (production 实际由 worker.frame_sent_type → on_frame +
        #   parser.count_changed → on_alarm 两条路径并行喂, 1:1 对齐训练 max-aggregation).
        n_w = self._window_total_count
        n_attack_w = self._window_attack_count
        n_normal_w = self._window_normal_count
        detected = self._detected_count
        wm = self._window_eval.get_metrics()
        wtp = wm['tp']
        wfp = wm['fp']
        wtn = wm['tn']
        wfn = wm['fn']
        # ⭐⭐ 第六十一轮 (2026-10-11): 双源数据架构.
        #   - 4 个主 KPI (Total / Normal / Attack / Detected) ← 内部 state (_window_*_count
        #     + _detected_count). 这些是 PC 实际发送的 windows 累计 (per-window 累计).
        #   - TP / FP / TN / FN + 4 派生率 ← _window_eval.get_metrics() (WindowEvaluator).
        #   这样 _on_window_stats 喂 state 即时驱动主 KPI label (test 友好), WindowEvaluator
        #   喂 wm 驱动评估指标 (production 实际由 worker.frame_sent_type → on_frame +
        #   parser.count_changed → on_alarm 两条路径并行喂, 1:1 对齐训练 max-aggregation).
        n_w = self._window_total_count
        n_attack_w = self._window_attack_count
        n_normal_w = self._window_normal_count
        detected = self._detected_count
        wm = self._window_eval.get_metrics()
        wtp = wm['tp']
        wfp = wm['fp']
        wtn = wm['tn']
        wfn = wm['fn']

        # 1-4. Total / Normal / Attack / Detected
        self.wn_label.setText(str(n_w))
        self.wnormal_label.setText(str(n_normal_w))
        self.wattack_label.setText(str(n_attack_w))
        self.wdetected_label.setText(str(detected))

        # 5-8. TP / FP / TN / FN
        self.wtp_label.setText(str(wtp))
        self.wfp_label.setText(str(wfp))
        self.wtn_label.setText(str(wtn))
        self.wfn_label.setText(str(wfn))

        if wm['n_windows'] > 0:
            # 9. Precision (per-window 精确率, 跟训练 Precision 直接对比)
            self.wprecision_label.setText(f"{100.0 * wm['precision']:.1f}%")
            # 10. Recall (per-window 召回率, ⭐ 跟训练 Recall 字段直接对比)
            #    clamp 100% 防御性保护 (跟训练 100% 上限对齐)
            raw_rec = 100.0 * wm['recall']
            rec = min(raw_rec, 100.0)
            self.wrecall_label.setText(f"{rec:.1f}%")
            # 11. F1 (⭐ 跟训练 F1m 直接对比的核心指标)
            self.wf1_label.setText(f"{100.0 * wm['f1']:.1f}%")
            # 12. Accuracy (per-window (TP+TN)/Total, 跟训练 Accuracy 直接对比)
            self.waccuracy_label.setText(f"{100.0 * wm['accuracy']:.1f}%")
        else:
            self.wprecision_label.setText("—")
            self.wrecall_label.setText("—")
            self.wf1_label.setText("—")
            self.waccuracy_label.setText("—")

        # 13. FPR (per-window FP / Normal windows) — WindowEvaluator 不直接算, 派生
        if n_normal_w > 0:
            fpr = 100.0 * wfp / n_normal_w
            self.wfpr_label.setText(f"{fpr:.1f}%")
        else:
            self.wfpr_label.setText("—")

        # ⭐ 第四十轮 (2026-10-02): 推送 5 个派生率给 MetricsChartPanel.
        #   ⭐⭐ 第六十一轮: 数据源改 per-window (跟训练 1:1, 跟 UI 显示 1:1).
        #   字段名 (p/r/f1/acc/fpr) 保持不变, MetricsChartPanel 自动适配.
        #   每个值可能为 None (数据不足, 标签显示 "—").
        #   即使全 None 也 emit, 让曲线面板知道"过了一个事件点" (用于 X 轴计数).
        self.metrics_updated.emit({
            'p':   (100.0 * wm['precision']) if n_w > 0 else None,
            'r':   (min(100.0, 100.0 * wm['recall'])) if n_w > 0 else None,
            'f1':  (100.0 * wm['f1']) if n_w > 0 else None,
            'acc': (100.0 * wm['accuracy']) if n_w > 0 else None,
            'fpr': (100.0 * wfp / n_normal_w) if n_normal_w > 0 else None,
        })

    def reset_counters(self):
        """重置全部 13 个 per-window KPI + 内部 state (供测试开始前 / 用户手动清零).

        ⭐ 2026-10-19: 删 per-frame state 重置 (per-frame 已删除).
        """
        # ⭐⭐ 第五十二轮: per-window 内部 state 也清零, 避免残留污染下次测试.
        self._window_total_count = 0
        self._window_normal_count = 0
        self._window_attack_count = 0
        # ⭐ 第二十八轮: 同时清 Detected baseline, 让下次首个 [ALARM] 重新捕获.
        #   ⭐ 第五十二轮: Detected 现在 = H743 windows_attack delta (per-window 累计).
        self._detected_count = 0
        self._detected_baseline = None
        # ⭐⭐ 第六十一轮: 删所有 per-frame label setText (label 已删). 改 13 个
        #   per-window label 重置 (5 个新增 + 8 个原有).
        self.wn_label.setText("0")
        self.wnormal_label.setText("0")
        self.wattack_label.setText("0")
        self.wdetected_label.setText("0")
        self.wtp_label.setText("0")
        self.wfp_label.setText("0")
        self.wtn_label.setText("0")
        self.wfn_label.setText("0")
        self.wprecision_label.setText("—")
        self.wrecall_label.setText("—")
        self.wf1_label.setText("—")
        self.waccuracy_label.setText("—")
        self.wfpr_label.setText("—")
        # ⭐ 第四十八轮: 重置 per-window 评估器
        self._window_eval.reset()
        # ⭐ 2026-10-11: 重置 attack/alarm 实时颜色
        self._refresh_attack_alarm_color()
        # ⭐ 第二十轮: 同时清空 parser 的 _last_was_attack, 避免旧的 PC 帧类型污染新测试
        self._parser.reset_last_was_attack()
        # ⭐ 第四十轮: 通知 MetricsChartPanel 同步清空曲线 history
        self.metrics_reset.emit()


# === Metrics 曲线面板 (第四十轮 2026-10-02) ===============================
# 5 个 IDS 派生指标 (Precision / Recall / F1 / Accuracy / FPR) 实时曲线.
# 数据源: 订阅 IdsReportPanel.metrics_updated (每个 [ALARM] 事件一次).
# 渲染: matplotlib FigureCanvasQTAgg, 暗色主题 (与 GUI 暗色调一致).

class MetricsChartPanel(QWidget):
    """5 个 IDS 指标实时曲线面板.

    数据流:
      IdsReportPanel.metrics_updated(dict) → _on_metrics_updated
        → append 到 self._history (FIFO 500)
        → _redraw() 增量更新 matplotlib Line2D

    控件:
      - 🗑 清空曲线 按钮: 清空 history + x 索引归零
      - ⏸ 暂停 / ▶ 继续 切换: 暂停时不追加新点 (按钮 text 切换)
    """

    # FIFO 上限, 防止长测试无限增长导致 matplotlib 卡顿
    _FIFO_CAP = 500

    # 5 条曲线的颜色, 与 IdsReportPanel 标签配色对齐 (便于视觉关联)
    _COLORS = {
        'p':   '#aa6600',   # Precision — 橙棕
        'r':   '#0066cc',   # Recall    — 蓝 (区分 Precision)
        'f1':  '#cc0066',   # F1        — 粉
        'acc': '#7b1fa2',   # Accuracy  — 紫
        'fpr': '#d4006a',   # FPR       — 品红 (与 fpr_label 同色)
    }
    _LABELS = {
        'p':   'Precision',
        'r':   'Recall',
        'f1':  'F1',
        'acc': 'Accuracy',
        'fpr': 'FPR',
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._history: list[dict] = []   # [{p,r,f1,acc,fpr}, ...]
        self._x_idx: int = 0             # 累计事件数 (X 轴)
        self._paused: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # 顶部 toolbar: 清空 + 暂停
        toolbar = QHBoxLayout()
        self.chart_clear_btn = QPushButton("🗑 清空曲线")
        self.chart_clear_btn.setToolTip("清空历史曲线点 (X 轴归零)")
        self.chart_clear_btn.clicked.connect(self._on_clear_clicked)
        self.chart_pause_btn = QPushButton("⏸ 暂停")
        self.chart_pause_btn.setCheckable(True)
        self.chart_pause_btn.setToolTip("暂停时不追加新曲线点 (X 轴也不递增)")
        self.chart_pause_btn.toggled.connect(self._on_pause_toggled)
        toolbar.addWidget(self.chart_clear_btn)
        toolbar.addWidget(self.chart_pause_btn)
        toolbar.addStretch(1)
        # 状态标签: 显示当前 history 大小
        self.chart_status_label = QLabel("0 点")
        self.chart_status_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self.chart_status_label)
        outer.addLayout(toolbar)

        if not _MPL_AVAILABLE:
            # matplotlib 缺失: 显示占位提示
            placeholder = QLabel(
                "(matplotlib 未安装, 无法显示曲线)\n"
                "安装: pip install matplotlib"
            )
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; padding: 40px;")
            outer.addWidget(placeholder, stretch=1)
            self.canvas = None
            self.ax = None
            self.lines: dict[str, object] = {}
            return

        # matplotlib Figure + Canvas
        self.fig = Figure(figsize=(7, 3), dpi=80, facecolor='#2d2d2d')
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas = _MplCanvas(self.fig)
        outer.addWidget(self.canvas, stretch=1)

        # 创建 5 条 Line2D (初始全空)
        self.lines: dict[str, object] = {}
        xs: list[int] = []
        ys: list[float] = []
        for key in ('p', 'r', 'f1', 'acc', 'fpr'):
            line, = self.ax.plot(
                xs, ys,
                color=self._COLORS[key],
                label=self._LABELS[key],
                linewidth=1.5,
                marker='o',
                markersize=3,
            )
            self.lines[key] = line
        # 图例 (右上, 暗底)
        legend = self.ax.legend(
            loc='upper right',
            facecolor='#1e1e1e',
            edgecolor='#555',
            labelcolor='white',
            fontsize=9,
        )
        self.canvas.draw_idle()

    def _style_axes(self) -> None:
        """设置 matplotlib 暗色主题 (与 GUI 一致)."""
        assert self.ax is not None
        ax = self.ax
        ax.set_facecolor('#1e1e1e')
        ax.set_title('IDS 指标实时曲线', color='white', fontsize=11)
        ax.set_xlabel('事件 #', color='#aaa', fontsize=9)
        ax.set_ylabel('百分比 (%)', color='#aaa', fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max(1, self._FIFO_CAP))
        # 刻度颜色
        ax.tick_params(axis='x', colors='#aaa', labelsize=8)
        ax.tick_params(axis='y', colors='#aaa', labelsize=8)
        # 边框颜色
        for spine in ax.spines.values():
            spine.set_color('#555')
        # 网格
        ax.grid(True, color='#444', linestyle='--', linewidth=0.5, alpha=0.7)
        self.fig.tight_layout()

    # ===== Slots =====

    def _on_metrics_updated(self, m: dict) -> None:
        """IdsReportPanel.metrics_updated 信号回调 (每个 [ALARM] 事件一次)."""
        if self._paused:
            return
        self._history.append(m)
        if len(self._history) > self._FIFO_CAP:
            # FIFO 截断
            self._history = self._history[-self._FIFO_CAP:]
        self._x_idx += 1
        self._redraw()

    def _on_metrics_reset(self) -> None:
        """IdsReportPanel.metrics_reset 信号回调 (用户/测试开始时清零)."""
        self._clear_internal()

    def _on_clear_clicked(self) -> None:
        """🗑 清空曲线 按钮回调."""
        self._clear_internal()

    def _on_pause_toggled(self, checked: bool) -> None:
        """⏸/▶ 切换按钮回调."""
        self._paused = checked
        self.chart_pause_btn.setText("▶ 继续" if checked else "⏸ 暂停")

    # ===== Internal =====

    def _clear_internal(self) -> None:
        self._history.clear()
        self._x_idx = 0
        self._redraw()

    def _redraw(self) -> None:
        """重绘 5 条曲线 + 状态标签."""
        self.chart_status_label.setText(f"{len(self._history)} 点")
        if self.canvas is None:
            return
        # 重建 X / Y 数组
        n = len(self._history)
        xs = list(range(n))
        for key in ('p', 'r', 'f1', 'acc', 'fpr'):
            ys = [h.get(key) for h in self._history]
            # matplotlib 不能直接 plot None — 替换为 nan 让线条断开
            import math
            ys_plot = [y if y is not None else math.nan for y in ys]
            line = self.lines[key]
            line.set_data(xs, ys_plot)
        # 重置 X 轴范围 (适应 FIFO 满的情况)
        if n > 0:
            self.ax.set_xlim(0, max(self._FIFO_CAP, n))
        self.canvas.draw_idle()


# === 测试日志 (第二十一轮 2026-09-20) ====================================
# 1000 条测试出现 TP=56/FP=185, 跟 ONNX 仿真 TP=177/FP=808 完全对不上.
# 根因分析需要更精细的 per-frame / per-alarm 事件流.
# TestLogger 把每个事件 (SEND 帧 / ALARM 上行 / KPI 快照) 写入 CSV 文件,
# 配合 logs/test_<时间戳>.csv + test_<时间戳>_summary.txt, 事后用 pandas /
# Excel / awk 分析. 不依赖 GUI 日志窗口 (后者会滚屏掉).
class TestLogger:
    """结构化测试日志 — 每个事件一行 CSV, 方便事后分析.

    ⭐ 2026-10-19: 删 per-frame 计数 (running_total/attack/normal/tp/fp), 只保留
    per-window KPI 跟踪. PC sim 跟 H743 1:1 对齐, evaluation 都是 per-window.

    写入 logs/test_<YYYYMMDD-HHMMSS>.csv, 列:
      t_event_ms | event | frame_idx | binary | n_bytes | alarm_count
      | prob_x1000 | is_fp | gap_after_send_ms | note

    同时写 .summary.txt (KPI 快照, 每次 close() 时更新).
    """
    HEADER = [
        't_event_ms', 'event', 'frame_idx', 'binary', 'n_bytes',
        'alarm_count', 'prob_x1000', 'is_fp',
        'gap_after_send_ms', 'note',
    ]

    def __init__(self):
        self._file = None
        self._writer = None
        self._path = None
        self._t0_ms: Optional[int] = None  # 测试开始 unix ms
        self._last_send_ms: Optional[int] = None  # 上一次 SEND 的 unix ms (算 gap)
        # ⭐⭐ 第五十二轮 c (2026-10-10): per-window KPI 跟踪, 跟训练 / H743 1:1.
        #   running_window_total/attack/normal: 累计 windows (由 ReplayWorker.window_stats 维护)
        #   running_window_detected: H743 windows_attack 累计 (= IdsLineParser.count_changed delta)
        #   running_window_tp/fp/tn/fn: per-window 评估 (由 main_window 在 close 前 push 进来)
        self.running_window_total = 0
        self.running_window_attack = 0
        self.running_window_normal = 0
        self.running_window_detected = 0  # H743 windows_attack 累计
        # per-window 混淆矩阵 (push from ids_report._window_eval.get_metrics())
        self.running_window_tp = 0
        self.running_window_fp = 0
        self.running_window_tn = 0
        self.running_window_fn = 0

    @property
    def is_open(self) -> bool:
        return self._file is not None

    @property
    def path(self) -> Optional[str]:
        return self._path

    def open(self, label: str = "") -> str:
        """打开新日志文件. label 可选 (e.g. 'auto-10x' / 'manual-40frames')."""
        import os, csv
        os.makedirs('logs', exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        suffix = f'_{label}' if label else ''
        self._path = os.path.join('logs', f'test_{ts}{suffix}.csv')
        self._file = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._file.flush()
        self._t0_ms = int(time.time() * 1000)
        self._last_send_ms = None
        return self._path

    def log_send(self, frame_idx: int, binary: int, n_bytes: int,
                 note: str = "") -> None:
        """记录一帧 PC 发送."""
        if not self.is_open:
            return
        import time as _t
        now_ms = int(_t.time() * 1000)
        t_event = now_ms - (self._t0_ms or now_ms)
        gap = '' if self._last_send_ms is None else (now_ms - self._last_send_ms)
        self._writer.writerow([
            t_event, 'SEND', frame_idx, binary, n_bytes,
            '', '', '',
            gap, note,
        ])
        self._file.flush()
        self._last_send_ms = now_ms

    def log_alarm(self, alarm_count: int, prob_x1000: Optional[int],
                  is_fp: Optional[bool], h743_frame: int = 0,
                  note: str = "") -> None:
        """记录一次 H743 上行 [ALARM] 事件.

        ⭐ 2026-10-19: 删 running_tp/running_fp 参数 (per-frame 计数已删除).

        ⭐ 第六十一轮 (2026-10-11): h743_frame 写到 CSV 第 3 列 (frame_idx), 诊断时
        可直接读 CSV 看到 H743 实际发的是哪个 frame, 不必依赖 GUI 解析逻辑.

        Args:
            alarm_count: H743 当前累计 frames_attack (本次 [ALARM] 上行携带的值)
            prob_x1000: H743 printf 输出的 prob=0.NNN × 1000
            is_fp: True=误报 (窗内全 normal), False=命中 (窗内有 attack), None=未判定
                   注: 这是单 [ALARM] 消息级 is_fp (legacy 标签), 不代表 H743 内部 alarm_count 增量
            h743_frame: ⭐ 第六十一轮: H743 上行的 frame=N 字段, 0=旧固件没发
            note: 备注 (如 'prob=0.469 frame=16384')
        """
        if not self.is_open:
            return
        import time as _t
        now_ms = int(_t.time() * 1000)
        t_event = now_ms - (self._t0_ms or now_ms)
        gap = '' if self._last_send_ms is None else (now_ms - self._last_send_ms)
        # ⭐ 第六十一轮: frame_idx 列写 H743 实际发的 frame (旧固件写空字符串)
        frame_idx_str = str(h743_frame) if h743_frame > 0 else ''
        self._writer.writerow([
            t_event, 'ALARM', frame_idx_str, '', '',
            alarm_count, prob_x1000 if prob_x1000 is not None else '',
            ('FP' if is_fp is True else ('TP' if is_fp is False else '')),
            gap, note,
        ])
        self._file.flush()

    def log_event(self, event: str, note: str = "") -> None:
        """记录任意非帧事件 (START / STOP / RESET / KILL)."""
        if not self.is_open:
            return
        import time as _t
        now_ms = int(_t.time() * 1000)
        t_event = now_ms - (self._t0_ms or now_ms)
        self._writer.writerow([
            t_event, event, '', '', '', '', '', '',
            '', note,
        ])
        self._file.flush()

    def write_summary(self) -> None:
        """关闭时写 per-window KPI 摘要到 <path>.summary.txt.

        ⭐ 2026-10-19: 删 per-frame 块, 只保留 per-window (跟训练 / H743 1:1).
        """
        if not self.is_open or not self._path:
            return
        summary_path = self._path + '.summary.txt'
        # ====== Per-Window (跟训练 1:1) ======
        w_total = self.running_window_total
        w_attack = self.running_window_attack
        w_normal = self.running_window_normal
        w_detected = self.running_window_detected
        w_tp = self.running_window_tp
        w_fp = self.running_window_fp
        w_tn = self.running_window_tn
        w_fn = self.running_window_fn
        # Per-Window 4 指标: 跟训练 macro_f1 1:1
        w_recall = w_tp / w_attack if w_attack else 0
        w_fpr = w_fp / w_normal if w_normal else 0
        w_precision = w_tp / (w_tp + w_fp) if (w_tp + w_fp) else 0
        w_f1 = (2 * w_precision * w_recall / (w_precision + w_recall)
                if (w_precision + w_recall) else 0)
        w_accuracy = ((w_tp + w_tn) / w_total if w_total else 0)
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('Test Summary\n')
                f.write('=' * 60 + '\n')
                f.write(f'Log file:        {self._path}\n\n')
                f.write('─── Per-Window 评估 (跟训练 max-aggregation 1:1, window=16) ───\n')
                f.write(f'Total windows:   {w_total}\n')
                f.write(f'  Attack (Sent): {w_attack}  (16 帧任一帧 attack → max-aggregation)\n')
                f.write(f'  Normal:        {w_normal}  (16 帧全 normal)\n')
                f.write(f'  Detected:      {w_detected}  (H743 windows_attack 累计, 含 FP)\n')
                # ⭐ Round 71 (2026-10-13): 公式 Attack - Detected 错误, 因为 Detected = TP + FP
                #   不是 TP. 当 FPs > 0 时 Detected > Attack, 公式出负数 (e.g. test_20261013-233626
                #   报 "Missed (FN): -4" 实际 FN=0). 改用 Attack - TP, 跟下方 FN 行一致.
                f.write(f'  Missed (FN):   {w_attack - w_tp}  (Attack - TP, 应跟下方 FN 一致)\n')
                f.write(f'TP:              {w_tp}\n')
                f.write(f'FP:              {w_fp}\n')
                f.write(f'TN:              {w_tn}\n')
                f.write(f'FN:              {w_fn}\n')
                f.write(f'Recall (TP/Atk): {w_recall:.4f}  ({w_recall*100:.2f}%)  ⭐ 跟训练 Recall 对比\n')
                f.write(f'FPR    (FP/Norm):{w_fpr:.4f}  ({w_fpr*100:.2f}%)\n')
                f.write(f'Precision:       {w_precision:.4f}  ({w_precision*100:.2f}%)\n')
                f.write(f'F1:              {w_f1:.4f}  ({w_f1*100:.2f}%)  ⭐ 跟训练 F1m 对比\n')
                f.write(f'Accuracy:        {w_accuracy:.4f}  ({w_accuracy*100:.2f}%)\n')
        except Exception:
            pass

    def set_window_metrics(self, wm: dict) -> None:
        """从 IdsReportPanel._window_eval.get_metrics() 推 per-window 4 指标到 summary.

        ⭐ 第五十二轮 c: main_window._on_finished 调本方法把 WindowEvaluator 当前状态
        (TP/FP/TN/FN/n_windows) 喂给 TestLogger, write_summary 算 per-window 块.
        """
        if not wm:
            return
        self.running_window_tp = int(wm.get('tp', 0))
        self.running_window_fp = int(wm.get('fp', 0))
        self.running_window_tn = int(wm.get('tn', 0))
        self.running_window_fn = int(wm.get('fn', 0))
        # n_windows 也存一下 (跟 running_window_total 概念一致, 优先用 wm 给的值)
        n_w = int(wm.get('n_windows', 0))
        if n_w > 0:
            self.running_window_total = n_w

    def close(self) -> None:
        if self._file:
            self.write_summary()
            self._file.close()
            self._file = None
            self._writer = None


class SerialReader(QThread):
    """后台线程,100ms 轮询 Serial.in_waiting, 收到字节就 emit data_received."""
    data_received = pyqtSignal(bytes)
    POLL_INTERVAL_MS = 100

    def __init__(self, serial_handle, parent=None):
        super().__init__(parent)
        self._serial = serial_handle
        self._stopping = False

    def stop(self):
        self._stopping = True

    def run(self):
        while not self._stopping:
            try:
                n = self._serial.in_waiting
                if n:
                    chunk = self._serial.read(n)
                    if chunk:
                        self.data_received.emit(chunk)
                else:
                    self.msleep(self.POLL_INTERVAL_MS)
            except Exception:
                # 串口被关闭 / pyserial 句柄失效 / 线程退出都走这里
                return


# === GUI ==============================================================
class RTUReplayGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modbus RTU Replay  --  H743 IDS 测试工具")
        self.resize(820, 980)
        self.worker: Optional[ReplayWorker] = None
        # 手动模式状态
        self.manual_ser: Optional[serial.Serial] = None
        self.manual_rows: List[dict] = []
        self.manual_index = 0
        self.manual_loaded = False
        # 全局统计
        self.total_frames = 0
        self.total_bytes = 0
        self.start_time: Optional[float] = None
        # 帧大小历史 (每帧字节数 → 用于分布分析)
        self.frame_sizes: List[int] = []
        # ⭐ 第十九轮 (2026-09-20): 手动模式 IDS KPI 累计 (与 worker 计数器独立).
        #   手动发送一帧后, _record_send 据 row.binary result 决定 +attack 或 +normal,
        #   同步通知 IdsReportPanel. 自动模式 (worker) 走 worker.sent_stats 信号.
        self.manual_total_count = 0
        self.manual_attack_count = 0
        self.manual_normal_count = 0
        # RX 区状态
        self._rx_reader: Optional[SerialReader] = None
        # ⭐ 2026-10-19 SITE 4 capture: raw RX 字节落到文件 (无 buffer 上限)
        self._rx_dump_file: Optional[object] = None
        self._rx_dump_path: Optional[str] = None
        self._rx_dump_n_bytes: int = 0
        # ⭐ 第二十一轮 (2026-09-20): 结构化测试日志 (CSV + summary)
        self.test_logger = TestLogger()
        self._build_ui()
        self._wire()
        self._refresh_ports()

    # --- UI ---
    def _build_ui(self):
        # ⭐ 第四十一轮 (2026-10-02): QScrollArea 包装 root, 让 GUI 可上下滚动.
        #   起因: Round 40 加 MetricsChartPanel (matplotlib 暗色曲线) 后, 总高度超过常见
        #   笔记本屏幕 (典型 800-1080px 高), 用户看不到新加的曲线图表. 用 QScrollArea
        #   让 root 内容放进一个可滚动容器, 内容超出窗口高度时自动显示垂直滚动条.
        #   setWidgetResizable(True) 让容器宽度跟随窗口缩放, 高度按内容自动.
        #   Qt.ScrollBarAsNeeded 让滚动条只在需要时出现 (不撑高度).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 只允许垂直滚动
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)  # 无边框, 与窗口融为一体
        _apply_scrollbar_style(scroll)  # ⭐ 第四十一轮: 统一滚动条样式 (宽 12px, hover 蓝)
        self.setCentralWidget(scroll)
        central = QWidget()                # 容器 widget
        scroll.setWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)  # 留 8px 边距, 滚动条不贴边
        # ⭐ 第四十一轮: 暴露滚动区供测试用, 也供调试时手动滚动.
        self._scroll_area = scroll

        # 串口设置
        port_box = QGroupBox("串口设置")
        _style_chinese_groupbox(port_box)
        grid = QGridLayout(port_box)
        grid.addWidget(QLabel("COM:"), 0, 0)
        self.port_combo = QComboBox()
        grid.addWidget(self.port_combo, 0, 1)
        self.refresh_btn = QPushButton("🔄 刷新")
        grid.addWidget(self.refresh_btn, 0, 2)
        grid.addWidget(QLabel("波特率:"), 1, 0)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['9600', '19200', '38400', '57600', '115200', '230400', '460800'])
        self.baud_combo.setCurrentText('115200')
        grid.addWidget(self.baud_combo, 1, 1)
        for col, (lab, default, opts) in enumerate([
            ("数据位:", "8", ['7', '8']),
            ("校验:",   "N", ['N', 'E', 'O']),
            ("停止位:", "1", ['1', '2']),
        ]):
            grid.addWidget(QLabel(lab), 2, col * 2)
            cb = QComboBox()
            cb.addItems(opts)
            cb.setCurrentText(default)
            grid.addWidget(cb, 2, col * 2 + 1)
            if '数据位' in lab: self.databits_combo = cb
            elif '校验' in lab: self.parity_combo = cb
            else: self.stopbits_combo = cb

        # 打开/关闭按钮
        self.open_btn = QPushButton("📡 打开串口")
        self.open_btn.setStyleSheet("font-weight: bold; color: green;")
        self.close_btn = QPushButton("🔌 关闭串口")
        self.close_btn.setStyleSheet("font-weight: bold; color: #cc6600;")
        self.close_btn.setEnabled(False)
        port_btn_row = QHBoxLayout()
        port_btn_row.addWidget(self.open_btn)
        port_btn_row.addWidget(self.close_btn)
        port_btn_row.addStretch(1)
        grid.addLayout(port_btn_row, 3, 0, 1, 3)

        # 状态指示
        self.port_status_label = QLabel("● 串口: 未打开")
        self.port_status_label.setStyleSheet(
            "padding: 4px; background: #f4f4f4; border: 1px solid #ccc;"
        )
        grid.addWidget(self.port_status_label, 4, 0, 1, 3)

        # ⭐ 2026-10-01 第二十四轮: 串口设置 + 回放选项 同一行 (横向并排, 节省垂直空间)
        #   两者各有自己的 GroupBox, 用 QHBoxLayout 横向并排, 各自 stretch=1 平分宽度.
        top_row = QHBoxLayout()
        top_row.addWidget(port_box, stretch=1)

        # CSV
        csv_box = QGroupBox("数据集 (IanArffDataset.csv)")
        _style_chinese_groupbox(csv_box)
        csv_layout = QHBoxLayout(csv_box)
        csv_layout.addWidget(QLabel("路径:"))
        self.csv_edit = QLineEdit(DEFAULT_CSV)
        csv_layout.addWidget(self.csv_edit, stretch=1)
        self.csv_btn = QPushButton("📂 浏览...")
        csv_layout.addWidget(self.csv_btn)
        root.addWidget(csv_box)

        # 回放选项
        opt_box = QGroupBox("回放选项")
        _style_chinese_groupbox(opt_box)
        opt = QGridLayout(opt_box)
        # ⭐ 2026-10-01 第二十三轮: 删速度倍率 (CSV time × speed 倍率, 1x 跑 76h 太慢的旧逻辑),
        #   删循环播放 (用户只跑一次看 KPI), 删合成时序 (代码逻辑已不用, 留 UI 造成混淆).
        #   保留"自动发送间隔"作为唯一速度控制 (interval_spin: 0-1000ms), 默认 40ms (auto-40ms).
        opt.addWidget(QLabel("攻击过滤:"), 0, 0)
        self.attack_combo = QComboBox()
        self.attack_combo.addItems(['all (全部)'] + ATTACK_LABELS)
        opt.addWidget(self.attack_combo, 0, 1)
        opt.addWidget(QLabel("自动发送间隔 (ms):"), 1, 0)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(0, 1000)
        self.interval_spin.setValue(40)  # ⭐ 默认 40ms = auto-40ms 测试档
        self.interval_spin.setSuffix(" ms/帧 (0=禁用,走 CSV time)")
        self.interval_spin.setToolTip(
            "🟢 自动发送间隔 (覆盖 CSV time 列):\n"
            "  • 0   = 禁用, 用 CSV time 列作为自然间隔\n"
            "  • 1-1000 = 自动按 N 毫秒间隔发送每帧\n"
            "• 推荐值: 40ms (auto-40ms 标准), 100ms 慢观察, 10ms 压测, 500ms dump 测试"
        )
        opt.addWidget(self.interval_spin, 1, 1)
        # ⭐ 2026-10-01 第二十三轮: 新增起始帧选择, 让用户从指定 frame_idx 开始自动发送.
        opt.addWidget(QLabel("起始帧 (0-based):"), 2, 0)
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 9999999)
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.setSuffix(" (CSV 第 N 行)")
        self.start_frame_spin.setToolTip(
            "🟢 自动发送起始帧 (0 = 从头开始):\n"
            "  • 0 = 从 CSV 第 1 行开始\n"
            "  • N = 从 CSV 第 N+1 行开始 (过滤后行号)\n"
            "• 用途: 跳过 ramp-up, 或续测上次中断位置"
        )
        opt.addWidget(self.start_frame_spin, 2, 1)
        self.verbose_chk = QCheckBox("详细 TX 日志 (会刷屏)")
        opt.addWidget(self.verbose_chk, 3, 0)
        # ⭐ 2026-10-01 第二十四轮: 回放选项放进 top_row 与 port_box 并排
        top_row.addWidget(opt_box, stretch=1)
        root.addLayout(top_row)

        # === 手动模式 (单帧发送) ===
        man_box = QGroupBox("手动模式 (单帧发送 — 不走时间间隔)")
        _style_chinese_groupbox(man_box)
        # ⭐ 第三十七轮 (2026-10-02): 改 QVBoxLayout → QHBoxLayout, 帧预览在左,
        #   所有按钮 (row1 pos/cat, row2 status+load, row3 nav, row4 manage) 在右 stacked.
        man = QHBoxLayout(man_box)
        # 左侧: 帧预览 (大, 150px 高, stretch=1)
        left_col = QVBoxLayout()
        self.preview_label = QLabel("帧预览: (加载后显示)")
        self.preview_label.setStyleSheet(
            "font-family: Consolas, monospace; padding: 8px; "
            "background: #f4f4f4; border: 1px solid #ccc;"
        )
        self.preview_label.setWordWrap(True)
        # ⭐ 第三十三轮 (2026-10-02): 帧预览框放大, 96px → 150px (6+ 行 hex).
        #   第三十七轮: 移到 left_col 左侧, 高度保留 150px.
        self.preview_label.setMinimumHeight(150)
        self.preview_label.setMaximumHeight(150)
        left_col.addWidget(self.preview_label, stretch=1)
        man.addLayout(left_col, stretch=1)

        # 右侧: 所有按钮 stacked (stretch=2 给按钮足够宽度)
        right_col = QVBoxLayout()
        # ⭐ 第三十三轮 (2026-10-02): "当前位置 / 类别" 上移到顶部 (状态信息优先),
        #   原 row2 (pos/cat) 现在变 row1, 原 row1 (status + load) 变 row2.
        # ⭐ 第三十八轮 (2026-10-02): 加载按钮从 row2 移到 row1 最左侧, 在 "当前位置" 之前.
        #   用户要求加载按钮放最左, 操作流: 先点"加载", 再观察"当前位置/类别"变化.
        row1 = QHBoxLayout()
        # ⭐ 第三十轮: 加载按钮改深蓝色, 突出主操作
        # ⭐ 第三十八轮: 加载按钮位置从 row2 移到这里 (row1 最左)
        self.load_btn = QPushButton("📥 加载 (应用过滤)")
        self.load_btn.setStyleSheet(
            "background-color: #003366; color: white; font-weight: bold; padding: 6px 12px;"
        )
        row1.addWidget(self.load_btn)
        row1.addWidget(QLabel("当前位置:"))
        self.pos_label = QLabel("— / —")
        self.pos_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        row1.addWidget(self.pos_label)
        row1.addStretch(1)
        self.cat_label = QLabel("类别: —")
        self.cat_label.setStyleSheet("font-weight: bold;")
        row1.addWidget(self.cat_label)
        right_col.addLayout(row1)

        row2 = QHBoxLayout()
        # ⭐ 第三十轮 (2026-10-02): 删除 "启用手动模式" 复选框, 改为状态指示标签.
        #   手动模式状态完全由 manual_loaded 驱动 (加载=开启, 清空=关闭),
        #   不再独立 toggle, 避免与"加载"按钮逻辑冲突.
        # ⭐ 第三十八轮: 加载按钮已移走, row2 现在只剩 manual_status_label.
        self.manual_status_label = QLabel("⏸ 手动模式: 未加载")
        self.manual_status_label.setStyleSheet("color: #888; font-weight: bold;")
        row2.addWidget(self.manual_status_label)
        row2.addStretch(1)
        right_col.addLayout(row2)

        row3 = QHBoxLayout()
        # ⭐ 第三十九轮 (2026-10-02): "发送并前进" (主操作绿色) 放最左, "上一帧" 跟在后面.
        #   操作流: 先点 "发送并前进", 然后想反悔才点 "上一帧". 主操作靠左, 符合 GUI 习惯.
        self.send_next_btn = QPushButton("📤 发送并前进")
        self.send_next_btn.setStyleSheet(
            "font-weight: bold; background: #4CAF50; color: white; padding: 8px;"
        )
        self.prev_btn = QPushButton("◀ 上一帧")
        # ⭐ 第三十九轮: "重发当前帧" 移到 row4 (与 reset/reload/unload 同组, 都是行级操作).
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(1, 274628)
        self.goto_spin.setPrefix("跳到: ")
        self.goto_btn = QPushButton("🎯 跳转")
        for b in (self.send_next_btn, self.prev_btn, self.goto_spin, self.goto_btn):
            row3.addWidget(b)
        self.prev_btn.setEnabled(False)
        self.send_next_btn.setEnabled(False)
        self.goto_btn.setEnabled(False)
        right_col.addLayout(row3)

        # 第 4 行: 发送区管理 (重发当前帧 / 回到首帧 / 重新加载 / 清空加载)
        # ⭐ 第三十九轮 (2026-10-02): "重发当前帧" 移到 row4 最左, 在 "回到首帧" 之前.
        #   row4 现在是行级操作组: 重发 → 回到首帧 → 重新加载 → 清空加载.
        row4 = QHBoxLayout()
        self.resend_btn = QPushButton("🔁 重发当前帧")
        self.reset_ptr_btn = QPushButton("⏮ 回到首帧")
        self.reset_ptr_btn.setToolTip("手动指针回到 #1 (不丢已加载行)")
        self.reload_btn = QPushButton("🔄 重新加载")
        self.reload_btn.setToolTip("按当前过滤重新读 CSV")
        self.unload_btn = QPushButton("🗑 清空加载")
        self.unload_btn.setToolTip("丢弃已加载的所有行,关闭手动模式")
        for b in (self.resend_btn, self.reset_ptr_btn, self.reload_btn, self.unload_btn):
            b.setEnabled(False)
            row4.addWidget(b)
        row4.addStretch(1)
        right_col.addLayout(row4)
        man.addLayout(right_col, stretch=2)  # ⭐ 第三十七轮: right_col 加 stretch=2 给按钮足够宽度

        # ⭐ 第三十七轮 (2026-10-02): 手动模式卡片单独一行 (全宽), 内部 preview LEFT + 按钮 RIGHT.
        #   新 root 顺序:
        #     1. top_row (port + csv + opt)
        #     2. man_box (单独一行, 全宽, 内部 QHBoxLayout: preview LEFT, 按钮 RIGHT)
        #     3. ctrl (start/pause/stop)
        #     4. progress_bar
        #     5. stats_label (单独一行, 加 padding 突出)
        #     6. log_rx_row (rx_box | log_box) ← Round 37 新横向容器
        #     7. ids_report
        # 移除旧的 self.stats_rx_row / self.tx_rx_row (Round 30/34/36 引入, 现已废弃).
        root.addWidget(man_box)

        # 控制 (自动回放)
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("▶ 自动开始")
        # ⭐ 2026-10-01 第二十四轮: 自动开始按钮浅蓝色 (突出主操作)
        self.start_btn.setStyleSheet(
            "background-color: #ADD8E6; font-weight: bold; color: #003366;"
        )
        self.pause_btn = QPushButton("⏸ 暂停")
        self.stop_btn = QPushButton("⏹ 停止")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        for b in (self.start_btn, self.pause_btn, self.stop_btn):
            b.setMinimumHeight(36)
            ctrl.addWidget(b)
        root.addLayout(ctrl)

        # 进度
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        # ⭐ 第三十七轮 (2026-10-02): 接收区 + 日志 同行横向排布.
        #   log_rx_row (QHBoxLayout) 把 rx_box 与 log_box 放在同一 row.
        #   旧布局 (Round 36): stats_rx_row (stats + rx) + log_box 单独一行.
        #   新布局 (Round 37): stats_label 单独一行 + log_rx_row (rx + log).
        #   rx_box 与 log_box 横向并排, 节省垂直空间, 让手动模式卡片更突出.
        # 先创建 stats_label (单独一行, 全宽, 加 padding 突出"已发送"实时计数).
        self.stats_label = QLabel("已发送: 0 帧 / 0 字节 / 平均 0 B/帧   状态: 空闲")
        self.stats_label.setStyleSheet("font-weight: bold; padding: 4px 6px; "
                                        "background: #f9f9f9; border: 1px solid #e0e0e0;")
        root.addWidget(self.stats_label)

        # ⭐ 第三十二轮 (2026-10-02): "📊 帧大小分布" 按钮从 stats_row 移到
        #   log_toolbar (放 "💾 关闭并写 summary" 之后), 与日志功能按钮集中.
        self.dist_btn = QPushButton("📊 帧大小分布")

        # 接收区 (RX) — 显示从串口收到的字节 (H743 主站轮询请求等)
        rx_box = QGroupBox("📥 接收区 (RX)")
        _style_chinese_groupbox(rx_box)
        rx_layout = QVBoxLayout(rx_box)
        self.rx_panel = RxPanel()
        rx_layout.addWidget(self.rx_panel, stretch=1)
        # ⭐ 第三十轮 (2026-10-02): 去除 maxHeight=220 限制, 加 minHeight 与手动模式卡片对齐.
        rx_box.setMinimumHeight(380)

        # 日志 (顶部带清空按钮的工具栏)
        log_box = QGroupBox("📋 日志")
        _style_chinese_groupbox(log_box)
        log_layout = QVBoxLayout(log_box)
        log_toolbar = QHBoxLayout()
        self.log_clear_btn = QPushButton("\U0001f5d1 清空日志")
        self.log_clear_btn.clicked.connect(lambda: self.log_text.clear())
        self.log_clear_btn.setToolTip("清空日志窗口 (不影响已发送统计)")
        log_toolbar.addWidget(self.log_clear_btn)
        # ⭐ 第二十一轮 (2026-09-20): 测试日志按钮
        self.log_open_btn = QPushButton("📂 开启 CSV 日志")
        self.log_open_btn.setToolTip(
            "创建 logs/test_<时间戳>.csv, 自动记录每帧 SEND + 每个 [ALARM] 的 "
            "TP/FP/prob + 派生 KPI. 用于 1000 条测试根因分析."
        )
        log_toolbar.addWidget(self.log_open_btn)
        self.log_close_btn = QPushButton("💾 关闭并写 summary")
        self.log_close_btn.setToolTip("关闭 CSV, 写 .summary.txt 摘要")
        self.log_close_btn.setEnabled(False)
        log_toolbar.addWidget(self.log_close_btn)
        # ⭐ 第三十二轮 (2026-10-02): 帧大小分布按钮放日志组关闭按钮之后
        log_toolbar.addWidget(self.dist_btn)
        # ⭐ 2026-10-19 SITE 4 capture: raw RX dump to file (无 buffer 上限)
        self.rx_dump_btn = QPushButton("📥 RX 原始字节抓包")
        self.rx_dump_btn.setToolTip(
            "把 SerialReader 收到的全部字节写到 rx_raw_<时间戳>.log (无 5KB buffer 上限).\n"
            "用于 SITE 4 capture: 解锁多 [RAW18-DBG] dumps 时不被 terminal scrollback 截断.\n"
            "再次点击停止抓包并 close 文件."
        )
        self.rx_dump_btn.setCheckable(True)
        self.rx_dump_btn.toggled.connect(self._toggle_rx_dump)
        log_toolbar.addWidget(self.rx_dump_btn)
        log_toolbar.addStretch(1)
        log_layout.addLayout(log_toolbar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        log_layout.addWidget(self.log_text, stretch=1)

        # ⭐ 第三十七轮: log_rx_row (rx_box | log_box 横向).
        #   rx_box ~1/3 (stretch=1), log_box ~2/3 (stretch=2, 主显示区).
        #   log_box 仍 setMinimumHeight(380) via log_text stretch=1, 保证视觉对齐.
        self.log_rx_row = QHBoxLayout()
        self.log_rx_row.setContentsMargins(0, 0, 0, 0)
        self.log_rx_row.setSpacing(6)
        self.log_rx_row.addWidget(rx_box, stretch=1)
        self.log_rx_row.addWidget(log_box, stretch=2)
        root.addLayout(self.log_rx_row, stretch=1)

        # ⭐ 第三十七轮: 旧的 self.stats_rx_row / self.tx_rx_row 已废弃移除.
        #   rx_box 现在属于 log_rx_row.

        # IDS Report 面板 — 解析 H743 上行的 [ALARM] N printf (第十六轮)
        self.ids_parser = IdsLineParser()
        self.ids_report = IdsReportPanel(self.ids_parser)
        root.addWidget(self.ids_report)

        # ⭐ 第四十轮 (2026-10-02): IDS 5 指标曲线面板 (实时)
        #   matplotlib FigureCanvasQTAgg, 5 条曲线 (Precision/Recall/F1/Accuracy/FPR)
        #   数据源: IdsReportPanel.metrics_updated 信号
        #   控件: 🗑 清空曲线 + ⏸/▶ 暂停/继续
        self.metrics_chart = MetricsChartPanel(self)
        self.metrics_chart_box = QGroupBox("📈 IDS 曲线图表 (实时性能)")
        _style_chinese_groupbox(self.metrics_chart_box)
        chart_layout = QVBoxLayout(self.metrics_chart_box)
        # toolbar (清空 + 暂停) — 已在 MetricsChartPanel 内部, 这里直接 addWidget
        chart_layout.addWidget(self.metrics_chart, stretch=1)
        root.addWidget(self.metrics_chart_box, stretch=1)

    # --- 信号 ---
    def _wire(self):
        self.refresh_btn.clicked.connect(self._refresh_ports)
        self.csv_btn.clicked.connect(self._browse_csv)
        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)
        # 串口开关
        self.open_btn.clicked.connect(self._open_port)
        self.close_btn.clicked.connect(self._close_port)
        # 手动模式
        self.load_btn.clicked.connect(self._manual_load)
        self.send_next_btn.clicked.connect(self._manual_send_next)
        self.prev_btn.clicked.connect(self._manual_prev)
        self.resend_btn.clicked.connect(self._manual_resend)
        self.goto_btn.clicked.connect(self._manual_goto)
        # ⭐ 第三十轮 (2026-10-02): 删除 manual_enable_chk 复选框及对应 stateChanged 信号.
        #   手动模式状态由 manual_loaded 标志驱动 (在 _manual_load/_manual_unload 更新).
        # 发送区管理按钮
        self.reset_ptr_btn.clicked.connect(self._manual_reset_ptr)
        self.reload_btn.clicked.connect(self._manual_load)  # 等同于按 加载
        self.unload_btn.clicked.connect(self._manual_unload)
        # 统计
        self.dist_btn.clicked.connect(self._show_size_distribution)
        # ⭐ 第二十一轮 (2026-09-20): 测试日志按钮
        self.log_open_btn.clicked.connect(self._open_test_log)
        self.log_close_btn.clicked.connect(self._close_test_log)
        # ⭐ 第四十轮 (2026-10-02): IDS 曲线面板数据流
        #   每个 [ALARM] 事件后 IdsReportPanel 推送 5 个派生率, 曲线面板追加点
        #   测试开始/重置时, 同步清空曲线 history
        self.ids_report.metrics_updated.connect(self.metrics_chart._on_metrics_updated)
        self.ids_report.metrics_reset.connect(self.metrics_chart._on_metrics_reset)
        # ALARM 事件钩子: IdsLineParser.alarm_with_prob → TestLogger.log_alarm
        # (用含 prob 的信号, 而不是 fp_changed, 这样日志能记录 prob_x1000)
        self.ids_parser.alarm_with_prob.connect(self._on_alarm_for_log)
        # ⭐ 第六十五轮 (2026-10-13): alarm_annotated → RX 区域追加 "实际是 attack/normal"
        #   标注, 让用户实时看到 H743 实际判到的帧是不是 attack. 数据源: IdsLineParser
        #   在 _parse_line 用 h743_frame 查 _pc_frame_binary (PC 端 frame→binary 映射).
        self.ids_parser.alarm_annotated.connect(self._on_alarm_annotated)

    def _refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("(无可用串口)", None)
            return
        # ⭐ 2026-10-11: 优先选中 COM4 (用户主用串口), 没有则选第一个.
        preferred = self.port_combo.findData('COM4')
        if preferred >= 0:
            self.port_combo.setCurrentIndex(preferred)
        else:
            self.port_combo.setCurrentIndex(0)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV", str(Path(self.csv_edit.text()).parent),
            "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            self.csv_edit.setText(path)

    def _get_port(self) -> Optional[str]:
        return self.port_combo.currentData()

    def _get_attack_filter(self) -> Optional[str]:
        t = self.attack_combo.currentText().split(' ')[0]
        return None if t == 'all' else t

    # --- 控制 ---
    def _start(self):
        if self.worker and self.worker.isRunning():
            return
        # ⭐ 第三十九轮 (2026-10-02): 处于手动模式时点自动开始, 先自动清空手动模式.
        #   用户选择: 自动清空手动模式后再启动 (无需用户手动点清空).
        #   避免手动行 + 自动行在统计/CSV 上混淆.
        if self.manual_loaded:
            self._log("🗑 自动模式启动: 先自动清空手动模式...")
            self._manual_unload()
        port = self._get_port()
        if not port:
            QMessageBox.warning(self, "无串口", "请先选择 COM 口 (点 🔄 刷新)")
            return
        try:
            baud = int(self.baud_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "波特率", "波特率无效")
            return
        csv_path = self.csv_edit.text().strip()
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "CSV", f"文件不存在:\n{csv_path}")
            return

        # ⭐ 第二十一轮 (2026-09-20): 自动模式也要打开 manual_ser + 启 RX reader
        #   之前 worker 自己开 serial, GUI 的 manual_ser 没开, SerialReader 不跑,
        #   H743 的 [ALARM] N 行根本没进 IdsLineParser. 修复: 先确保 manual_ser
        #   打开, 启 RX reader, 然后把 manual_ser 共享给 worker (避免双开 COM 失败).
        if not self._ensure_manual_ser():
            return  # 用户取消了 QMessageBox
        # _ensure_manual_ser() 不启 RX reader, 手动模式也不会自动启.
        # 但 _open_port() 会. 这里直接调一下, 幂等.
        self._start_rx_reader()

        # ⭐ 第四十五轮 (2026-10-08): 自动模式 start 必先清零 IDS KPI + parser 状态.
        #   旧 _start() 不调 reset, 旧测试的 _tp_budget / _binary_history / _last_alarm_count /
        #   _detected_baseline 全部残留. H743 modbus_ids SW=16 仍含旧测试尾部 attack features,
        #   新测试首个 alarm 触发时 window_has_attack=True (旧 binary_history) + budget 充足
        #   (旧 budget) → 错标 TP, 用户报告"没下发攻击 TP 也会计数".
        #   修复: start = 新测试, 必清. 配合 start_frame 已能处理"从某帧继续跑".
        self.ids_report.reset_counters()
        self._log("🔄 IDS 计数器已重置 (新一轮测试)")

        # 重置统计
        self.total_frames = 0
        self.total_bytes = 0
        self.frame_sizes = []
        self.start_time = time.time()
        # ⭐ 2026-10-01 第二十三轮: 测试日志 label 简化 (无 speed/loop/synth_timing 字段)
        interval_ms = self.interval_spin.value()
        start_frame = self.start_frame_spin.value()
        if not self.test_logger.is_open:
            if interval_ms > 0:
                label = f'auto-{interval_ms}ms'
            else:
                label = 'auto-csv-time'  # ⭐ CSV time 自然间隔 (1s 精度整数)
            path = self.test_logger.open(label=label)
            self.test_logger.log_event('START', note=f'interval_ms={interval_ms} '
                                                     f'start_frame={start_frame} '
                                                     f'filter={self._get_attack_filter() or "all"}')
            self.log_open_btn.setEnabled(False)
            self.log_close_btn.setEnabled(True)
            self._log(f"📂 测试日志已自动开启: {path}")

        self.worker = ReplayWorker(
            csv_path=csv_path, port=port, baud=baud,
            attack_filter=self._get_attack_filter(),
            start_frame=start_frame,
            verbose=self.verbose_chk.isChecked(),
            interval_ms=interval_ms,
            serial_obj=self.manual_ser,  # ⭐ 第二十一轮: 共享 manual_ser, worker 不重开
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        # ⭐⭐ 第十九轮: 3 元组 (总发送, 攻击次数, 正常数) → IdsReportPanel
        #   ⭐ 2026-10-19: 删 per-frame state 维护. sent_stats 仅触发 _update_rates.
        self.worker.sent_stats.connect(self.ids_report._on_sent_stats)
        # ⭐ 2026-10-19: 删 attack_sent 连接 (per-frame KPI 已删除)
        # ⭐⭐ 第五十二轮 (2026-10-09): 4 个主 KPI (Total / Normal / Attack / Detected)
        #   全部按 16 帧 1 window 算 (用户要求). 跟训练 / H743 1:1 对齐.
        #   window_stats 信号: 每凑齐 1 个 16 帧 window, emit 一次 (Round 51 新增).
        #   Detected label 由 _on_detected 单独维护 (H743 windows_attack 累计, per-window).
        self.worker.window_stats.connect(self.ids_report._on_window_stats)
        # ⭐⭐ 第五十二轮 c (2026-10-10): window_stats → TestLogger 维护 per-window 总/attack/normal
        #   计数, write_summary 加 per-window 块 (跟训练 1:1 评估, Recall=TP/Attack_windows).
        #   TestLogger 不监听 count_changed (那是 per-alarm 累计, 用 baseline 锚点), Detected
        #   数字从 H743 windows_attack 累计拿. 这里改用 main_window._on_detected_for_summary
        #   做 baseline 校正 (跟 IdsReportPanel._on_detected 一样的逻辑).
        self.worker.window_stats.connect(self._on_window_stats_for_summary)
        self.ids_parser.count_changed.connect(self._on_detected_for_summary)
        # ⭐⭐ 第二十轮: 每帧类型 → IdsLineParser (为下次 [ALARM] 准备 TP/FP 判定)
        self.worker.frame_sent_type.connect(self.ids_parser.set_last_was_attack)
        # ⭐ 第六十三轮 (2026-10-12): 每帧类型 → IdsLineParser._on_pc_frame_sent
        #   记每 PC 发送帧的 unix-ms 时间戳, [ALARM] 上行时算 lag_ms 反向 shift 归属.
        #   跟 set_last_was_attack 共用 frame_sent_type 信号 (multi-subscriber fan-out).
        self.worker.frame_sent_type.connect(self.ids_parser._on_pc_frame_sent)
        # ⭐ 第四十八轮: 每帧类型 → WindowEvaluator (per-window 评估, 跟训练 1:1)
        self.worker.frame_sent_type.connect(self.ids_report._window_eval.on_frame)
        # ⭐ 2026-10-13 Round 69: 修复 per-window summary TP/FP=0 bug.
        #   IdsLineParser 解析到 h743_frame 时 emit alarm_frame_index(int), 这里转发到
        #   IdsReportPanel._window_eval.on_alarm_with_frame, 让 summary 跟训练 1:1 的
        #   WindowEvaluator 同时拿到 attack (per-frame 信号) + alarm (此信号) 两侧信息.
        self.ids_parser.alarm_frame_index.connect(self.ids_report._window_eval.on_alarm_with_frame)
        # ⭐ 2026-10-11: 每帧 attack 状态 → IdsReportPanel 实时红色反馈
        self.worker.frame_sent_type.connect(self.ids_report._refresh_attack_alarm_color)
        # ⭐⭐ 第二十一轮: 每帧 (n_bytes, is_attack) → TestLogger
        #   这是关键修复: 之前 TestLogger 只接到手动 _record_send, 自动回放完全没记录
        #   (1073 帧自动回放后 CSV 仍只有 START+FINISH 两行, 0 SEND 事件)
        self.worker.frame_sent_log.connect(self._record_send)

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        self.worker.start()
        interval_str = f"interval={interval_ms}ms/帧" if interval_ms > 0 else "interval=CSV-time"
        self._log(f"▶ 开始: {port} @ {baud}  {interval_str}"
                  f"  start_frame={start_frame}"
                  f"  filter={self._get_attack_filter() or 'all'}")

    def _toggle_pause(self):
        if self.worker:
            self.worker.toggle_pause()
            self.pause_btn.setText("▶ 继续" if self.worker._paused else "⏸ 暂停")
            self._log("⏸ 暂停" if self.worker._paused else "▶ 继续")

    def _stop(self):
        if self.worker:
            self._log("⏹ 停止中...")
            self.worker.stop()

    # --- 回调 ---
    def _on_progress(self, current: int, total: int,
                     frames_sent: int, bytes_sent: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        # 同步到 GUI 全局计数 (worker 可能没 emit 完, 兜底)
        if frames_sent > self.total_frames:
            delta = frames_sent - self.total_frames
            self.total_frames = frames_sent
            # bytes 由 worker 自己算,这里只更新 frames
            # 如果 bytes_sent 没增加, 就用估算
            if bytes_sent > self.total_bytes:
                self.total_bytes = bytes_sent
        self._update_stats_label(state="运行中")

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self._update_stats_label(state="完成")
        # ⭐ 2026-10-11: 测试关闭时强制刷一次 GUI label, 跟 summary 数字 1:1.
        #   用户报告 GUI 显示 ~80% F1 但 summary 写 89.66% — GUI 标签停在 mid-test
        #   最后一个信号触发的 _update_rates, 而 H743 alarm 在那之后又来了几个,
        #   WindowEvaluator 累计 TP/FP 改善但 GUI 没刷新. 修复: close 时再算一次.
        try:
            self.ids_report._update_rates()
        except Exception:
            pass
        # ⭐ 第二十一轮: 自动回放结束时关闭测试日志
        if self.test_logger.is_open:
            # ⭐ 第五十二轮 c (2026-10-10): close 前 push WindowEvaluator 状态
            #   给 test_logger, write_summary 算 per-window 块 (跟训练 1:1).
            try:
                wm = self.ids_report._window_eval.get_metrics()
                self.test_logger.set_window_metrics(wm)
            except Exception:
                pass
            self.test_logger.log_event('FINISH', note='auto replay done')
            self.test_logger.close()
            self.log_open_btn.setEnabled(True)
            self.log_close_btn.setEnabled(False)
            self._log(f"💾 测试日志已自动关闭: {self.test_logger.path}")
            self._log(f"   摘要: {self.test_logger.path}.summary.txt")
            self.test_logger = TestLogger()  # 重置, 下次开启生成新文件

    def _on_error(self, msg: str):
        self._log(f"❌ {msg}")
        # ⭐ 第二十一轮: 错误时也保存测试日志
        if self.test_logger.is_open:
            # ⭐ 第五十二轮 c: 错误时也 push WindowEvaluator 状态
            try:
                wm = self.ids_report._window_eval.get_metrics()
                self.test_logger.set_window_metrics(wm)
            except Exception:
                pass
            self.test_logger.log_event('ERROR', note=msg[:200])
            self.test_logger.close()
            self.log_open_btn.setEnabled(True)
            self.log_close_btn.setEnabled(False)
            self._log(f"💾 测试日志已保存: {self.test_logger.path}")
            self.test_logger = TestLogger()

    # ⭐ 第五十二轮 c: TestLogger 维护 per-window 4 KPI 计数 (跟训练 1:1 评估)
    #   跟 IdsReportPanel 的 _on_window_stats / _on_detected 独立, 专门给 summary.txt 用.
    def _on_window_stats_for_summary(self, total_w: int, attack_w: int, normal_w: int) -> None:
        """ReplayWorker.window_stats 槽函数 — 更新 test_logger.running_window_*."""
        if not self.test_logger.is_open:
            return
        self.test_logger.running_window_total = total_w
        self.test_logger.running_window_attack = attack_w
        self.test_logger.running_window_normal = normal_w

    def _on_detected_for_summary(self, count: int) -> None:
        """IdsLineParser.count_changed 槽函数 — H743 windows_attack 累计 (per-window).

        ⭐ baseline 锚点逻辑跟 IdsReportPanel._on_detected 一样:
          首条 [ALARM] N → Detected = N - (N-1) = 1, 避免 H743 不重启时首条是旧值.
        """
        if not self.test_logger.is_open:
            return
        if self.test_logger.running_window_detected is None:  # type: ignore[unreachable]
            # 第一次: 锚定 baseline (类似 _detected_baseline)
            self.test_logger._detected_baseline = max(count - 1, 0)  # type: ignore[attr-defined]
        self.test_logger.running_window_detected = count - getattr(
            self.test_logger, '_detected_baseline', 0)

    # --- ⭐ 第二十一轮: 结构化测试日志 (CSV + summary.txt) ---
    def _open_test_log(self):
        """用户手动开启 CSV 测试日志. 若 worker 在跑则用 'auto-{speed}x' 后缀.

        ⭐ 2026-10-19: 删 per-frame KPI 同步 (running_total/attack/normal/tp/fp).
        """
        if self.test_logger.is_open:
            QMessageBox.information(self, "已开启", f"日志已在: {self.test_logger.path}")
            return
        label = ''
        if self.worker and self.worker.isRunning():
            interval_ms = self.interval_spin.value()
            if interval_ms > 0:
                label = f'auto-{interval_ms}ms'
            else:
                label = 'auto-csv-time'
        elif self.manual_loaded:
            label = f'manual-{len(self.manual_rows)}rows'
        path = self.test_logger.open(label=label)
        self.test_logger.log_event('OPEN', note=f'label={label}')
        self.log_open_btn.setEnabled(False)
        self.log_close_btn.setEnabled(True)
        self._log(f"📂 测试日志已开启: {path}")
        # ⭐ 2026-10-19: 删 per-frame KPI baseline 同步, 只保留日志开启功能.

    def _close_test_log(self):
        """用户手动关闭 CSV, 写 .summary.txt."""
        if not self.test_logger.is_open:
            return
        self.test_logger.log_event('CLOSE')
        path = self.test_logger.path
        self.test_logger.close()
        self.log_open_btn.setEnabled(True)
        self.log_close_btn.setEnabled(False)
        self._log(f"💾 测试日志已关闭: {path}")
        self._log(f"   摘要: {path}.summary.txt")

    def _on_alarm_for_log(self, count: int, prob_x1000: int, is_fp: bool, h743_frame: int = 0):
        """IdsLineParser.alarm_with_prob 信号回调 — 写一行 ALARM 事件到 CSV.

        ⭐ 2026-10-19: 删 running_tp/running_fp 参数 (per-frame 计数已删除).

        Args:
            count: H743 当前累计 frames_attack
            prob_x1000: H743 printf 输出的 prob=0.NNN × 1000 (范围 [0, 1000],
                       -1 表示 H743 没输出 prob 字段, 旧格式)
            is_fp: True=误报 (上一帧 normal), False=命中 (上一帧 attack)
            h743_frame: ⭐ 第六十一轮: H743 上行 frame=N 字段 (0 = 旧固件没发)
        """
        if not self.test_logger.is_open:
            return
        # ⭐ 第六十一轮: note 列加 frame= (旧固件无 frame= 时只写 prob=)
        prob_note = f'prob={prob_x1000/1000:.3f}' if prob_x1000 >= 0 else 'no_prob'
        if h743_frame > 0:
            note_with_frame = f'{prob_note} frame={h743_frame}'
        else:
            note_with_frame = prob_note
        self.test_logger.log_alarm(
            alarm_count=count,
            prob_x1000=prob_x1000 if prob_x1000 >= 0 else None,
            is_fp=is_fp,
            h743_frame=h743_frame,
            note=note_with_frame,
        )

    def _on_alarm_annotated(self, count: int, prob_x1000: int,
                             label: str, h743_frame: int) -> None:
        """⭐ 第六十五轮 (2026-10-13): IdsLineParser.alarm_annotated 信号回调.

        在 RX 区域最后一行 (即 [ALARM] 原始 RX 行) 末尾追加 "(attack)" /
        "(normal)" / "(?)" 简注. 不换行不重新起行, 直接 inline 紧贴.

        Args:
            count: H743 当前累计 frames_attack
            prob_x1000: H743 printf 输出的 prob=0.NNN × 1000 (-1 = 旧固件)
            label: "attack" / "normal" / "?" (h743_frame 在 _pc_frame_binary 里查不到时)
            h743_frame: H743 上行 frame=K 字段
        """
        # 颜色: attack 红色, normal 灰色, ? 黄色
        if label == "attack":
            color = "#cc3333"
            tag = "(attack)"
        elif label == "normal":
            color = "#888888"
            tag = "(normal)"
        else:
            color = "#cc9933"
            tag = "(?)"
        if hasattr(self, 'rx_panel') and self.rx_panel is not None:
            # ⭐ Round 70 (2026-10-13): 用 QTextCharFormat 显式 setForeground, 避免
            #   insertHtml 留下 cursor format 被下一个 append() 继承, 之前观察到的
            #   "[ALARM] N (attack)" 整行变灰 / "(normal)" 变红 即此问题.
            #   insertHtml 后 cursor.charFormat().foreground() 仍带新色, 下次 text_edit.append
            #   新段落继承该色, 出现颜色错位. 修复: insert 前 reset 字符格式为默认.
            cursor = self.rx_panel.text_edit.textCursor()
            cursor.movePosition(cursor.End)
            # 显式重置字符格式 (避免继承前一次 insertHtml 留下的彩色 format)
            default_fmt = QTextCharFormat()
            default_fmt.setForeground(QColor())  # 默认前景 (黑)
            cursor.setCharFormat(default_fmt)
            cursor.insertHtml(f' <span style="color: {color};"><b>{tag}</b></span>')
            # 再次重置, 避免下一个 append 继承
            cursor.movePosition(cursor.End)
            cursor.setCharFormat(default_fmt)
            self.rx_panel.text_edit.setTextCursor(cursor)
            self.rx_panel.text_edit.ensureCursorVisible()

    # --- 全局统计 ---
    def _record_send(self, n_bytes: int, is_attack: bool = False):
        """记录一帧发送 (手动 + 自动都用)。

        Args:
            n_bytes:   本帧字节数
            is_attack: 本帧是否是攻击帧 (binary==1).
                       手动模式从 row['binary result'] 取, 决定是否计入
                       manual_attack_count / manual_normal_count.
                       自动模式下 worker 已独立 emit sent_stats, 此处不再重复
                       (worker 信号会被 IdsReportPanel 直接消费).

        第二十轮 (2026-09-20): 同时通知 IdsLineParser 本帧类型,
          让 parser 在下次 [ALARM] 时知道是 TP 还是 FP.
        第二十一轮 (2026-09-20): 若 TestLogger 已开启, 写一行 SEND 事件.
        """
        self.total_frames += 1
        self.total_bytes += n_bytes
        self.frame_sizes.append(n_bytes)
        # ⭐ 第二十轮: 告诉 IdsLineParser 最近一帧是不是 attack (用于 FP 判定).
        #   ⚠️ 2026-10-01 第二十五轮 bug 修复: 仅手动模式下调用. 自动模式下 worker.frame_sent_type
        #   已直接 connect 到 ids_parser.set_last_was_attack (line 1781), 这里再调一次会让
        #   _binary_history 每个 binary 存 2 份, H743 [ALARM] N 读 _binary_history[-N..-1]
        #   时映射到错误的帧, TP/FP 分类彻底错乱 (测试中 F1=47% 几乎一半是 duplicate 导致的
        #   wrong window offset, carry-over 救不了, 真根因在此).
        if self.worker is None:  # 仅手动模式需要此处显式通知 parser
            self.ids_parser.set_last_was_attack(is_attack)
            # ⭐ 第四十八轮: 手动模式也喂 per-window 评估器
            self.ids_report._window_eval.on_frame(is_attack)
        # ⭐ 第二十一轮: 写测试日志 SEND 事件
        if self.test_logger.is_open:
            self.test_logger.log_send(
                frame_idx=self.total_frames,
                binary=1 if is_attack else 0,
                n_bytes=n_bytes,
                note='auto' if self.worker else 'manual',
            )
        # ⭐ 第十九轮: 手动模式累加 3 个 IDS KPI, 通知 IdsReportPanel.
        if self.worker is None:
            self.manual_total_count += 1
            if is_attack:
                self.manual_attack_count += 1
            else:
                self.manual_normal_count += 1
            self.ids_report._on_sent_stats(
                self.manual_total_count,
                self.manual_attack_count,
                self.manual_normal_count,
            )
        else:
            # 自动模式下也同步 worker 的统计 (避免进度回调延迟造成显示滞后)
            self.worker.frames_sent = self.total_frames
            self.worker.bytes_sent = self.total_bytes
        self._update_stats_label(state="运行中" if self.worker else "手动")

    def _update_stats_label(self, state: str = "空闲"):
        avg = self.total_bytes / max(1, self.total_frames)
        elapsed = 0.0
        if self.start_time:
            elapsed = time.time() - self.start_time
        bps = self.total_bytes / elapsed if elapsed > 0 else 0.0
        fps = self.total_frames / elapsed if elapsed > 0 else 0.0
        self.stats_label.setText(
            f"已发送: {self.total_frames} 帧 / {self.total_bytes} 字节 / "
            f"平均 {avg:.1f} B/帧 / {fps:.1f} 帧·s⁻¹ / {bps:.0f} B·s⁻¹   "
            f"状态: {state}"
        )

    def _show_size_distribution(self):
        """弹出帧大小分布表 (按 size 分组 + min/max/median)。"""
        if not self.frame_sizes:
            QMessageBox.information(self, "无数据", "还没发送任何帧")
            return
        from collections import Counter
        c = Counter(self.frame_sizes)
        total = sum(c.values())
        sizes = sorted(c.keys())
        # 算统计
        import statistics
        all_sizes = self.frame_sizes
        lines = [
            "帧大小分布统计",
            "=" * 60,
            f"总帧数: {total}",
            f"总字节: {sum(all_sizes)}",
            f"最小: {min(all_sizes)} B",
            f"最大: {max(all_sizes)} B",
            f"平均: {statistics.mean(all_sizes):.2f} B",
            f"中位: {statistics.median(all_sizes):.0f} B",
            f"众数: {statistics.mode(all_sizes)} B",
            "",
            "按字节数分组 (size | count | %) :",
            "-" * 40,
        ]
        for s in sizes:
            cnt = c[s]
            pct = 100.0 * cnt / total
            bar = '#' * int(pct / 2)  # 简单的 ascii bar
            lines.append(f"  {s:>4} B  | {cnt:>5} ({pct:5.1f}%) {bar}")

        msg = "\n".join(lines)
        # 用 dialog 显示, 可滚动
        dlg = QMessageBox(self)
        dlg.setWindowTitle("帧大小分布")
        dlg.setText(f"已发送 {total} 帧, {len(sizes)} 种不同字节数")
        dlg.setDetailedText(msg)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec_()

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

    # --- 手动模式 ---
    CAT_NAMES = {0: 'normal', 1: 'nmri', 2: 'cmri', 3: 'msci',
                 4: 'mpci', 5: 'mfci', 6: 'dos', 7: 'recon'}

    def _set_manual_state(self, loaded: bool):
        """⭐ 第三十轮 (2026-10-02): 手动模式状态完全由 manual_loaded 驱动.
        加载 → 启用手动 (禁用自动按钮 + 状态标签变绿).
        清空 → 关闭手动 (启用自动按钮 + 状态标签变灰).
        """
        if loaded:
            self.manual_status_label.setText("▶ 手动模式: 已加载")
            self.manual_status_label.setStyleSheet(
                "color: #006600; font-weight: bold;"
            )
            # 启用手动时, 禁用自动按钮 (避免冲突)
            self.start_btn.setEnabled(False)
        else:
            self.manual_status_label.setText("⏸ 手动模式: 未加载")
            self.manual_status_label.setStyleSheet(
                "color: #888; font-weight: bold;"
            )
            # 清空后允许自动回放
            self.start_btn.setEnabled(True)

    def _manual_load(self):
        """读取 CSV + 应用 attack 过滤 + 缓存到内存 (不启动 worker)。"""
        csv_path = self.csv_edit.text().strip()
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "CSV", f"文件不存在:\n{csv_path}")
            return

        attack_filter = self._get_attack_filter()
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            QMessageBox.critical(self, "CSV 错误", str(e))
            return

        def passes(row):
            if not attack_filter:
                return True
            try:
                ci = int((row.get('categorized result') or '').strip())
            except (ValueError, TypeError):
                return False
            if attack_filter == 'normal':
                return ci == 0
            return self.CAT_NAMES.get(ci, '') == attack_filter

        self.manual_rows = [r for r in rows if passes(r)]
        self.manual_index = 0
        self.manual_loaded = True
        self.goto_spin.setMaximum(len(self.manual_rows))

        self._log(f"✓ 加载完成: {len(self.manual_rows)}/{len(rows)} 行 "
                  f"(filter={attack_filter or 'all'})")
        self._update_manual_preview()

        self.send_next_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.resend_btn.setEnabled(True)
        self.goto_btn.setEnabled(True)
        # 第 4 行管理按钮:加载成功后启用
        self.reset_ptr_btn.setEnabled(True)
        self.reload_btn.setEnabled(True)
        self.unload_btn.setEnabled(True)
        # ⭐ 第三十轮: 手动模式开启 → 禁用自动按钮, 状态标签变绿
        self._set_manual_state(loaded=True)

    def _manual_reset_ptr(self):
        """手动指针回到首帧 (保留已加载行)。"""
        if not self.manual_loaded or not self.manual_rows:
            self._log("⚠ 未加载,无法重置指针")
            return
        self.manual_index = 0
        self._log("⏮ 手动指针已回到首帧 (#1)")
        self._update_manual_preview()

    def _manual_unload(self):
        """丢弃已加载的所有行,关闭手动模式。"""
        self.manual_rows = []
        self.manual_index = 0
        self.manual_loaded = False
        self.goto_spin.setMaximum(1)
        self.send_next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.resend_btn.setEnabled(False)
        self.goto_btn.setEnabled(False)
        self.reset_ptr_btn.setEnabled(False)
        self.reload_btn.setEnabled(False)
        self.unload_btn.setEnabled(False)
        # 清空预览
        self.preview_label.setText("帧预览: (加载后显示)")
        self.pos_label.setText("— / —")
        # ⭐ 第三十轮: 手动模式关闭 → 启用自动按钮, 状态标签变灰
        self._set_manual_state(loaded=False)
        self.cat_label.setText("类别: —")
        self._log("🗑 已清空手动加载")

    def _update_manual_preview(self):
        if not self.manual_loaded or not self.manual_rows:
            return
        if self.manual_index >= len(self.manual_rows):
            self.manual_index = len(self.manual_rows) - 1
        if self.manual_index < 0:
            self.manual_index = 0

        row = self.manual_rows[self.manual_index]
        frame = build_modbus_rtu_frame(row)
        cat_int = int(row.get('categorized result', '0') or 0)
        cat_name = self.CAT_NAMES.get(cat_int, '?')

        self.pos_label.setText(f"{self.manual_index + 1} / {len(self.manual_rows)}")
        self.cat_label.setText(f"类别: {cat_name} (cat={cat_int})")
        self.preview_label.setText(
            f"帧预览 ({len(frame)} B):\n{frame.hex(' ')}\n"
            f"addr={row['address']} fc={row['function']} len={row['length']} "
            f"command={row.get('command response', '?')} time={row.get('time', '?')}"
        )
        self.goto_spin.blockSignals(True)
        self.goto_spin.setValue(self.manual_index + 1)
        self.goto_spin.blockSignals(False)

    def _ensure_manual_ser(self) -> bool:
        # 优先用主串口句柄 (打开按钮创建的),没有则自动开
        if self.manual_ser and self.manual_ser.is_open:
            return True
        port = self._get_port()
        if not port:
            QMessageBox.warning(self, "无串口", "请先选择 COM 口")
            return False
        try:
            baud = int(self.baud_combo.currentText())
            self.manual_ser = serial.Serial(
                port=port, baudrate=baud,
                bytesize=8, parity='N', stopbits=1,
                timeout=0.1, write_timeout=0.5,
            )
            self._log(f"✓ 手动串口打开: {port} @ {baud} 8N1")
            self._update_port_status()
            return True
        except Exception as e:
            QMessageBox.critical(self, "串口错误", str(e))
            self._update_port_status(error=str(e))
            return False

    # --- 串口打开/关闭 ---
    def _open_port(self):
        """手动打开主串口 (供手动模式和监控用)。"""
        port = self._get_port()
        if not port:
            QMessageBox.warning(self, "无串口", "请先选择 COM 口")
            return
        if self.manual_ser and self.manual_ser.is_open:
            self._log("⚠ 串口已打开,无需重复")
            return
        try:
            baud = int(self.baud_combo.currentText())
            self.manual_ser = serial.Serial(
                port=port, baudrate=baud,
                bytesize=8, parity='N', stopbits=1,
                timeout=0.1, write_timeout=0.5,
            )
            self._log(f"✓ 串口已打开: {port} @ {baud} 8N1")
            self._update_port_status()
            self.open_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            # 串口开好后立刻启动 RX reader,这样手动模式发帧时也能看到 H743 的回包
            self._start_rx_reader()
        except Exception as e:
            self._log(f"❌ 打开失败: {e}")
            self._update_port_status(error=str(e))
            QMessageBox.critical(self, "串口错误", f"打开失败:\n{e}")

    def _close_port(self):
        """关闭主串口。"""
        # 先停 RX reader,避免它读已关闭的句柄
        self._stop_rx_reader()
        if self.manual_ser and self.manual_ser.is_open:
            try:
                self.manual_ser.close()
                self._log("✓ 串口已关闭")
            except Exception as e:
                self._log(f"⚠ 关闭时异常: {e}")
        self.manual_ser = None
        self._update_port_status()
        self.open_btn.setEnabled(True)
        self.close_btn.setEnabled(False)

    def _start_rx_reader(self):
        """打开串口后,启动后台 RX 线程,把收到的字节转发到 RxPanel."""
        if self._rx_reader is not None:
            return  # 已启动
        if self.manual_ser is None or not self.manual_ser.is_open:
            return
        try:
            self._rx_reader = SerialReader(self.manual_ser)
            self._rx_reader.data_received.connect(self.rx_panel.append_bytes)
            # ⭐ 同时把字节流喂给 IdsLineParser (解析 [ALARM] N 行)
            self._rx_reader.data_received.connect(self.ids_parser.feed)
            # ⭐ 2026-10-19 SITE 4 capture: 把原始字节同时转发给 raw dump sink
            self._rx_reader.data_received.connect(self._on_rx_dump_data)
            self._rx_reader.start()
            self._log("✓ RX reader 启动 (100ms 轮询)")
        except Exception as e:
            self._log(f"⚠ RX reader 启动失败: {e}")
            self._rx_reader = None

    # ⭐ 2026-10-19 SITE 4 capture: 原始字节落到文件, 无 buffer 上限.
    #   跟现有 rx_panel/ids_parser 并联 (Qt 信号多 slot 安全).
    def _toggle_rx_dump(self, checked: bool):
        if checked:
            ts = time.strftime("%Y%m%d-%H%M%S")
            os.makedirs('logs', exist_ok=True)
            path = os.path.join('logs', f'rx_raw_{ts}.log')
            try:
                f = open(path, 'wb', buffering=0)  # unbuffered, 立即落盘
            except (OSError, IOError) as e:
                self._log(f"❌ RX dump 打开失败: {e}")
                self.rx_dump_btn.setChecked(False)
                return
            self._rx_dump_file = f
            self._rx_dump_path = path
            self._rx_dump_n_bytes = 0
            self._log(f"📥 RX dump 已开启 → {path}")
        else:
            f = self._rx_dump_file
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
                n = self._rx_dump_n_bytes
                self._log(f"💾 RX dump 已关闭: {self._rx_dump_path} ({n} bytes)")
            self._rx_dump_file = None

    def _on_rx_dump_data(self, chunk: bytes):
        """Qt slot — 把 RX 字节原样写盘. 在主线程同步, 简单可靠."""
        f = self._rx_dump_file
        if f is None:
            return
        try:
            f.write(chunk)
            self._rx_dump_n_bytes += len(chunk)
        except Exception as e:
            self._log(f"⚠ RX dump write failed: {e}")
            # 自动停
            self.rx_dump_btn.setChecked(False)

    def _stop_rx_reader(self):
        """关闭串口前/窗口关闭前停 RX 线程."""
        if self._rx_reader is None:
            return
        try:
            self._rx_reader.stop()
            self._rx_reader.wait(1000)
        except Exception as e:
            self._log(f"⚠ RX reader 停止异常: {e}")
        finally:
            self._rx_reader = None

    def _update_port_status(self, error: Optional[str] = None):
        """更新状态标签的颜色 + 文字。"""
        if error:
            self.port_status_label.setText(f"● 串口错误: {error}")
            self.port_status_label.setStyleSheet(
                "padding: 4px; background: #ffe0e0; border: 1px solid #cc0000; "
                "color: #cc0000; font-weight: bold;"
            )
            return
        if self.manual_ser and self.manual_ser.is_open:
            port = self.manual_ser.port
            baud = self.manual_ser.baudrate
            self.port_status_label.setText(
                f"● 串口已打开: {port} @ {baud} 8N1"
            )
            self.port_status_label.setStyleSheet(
                "padding: 4px; background: #e0ffe0; border: 1px solid #00aa00; "
                "color: #008800; font-weight: bold;"
            )
        else:
            self.port_status_label.setText("● 串口: 未打开")
            self.port_status_label.setStyleSheet(
                "padding: 4px; background: #f4f4f4; border: 1px solid #ccc;"
            )

    def closeEvent(self, event):
        """窗口关闭时清理串口 + RX reader。"""
        self._stop_rx_reader()
        if self.manual_ser and self.manual_ser.is_open:
            try:
                self.manual_ser.close()
            except Exception:
                pass
        super().closeEvent(event)

    def _manual_send_next(self):
        if not self.manual_loaded:
            self._manual_load()
            if not self.manual_loaded:
                return
        if self.manual_index >= len(self.manual_rows):
            self._log("⚠ 已到末尾")
            return
        if not self._ensure_manual_ser():
            return
        row = self.manual_rows[self.manual_index]
        frame = build_modbus_rtu_frame(row)
        if frame is None:
            self._log(f"⚠ row {self.manual_index + 1}: 主站过滤, 无帧")
            return
        # ⭐ 第十九轮: 取 row['binary result'] 决定是否计为攻击 (binary==1 时 +1)
        try:
            binary_int = int((row.get('binary result') or '0').strip())
        except (ValueError, TypeError):
            binary_int = 0
        try:
            self.manual_ser.write(frame)
            self._record_send(len(frame), is_attack=(binary_int == 1))
            self._log(f"[MANUAL] #{self.manual_index + 1}/{len(self.manual_rows)} "
                      f"({len(frame)} B, binary={binary_int}) → {frame.hex(' ')}")
        except Exception as e:
            self._log(f"❌ 写入失败: {e}")
            return
        self.manual_index += 1
        self._update_manual_preview()

    def _manual_prev(self):
        if not self.manual_loaded:
            return
        if self.manual_index <= 0:
            self._log("⚠ 已在第一帧")
            return
        self.manual_index -= 1
        self._update_manual_preview()
        self._log(f"← 指针移到 #{self.manual_index + 1} (未发送)")

    def _manual_resend(self):
        """重发当前帧 (不前进)。"""
        if not self.manual_loaded or not self.manual_rows:
            return
        if self.manual_index >= len(self.manual_rows):
            return
        if not self._ensure_manual_ser():
            return
        row = self.manual_rows[self.manual_index]
        frame = build_modbus_rtu_frame(row)
        if frame is None:
            self._log(f"⚠ row {self.manual_index + 1}: 主站过滤, 无帧")
            return
        # ⭐ 第十九轮: binary==1 时计入攻击次数
        try:
            binary_int = int((row.get('binary result') or '0').strip())
        except (ValueError, TypeError):
            binary_int = 0
        try:
            self.manual_ser.write(frame)
            self._record_send(len(frame), is_attack=(binary_int == 1))
            self._log(f"[MANUAL RESEND] #{self.manual_index + 1} "
                      f"({len(frame)} B, binary={binary_int}) → {frame.hex(' ')}")
        except Exception as e:
            self._log(f"❌ 写入失败: {e}")

    def _manual_goto(self):
        if not self.manual_loaded:
            return
        target = self.goto_spin.value() - 1
        if 0 <= target < len(self.manual_rows):
            self.manual_index = target
            self._update_manual_preview()
            self._log(f"🎯 跳到 #{target + 1}")
        else:
            self._log(f"⚠ 越界: {target + 1}, 范围 1..{len(self.manual_rows)}")


# === 入口 ============================================================
def main():
    app = QApplication(sys.argv)
    win = RTUReplayGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
