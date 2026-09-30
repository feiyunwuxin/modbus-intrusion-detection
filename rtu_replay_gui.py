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
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QTextEdit,
    QVBoxLayout, QWidget,
)
import datetime

# --- 攻击标签常量 (与 arff 行的 binary label 对应) ----------------------
ATTACK_LABELS = ['normal', 'nmri', 'cmri', 'msci', 'mpci', 'mfci', 'dos', 'recon']
DEFAULT_CSV = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"  # ⭐ 2026-09-19 第二轮: 用 ARFF 衍生的高精度 CSV (time 字段保留小数秒), 否则 H743 算 time_diff 跟训练对不上 → 全误报


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
    # ⭐ 第二十轮 (2026-09-20): 每发一帧 emit 本帧类型 (binary==1?),
    #   让 RTUReplayGUI 通知 IdsLineParser, 用于下次 [ALARM] 时判 TP/FP.
    frame_sent_type = pyqtSignal(bool)
    # ⭐ 第二十一轮 (2026-09-20): 每发一帧 emit (n_bytes, is_attack),
    #   让 RTUReplayGUI 写 TestLogger. 之前只 emit bool, TestLogger 拿不到 n_bytes.
    frame_sent_log = pyqtSignal(int, bool)

    def __init__(self, csv_path: str, port: str, baud: int,
                 attack_filter: Optional[str], speed: float,
                 loop: bool, verbose: bool = False,
                 synth_timing: bool = True,
                 fixed_10ms: bool = False,
                 interval_ms: int = 0,
                 serial_obj: Optional['serial.Serial'] = None):
        super().__init__()
        self.csv_path = csv_path
        self.port = port
        self.baud = baud
        self.attack_filter = attack_filter  # None=all, 'normal', 'mfci', ...
        self.speed = max(0.01, float(speed)) if speed > 0 else 1.0
        self.loop = loop
        self.verbose = verbose
        self.synth_timing = synth_timing
        # ⭐ 第二十一轮 (2026-09-20): 固定 10ms 间隔模式 (诊断 DMA 解析瓶颈用)
        self.fixed_10ms = fixed_10ms
        # ⭐ 第二十二轮 (2026-09-28): 1-1000ms 任意 interval (dump 测试用, 避免 4.3KB/alarm dump
        #   阻塞时撞上下一个 alarm cycle). interval_ms > 0 覆盖 fixed_10ms.
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

            while not self._stopping:
                # 关键: 用过滤后第一行的时间当 baseline (CSV time 是 Unix 时间戳)
                baseline_t = 0.0
                if play_rows:
                    try:
                        baseline_t = float(play_rows[0].get('time', 0) or 0)
                    except (ValueError, TypeError):
                        baseline_t = 0.0
                last_t = baseline_t

                for i, row in enumerate(play_rows):
                    if self._stopping:
                        break
                    while self._paused and not self._stopping:
                        time.sleep(0.05)

                    # 时间间隔 (相对 baseline → 第一帧 dt=0, 不等)
                    # ⭐⭐ 第十八轮 (2026-09-19): 取消 synth_timing 合成时序选项.
                    #   自动回放不需模拟训练分布的 1.5s/0.1s master/slave 交替模式,
                    #   直接用 CSV `time` 列作为自然间隔, 配合 speed_combo 倍率加速.
                    # ⭐ 第二十一轮 (2026-09-20): fixed_10ms 模式忽略 CSV time,
                    #   每帧之间 time.sleep(0.010) 给 H743 DMA 256-byte 缓冲留解析时间.
                    # ⭐ 第二十二轮 (2026-09-28): interval_ms > 0 时强制固定间隔
                    #   (1-1000ms spinbox), 覆盖 fixed_10ms. 给 dump 测试用, 避免
                    #   4.3KB/alarm dump 输出阻塞时撞上下一个 alarm cycle.
                    if self.interval_ms > 0:
                        if self.frames_sent > 0:  # 第一帧不睡
                            time.sleep(self.interval_ms / 1000.0)
                    elif self.fixed_10ms:
                        if self.frames_sent > 0:  # 第一帧不睡
                            time.sleep(0.010)
                    else:
                        try:
                            t = float(row.get('time', 0) or 0)
                            dt_ms = max(0.0, (t - last_t) * 1000.0 / self.speed)
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

                    if i % 50 == 0:
                        self.progress.emit(i + 1, total, self.frames_sent,
                                            self.bytes_sent)

                if not self.loop or self._stopping:
                    break
                last_t = 0.0
                self.log.emit("--- 循环播放 ---")

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
    """Hex+ASCII 接收查看器 (清空按钮 + 格式下拉 + 字节计数)."""

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
        self.format_combo.addItems(["hex+ASCII", "hex", "ASCII"])
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
        self.text_edit.setFont(font)
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
        fmt = self.format_combo.currentText()
        if fmt == "hex":
            body = _format_hex_only(data)
        elif fmt == "ASCII":
            body = _format_ascii_only(data)
        else:
            body = _format_hex_ascii(data)
        self.text_edit.append(f"[{ts}] RX ({len(data)} B):\n{body}")
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
    alarm_with_prob = pyqtSignal(int, int, bool)  # (count, prob_x1000, is_fp)

    # ⭐ 第二十二轮: 滑动窗口大小 (H743 TCN+SE 训练时 window=16)
    _SW_SIZE = 16

    _ALARM_RE = _re.compile(rb'\[ALARM\]\s+(\d+)')
    _PROB_RE  = _re.compile(rb'prob=0\.(\d{1,3})')

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buf = bytearray()
        # ⭐ 第二十二轮 (2026-09-30): 滑动窗口 (deque maxlen=16) 跟踪最近 16 帧 binary.
        #   H743 IDS 推理基于 sliding window=16 累积特征, ALARM 反映整窗口.
        #   当 [ALARM] 上行时, 应检查最近 16 帧内是否有 attack (binary=1):
        #     - 任一帧 attack → TP (本次报警命中)
        #     - 全 normal      → FP (误报)
        #   旧逻辑只看最后一帧, attack burst 内除首帧外都误标 FP.
        self._window: "deque[bool]" = deque(maxlen=self._SW_SIZE)

    def set_last_was_attack(self, val: Optional[bool]) -> None:
        """PC GUI 每次发完一帧后调用,推送该帧 binary 到滑动窗口.

        Args:
            val: True=attack (binary==1), False=normal (binary==0),
                 None=未初始化 (清空窗口, 不再 emit TP/FP 计数)
        """
        if val is None:
            self.reset_last_was_attack()
            return
        # 兼容旧 API: val=True/False 都推入窗口
        # 旧名 set_last_was_attack 仍保留以兼容 frame_sent_type.connect(...)
        self._window.append(bool(val))

    def reset_last_was_attack(self) -> None:
        """重置 PC 滑动窗口 (测试开始前/手动清零时调用)."""
        self._window.clear()

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
        # count_changed 始终 emit (兼容旧 IdsReportPanel._on_detected)
        self.count_changed.emit(count)
        # ⭐ 第二十二轮 (2026-09-30): 滑动窗口评估 TP/FP (H743 推理基于 SW=16)
        #   - 窗口内任一帧 attack (binary=1) → TP (本次报警命中真实攻击)
        #   - 窗口内全 normal (binary=0)       → FP (本次报警是误报)
        #   - 窗口未填满 (< 16 帧)            → 不判定 (避免 ramp-up 期误分类)
        # 旧逻辑只看上一帧, attack burst 内除首帧外 15 个 alarm 全误标 FP.
        is_fp: Optional[bool] = None
        if len(self._window) >= self._SW_SIZE:
            is_fp = not any(self._window)   # True=FP (全 normal), False=TP (有 attack)
            self.fp_changed.emit(count, is_fp)
        # ⭐ 第二十一轮: 含 prob 的报警信号 (供测试日志用)
        if is_fp is not None:
            self.alarm_with_prob.emit(count, prob_x1000, is_fp)


# === IDS Report 面板 (第十七轮 双计数 → 第十九轮 4 KPI → 第二十轮 FP/TP/FPR) ==
# 显示 6 个 KPI + 2 个派生率:
#   - Total:    PC 已发送的总帧数 (binary=0 + binary=1, 即总发送)
#   - Sent:     PC 已发送的攻击帧数 (binary==1 时 +1, 即攻击次数)
#   - Normal:   PC 已发送的正常帧数 (binary==0 时 +1, 即正常数)
#   - Detected: H743 上行 [ALARM] N 的累计计数 (累计 frames_attack)
#   - TP:       真阳性 — [ALARM] 到达时最近 16 帧滑动窗口内有 attack (binary==1)
#   - FP:       假阳性 — [ALARM] 到达时最近 16 帧滑动窗口内全 normal (binary==0)
#   - Recall:   TP / Sent × 100% (召回率, H743 命中攻击的能力)
#   - FPR:      FP / Normal × 100% (误报率, 正常帧被误报的比例)
# 调试详情请看串口 raw 终端.

class IdsReportPanel(QWidget):
    """IDS 面板: 显示 6 个 KPI + 4 个派生率 (Precision/Recall/F1/Accuracy) + 1 个 FPR.

    布局:
      - 左: Total/Sent/Normal/Detected/TP/FP (6 KPI)
      - 中: Precision/Recall/F1/Accuracy (4 核心指标, 二分类论文标准)
      - 右: FPR (误报率)

    数据来源:
      - Total/Sent/Normal ← ReplayWorker.sent_stats (total, attack, normal)
      - Detected          ← IdsLineParser.count_changed (H743 上行 [ALARM] N)
      - TP / FP           ← IdsLineParser.fp_changed (最近 16 帧滑动窗口内是否有 attack)
    """

    def __init__(self, parser: IdsLineParser, parent=None):
        super().__init__(parent)
        self._parser = parser
        self._total_count = 0        # 总发送
        self._sent_count = 0         # 攻击次数 (binary==1 累计)
        self._normal_count = 0       # 正常数 (binary==0 累计)
        self._detected_count = 0     # 识别攻击次数 (H743 上行累计)
        self._tp_count = 0           # 真阳性 (alarm on attack frame)
        self._fp_count = 0           # 假阳性 (alarm on normal frame)
        self._build_ui()
        # 接 parser 信号 (H743 上行的 [ALARM] N)
        self._parser.count_changed.connect(self._on_detected)
        self._parser.fp_changed.connect(self._on_fp_changed)

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # 1. Total (总发送帧数)
        total_cap = QLabel("Total:")
        total_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.total_label = QLabel("0")
        self.total_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #333333; padding: 4px;"
        )
        outer.addWidget(total_cap)
        outer.addWidget(self.total_label)

        outer.addSpacing(16)

        # 2. Sent (攻击次数, binary==1 累计)
        sent_cap = QLabel("Sent:")
        sent_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.sent_label = QLabel("0")
        self.sent_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #0055aa; padding: 4px;"
        )
        outer.addWidget(sent_cap)
        outer.addWidget(self.sent_label)

        outer.addSpacing(16)

        # 3. Normal (正常数, binary==0 累计)
        normal_cap = QLabel("Normal:")
        normal_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.normal_label = QLabel("0")
        self.normal_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #008800; padding: 4px;"
        )
        outer.addWidget(normal_cap)
        outer.addWidget(self.normal_label)

        outer.addSpacing(16)

        # 4. Detected (识别攻击次数, H743 上行累计)
        det_cap = QLabel("Detected:")
        det_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.detected_label = QLabel("0")
        self.detected_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #cc0000; padding: 4px;"
        )
        outer.addWidget(det_cap)
        outer.addWidget(self.detected_label)

        outer.addSpacing(16)

        # 5. TP (真阳性, H743 上行 alarm 命中 attack 帧)
        tp_cap = QLabel("TP:")
        tp_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.tp_label = QLabel("0")
        self.tp_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #cc0000; padding: 4px;"
        )
        outer.addWidget(tp_cap)
        outer.addWidget(self.tp_label)

        outer.addSpacing(16)

        # 6. FP (假阳性, H743 上行 alarm 命中 normal 帧 = 误报!)
        fp_cap = QLabel("FP:")
        fp_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.fp_label = QLabel("0")
        self.fp_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #ff6600; padding: 4px;"
        )
        outer.addWidget(fp_cap)
        outer.addWidget(self.fp_label)

        outer.addSpacing(16)

        # 7. Precision (派生: TP / (TP+FP) × 100%, 预测为攻击中真攻击占比)
        prec_cap = QLabel("Precision:")
        prec_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.precision_label = QLabel("—")
        self.precision_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #aa6600; padding: 4px;"
        )
        outer.addWidget(prec_cap)
        outer.addWidget(self.precision_label)

        outer.addSpacing(16)

        # 8. Recall (派生: TP / Sent × 100%, 攻击召回率)
        rec_cap = QLabel("Recall:")
        rec_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.recall_label = QLabel("—")
        self.recall_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #aa6600; padding: 4px;"
        )
        outer.addWidget(rec_cap)
        outer.addWidget(self.recall_label)

        outer.addSpacing(16)

        # 9. F1 (派生: 2*P*R/(P+R), 精确率和召回率的调和平均 — 最核心指标)
        f1_cap = QLabel("F1:")
        f1_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.f1_label = QLabel("—")
        self.f1_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 20px; color: #cc0066; padding: 4px;"  # 稍大稍红, 最核心
        )
        outer.addWidget(f1_cap)
        outer.addWidget(self.f1_label)

        outer.addSpacing(16)

        # 10. Accuracy (派生: (TP+TN)/Total, 整体正确率)
        acc_cap = QLabel("Accuracy:")
        acc_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.accuracy_label = QLabel("—")
        self.accuracy_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #aa6600; padding: 4px;"
        )
        outer.addWidget(acc_cap)
        outer.addWidget(self.accuracy_label)

        outer.addSpacing(16)

        # 11. FPR (派生: FP / Normal × 100%, 误报率 ⭐)
        fpr_cap = QLabel("FPR:")
        fpr_cap.setStyleSheet("color: #555; font-size: 14px;")
        self.fpr_label = QLabel("—")
        self.fpr_label.setStyleSheet(
            "font-weight: bold; font-family: Consolas, monospace; "
            "font-size: 18px; color: #d4006a; padding: 4px;"
        )
        outer.addWidget(fpr_cap)
        outer.addWidget(self.fpr_label)

        outer.addStretch(1)

    def _on_sent_stats(self, total: int, attack: int, normal: int) -> None:
        """ReplayWorker.sent_stats 信号回调 (每帧一次).

        Args:
            total:  累计总发送帧数 (= attack + normal)
            attack: 累计攻击帧数 (binary==1 时 +1)
            normal: 累计正常帧数 (binary==0 时 +1)
        """
        self._total_count = total
        self._sent_count = attack
        self._normal_count = normal
        self.total_label.setText(str(total))
        self.sent_label.setText(str(attack))
        self.normal_label.setText(str(normal))
        self._update_rates()

    def _on_sent(self, count: int) -> None:
        """(遗留) ReplayWorker.attack_sent 信号回调. 与 sent_stats 共存时调用, 不冲突."""
        # 仅刷新 Sent 标签; 若 _on_sent_stats 已先到, 此处会覆盖一次相同值, 无副作用.
        self._sent_count = count
        self.sent_label.setText(str(count))
        self._update_rates()

    def _on_detected(self, count: int) -> None:
        """IdsLineParser.count_changed 信号回调 (H743 上行 [ALARM] N 累计)."""
        self._detected_count = count
        self.detected_label.setText(str(count))
        # 不在 _on_detected 里 +FP/+TP: 那由 _on_fp_changed 单独做 (避免重复)
        self._update_rates()

    def _on_fp_changed(self, count: int, is_fp: bool) -> None:
        """IdsLineParser.fp_changed 信号回调 (H743 上行 alarm 时附带 FP 标记).

        Args:
            count: H743 当前累计 frames_attack (与 _on_detected 的 count 同源)
            is_fp: True = 最近 16 帧滑动窗口内全 normal → 本次 alarm 算误报
                   False = 最近 16 帧滑动窗口内有 attack → 本次 alarm 算命中
        """
        # count 跟 _on_detected 重复,这里只更新 TP/FP 计数 + 标签
        if is_fp:
            self._fp_count += 1
        else:
            self._tp_count += 1
        self.tp_label.setText(str(self._tp_count))
        self.fp_label.setText(str(self._fp_count))
        self._update_rates()

    def _update_rates(self) -> None:
        """计算并显示 5 个派生指标: Precision / Recall / F1 / Accuracy / FPR.

        公式:
          - Precision = TP / (TP + FP)           # 预测为攻击中真攻击的占比
          - Recall    = TP / Sent                # 实际攻击被检出的占比
          - F1        = 2*P*R / (P + R)          # P 与 R 的调和平均
          - Accuracy  = (TP + TN) / Total        # 整体正确率
            其中 TN = Normal - FP (没报警的 normal 帧数)
            Total = Sent + Normal
          - FPR       = FP / Normal               # 误报率

        注:
          - Sent 计数的是 PC 已发送的攻击帧数 (binary==1 时 +1)
          - Normal 计数的是 PC 已发送的正常帧数 (binary==0 时 +1)
          - TP 是 [ALARM] 落在 attack 帧后的累计次数
          - FP 是 [ALARM] 落在 normal 帧后的累计次数
        """
        tp, fp = self._tp_count, self._fp_count
        sent, normal = self._sent_count, self._normal_count
        total = self._total_count

        # Precision
        if (tp + fp) > 0:
            prec = 100.0 * tp / (tp + fp)
            self.precision_label.setText(f"{prec:.1f}%")
        else:
            self.precision_label.setText("—")

        # Recall
        if sent > 0:
            rec = 100.0 * tp / sent
            self.recall_label.setText(f"{rec:.1f}%")
        else:
            self.recall_label.setText("—")

        # F1 = 2 * P * R / (P + R)
        if (tp + fp) > 0 and sent > 0:
            p = tp / (tp + fp)
            r = tp / sent
            if (p + r) > 0:
                f1 = 2.0 * p * r / (p + r) * 100.0
                self.f1_label.setText(f"{f1:.1f}%")
            else:
                self.f1_label.setText("—")
        else:
            self.f1_label.setText("—")

        # Accuracy = (TP + TN) / Total, 其中 TN = Normal - FP
        if total > 0:
            tn = max(0, normal - fp)  # 防御性 max, FP 不会超过 Normal
            acc = 100.0 * (tp + tn) / total
            self.accuracy_label.setText(f"{acc:.1f}%")
        else:
            self.accuracy_label.setText("—")

        # FPR
        if normal > 0:
            fpr = 100.0 * fp / normal
            self.fpr_label.setText(f"{fpr:.1f}%")
        else:
            self.fpr_label.setText("—")

    def reset_counters(self):
        """重置全部 6 个 KPI + 5 个派生率 (供测试开始前 / 用户手动清零)."""
        self._total_count = 0
        self._sent_count = 0
        self._normal_count = 0
        self._detected_count = 0
        self._tp_count = 0
        self._fp_count = 0
        self.total_label.setText("0")
        self.sent_label.setText("0")
        self.normal_label.setText("0")
        self.detected_label.setText("0")
        self.tp_label.setText("0")
        self.fp_label.setText("0")
        self.precision_label.setText("—")
        self.recall_label.setText("—")
        self.f1_label.setText("—")
        self.accuracy_label.setText("—")
        self.fpr_label.setText("—")
        # ⭐ 第二十轮: 同时清空 parser 的 _last_was_attack, 避免旧的 PC 帧类型污染新测试
        self._parser.reset_last_was_attack()


# === 测试日志 (第二十一轮 2026-09-20) ====================================
# 1000 条测试出现 TP=56/FP=185, 跟 ONNX 仿真 TP=177/FP=808 完全对不上.
# 根因分析需要更精细的 per-frame / per-alarm 事件流.
# TestLogger 把每个事件 (SEND 帧 / ALARM 上行 / KPI 快照) 写入 CSV 文件,
# 配合 logs/test_<时间戳>.csv + test_<时间戳>_summary.txt, 事后用 pandas /
# Excel / awk 分析. 不依赖 GUI 日志窗口 (后者会滚屏掉).
class TestLogger:
    """结构化测试日志 — 每个事件一行 CSV, 方便事后分析.

    写入 logs/test_<YYYYMMDD-HHMMSS>.csv, 列:
      t_event_ms | event | frame_idx | binary | n_bytes | alarm_count
      | prob_x1000 | is_fp | running_total | running_attack | running_normal
      | running_tp | running_fp | gap_after_send_ms | note

    同时写 .summary.txt (KPI 快照, 每次 close() 时更新).
    """
    HEADER = [
        't_event_ms', 'event', 'frame_idx', 'binary', 'n_bytes',
        'alarm_count', 'prob_x1000', 'is_fp',
        'running_total', 'running_attack', 'running_normal',
        'running_tp', 'running_fp', 'gap_after_send_ms', 'note',
    ]

    def __init__(self):
        self._file = None
        self._writer = None
        self._path = None
        self._t0_ms: Optional[int] = None  # 测试开始 unix ms
        self._last_send_ms: Optional[int] = None  # 上一次 SEND 的 unix ms (算 gap)
        # KPI 跟踪
        self.running_total = 0
        self.running_attack = 0
        self.running_normal = 0
        self.running_tp = 0
        self.running_fp = 0

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
        self.running_total += 1
        if binary == 1:
            self.running_attack += 1
        else:
            self.running_normal += 1
        self._writer.writerow([
            t_event, 'SEND', frame_idx, binary, n_bytes,
            '', '', '',
            self.running_total, self.running_attack, self.running_normal,
            self.running_tp, self.running_fp, gap, note,
        ])
        self._file.flush()
        self._last_send_ms = now_ms

    def log_alarm(self, alarm_count: int, prob_x1000: Optional[int],
                  is_fp: Optional[bool], note: str = "") -> None:
        """记录一次 H743 上行 [ALARM] 事件 (含 TP/FP 分类)."""
        if not self.is_open:
            return
        import time as _t
        now_ms = int(_t.time() * 1000)
        t_event = now_ms - (self._t0_ms or now_ms)
        gap = '' if self._last_send_ms is None else (now_ms - self._last_send_ms)
        if is_fp is True:
            self.running_fp += 1
        elif is_fp is False:
            self.running_tp += 1
        self._writer.writerow([
            t_event, 'ALARM', '', '', '',
            alarm_count, prob_x1000 if prob_x1000 is not None else '',
            ('FP' if is_fp is True else ('TP' if is_fp is False else '')),
            self.running_total, self.running_attack, self.running_normal,
            self.running_tp, self.running_fp, gap, note,
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
            self.running_total, self.running_attack, self.running_normal,
            self.running_tp, self.running_fp, '', note,
        ])
        self._file.flush()

    def write_summary(self) -> None:
        """关闭时写 KPI 摘要到 <path>.summary.txt."""
        if not self.is_open or not self._path:
            return
        summary_path = self._path + '.summary.txt'
        sent = self.running_attack
        normal = self.running_normal
        recall = self.running_tp / sent if sent else 0
        fpr = self.running_fp / normal if normal else 0
        precision = self.running_tp / (self.running_tp + self.running_fp) if (self.running_tp + self.running_fp) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        # Accuracy = (TP + TN) / Total, 其中 TN = Normal - FP
        tn = max(0, self.running_normal - self.running_fp)
        accuracy = (self.running_tp + tn) / self.running_total if self.running_total else 0
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f'Test Summary\n')
                f.write(f'=' * 60 + '\n')
                f.write(f'Log file:        {self._path}\n')
                f.write(f'Total frames:    {self.running_total}\n')
                f.write(f'  Attack (Sent): {self.running_attack}\n')
                f.write(f'  Normal:        {self.running_normal}\n')
                f.write(f'TP:              {self.running_tp}\n')
                f.write(f'FP:              {self.running_fp}\n')
                f.write(f'TN:              {tn}  (= Normal - FP)\n')
                f.write(f'FN:              {self.running_attack - self.running_tp}  (= Attack - TP)\n')
                f.write(f'Recall (TP/Sent):{recall:.4f}  ({recall*100:.2f}%)\n')
                f.write(f'FPR    (FP/Norm):{fpr:.4f}  ({fpr*100:.2f}%)\n')
                f.write(f'Precision:       {precision:.4f}  ({precision*100:.2f}%)\n')
                f.write(f'F1:              {f1:.4f}  ({f1*100:.2f}%)\n')
                f.write(f'Accuracy:        {accuracy:.4f}  ({accuracy*100:.2f}%)\n')
        except Exception:
            pass

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
        # ⭐ 第二十一轮 (2026-09-20): 结构化测试日志 (CSV + summary)
        self.test_logger = TestLogger()
        self._build_ui()
        self._wire()
        self._refresh_ports()

    # --- UI ---
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 串口设置
        port_box = QGroupBox("串口设置")
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

        root.addWidget(port_box)

        # CSV
        csv_box = QGroupBox("数据集 (IanArffDataset.csv)")
        csv_layout = QHBoxLayout(csv_box)
        csv_layout.addWidget(QLabel("路径:"))
        self.csv_edit = QLineEdit(DEFAULT_CSV)
        csv_layout.addWidget(self.csv_edit, stretch=1)
        self.csv_btn = QPushButton("📂 浏览...")
        csv_layout.addWidget(self.csv_btn)
        root.addWidget(csv_box)

        # 回放选项
        opt_box = QGroupBox("回放选项")
        opt = QGridLayout(opt_box)
        opt.addWidget(QLabel("速度 (倍率):"), 0, 0)
        self.speed_combo = QComboBox()
        # ⭐ 第二十一轮: 加 "10ms/帧" 固定间隔选项, 给 DMA 256-byte 解析瓶颈测试用
        # ⭐ 第二十二轮: 加 "自定义 ms/帧" 选项, 让 interval spinbox 生效
        self.speed_combo.addItems(['0.1x', '0.5x', '1x', '2x', '5x', '10x', '50x', '100x', 'Max',
                                       '10ms/帧 (固定)', '自定义 (ms/帧, 见右侧)'])
        self.speed_combo.setCurrentText('10x')  # 默认 10x,1x 跑 76h 太慢
        opt.addWidget(self.speed_combo, 0, 1)
        opt.addWidget(QLabel("攻击过滤:"), 1, 0)
        self.attack_combo = QComboBox()
        self.attack_combo.addItems(['all (全部)'] + ATTACK_LABELS)
        opt.addWidget(self.attack_combo, 1, 1)
        # ⭐ 第二十三轮 (2026-09-29): 重命名为"自动发送间隔", 让用户清楚这是固定间隔模式.
        #   0   = 禁用, 走 speed_combo (CSV time × 倍率) 或 "10ms/帧 (固定)"
        #   1-1000 = 自动发送模式: 每帧之间 sleep N ms, 忽略 CSV time 列
        #   默认 100ms 给 H743 sniffer 测试用 (≈10 帧/秒, 既不丢帧又能实时观察 KPI)
        opt.addWidget(QLabel("自动发送间隔 (ms):"), 2, 0)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(0, 1000)
        self.interval_spin.setValue(100)  # 默认 100ms = 10 帧/秒
        self.interval_spin.setSuffix(" ms/帧 (0=禁用)")
        self.interval_spin.setToolTip(
            "🟢 自动发送间隔 (覆盖 CSV time 列):\n"
            "  • 0   = 禁用, 走速度倍率 或 10ms/帧\n"
            "  • 1-1000 = 自动按 N 毫秒间隔发送每帧\n"
            "• 用途: 实时观察 IDS 面板 KPI, 避免发送过快掩盖 H743 漏报\n"
            "• 推荐值: 100ms (10帧/秒) 观察, 10ms 压测, 500ms dump 测试"
        )
        opt.addWidget(self.interval_spin, 2, 1)
        self.loop_chk = QCheckBox("循环播放")
        self.verbose_chk = QCheckBox("详细 TX 日志 (会刷屏)")
        opt.addWidget(self.loop_chk, 3, 0)
        opt.addWidget(self.verbose_chk, 3, 1)
        # 2026-09-19: 合成 1.5s/0.1s 时间模式 (匹配 SCADA 训练分布, 见 rtu_replay_gui.py 行 296 注释)
        self.synth_timing_chk = QCheckBox("合成时序 (1.5s poll / 0.1s resp, 训练分布)")
        self.synth_timing_chk.setChecked(True)  # 默认开,因为 CSV time 列是 1s 精度整数,与训练不符
        opt.addWidget(self.synth_timing_chk, 4, 0, 1, 2)
        root.addWidget(opt_box)

        # === 手动模式 (单帧发送) ===
        man_box = QGroupBox("手动模式 (单帧发送 — 不走时间间隔)")
        man = QVBoxLayout(man_box)
        row1 = QHBoxLayout()
        self.manual_enable_chk = QCheckBox("启用手动模式")
        row1.addWidget(self.manual_enable_chk)
        row1.addStretch(1)
        self.load_btn = QPushButton("📥 加载 (应用过滤)")
        row1.addWidget(self.load_btn)
        man.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("当前位置:"))
        self.pos_label = QLabel("— / —")
        self.pos_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        row2.addWidget(self.pos_label)
        row2.addStretch(1)
        self.cat_label = QLabel("类别: —")
        self.cat_label.setStyleSheet("font-weight: bold;")
        row2.addWidget(self.cat_label)
        man.addLayout(row2)

        self.preview_label = QLabel("帧预览: (加载后显示)")
        self.preview_label.setStyleSheet(
            "font-family: Consolas, monospace; padding: 6px; "
            "background: #f4f4f4; border: 1px solid #ccc;"
        )
        self.preview_label.setWordWrap(True)
        # 固定高度:布局不再随 hex 内容长度伸缩,长帧内部换行即可
        # (96px ≈ 4 行 hex + metadata 行,够用又紧凑)
        self.preview_label.setMinimumHeight(96)
        self.preview_label.setMaximumHeight(96)
        man.addWidget(self.preview_label)

        row3 = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一帧")
        self.send_next_btn = QPushButton("📤 发送并前进")
        self.send_next_btn.setStyleSheet(
            "font-weight: bold; background: #4CAF50; color: white; padding: 8px;"
        )
        self.resend_btn = QPushButton("🔁 重发当前帧")
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(1, 274628)
        self.goto_spin.setPrefix("跳到: ")
        self.goto_btn = QPushButton("🎯 跳转")
        for b in (self.prev_btn, self.send_next_btn, self.resend_btn,
                  self.goto_spin, self.goto_btn):
            row3.addWidget(b)
        self.prev_btn.setEnabled(False)
        self.send_next_btn.setEnabled(False)
        self.resend_btn.setEnabled(False)
        self.goto_btn.setEnabled(False)
        man.addLayout(row3)

        # 第 4 行: 发送区管理 (清空加载 / 重置指针 / 重新加载)
        row4 = QHBoxLayout()
        self.reset_ptr_btn = QPushButton("⏮ 回到首帧")
        self.reset_ptr_btn.setToolTip("手动指针回到 #1 (不丢已加载行)")
        self.reload_btn = QPushButton("🔄 重新加载")
        self.reload_btn.setToolTip("按当前过滤重新读 CSV")
        self.unload_btn = QPushButton("🗑 清空加载")
        self.unload_btn.setToolTip("丢弃已加载的所有行,关闭手动模式")
        for b in (self.reset_ptr_btn, self.reload_btn, self.unload_btn):
            b.setEnabled(False)
            row4.addWidget(b)
        row4.addStretch(1)
        man.addLayout(row4)

        # ⭐ 发送区 (man_box) 与 接收区 (rx_box) 同一行, 左右分布.
        # 创建横向容器, 稍后把 rx_box 也加进来.
        self.tx_rx_row = QHBoxLayout()
        self.tx_rx_row.setContentsMargins(0, 0, 0, 0)
        self.tx_rx_row.setSpacing(6)
        man_box.setMinimumWidth(380)
        self.tx_rx_row.addWidget(man_box, stretch=2)
        # rx_box 在下面创建后再 addWidget 进这里

        # 控制 (自动回放)
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("▶ 自动开始")
        self.pause_btn = QPushButton("⏸ 暂停")
        self.stop_btn = QPushButton("⏹ 停止")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        for b in (self.start_btn, self.pause_btn, self.stop_btn):
            b.setMinimumHeight(36)
            ctrl.addWidget(b)
        root.addLayout(ctrl)

        # 进度 + 统计
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)
        stats_row = QHBoxLayout()
        self.stats_label = QLabel("已发送: 0 帧 / 0 字节 / 平均 0 B/帧   状态: 空闲")
        self.stats_label.setStyleSheet("font-weight: bold;")
        stats_row.addWidget(self.stats_label, stretch=1)
        self.dist_btn = QPushButton("📊 帧大小分布")
        stats_row.addWidget(self.dist_btn)
        root.addLayout(stats_row)

        # 日志 (顶部带清空按钮的工具栏)
        log_box = QGroupBox("📋 日志")
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
        log_toolbar.addStretch(1)
        log_layout.addLayout(log_toolbar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        log_layout.addWidget(self.log_text, stretch=1)
        root.addWidget(log_box, stretch=1)

        # 接收区 (RX) — 显示从串口收到的字节 (H743 主站轮询请求等)
        rx_box = QGroupBox("📥 接收区 (RX)")
        rx_layout = QVBoxLayout(rx_box)
        self.rx_panel = RxPanel()
        rx_layout.addWidget(self.rx_panel)
        # 接收区高度 = 总高度的 ~1/4,不让它喧宾夺主
        rx_box.setMaximumHeight(220)
        # ⭐ 加入左侧发送区同一行 (左右分布), 不用 root.addWidget
        self.tx_rx_row.addWidget(rx_box, stretch=3)
        root.addLayout(self.tx_rx_row)

        # IDS Report 面板 — 解析 H743 上行的 [ALARM] N printf (第十六轮)
        self.ids_parser = IdsLineParser()
        self.ids_report = IdsReportPanel(self.ids_parser)
        root.addWidget(self.ids_report)

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
        self.manual_enable_chk.stateChanged.connect(self._on_manual_toggle)
        # 发送区管理按钮
        self.reset_ptr_btn.clicked.connect(self._manual_reset_ptr)
        self.reload_btn.clicked.connect(self._manual_load)  # 等同于按 加载
        self.unload_btn.clicked.connect(self._manual_unload)
        # 统计
        self.dist_btn.clicked.connect(self._show_size_distribution)
        # ⭐ 第二十一轮 (2026-09-20): 测试日志按钮
        self.log_open_btn.clicked.connect(self._open_test_log)
        self.log_close_btn.clicked.connect(self._close_test_log)
        # ALARM 事件钩子: IdsLineParser.alarm_with_prob → TestLogger.log_alarm
        # (用含 prob 的信号, 而不是 fp_changed, 这样日志能记录 prob_x1000)
        self.ids_parser.alarm_with_prob.connect(self._on_alarm_for_log)

    def _refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("(无可用串口)", None)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV", str(Path(self.csv_edit.text()).parent),
            "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            self.csv_edit.setText(path)

    def _get_port(self) -> Optional[str]:
        return self.port_combo.currentData()

    def _get_speed(self) -> float:
        """返回 speed 倍率.
        'Max'  -> 1000.0 (实际效果 ≈ 不 sleep)
        '10ms/帧 (固定)' -> 0.0 (特殊: worker 走 fixed_10ms 分支, 忽略 CSV time)
        其余 (e.g. '10x') -> 10.0
        """
        t = self.speed_combo.currentText()
        if t.startswith('10ms'):
            return 0.0  # 0.0 = 触发 fixed 10ms 分支
        t = t.replace('x', '')
        if t == 'Max':
            return 1000.0
        try:
            return float(t)
        except ValueError:
            return 1.0

    def _is_fixed_10ms(self) -> bool:
        """是否选中 '10ms/帧 (固定)' 模式."""
        return self.speed_combo.currentText().startswith('10ms')

    def _get_attack_filter(self) -> Optional[str]:
        t = self.attack_combo.currentText().split(' ')[0]
        return None if t == 'all' else t

    # --- 控制 ---
    def _start(self):
        if self.worker and self.worker.isRunning():
            return
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

        # 重置统计
        self.total_frames = 0
        self.total_bytes = 0
        self.frame_sizes = []
        self.start_time = time.time()
        # ⭐ 第二十一轮: 自动开启测试日志 (用 'auto-{speed}x' 后缀)
        # ⭐ 第二十二轮: 加 interval_ms 后缀 (用于 dump 测试)
        interval_ms = self.interval_spin.value()
        if not self.test_logger.is_open:
            if interval_ms > 0:
                label = f'auto-{interval_ms}ms'
            elif self._is_fixed_10ms():
                label = 'auto-fixed-10ms'
            else:
                label = f'auto-{self._get_speed()}x'
            path = self.test_logger.open(label=label)
            self.test_logger.log_event('START', note=f'speed={self.speed_combo.currentText()} '
                                                     f'interval_ms={interval_ms} '
                                                     f'filter={self._get_attack_filter() or "all"} '
                                                     f'loop={self.loop_chk.isChecked()}')
            self.log_open_btn.setEnabled(False)
            self.log_close_btn.setEnabled(True)
            self._log(f"📂 测试日志已自动开启: {path}")

        self.worker = ReplayWorker(
            csv_path=csv_path, port=port, baud=baud,
            attack_filter=self._get_attack_filter(),
            speed=self._get_speed(),
            loop=self.loop_chk.isChecked(),
            verbose=self.verbose_chk.isChecked(),
            synth_timing=self.synth_timing_chk.isChecked(),
            fixed_10ms=self._is_fixed_10ms(),
            interval_ms=interval_ms,  # ⭐ 第二十二轮: 1-1000ms 自定义间隔 (dump 测试用)
            serial_obj=self.manual_ser,  # ⭐ 第二十一轮: 共享 manual_ser, worker 不重开
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        # ⭐⭐ 第十七轮: 攻击帧发送计数 → IdsReportPanel "Sent" KPI
        self.worker.attack_sent.connect(self.ids_report._on_sent)
        # ⭐⭐ 第十九轮: 3 元组 (总发送, 攻击次数, 正常数) → IdsReportPanel
        self.worker.sent_stats.connect(self.ids_report._on_sent_stats)
        # ⭐⭐ 第二十轮: 每帧类型 → IdsLineParser (为下次 [ALARM] 准备 TP/FP 判定)
        self.worker.frame_sent_type.connect(self.ids_parser.set_last_was_attack)
        # ⭐⭐ 第二十一轮: 每帧 (n_bytes, is_attack) → TestLogger
        #   这是关键修复: 之前 TestLogger 只接到手动 _record_send, 自动回放完全没记录
        #   (1073 帧自动回放后 CSV 仍只有 START+FINISH 两行, 0 SEND 事件)
        self.worker.frame_sent_log.connect(self._record_send)

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        self.worker.start()
        interval_str = f"  interval={interval_ms}ms/帧" if interval_ms > 0 else ""
        self._log(f"▶ 开始: {port} @ {baud}  speed={self._get_speed()}x"
                  f"{interval_str}  filter={self._get_attack_filter() or 'all'}")

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
        # ⭐ 第二十一轮: 自动回放结束时关闭测试日志
        if self.test_logger.is_open:
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
            self.test_logger.log_event('ERROR', note=msg[:200])
            self.test_logger.close()
            self.log_open_btn.setEnabled(True)
            self.log_close_btn.setEnabled(False)
            self._log(f"💾 测试日志已保存: {self.test_logger.path}")
            self.test_logger = TestLogger()

    # --- ⭐ 第二十一轮: 结构化测试日志 (CSV + summary.txt) ---
    def _open_test_log(self):
        """用户手动开启 CSV 测试日志. 若 worker 在跑则用 'auto-{speed}x' 后缀."""
        if self.test_logger.is_open:
            QMessageBox.information(self, "已开启", f"日志已在: {self.test_logger.path}")
            return
        label = ''
        if self.worker and self.worker.isRunning():
            label = f'auto-{self._get_speed()}x'
        elif self.manual_loaded:
            label = f'manual-{len(self.manual_rows)}rows'
        path = self.test_logger.open(label=label)
        self.test_logger.log_event('OPEN', note=f'label={label}')
        self.log_open_btn.setEnabled(False)
        self.log_close_btn.setEnabled(True)
        self._log(f"📂 测试日志已开启: {path}")
        # 同步 KPI baseline 到 logger
        self.test_logger.running_total = self.total_frames
        self.test_logger.running_attack = (
            self.manual_attack_count if self.worker is None
            else int(self.worker.attack_count)
        )
        self.test_logger.running_normal = (
            self.manual_normal_count if self.worker is None
            else int(self.worker.normal_count)
        )
        self.test_logger.running_tp = self.ids_report._tp_count
        self.test_logger.running_fp = self.ids_report._fp_count

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

    def _on_alarm_for_log(self, count: int, prob_x1000: int, is_fp: bool):
        """IdsLineParser.alarm_with_prob 信号回调 — 写一行 ALARM 事件到 CSV.

        Args:
            count: H743 当前累计 frames_attack
            prob_x1000: H743 printf 输出的 prob=0.NNN × 1000 (范围 [0, 1000],
                       -1 表示 H743 没输出 prob 字段, 旧格式)
            is_fp: True=误报 (上一帧 normal), False=命中 (上一帧 attack)
        """
        if not self.test_logger.is_open:
            return
        self.test_logger.log_alarm(
            alarm_count=count,
            prob_x1000=prob_x1000 if prob_x1000 >= 0 else None,
            is_fp=is_fp,
            note=f'prob={prob_x1000/1000:.3f}' if prob_x1000 >= 0 else 'no_prob',
        )

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
        # ⭐ 第二十轮: 告诉 IdsLineParser 最近一帧是不是 attack (用于 FP 判定)
        self.ids_parser.set_last_was_attack(is_attack)
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

    def _on_manual_toggle(self, state: int):
        enabled = bool(state)
        # 启用手动时,禁用自动按钮 (避免冲突)
        self.start_btn.setEnabled(not enabled)
        if enabled and not self.manual_loaded:
            self._log("手动模式已启用,点 [📥 加载] 应用过滤")

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
            self._rx_reader.start()
            self._log("✓ RX reader 启动 (100ms 轮询)")
        except Exception as e:
            self._log(f"⚠ RX reader 启动失败: {e}")
            self._rx_reader = None

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
