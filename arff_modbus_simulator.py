#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arff_modbus_simulator.py
========================
基于 IanArffDataset.arff 的 Modbus 通信模拟器 — 真实数据回放

设计目标:
  1. 从 IanArffDataset.arff 流式读取 274k 条 SCADA 记录
  2. 把每条记录翻译成 Modbus TCP 帧,作为真实 Modbus 流量输出
  3. 支持 --rate Nx 调节回放节奏 (1x=实时, 100x=压缩, 0=burst)
  4. 100% 自实现 Modbus TCP 协议 (绕过 pymodbus 3.x 弃用陷阱)
  5. 与 modbus_scada_server.py 协议级兼容 (FC=3/6/16, 同一寄存器布局)

寄存器布局 (Modbus 地址 0-25, 与 IanArffDataset 17 维对齐):
  Holding Registers (FC=3/6/16):
    0  setpoint         (实数 × 1)
    1  gain             (实数 × 1)
    2  reset_rate_x100  (实数 × 100)
    3  deadband_x100    (实数 × 100)
    4  cycle_time       (实数 × 1)
    5  rate_x100        (实数 × 100)
    6  system_mode      (0/1/2)
    7  control_scheme   (1/2)
    8  pump             (0/1)
    9  solenoid         (0/1)
    20 pressure_x100    (实数 × 100)
    21 crc_rate         (整数)
    22 cmd_response     (0/1)
    23 flow_x100        (保留)
    24 temperature_x10  (保留)

用法:
  # 1. 启动回放服务器 (默认 100x 压缩)
  python arff_modbus_simulator.py start --port 5020 --rate 100x

  # 2. 客户端测试 — 读取 8 个寄存器
  python arff_modbus_simulator.py client --port 5020 --read 8

  # 3. 把 ARFF 数据导出为 JSONL (无需启动服务器)
  python arff_modbus_simulator.py dump --n 100 --out scada_replay.jsonl

  # 4. 验证: 把 ARFF 数据转回 Modbus 帧 (offline)
  python arff_modbus_simulator.py dump-frames --n 10

Author: SCADA IDS Team
Date: 2026-09-06
"""

import os
import sys
import csv
import time
import struct
import socket
import argparse
import logging
import threading
from typing import Optional, Dict, List, Tuple, Iterator
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arff_modbus")

# ───────────────────────────────────────────────────────────────────
# Modbus 协议常量 (与 modbus_scada_server.py 一致)
# ───────────────────────────────────────────────────────────────────
MBAP_HEADER_LEN = 7
MAX_REGISTERS = 65536

# Function codes
FC_READ_HOLDING = 0x03
FC_WRITE_SINGLE = 0x06
FC_WRITE_MULTI = 0x10

# 异常码
EXC_ILLEGAL_FC = 0x01
EXC_ILLEGAL_ADDR = 0x02
EXC_ILLEGAL_VALUE = 0x03

# ARFF 中的非法 FC (MFCI / Recon 攻击)
ILLEGAL_FCS = {43, 73, 90, 100, 136, 137, 138, 139, 140, 171, 172}

# ───────────────────────────────────────────────────────────────────
# 寄存器映射
# ───────────────────────────────────────────────────────────────────
# ARFF 字段索引 (与 IanArffDataset.csv 表头顺序一致)
ARFF_COLS = [
    "address", "function", "length", "setpoint", "gain", "reset rate",
    "deadband", "cycle time", "rate", "system mode", "control scheme",
    "pump", "solenoid", "pressure measurement", "crc rate",
    "command response", "time", "binary result", "categorized result",
    "specific result",
]
COL = {name: i for i, name in enumerate(ARFF_COLS)}

# Modbus Holding Register 地址分配
REG = {
    "setpoint":         0,
    "gain":             1,
    "reset_rate_x100":  2,
    "deadband_x100":    3,
    "cycle_time":       4,
    "rate_x100":        5,
    "system_mode":      6,
    "control_scheme":   7,
    "pump":             8,
    "solenoid":         9,
    "pressure_x100":    20,
    "crc_rate":         21,
    "cmd_response":     22,
    "flow_x100":        23,
    "temperature_x10":  24,
}
NUM_REGS = 25

# ARFF 行 → 寄存器值列表 (按 REG 顺序填,缺失/NaN 填 0)
def _safe_float(v: str, scale: float = 1.0, default: int = 0) -> int:
    """把 ARFF 字符串转成 Modbus 寄存器值 (16-bit unsigned)."""
    if v in ("", "?", "nan", None):
        return default
    try:
        f = float(v)
        return int(round(f * scale)) & 0xFFFF
    except (ValueError, TypeError):
        return default


def arff_row_to_regs(row: List[str]) -> List[int]:
    """ARFF 一行 (20 列) → 16 个 Holding Register 值 (NUM_REGS 长)."""
    regs = [0] * NUM_REGS
    regs[REG["setpoint"]]        = _safe_float(row[COL["setpoint"]])
    regs[REG["gain"]]            = _safe_float(row[COL["gain"]])
    regs[REG["reset_rate_x100"]] = _safe_float(row[COL["reset rate"]], scale=100.0)
    regs[REG["deadband_x100"]]   = _safe_float(row[COL["deadband"]], scale=100.0)
    regs[REG["cycle_time"]]      = _safe_float(row[COL["cycle time"]])
    regs[REG["rate_x100"]]       = _safe_float(row[COL["rate"]], scale=100.0)
    regs[REG["system_mode"]]     = _safe_float(row[COL["system mode"]])
    regs[REG["control_scheme"]]  = _safe_float(row[COL["control scheme"]])
    regs[REG["pump"]]            = _safe_float(row[COL["pump"]])
    regs[REG["solenoid"]]        = _safe_float(row[COL["solenoid"]])
    regs[REG["pressure_x100"]]   = _safe_float(row[COL["pressure measurement"]], scale=100.0)
    regs[REG["crc_rate"]]        = _safe_float(row[COL["crc rate"]])
    regs[REG["cmd_response"]]    = _safe_float(row[COL["command response"]])
    return regs


# ───────────────────────────────────────────────────────────────────
# ARFF 流式读取器 (274k 行,内存 < 1MB)
# ───────────────────────────────────────────────────────────────────
class ARFFReplayReader:
    """流式读取 IanArffDataset.csv 或 .arff 文件.

    支持:
      - .csv (首选,无 @attribute 头)
      - .arff (自动跳过 @attribute 直到 @data)
    """

    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop  # 读到末尾后是否回到开头
        self._row_count = 0
        self._time_start = None
        self._time_end = None
        self._attack_dist: Dict[int, int] = {}

    def __iter__(self) -> Iterator[List[str]]:
        while True:
            n = 0
            for row in self._iter_rows():
                yield row
                n += 1
            log.info(f"ARFF replay finished: {n} rows streamed")
            if not self.loop:
                return

    def _iter_rows(self) -> Iterator[List[str]]:
        """生成一行 CSV 数据 (20 列)."""
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".arff":
            yield from self._iter_arff()
        else:
            yield from self._iter_csv()

    def _iter_csv(self) -> Iterator[List[str]]:
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 20:
                    yield row

    def _iter_arff(self) -> Iterator[List[str]]:
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            in_data = False
            for raw in f:
                line = raw.strip()
                if not in_data:
                    if line.lower().startswith("@data"):
                        in_data = True
                    continue
                # CSV 格式的 @data 段
                if not line:
                    continue
                # ARFF 数据行是逗号分隔
                row = next(iter(csv.reader([line])), None)
                if row and len(row) >= 20:
                    yield row


# ───────────────────────────────────────────────────────────────────
# Modbus TCP 协议处理 (与 modbus_scada_server.py 同结构)
# ───────────────────────────────────────────────────────────────────
@dataclass
class ARFFContext:
    """当前 ARFF 行状态 — 服务器用它来构造响应."""
    row: List[str] = field(default_factory=list)
    row_idx: int = 0
    regs: List[int] = field(default_factory=list)
    attack_type: int = 0    # 0=Normal, 1-7 = NMRI/CMRI/MSCI/MPCI/MFCI/DoS/Recon
    attack_name: str = "Normal"
    time: float = 0.0       # ARFF 时间戳
    is_response: bool = False  # command_response 字段 (1=request, 0=response)
    function: int = 3       # ARFF function 字段
    length: int = 0
    is_illegal_fc: bool = False


ATTACK_NAMES = {
    0: "Normal", 1: "NMRI", 2: "CMRI", 3: "MSCI", 4: "MPCI",
    5: "MFCI", 6: "DoS", 7: "Recon",
}


class ModbusRequestHandler:
    """处理 MBAP + PDU 请求,返回基于 ARFF 数据的响应."""

    def __init__(self, unit_id: int = 4):
        self.unit_id = unit_id

    def handle(self, data: bytes, ctx: ARFFContext) -> bytes:
        if len(data) < MBAP_HEADER_LEN + 1:
            return self._exception(0, EXC_ILLEGAL_VALUE)

        tx_id, proto_id, length, _uid = struct.unpack(">HHHB", data[:MBAP_HEADER_LEN])
        if proto_id != 0:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        pdu = data[MBAP_HEADER_LEN: MBAP_HEADER_LEN + length - 1]
        if len(pdu) < 1:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        fc = pdu[0]
        pdu_data = pdu[1:]

        # 非法 FC 直接返回异常 (匹配 ARFF 中的 MFCI/Recon 攻击)
        if fc in ILLEGAL_FCS:
            log.debug(f"  Illegal FC={fc} (MFCI/Recon) → exception")
            return self._exception(tx_id, EXC_ILLEGAL_FC)

        if fc == FC_READ_HOLDING:
            return self._handle_fc3(tx_id, pdu_data, ctx)
        elif fc == FC_WRITE_SINGLE:
            return self._handle_fc6(tx_id, pdu_data, ctx)
        elif fc == FC_WRITE_MULTI:
            return self._handle_fc16(tx_id, pdu_data, ctx)
        else:
            return self._exception(tx_id, EXC_ILLEGAL_FC)

    def _handle_fc3(self, tx_id: int, data: bytes, ctx: ARFFContext) -> bytes:
        """FC=3: Read Holding Registers — 返回 ARFF 当前行的寄存器值."""
        if len(data) < 4:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        start_addr, count = struct.unpack(">HH", data[:4])
        if count < 1 or count > 125:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)
        if start_addr + count > len(ctx.regs):
            return self._exception(tx_id, EXC_ILLEGAL_ADDR)

        values = ctx.regs[start_addr: start_addr + count]
        byte_count = count * 2
        pdu = struct.pack(">BB", FC_READ_HOLDING, byte_count)
        for v in values:
            pdu += struct.pack(">H", v & 0xFFFF)

        log.debug(f"  FC3 read: addr={start_addr} count={count} → "
                  f"attack={ctx.attack_name} t={ctx.time:.0f}")
        return self._build_response(tx_id, pdu)

    def _handle_fc6(self, tx_id: int, data: bytes, ctx: ARFFContext) -> bytes:
        """FC=6: Write Single Register — ARFF 回放模式下不修改状态."""
        if len(data) < 4:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)
        addr, value = struct.unpack(">HH", data[:4])
        if addr >= len(ctx.regs):
            return self._exception(tx_id, EXC_ILLEGAL_ADDR)
        log.debug(f"  FC6 write (ignored): addr={addr} value={value}")
        pdu = struct.pack(">BHH", FC_WRITE_SINGLE, addr, value)
        return self._build_response(tx_id, pdu)

    def _handle_fc16(self, tx_id: int, data: bytes, ctx: ARFFContext) -> bytes:
        """FC=16: Write Multiple Registers — ARFF 回放模式下不修改状态."""
        if len(data) < 4:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)
        start_addr, count = struct.unpack(">HH", data[:4])
        if count < 1 or count > 123:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)
        if start_addr + count > len(ctx.regs):
            return self._exception(tx_id, EXC_ILLEGAL_ADDR)
        log.debug(f"  FC16 write (ignored): start={start_addr} count={count}")
        pdu = struct.pack(">BHH", FC_WRITE_MULTI, start_addr, count)
        return self._build_response(tx_id, pdu)

    def _build_response(self, tx_id: int, pdu: bytes) -> bytes:
        length = len(pdu) + 1
        mbap = struct.pack(">HHHB", tx_id, 0, length, self.unit_id)
        return mbap + pdu

    def _exception(self, tx_id: int, exc_code: int) -> bytes:
        pdu = struct.pack(">BB", 0x80 | FC_READ_HOLDING, exc_code)
        return self._build_response(tx_id, pdu)


# ───────────────────────────────────────────────────────────────────
# 回放服务器
# ───────────────────────────────────────────────────────────────────
class ARFFReplayServer:
    """Modbus TCP 服务器 — 从 ARFF 流式回放真实 SCADA 数据.

    行为:
      - 每收到一个客户端请求 → 推进一行 ARFF (row_idx++)
      - 当前 ARFF 行作为响应寄存器值
      - 攻击标签从 ARFF categorized result 字段提取
      - --rate 控制节奏: rate=Nx → 两次请求之间 sleep(real_dt / N)
                      rate=0  → burst 模式,尽快返回
    """

    def __init__(self, arff_path: str, host: str = "0.0.0.0",
                 port: int = 5020, unit_id: int = 4,
                 rate: str = "100x", loop: bool = True):
        self.arff_path = arff_path
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.rate = self._parse_rate(rate)
        self.loop = loop
        self.reader = ARFFReplayReader(arff_path, loop=loop)
        self._row_iter = iter(self.reader)
        self.ctx = ARFFContext()
        self._handler = ModbusRequestHandler(unit_id=unit_id)
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._last_row_time: Optional[float] = None
        self._stats = {"req": 0, "fc3": 0, "fc6": 0, "fc16": 0, "exceptions": 0,
                       "attacks": {i: 0 for i in range(8)}}

    @staticmethod
    def _parse_rate(s: str) -> Tuple[float, bool]:
        """解析 'Nx' 或 '0' (burst). 返回 (rate_multiplier, is_burst)."""
        s = s.strip().lower()
        if s == "0" or s == "burst":
            return (0.0, True)
        if s.endswith("x"):
            try:
                return (float(s[:-1]), False)
            except ValueError:
                pass
        log.warning(f"无法解析 rate '{s}', 用默认 100x")
        return (100.0, False)

    def _advance_row(self) -> bool:
        """推进到下一行 ARFF, 写入 ctx. 返回是否成功."""
        try:
            row = next(self._row_iter)
        except StopIteration:
            return False
        self.ctx.row = row
        self.ctx.row_idx += 1
        self.ctx.regs = arff_row_to_regs(row)
        self.ctx.function = int(float(row[COL["function"]])) if row[COL["function"]] not in ("", "?") else 0
        self.ctx.length = int(float(row[COL["length"]])) if row[COL["length"]] not in ("", "?") else 0
        self.ctx.attack_type = int(float(row[COL["categorized result"]])) if row[COL["categorized result"]] not in ("", "?") else 0
        self.ctx.attack_name = ATTACK_NAMES.get(self.ctx.attack_type, "Unknown")
        self.ctx.is_response = (row[COL["command response"]] == "0")
        self.ctx.time = float(row[COL["time"]]) if row[COL["time"]] not in ("", "?") else 0.0
        self.ctx.is_illegal_fc = self.ctx.function in ILLEGAL_FCS

        # 速率控制: 根据 ARFF time 字段差值 sleep
        if not self.rate[1]:  # 非 burst
            if self._last_row_time is not None and self.ctx.time > self._last_row_time:
                dt = self.ctx.time - self._last_row_time
                sleep_dt = dt / self.rate[0]
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
        self._last_row_time = self.ctx.time
        return True

    def _serve_client(self, conn: socket.socket, addr):
        log.info(f"Client connected: {addr}")
        try:
            while self._running:
                data = conn.recv(1024)
                if not data:
                    break
                # 每个 TCP recv 触发 1 个 ARFF 行推进 + 1 个 Modbus 响应
                if not self._advance_row():
                    log.warning("ARFF 数据用完,关闭连接")
                    break
                self._stats["req"] += 1
                self._stats["attacks"][self.ctx.attack_type] += 1

                # 检查客户端请求的 FC, 分类统计
                if len(data) >= MBAP_HEADER_LEN + 1:
                    fc = data[MBAP_HEADER_LEN]
                    if fc == FC_READ_HOLDING:
                        self._stats["fc3"] += 1
                    elif fc == FC_WRITE_SINGLE:
                        self._stats["fc6"] += 1
                    elif fc == FC_WRITE_MULTI:
                        self._stats["fc16"] += 1
                    else:
                        self._stats["exceptions"] += 1

                response = self._handler.handle(data, self.ctx)
                conn.sendall(response)
        except (ConnectionResetError, BrokenPipeError):
            log.debug(f"Client {addr} disconnected")
        finally:
            conn.close()

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)
        self._sock.settimeout(0.5)
        self._running = True
        rate_str = f"{self.rate[0]}x" if not self.rate[1] else "burst"
        log.info(f"ARFF Modbus server listening on {self.host}:{self.port} "
                 f"(rate={rate_str}, loop={self.loop})")
        log.info(f"Source: {self.arff_path}")
        log.info(f"Press Ctrl+C to stop")

        try:
            while self._running:
                try:
                    conn, addr = self._sock.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(target=self._serve_client,
                                     args=(conn, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            log.info("Stopping...")
        finally:
            self._running = False
            self._sock.close()
            self._print_stats()

    def _print_stats(self):
        log.info(f"Stats: total_req={self._stats['req']} "
                 f"(FC3={self._stats['fc3']}, FC6={self._stats['fc6']}, "
                 f"FC16={self._stats['fc16']}, exc={self._stats['exceptions']})")
        log.info(f"Attack distribution: " +
                 ", ".join(f"{ATTACK_NAMES[i]}={self._stats['attacks'][i]}"
                          for i in range(8) if self._stats['attacks'][i] > 0))


# ───────────────────────────────────────────────────────────────────
# 客户端 (验证用)
# ───────────────────────────────────────────────────────────────────
def modbus_read_holding(host: str, port: int, unit_id: int,
                        start_addr: int, count: int) -> List[int]:
    """发送 FC=3 read holding 寄存器, 返回解码后的值列表."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((host, port))
    try:
        # 构造请求: MBAP(7) + FC=3 + start(2) + count(2)
        pdu = struct.pack(">BHH", FC_READ_HOLDING, start_addr, count)
        length = len(pdu) + 1  # +1 for unit_id
        mbap = struct.pack(">HHHB", 1, 0, length, unit_id)
        s.sendall(mbap + pdu)

        # 接收响应: MBAP(7) + FC(1) + byte_count(1) + data(count*2)
        resp = s.recv(MBAP_HEADER_LEN + 2 + count * 2)
        if len(resp) < MBAP_HEADER_LEN + 2:
            raise RuntimeError(f"响应长度不足: {len(resp)}")
        fc, byte_count = struct.unpack(">BB", resp[MBAP_HEADER_LEN: MBAP_HEADER_LEN + 2])
        if fc == 0x80 | FC_READ_HOLDING:
            raise RuntimeError(f"Modbus exception: code=0x{byte_count:02X}")
        if byte_count != count * 2:
            raise RuntimeError(f"byte_count={byte_count}, 期望 {count * 2}")
        values = []
        for i in range(count):
            v = struct.unpack(">H", resp[MBAP_HEADER_LEN + 2 + i * 2:
                                          MBAP_HEADER_LEN + 2 + i * 2 + 2])[0]
            values.append(v)
        return values
    finally:
        s.close()


def run_client(args):
    host = args.host
    port = args.port
    rate = args.rate
    rate_mult, is_burst = ARFFReplayServer._parse_rate(rate)

    print(f"[Client] Connecting to {host}:{port}, read={args.read} regs, "
          f"rate={rate}, burst={args.burst}")
    total = 0
    attack_seen = {i: 0 for i in range(8)}
    t0 = time.time()
    try:
        i = 0
        while True:
            if args.burst or is_burst:
                pass  # 不 sleep
            elif rate_mult > 0:
                time.sleep(1.0 / rate_mult)
            regs = modbus_read_holding(host, port, 4, 0, args.read)
            total += 1
            if i < 5 or (i % 100 == 0 and i < 500) or args.verbose:
                print(f"  [{i:4d}] regs={[r & 0xFFFF for r in regs[:6]]}...")
            i += 1
            if args.n > 0 and total >= args.n:
                break
    except KeyboardInterrupt:
        pass
    dt = time.time() - t0
    print(f"\n[Client] 共读取 {total} 次, 用时 {dt:.2f}s, "
          f"速率 {total / max(dt, 0.01):.1f} req/s")


def run_dump(args):
    """把 ARFF 数据导出为 JSONL,无需启动服务器."""
    import json
    reader = ARFFReplayReader(args.arff, loop=False)
    out = open(args.out, "w", encoding="utf-8")
    n = 0
    for row in reader:
        if args.n > 0 and n >= args.n:
            break
        rec = {col: row[i] for i, col in enumerate(ARFF_COLS)}
        regs = arff_row_to_regs(row)
        rec["regs"] = regs
        out.write(json.dumps(rec) + "\n")
        n += 1
        if n % 1000 == 0:
            print(f"  dumped {n} rows...", end="\r")
    out.close()
    print(f"\n[Dump] {n} rows → {args.out}")


def run_dump_frames(args):
    """把 ARFF 行翻译成 Modbus 帧 (二进制),输出到 stdout / 文件."""
    reader = ARFFReplayReader(args.arff, loop=False)
    handler = ModbusRequestHandler(unit_id=4)
    n = 0
    if args.out:
        out = open(args.out, "wb")
    else:
        out = sys.stdout.buffer
    try:
        for row in reader:
            if args.n > 0 and n >= args.n:
                break
            ctx = ARFFContext(
                row=row,
                row_idx=n,
                regs=arff_row_to_regs(row),
                function=int(float(row[COL["function"]])) if row[COL["function"]] not in ("", "?") else 0,
                length=int(float(row[COL["length"]])) if row[COL["length"]] not in ("", "?") else 0,
                attack_type=int(float(row[COL["categorized result"]])) if row[COL["categorized result"]] not in ("", "?") else 0,
                attack_name=ATTACK_NAMES.get(int(float(row[COL["categorized result"]]))
                                              if row[COL["categorized result"]] not in ("", "?") else 0,
                                              "Unknown"),
                time=float(row[COL["time"]]) if row[COL["time"]] not in ("", "?") else 0.0,
                is_response=(row[COL["command response"]] == "0"),
                is_illegal_fc=(int(float(row[COL["function"]]))
                               if row[COL["function"]] not in ("", "?") else 0) in ILLEGAL_FCS,
            )
            # 构造客户端 read 请求 + 服务器响应 (完整 Modbus TCP 帧对)
            pdu_req = struct.pack(">BHH", FC_READ_HOLDING, 0, args.read)
            mbap_req = struct.pack(">HHHB", n, 0, len(pdu_req) + 1, 4)
            req_frame = mbap_req + pdu_req

            resp_frame = handler.handle(req_frame, ctx)
            out.write(req_frame)
            out.write(b"  -->  ")
            out.write(resp_frame)
            out.write(b"\n")
            n += 1
            if n % 100 == 0:
                print(f"  dumped {n} frames...", file=sys.stderr, end="\r")
    finally:
        if args.out:
            out.close()
    print(f"\n[Dump-Frames] {n} request/response pairs", file=sys.stderr)


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ARFF 驱动的 Modbus 通信模拟器 — 真实 SCADA 数据回放",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    DEFAULT_ARFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "IanArffDataset.csv")

    # start
    p_start = sub.add_parser("start", help="启动 Modbus TCP 回放服务器")
    p_start.add_argument("--arff", default=DEFAULT_ARFF,
                         help=f"ARFF/CSV 数据文件路径 (默认: {DEFAULT_ARFF})")
    p_start.add_argument("--host", default="0.0.0.0")
    p_start.add_argument("--port", type=int, default=5020)
    p_start.add_argument("--unit-id", type=int, default=4)
    p_start.add_argument("--rate", default="100x",
                         help="回放速率 (Nx 或 0=burst, 例: 100x 1x 0)")
    p_start.add_argument("--no-loop", action="store_true",
                         help="到末尾停止 (默认循环)")

    # client
    p_cli = sub.add_parser("client", help="客户端测试 (验证服务器)")
    p_cli.add_argument("--host", default="127.0.0.1")
    p_cli.add_argument("--port", type=int, default=5020)
    p_cli.add_argument("--read", type=int, default=8,
                       help="每次读取的寄存器数 (默认 8)")
    p_cli.add_argument("--rate", default="100x",
                       help="客户端请求速率")
    p_cli.add_argument("--burst", action="store_true",
                       help="最快速度 (忽略 --rate)")
    p_cli.add_argument("--n", type=int, default=20,
                       help="总共请求次数 (-1 = 无限)")
    p_cli.add_argument("--verbose", action="store_true")

    # dump
    p_dump = sub.add_parser("dump", help="把 ARFF 数据导出为 JSONL")
    p_dump.add_argument("--arff", default=DEFAULT_ARFF)
    p_dump.add_argument("--out", default="scada_replay.jsonl")
    p_dump.add_argument("--n", type=int, default=-1, help="导出行数 (-1=全部)")

    # dump-frames
    p_df = sub.add_parser("dump-frames", help="把 ARFF 翻译成 Modbus 帧 (二进制)")
    p_df.add_argument("--arff", default=DEFAULT_ARFF)
    p_df.add_argument("--out", default="", help="输出文件 (默认 stdout)")
    p_df.add_argument("--n", type=int, default=10)
    p_df.add_argument("--read", type=int, default=8)

    args = parser.parse_args()

    if args.cmd == "start":
        if not os.path.exists(args.arff):
            log.error(f"ARFF 文件不存在: {args.arff}")
            sys.exit(1)
        server = ARFFReplayServer(
            arff_path=args.arff, host=args.host, port=args.port,
            unit_id=args.unit_id, rate=args.rate, loop=not args.no_loop,
        )
        server.start()
    elif args.cmd == "client":
        run_client(args)
    elif args.cmd == "dump":
        if not os.path.exists(args.arff):
            log.error(f"ARFF 文件不存在: {args.arff}")
            sys.exit(1)
        run_dump(args)
    elif args.cmd == "dump-frames":
        if not os.path.exists(args.arff):
            log.error(f"ARFF 文件不存在: {args.arff}")
            sys.exit(1)
        run_dump_frames(args)


if __name__ == "__main__":
    main()
