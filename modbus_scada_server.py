#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modbus_scada_server.py
======================
PC 端 Modbus TCP 服务器 — 完整模拟 IanArffDataset 气管道 SCADA 系统

✅ 100% 自实现 Modbus TCP 协议 (绕过 pymodbus 3.x 弃用陷阱)
✅ 支持 FC=3 (read) / FC=6 (write single) / FC=16 (write multiple)
✅ 内置工艺仿真 (ProcessModel) — 压力/温度/CRC 实时变化
✅ Python 3.14 兼容 (无 pymodbus 依赖)

寄存器布局 (Modbus 地址 0-99, 与 IanArffDataset 对齐):
  Holding Registers (FC=3/6/16):
    0-9   : 命令参数 (setpoint, gain, reset_rate, deadband, cycle_time, rate, ...)
    10-19 : 状态 (system_mode, control_scheme, pump, solenoid, ...)
    20-29 : 传感器 (pressure, crc_rate, command_response, ...)

用法:
  # 1. 启动服务器
  python modbus_scada_server.py start --port 5020

  # 2. 客户端测试 (读 + 写)
  python modbus_scada_server.py client --port 5020 --read 5

  # 3. 写 setpoint
  python modbus_scada_server.py client --port 5020 --write-sp 25

  # 4. 一键演示 (服务器 + 客户端 + 攻击注入)
  python modbus_scada_server.py demo --port 5020

Author: SCADA IDS Team
Date: 2026-09-02
"""

import os
import sys
import time
import struct
import random
import socket
import argparse
import logging
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("modbus_scada")


# ───────────────────────────────────────────────────────────────────
# Modbus 协议常量
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


# ───────────────────────────────────────────────────────────────────
# 寄存器映射 (与 IanArffDataset 17 维对齐)
# ───────────────────────────────────────────────────────────────────
REG = {
    "setpoint":       0,   # 目标压力
    "gain":           1,   # 比例增益
    "reset_rate":     2,   # 积分速率
    "deadband":       3,   # 死区
    "cycle_time":     4,   # 周期
    "rate":           5,   # 输出变化率
    "system_mode":    6,   # 0=auto, 1=manual, 2=off
    "control_scheme": 7,   # 1=PI, 2=PID
    "pump":           8,   # 0=off, 1=on
    "solenoid":       9,   # 0=close, 1=open
    "status_word":   10,
    "alarm":         11,
    "runtime_h":     12,
    "fault_code":    13,
    "pressure_x100": 20,  # 压力 × 100 (0-100)
    "pressure_raw":  21,  # 原始 ADC
    "crc_rate":      22,  # CRC 通过率
    "cmd_response":  23,  # 命令响应
    "flow_rate":     24,  # 流量 × 100
    "temperature":   25,  # 温度 × 10
}

INIT_VALUES = {
    0: 10,    1: 115,  2: 2,    3: 5,   4: 1,   5: 0,
    6: 0,     7: 1,    8: 0,    9: 0,
    10: 0, 11: 0, 12: 0, 13: 0,
    20: 70, 21: 0, 22: 17000, 23: 0, 24: 50, 25: 250,
}


# ───────────────────────────────────────────────────────────────────
# 工艺模型
# ───────────────────────────────────────────────────────────────────
@dataclass
class ProcessModel:
    """气管道 SCADA 工艺物理模型"""
    pressure: float = 0.7
    target_pressure: float = 0.7
    flow: float = 0.5
    temperature: float = 25.0
    crc_pass: int = 17000
    setpoint: float = 10.0
    pump_on: bool = False
    solenoid_open: bool = False

    def step(self, dt: float = 0.5):
        """工艺时间步进"""
        self.target_pressure = self.setpoint * 0.07

        # 流量由泵+阀门控制
        if self.pump_on and self.solenoid_open:
            self.flow = min(1.0, self.flow + 0.05 * dt)
        elif not self.pump_on:
            self.flow = max(0.0, self.flow - 0.08 * dt)
        else:
            self.flow = max(0.0, self.flow - 0.03 * dt)

        # 压力一阶滞后
        alpha = 0.15 * dt
        self.pressure = (1 - alpha) * self.pressure + alpha * self.target_pressure
        self.pressure += random.gauss(0, 0.005)
        self.pressure = max(0.0, min(1.5, self.pressure))

        # CRC 波动
        self.crc_pass = max(10000, min(20000, self.crc_pass + random.randint(-50, 50)))

        # 温度漂移
        self.temperature += random.gauss(0, 0.05)
        self.temperature = max(20.0, min(40.0, self.temperature))


# ───────────────────────────────────────────────────────────────────
# Modbus TCP 协议处理
# ───────────────────────────────────────────────────────────────────
class ModbusRequestHandler:
    """处理单个 MBAP + PDU 请求"""

    def __init__(self, registers: List[int], process: ProcessModel,
                 unit_id: int = 4):
        self.regs = registers
        self.process = process
        self.unit_id = unit_id

    def handle(self, data: bytes) -> bytes:
        """处理完整 Modbus TCP 帧
        data = MBAP(7) + PDU
        Returns: response MBAP + PDU (或 exception)
        """
        if len(data) < MBAP_HEADER_LEN + 1:
            return self._exception(0, EXC_ILLEGAL_VALUE)

        # 解析 MBAP
        tx_id, proto_id, length, unit_id = struct.unpack(">HHHB", data[:MBAP_HEADER_LEN])

        if proto_id != 0:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        # 解析 PDU
        pdu = data[MBAP_HEADER_LEN: MBAP_HEADER_LEN + length - 1]
        if len(pdu) < 1:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        fc = pdu[0]
        pdu_data = pdu[1:]

        # 路由到具体 handler
        if fc == FC_READ_HOLDING:
            return self._handle_fc3(tx_id, pdu_data)
        elif fc == FC_WRITE_SINGLE:
            return self._handle_fc6(tx_id, pdu_data)
        elif fc == FC_WRITE_MULTI:
            return self._handle_fc16(tx_id, pdu_data)
        else:
            return self._exception(tx_id, EXC_ILLEGAL_FC)

    def _handle_fc3(self, tx_id: int, data: bytes) -> bytes:
        """FC=3: Read Holding Registers"""
        if len(data) < 4:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        start_addr, count = struct.unpack(">HH", data[:4])

        # 同步工艺模型到寄存器 (动态更新)
        self._sync_process_to_regs()

        # 验证地址范围
        if count < 1 or count > 125:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)
        if start_addr + count > len(self.regs):
            return self._exception(tx_id, EXC_ILLEGAL_ADDR)

        # 读取
        values = self.regs[start_addr: start_addr + count]
        byte_count = count * 2
        # 构造响应
        pdu = struct.pack(">B", FC_READ_HOLDING) + struct.pack(">B", byte_count)
        for v in values:
            pdu += struct.pack(">H", v & 0xFFFF)

        return self._build_response(tx_id, pdu)

    def _handle_fc6(self, tx_id: int, data: bytes) -> bytes:
        """FC=6: Write Single Register"""
        if len(data) < 4:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        addr, value = struct.unpack(">HH", data[:4])
        if addr >= len(self.regs):
            return self._exception(tx_id, EXC_ILLEGAL_ADDR)

        # 写入
        self.regs[addr] = value
        self._sync_regs_to_process(addr, value)
        log.debug(f"  FC6 write: addr={addr} value={value}")

        # 响应 = 请求回显
        pdu = struct.pack(">BHH", FC_WRITE_SINGLE, addr, value)
        return self._build_response(tx_id, pdu)

    def _handle_fc16(self, tx_id: int, data: bytes) -> bytes:
        """FC=16: Write Multiple Registers"""
        if len(data) < 4:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        start_addr, count = struct.unpack(">HH", data[:4])
        if count < 1 or count > 123:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)
        if start_addr + count > len(self.regs):
            return self._exception(tx_id, EXC_ILLEGAL_ADDR)

        byte_count = data[4]
        if byte_count != count * 2 or len(data) < 5 + byte_count:
            return self._exception(tx_id, EXC_ILLEGAL_VALUE)

        # 写入
        for i in range(count):
            value = struct.unpack(">H", data[5 + i * 2: 7 + i * 2])[0]
            self.regs[start_addr + i] = value
            self._sync_regs_to_process(start_addr + i, value)
        log.debug(f"  FC16 write: start={start_addr} count={count}")

        # 响应
        pdu = struct.pack(">BHH", FC_WRITE_MULTI, start_addr, count)
        return self._build_response(tx_id, pdu)

    def _sync_process_to_regs(self):
        """工艺模型 → 寄存器 (动态值)"""
        self.regs[REG["pressure_x100"]] = int(self.process.pressure * 100)
        self.regs[REG["flow_rate"]] = int(self.process.flow * 100)
        self.regs[REG["crc_rate"]] = self.process.crc_pass
        self.regs[REG["temperature"]] = int(self.process.temperature * 10)
        self.regs[REG["cmd_response"]] = (self.regs[REG["cmd_response"]] + 1) & 0xFFFF

    def _sync_regs_to_process(self, addr: int, value: int):
        """寄存器 → 工艺模型 (写入同步)"""
        if addr == REG["setpoint"]:
            self.process.setpoint = float(value)
        elif addr == REG["pump"]:
            self.process.pump_on = bool(value)
        elif addr == REG["solenoid"]:
            self.process.solenoid_open = bool(value)

    def _build_response(self, tx_id: int, pdu: bytes) -> bytes:
        """构造 MBAP 响应"""
        length = len(pdu) + 1  # +1 for unit_id
        mbap = struct.pack(">HHHB", tx_id, 0, length, self.unit_id)
        return mbap + pdu

    def _exception(self, tx_id: int, exc_code: int) -> bytes:
        """构造异常响应"""
        pdu = struct.pack(">BB", 0x80 | FC_READ_HOLDING, exc_code)
        return self._build_response(tx_id, pdu)


# ───────────────────────────────────────────────────────────────────
# Modbus TCP 服务器
# ───────────────────────────────────────────────────────────────────
class ModbusSCADAServer:
    """完整 Modbus TCP 服务器 — 自实现协议, 无 pymodbus 依赖"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5020,
                 unit_id: int = 4, num_registers: int = 100, auto_step: bool = True):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.auto_step = auto_step

        self.process = ProcessModel()
        self.regs = [0] * num_registers
        for addr, val in INIT_VALUES.items():
            self.regs[addr] = val

        self.handler = ModbusRequestHandler(self.regs, self.process, unit_id)
        self.sock: Optional[socket.socket] = None
        self.running = False
        self._step_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = []

    def start(self):
        log.info("=" * 60)
        log.info(f"Modbus SCADA Server (raw TCP, pymodbus-free)")
        log.info(f"  Address : {self.host}:{self.port}")
        log.info(f"  Unit ID : {self.unit_id}")
        log.info(f"  Registers: {len(self.regs)}")
        log.info(f"  Auto-step: {self.auto_step}")
        log.info("=" * 60)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.sock.settimeout(0.5)
        self.running = True

        # 启动工艺仿真
        if self.auto_step:
            self._step_thread = threading.Thread(target=self._step_loop, daemon=True)
            self._step_thread.start()

        log.info(f"Listening on {self.host}:{self.port}")
        try:
            while self.running:
                try:
                    conn, addr = self.sock.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
                self._client_threads.append(t)
        except KeyboardInterrupt:
            log.info("Stopping...")
        finally:
            self.stop()

    def _step_loop(self):
        """工艺仿真循环"""
        while self.running:
            try:
                self.process.step(dt=0.5)
            except Exception as e:
                log.error(f"step error: {e}")
            time.sleep(0.5)

    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """处理单个客户端"""
        log.info(f"Client connected: {addr}")
        conn.settimeout(60)
        try:
            while self.running:
                # 读取 MBAP 头
                header = self._recv_exact(conn, MBAP_HEADER_LEN)
                if not header:
                    break

                # 解析 length
                _, _, length, unit_id = struct.unpack(">HHHB", header)
                if unit_id != self.unit_id:
                    log.debug(f"  ignore unit_id={unit_id}")
                    continue

                # 读取 PDU
                pdu_len = length - 1  # 减去 unit_id
                pdu = self._recv_exact(conn, pdu_len)
                if not pdu:
                    break

                # 处理
                response = self.handler.handle(header + pdu)

                # 发送
                conn.sendall(response)
        except socket.timeout:
            log.debug(f"  Client {addr} timeout")
        except Exception as e:
            log.error(f"  Client {addr} error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            log.info(f"Client disconnected: {addr}")

    def _recv_exact(self, conn: socket.socket, n: int) -> Optional[bytes]:
        """接收恰好 n 字节"""
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if self._step_thread:
            self._step_thread.join(timeout=2)


# ───────────────────────────────────────────────────────────────────
# Modbus TCP 客户端
# ───────────────────────────────────────────────────────────────────
class ModbusSCADAClient:
    """Modbus TCP 客户端 — 自实现协议"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5020, unit_id: int = 4):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.tx_id = 0
        self.sock: Optional[socket.socket] = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))
        log.info(f"Connected to {self.host}:{self.port}")

    def close(self):
        if self.sock:
            self.sock.close()

    def _next_tx(self) -> int:
        self.tx_id = (self.tx_id + 1) & 0xFFFF
        return self.tx_id

    def read_holding_registers(self, addr: int, count: int = 1) -> List[int]:
        """FC=3 读保持寄存器"""
        # 请求
        tx = self._next_tx()
        pdu = struct.pack(">BHH", FC_READ_HOLDING, addr, count)
        mbap = struct.pack(">HHHB", tx, 0, len(pdu) + 1, self.unit_id)
        self.sock.sendall(mbap + pdu)

        # 响应
        resp = self._recv_full()
        if not resp:
            return []
        # 解析
        _, _, length, _ = struct.unpack(">HHHB", resp[:MBAP_HEADER_LEN])
        pdu = resp[MBAP_HEADER_LEN: MBAP_HEADER_LEN + length - 1]
        if pdu[0] == 0x80 | FC_READ_HOLDING:
            log.error(f"  Exception: code={pdu[1]}")
            return []
        byte_count = pdu[1]
        values = []
        for i in range(count):
            v = struct.unpack(">H", pdu[2 + i * 2: 4 + i * 2])[0]
            values.append(v)
        return values

    def write_register(self, addr: int, value: int) -> bool:
        """FC=6 写单个寄存器"""
        tx = self._next_tx()
        pdu = struct.pack(">BHH", FC_WRITE_SINGLE, addr, value)
        mbap = struct.pack(">HHHB", tx, 0, len(pdu) + 1, self.unit_id)
        self.sock.sendall(mbap + pdu)
        resp = self._recv_full()
        return resp is not None

    def write_registers(self, addr: int, values: List[int]) -> bool:
        """FC=16 写多个寄存器"""
        count = len(values)
        byte_count = count * 2
        tx = self._next_tx()
        pdu = struct.pack(">BHHB", FC_WRITE_MULTI, addr, count, byte_count)
        for v in values:
            pdu += struct.pack(">H", v & 0xFFFF)
        mbap = struct.pack(">HHHB", tx, 0, len(pdu) + 1, self.unit_id)
        self.sock.sendall(mbap + pdu)
        resp = self._recv_full()
        return resp is not None

    def _recv_full(self) -> Optional[bytes]:
        """接收完整 MBAP + PDU"""
        # 读 MBAP
        header = self._recv_exact(MBAP_HEADER_LEN)
        if not header:
            return None
        _, _, length, _ = struct.unpack(">HHHB", header)
        # 读 PDU
        pdu = self._recv_exact(length - 1)
        if not pdu:
            return None
        return header + pdu

    def _recv_exact(self, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def read_pressure(self) -> float:
        v = self.read_holding_registers(REG["pressure_x100"], 1)
        return v[0] / 100.0 if v else -1.0

    def read_all(self) -> Dict[str, int]:
        regs = self.read_holding_registers(0, 30)
        if not regs:
            return {}
        decoded = {"raw": regs}
        for name, addr in REG.items():
            if addr < len(regs):
                decoded[name] = regs[addr]
        return decoded

    def demo_loop(self, n: int = 5, interval: float = 1.0):
        """演示: 循环读状态"""
        log.info(f"Client demo: 读 {n} 次 (间隔 {interval}s)")
        log.info("-" * 60)
        for i in range(n):
            regs = self.read_all()
            t = time.strftime("%H:%M:%S")
            p = regs.get("pressure_x100", 0) / 100.0
            sp = regs.get("setpoint", 0)
            pump = regs.get("pump", 0)
            sol = regs.get("solenoid", 0)
            crc = regs.get("crc_rate", 0)
            log.info(f"  [{t}] i={i+1}/{n}  P={p:.3f}  SP={sp:>4}  泵={pump}  阀={sol}  CRC={crc}")
            time.sleep(interval)
        log.info("-" * 60)


# ───────────────────────────────────────────────────────────────────
# 一键演示
# ───────────────────────────────────────────────────────────────────
def run_demo(port: int = 5020):
    """启动服务器 + 客户端 + 攻击注入"""
    log.info("=" * 60)
    log.info("Modbus SCADA 一键演示 (pymodbus-free)")
    log.info("=" * 60)

    server = ModbusSCADAServer(host="0.0.0.0", port=port, auto_step=True)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    time.sleep(2)

    try:
        client = ModbusSCADAClient("127.0.0.1", port)
        client.connect()
        log.info(">>> 阶段 1: 正常流量")
        client.demo_loop(n=3, interval=0.5)

        log.info("")
        log.info(">>> 阶段 2: 注入攻击")
        log.info("  [MPCI] 写 setpoint=999, gain=-500 (越限)")
        client.write_registers(0, [999, -500 & 0xFFFF, 2, 5, 1, 0, 0, 1, 1, 1])
        time.sleep(0.5)

        log.info("  [MSCI] 写 system_mode=99 (非法)")
        client.write_register(6, 99)
        time.sleep(0.5)

        log.info("  [NMRI] 写 pressure_x100=999 (异常高)")
        client.write_register(20, 999)
        time.sleep(0.5)

        log.info("")
        log.info(">>> 阶段 3: 验证最终状态")
        time.sleep(1)
        regs = client.read_all()
        log.info(f"  寄存器状态: {regs}")
        client.close()
    finally:
        server.stop()
        log.info("演示结束")


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Modbus TCP 服务器模拟器 (气管道 SCADA) — pymodbus-free",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # start
    p_start = sub.add_parser("start", help="启动 Modbus TCP 服务器")
    p_start.add_argument("--host", default="0.0.0.0")
    p_start.add_argument("--port", type=int, default=5020)
    p_start.add_argument("--unit-id", type=int, default=4)
    p_start.add_argument("--no-auto-step", action="store_true")

    # client
    p_client = sub.add_parser("client", help="客户端测试")
    p_client.add_argument("--host", default="127.0.0.1")
    p_client.add_argument("--port", type=int, default=5020)
    p_client.add_argument("--unit-id", type=int, default=4)
    p_client.add_argument("--read", type=int, default=5)
    p_client.add_argument("--interval", type=float, default=1.0)
    p_client.add_argument("--write-sp", type=int, default=None)

    # demo
    p_demo = sub.add_parser("demo", help="一键演示")
    p_demo.add_argument("--port", type=int, default=5020)

    args = parser.parse_args()

    if args.cmd == "start":
        srv = ModbusSCADAServer(args.host, args.port, args.unit_id,
                                 auto_step=not args.no_auto_step)
        try:
            srv.start()
        except OSError as e:
            if "10048" in str(e) or "Address already" in str(e):
                log.error(f"端口 {args.port} 已被占用, 用 --port 指定其他端口")
            raise

    elif args.cmd == "client":
        cli = ModbusSCADAClient(args.host, args.port, args.unit_id)
        cli.connect()
        if args.write_sp is not None:
            cli.write_register(0, args.write_sp)
            log.info(f"  setpoint = {args.write_sp}")
        else:
            cli.demo_loop(n=args.read, interval=args.interval)
        cli.close()

    elif args.cmd == "demo":
        run_demo(args.port)


if __name__ == "__main__":
    main()
