#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attack_injector.py
==================
SCADA / Modbus 攻击注入器 — 完整 7 类攻击生成 + 23-dim 特征工程 + MCU 推理接口

设计目标:
  1. 完全复现 IanArffDataset 的 4 行 Modbus cycle 模式
     (FC=3 ping → FC=3 read_resp → FC=16 write_cmd → FC=16 write_ack)
  2. 实现 7 类攻击 (NMRI / CMRI / MSCI / MPCI / MFCI / DoS / Recon)
  3. 输出与训练时一致的 27-dim SCADA 特征
  4. 支持 23-dim (KEEP_23) 切片,直接喂给 TCN+SE ch=32 5-seed
  5. 导出 JSONL 给 MCU 端做 bit-perfect 推理验证

用法:
  # 1. 仅生成正常流量 + 攻击样本 (CSV + NPY)
  python attack_injector.py generate --n-normal 5000 --n-attack 1000 \
      --out-dir ./attack_test_data

  # 2. 启动 Modbus TCP 仿真服务器,实时注入攻击
  python attack_injector.py server --host 0.0.0.0 --port 5020 \
      --inject-cmd "5:0.02,4:0.05,2:0.01"  # 每 5% 概率注入 MSCI / 5% MPCI

  # 3. 从 PC 已有 .npy 喂入 TCN+SE (无需 MCU)
  python attack_injector.py infer --input X_test_attack.npy \
      --model tcns_ch32_seed42.pt --threshold 0.5

Author: SCADA IDS Team
Date: 2026-09-02
"""

import os
import sys
import json
import time
import socket
import struct
import random
import argparse
import logging
import threading
import csv
from collections import deque, defaultdict
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict

# ───────────────────────────────────────────────────────────────────
# 配置
# ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attack_injector")

# 17 维原始列名 (与 IanArffDataset.csv 表头一致)
RAW_COLS = [
    "address", "function", "length", "setpoint", "gain", "reset rate",
    "deadband", "cycle time", "rate", "system mode", "control scheme",
    "pump", "solenoid", "pressure measurement", "crc rate",
    "command response", "time",
]

# 27-dim SCADA 特征 (含 17 原始 + time_diff + 10 SCADA)
SCADA27_COLS = RAW_COLS + ["time_diff"] + [
    "time_since_same", "is_unusual_fc", "is_response",  # row-level
    "cmd_count_w", "resp_count_w", "cmd_resp_balance_w",  # window-level
    "crc_mean_w", "crc_max_w", "press_mean_w", "length_nunique_w",
]

# 23-dim TCN+SE ch=32 冠军版 (drop {2, 3, 20, 22})
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]

# 7 类攻击 (与 IanArffDataset 'categorized result' 一致)
ATTACK_TYPES = {
    0: "Normal",
    1: "NMRI",   # Naive Malicious Response Injection
    2: "CMRI",   # Complex Malicious Response Injection
    3: "MSCI",   # Malicious State Command Injection
    4: "MPCI",   # Malicious Parameter Command Injection
    5: "MFCI",   # Malicious Function Code Injection
    6: "DoS",    # Denial of Service
    7: "Recon",  # Reconnaissance
}

# IanArffDataset 函数码分布 (用于正常流量)
NORMAL_FC_DIST = {3: 0.50, 16: 0.49, 8: 0.01}  # 97% 是 FC=3/16

# 异常 FC (MFCI 攻击常用)
ILLEGAL_FCS = [43, 73, 90, 100, 136, 137, 138, 139, 140, 171, 172]

# Modbus 标准功能码
MODBUS_FC = {
    1: "Read Coils",
    2: "Read Discrete Inputs",
    3: "Read Holding Registers",
    4: "Read Input Registers",
    5: "Write Single Coil",
    6: "Write Single Register",
    7: "Read Exception Status",
    8: "Diagnostic / Echo",
    15: "Write Multiple Coils",
    16: "Write Multiple Registers",
    43: "Read Device Identification (illegal if not on dev)",
}

# ───────────────────────────────────────────────────────────────────
# 数据类
# ───────────────────────────────────────────────────────────────────
@dataclass
class SCADARow:
    """单条 Modbus 记录 (17 维 + 攻击标签)"""
    address: int = 4
    function: int = 3
    length: int = 16
    setpoint: float = float("nan")
    gain: float = float("nan")
    reset_rate: float = float("nan")
    deadband: float = float("nan")
    cycle_time: float = float("nan")
    rate: float = float("nan")
    system_mode: int = -1
    control_scheme: int = -1
    pump: int = -1
    solenoid: int = -1
    pressure: float = float("nan")
    crc_rate: int = 0
    command_response: int = 1  # 1=command, 0=response
    time: int = 0

    # 攻击标签
    attack_type: int = 0  # 0=normal, 1-7
    attack_subtype: int = 0  # 'specific result' (1-35)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "function": self.function,
            "length": self.length,
            "setpoint": self.setpoint,
            "gain": self.gain,
            "reset rate": self.reset_rate,
            "deadband": self.deadband,
            "cycle time": self.cycle_time,
            "rate": self.rate,
            "system mode": self.system_mode,
            "control scheme": self.control_scheme,
            "pump": self.pump,
            "solenoid": self.solenoid,
            "pressure measurement": self.pressure,
            "crc rate": self.crc_rate,
            "command response": self.command_response,
            "time": self.time,
        }


@dataclass
class ProcessState:
    """PLC 工艺状态 (用于物理一致性约束)"""
    setpoint: float = 10.0
    gain: float = 115.0
    reset_rate: float = 0.2
    deadband: float = 0.5
    cycle_time: int = 1
    rate: float = 0.0
    system_mode: int = 0
    control_scheme: int = 1
    pump: int = 0
    solenoid: int = 0
    pressure: float = 0.7  # 工艺正常值 0.5-0.8
    crc_rate: int = 17000
    last_update: float = 0.0

    def step(self, dt: float = 1.0):
        """物理时间步进: 压力受 setpoint 驱动, 加上噪声"""
        # 一阶滞后模型: pressure → setpoint 方向
        target = self.setpoint * 0.07 + random.gauss(0, 0.01)
        alpha = 0.1 * dt
        self.pressure = (1 - alpha) * self.pressure + alpha * target
        # CRC rate 工艺相关
        self.crc_rate = max(10000, min(20000, int(self.crc_rate + random.gauss(0, 50))))


# ───────────────────────────────────────────────────────────────────
# SCADA 仿真器 (正常流量生成)
# ───────────────────────────────────────────────────────────────────
class SCADASimulator:
    """4 行 Modbus cycle 生成器 (FC=3 ping → FC=3 read → FC=16 write → FC=16 ack)"""

    def __init__(self, seed: int = 42, start_time: int = 1418682163, cycle_dt: float = 2.0):
        self.rng = random.Random(seed)
        self.state = ProcessState()
        self.start_time = start_time
        self.cycle_dt = cycle_dt
        self.current_time = start_time
        self.row_counter = 0

    def tick(self) -> List[SCADARow]:
        """生成 1 个完整 cycle (4 rows)"""
        t = self.current_time
        rows = []

        # 工艺步进
        self.state.step(dt=1.0)

        # Row 1: FC=3 len=16 (read request, ping)
        rows.append(SCADARow(
            address=4, function=3, length=16,
            command_response=1, time=t,
        ))

        # Row 2: FC=3 len=46 (read response with sensor data)
        rows.append(SCADARow(
            address=4, function=3, length=46,
            pressure=self.state.pressure,
            crc_rate=self.state.crc_rate,
            command_response=0, time=t,
        ))

        # Row 3: FC=16 len=90 (write multiple registers)
        rows.append(SCADARow(
            address=4, function=16, length=90,
            setpoint=self.state.setpoint,
            gain=self.state.gain,
            reset_rate=self.state.reset_rate,
            deadband=self.state.deadband,
            cycle_time=self.state.cycle_time,
            rate=self.state.rate,
            system_mode=self.state.system_mode,
            control_scheme=self.state.control_scheme,
            pump=self.state.pump,
            solenoid=self.state.solenoid,
            command_response=1, time=t + 2,
        ))

        # Row 4: FC=16 len=16 (write response/ack)
        rows.append(SCADARow(
            address=4, function=16, length=16,
            command_response=0, time=t + 2,
        ))

        # 时间推进 (2-3s per cycle)
        self.current_time += self.cycle_dt
        self.row_counter += 4
        return rows


# ───────────────────────────────────────────────────────────────────
# 7 类攻击注入器
# ───────────────────────────────────────────────────────────────────
class AttackInjector:
    """7 类 SCADA 攻击生成器 (与 IanArffDataset categorized result 对齐)"""

    def __init__(self, simulator: SCADASimulator, attack_type: int, intensity: float = 1.0):
        self.sim = simulator
        self.attack_type = attack_type
        self.intensity = max(0.0, min(1.0, intensity))
        self.attack_id_counter = 0
        self.rng = random.Random(simulator.rng.random() * 1e6)

    def _new_id(self) -> int:
        """specific result 子类编号 (1-35)"""
        # 简化: 1-5 子类映射
        sub = (self.attack_id_counter % 5) + 1
        self.attack_id_counter += 1
        return sub

    def inject_into_cycle(self, rows: List[SCADARow]) -> List[SCADARow]:
        """对 1 个 cycle (4 rows) 应用攻击"""
        if self.attack_type == 0:
            return rows
        method = getattr(self, f"_attack_{['normal','nmri','cmri','msci','mpci','mfci','dos','recon'][self.attack_type]}")
        return method(rows)

    # ───────── L3 数据层攻击 ─────────
    def _attack_nmri(self, rows: List[SCADARow]) -> List[SCADARow]:
        """NMRI: Naive Malicious Response Injection
        - 修改 read response (FC=3 len=46) 的压力值
        - 随机化 + 大幅偏离工艺合理范围
        """
        for r in rows:
            if r.function == 3 and r.length == 46 and r.command_response == 0:
                # 注入极端值 (0.0 或 1.0)
                r.pressure = self.rng.choice([0.0, 1.0, 0.01, 0.99])
                r.crc_rate = self.rng.randint(20000, 30000)  # CRC rate 异常
                r.attack_type = 1
                r.attack_subtype = self._new_id()
        return rows

    def _attack_cmri(self, rows: List[SCADARow]) -> List[SCADARow]:
        """CMRI: Complex Malicious Response Injection
        - 智能篡改: 修改 response 让数据看起来"合理"但工艺不一致
        - 例: 流量大但压力低 (物理不可能)
        """
        for r in rows:
            if r.function == 3 and r.length == 46 and r.command_response == 0:
                # 工艺一致但缓慢漂移
                r.pressure = self.sim.state.pressure * self.rng.uniform(0.3, 1.7)
                r.pressure = max(0.0, min(1.5, r.pressure))
                # CRC 异常递增
                r.crc_rate = self.rng.randint(15000, 25000)
                r.attack_type = 2
                r.attack_subtype = self._new_id()
        return rows

    def _attack_msci(self, rows: List[SCADARow]) -> List[SCADARow]:
        """MSCI: Malicious State Command Injection
        - 注入异常状态 (pump/solenoid/system mode)
        - 例如: 紧急关断 (solenoid=1) 但流量未降
        """
        for r in rows:
            if r.function == 16 and r.length == 90 and r.command_response == 1:
                # 强制切换状态
                r.system_mode = self.rng.choice([0, 1, 2, 3, 99])  # 99=非法
                r.control_scheme = self.rng.choice([0, 1, 2])
                r.pump = self.rng.choice([0, 1])
                r.solenoid = 1 if self.rng.random() < 0.5 else 0
                r.attack_type = 3
                r.attack_subtype = self._new_id()
        return rows

    def _attack_mpci(self, rows: List[SCADARow]) -> List[SCADARow]:
        """MPCI: Malicious Parameter Command Injection
        - 篡改 setpoint/gain/reset_rate/deadband/cycle_time
        - 比例控制参数 (gain) 异常最致命
        """
        for r in rows:
            if r.function == 16 and r.length == 90 and r.command_response == 1:
                r.setpoint = self.rng.uniform(-100, 500)  # 越限
                r.gain = self.rng.uniform(-500, 5000)  # 比例异常
                r.reset_rate = self.rng.uniform(-1, 5)  # 积分异常
                r.deadband = self.rng.uniform(-1, 10)
                r.cycle_time = self.rng.choice([0, -1, 999, 65535])
                r.attack_type = 4
                r.attack_subtype = self._new_id()
        return rows

    # ───────── L2 功能码层攻击 ─────────
    def _attack_mfci(self, rows: List[SCADARow]) -> List[SCADARow]:
        """MFCI: Malicious Function Code Injection
        - 注入非法 function code
        - length 也异常 (与 FC 不匹配)
        """
        illegal_fc = self.rng.choice(ILLEGAL_FCS)
        illegal_len = self.rng.choice([10, 12, 14, 16, 46, 90, 200, 999])
        # 替换第一行 (ping) 为非法 FC
        rows[0].function = illegal_fc
        rows[0].length = illegal_len
        rows[0].command_response = 1
        rows[0].attack_type = 5
        rows[0].attack_subtype = self._new_id()
        # 后续行可能也有异常 FC
        if self.rng.random() < 0.3:
            rows[1].function = self.rng.choice(ILLEGAL_FCS)
            rows[1].attack_type = 5
        return rows

    # ───────── L4 时序层攻击 ─────────
    def _attack_dos(self, rows: List[SCADARow]) -> List[SCADARow]:
        """DoS: Denial of Service
        - 高频发包 (cycle_dt → 0)
        - 全部写命令 (FC=16) 压满带宽
        - 或: 时间戳重复 (旧包重放)
        """
        attack_mode = self.rng.choice(["burst", "replay", "spam"])
        if attack_mode == "burst":
            # 在原 cycle 内插入 5-10 个额外 write 命令
            extra = []
            for _ in range(self.rng.randint(5, 10)):
                extra.append(SCADARow(
                    address=4, function=16, length=90,
                    setpoint=self.sim.state.setpoint,
                    gain=self.sim.state.gain,
                    system_mode=self.sim.state.system_mode,
                    control_scheme=self.sim.state.control_scheme,
                    pump=self.sim.state.pump,
                    solenoid=self.sim.state.solenoid,
                    command_response=1, time=rows[0].time,
                    attack_type=6, attack_subtype=self._new_id(),
                ))
            rows = extra + rows
        elif attack_mode == "replay":
            # 时间戳倒回
            old_time = self.sim.start_time + self.rng.randint(0, 1000)
            for r in rows:
                r.time = old_time
                r.attack_type = 6
                r.attack_subtype = self._new_id()
        else:  # spam
            for r in rows:
                r.attack_type = 6
                r.attack_subtype = self._new_id()
        return rows

    # ───────── L1+L4 协议扫描 ─────────
    def _attack_recon(self, rows: List[SCADARow]) -> List[SCADARow]:
        """Reconnaissance: 扫描/嗅探
        - 轮询所有 address (1-247)
        - 短 length (10/12/14, 探测用)
        - 频率低 (避免被 DoS 检测)
        """
        # 替换为扫描模式
        for r in rows[:2]:
            r.address = self.rng.randint(1, 247)  # 扫描其他设备
            r.function = self.rng.choice([1, 2, 7])  # 探测类 FC
            r.length = self.rng.choice([10, 12, 14])
            r.command_response = 1
            r.attack_type = 7
            r.attack_subtype = self._new_id()
        return rows


# ───────────────────────────────────────────────────────────────────
# 特征工程 (17 维 + time_diff + 10 SCADA = 27 维)
# ───────────────────────────────────────────────────────────────────
class FeatureExtractor:
    """从 17 维原始数据生成 27-dim SCADA 特征"""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size

    def add_row_level(self, rows: List[Dict]) -> List[Dict]:
        """添加行级特征: time_diff, time_since_same, is_unusual_fc, is_response"""
        # 1. time_diff
        for i, r in enumerate(rows):
            r["time_diff"] = 0 if i == 0 else r["time"] - rows[i - 1]["time"]
        # 2. time_since_last_same_addr_func
        last_seen: Dict[Tuple[int, int], int] = {}
        for r in rows:
            key = (r["address"], r["function"])
            r["time_since_same"] = r["time"] - last_seen.get(key, r["time"])
            last_seen[key] = r["time"]
        # 3. is_unusual_fc
        for r in rows:
            r["is_unusual_fc"] = 1 if r["function"] in ILLEGAL_FCS else 0
        # 4. is_response (alias of command response, 1=cmd, 0=resp)
        for r in rows:
            r["is_response"] = r["command response"]
        return rows

    def add_window_level(self, rows: List[Dict]) -> List[Dict]:
        """添加窗口聚合特征 (window=self.window_size, broadcasted to all rows in window)"""
        n = len(rows)
        ws = self.window_size
        for i in range(n):
            start = max(0, i - ws + 1)
            window = rows[start: i + 1]
            # cmd_count (function=16 且 command_response=1)
            cmd = sum(1 for r in window if r["function"] == 16 and r["command response"] == 1)
            # resp_count
            resp = sum(1 for r in window if r["command response"] == 0)
            # cmd_resp_balance
            rwin = rows[i]
            rwin["cmd_count_w"] = cmd
            rwin["resp_count_w"] = resp
            rwin["cmd_resp_balance_w"] = (cmd - resp) / ws
            # crc_mean / crc_max
            crcs = [r["crc rate"] for r in window if r["crc rate"] > 0]
            rwin["crc_mean_w"] = sum(crcs) / len(crcs) if crcs else 0
            rwin["crc_max_w"] = max(crcs) if crcs else 0
            # press_mean
            presses = [r["pressure measurement"] for r in window
                       if r["pressure measurement"] not in (None, "", "?", "nan")]
            rwin["press_mean_w"] = (sum(float(p) for p in presses) / len(presses)) if presses else 0.0
            # length_nunique
            rwin["length_nunique_w"] = len(set(r["length"] for r in window))
        return rows

    def to_27dim_matrix(self, rows: List[Dict]) -> Tuple["np.ndarray", "np.ndarray"]:
        """输出 (N, 27) 特征矩阵 + (N,) 攻击标签"""
        try:
            import numpy as np
        except ImportError:
            log.error("需要 numpy: pip install numpy")
            raise
        feats, labels = [], []
        for r in rows:
            row_vec = []
            for col in SCADA27_COLS:
                v = r.get(col, 0)
                if v in (None, "", "?", "nan"):
                    v = 0
                try:
                    row_vec.append(float(v))
                except (ValueError, TypeError):
                    row_vec.append(0.0)
            feats.append(row_vec)
            labels.append(r.get("attack_type", 0))
        return np.array(feats, dtype="float32"), np.array(labels, dtype="int64")

    def to_23dim(self, X_27: "np.ndarray") -> "np.ndarray":
        """KEEP_23 切片: drop {2,3,20,22}"""
        return X_27[:, KEEP_23]


# ───────────────────────────────────────────────────────────────────
# MCU 推理接口 (JSONL 日志)
# ───────────────────────────────────────────────────────────────────
class MCULogger:
    """记录 (window_features, label, mcu_prediction) 三元组
    用于事后 bit-perfect 验证 + 指标计算

    window_features 支持 2 种格式:
      - 1D list[float]: 单个 timestep 的 23 维特征 (向后兼容)
      - 2D list[list[float]]: (ws, 23) 窗口矩阵,匹配 TCN+SE 训练输入
    """

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.count = 0

    def log(self, window_features, label: int,
            mcu_prob: Optional[float] = None, mcu_pred: Optional[int] = None,
            timestamp: int = 0):
        # 自动检测 1D / 2D
        if window_features and isinstance(window_features[0], list):
            x = [[float(v) for v in row] for row in window_features]
            ws = len(x)
            n_feat = len(x[0]) if ws > 0 else 0
        else:
            x = [list(map(float, window_features))]
            ws, n_feat = 1, len(x[0]) if x[0] else 0
        rec = {
            "idx": self.count,
            "t": int(timestamp) if timestamp else 0,
            "x": x,
            "shape": [ws, n_feat],
            "y": int(label),
            "y_name": ATTACK_TYPES.get(int(label), "Unknown"),
            "p_mcu": mcu_prob,
            "pred_mcu": mcu_pred,
        }
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.count += 1

    def close(self):
        self.f.close()
        log.info(f"MCU log saved: {self.path} ({self.count} records)")


# ───────────────────────────────────────────────────────────────────
# Modbus TCP 仿真服务器 (可选)
# ───────────────────────────────────────────────────────────────────
class ModbusTCPServer:
    """最小 Modbus TCP 服务器 (FC=3 read, FC=16 write)
    接收客户端命令, 用 SCADASimulator 生成数据, 可选注入攻击
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5020,
                 inject_cmd: Optional[Dict[int, float]] = None):
        self.host = host
        self.port = port
        self.sim = SCADASimulator()
        self.inject_cmd = inject_cmd or {}  # {attack_type: prob_per_cycle}
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.attack_log: List[Dict] = []

    def _maybe_inject(self) -> int:
        """按概率决定是否注入攻击, 返回 attack_type"""
        for atype, prob in self.inject_cmd.items():
            if random.random() < prob:
                injector = AttackInjector(self.sim, atype)
                return atype
        return 0

    def _handle_client(self, conn: socket.socket, addr):
        log.info(f"Client connected: {addr}")
        cycle_buf = []
        try:
            while self.running:
                data = conn.recv(1024)
                if not data:
                    break
                # 简化: 每次 recv 触发 1 个 cycle
                cycle = self.sim.tick()
                atype = self._maybe_inject()
                if atype > 0:
                    injector = AttackInjector(self.sim, atype)
                    cycle = injector.inject_into_cycle(cycle)
                # 推送给客户端
                for row in cycle:
                    payload = json.dumps({
                        "function": row.function,
                        "length": row.length,
                        "pressure": row.pressure,
                        "setpoint": row.setpoint,
                        "attack": ATTACK_TYPES[row.attack_type],
                    }).encode() + b"\n"
                    conn.sendall(payload)
                if atype > 0:
                    self.attack_log.append({
                        "t": time.time(), "type": atype,
                        "name": ATTACK_TYPES[atype],
                    })
        except Exception as e:
            log.error(f"Client {addr} error: {e}")
        finally:
            conn.close()
            log.info(f"Client disconnected: {addr}")

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.running = True
        log.info(f"Modbus TCP server listening on {self.host}:{self.port}")
        log.info(f"Injection config: {self.inject_cmd}")
        try:
            while self.running:
                conn, addr = self.sock.accept()
                t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        log.info(f"Total attacks injected: {len(self.attack_log)}")


# ───────────────────────────────────────────────────────────────────
# 数据集生成 (generate 子命令)
# ───────────────────────────────────────────────────────────────────
def generate_dataset(
    n_normal: int = 5000,
    n_attack: int = 1000,
    out_dir: str = "./attack_test_data",
    attack_mix: Optional[Dict[int, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """生成正常 + 攻击混合数据集
    Returns: 路径字典 {csv, npy, jsonl}
    """
    os.makedirs(out_dir, exist_ok=True)
    sim = SCADASimulator(seed=seed)

    # 默认攻击比例 (与 IanArffDataset 接近: MPCI 最大, DoS 最小)
    if attack_mix is None:
        attack_mix = {1: 0.13, 2: 0.22, 3: 0.13, 4: 0.34,
                      5: 0.08, 6: 0.04, 7: 0.06}

    all_rows: List[Dict] = []

    # 1. 生成正常流量
    log.info(f"生成正常流量: {n_normal} 行 (≈ {n_normal // 4} cycles)")
    n_cycles = n_normal // 4
    for _ in range(n_cycles):
        for r in sim.tick():
            d = r.to_dict()
            d["attack_type"] = 0
            d["attack_subtype"] = 0
            all_rows.append(d)

    # 2. 注入攻击
    log.info(f"注入攻击: 总 {n_attack} 行, 比例 {attack_mix}")
    n_attack_cycles = n_attack // 4
    attack_counts = {a: 0 for a in attack_mix}
    for atype, prob in attack_mix.items():
        n_cycles_for_type = int(n_attack_cycles * prob)
        injector = AttackInjector(sim, atype)
        for _ in range(n_cycles_for_type):
            cycle = sim.tick()
            cycle = injector.inject_into_cycle(cycle)
            for r in cycle:
                d = r.to_dict()
                d["attack_type"] = r.attack_type
                d["attack_subtype"] = r.attack_subtype
                all_rows.append(d)
                if r.attack_type == atype:
                    attack_counts[atype] += 1

    log.info(f"生成完成: {len(all_rows)} 行, 攻击分布: {attack_counts}")

    # 3. 特征工程
    log.info("特征工程: 17 维 → 27 维 → 23-dim")
    extractor = FeatureExtractor(window_size=8)
    all_rows = extractor.add_row_level(all_rows)
    all_rows = extractor.add_window_level(all_rows)

    # 4. 保存
    paths = {}

    # 4a. CSV (原始 17 + 标签)
    csv_path = os.path.join(out_dir, "raw_attack_dataset.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLS + ["attack_type", "attack_subtype"])
        writer.writeheader()
        for r in all_rows:
            row = {col: r.get(col, "") for col in RAW_COLS}
            row["attack_type"] = r.get("attack_type", 0)
            row["attack_subtype"] = r.get("attack_subtype", 0)
            writer.writerow(row)
    paths["csv"] = csv_path
    log.info(f"  CSV saved: {csv_path}")

    # 4b. NPY (27-dim + 23-dim)
    try:
        import numpy as np
        X_27, y = extractor.to_27dim_matrix(all_rows)
        X_23 = extractor.to_23dim(X_27)
        np.save(os.path.join(out_dir, "X_attack_27dim.npy"), X_27)
        np.save(os.path.join(out_dir, "X_attack_23dim.npy"), X_23)
        np.save(os.path.join(out_dir, "y_attack.npy"), y)
        paths["X_27"] = os.path.join(out_dir, "X_attack_27dim.npy")
        paths["X_23"] = os.path.join(out_dir, "X_attack_23dim.npy")
        paths["y"] = os.path.join(out_dir, "y_attack.npy")
        log.info(f"  NPY saved: X_27{X_27.shape}, X_23{X_23.shape}, y{y.shape}")
    except ImportError:
        log.warning("numpy 未安装,跳过 NPY 输出")

    # 4c. JSONL (MCU 推理接口) — 输出 (ws, 23) 窗口矩阵,匹配训练格式
    jsonl_path = os.path.join(out_dir, "mcu_attack_log.jsonl")
    mcu_logger = MCULogger(jsonl_path)
    ws = extractor.window_size
    n_total_windows = len(all_rows) - ws + 1
    for i in range(ws - 1, len(all_rows)):
        # 当前 window (含当前行 + 前 ws-1 行)
        window_rows = all_rows[i - ws + 1: i + 1]
        # 对每个 timestep 提取 23-dim 特征 (NaN → 0)
        window_23 = []
        for r in window_rows:
            timestep_23 = []
            for j in KEEP_23:
                col = SCADA27_COLS[j]
                v = r.get(col, 0)
                if v is None or (isinstance(v, float) and v != v):  # NaN check
                    v = 0.0
                try:
                    timestep_23.append(float(v))
                except (ValueError, TypeError):
                    timestep_23.append(0.0)
            window_23.append(timestep_23)
        mcu_logger.log(
            window_features=window_23,  # 现在是 (ws, 23) 矩阵
            label=all_rows[i].get("attack_type", 0),
            timestamp=all_rows[i].get("time", 0),
        )
    mcu_logger.close()
    paths["jsonl"] = jsonl_path
    paths["n_windows"] = n_total_windows

    # 4d. 报告
    report = {
        "n_total_rows": len(all_rows),
        "n_attack_rows": int(sum(1 for r in all_rows if r.get("attack_type", 0) > 0)),
        "attack_counts_by_type": attack_counts,
        "feature_dims": {"27dim": 27, "23dim_KEEP_23": len(KEEP_23)},
        "out_dir": out_dir,
        "paths": paths,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"  Report: {os.path.join(out_dir, 'report.json')}")

    return report


# ───────────────────────────────────────────────────────────────────
# 推理 / 评估 (infer 子命令)
# ───────────────────────────────────────────────────────────────────
def infer_and_evaluate(
    input_npy: str,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """加载 .npy + 可选模型, 输出指标"""
    try:
        import numpy as np
    except ImportError:
        log.error("需要 numpy")
        return {}

    log.info(f"Loading {input_npy}")
    X = np.load(input_npy)
    log.info(f"  Shape: {X.shape}")

    # 加载标签 (假设同名 y_*.npy)
    y_path = input_npy.replace("X_", "y_").replace("_23dim", "").replace("_27dim", "")
    if not os.path.exists(y_path):
        # 尝试同级
        candidates = [y_path,
                      os.path.join(os.path.dirname(input_npy), "y_attack.npy"),
                      os.path.join(os.path.dirname(input_npy), "y_test_binary_v2_scada.npy")]
        y_path = next((c for c in candidates if os.path.exists(c)), None)
    if y_path is None:
        log.warning("未找到标签 .npy, 无法评估")
        return {}

    y = np.load(y_path)
    log.info(f"  Labels: {y.shape}, attack ratio: {y.mean():.3f}")

    # 加载模型
    probs = None
    if model_path and os.path.exists(model_path):
        try:
            import torch
            log.info(f"Loading model: {model_path}")
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            model = ckpt.get("model", ckpt)
            model.eval()
            with torch.no_grad():
                # 滑动窗口
                ws = 8
                preds = []
                for i in range(0, len(X) - ws + 1):
                    x_win = torch.tensor(X[i: i + ws], dtype=torch.float32).unsqueeze(0)
                    if hasattr(model, "forward"):
                        out = model(x_win)
                    else:
                        out = model(x_win)
                    preds.append(torch.sigmoid(out).squeeze().item())
                probs = np.array(preds)
                # 对齐标签
                y_eval = y[ws - 1:]
        except Exception as e:
            log.error(f"模型推理失败: {e}")
            return {}
    else:
        log.info("无模型, 仅输出统计 (probs=None)")
        return {"n_samples": int(len(y)), "attack_ratio": float(y.mean())}

    # 评估
    from collections import Counter
    y_pred = (probs >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_eval == 1)).sum())
    fp = int(((y_pred == 1) & (y_eval == 0)).sum())
    fn = int(((y_pred == 0) & (y_eval == 1)).sum())
    tn = int(((y_pred == 0) & (y_eval == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    f1_normal = tn / max(1, tn + fn) * 2 * (tn / max(1, tn + fn)) / (
        2 * tn / max(1, tn + fn) + tp / max(1, tp + fp) + 1e-9)
    macro_f1 = (f1 + tn / max(1, tn + fn)) / 2

    metrics = {
        "n_samples": int(len(y_eval)),
        "threshold": threshold,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1_attack": f1,
        "f1_normal": tn / max(1, tn + fn),
        "macro_f1": macro_f1,
        "prob_mean": float(probs.mean()),
        "prob_attack_mean": float(probs[y_eval == 1].mean()) if (y_eval == 1).any() else 0,
        "prob_normal_mean": float(probs[y_eval == 0].mean()) if (y_eval == 0).any() else 0,
    }
    log.info("=" * 50)
    log.info("EVALUATION METRICS:")
    for k, v in metrics.items():
        log.info(f"  {k:20s} = {v}")
    log.info("=" * 50)
    return metrics


# ───────────────────────────────────────────────────────────────────
# CLI 主入口
# ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SCADA/Modbus 攻击注入器 + 23-dim 特征工程 + MCU 推理接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate 子命令
    p_gen = sub.add_parser("generate", help="生成正常+攻击数据集 (CSV/NPY/JSONL)")
    p_gen.add_argument("--n-normal", type=int, default=5000)
    p_gen.add_argument("--n-attack", type=int, default=1000)
    p_gen.add_argument("--out-dir", default="./attack_test_data")
    p_gen.add_argument("--seed", type=int, default=42)

    # server 子命令
    p_srv = sub.add_parser("server", help="启动 Modbus TCP 仿真服务器,实时注入攻击")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=5020)
    p_srv.add_argument("--inject-cmd", default="5:0.05,4:0.10",
                       help="攻击注入配置, 格式 TYPE:PROB,TYPE:PROB "
                            "(TYPE: 1-7, PROB: 0-1)")

    # infer 子命令
    p_inf = sub.add_parser("infer", help="加载 .npy 跑 TCN+SE 推理评估")
    p_inf.add_argument("--input", required=True, help="X_attack_23dim.npy 路径")
    p_inf.add_argument("--model", default=None, help="PyTorch .pt 模型路径")
    p_inf.add_argument("--threshold", type=float, default=0.5)

    # demo 子命令 (不依赖 pymodbus)
    p_demo = sub.add_parser("demo", help="快速演示: 生成 100 攻击 + JSONL")
    p_demo.add_argument("--out", default="./demo_attack.jsonl")

    args = parser.parse_args()

    if args.cmd == "generate":
        report = generate_dataset(
            n_normal=args.n_normal,
            n_attack=args.n_attack,
            out_dir=args.out_dir,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))

    elif args.cmd == "server":
        # 解析 inject-cmd
        inject = {}
        for pair in args.inject_cmd.split(","):
            k, v = pair.split(":")
            inject[int(k)] = float(v)
        srv = ModbusTCPServer(args.host, args.port, inject)
        srv.start()

    elif args.cmd == "infer":
        infer_and_evaluate(args.input, args.model, args.threshold)

    elif args.cmd == "demo":
        log.info("Running quick demo (100 cycles = 400 rows)...")
        report = generate_dataset(
            n_normal=400,
            n_attack=100,
            out_dir=os.path.dirname(args.out) or ".",
            seed=42,
        )
        log.info("Demo complete!")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
