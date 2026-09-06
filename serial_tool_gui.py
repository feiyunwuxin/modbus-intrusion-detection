#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serial_tool_gui.py
==================
SCADA Modbus 串口 + ARFF 联动 GUI 工具

设计目标:
  1. 选择 PC 串口 (COM1-COM256) + 波特率 (1200-921600)
  2. 收发 HEX/ASCII 数据 (类似 XCOM / Putty)
  3. 联动 arff_modbus_simulator.py: 启动/停止 ARFF 回放服务器
  4. 内置 Modbus TCP 客户端 (FC=3/6/16), 直连 ARFF Server
  5. 一站式调试: MCU 串口输出 + PC Modbus 回放

依赖:
  pip install pyserial

用法:
  python serial_tool_gui.py
  # 或双击 serial_tool_gui.bat

Author: SCADA IDS Team
Date: 2026-09-06
"""

import os
import sys
import time
import queue
import struct
import socket
import threading
import subprocess
from typing import Optional

# ───────────────────────────────────────────────────────────────────
# 依赖检查
# ───────────────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("错误: tkinter 不可用 (Python 3.13+ 已默认移除)")
    print("解决: 重新安装 Python 时勾选 tcl/tk, 或使用 python -m tkinter 测试")
    sys.exit(1)

try:
    import serial  # pyserial
    import serial.tools.list_ports
except ImportError:
    # 不立即退出, 让 GUI 起来后再提示
    serial = None

# ───────────────────────────────────────────────────────────────────
# 常量
# ───────────────────────────────────────────────────────────────────
APP_TITLE = "SCADA Modbus 串口工具 v1.0"
APP_WIDTH = 920
APP_HEIGHT = 760

# Modbus 协议 (复用 arff_modbus_simulator 的常量)
MBAP_HEADER_LEN = 7
FC_READ_HOLDING = 0x03
FC_WRITE_SINGLE = 0x06
FC_WRITE_MULTI = 0x10
ILLEGAL_FCS = {43, 73, 90, 100, 136, 137, 138, 139, 140, 171, 172}

BAUDRATES = [1200, 2400, 4800, 9600, 19200, 38400, 57600,
             115200, 230400, 460800, 921600]
DEFAULT_BAUDRATE = 115200

DATA_BITS = [5, 6, 7, 8]
PARITIES = ["None", "Even", "Odd", "Mark", "Space"]
STOP_BITS = [1, 1.5, 2]
DEFAULT_DATA_BITS = 8
DEFAULT_STOP_BITS = 1
DEFAULT_PARITY = "None"

DEFAULT_ARFF_PORT = 5020
DEFAULT_ARFF_RATE = "100x"
ARFF_SIMULATOR_SCRIPT = "arff_modbus_simulator.py"


# ───────────────────────────────────────────────────────────────────
# 辅助函数
# ───────────────────────────────────────────────────────────────────
def hex_str_to_bytes(s: str) -> bytes:
    """HEX 字符串 ('AA 01 02' 或 'AA0102') → bytes."""
    s = s.strip().replace(" ", "").replace("\n", "").replace(",", "")
    if len(s) % 2 != 0:
        raise ValueError("HEX 字符串长度必须为偶数")
    return bytes.fromhex(s)


def bytes_to_hex_str(data: bytes, sep: str = " ") -> str:
    """bytes → HEX 字符串 ('AA 01 02')."""
    return sep.join(f"{b:02X}" for b in data)


def bytes_to_ascii_str(data: bytes) -> str:
    """bytes → ASCII (不可打印字符用 . 替换)."""
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data)


def list_serial_ports() -> list:
    """列出可用串口."""
    if serial is None:
        return []
    return [f"{p.device} - {p.description}" for p in serial.tools.list_ports.comports()]


# ───────────────────────────────────────────────────────────────────
# 主应用
# ───────────────────────────────────────────────────────────────────
class SerialToolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.root.minsize(800, 600)

        # 状态
        self.ser: Optional[serial.Serial] = None
        self.rx_thread: Optional[threading.Thread] = None
        self.rx_queue: queue.Queue = queue.Queue()
        self.rx_count = 0
        self.tx_count = 0
        self.err_count = 0
        self.start_time = time.time()
        self.auto_send_job = None
        self.show_hex = tk.BooleanVar(value=True)
        self.show_ascii = tk.BooleanVar(value=True)
        self.auto_scroll = tk.BooleanVar(value=True)

        # ARFF 子进程
        self.arff_proc: Optional[subprocess.Popen] = None

        # Modbus TCP 客户端
        self.mb_sock: Optional[socket.socket] = None
        self.mb_connected = False

        self._build_ui()
        self._refresh_ports()
        self._update_clock()
        self._poll_rx_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ────────────────────────────────────────────────────────────
    # UI 构建
    # ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # 主容器
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # === 第 1 区: 串口设置 ===
        serial_frame = ttk.LabelFrame(main, text=" 串口设置 ", padding=8)
        serial_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(serial_frame, text="端口:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var,
                                        width=28, state="readonly")
        self.port_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 12))

        ttk.Label(serial_frame, text="波特率:").grid(row=0, column=2, sticky=tk.W, padx=(0, 4))
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUDRATE))
        self.baud_combo = ttk.Combobox(serial_frame, textvariable=self.baud_var,
                                       values=[str(b) for b in BAUDRATES],
                                       width=10, state="readonly")
        self.baud_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 12))

        ttk.Label(serial_frame, text="数据位:").grid(row=0, column=4, sticky=tk.W, padx=(0, 4))
        self.data_bits_var = tk.StringVar(value=str(DEFAULT_DATA_BITS))
        ttk.Combobox(serial_frame, textvariable=self.data_bits_var,
                     values=[str(d) for d in DATA_BITS],
                     width=4, state="readonly").grid(row=0, column=5, padx=(0, 8))

        ttk.Label(serial_frame, text="停止位:").grid(row=0, column=6, sticky=tk.W, padx=(0, 4))
        self.stop_bits_var = tk.StringVar(value=str(DEFAULT_STOP_BITS))
        ttk.Combobox(serial_frame, textvariable=self.stop_bits_var,
                     values=[str(s) for s in STOP_BITS],
                     width=4, state="readonly").grid(row=0, column=7, padx=(0, 8))

        ttk.Label(serial_frame, text="校验:").grid(row=0, column=8, sticky=tk.W, padx=(0, 4))
        self.parity_var = tk.StringVar(value=DEFAULT_PARITY)
        ttk.Combobox(serial_frame, textvariable=self.parity_var,
                     values=PARITIES, width=6, state="readonly").grid(row=0, column=9, padx=(0, 8))

        ttk.Button(serial_frame, text="刷新端口",
                   command=self._refresh_ports).grid(row=0, column=10, padx=(0, 4))
        self.connect_btn = ttk.Button(serial_frame, text="连接", width=8,
                                       command=self._on_connect)
        self.connect_btn.grid(row=0, column=11, padx=(0, 4))
        self.disconnect_btn = ttk.Button(serial_frame, text="断开", width=8,
                                         command=self._on_disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=12, padx=(0, 4))

        # 显示选项
        opt_frame = ttk.Frame(serial_frame)
        opt_frame.grid(row=1, column=0, columnspan=13, sticky=tk.W, pady=(8, 0))
        ttk.Checkbutton(opt_frame, text="HEX", variable=self.show_hex).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(opt_frame, text="ASCII", variable=self.show_ascii).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(opt_frame, text="自动滚动", variable=self.auto_scroll).pack(side=tk.LEFT, padx=(0, 8))

        # === 第 2 区: Modbus / ARFF 联动 ===
        mb_frame = ttk.LabelFrame(main, text=" Modbus / ARFF 联动 ", padding=8)
        mb_frame.pack(fill=tk.X, pady=(0, 6))

        # ARFF Server 控制行
        ttk.Label(mb_frame, text="ARFF Server:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.arff_status_var = tk.StringVar(value="● 未启动")
        ttk.Label(mb_frame, textvariable=self.arff_status_var, foreground="grey",
                  width=14).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(mb_frame, text="端口:").grid(row=0, column=2, padx=(0, 4))
        self.arff_port_var = tk.StringVar(value=str(DEFAULT_ARFF_PORT))
        ttk.Entry(mb_frame, textvariable=self.arff_port_var, width=6).grid(row=0, column=3, padx=(0, 8))
        ttk.Label(mb_frame, text="速率:").grid(row=0, column=4, padx=(0, 4))
        self.arff_rate_var = tk.StringVar(value=DEFAULT_ARFF_RATE)
        ttk.Combobox(mb_frame, textvariable=self.arff_rate_var,
                     values=["100x", "10x", "1x", "0"], width=6).grid(row=0, column=5, padx=(0, 8))
        self.arff_start_btn = ttk.Button(mb_frame, text="启动", width=6,
                                          command=self._on_arff_start)
        self.arff_start_btn.grid(row=0, column=6, padx=(0, 4))
        self.arff_stop_btn = ttk.Button(mb_frame, text="停止", width=6,
                                         command=self._on_arff_stop, state=tk.DISABLED)
        self.arff_stop_btn.grid(row=0, column=7, padx=(0, 12))

        # Modbus 客户端行
        ttk.Label(mb_frame, text="Modbus 客户端:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=(8, 0))
        ttk.Label(mb_frame, text="IP:").grid(row=1, column=1, sticky=tk.E, padx=(0, 4), pady=(8, 0))
        self.mb_ip_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(mb_frame, textvariable=self.mb_ip_var, width=14).grid(row=1, column=2, padx=(0, 8), pady=(8, 0))
        ttk.Label(mb_frame, text="Port:").grid(row=1, column=3, sticky=tk.E, padx=(0, 4), pady=(8, 0))
        self.mb_port_var = tk.StringVar(value=str(DEFAULT_ARFF_PORT))
        ttk.Entry(mb_frame, textvariable=self.mb_port_var, width=6).grid(row=1, column=4, padx=(0, 8), pady=(8, 0))
        self.mb_connect_btn = ttk.Button(mb_frame, text="连接", width=6,
                                          command=self._on_mb_connect)
        self.mb_connect_btn.grid(row=1, column=5, padx=(0, 4), pady=(8, 0))
        self.mb_disconnect_btn = ttk.Button(mb_frame, text="断开", width=6,
                                             command=self._on_mb_disconnect, state=tk.DISABLED)
        self.mb_disconnect_btn.grid(row=1, column=6, padx=(0, 12), pady=(8, 0))

        # FC 选择 + 发送
        ttk.Label(mb_frame, text="FC:").grid(row=2, column=0, sticky=tk.W, padx=(0, 4), pady=(8, 0))
        self.mb_fc_var = tk.StringVar(value="3")
        ttk.Combobox(mb_frame, textvariable=self.mb_fc_var,
                     values=["3", "6", "16"], width=4, state="readonly").grid(row=2, column=1, padx=(0, 8), pady=(8, 0))
        ttk.Label(mb_frame, text="起始:").grid(row=2, column=2, sticky=tk.E, padx=(0, 4), pady=(8, 0))
        self.mb_addr_var = tk.StringVar(value="0")
        ttk.Entry(mb_frame, textvariable=self.mb_addr_var, width=6).grid(row=2, column=3, padx=(0, 8), pady=(8, 0))
        ttk.Label(mb_frame, text="数量:").grid(row=2, column=4, sticky=tk.E, padx=(0, 4), pady=(8, 0))
        self.mb_count_var = tk.StringVar(value="25")
        ttk.Entry(mb_frame, textvariable=self.mb_count_var, width=6).grid(row=2, column=5, padx=(0, 8), pady=(8, 0))
        ttk.Label(mb_frame, text="Unit:").grid(row=2, column=6, sticky=tk.E, padx=(0, 4), pady=(8, 0))
        self.mb_unit_var = tk.StringVar(value="4")
        ttk.Entry(mb_frame, textvariable=self.mb_unit_var, width=4).grid(row=2, column=7, padx=(0, 8), pady=(8, 0))
        ttk.Button(mb_frame, text="发送请求", width=10,
                   command=self._on_mb_send).grid(row=2, column=8, padx=(4, 0), pady=(8, 0))

        # === 第 3 区: 接收区 ===
        rx_frame = ttk.LabelFrame(main, text=" 接收区 (串口 + Modbus) ", padding=4)
        rx_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.rx_text = scrolledtext.ScrolledText(rx_frame, wrap=tk.NONE, height=12,
                                                  font=("Consolas", 9), bg="#1e1e1e",
                                                  fg="#d4d4d4", insertbackground="white")
        self.rx_text.pack(fill=tk.BOTH, expand=True)
        self.rx_text.tag_config("rx_serial", foreground="#9cdcfe")
        self.rx_text.tag_config("rx_mb_req", foreground="#ce9178")
        self.rx_text.tag_config("rx_mb_resp", foreground="#b5cea8")
        self.rx_text.tag_config("rx_err", foreground="#f48771")
        self.rx_text.tag_config("rx_info", foreground="#808080")

        # === 第 4 区: 发送区 ===
        tx_frame = ttk.LabelFrame(main, text=" 发送区 (HEX 字符串, 如 AA 01 02 03) ", padding=4)
        tx_frame.pack(fill=tk.X, pady=(0, 6))
        self.tx_text = tk.Text(tx_frame, height=3, font=("Consolas", 9),
                                bg="#2d2d2d", fg="#d4d4d4", insertbackground="white")
        self.tx_text.pack(fill=tk.X, expand=True)

        tx_btn_frame = ttk.Frame(tx_frame)
        tx_btn_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(tx_btn_frame, text="发送", width=8, command=self._on_send).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(tx_btn_frame, text="清空发送区", width=10,
                   command=lambda: self.tx_text.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(tx_btn_frame, text="清空接收区", width=10,
                   command=lambda: self.rx_text.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(0, 12))
        self.auto_send_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tx_btn_frame, text="自动发送", variable=self.auto_send_var,
                        command=self._toggle_auto_send).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(tx_btn_frame, text="间隔:").pack(side=tk.LEFT)
        self.auto_send_ms = tk.StringVar(value="1000")
        ttk.Entry(tx_btn_frame, textvariable=self.auto_send_ms, width=6).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Label(tx_btn_frame, text="ms").pack(side=tk.LEFT)

        # === 第 5 区: 状态栏 ===
        self.status_var = tk.StringVar(value="就绪 | pyserial: " + ("OK" if serial else "未安装"))
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ────────────────────────────────────────────────────────────
    # 串口操作
    # ────────────────────────────────────────────────────────────
    def _refresh_ports(self):
        if serial is None:
            self.port_combo["values"] = ["(pyserial 未安装)"]
            return
        ports = list_serial_ports()
        if not ports:
            ports = ["(未发现串口)"]
        self.port_combo["values"] = ports
        if ports:
            self.port_combo.current(0)

    def _on_connect(self):
        if serial is None:
            messagebox.showerror("错误", "pyserial 未安装\n\n请运行: pip install pyserial")
            return
        if self.ser and self.ser.is_open:
            return
        port_str = self.port_var.get().split(" - ")[0]
        if not port_str or port_str.startswith("("):
            messagebox.showwarning("提示", "请先选择有效串口")
            return

        # 检查串口是否真实存在
        available = [p.device for p in serial.tools.list_ports.comports()]
        if port_str not in available:
            messagebox.showerror("串口不存在",
                                f"{port_str} 不在系统可用串口列表中\n\n"
                                f"当前可用: {', '.join(available) if available else '(无)'}\n\n"
                                f"解决:\n"
                                f"  1. 检查 USB 是否插好\n"
                                f"  2. 设备管理器查看端口号是否变化\n"
                                f"  3. 点击 [刷新端口] 重新扫描")
            self.err_count += 1
            return

        try:
            self.ser = serial.Serial(
                port=port_str,
                baudrate=int(self.baud_var.get()),
                bytesize=int(self.data_bits_var.get()),
                parity={"None": "N", "Even": "E", "Odd": "O",
                        "Mark": "M", "Space": "S"}[self.parity_var.get()],
                stopbits=float(self.stop_bits_var.get()),
                timeout=0.1,
            )
        except serial.SerialException as e:
            # 解析常见错误并给出解决建议
            err = str(e)
            if "PermissionError" in err or "Access is denied" in err:
                msg = (f"{port_str} 被其他程序占用!\n\n"
                       f"常见占用者:\n"
                       f"  • Arduino IDE 串口监视器 (最常见)\n"
                       f"  • PlatformIO 串口监视器\n"
                       f"  • PuTTY / XCOM / 串口调试助手\n"
                       f"  • STM32CubeIDE / Keil 调试器\n"
                       f"  • 上次未关闭的串口工具\n\n"
                       f"解决步骤:\n"
                       f"  1. 关闭 Arduino IDE 等工具的串口监视器\n"
                       f"  2. 重新插入 USB 线\n"
                       f"  3. 再次点击 [连接]\n\n"
                       f"原始错误: {err}")
            elif "FileNotFoundError" in err or "cannot find" in err.lower():
                msg = (f"{port_str} 已被系统移除\n\n"
                       f"可能原因: USB 设备刚拔出,或驱动异常\n\n"
                       f"解决:\n"
                       f"  1. 重新插入 USB\n"
                       f"  2. 点击 [刷新端口]\n\n"
                       f"原始错误: {err}")
            else:
                msg = f"无法打开 {port_str}:\n{err}"
            messagebox.showerror("连接失败", msg)
            self.err_count += 1
            return
        except Exception as e:
            messagebox.showerror("连接失败", f"无法打开 {port_str}:\n{e}")
            self.err_count += 1
            return

        # 启动接收线程
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.NORMAL)
        self._rx_append(f"=== 连接 {port_str} @ {self.baud_var.get()} ===\n", "rx_info")
        self._update_status()

    def _on_disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self._rx_append("=== 已断开 ===\n", "rx_info")
        self._update_status()

    def _rx_loop(self):
        """串口接收循环 (后台线程)."""
        while self.ser and self.ser.is_open:
            try:
                data = self.ser.read(256)
                if data:
                    self.rx_queue.put(("serial", data))
                    self.rx_count += len(data)
            except Exception as e:
                self.rx_queue.put(("err", f"接收错误: {e}".encode("utf-8")))
                self.err_count += 1
                break

    # ────────────────────────────────────────────────────────────
    # ARFF Server 子进程控制
    # ────────────────────────────────────────────────────────────
    def _on_arff_start(self):
        if self.arff_proc and self.arff_proc.poll() is None:
            messagebox.showinfo("提示", "ARFF Server 已在运行")
            return
        if not os.path.exists(ARFF_SIMULATOR_SCRIPT):
            messagebox.showerror("错误", f"找不到脚本: {ARFF_SIMULATOR_SCRIPT}")
            return
        port = self.arff_port_var.get()
        rate = self.arff_rate_var.get()
        try:
            # 在新窗口运行 (Windows 下用 CREATE_NEW_CONSOLE)
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_CONSOLE
            self.arff_proc = subprocess.Popen(
                [sys.executable, ARFF_SIMULATOR_SCRIPT, "start",
                 "--port", port, "--rate", rate],
                creationflags=creationflags,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            messagebox.showerror("启动失败", f"无法启动 ARFF Server:\n{e}")
            return
        self.arff_start_btn.config(state=tk.DISABLED)
        self.arff_stop_btn.config(state=tk.NORMAL)
        self.arff_status_var.set(f"● 运行中 (PID {self.arff_proc.pid})")
        self._rx_append(f"=== ARFF Server 启动: port={port} rate={rate} PID={self.arff_proc.pid} ===\n", "rx_info")
        # 启动日志监控线程
        threading.Thread(target=self._arff_log_loop, daemon=True).start()

    def _on_arff_stop(self):
        if self.arff_proc and self.arff_proc.poll() is None:
            self.arff_proc.terminate()
            try:
                self.arff_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.arff_proc.kill()
        self.arff_proc = None
        self.arff_start_btn.config(state=tk.NORMAL)
        self.arff_stop_btn.config(state=tk.DISABLED)
        self.arff_status_var.set("● 未启动")
        self._rx_append("=== ARFF Server 已停止 ===\n", "rx_info")

    def _arff_log_loop(self):
        """捕获 ARFF 子进程输出到接收区."""
        if not self.arff_proc:
            return
        try:
            for line in iter(self.arff_proc.stdout.readline, b""):
                if not line:
                    break
                self.rx_queue.put(("arff", line))
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────
    # Modbus 客户端
    # ────────────────────────────────────────────────────────────
    def _on_mb_connect(self):
        if self.mb_connected:
            return
        try:
            self.mb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.mb_sock.settimeout(3.0)
            self.mb_sock.connect((self.mb_ip_var.get(), int(self.mb_port_var.get())))
        except Exception as e:
            messagebox.showerror("Modbus 连接失败", f"{e}")
            self.mb_sock = None
            self.err_count += 1
            return
        self.mb_connected = True
        self.mb_connect_btn.config(state=tk.DISABLED)
        self.mb_disconnect_btn.config(state=tk.NORMAL)
        self._rx_append(f"=== Modbus 连接 {self.mb_ip_var.get()}:{self.mb_port_var.get()} ===\n", "rx_info")
        self._update_status()

    def _on_mb_disconnect(self):
        if self.mb_sock:
            try:
                self.mb_sock.close()
            except Exception:
                pass
        self.mb_sock = None
        self.mb_connected = False
        self.mb_connect_btn.config(state=tk.NORMAL)
        self.mb_disconnect_btn.config(state=tk.DISABLED)
        self._rx_append("=== Modbus 已断开 ===\n", "rx_info")
        self._update_status()

    def _on_mb_send(self):
        if not self.mb_connected or not self.mb_sock:
            messagebox.showwarning("提示", "请先连接 Modbus 服务器")
            return
        try:
            fc = int(self.mb_fc_var.get())
            start_addr = int(self.mb_addr_var.get())
            count = int(self.mb_count_var.get())
            unit_id = int(self.mb_unit_var.get())
        except ValueError:
            messagebox.showerror("错误", "FC/起始/数量/Unit 必须是整数")
            return

        if fc not in (FC_READ_HOLDING, FC_WRITE_SINGLE, FC_WRITE_MULTI):
            messagebox.showerror("错误", f"不支持 FC={fc}")
            return

        try:
            # 构造请求
            pdu = struct.pack(">BHH", fc, start_addr, count)
            length = len(pdu) + 1
            mbap = struct.pack(">HHHB", 1, 0, length, unit_id)
            req = mbap + pdu
            self.mb_sock.sendall(req)
            self.tx_count += len(req)
            self._rx_append(f"[REQ ] {bytes_to_hex_str(req)}\n", "rx_mb_req")

            # 接收响应
            resp = self.mb_sock.recv(MBAP_HEADER_LEN + 2 + count * 2)
            self.rx_count += len(resp)
            self._rx_append(f"[RESP] {bytes_to_hex_str(resp)}\n", "rx_mb_resp")

            # 解析
            if len(resp) >= MBAP_HEADER_LEN + 2:
                resp_fc = resp[MBAP_HEADER_LEN]
                if resp_fc & 0x80:
                    exc_code = resp[MBAP_HEADER_LEN + 1]
                    self._rx_append(f"  → Modbus Exception code=0x{exc_code:02X}\n", "rx_err")
                elif resp_fc == FC_READ_HOLDING:
                    byte_count = resp[MBAP_HEADER_LEN + 1]
                    regs = []
                    for i in range(byte_count // 2):
                        v = struct.unpack(">H", resp[MBAP_HEADER_LEN + 2 + i * 2:
                                                      MBAP_HEADER_LEN + 2 + i * 2 + 2])[0]
                        regs.append(v)
                    self._rx_append(f"  → 寄存器: {regs}\n", "rx_mb_resp")
                elif resp_fc == FC_WRITE_SINGLE:
                    addr, val = struct.unpack(">HH", resp[MBAP_HEADER_LEN + 1:
                                                          MBAP_HEADER_LEN + 5])
                    self._rx_append(f"  → 写入 OK: addr={addr} value={val}\n", "rx_mb_resp")
        except socket.timeout:
            self._rx_append("  → 超时\n", "rx_err")
            self.err_count += 1
        except Exception as e:
            self._rx_append(f"  → 错误: {e}\n", "rx_err")
            self.err_count += 1
        self._update_status()

    # ────────────────────────────────────────────────────────────
    # 串口发送
    # ────────────────────────────────────────────────────────────
    def _on_send(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("提示", "请先连接串口")
            return
        text = self.tx_text.get("1.0", tk.END).strip()
        if not text:
            return
        try:
            data = hex_str_to_bytes(text)
        except ValueError as e:
            messagebox.showerror("HEX 解析错误", str(e))
            return
        try:
            self.ser.write(data)
            self.tx_count += len(data)
            self._rx_append(f"[TX  ] {bytes_to_hex_str(data)}\n", "rx_serial")
        except Exception as e:
            self._rx_append(f"[TX ERR] {e}\n", "rx_err")
            self.err_count += 1

    def _toggle_auto_send(self):
        if self.auto_send_var.get():
            self._auto_send_tick()
        else:
            if self.auto_send_job:
                self.root.after_cancel(self.auto_send_job)
                self.auto_send_job = None

    def _auto_send_tick(self):
        if not self.auto_send_var.get():
            return
        self._on_send()
        try:
            ms = max(50, int(self.auto_send_ms.get()))
        except ValueError:
            ms = 1000
        self.auto_send_job = self.root.after(ms, self._auto_send_tick)

    # ────────────────────────────────────────────────────────────
    # 接收区更新 (UI 线程, 100ms 轮询)
    # ────────────────────────────────────────────────────────────
    def _poll_rx_queue(self):
        try:
            for _ in range(50):  # 每帧最多处理 50 条
                source, data = self.rx_queue.get_nowait()
                if source == "err":
                    self._rx_append(f"[ERR] {data.decode('utf-8', errors='replace')}\n", "rx_err")
                elif source == "arff":
                    text = data.decode('utf-8', errors='replace').rstrip()
                    if text:
                        self._rx_append(f"[ARFF] {text}\n", "rx_info")
                elif source == "serial":
                    self._append_data(data, "rx_serial")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_rx_queue)

    def _append_data(self, data: bytes, tag: str):
        if self.show_hex.get():
            self.rx_text.insert(tk.END, bytes_to_hex_str(data) + "\n", tag)
        if self.show_ascii.get():
            self.rx_text.insert(tk.END, bytes_to_ascii_str(data) + "\n", tag)
        if self.auto_scroll.get():
            self.rx_text.see(tk.END)

    def _rx_append(self, text: str, tag: str = ""):
        self.rx_text.insert(tk.END, text, tag)
        if self.auto_scroll.get():
            self.rx_text.see(tk.END)

    # ────────────────────────────────────────────────────────────
    # 状态栏
    # ────────────────────────────────────────────────────────────
    def _update_status(self):
        ser_status = "未连接"
        if self.ser and self.ser.is_open:
            ser_status = f"{self.ser.port} @ {self.ser.baudrate}"
        mb_status = "未连接" if not self.mb_connected else f"{self.mb_ip_var.get()}:{self.mb_port_var.get()}"
        self.status_var.set(
            f"串口: {ser_status} | Modbus: {mb_status} | "
            f"RX: {self.rx_count} | TX: {self.tx_count} | ERR: {self.err_count} | "
            f"pyserial: {'OK' if serial else '未安装'}"
        )

    def _update_clock(self):
        elapsed = int(time.time() - self.start_time)
        self.root.after(1000, self._update_clock)
        self._update_status()

    # ────────────────────────────────────────────────────────────
    # 退出
    # ────────────────────────────────────────────────────────────
    def _on_close(self):
        if self.auto_send_job:
            self.root.after_cancel(self.auto_send_job)
        self._on_arff_stop()
        self._on_disconnect()
        self._on_mb_disconnect()
        self.root.destroy()


# ───────────────────────────────────────────────────────────────────
# 入口
# ───────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    try:
        # Windows: 使用更现代的主题
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    SerialToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
