"""Modbus 模拟器 - 主程序入口。

CLI 模式:
    python modbus_simulator.py [data_file]   # 启动 GUI
    python modbus_simulator.py --self-test   # 自检模式
"""
import csv
import struct
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
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


def _self_test() -> int:
    """打印前 5 条记录的 hex 帧用于验证。"""
    default_csv = Path(__file__).parent / "IanArffDataset.csv"
    if not default_csv.exists():
        print(f"ERROR: 默认数据文件不存在: {default_csv}")
        return 1
    records = load_records(str(default_csv))
    print(f"Loaded {len(records)} records from {default_csv.name}")
    print("-" * 60)
    for i, rec in enumerate(records[:5]):
        frame = build_frame(rec)
        hex_str = frame_to_hex(frame)
        tag = "M" if rec["command"] == 1 else "S"
        print(f"[{i+1}] [{tag}] {hex_str}")
    print("-" * 60)
    print(f"Frame length: {len(build_frame(records[0]))} bytes")
    return 0


class ModbusSimulatorApp:
    """Modbus 主从通信模拟器 GUI。"""

    COLOR_MASTER = "#0066CC"
    COLOR_SLAVE = "#CC0033"

    def __init__(self, root: tk.Tk, initial_file: str | None = None):
        self.root = root
        self.root.title("Modbus 模拟器")
        self.root.geometry("900x650")

        self.records: list[dict] = []
        self.index: int = 0
        self.current_file: str | None = None
        self._auto_after_id: str | None = None
        self._auto_enabled: bool = False
        self._last_valid_interval: int = 500

        self._build_ui()

        if initial_file:
            self._load_file(initial_file)

    def _build_ui(self) -> None:
        """构建完整 UI。"""
        self._build_top_bar()
        self._build_control_bar()
        self._build_log_area()
        self._build_detail_panel()

    def _build_top_bar(self) -> None:
        frame = ttk.Frame(self.root, padding=5)
        frame.pack(fill="x")
        ttk.Label(frame, text="Modbus 模拟器", font=("Microsoft YaHei", 12, "bold")).pack(side="left")
        ttk.Label(frame, text="    记录总数:").pack(side="left")
        self.total_label = ttk.Label(frame, text="0")
        self.total_label.pack(side="left")
        # 第二行
        file_frame = ttk.Frame(self.root, padding=(5, 0, 5, 5))
        file_frame.pack(fill="x")
        ttk.Label(file_frame, text="当前文件:").pack(side="left")
        self.file_label = ttk.Label(file_frame, text="未加载文件", foreground="gray")
        self.file_label.pack(side="left", padx=(0, 10))
        ttk.Button(file_frame, text="打开文件...", command=self._on_open_file).pack(side="left")

    def _build_control_bar(self) -> None:
        frame = ttk.Frame(self.root, padding=(5, 0, 5, 5))
        frame.pack(fill="x")
        ttk.Label(frame, text="已发送:").pack(side="left")
        self.sent_label = ttk.Label(frame, text="0")
        self.sent_label.pack(side="left", padx=(0, 10))
        ttk.Button(frame, text="发送下一条", command=self._on_send_next).pack(side="left", padx=(0, 5))
        ttk.Button(frame, text="重置", command=self._on_reset).pack(side="left", padx=(0, 15))
        # 自动发送
        self.auto_btn = ttk.Button(frame, text="自动发送: 关", command=self._on_toggle_auto)
        self.auto_btn.pack(side="left", padx=(0, 10))
        ttk.Label(frame, text="间隔:").pack(side="left")
        self.interval_var = tk.StringVar(value="500")
        self.interval_spin = ttk.Spinbox(
            frame, from_=1, to=10000, width=6, textvariable=self.interval_var,
            command=self._on_interval_change
        )
        self.interval_spin.pack(side="left", padx=(0, 3))
        ttk.Label(frame, text="ms (1-10000)").pack(side="left")

    def _build_log_area(self) -> None:
        frame = ttk.LabelFrame(self.root, text="通信日志", padding=5)
        frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text = tk.Text(
            frame, wrap="none", font=("Consolas", 10),
            state="disabled", bg="#1E1E1E", fg="#E0E0E0"
        )
        scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        # 配置 M/S 标签颜色 tag
        self.log_text.tag_configure("M", foreground=self.COLOR_MASTER)
        self.log_text.tag_configure("S", foreground=self.COLOR_SLAVE)
        self.log_text.tag_configure("timestamp", foreground="#888888")

    def _build_detail_panel(self) -> None:
        frame = ttk.LabelFrame(self.root, text="当前帧详情", padding=5)
        frame.pack(fill="x", padx=5, pady=(0, 5))
        self.detail_text = tk.Text(frame, height=3, font=("Consolas", 9), state="disabled",
                                    bg="#F5F5F5")
        self.detail_text.pack(fill="x")

    # ---- 占位方法（后续任务实现） ----
    def _on_open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("数据文件", "*.xlsx *.csv"),
                ("Excel", "*.xlsx"),
                ("CSV", "*.csv"),
            ],
        )
        if not path:
            return  # 用户取消
        self._stop_auto()  # 切换文件前关闭自动发送
        self._load_file(path)

    def _on_send_next(self) -> None:
        """发送当前索引对应的记录，索引 +1。"""
        if not self.records:
            messagebox.showinfo("提示", "请先加载数据文件")
            return
        if self.index >= len(self.records):
            return  # 已在末尾
        record = self.records[self.index]
        frame = build_frame(record)
        hex_str = frame_to_hex(frame)
        tag = "M" if record["command"] == 1 else "S"
        self._append_log(tag, hex_str)
        self._update_detail(record, hex_str)
        self.index += 1
        self.sent_label.configure(text=str(self.index))

    def _append_log(self, tag: str, hex_str: str) -> None:
        """追加一条日志行：[HH:MM:SS] [M/S] HEX"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{tag}] {hex_str}\n"
        self.log_text.configure(state="normal")
        # 先写时间戳（灰色），再写 M/S 标签（颜色），再写 hex（默认色）
        self.log_text.insert("end", f"[{ts}] ", "timestamp")
        self.log_text.insert("end", f"[{tag}] ", tag)
        self.log_text.insert("end", f"{hex_str}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_detail(self, record: dict, hex_str: str) -> None:
        """更新详情面板。"""
        from datetime import datetime
        try:
            ts_str = datetime.fromtimestamp(int(record["time"])).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            ts_str = str(record["time"])
        cmd = int(record["command"])
        role = "Master" if cmd == 1 else "Slave"
        detail = (
            f"address={record['address']}  function={record['function']}  "
            f"length={record['length']}  crc={record['crc']}\n"
            f"command={cmd} [{role}]  time={ts_str} ({record['time']})"
        )
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", detail)
        self.detail_text.configure(state="disabled")

    def _on_reset(self, clear_log: bool = True) -> None:
        """重置索引和日志（可不清空日志，用于加载新文件时）。"""
        self._stop_auto()
        self.index = 0
        self.sent_label.configure(text="0")
        if clear_log:
            self._clear_log()
        self._clear_detail()

    def _on_toggle_auto(self) -> None:
        if not self.records:
            messagebox.showinfo("提示", "请先加载数据文件")
            return
        if self._auto_enabled:
            self._stop_auto()
        else:
            self._start_auto()

    def _start_auto(self) -> None:
        interval = self._validate_interval()
        self._auto_enabled = True
        self.auto_btn.configure(text="自动发送: 开")
        self._schedule_next_tick(interval)

    def _schedule_next_tick(self, interval_ms: int) -> None:
        if not self._auto_enabled:
            return
        self._auto_after_id = self.root.after(interval_ms, self._auto_tick)

    def _auto_tick(self) -> None:
        """自动发送 tick：发一条 + 调度下一条。"""
        if not self._auto_enabled:
            return
        if self.index >= len(self.records):
            self._stop_auto()
            return
        self._on_send_next()
        interval = self._validate_interval()
        self._schedule_next_tick(interval)

    def _validate_interval(self) -> int:
        """读取并校验间隔输入，返回合法值（非法回退到上次合法值）。"""
        try:
            v = int(self.interval_var.get())
            if 1 <= v <= 10000:
                self._last_valid_interval = v
                return v
        except ValueError:
            pass
        # 回退
        self.interval_var.set(str(self._last_valid_interval))
        return self._last_valid_interval

    def _on_interval_change(self) -> None:
        """Spinbox 值变更：实时校验，更新最后合法值（不影响正在运行的 after）。"""
        try:
            v = int(self.interval_var.get())
            if 1 <= v <= 10000:
                self._last_valid_interval = v
        except ValueError:
            pass

    def _stop_auto(self) -> None:
        """停止自动发送。"""
        if self._auto_after_id is not None:
            try:
                self.root.after_cancel(self._auto_after_id)
            except Exception:
                pass
            self._auto_after_id = None
        if self._auto_enabled:
            self._auto_enabled = False
            self.auto_btn.configure(text="自动发送: 关")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _clear_detail(self) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.configure(state="disabled")

    def _load_file(self, path: str) -> None:
        """加载数据文件，更新 UI 状态。失败时弹窗并保持当前状态。"""
        try:
            records = load_records(path)
        except FileNotFoundError:
            messagebox.showerror("加载失败", f"文件不存在:\n{path}")
            return
        except RuntimeError as e:
            messagebox.showinfo("缺少依赖", str(e))
            return
        except ValueError as e:
            messagebox.showerror("加载失败", str(e))
            return
        except Exception as e:
            messagebox.showerror("加载失败", f"未知错误: {e}")
            return
        # 成功：更新状态
        self.records = records
        self.index = 0
        self.current_file = path
        self._on_reset(clear_log=False)
        self.file_label.configure(text=path, foreground="black")
        self.total_label.configure(text=str(len(records)))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return _self_test()
    root = tk.Tk()
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = ModbusSimulatorApp(root, initial_file=initial)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
