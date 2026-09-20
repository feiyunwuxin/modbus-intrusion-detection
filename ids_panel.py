"""入侵检测 (IDS) 面板 - Tkinter 组件。

布局:
    ┌─ 模型选择 ────────────┐
    │ 模型: [combo ▼] [刷新] │
    │ 状态: ...              │
    │ 阈值: [====●====] 0.50 │
    └────────────────────────┘
    ┌─ 统计 ────────────────┐
    │ 总:0 正常:0 攻击:0 ... │
    └────────────────────────┘
    ┌─ 检测结果 (滚动) ──────┐
    │ Treeview: #/hex/真/预/...│
    └────────────────────────┘
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

from ids import FEATURE_COLUMNS, extract_features, list_available_models, load_model

if TYPE_CHECKING:
    from ids.model_loader import ModelWrapper


_MAX_RESULTS = 1000


class IDsPanel(ttk.Frame):
    """右侧入侵检测面板。"""

    def __init__(self, parent: tk.Widget, app=None):
        super().__init__(parent, padding=5)
        self._app = app  # ModbusSimulatorApp reference (for stats etc.)
        self.wrapper: ModelWrapper | None = None
        self.threshold: float = 0.5
        self._stats = {"total": 0, "normal": 0, "attack": 0, "correct": 0}
        self._build_ui()
        self._refresh_models()

    # ---- UI 构建 ----

    def _build_ui(self) -> None:
        self._build_model_section()
        self._build_stats_section()
        self._build_results_section()

    def _build_model_section(self) -> None:
        frame = ttk.LabelFrame(self, text="模型", padding=5)
        frame.pack(fill="x", pady=(0, 5))
        ttk.Label(frame, text="模型:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            frame, textvariable=self.model_var, state="readonly", width=40
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        ttk.Button(frame, text="刷新", command=self._refresh_models).grid(
            row=0, column=2
        )
        # 状态
        ttk.Label(frame, text="状态:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.status_var = tk.StringVar(value="未加载模型")
        ttk.Label(frame, textvariable=self.status_var, foreground="gray").grid(
            row=1, column=1, columnspan=2, sticky="w", padx=(5, 0), pady=(5, 0)
        )
        # 阈值
        ttk.Label(frame, text="阈值:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(
            frame, from_=0.0, to=1.0, variable=self.threshold_var,
            orient="horizontal", command=self._on_threshold_change
        )
        self.threshold_scale.grid(row=2, column=1, sticky="ew", padx=(5, 5), pady=(5, 0))
        self.threshold_label = ttk.Label(frame, text="0.50")
        self.threshold_label.grid(row=2, column=2, pady=(5, 0))
        frame.columnconfigure(1, weight=1)

    def _build_stats_section(self) -> None:
        frame = ttk.LabelFrame(self, text="统计", padding=5)
        frame.pack(fill="x", pady=(0, 5))
        self.stats_var = tk.StringVar(value="总:0  正常:0  攻击:0  准确率:-")
        ttk.Label(frame, textvariable=self.stats_var, font=("Consolas", 10)).pack()

    def _build_results_section(self) -> None:
        frame = ttk.LabelFrame(self, text="检测结果", padding=5)
        frame.pack(fill="both", expand=True)
        cols = ("idx", "hex", "truth", "pred", "prob", "ok")
        self.results_tree = ttk.Treeview(
            frame, columns=cols, show="headings", height=15
        )
        for col, label, width in [
            ("idx", "#", 40), ("hex", "帧前8B", 110),
            ("truth", "真", 40), ("pred", "预", 40),
            ("prob", "概率", 70), ("ok", "✓/✗", 40),
        ]:
            self.results_tree.heading(col, text=label)
            self.results_tree.column(col, width=width, anchor="center")
        # 颜色标签
        self.results_tree.tag_configure("normal", foreground="#006633")
        self.results_tree.tag_configure("attack", foreground="#CC0033")
        self.results_tree.tag_configure("error", foreground="#888888")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ---- 模型管理 ----

    def _refresh_models(self) -> None:
        from pathlib import Path
        project_dir = Path(__file__).parent
        paths = list_available_models(str(project_dir))
        self._available_paths = paths
        names = [Path(p).name for p in paths]
        self.model_combo["values"] = names
        if names and not self.model_var.get():
            self.model_combo.current(0)
            self._on_model_selected()

    def _on_model_selected(self, event=None) -> None:
        from pathlib import Path
        idx = self.model_combo.current()
        if idx < 0 or idx >= len(self._available_paths):
            return
        path = self._available_paths[idx]
        try:
            wrapper = load_model(path)
            self.wrapper = wrapper
            name = Path(path).name
            self.status_var.set(f"已加载 {name} ({wrapper.input_features} features)")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            self.wrapper = None
            self.status_var.set(f"加载失败: {e}")
            messagebox.showerror("模型加载失败", str(e))

    # ---- 阈值 ----

    def _on_threshold_change(self, value) -> None:
        self.threshold = float(value)
        self.threshold_label.configure(text=f"{self.threshold:.2f}")

    # ---- 推理接口 ----

    def process_frame(self, record: dict, frame_index: int) -> None:
        """每帧调用一次。已有模型 → 推理；无模型 → 跳过。"""
        if self.wrapper is None:
            return
        truth = int(record.get("binary", 0))
        hex_bytes = self._hex_preview(record)
        try:
            features = extract_features(record)
            label, prob = self.wrapper.infer(features)
            # 重新应用当前阈值
            label = 1 if prob >= self.threshold else 0
            correct = "✓" if label == truth else "✗"
            tag = "attack" if label == 1 else "normal"
            prob_str = f"{prob:.3f}"
        except Exception as e:
            label, prob, correct, tag, prob_str = -1, 0.0, "?", "error", f"ERR"
            print(f"[IDS] 推理失败 (frame {frame_index}): {e}")

        # 更新列表
        self.results_tree.insert(
            "", "end",
            values=(frame_index + 1, hex_bytes, truth, label, prob_str, correct),
            tags=(tag,)
        )
        # LRU 截断
        children = self.results_tree.get_children()
        if len(children) > _MAX_RESULTS:
            for c in children[:len(children) - _MAX_RESULTS]:
                self.results_tree.delete(c)
        # 更新统计
        self._update_stats(label, truth, label >= 0)

    def clear(self) -> None:
        for c in self.results_tree.get_children():
            self.results_tree.delete(c)
        self._stats = {"total": 0, "normal": 0, "attack": 0, "correct": 0}
        self._render_stats()

    def _update_stats(self, pred: int, truth: int, counted: bool) -> None:
        if not counted:
            return
        self._stats["total"] += 1
        if pred == 1:
            self._stats["attack"] += 1
        else:
            self._stats["normal"] += 1
        if pred == truth:
            self._stats["correct"] += 1
        self._render_stats()

    def _render_stats(self) -> None:
        s = self._stats
        if s["total"] == 0:
            acc_str = "-"
        else:
            acc = s["correct"] / s["total"]
            acc_str = f"{acc:.3f}"
        self.stats_var.set(
            f"总:{s['total']}  正常:{s['normal']}  攻击:{s['attack']}  准确率:{acc_str}"
        )

    @staticmethod
    def _hex_preview(record: dict) -> str:
        """从 record 取前 4 字段拼成 8 字节 hex 预览。"""
        from modbus_simulator import build_frame, frame_to_hex
        try:
            frame = build_frame(record)
            return frame_to_hex(frame[:4])  # 4 bytes = 8 hex chars + 3 spaces
        except Exception:
            return "??"