"""入侵检测 (IDS) 面板 - Tkinter 组件。

布局:
    ┌─ 模型选择 ────────────┐
    │ 模型: [combo ▼] [刷新] │
    │ 状态: ...              │
    │ 阈值: [====●====] 0.50 │
    └────────────────────────┘
    ┌─ 统计 ────────────────┐
    │ 总:0 正常:0 攻击:0 正确:0│
    │ 准确率:- 精确率:- 召回率:- F1:- │
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
        self.window_size: int = 1  # 1=单帧；>1=3D 窗口 (LSTM/CNN1d)
        # Path of the model whose predictions are currently in the
        # stats counters. Used by _on_model_selected to detect "user
        # actually switched models" vs "we just re-loaded the same
        # model" — only the former resets the stats. ``None`` until the
        # first successful load.
        self._active_path: str | None = None
        # Confusion-matrix counters for binary classification
        # (positive class = attack / label 1). Total / normal / attack /
        # correct are kept for the existing display; the four tp/fp/tn/fn
        # cells power the new Precision / Recall / F1 metrics.
        self._stats = {
            "total": 0, "normal": 0, "attack": 0, "correct": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        }
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
        # 窗口大小 (1=单帧；>1=3D 窗口模型所需，如 BiLSTM/Conv1d)
        ttk.Label(frame, text="窗口:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.window_size_var = tk.IntVar(value=16)
        self.window_size_spin = ttk.Spinbox(
            frame, from_=1, to=64, increment=1, textvariable=self.window_size_var,
            width=8, command=self._on_window_size_change,
        )
        self.window_size_spin.grid(row=3, column=1, sticky="w", padx=(5, 5), pady=(5, 0))
        ttk.Label(frame, text="(1=单帧；16=BiLSTM 默认)").grid(
            row=3, column=2, sticky="w", pady=(5, 0)
        )
        frame.columnconfigure(1, weight=1)

    def _build_stats_section(self) -> None:
        frame = ttk.LabelFrame(self, text="统计", padding=5)
        frame.pack(fill="x", pady=(0, 5))
        # Two rows: counts on top (existing), classification metrics on
        # bottom (Accuracy / Precision / Recall / F1). Both update on the
        # same _render_stats() tick so they can never disagree.
        self.stats_var = tk.StringVar(value="总:0  正常:0  攻击:0  正确:0")
        ttk.Label(frame, textvariable=self.stats_var, font=("Consolas", 10)).pack(anchor="w")
        self.metrics_var = tk.StringVar(value="准确率:-  精确率:-  召回率:-  F1:-")
        ttk.Label(frame, textvariable=self.metrics_var, font=("Consolas", 10)).pack(anchor="w")

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
        self.results_tree.tag_configure("warmup", foreground="#999999")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ---- 模型管理 ----

    @staticmethod
    def _probe_model(path: str) -> dict:
        """快速预分类模型，返回 {"compatible": bool, "n_features": int|None,
        "kind": str, "window_size": int}。不抛异常。

        kind 取值: "BiLSTM", "GRU", "FNN", "Conv1d", "sklearn-17",
                   "wrong_features", "incompatible_3d", "unknown",
                   "missing_dep", "load_error"。
        """
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return {"compatible": False, "n_features": None,
                    "kind": "missing_dep", "window_size": 1}
        try:
            import joblib
        except ImportError:
            joblib = None
        try:
            import torch
        except ImportError:
            torch = None
        try:
            if p.suffix.lower() == ".joblib" and joblib is not None:
                m = joblib.load(p)
                n = getattr(m, "n_features_in_", None)
                if n == 17:
                    return {"compatible": True, "n_features": 17,
                            "kind": "sklearn-17", "window_size": 1}
                return {"compatible": False, "n_features": n,
                        "kind": "wrong_features", "window_size": 1}
            if p.suffix.lower() == ".pt" and torch is not None:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(obj, dict) and "state_dict" in obj:
                    sd = obj["state_dict"]
                elif isinstance(obj, dict):
                    sd = obj
                else:
                    # nn.Module: 探测第一个 Linear
                    n = None
                    for mod in obj.modules():
                        if isinstance(mod, torch.nn.Linear):
                            n = mod.in_features
                            break
                    if n == 17:
                        return {"compatible": True, "n_features": 17,
                                "kind": "FNN", "window_size": 1}
                    return {"compatible": False, "n_features": n,
                            "kind": "wrong_features", "window_size": 1}
                # state_dict: 探测 lstm/gru/conv 关键字 + 第一层 Linear in_features
                has_lstm = any(k.startswith(("lstm.", "gru.")) for k in sd)
                has_3d = any(hasattr(v, "dim") and v.dim() == 3 for v in sd.values())
                # BiLSTM/GRU：input_size 必须从 weight_ih_l0 读，classifier fc1 形状不可信
                if has_lstm:
                    import re
                    pat = re.compile(r"^(lstm|gru)\.weight_ih_l0(_reverse)?$")
                    in_size = None
                    gate = None
                    has_reverse = False
                    is_lstm_module = False
                    for k, v in sd.items():
                        m = pat.match(k)
                        if m and hasattr(v, "dim") and v.dim() == 2:
                            in_size = int(v.shape[1])
                            gate = int(v.shape[0])
                            mod_name = m.group(1)  # lstm or gru
                            if "_reverse" in k:
                                has_reverse = True
                            if mod_name == "lstm":
                                is_lstm_module = True
                    if in_size is not None and gate is not None:
                        family = "LSTM" if is_lstm_module else "GRU"
                        prefix = "Bi" if has_reverse else ""
                        mod_kind = f"{prefix}{family}"
                        return {"compatible": in_size == 17,
                                "n_features": in_size,
                                "kind": mod_kind if in_size == 17 else "wrong_features",
                                "window_size": 16}
                # 非 RNN：从第一个非-RNN/Conv 的 2D weight 取 in_features
                first_linear_in = None
                for k, v in sd.items():
                    if hasattr(v, "dim") and v.dim() == 2 and not (
                            "lstm" in k.lower() or "gru" in k.lower() or "conv" in k.lower()):
                        first_linear_in = int(v.shape[1])
                        break
                if has_3d:
                    # CNN/Conv1d 风格：需要更细探测；这里仅粗分类
                    return {"compatible": first_linear_in == 17,
                            "n_features": first_linear_in,
                            "kind": "Conv1d" if first_linear_in == 17 else "wrong_features",
                            "window_size": 16}
                if first_linear_in is None:
                    return {"compatible": False, "n_features": None,
                            "kind": "unknown", "window_size": 1}
                if first_linear_in != 17:
                    return {"compatible": False, "n_features": first_linear_in,
                            "kind": "wrong_features", "window_size": 1}
                return {"compatible": True, "n_features": 17,
                        "kind": "FNN", "window_size": 1}
            return {"compatible": False, "n_features": None,
                    "kind": "missing_dep", "window_size": 1}
        except Exception:
            return {"compatible": False, "n_features": None,
                    "kind": "load_error", "window_size": 1}

    def _refresh_models(self) -> None:
        from pathlib import Path
        project_dir = Path(__file__).parent
        paths = list_available_models(str(project_dir))
        self._available_paths = paths
        # 预分类每个模型，UI 上加标签提示兼容性
        tagged_names: list[str] = []
        first_compatible_idx: int | None = None
        for i, p in enumerate(paths):
            info = self._probe_model(p)
            base = Path(p).name
            if info["compatible"]:
                tag = f"✅ {info['kind']} ({info['n_features']}f)"
                if info["window_size"] > 1:
                    tag += f" window={info['window_size']}"
                if first_compatible_idx is None:
                    first_compatible_idx = i
            elif info["kind"] == "wrong_features":
                tag = f"⚠ {info['n_features']}特征(需17)"
            elif info["kind"] == "load_error":
                tag = "❌ 加载错误"
            else:
                tag = f"⚠ {info['kind']}"
            tagged_names.append(f"{base}  {tag}")
        self.model_combo["values"] = tagged_names
        # 自动选第一个兼容模型；若已有用户选择则保留
        if first_compatible_idx is not None and not self.model_var.get():
            self.model_combo.current(first_compatible_idx)
            self._on_model_selected()
            return
        if paths and not self.model_var.get():
            self.model_combo.current(0)
            self._on_model_selected()

    def _on_model_selected(self, event=None) -> None:
        from pathlib import Path
        idx = self.model_combo.current()
        if idx < 0 or idx >= len(self._available_paths):
            return
        path = self._available_paths[idx]
        # 从 Spinbox 读取窗口大小 (Spinbox 内部值字符串可能非数字)
        try:
            ws = int(self.window_size_var.get())
        except (tk.TclError, ValueError):
            ws = 1
        ws = max(1, ws)
        self.window_size = ws
        try:
            wrapper = load_model(path, window_size=ws)
            self.wrapper = wrapper
            name = Path(path).name
            ws_tag = f", window={ws}" if ws > 1 else ""
            base_status = f"已加载 {name} ({wrapper.input_features} features{ws_tag})"
            self.status_var.set(
                base_status + self._maybe_reset_for_new_model(path)
            )
        except ValueError as e:
            # 3D 模型 + window_size=1 的常见情形：自动升级到 16 并重试
            msg = str(e)
            if ws == 1 and ("LSTM" in msg or "3D" in msg or "窗口" in msg):
                try:
                    wrapper = load_model(path, window_size=16)
                    self.wrapper = wrapper
                    self.window_size = 16
                    try:
                        self.window_size_var.set(16)
                    except (tk.TclError, Exception):
                        pass
                    name = Path(path).name
                    base_status = (
                        f"已加载 {name} ({wrapper.input_features} features, "
                        f"window=16 自动升级)"
                    )
                    self.status_var.set(
                        base_status + self._maybe_reset_for_new_model(path)
                    )
                    return
                except Exception as e2:
                    self.wrapper = None
                    self.status_var.set(f"加载失败: {e2}")
                    messagebox.showerror("模型加载失败", str(e2))
                    return
            self.wrapper = None
            self.status_var.set(f"加载失败: {e}")
            messagebox.showerror("模型加载失败", msg)
        except (FileNotFoundError, RuntimeError, AttributeError) as e:
            self.wrapper = None
            self.status_var.set(f"加载失败: {e}")
            messagebox.showerror("模型加载失败", str(e))

    def _maybe_reset_for_new_model(self, path: str) -> str:
        """Reset stats + results tree iff ``path`` differs from the model
        currently held in ``self._active_path``. Returns a status suffix
        so the caller can tell the user a reset just happened.

        Re-selecting the same model (or the bootstrap call from
        ``_refresh_models``) is a no-op — counters keep their values so
        the user is not silently wiped on a no-op refresh.
        """
        if path == self._active_path:
            return ""
        self._active_path = path
        self.clear()
        return "（统计已重置）"

    def _on_window_size_change(self) -> None:
        """窗口大小变化时，若已加载模型则提示需重新选择。"""
        try:
            ws = int(self.window_size_var.get())
        except (tk.TclError, ValueError):
            return
        ws = max(1, ws)
        self.window_size = ws
        if self.wrapper is not None:
            self.status_var.set(
                f"窗口已改为 {ws}，请重新选择模型以重新加载"
            )

    # ---- 阈值 ----

    def _on_threshold_change(self, value) -> None:
        self.threshold = float(value)
        self.threshold_label.configure(text=f"{self.threshold:.2f}")

    # ---- 推理接口 ----

    def process_frame(self, record: dict, frame_index: int) -> None:
        """每帧调用一次。已有模型 → 推理；无模型 → 跳过。

        输入特征维度由 wrapper.input_features 决定：17 时喂 (17,)
        数组（走 ``extract_features``），23 时喂原始 record dict
        （23-dim wrapper 自己解 19 行级特征 + RobustScaler + 6 个
        窗口聚合列）。
        """
        if self.wrapper is None:
            return
        truth = int(record.get("binary", 0))
        hex_bytes = self._hex_preview(record)
        try:
            n_feat = self.wrapper.input_features
            if n_feat == 23:
                # 23-dim KEEP_23 wrapper 需要 record（含 time / addr /
                # function 派生列），不能传预提取特征。
                result = self.wrapper.infer(record)
            else:
                features = extract_features(record)
                result = self.wrapper.infer(features)
        except Exception as e:
            label, prob, correct, tag, prob_str = -1, 0.0, "?", "error", f"ERR"
            print(f"[IDS] 推理失败 (frame {frame_index}): {e}")
            self._record_frame(frame_index, hex_bytes, truth, label, prob_str, correct, tag)
            self._update_stats(label, truth, label >= 0)
            return

        # 3D 模型 warm-up：窗口未填满时 wrapper 返回 None
        if result is None:
            label, prob, correct, tag, prob_str = -2, 0.0, "·", "warmup", "warm"
            self._record_frame(frame_index, hex_bytes, truth, label, prob_str, correct, tag)
            # warm-up 不计入准确率统计（既非预测也非错误）
            return

        label, prob = result
        # 重新应用当前阈值
        label = 1 if prob >= self.threshold else 0
        correct = "✓" if label == truth else "✗"
        tag = "attack" if label == 1 else "normal"
        prob_str = f"{prob:.3f}"
        self._record_frame(frame_index, hex_bytes, truth, label, prob_str, correct, tag)
        self._update_stats(label, truth, label >= 0)

    def _record_frame(self, frame_index: int, hex_bytes: str, truth: int,
                      label: int, prob_str: str, correct: str, tag: str) -> None:
        """插入一条记录到 Treeview 并做 LRU 截断。"""
        self.results_tree.insert(
            "", "end",
            values=(frame_index + 1, hex_bytes, truth, label, prob_str, correct),
            tags=(tag,)
        )
        children = self.results_tree.get_children()
        if len(children) > _MAX_RESULTS:
            for c in children[:len(children) - _MAX_RESULTS]:
                self.results_tree.delete(c)

    def clear(self) -> None:
        for c in self.results_tree.get_children():
            self.results_tree.delete(c)
        self._stats = {
            "total": 0, "normal": 0, "attack": 0, "correct": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        }
        self._render_stats()

    def _update_stats(self, pred: int, truth: int, counted: bool) -> None:
        """Increment the right confusion-matrix cell for ``(pred, truth)``.

        ``counted`` is False for warm-up / error frames — those don't
        enter the metric calculation at all. Truth / pred are 0 (normal)
        or 1 (attack); anything else is treated as not-counted.
        """
        if not counted:
            return
        if pred not in (0, 1) or truth not in (0, 1):
            return
        s = self._stats
        s["total"] += 1
        if pred == 1:
            s["attack"] += 1
        else:
            s["normal"] += 1
        if pred == truth:
            s["correct"] += 1
        # Confusion matrix: positive class = attack
        if pred == 1 and truth == 1:
            s["tp"] += 1
        elif pred == 1 and truth == 0:
            s["fp"] += 1
        elif pred == 0 and truth == 0:
            s["tn"] += 1
        else:  # pred == 0, truth == 1
            s["fn"] += 1
        self._render_stats()

    def _render_stats(self) -> None:
        s = self._stats
        if s["total"] == 0:
            self.stats_var.set("总:0  正常:0  攻击:0  正确:0")
            self.metrics_var.set("准确率:-  精确率:-  召回率:-  F1:-")
            return
        # Accuracy
        acc = s["correct"] / s["total"]
        # Precision = TP / (TP + FP); undefined if model never predicted
        # attack — show "-" so the user isn't misled by a 0 from
        # "denominator was zero".
        if (s["tp"] + s["fp"]) > 0:
            precision = s["tp"] / (s["tp"] + s["fp"])
            prec_str = f"{precision:.3f}"
        else:
            prec_str = "-"
        # Recall = TP / (TP + FN); undefined if there were no positive
        # truths in the stream.
        if (s["tp"] + s["fn"]) > 0:
            recall = s["tp"] / (s["tp"] + s["fn"])
            rec_str = f"{recall:.3f}"
        else:
            rec_str = "-"
        # F1 = 2*P*R/(P+R); undefined when either is (because both
        # formulas already produced "-"). NaN guard for the floating
        # case where P or R is exactly 0 but the other is defined.
        if prec_str == "-" or rec_str == "-":
            f1_str = "-"
        else:
            p, r = precision, recall
            if (p + r) > 0:
                f1 = 2 * p * r / (p + r)
                f1_str = f"{f1:.3f}"
            else:
                f1_str = "-"
        self.stats_var.set(
            f"总:{s['total']}  正常:{s['normal']}  攻击:{s['attack']}  正确:{s['correct']}"
        )
        self.metrics_var.set(
            f"准确率:{acc:.3f}  精确率:{prec_str}  召回率:{rec_str}  F1:{f1_str}"
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