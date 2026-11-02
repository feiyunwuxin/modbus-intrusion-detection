#!/usr/bin/env python3
"""验证 Round 40: IDS 5 指标曲线面板 (Precision/Recall/F1/Accuracy/FPR).

⭐ 2026-10-02 第四十轮:
  用户要求在 GUI 上增加 precision, recall, F1, Accuracy, FRR 的曲线图.
  "FRR" 解释为 FPR (False Positive Rate), 与现有 IdsReportPanel KPI 配色对齐.

  新增 MetricsChartPanel(QWidget):
    - matplotlib FigureCanvasQTAgg, 暗色主题
    - 5 条 Line2D: Precision / Recall / F1 / Accuracy / FPR
    - FIFO 上限 500 点
    - 🗑 清空曲线 + ⏸ 暂停/▶ 继续 按钮

  数据流:
    IdsReportPanel.metrics_updated (dict) → MetricsChartPanel._on_metrics_updated
    IdsReportPanel.metrics_reset → MetricsChartPanel._on_metrics_reset

  布局: root 底部, ids_report 之后 (root.addWidget, stretch=1)
"""
from __future__ import annotations
import os
import sys
import inspect

# === Dummy PyQt5 (与 zround28-39 一致) ===
class _DummySignal:
    def __init__(self, *a, **k):
        self._calls = []
        self._slots = []
    def emit(self, *a):
        self._calls.append(a)
        for s in self._slots: s(*a)
    def connect(self, slot, *a, **k):
        self._slots.append(slot)

class _DummyWidget:
    def __init__(self, *a, **k):
        self._style = ""
        self._text = ""
        self.clicked = _DummySignal()
        self.toggled = _DummySignal()
    def setStyleSheet(self, qss): self._style = qss
    def styleSheet(self): return self._style
    def setText(self, t): self._text = t
    def text(self): return self._text
    def setCheckable(self, *a, **k): pass
    def setToolTip(self, *a, **k): pass
    def setAlignment(self, *a, **k): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass

_dummy_qt = type(sys)('PyQt5')
sys.modules['PyQt5'] = _dummy_qt
_dummy_qtcore = type(sys)('PyQt5.QtCore')
sys.modules['PyQt5.QtCore'] = _dummy_qtcore
_dummy_qtcore.Qt = type('Qt', (), {'AlignCenter': 0})
_dummy_qtcore.QObject = _DummyWidget
_dummy_qtcore.QThread = _DummyWidget
_dummy_qtcore.QTimer = type('QTimer', (), {'singleShot': staticmethod(lambda *a, **k: None)})
_dummy_qtcore.QTextCursor = _DummyWidget
_dummy_qtcore.pyqtSignal = lambda *a, **k: _DummySignal()
_dummy_qtgui = type(sys)('PyQt5.QtGui')
sys.modules['PyQt5.QtGui'] = _dummy_qtgui
_dummy_qtgui.QTextCursor = _DummyWidget
_dummy_qtgui.QFont = type('QFont', (), {'Monospace': 7})
_dummy_qtwidgets = type(sys)('PyQt5.QtWidgets')
sys.modules['PyQt5.QtWidgets'] = _dummy_qtwidgets
_dummy_qtwidgets.WidgetWidth = 1
for _name in ('QApplication', 'QWidget', 'QHBoxLayout', 'QVBoxLayout',
              'QLabel', 'QPushButton', 'QPlainTextEdit', 'QTextEdit',
              'QComboBox', 'QSpinBox', 'QDoubleSpinBox', 'QFileDialog',
              'QMessageBox', 'QMainWindow', 'QDialog', 'QCheckBox',
              'QRadioButton', 'QGroupBox', 'QFrame', 'QSizePolicy',
              'QLineEdit', 'QTabWidget', 'QGridLayout', 'QStatusBar',
              'QMenuBar', 'QAction', 'QListWidget', 'QTreeWidget',
              'QTableWidget', 'QHeaderView', 'QProgressBar', 'QScrollArea', 'QSlider'):
    setattr(_dummy_qtwidgets, _name, _DummyWidget)
sys.modules['serial'] = type(sys)('serial')
sys.modules['serial'].Serial = type('Serial', (), {})
sys.modules['serial.tools'] = type(sys)('serial.tools')
sys.modules['serial.tools.list_ports'] = type(sys)('serial.tools.list_ports')
sys.modules['serial.tools.list_ports'].comports = lambda: []

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.modules.pop('rtu_replay_gui', None)
import rtu_replay_gui  # noqa: E402


# === Tests ===

def test_metrics_chart_panel_class_exists():
    """验证 MetricsChartPanel 类已定义."""
    assert hasattr(rtu_replay_gui, 'MetricsChartPanel'), (
        "BUG: MetricsChartPanel 类未定义 (Round 40 应加)"
    )
    cls = rtu_replay_gui.MetricsChartPanel
    # 验证是 QWidget 子类
    assert issubclass(cls, rtu_replay_gui.QWidget), (
        f"BUG: MetricsChartPanel 应继承 QWidget, got {cls.__mro__[:3]}"
    )


def test_metrics_chart_added_to_root():
    """验证 metrics_chart_box 加入 root (Round 40 必须)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'root.addWidget(self.metrics_chart_box' in src, (
        "BUG: root.addWidget(self.metrics_chart_box) 未调用"
    )


def test_metrics_chart_added_with_stretch():
    """验证 metrics_chart_box 用 stretch=1 加入 root (与 log_rx_row 对称)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'root.addWidget(self.metrics_chart_box, stretch=1)' in src, (
        "BUG: metrics_chart_box 应 stretch=1 加入 (与 log_rx_row 一致)"
    )


def test_metrics_chart_after_ids_report():
    """验证 metrics_chart_box 在 ids_report 之后 (用户要求: 放在 IDS 面板下)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    ids_idx = src.find('root.addWidget(self.ids_report)')
    chart_idx = src.find('root.addWidget(self.metrics_chart_box')
    assert ids_idx > 0, "BUG: ids_report addWidget 缺失"
    assert chart_idx > 0, "BUG: metrics_chart_box addWidget 缺失"
    assert ids_idx < chart_idx, (
        f"BUG: ids_report ({ids_idx}) 必须在 metrics_chart_box ({chart_idx}) 之前 "
        "(Round 40: 用户要求曲线在 IDS 面板下方)"
    )


def test_metrics_updated_signal_exists():
    """验证 IdsReportPanel.metrics_updated pyqtSignal 已定义."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.IdsReportPanel)
    assert 'metrics_updated' in src, (
        "BUG: IdsReportPanel.metrics_updated 信号缺失 (Round 40 必须)"
    )


def test_metrics_reset_signal_exists():
    """验证 IdsReportPanel.metrics_reset pyqtSignal 已定义."""
    src = inspect.getsource(rtu_replay_gui.IdsReportPanel)
    assert 'metrics_reset' in src, (
        "BUG: IdsReportPanel.metrics_reset 信号缺失 (Round 40 必须)"
    )


def test_metrics_signals_at_class_level():
    """⭐ 第四十轮 bug 修复: pyqtSignal 必须在类级别声明, 不能在 __init__ 里 self.X = pyqtSignal().

    历史 bug: Round 40 初始实现把声明写在 __init__ 内:
        def __init__(self, ...):
            self.metrics_updated = pyqtSignal(dict)  # ❌ 错误!
    运行时 `self.metrics_updated.connect(slot)` 报 AttributeError:
        "'PyQt5.QtCore.pyqtSignal' object has no attribute 'connect'"

    根因: PyQt5 metaclass 只在类级别绑定 pyqtSignal 描述符为 bound signal.
    在 __init__ 里赋值得到的是未绑定的 pyqtSignal 描述符 (没 connect/emit).

    修复: 把 pyqtSignal 移到类级 (与 IdsLineParser 第 704-715 行一致).
    """
    cls = rtu_replay_gui.IdsReportPanel
    # 必须在类 __dict__ 里能找到 (而不是只在 __init__ 源码里出现)
    cls_attrs = cls.__dict__
    assert 'metrics_updated' in cls_attrs, (
        "BUG: metrics_updated 必须 IdsReportPanel 类级别声明. "
        "在 __init__ 里 self.X = pyqtSignal(...) 是错的, PyQt5 metaclass 不会绑定它. "
        "修复: 把声明移到类体 (__init__ 之外). "
        "参考 IdsLineParser class IdsLineParser(QObject): 下面的 count_changed = pyqtSignal(int)"
    )
    assert 'metrics_reset' in cls_attrs, (
        "BUG: metrics_reset 必须 IdsReportPanel 类级别声明. 同上."
    )
    # 反向检查: 不能在 __init__ 里出现 self.metrics_updated = pyqtSignal(...)
    import inspect
    init_src = inspect.getsource(cls.__init__)
    assert 'self.metrics_updated = pyqtSignal' not in init_src, (
        "BUG: __init__ 里赋值 self.metrics_updated = pyqtSignal(...). "
        "PyQt5 metaclass 不在实例化时绑定. 应移除, 仅在类级声明即可."
    )
    assert 'self.metrics_reset = pyqtSignal' not in init_src, (
        "BUG: __init__ 里赋值 self.metrics_reset = pyqtSignal(...). 同上."
    )


def test_update_rates_emits_metrics_updated():
    """验证 _update_rates() 末尾 emit metrics_updated."""
    src = inspect.getsource(rtu_replay_gui.IdsReportPanel)
    # 找 _update_rates 方法体 (兼容 'def _update_rates(self):' 或 'def _update_rates(self) -> None:')
    start = src.find('def _update_rates(self)')
    end = src.find('def reset_counters(self)')
    assert start > 0, "BUG: _update_rates 未找到"
    assert end > start, "BUG: reset_counters 未在 _update_rates 之后"
    block = src[start:end]
    assert 'metrics_updated.emit' in block, (
        "BUG: _update_rates 不 emit metrics_updated (Round 40 必须)"
    )


def test_reset_counters_emits_metrics_reset():
    """验证 reset_counters() 末尾 emit metrics_reset."""
    src = inspect.getsource(rtu_replay_gui.IdsReportPanel)
    start = src.find('def reset_counters(self)')
    assert start > 0, "BUG: reset_counters 未找到"
    # 取整个函数体 (到下一个 def 或类结尾)
    next_def = src.find('\n    def ', start + 1)
    if next_def < 0:
        next_def = start + 3000
    block = src[start:next_def]
    assert 'metrics_reset.emit' in block, (
        "BUG: reset_counters 不 emit metrics_reset (Round 40 必须)"
    )


def test_chart_wired_in_wire():
    """验证 _wire() 中 metrics_updated + metrics_reset 已 connect."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._wire)
    assert 'metrics_updated.connect' in src, (
        "BUG: metrics_updated.connect 未在 _wire 中调用"
    )
    assert 'metrics_reset.connect' in src, (
        "BUG: metrics_reset.connect 未在 _wire 中调用"
    )
    # 验证连到 chart panel
    assert '_on_metrics_updated' in src, (
        "BUG: _on_metrics_updated 未连 (应为 MetricsChartPanel 的 slot)"
    )
    assert '_on_metrics_reset' in src, (
        "BUG: _on_metrics_reset 未连"
    )


def test_chinese_title_styled():
    """验证 metrics_chart_box 调用 _style_chinese_groupbox (防 Windows 中文标题截断)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert '_style_chinese_groupbox(self.metrics_chart_box)' in src, (
        "BUG: metrics_chart_box 未调 _style_chinese_groupbox "
        "(中文标题 'IDS 曲线图表 (实时性能)' 会被 Windows Keil/PyQt 截断)"
    )


def test_chart_fifo_500_cap():
    """验证 MetricsChartPanel._FIFO_CAP = 500 (FIFO 上限)."""
    cls = rtu_replay_gui.MetricsChartPanel
    assert hasattr(cls, '_FIFO_CAP'), "BUG: _FIFO_CAP 未定义"
    assert cls._FIFO_CAP == 500, (
        f"BUG: _FIFO_CAP 应 500, got {cls._FIFO_CAP}"
    )


def test_clear_and_pause_buttons_exist():
    """验证 chart_clear_btn + chart_pause_btn 已创建."""
    # 通过 source 扫描 (不需要实例化)
    src = inspect.getsource(rtu_replay_gui.MetricsChartPanel)
    assert 'self.chart_clear_btn = QPushButton' in src, (
        "BUG: chart_clear_btn 未创建"
    )
    assert 'self.chart_pause_btn = QPushButton' in src, (
        "BUG: chart_pause_btn 未创建"
    )


def test_metrics_dict_emits_5_keys():
    """验证 _update_rates emit 的 dict 含 5 个键 (p/r/f1/acc/fpr)."""
    src = inspect.getsource(rtu_replay_gui.IdsReportPanel)
    start = src.find('def _update_rates(self)')
    end = src.find('def reset_counters(self)')
    block = src[start:end]
    for key in ("'p'", "'r'", "'f1'", "'acc'", "'fpr'"):
        assert key in block, (
            f"BUG: metrics_updated emit dict 缺键 {key}"
        )


def test_mpl_lazy_import():
    """验证 matplotlib 是按需 import (顶部 try/except 模式)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    # 找 try 块
    assert 'try:' in src and 'matplotlib' in src, (
        "BUG: matplotlib 未做 lazy import 保护"
    )
    # 验证 _MPL_AVAILABLE 标志
    assert '_MPL_AVAILABLE' in src, (
        "BUG: _MPL_AVAILABLE 标志未定义 (Round 40 引入)"
    )


def test_round40_marker_comment():
    """验证源码有 Round 40 标记注释."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    assert '第四十轮' in src, "BUG: 缺 Round 40 标记注释"


if __name__ == '__main__':
    test_metrics_chart_panel_class_exists()
    print("[PASS] test_metrics_chart_panel_class_exists")

    test_metrics_chart_added_to_root()
    print("[PASS] test_metrics_chart_added_to_root")

    test_metrics_chart_added_with_stretch()
    print("[PASS] test_metrics_chart_added_with_stretch")

    test_metrics_chart_after_ids_report()
    print("[PASS] test_metrics_chart_after_ids_report")

    test_metrics_updated_signal_exists()
    print("[PASS] test_metrics_updated_signal_exists")

    test_metrics_reset_signal_exists()
    print("[PASS] test_metrics_reset_signal_exists")

    test_metrics_signals_at_class_level()
    print("[PASS] test_metrics_signals_at_class_level (⭐ Round 40 bug 回归)")

    test_update_rates_emits_metrics_updated()
    print("[PASS] test_update_rates_emits_metrics_updated")

    test_reset_counters_emits_metrics_reset()
    print("[PASS] test_reset_counters_emits_metrics_reset")

    test_chart_wired_in_wire()
    print("[PASS] test_chart_wired_in_wire")

    test_chinese_title_styled()
    print("[PASS] test_chinese_title_styled")

    test_chart_fifo_500_cap()
    print("[PASS] test_chart_fifo_500_cap (_FIFO_CAP=500)")

    test_clear_and_pause_buttons_exist()
    print("[PASS] test_clear_and_pause_buttons_exist")

    test_metrics_dict_emits_5_keys()
    print("[PASS] test_metrics_dict_emits_5_keys (p/r/f1/acc/fpr)")

    test_mpl_lazy_import()
    print("[PASS] test_mpl_lazy_import (try/except + _MPL_AVAILABLE)")

    test_round40_marker_comment()
    print("[PASS] test_round40_marker_comment (Round 40 注释)")

    print("\n所有 16 个 Round 40 测试通过!")
    print("MetricsChartPanel: 5 条曲线 + FIFO 500 + 清空/暂停按钮.")
    print("IdsReportPanel: 新增 metrics_updated/metrics_reset 信号.")
    print("root 布局: ids_report 下方新增 metrics_chart_box (stretch=1).")