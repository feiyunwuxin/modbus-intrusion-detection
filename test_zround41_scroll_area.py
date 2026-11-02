#!/usr/bin/env python3
"""验证 Round 41: 用 QScrollArea 包装 root, 让 GUI 可上下滚动.

⭐ 2026-10-02 第四十一轮:
  Round 40 加 MetricsChartPanel (matplotlib 暗色曲线) 后, 总高度超过常见笔记本屏幕
  (典型 800-1080px 高), 用户看不到新加的曲线图表. 需求: 把 root 布局放进 QScrollArea,
  内容超出窗口高度时自动出现垂直滚动条.

  修改:
    1. rtu_replay_gui.py:1720-1740 — 创建 QScrollArea, setCentralWidget(scroll),
       容器 QWidget 作为 scroll 的 widget, root layout 应用于容器.
    2. _apply_scrollbar_style() — QScrollArea 应用统一滚动条 QSS (宽 12px, hover 蓝).
    3. self._scroll_area 暴露给调试和测试用.

  测试设计:
    本测试只验 _build_ui 的源码 + scroll 属性存在, 不实例化 RTUReplayGUI (避免 _build_ui
    全部跑通需要的 mock). 用 source inspection (zround28+ 通用的 dummy PyQt5 模式).
"""
from __future__ import annotations
import os
import sys
import inspect

# === Dummy PyQt5 (与 zround28-40 一致) ===
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
    def setWidget(self, *a, **k): pass
    def setWidgetResizable(self, *a, **k): pass
    def setHorizontalScrollBarPolicy(self, *a, **k): pass
    def setVerticalScrollBarPolicy(self, *a, **k): pass
    def setFrameShape(self, *a, **k): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass

_dummy_qt = type(sys)('PyQt5')
sys.modules['PyQt5'] = _dummy_qt
_dummy_qtcore = type(sys)('PyQt5.QtCore')
sys.modules['PyQt5.QtCore'] = _dummy_qtcore
_dummy_qtcore.Qt = type('Qt', (), {'AlignCenter': 0, 'ScrollBarAlwaysOff': 0, 'ScrollBarAsNeeded': 0})
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
              'QTableWidget', 'QHeaderView', 'QProgressBar', 'QSlider',
              'QScrollArea'):
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

def test_qscrollarea_imported():
    """验证 QScrollArea 已 import."""
    assert hasattr(rtu_replay_gui, 'QScrollArea'), (
        "BUG: QScrollArea 未 import (Round 41 必须)"
    )


def test_scroll_area_created_in_build_ui():
    """验证 _build_ui 中创建 QScrollArea 实例."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'QScrollArea()' in src, (
        "BUG: QScrollArea() 未在 _build_ui 中创建"
    )


def test_scroll_area_set_as_central_widget():
    """验证 QScrollArea 通过 setCentralWidget 设为中心窗口."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'self.setCentralWidget(scroll)' in src, (
        "BUG: setCentralWidget(scroll) 未调用 (滚动区必须是 central widget)"
    )


def test_scroll_area_set_widget_resizable():
    """验证 setWidgetResizable(True) — 容器宽度跟随窗口缩放."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'setWidgetResizable(True)' in src, (
        "BUG: setWidgetResizable(True) 未调用 (容器不会自适应宽度)"
    )


def test_scroll_area_horizontal_disabled():
    """验证水平滚动条禁用 (只允许垂直滚动)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)' in src, (
        "BUG: 水平滚动条未禁用. 横向不应滚动 (窗口宽度 820 固定)"
    )


def test_scroll_area_vertical_as_needed():
    """验证垂直滚动条 AsNeeded (只在需要时显示)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)' in src, (
        "BUG: 垂直滚动条策略未设为 AsNeeded (会在内容少时也占空间)"
    )


def test_scroll_area_widget_set():
    """验证 QScrollArea.setWidget(central) 装了容器 widget."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'scroll.setWidget(central)' in src, (
        "BUG: scroll.setWidget(central) 未调用. 容器 widget 必须 setWidget 到 scroll 上"
    )


def test_root_layout_on_container():
    """验证 root QVBoxLayout 应用于 central (容器 widget), 不是 scroll 本身."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'QVBoxLayout(central)' in src, (
        "BUG: root layout 应在 central 容器上, 不是 scroll 上"
    )


def test_scroll_area_attribute_exposed():
    """验证 self._scroll_area 暴露 (供测试 + 调试用)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'self._scroll_area = scroll' in src, (
        "BUG: self._scroll_area 暴露属性未设置 (测试和调试需要访问 scroll)"
    )


def test_apply_scrollbar_style_function():
    """验证 _apply_scrollbar_style() helper 函数已定义."""
    assert hasattr(rtu_replay_gui, '_apply_scrollbar_style'), (
        "BUG: _apply_scrollbar_style() helper 函数未定义"
    )
    assert callable(rtu_replay_gui._apply_scrollbar_style), (
        "BUG: _apply_scrollbar_style 不是 callable"
    )


def test_scrollbar_style_applied_in_build_ui():
    """验证 _build_ui 中调用 _apply_scrollbar_style(scroll)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert '_apply_scrollbar_style(scroll)' in src, (
        "BUG: _apply_scrollbar_style(scroll) 未在 _build_ui 调用 (滚动条样式未应用)"
    )


def test_scrollbar_qss_constants():
    """验证 _SCROLLBAR_QSS 常量定义 (滚动条样式)."""
    assert hasattr(rtu_replay_gui, '_SCROLLBAR_QSS'), (
        "BUG: _SCROLLBAR_QSS 常量未定义"
    )
    qss = rtu_replay_gui._SCROLLBAR_QSS
    assert 'QScrollBar:vertical' in qss, (
        "BUG: _SCROLLBAR_QSS 缺 QScrollBar:vertical 块"
    )
    assert 'width: 12px' in qss, (
        "BUG: _SCROLLBAR_QSS 应设宽度 12px (Windows 默认太窄)"
    )
    assert '#0066cc' in qss, (
        "BUG: _SCROLLBAR_QSS 应含 hover 蓝色 (#0066cc) 与 GUI 配色一致"
    )


def test_round41_marker_comment():
    """验证源码有 Round 41 标记注释."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    assert '第四十一轮' in src, "BUG: 缺 Round 41 标记注释"


def test_no_horizontal_scrollbar_in_qss():
    """验证 _SCROLLBAR_QSS 没设 horizontal 样式 (避免双滚动条)."""
    qss = rtu_replay_gui._SCROLLBAR_QSS
    # 我们只允许垂直滚动, 横向 setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 控制
    # QSS 没必要也不应包含 horizontal 块
    assert 'QScrollBar:horizontal' not in qss, (
        "BUG: _SCROLLBAR_QSS 不应含 horizontal 样式 (横向滚动已禁用)"
    )


if __name__ == '__main__':
    test_qscrollarea_imported()
    print("[PASS] test_qscrollarea_imported")

    test_scroll_area_created_in_build_ui()
    print("[PASS] test_scroll_area_created_in_build_ui")

    test_scroll_area_set_as_central_widget()
    print("[PASS] test_scroll_area_set_as_central_widget")

    test_scroll_area_set_widget_resizable()
    print("[PASS] test_scroll_area_set_widget_resizable")

    test_scroll_area_horizontal_disabled()
    print("[PASS] test_scroll_area_horizontal_disabled (AlwaysOff)")

    test_scroll_area_vertical_as_needed()
    print("[PASS] test_scroll_area_vertical_as_needed (AsNeeded)")

    test_scroll_area_widget_set()
    print("[PASS] test_scroll_area_widget_set")

    test_root_layout_on_container()
    print("[PASS] test_root_layout_on_container")

    test_scroll_area_attribute_exposed()
    print("[PASS] test_scroll_area_attribute_exposed")

    test_apply_scrollbar_style_function()
    print("[PASS] test_apply_scrollbar_style_function")

    test_scrollbar_style_applied_in_build_ui()
    print("[PASS] test_scrollbar_style_applied_in_build_ui")

    test_scrollbar_qss_constants()
    print("[PASS] test_scrollbar_qss_constants")

    test_round41_marker_comment()
    print("[PASS] test_round41_marker_comment (Round 41 注释)")

    test_no_horizontal_scrollbar_in_qss()
    print("[PASS] test_no_horizontal_scrollbar_in_qss")

    print("\n所有 14 个 Round 41 测试通过!")
    print("QScrollArea 包装 root, GUI 可上下滚动看完整界面 (含 MetricsChartPanel).")
    print("滚动条: 12px 宽, hover 蓝色, 内容溢出自动出现.")
