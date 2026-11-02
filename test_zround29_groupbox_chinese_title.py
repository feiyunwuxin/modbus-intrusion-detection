#!/usr/bin/env python3
"""验证 Round 29: 所有含中文的 QGroupBox 应用 _GROUPBOX_TITLE_QSS, 防 Windows 默认 8pt Tahoma 截断.

⭐ 2026-10-02 第二十九轮:
  用户反馈 "📋 日志" 和 "📥 接收区 (RX)" 标题文字显示不全 (中文字符顶端/底端被裁切).
  根因: Windows PyQt5 QGroupBox::title 默认用系统 8pt Tahoma, 中文 glyph 高度超过默认
  title subcontrol, 渲染时上下裁切. 解法: 定义 _GROUPBOX_TITLE_QSS 常量 + _style_chinese_groupbox
  辅助函数, 给所有含中文的 QGroupBox 显式调用.
"""
from __future__ import annotations
import os
import re
import sys
import inspect

# Dummy PyQt5 (与其他测试一致)
class _DummySignal:
    def __init__(self, *args, **kwargs):
        self._calls = []
        self._slots = []
    def emit(self, *args):
        self._calls.append(args)
        for s in self._slots:
            s(*args)
    def connect(self, slot, *args, **kwargs):
        self._slots.append(slot)

class _DummyQObject:
    def __init__(self, *args, **kwargs):
        self._style = ""
    def setStyleSheet(self, qss):
        self._style = qss
    def styleSheet(self):
        return self._style
    def setMinimumWidth(self, w): pass
    def setMaximumHeight(self, h): pass
    def setMinimumHeight(self, h): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass
    def setReadOnly(self, *a, **k): pass

_dummy_qt = type(sys)('PyQt5')
sys.modules['PyQt5'] = _dummy_qt
_dummy_qtcore = type(sys)('PyQt5.QtCore')
sys.modules['PyQt5.QtCore'] = _dummy_qtcore
_dummy_qtcore.Qt = type('Qt', (), {'AlignCenter': 0})
_dummy_qtcore.QObject = _DummyQObject
_dummy_qtcore.QThread = _DummyQObject
_dummy_qtcore.QTimer = type('QTimer', (), {'singleShot': staticmethod(lambda *a, **k: None)})
_dummy_qtcore.QTextCursor = _DummyQObject
_dummy_qtcore.pyqtSignal = lambda *a, **k: _DummySignal()
_dummy_qtgui = type(sys)('PyQt5.QtGui')
sys.modules['PyQt5.QtGui'] = _dummy_qtgui
_dummy_qtgui.QTextCursor = _DummyQObject
_dummy_qtwidgets = type(sys)('PyQt5.QtWidgets')
sys.modules['PyQt5.QtWidgets'] = _dummy_qtwidgets
for _name in ('QApplication', 'QWidget', 'QHBoxLayout', 'QVBoxLayout',
              'QLabel', 'QPushButton', 'QPlainTextEdit', 'QComboBox',
              'QSpinBox', 'QDoubleSpinBox', 'QFileDialog', 'QMessageBox',
              'QMainWindow', 'QDialog', 'QCheckBox', 'QRadioButton',
              'QGroupBox', 'QFrame', 'QSizePolicy', 'QTextEdit', 'QLineEdit',
              'QTabWidget', 'QGridLayout', 'QStatusBar', 'QMenuBar', 'QAction',
              'QListWidget', 'QTreeWidget', 'QTableWidget', 'QHeaderView',
              'QProgressBar', 'QScrollArea', 'QSlider'):
    setattr(_dummy_qtwidgets, _name, _DummyQObject)
sys.modules['serial'] = type(sys)('serial')
sys.modules['serial'].Serial = type('Serial', (), {})
sys.modules['serial.tools'] = type(sys)('serial.tools')
sys.modules['serial.tools.list_ports'] = type(sys)('serial.tools.list_ports')
sys.modules['serial.tools.list_ports'].comports = lambda: []

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import rtu_replay_gui  # noqa: E402


# === Tests ===

def test_groupbox_title_qss_defined():
    """验证 _GROUPBOX_TITLE_QSS 常量存在且包含关键修复规则."""
    assert hasattr(rtu_replay_gui, '_GROUPBOX_TITLE_QSS'), (
        "缺少 _GROUPBOX_TITLE_QSS 常量"
    )
    qss = rtu_replay_gui._GROUPBOX_TITLE_QSS
    # 必须含 QGroupBox::title subcontrol 规则 (核心修复)
    assert 'QGroupBox::title' in qss, "QSS 缺 QGroupBox::title 块"
    assert 'subcontrol-origin: margin' in qss, "title subcontrol-origin 必须 margin"
    assert 'subcontrol-position: top left' in qss, "title 必须 top left"
    assert 'padding' in qss, "title 必须设 padding 防裁切"


def test_helper_function_defined():
    """验证 _style_chinese_groupbox 辅助函数存在."""
    assert hasattr(rtu_replay_gui, '_style_chinese_groupbox'), (
        "缺少 _style_chinese_groupbox 辅助函数"
    )
    helper = rtu_replay_gui._style_chinese_groupbox
    assert callable(helper), "_style_chinese_groupbox 不可调用"
    # 必须接收 box 参数
    sig = inspect.signature(helper)
    params = list(sig.parameters.keys())
    assert 'box' in params, f"_style_chinese_groupbox 缺 'box' 参数: {params}"


def test_helper_applies_qss():
    """验证 _style_chinese_groupbox 实际把 QSS 写到 box 上."""
    QGroupBox = sys.modules['PyQt5.QtWidgets'].QGroupBox
    box = QGroupBox("📋 日志")
    assert box.styleSheet() == "", "初始 stylesheet 应为空"
    rtu_replay_gui._style_chinese_groupbox(box)
    assert box.styleSheet() == rtu_replay_gui._GROUPBOX_TITLE_QSS, (
        "_style_chinese_groupbox 没正确应用 QSS"
    )


def test_all_chinese_groupboxes_styled():
    """验证所有 6 个含中文标题的 QGroupBox 都调用了 _style_chinese_groupbox.
    通过源码静态扫描确认, 不依赖运行 GUI (GUI 实例化涉及大量 dummy).
    """
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 6 个含中文/emoji 标题的 QGroupBox
    chinese_titles = [
        'QGroupBox("串口设置")',
        'QGroupBox("数据集 (IanArffDataset.csv)")',
        'QGroupBox("回放选项")',
        'QGroupBox("手动模式 (单帧发送 — 不走时间间隔)")',
        'QGroupBox("📋 日志")',
        'QGroupBox("📥 接收区 (RX)")',
    ]
    for title_pat in chinese_titles:
        # 找创建行
        idx = src.find(title_pat)
        assert idx >= 0, f"源码找不到 {title_pat}"
        # 找 _style_chinese_groupbox 调用, 必须在同一块 (前后 200 字符内)
        window_end = idx + 200
        block = src[idx:window_end]
        assert '_style_chinese_groupbox' in block, (
            f"{title_pat} 之后没调用 _style_chinese_groupbox (中文截断)"
        )


def test_helper_handles_dummy_gracefully():
    """验证 helper 在 dummy QGroupBox (无 setStyleSheet) 上不抛错.
    当前 dummy_QObject 已有 setStyleSheet, 但保险起见显式测一遍.
    """
    QGroupBox = sys.modules['PyQt5.QtWidgets'].QGroupBox
    box = QGroupBox("日志")
    # 应不抛错
    try:
        rtu_replay_gui._style_chinese_groupbox(box)
    except Exception as e:
        raise AssertionError(f"_style_chinese_groupbox 抛错: {e}")


if __name__ == '__main__':
    test_groupbox_title_qss_defined()
    print("[PASS] test_groupbox_title_qss_defined (QSS 规则完整)")

    test_helper_function_defined()
    print("[PASS] test_helper_function_defined")

    test_helper_applies_qss()
    print("[PASS] test_helper_applies_qss")

    test_all_chinese_groupboxes_styled()
    print("[PASS] test_all_chinese_groupboxes_styled (6 个中文 QGroupBox 都调用 helper)")

    test_helper_handles_dummy_gracefully()
    print("[PASS] test_helper_handles_dummy_gracefully")

    print("\n所有 5 个 QGroupBox 中文标题测试通过!")
    print("Round 29: Windows PyQt5 默认 8pt Tahoma 裁切中文 glyph 已修复.")