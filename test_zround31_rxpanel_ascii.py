#!/usr/bin/env python3
"""验证 Round 31: RxPanel 默认格式 ASCII + 时间内容同行.

⭐ 2026-10-02 第三十一轮:
  1. RxPanel 默认格式: hex+ASCII → ASCII (用户偏好, 看接 ASCII 可读性最高)
  2. 接收区显示 "[时间 + 字节数]:  内容" (原 JSON-style 两行, 现一行加一列)
"""
from __future__ import annotations
import os
import sys
import inspect

# === Dummy PyQt5 ===
class _DummySignal:
    def __init__(self, *a, **k):
        self._calls = []
        self._slots = []
    def emit(self, *a):
        self._calls.append(a)
        for s in self._slots:
            s(*a)
    def connect(self, slot, *a, **k):
        self._slots.append(slot)

class _DummyTextEdit:
    # ⭐ Round 32: QTextEdit.WidgetWidth 是 class attribute, 不是 instance attribute
    WidgetWidth = 1
    NoWrap = 0

    def __init__(self, *a, **k):
        self._buf = []
        self._style = ""
        self._font = None
        self._line_wrap = None
        self._cursor = type('Cursor', (), {
            'End': 0,
            'movePosition': lambda *a, **k: None,
            'ensureCursorVisible': lambda *a, **k: None,
        })()
    def append(self, s):
        self._buf.append(s)
    def toPlainText(self):
        return "\n".join(self._buf)
    def setReadOnly(self, *a, **k): pass
    def setFont(self, *a, **k): pass
    def setStyleSheet(self, qss): self._style = qss
    def styleSheet(self): return self._style
    def setLineWrapMode(self, m): self._line_wrap = m
    def lineWrapMode(self): return self._line_wrap
    def textCursor(self): return self._cursor
    def setTextCursor(self, *a, **k): pass
    def ensureCursorVisible(self, *a, **k): pass
    def clear(self): self._buf = []
    def isReadOnly(self): return True
    def font(self): return type('Font', (), {'setFamily': lambda *a: None, 'setPointSize': lambda *a: None})()

class _DummyWidget:
    def __init__(self, *a, **k):
        self._style = ""
        self._text = ""
        # Signal-like stubs (QPushButton.clicked.connect pattern)
        self.clicked = _DummySignal()
        self.pressed = _DummySignal()
        self.released = _DummySignal()
        self.toggled = _DummySignal()
        self.stateChanged = _DummySignal()
    def setStyleSheet(self, qss): self._style = qss
    def styleSheet(self): return self._style
    def setText(self, t): self._text = t
    def text(self): return self._text
    def setEnabled(self, e): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass

class _DummyComboBox(_DummyWidget):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._items = []
        self._current = 0
    def addItems(self, items):
        self._items = list(items)
    def currentText(self):
        return self._items[self._current] if self._items else ""
    def setCurrentText(self, t):
        if t in self._items:
            self._current = self._items.index(t)

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
_dummy_qtwidgets = type(sys)('PyQt5.QtWidgets')
sys.modules['PyQt5.QtWidgets'] = _dummy_qtwidgets
_dummy_qtwidgets.QApplication = _DummyWidget
_dummy_qtwidgets.QWidget = _DummyWidget
_dummy_qtwidgets.QHBoxLayout = _DummyWidget
_dummy_qtwidgets.QVBoxLayout = _DummyWidget


class _DummyLabel(_DummyWidget):
    def __init__(self, text="", *a, **k):
        super().__init__(*a, **k)
        self._text = text

_dummy_qtwidgets.QLabel = _DummyLabel
_dummy_qtwidgets.QPushButton = _DummyWidget
_dummy_qtwidgets.QPlainTextEdit = _DummyTextEdit
_dummy_qtwidgets.QTextEdit = _DummyTextEdit  # ⭐ RxPanel 用 QTextEdit
_dummy_qtwidgets.QComboBox = _DummyComboBox
# WidgetWidth / NoWrap 已定义为 _DummyTextEdit 类属性 (QTextEdit class attr)
for _name in ('QSpinBox', 'QDoubleSpinBox', 'QFileDialog', 'QMessageBox',
              'QMainWindow', 'QDialog', 'QCheckBox', 'QRadioButton',
              'QGroupBox', 'QFrame', 'QSizePolicy', 'QLineEdit',
              'QTabWidget', 'QGridLayout', 'QStatusBar', 'QMenuBar', 'QAction',
              'QListWidget', 'QTreeWidget', 'QTableWidget', 'QHeaderView',
              'QProgressBar', 'QScrollArea', 'QSlider'):
    setattr(_dummy_qtwidgets, _name, _DummyWidget)
sys.modules['serial'] = type(sys)('serial')
sys.modules['serial'].Serial = type('Serial', (), {})
sys.modules['serial.tools'] = type(sys)('serial.tools')
sys.modules['serial.tools.list_ports'] = type(sys)('serial.tools.list_ports')
sys.modules['serial.tools.list_ports'].comports = lambda: []

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
# ⭐ pytest 跑多文件时, 其他 test_* 可能先 import rtu_replay_gui, 把模块缓存下来.
#   我们的 dummy 必须在 rtu_replay_gui 首次 import 时生效. 显式清缓存, 强制重新 import.
sys.modules.pop('rtu_replay_gui', None)
import rtu_replay_gui  # noqa: E402


# === Tests ===

def test_default_format_is_ascii():
    """验证 RxPanel 默认格式是 ASCII (Round 31 改动)."""
    panel = rtu_replay_gui.RxPanel()
    assert panel.format_combo.currentText() == "ASCII", (
        f"Round 31: RxPanel 默认格式应 ASCII, got '{panel.format_combo.currentText()}'"
    )


def test_format_combo_has_all_three_options():
    """验证 format_combo 仍含 3 个选项 (ASCII/hex/hex+ASCII)."""
    panel = rtu_replay_gui.RxPanel()
    items = panel.format_combo._items
    assert "ASCII" in items, "缺 ASCII 选项"
    assert "hex" in items, "缺 hex 选项"
    assert "hex+ASCII" in items, "缺 hex+ASCII 选项"


def test_append_bytes_single_line():
    """验证 append_bytes 输出单行 (时间 + 内容同行, 不换行)."""
    panel = rtu_replay_gui.RxPanel()
    panel.append_bytes(b"\x01\x02hi")  # 默认 ASCII 格式
    text = panel.text_edit.toPlainText()
    # 整段应是单行 (1 个 \n, 因为 toPlainText 在多行间插 \n; 单 append 单行 = 0 个 \n)
    newline_count = text.count("\n")
    assert newline_count == 0, (
        f"Round 31: append_bytes 应单行, 但有 {newline_count} 个 \\n. text={text!r}"
    )
    # 含时间戳 + 字节数 + 内容
    assert "RX (" in text, "缺 RX (...) 字节数标签"
    assert "] RX (" in text, "缺 [时间] RX 格式"
    # 默认 ASCII 格式, 内容 = ".hi" (0x01='.', 0x02='.', 'h', 'i')
    assert "..hi" in text, (
        f"ASCII 格式内容应 '.hi', got: {text!r}"
    )


def test_append_bytes_timestamp_and_content_same_line():
    """验证 时间戳 和 内容 在同一行 (用空格分隔, 不是 \\n 分隔)."""
    panel = rtu_replay_gui.RxPanel()
    panel.append_bytes(b"hello")
    text = panel.text_edit.toPlainText()
    # 关键: 不应出现 "] RX (5 B):\n" (旧两行格式)
    assert "]:\n" not in text and "RX (5 B):\n" not in text, (
        f"Round 31 BUG: 仍是两行显示 (时间换行后才显示内容): {text!r}"
    )
    # 应有 "] RX (5 B): hello" (同行)
    assert "] RX (5 B): hello" in text, (
        f"时间与内容应同行, 缺 '] RX (5 B): hello': {text!r}"
    )


def test_append_bytes_hex_format_still_works():
    """回归: hex 格式 (切到 hex) 仍正常工作 (Round 31 不破坏 hex 选项)."""
    panel = rtu_replay_gui.RxPanel()
    panel.format_combo.setCurrentText("hex")
    panel.append_bytes(b"\xde\xad\xbe\xef")
    text = panel.text_edit.toPlainText()
    assert "de ad be ef" in text, f"hex 格式内容缺失: {text!r}"
    # 仍是单行
    assert text.count("\n") == 0, f"hex 格式应为单行: {text!r}"


def test_append_bytes_hex_ascii_format_still_works():
    """回归: hex+ASCII 格式仍正常 (Round 31 不破坏第三个选项)."""
    panel = rtu_replay_gui.RxPanel()
    panel.format_combo.setCurrentText("hex+ASCII")
    panel.append_bytes(b"hi")
    text = panel.text_edit.toPlainText()
    # hex+ASCII 格式含 "|" 列
    assert "|" in text, f"hex+ASCII 应含 '|' 分隔符: {text!r}"


def test_append_bytes_byte_count_updates():
    """验证 字节计数仍累加 (Round 31 不破坏字节统计)."""
    panel = rtu_replay_gui.RxPanel()
    panel.append_bytes(b"abc")
    panel.append_bytes(b"defgh")
    assert panel.byte_count_label.text() == "8 bytes", (
        f"字节计数应累加到 8, got '{panel.byte_count_label.text()}'"
    )


def test_append_bytes_empty_no_op():
    """回归: 空字节仍跳过 (Round 31 不破空保护)."""
    panel = rtu_replay_gui.RxPanel()
    panel.append_bytes(b"")
    assert panel.byte_count_label.text() == "0 bytes"
    assert panel.text_edit.toPlainText() == ""


def test_clear_button_resets():
    """验证 清空按钮 仍重置 (Round 31 不破清空功能)."""
    panel = rtu_replay_gui.RxPanel()
    panel.append_bytes(b"abc")
    panel._on_clear()
    assert panel.byte_count_label.text() == "0 bytes"
    assert panel.text_edit.toPlainText() == ""


if __name__ == '__main__':
    test_default_format_is_ascii()
    print("[PASS] test_default_format_is_ascii")

    test_format_combo_has_all_three_options()
    print("[PASS] test_format_combo_has_all_three_options")

    test_append_bytes_single_line()
    print("[PASS] test_append_bytes_single_line")

    test_append_bytes_timestamp_and_content_same_line()
    print("[PASS] test_append_bytes_timestamp_and_content_same_line")

    test_append_bytes_hex_format_still_works()
    print("[PASS] test_append_bytes_hex_format_still_works (hex 回归)")

    test_append_bytes_hex_ascii_format_still_works()
    print("[PASS] test_append_bytes_hex_ascii_format_still_works (hex+ASCII 回归)")

    test_append_bytes_byte_count_updates()
    print("[PASS] test_append_bytes_byte_count_updates")

    test_append_bytes_empty_no_op()
    print("[PASS] test_append_bytes_empty_no_op")

    test_clear_button_resets()
    print("[PASS] test_clear_button_resets")

    print("\n所有 9 个 Round 31 测试通过!")
    print("RxPanel 默认 ASCII + 时间内容同行, 接收区更紧凑可读.")