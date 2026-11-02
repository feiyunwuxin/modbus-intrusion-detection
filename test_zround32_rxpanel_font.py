#!/usr/bin/env python3
"""验证 Round 32: RxPanel 文字显示不全修复.

⭐ 2026-10-02 第三十二轮:
  用户反馈接收区文字仍显示不全 (Round 31 把时间和内容合一行后, 长 hex+ASCII 行
  字符底端被裁切 / 上下行 padding 太紧).
  修复:
    1. QFont.setPointSize(10) (默认 8pt 太小, 字符底端裁切)
    2. QFont.setStyleHint(Monospace) (无 Consolas 时降级等宽)
    3. QTextEdit stylesheet: padding 4px 6px + line-height 1.4em (字符上下不挤)
    4. setLineWrapMode(WidgetWidth) (显式声明, 长行按宽度换行)
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

class _DummyFont:
    """QFont stub: 记录 setFamily / setPointSize / setStyleHint 调用."""
    # ⭐ QFont class attribute (QFont.Monospace = 7 in real PyQt)
    Monospace = 7

    def __init__(self, *a, **k):
        self._family = ""
        self._point_size = 0
        self._style_hint = None
    def setFamily(self, f): self._family = f
    def family(self): return self._family
    def setPointSize(self, s): self._point_size = s
    def pointSize(self): return self._point_size
    def setStyleHint(self, h, *a, **k): self._style_hint = h
    def styleHint(self): return self._style_hint

class _DummyTextEdit:
    """QTextEdit stub: 记录 stylesheet + LineWrapMode + font."""
    # ⭐ QTextEdit class attribute (WidgetWidth = 1 in real PyQt)
    WidgetWidth = 1
    NoWrap = 0

    def __init__(self, *a, **k):
        self._buf = []
        self._style = ""
        self._font = _DummyFont()  # ⭐ 必须默认 _DummyFont 实例, 不然 font() 返回 None
        self._line_wrap = None
        self._readonly = False
        self._cursor = type('Cursor', (), {
            'End': 0,
            'movePosition': lambda *a, **k: None,
            'ensureCursorVisible': lambda *a, **k: None,
        })()
    def append(self, s): self._buf.append(s)
    def toPlainText(self): return "\n".join(self._buf)
    def setReadOnly(self, e): self._readonly = bool(e)
    def isReadOnly(self): return self._readonly
    def setFont(self, f): self._font = f
    def font(self): return self._font
    def setStyleSheet(self, qss): self._style = qss
    def styleSheet(self): return self._style
    def setLineWrapMode(self, m): self._line_wrap = m
    def lineWrapMode(self): return self._line_wrap
    def textCursor(self): return self._cursor
    def setTextCursor(self, *a, **k): pass
    def ensureCursorVisible(self, *a, **k): pass
    def clear(self): self._buf = []

class _DummyWidget:
    def __init__(self, *a, **k):
        self._style = ""
        self._text = ""
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

class _DummyLabel(_DummyWidget):
    def __init__(self, text="", *a, **k):
        super().__init__(*a, **k)
        self._text = text

class _DummyComboBox(_DummyWidget):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._items = []
        self._current = 0
    def addItems(self, items): self._items = list(items)
    def currentText(self): return self._items[self._current] if self._items else ""
    def setCurrentText(self, t):
        if t in self._items: self._current = self._items.index(t)

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
# ⭐ QFont 必须存在 (Round 32 显式 import)
_dummy_qtgui.QFont = _DummyFont
_dummy_qtgui.Monospace = 7  # Qt 实际值: QFont.Monospace = 7 (随便填 dummy 值)
_dummy_qtwidgets = type(sys)('PyQt5.QtWidgets')
sys.modules['PyQt5.QtWidgets'] = _dummy_qtwidgets
_dummy_qtwidgets.QApplication = _DummyWidget
_dummy_qtwidgets.QWidget = _DummyWidget
_dummy_qtwidgets.QHBoxLayout = _DummyWidget
_dummy_qtwidgets.QVBoxLayout = _DummyWidget
_dummy_qtwidgets.QLabel = _DummyLabel
_dummy_qtwidgets.QPushButton = _DummyWidget
_dummy_qtwidgets.QPlainTextEdit = _DummyTextEdit
_dummy_qtwidgets.QTextEdit = _DummyTextEdit
_dummy_qtwidgets.QComboBox = _DummyComboBox
_dummy_qtwidgets.WidgetWidth = 1  # QTextEdit.WidgetWidth = 1
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
sys.modules.pop('rtu_replay_gui', None)  # 防 pytest 模块缓存
import rtu_replay_gui  # noqa: E402


# === Tests ===

def test_qfont_imported():
    """验证 QFont 已导入 (Round 32 修复需要)."""
    src = inspect.getsource(rtu_replay_gui)
    assert 'QFont' in src, "BUG: 源码未引用 QFont (Round 32 字体修复缺失)"


def test_qfont_setpointsize_10():
    """验证 font.setPointSize(10) — 防默认 8pt 字符裁切."""
    panel = rtu_replay_gui.RxPanel()
    font = panel.text_edit.font()
    assert font.pointSize() == 10, (
        f"BUG: font point size 应 10pt (防字符裁切), got {font.pointSize()}"
    )


def test_qfont_setstylehint_monospace():
    """验证 font.setStyleHint(QFont.Monospace) — 无 Consolas 时降级等宽."""
    panel = rtu_replay_gui.RxPanel()
    font = panel.text_edit.font()
    # QFont.Monospace = 7 in Qt (or our dummy value)
    assert font.styleHint() is not None, (
        "BUG: font style hint 未设置, 系统无 Consolas 时可能字体非等宽"
    )
    assert font.styleHint() == rtu_replay_gui.QFont.Monospace, (
        f"BUG: font style hint 应 Monospace, got {font.styleHint()}"
    )


def test_textedit_stylesheet_padding():
    """验证 text_edit stylesheet 含 padding + line-height (防字符上下挤)."""
    panel = rtu_replay_gui.RxPanel()
    qss = panel.text_edit.styleSheet()
    assert 'padding' in qss, "BUG: text_edit QSS 缺 padding (字符贴边)"
    assert 'line-height' in qss, "BUG: text_edit QSS 缺 line-height (字符行距紧)"


def test_textedit_linewrap_widget_width():
    """验证 setLineWrapMode(QTextEdit.WidgetWidth) — 长 hex+ASCII 行按宽度 wrap.
    注意: 不能直接用 QTextEdit.WidgetWidth 比较, 因为 pytest 跑多文件时,
    后加载的 _DummyTextEdit 覆盖了此文件的版本, 类属性 WidgetWidth 会丢.
    用字面值 1 (QTextEdit.WidgetWidth 在 PyQt5 实际值).
    """
    panel = rtu_replay_gui.RxPanel()
    wrap_mode = panel.text_edit.lineWrapMode()
    # QTextEdit.WidgetWidth 在 PyQt5 实际是 1
    assert wrap_mode == 1, (
        f"BUG: line wrap mode 应 WidgetWidth (1), got {wrap_mode}"
    )


def test_textedit_is_readonly():
    """回归: text_edit 仍只读 (Round 32 不破坏只读)."""
    panel = rtu_replay_gui.RxPanel()
    assert panel.text_edit.isReadOnly() is True


def test_append_bytes_still_single_line():
    """回归: append_bytes 仍单行 (Round 31 + Round 32 兼容)."""
    panel = rtu_replay_gui.RxPanel()
    panel.append_bytes(b"hello")
    text = panel.text_edit.toPlainText()
    assert text.count("\n") == 0, f"BUG: 应单行: {text!r}"


def test_append_bytes_hex_ascii_still_single_line():
    """回归: hex+ASCII 格式仍单行 (Round 31 行为保持)."""
    panel = rtu_replay_gui.RxPanel()
    panel.format_combo.setCurrentText("hex+ASCII")
    panel.append_bytes(b"\xde\xad\xbe\xef" * 4)  # 16 bytes
    text = panel.text_edit.toPlainText()
    assert text.count("\n") == 0, f"hex+ASCII 应单行: {text!r}"


if __name__ == '__main__':
    test_qfont_imported()
    print("[PASS] test_qfont_imported (QFont 已 import)")

    test_qfont_setpointsize_10()
    print("[PASS] test_qfont_setpointsize_10 (10pt 防裁切)")

    test_qfont_setstylehint_monospace()
    print("[PASS] test_qfont_setstylehint_monospace (Monospace 降级)")

    test_textedit_stylesheet_padding()
    print("[PASS] test_textedit_stylesheet_padding (padding + line-height)")

    test_textedit_linewrap_widget_width()
    print("[PASS] test_textedit_linewrap_widget_width (WidgetWidth wrap)")

    test_textedit_is_readonly()
    print("[PASS] test_textedit_is_readonly")

    test_append_bytes_still_single_line()
    print("[PASS] test_append_bytes_still_single_line (Round 31 兼容)")

    test_append_bytes_hex_ascii_still_single_line()
    print("[PASS] test_append_bytes_hex_ascii_still_single_line")

    print("\n所有 8 个 Round 32 测试通过!")
    print("QFont 10pt + Monospace 降级 + padding + line-height, 接收区字符不再裁切.")