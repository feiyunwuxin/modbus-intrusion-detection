#!/usr/bin/env python3
"""验证 Round 37: 手动模式卡片内部布局拆分 — preview LEFT, 所有按钮 RIGHT.

⭐ 2026-10-02 第三十七轮:
  用户要求新布局:
    1. man_box 卡片内: 帧预览在左侧, 所有按钮 (row1..row4) 在右侧 stacked
    2. 接收区 + 日志 同行横向排布 (在另一个测试 zround36 验证)

  手动模式卡片内部结构变化:
    旧 (QVBoxLayout): row1 → row2 → preview_label → row3 → row4 (全部 stacked 垂直)
    新 (QHBoxLayout):
      ├── left_col (QVBoxLayout, stretch=1):
      │     └── preview_label (150px 高)
      └── right_col (QVBoxLayout, stretch=2):
            ├── row1 (pos/cat)
            ├── row2 (status+load_btn)
            ├── row3 (nav: prev/next/resend/goto)
            └── row4 (manage: reset/reload/unload)
"""
from __future__ import annotations
import os
import sys
import re

# Dummy PyQt5 (与其他测试一致)
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
    def setStyleSheet(self, qss): self._style = qss
    def styleSheet(self): return self._style
    def setText(self, t): self._text = t
    def text(self): return self._text
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass
    def setMaximumHeight(self, *a, **k): pass
    def setMinimumHeight(self, *a, **k): pass
    def setWordWrap(self, *a, **k): pass

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

def _get_man_box_code():
    """提取 man_box 内部的代码段 (从 'man_box = QGroupBox' 到 'root.addWidget(man_box)')."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    # 找到 man_box 定义
    start = src.find('man_box = QGroupBox("手动模式')
    assert start > 0, "BUG: man_box 定义未找到"
    # 找到 man_box addWidget (调用根)
    end = src.find('root.addWidget(man_box)', start)
    assert end > 0, "BUG: man_box addWidget 未找到"
    return src[start:end + len('root.addWidget(man_box)')]


def test_man_is_hbox_layout():
    """验证 man = QHBoxLayout (Round 37 改动)."""
    code = _get_man_box_code()
    assert 'man = QHBoxLayout(man_box)' in code, (
        "BUG: man 仍 QVBoxLayout (Round 37 应改 QHBoxLayout)"
    )


def test_man_not_qvbox_layout():
    """验证 man 不是 QVBoxLayout (Round 37 移除)."""
    code = _get_man_box_code()
    # 注意: 内层 left_col / right_col 仍可能是 QVBoxLayout, 这里只看 man 主布局
    lines = [l for l in code.splitlines() if 'man = ' in l and 'Layout' in l]
    for line in lines:
        # 'man = QHBoxLayout' 才正确
        assert 'QVBoxLayout' not in line, (
            f"BUG: man 主布局出现 QVBoxLayout: {line}"
        )


def test_left_col_contains_preview_label():
    """验证 left_col (QVBoxLayout) 包含 preview_label."""
    code = _get_man_box_code()
    assert 'left_col = QVBoxLayout()' in code, (
        "BUG: left_col 未创建"
    )
    # left_col.addWidget(preview_label, ...)
    assert re.search(r'left_col\.addWidget\(self\.preview_label', code), (
        "BUG: preview_label 未加入 left_col"
    )


def test_left_col_added_to_man():
    """验证 left_col 加入 man with stretch=1."""
    code = _get_man_box_code()
    assert re.search(r'man\.addLayout\(left_col,\s*stretch=1\)', code), (
        "BUG: man.addLayout(left_col, stretch=1) 未调用 (left_col 应在左, stretch=1)"
    )


def test_right_col_contains_row1_row2_row3_row4():
    """验证 right_col (QVBoxLayout) 包含 row1, row2, row3, row4."""
    code = _get_man_box_code()
    assert 'right_col = QVBoxLayout()' in code, (
        "BUG: right_col 未创建"
    )
    # 4 个 row 都要 right_col.addLayout
    assert re.search(r'right_col\.addLayout\(row1\)', code), (
        "BUG: right_col 未加 row1 (pos/cat)"
    )
    assert re.search(r'right_col\.addLayout\(row2\)', code), (
        "BUG: right_col 未加 row2 (status+load)"
    )
    assert re.search(r'right_col\.addLayout\(row3\)', code), (
        "BUG: right_col 未加 row3 (nav)"
    )
    assert re.search(r'right_col\.addLayout\(row4\)', code), (
        "BUG: right_col 未加 row4 (manage)"
    )


def test_right_col_added_to_man_stretch2():
    """验证 right_col 加入 man with stretch=2 (给按钮足够宽度)."""
    code = _get_man_box_code()
    assert re.search(r'man\.addLayout\(right_col,\s*stretch=2\)', code), (
        "BUG: man.addLayout(right_col, stretch=2) 未调用 "
        "(right_col 应在右, stretch=2 给按钮宽度)"
    )


def test_preview_label_minheight_150():
    """回归: preview_label 仍 setMinimumHeight(150) (Round 33 不破)."""
    code = _get_man_box_code()
    assert 'self.preview_label.setMinimumHeight(150)' in code, (
        "BUG: preview_label minHeight 150 丢失 (Round 33 修复被破)"
    )


def test_old_man_addwidget_preview_removed():
    """验证旧的 'man.addWidget(self.preview_label)' 已移除 (应改用 left_col.addWidget)."""
    code = _get_man_box_code()
    assert 'man.addWidget(self.preview_label)' not in code, (
        "BUG: preview_label 仍直接 man.addWidget (Round 37 应改为 left_col.addWidget)"
    )


def test_old_man_addlayout_row_removed():
    """验证旧的 'man.addLayout(row1)'..row4 已移除 (应改用 right_col.addLayout)."""
    code = _get_man_box_code()
    assert 'man.addLayout(row1)' not in code, (
        "BUG: row1 仍直接 man.addLayout (Round 37 应改 right_col.addLayout)"
    )
    assert 'man.addLayout(row2)' not in code, (
        "BUG: row2 仍直接 man.addLayout"
    )
    assert 'man.addLayout(row3)' not in code, (
        "BUG: row3 仍直接 man.addLayout"
    )
    assert 'man.addLayout(row4)' not in code, (
        "BUG: row4 仍直接 man.addLayout"
    )


def test_round37_marker_comment():
    """验证源码有 Round 37 标记注释."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    assert '第三十七轮' in src, "BUG: 缺 Round 37 标记注释"
    assert 'preview LEFT' in src or '帧预览在左侧' in src, (
        "BUG: 缺 preview LEFT 说明注释"
    )


if __name__ == '__main__':
    test_man_is_hbox_layout()
    print("[PASS] test_man_is_hbox_layout (man = QHBoxLayout)")

    test_man_not_qvbox_layout()
    print("[PASS] test_man_not_qvbox_layout (man 主布局非 QVBoxLayout)")

    test_left_col_contains_preview_label()
    print("[PASS] test_left_col_contains_preview_label (preview_label 在 left_col)")

    test_left_col_added_to_man()
    print("[PASS] test_left_col_added_to_man (left_col stretch=1)")

    test_right_col_contains_row1_row2_row3_row4()
    print("[PASS] test_right_col_contains_row1_row2_row3_row4 (4 个 row 在 right_col)")

    test_right_col_added_to_man_stretch2()
    print("[PASS] test_right_col_added_to_man_stretch2 (right_col stretch=2)")

    test_preview_label_minheight_150()
    print("[PASS] test_preview_label_minheight_150 (minHeight 150 回归)")

    test_old_man_addwidget_preview_removed()
    print("[PASS] test_old_man_addwidget_preview_removed")

    test_old_man_addlayout_row_removed()
    print("[PASS] test_old_man_addlayout_row_removed")

    test_round37_marker_comment()
    print("[PASS] test_round37_marker_comment (Round 37 注释存在)")

    print("\n所有 10 个 Round 37 测试通过!")
    print("man_box 内部: preview LEFT (left_col stretch=1) + 所有按钮 RIGHT "
          "(right_col stretch=2, row1-4 stacked).")