#!/usr/bin/env python3
"""验证 Round 33: 帧预览方框放大 + 当前位置/类别上移.

⭐ 2026-10-02 第三十三轮:
  用户反馈: 帧预览方框太小 (96px), 长帧 (>=32 字节) 只能显示 4 行 hex, 信息不全.
            '当前位置' / '类别' 行应上移到顶部 (状态信息优先).
  修复:
    1. preview_label 高度 96px → 150px (容 6+ 行 hex + metadata)
    2. preview_label padding 6px → 8px (内部更宽松)
    3. man_box 内 row 顺序调整:
       旧: [状态 + 加载] → [位置 + 类别] → [预览]
       新: [位置 + 类别] → [状态 + 加载] → [预览]
"""
from __future__ import annotations
import os
import sys

# === Dummy PyQt5 ===
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
        self._min_h = 0
        self._max_h = 16777215
        self._min_w = 0
        self._max_w = 16777215
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
    def setMinimumHeight(self, h): self._min_h = h
    def setMaximumHeight(self, h): self._max_h = h
    def setMinimumWidth(self, w): self._min_w = w
    def setMaximumWidth(self, w): self._max_w = w
    def minimumHeight(self): return self._min_h
    def setWordWrap(self, *a, **k): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass

class _DummyLabel(_DummyWidget):
    def __init__(self, text="", *a, **k):
        super().__init__(*a, **k)
        self._text = text

class _DummyLayout:
    def __init__(self, *a, **k):
        self._items = []  # 记录加入的 widget/layout (顺序敏感)
    def addWidget(self, w, *a, **k):
        self._items.append(('widget', w))
    def addLayout(self, l, *a, **k):
        self._items.append(('layout', l))
    def addStretch(self, *a, **k):
        self._items.append(('stretch', None))
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass

class _DummyGroupBox(_DummyWidget):
    def __init__(self, title="", *a, **k):
        super().__init__(*a, **k)
        self._title = title

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
_dummy_qtgui.QFont = type('QFont', (), {})
_dummy_qtgui.Monospace = 7
_dummy_qtwidgets = type(sys)('PyQt5.QtWidgets')
sys.modules['PyQt5.QtWidgets'] = _dummy_qtwidgets
_dummy_qtwidgets.QApplication = _DummyWidget
_dummy_qtwidgets.QWidget = _DummyWidget
_dummy_qtwidgets.QHBoxLayout = _DummyLayout
_dummy_qtwidgets.QVBoxLayout = _DummyLayout
_dummy_qtwidgets.QLabel = _DummyLabel
_dummy_qtwidgets.QPushButton = _DummyWidget
_dummy_qtwidgets.QPlainTextEdit = _DummyWidget
_dummy_qtwidgets.QTextEdit = _DummyWidget
_dummy_qtwidgets.QComboBox = _DummyWidget
_dummy_qtwidgets.WidgetWidth = 1
_dummy_qtwidgets.QSpinBox = _DummyWidget
_dummy_qtwidgets.QDoubleSpinBox = _DummyWidget
_dummy_qtwidgets.QFileDialog = _DummyWidget
_dummy_qtwidgets.QMessageBox = _DummyWidget
_dummy_qtwidgets.QMainWindow = _DummyWidget
_dummy_qtwidgets.QDialog = _DummyWidget
_dummy_qtwidgets.QCheckBox = _DummyWidget
_dummy_qtwidgets.QRadioButton = _DummyWidget
_dummy_qtwidgets.QGroupBox = _DummyGroupBox
_dummy_qtwidgets.QFrame = _DummyWidget
_dummy_qtwidgets.QSizePolicy = _DummyWidget
_dummy_qtwidgets.QLineEdit = _DummyWidget
_dummy_qtwidgets.QTabWidget = _DummyWidget
_dummy_qtwidgets.QGridLayout = _DummyLayout
_dummy_qtwidgets.QStatusBar = _DummyWidget
_dummy_qtwidgets.QMenuBar = _DummyWidget
_dummy_qtwidgets.QAction = _DummyWidget
_dummy_qtwidgets.QListWidget = _DummyWidget
_dummy_qtwidgets.QTreeWidget = _DummyWidget
_dummy_qtwidgets.QTableWidget = _DummyWidget
_dummy_qtwidgets.QHeaderView = _DummyWidget
_dummy_qtwidgets.QProgressBar = _DummyWidget
_dummy_qtwidgets.QSlider = _DummyWidget
_dummy_qtwidgets.QScrollArea = _DummyWidget  # ⭐ 第四十一轮 (2026-10-02): QScrollArea stub

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

def test_preview_label_height_150():
    """验证 preview_label min/maxHeight = 150 (Round 33 放大)."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 找 preview_label.setMinimumHeight
    min_idx = src.find('self.preview_label.setMinimumHeight(150)')
    assert min_idx >= 0, "BUG: preview_label.setMinimumHeight(150) 不存在"
    max_idx = src.find('self.preview_label.setMaximumHeight(150)')
    assert max_idx >= 0, "BUG: preview_label.setMaximumHeight(150) 不存在"


def test_preview_label_old_96_removed():
    """验证旧 96px 高度已移除."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 找 preview_label.setMinimumHeight(96) 应为 96 行不存在
    # 允许注释中有 96 (旧高度说明)
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'setMinimumHeight(96)' not in code_only or 'preview_label' not in code_only.split('setMinimumHeight(96)')[0][-200:], (
        "BUG: preview_label.setMinimumHeight(96) 旧值仍在代码中"
    )


def test_pos_cat_moved_to_top():
    """验证 '当前位置' / '类别' 行在 '手动模式状态/加载' 之前."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 找 row1 创建: 第一行现在应是 pos_label
    # 找 "row1.addWidget(self.pos_label"
    pos_label_idx = src.find('row1.addWidget(self.pos_label')
    assert pos_label_idx >= 0, "BUG: pos_label 未加入 row1 (Round 33 应在顶部 row1)"
    # 找 status_label 加入行
    status_label_idx = src.find('row2.addWidget(self.manual_status_label')
    assert status_label_idx >= 0, "BUG: manual_status_label 未在 row2"
    # 顺序: row1 (pos) 必须在 row2 (status) 之前
    assert pos_label_idx < status_label_idx, (
        f"BUG: pos_label (idx={pos_label_idx}) 必须在 status_label (idx={status_label_idx}) 之前"
    )


def test_old_row2_pos_label_removed():
    """验证旧的 'row2.addWidget(self.pos_label)' (pos 在 row2) 已移除."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'row2.addWidget(self.pos_label)' not in src, (
        "BUG: pos_label 仍在 row2 (Round 33 应移到 row1 顶部)"
    )


def test_status_label_in_row2():
    """验证 manual_status_label 现在在 row2 (不是 row1)."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # row2 应只包含 manual_status_label (load_btn 已在 Round 38 移到 row1)
    row2_start = src.find('row2 = QHBoxLayout()')
    assert row2_start >= 0, "BUG: row2 未创建"
    # ⭐ Round 37: row2 加入 right_col 而非 man (man_box 改 QHBoxLayout)
    row2_end = src.find('right_col.addLayout(row2)')
    if row2_end < 0:
        row2_end = src.find('man.addLayout(row2)')
    row2_block = src[row2_start:row2_end]
    assert 'manual_status_label' in row2_block, (
        "BUG: manual_status_label 不在 row2 (Round 33 应在 row2)"
    )
    # ⭐ Round 38: load_btn 已从 row2 移到 row1, 不应在 row2_block
    assert 'load_btn' not in row2_block, (
        "BUG: load_btn 仍在 row2 (Round 38 应移到 row1 最左)"
    )


def test_load_btn_in_row1():
    """⭐ Round 38: 加载按钮移到 row1 最左侧, 在 '当前位置' 之前."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # row1 应包含 load_btn + 当前位置: + pos_label + cat_label
    row1_start = src.find('row1 = QHBoxLayout()')
    assert row1_start >= 0, "BUG: row1 未创建"
    row1_end = src.find('man.addLayout(row1)')
    row1_block = src[row1_start:row1_end]
    assert 'load_btn' in row1_block, (
        "BUG: load_btn 不在 row1 (Round 38 应在 row1 最左)"
    )
    assert 'self.pos_label' in row1_block, (
        "BUG: pos_label 不在 row1"
    )
    # 关键: load_btn.addWidget 必须在 "当前位置:" addWidget 之前
    # 找两个 addWidget 调用
    import re
    load_idx = row1_block.find('addWidget(self.load_btn)')
    pos_text_idx = row1_block.find('addWidget(QLabel("当前位置')
    assert load_idx >= 0, "BUG: load_btn 未 addWidget"
    assert pos_text_idx >= 0, "BUG: 当前位置 label 未 addWidget"
    assert load_idx < pos_text_idx, (
        f"BUG: load_btn addWidget ({load_idx}) 必须在 当前位置 ({pos_text_idx}) 之前 "
        "(Round 38: 加载按钮在当前位置左边)"
    )


def test_preview_label_padding_increased():
    """验证 preview_label padding 增加到 8px (Round 33 改)."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 找 preview_label stylesheet
    idx = src.find('self.preview_label.setStyleSheet')
    assert idx >= 0, "BUG: preview_label stylesheet 未设"
    block = src[idx:idx + 200]
    assert 'padding: 8px' in block, (
        f"BUG: preview_label padding 应 8px, 当前: {block}"
    )


if __name__ == '__main__':
    test_preview_label_height_150()
    print("[PASS] test_preview_label_height_150 (150px)")

    test_preview_label_old_96_removed()
    print("[PASS] test_preview_label_old_96_removed (旧 96 移除)")

    test_pos_cat_moved_to_top()
    print("[PASS] test_pos_cat_moved_to_top (pos/cat 在 row1 顶部)")

    test_old_row2_pos_label_removed()
    print("[PASS] test_old_row2_pos_label_removed (旧 row2.pos 移除)")

    test_status_label_in_row2()
    print("[PASS] test_status_label_in_row2 (status 在 row2, load 移到 row1)")

    test_load_btn_in_row1()
    print("[PASS] test_load_btn_in_row1 (load_btn 在 row1 最左, 当前位置之前)")

    test_preview_label_padding_increased()
    print("[PASS] test_preview_label_padding_increased (padding 8px)")

    print("\n所有 6 个 Round 33 测试通过!")
    print("帧预览 96px -> 150px + 位置/类别上移, 长帧可视性显著提升.")