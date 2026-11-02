#!/usr/bin/env python3
"""验证 Round 32: '📊 帧大小分布' 按钮位置移动.

⭐ 2026-10-02 第三十二轮:
  用户反馈: '帧大小分布' 应放在 '关闭并写 summary' 之后 (与日志功能按钮集中).
  旧: dist_btn 在 stats_row (进度条下方, 与 stats_label 同一行).
  新: dist_btn 在 log_toolbar (日志工具栏), 顺序:
       [清空日志] [开启 CSV 日志] [关闭并写 summary] [帧大小分布] (stretch)
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
_dummy_qtwidgets.QHBoxLayout = _DummyWidget
_dummy_qtwidgets.QVBoxLayout = _DummyWidget
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
_dummy_qtwidgets.QGroupBox = _DummyWidget
_dummy_qtwidgets.QFrame = _DummyWidget
_dummy_qtwidgets.QSizePolicy = _DummyWidget
_dummy_qtwidgets.QLineEdit = _DummyWidget
_dummy_qtwidgets.QTabWidget = _DummyWidget
_dummy_qtwidgets.QGridLayout = _DummyWidget
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

def test_dist_btn_not_in_stats_row():
    """验证 dist_btn 不在 stats_row (进度条下方)."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # dist_btn 创建后, 不能再 stats_row.addWidget(self.dist_btn)
    # 找 dist_btn 创建行
    dist_create_idx = src.find('self.dist_btn = QPushButton')
    assert dist_create_idx >= 0, "BUG: dist_btn 未创建"
    # dist_btn 创建之后的 250 行 (足够看到 stats_row 末尾 + log_toolbar 起点)
    block_after = src[dist_create_idx:dist_create_idx + 600]
    # 验证 stats_row.addWidget(self.dist_btn) 不存在
    assert 'stats_row.addWidget(self.dist_btn)' not in block_after, (
        "BUG: dist_btn 仍在 stats_row (Round 32 应移到 log_toolbar)"
    )


def test_dist_btn_in_log_toolbar():
    """验证 dist_btn 在 log_toolbar 中."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 找 log_toolbar.addWidget(self.dist_btn) 调用
    assert 'log_toolbar.addWidget(self.dist_btn)' in src, (
        "BUG: dist_btn 没加入 log_toolbar (Round 32 改动缺失)"
    )


def test_dist_btn_after_log_close_btn():
    """验证 dist_btn 加在 log_close_btn 之后 (用户要求顺序)."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 找 log_close_btn 加 log_toolbar 的位置
    log_close_idx = src.find('log_toolbar.addWidget(self.log_close_btn)')
    assert log_close_idx >= 0, "BUG: log_close_btn 未加入 log_toolbar"
    # dist_btn 加 log_toolbar 的位置
    dist_idx = src.find('log_toolbar.addWidget(self.dist_btn)')
    assert dist_idx >= 0, "BUG: dist_btn 未加入 log_toolbar"
    # dist_btn 必须在 log_close_btn 之后
    assert dist_idx > log_close_idx, (
        f"BUG: dist_btn (idx={dist_idx}) 必须在 log_close_btn (idx={log_close_idx}) 之后"
    )


def test_dist_btn_still_exists():
    """回归: dist_btn 仍作为 self.dist_btn 创建 (没改名)."""
    panel_attr = 'dist_btn'
    # 通过源码确认 self.dist_btn 仍创建
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'self.dist_btn = QPushButton' in src, (
        "BUG: self.dist_btn 未创建"
    )
    assert '📊 帧大小分布' in src, (
        "BUG: dist_btn 文本丢失"
    )


if __name__ == '__main__':
    test_dist_btn_not_in_stats_row()
    print("[PASS] test_dist_btn_not_in_stats_row (从 stats_row 移出)")

    test_dist_btn_in_log_toolbar()
    print("[PASS] test_dist_btn_in_log_toolbar (加入 log_toolbar)")

    test_dist_btn_after_log_close_btn()
    print("[PASS] test_dist_btn_after_log_close_btn (顺序在 summary 后)")

    test_dist_btn_still_exists()
    print("[PASS] test_dist_btn_still_exists (按钮保留)")

    print("\n所有 4 个 Round 32 (按钮顺序) 测试通过!")
    print("'Frame Size Distribution' moved to after 'Close & Write Summary' in log_toolbar.")