#!/usr/bin/env python3
"""验证 Round 36+37: 布局重组 — 手动模式独立一行, 接收区+日志同行.

⭐ 2026-10-02 第三十六轮 (历史):
  用户要求新布局:
    1. 手动模式卡片 (man_box) 单独一行 (全宽)
    2. "已发送" stats_label + 接收区 (rx_box) 同行横向排布
    3. 移除旧的 self.tx_rx_row (Round 30/34 引入的 man+rx 同行容器)

  Round 36 root 顺序:
    1. csv_box
    2. top_row (port + opt)
    3. man_box            ← 单独一行
    4. ctrl (start/pause/stop)
    5. progress_bar
    6. stats_rx_row       ← stats_label | rx_box 横向
    7. log_box
    8. ids_report

⭐ 2026-10-02 第三十七轮 (当前):
  用户进一步要求:
    1. 手动模式卡片内部: 帧预览 LEFT, 所有按钮 RIGHT stacked
    2. 接收区 + 日志 同行横向排布 (log_rx_row)
    3. stats_label 单独一行 (不再与 rx 同行)
    4. 旧的 self.stats_rx_row 容器已废弃

  Round 37 root 顺序:
    1. csv_box
    2. top_row (port + opt)
    3. man_box            ← 单独一行, 内部 QHBoxLayout
    4. ctrl
    5. progress_bar
    6. stats_label        ← 单独一行 (全宽, padding 突出)
    7. log_rx_row         ← rx_box | log_box 横向
    8. ids_report

本测试验证 Round 37 当前生效的最终布局 (Round 36 测试已迁移到 Round 37 规范).
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
    def styleSheet(self): return self._style  # ⭐ zround29 兼容性
    def setText(self, t): self._text = t
    def text(self): return self._text
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


# === Tests (Round 37 更新) ===

def test_no_tx_rx_row_attribute():
    """验证 self.tx_rx_row 已删除 (Round 36 移除)."""
    import inspect
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 排除注释行
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.tx_rx_row = QHBoxLayout' not in code_only, (
        "BUG: self.tx_rx_row 仍创建 (Round 36 应移除)"
    )
    assert 'self.tx_rx_row.addWidget' not in code_only, (
        "BUG: self.tx_rx_row.addWidget 仍存在 (Round 36 应移除)"
    )


def test_no_stats_rx_row_attribute():
    """⭐ Round 37 新增: self.stats_rx_row 已替换为 log_rx_row."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui) if False else open(
        'rtu_replay_gui.py', encoding='utf-8'
    ).read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.stats_rx_row = QHBoxLayout' not in code_only, (
        "BUG: self.stats_rx_row 仍创建 (Round 37 应替换为 log_rx_row)"
    )


def test_log_rx_row_attribute_exists():
    """验证 self.log_rx_row 创建 (Round 37 新容器)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.log_rx_row = QHBoxLayout' in code_only, (
        "BUG: self.log_rx_row 未创建 (Round 37 应加)"
    )


def test_log_rx_row_in_root():
    """验证 log_rx_row 加入 root (Round 37 必须)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'root.addLayout(self.log_rx_row' in code_only, (
        "BUG: root.addLayout(self.log_rx_row) 未调用"
    )


def test_root_order_round37():
    """验证 root 顺序符合 Round 37 规范:
    top_row → man_box → ctrl → progress_bar → stats_label → log_rx_row → ids_report.
    """
    src_lines = open('rtu_replay_gui.py', encoding='utf-8').readlines()
    in_build_ui = False
    order = []
    for l in src_lines:
        if 'def _build_ui' in l:
            in_build_ui = True
        if in_build_ui:
            stripped = l.lstrip()
            if stripped.startswith('#'):
                continue
            for m in re.finditer(r'root\.(addLayout|addWidget)\(([^)]+)\)', l):
                order.append(m.group(2).strip())

    top_idx = next((i for i, x in enumerate(order) if 'top_row' in x), -1)
    man_idx = next((i for i, x in enumerate(order) if x == 'man_box'), -1)
    ctrl_idx = next((i for i, x in enumerate(order) if x == 'ctrl'), -1)
    progress_idx = next((i for i, x in enumerate(order) if 'progress_bar' in x), -1)
    stats_idx = next((i for i, x in enumerate(order) if x == 'self.stats_label'), -1)
    log_rx_idx = next((i for i, x in enumerate(order) if 'log_rx_row' in x), -1)

    assert top_idx > 0, "top_row addLayout 缺失"
    assert man_idx > 0, "man_box addWidget 缺失"
    assert ctrl_idx > 0, "ctrl addLayout 缺失"
    assert progress_idx > 0, "progress_bar addWidget 缺失"
    assert stats_idx > 0, "stats_label addWidget 缺失"
    assert log_rx_idx > 0, "log_rx_row addLayout 缺失"

    # Round 37 顺序: top_row < man < ctrl < progress < stats_label < log_rx_row
    assert top_idx < man_idx, (
        f"BUG: top_row ({top_idx}) 必须在 man_box ({man_idx}) 之前"
    )
    assert man_idx < ctrl_idx, (
        f"BUG: man_box ({man_idx}) 必须在 ctrl ({ctrl_idx}) 之前"
    )
    assert ctrl_idx < progress_idx, (
        f"BUG: ctrl ({ctrl_idx}) 必须在 progress ({progress_idx}) 之前"
    )
    assert progress_idx < stats_idx, (
        f"BUG: progress ({progress_idx}) 必须在 stats_label ({stats_idx}) 之前"
    )
    assert stats_idx < log_rx_idx, (
        f"BUG: stats_label ({stats_idx}) 必须在 log_rx_row ({log_rx_idx}) 之前"
    )


def test_rx_box_in_log_rx_row():
    """验证 rx_box 加入 log_rx_row (Round 37)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.log_rx_row.addWidget(rx_box' in code_only, (
        "BUG: rx_box 未加入 log_rx_row (Round 37 应在 log_rx_row)"
    )


def test_log_box_in_log_rx_row():
    """⭐ Round 37 新增: log_box 加入 log_rx_row (而非独立 addWidget)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.log_rx_row.addWidget(log_box' in code_only, (
        "BUG: log_box 未加入 log_rx_row (Round 37 应在 log_rx_row)"
    )


def test_stats_label_standalone_row():
    """⭐ Round 37 新增: stats_label 通过 root.addWidget 单独一行 (非 mix 内的 self.xx_row)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    # stats_label 应直接通过 root.addWidget 加入 (而非通过 stats_rx_row.addWidget)
    assert 'root.addWidget(self.stats_label' in code_only, (
        "BUG: stats_label 未直接 root.addWidget (Round 37 应单独一行)"
    )
    # 不应再通过 stats_rx_row.addWidget 加入
    assert 'self.stats_rx_row.addWidget(self.stats_label' not in code_only, (
        "BUG: stats_label 仍通过旧 stats_rx_row.addWidget 加入 (Round 37 已废弃)"
    )


if __name__ == '__main__':
    test_no_tx_rx_row_attribute()
    print("[PASS] test_no_tx_rx_row_attribute (旧 tx_rx_row 容器移除)")

    test_no_stats_rx_row_attribute()
    print("[PASS] test_no_stats_rx_row_attribute (旧 stats_rx_row 容器也移除)")

    test_log_rx_row_attribute_exists()
    print("[PASS] test_log_rx_row_attribute_exists (log_rx_row 容器加)")

    test_log_rx_row_in_root()
    print("[PASS] test_log_rx_row_in_root (log_rx_row 加入 root)")

    test_root_order_round37()
    print("[PASS] test_root_order_round37 (root 顺序: top→man→ctrl→progress→stats→log_rx)")

    test_rx_box_in_log_rx_row()
    print("[PASS] test_rx_box_in_log_rx_row (rx_box 在 log_rx_row)")

    test_log_box_in_log_rx_row()
    print("[PASS] test_log_box_in_log_rx_row (log_box 也在 log_rx_row)")

    test_stats_label_standalone_row()
    print("[PASS] test_stats_label_standalone_row (stats_label 单独一行)")

    print("\n所有 8 个 Round 36+37 测试通过!")
    print("手动模式独立一行, 接收区+日志同行横向, stats_label 单独一行.")