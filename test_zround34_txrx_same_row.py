#!/usr/bin/env python3
"""验证 Round 34+37: 接收区与发送区同行 (Round 34 修复, Round 37 改布局).

⭐ 2026-10-02 第三十四轮:
  旧代码 bug: 'root.addLayout(self.tx_rx_row)' 在 rx_box 创建之后才调用,
  而 tx_rx_row 在那时还没加入 root, 导致 tx_rx_row (含 man_box) 被加到 root 底部.

  修复: 把 'root.addLayout(self.tx_rx_row)' 移到 man_box 加入 tx_rx_row 之后立即调用.

⭐ 2026-10-02 第三十六轮:
  用户改要求: 手动模式单独一行, stats_label + rx_box 同行横向.
  旧的 self.tx_rx_row 容器已移除 (man_box 不再与 rx_box 同行).

⭐ 2026-10-02 第三十七轮:
  用户再改要求: 接收区 + 日志 同行横向, stats_label 单独一行.
  旧的 self.stats_rx_row 容器已移除, 引入 self.log_rx_row.
  rx_box 与 log_box 在同一 QHBoxLayout.

  本测试更新为: 验证布局流程无重复 addLayout, 且 root 顺序符合 Round 37 规范.
"""
from __future__ import annotations
import os
import sys

# Dummy PyQt5
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

def test_no_duplicate_layout_adds():
    """验证 root 内的 layout_add() 不重复 (Round 34 修复保留)."""
    src_lines = open('rtu_replay_gui.py', encoding='utf-8').readlines()
    in_build_ui = False
    counts = {}
    for l in src_lines:
        if 'def _build_ui' in l:
            in_build_ui = True
        if in_build_ui:
            stripped = l.lstrip()
            if stripped.startswith('#'):
                continue
            if 'root.addLayout' in l:
                # 提取 layout 名称
                import re
                m = re.search(r'root\.addLayout\(([^)]+)\)', l)
                if m:
                    counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    # 所有 layout 都应只调 1 次 (Round 34 修复: 不能 root.addLayout(self.tx_rx_row) 调 2 次)
    for layout_name, cnt in counts.items():
        assert cnt == 1, (
            f"BUG: root.addLayout({layout_name}) 应只调 1 次, got {cnt} 次"
        )


def test_root_order_correct():
    """验证 root 顺序符合 Round 37 规范:
    csv_box → top_row → man_box → ctrl → progress_bar → stats_label → log_rx_row → ids_report.
    """
    import re
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

    # 关键顺序检查 (Round 37):
    # top_row → man_box → ctrl → progress_bar → stats_label → log_rx_row → ids_report
    top_idx = next((i for i, x in enumerate(order) if 'top_row' in x), -1)
    man_idx = next((i for i, x in enumerate(order) if x == 'man_box'), -1)
    ctrl_idx = next((i for i, x in enumerate(order) if x == 'ctrl'), -1)
    progress_idx = next((i for i, x in enumerate(order) if 'progress_bar' in x), -1)
    stats_idx = next((i for i, x in enumerate(order) if x == 'self.stats_label'), -1)
    log_rx_idx = next((i for i, x in enumerate(order) if 'log_rx_row' in x), -1)

    assert top_idx >= 0 and man_idx >= 0, "top_row / man_box 调用缺失"
    assert top_idx < man_idx, f"BUG: top_row ({top_idx}) 必须在 man_box ({man_idx}) 之前"
    assert man_idx < ctrl_idx, f"BUG: man_box ({man_idx}) 必须在 ctrl ({ctrl_idx}) 之前"
    assert ctrl_idx < progress_idx, (
        f"BUG: ctrl ({ctrl_idx}) 必须在 progress ({progress_idx}) 之前"
    )
    assert progress_idx < stats_idx, (
        f"BUG: progress ({progress_idx}) 必须在 stats_label ({stats_idx}) 之前 "
        "(Round 37: stats_label 单独一行, 在 progress 下方)"
    )
    assert stats_idx < log_rx_idx, (
        f"BUG: stats_label ({stats_idx}) 必须在 log_rx_row ({log_rx_idx}) 之前 "
        "(Round 37: log_rx_row 在 stats_label 下方)"
    )


def test_rx_box_in_log_rx_row_not_tx_rx_row():
    """验证 rx_box 加入 log_rx_row (Round 37) 而非 tx_rx_row (Round 30/34 旧)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.log_rx_row.addWidget(rx_box' in code_only, (
        "BUG: rx_box 未加入 log_rx_row (Round 37 应在 log_rx_row)"
    )
    # 旧的 tx_rx_row.addWidget(rx_box, ...) 不应存在
    assert 'self.tx_rx_row.addWidget(rx_box' not in code_only, (
        "BUG: rx_box 仍加入旧 tx_rx_row (Round 36 应迁移到 stats_rx_row, "
        "Round 37 又迁移到 log_rx_row)"
    )


def test_old_tx_rx_row_removed():
    """验证旧 self.tx_rx_row 容器已删除 (Round 36 移除)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.tx_rx_row = QHBoxLayout()' not in code_only, (
        "BUG: self.tx_rx_row 仍创建 (Round 36 应移除)"
    )


def test_old_stats_rx_row_removed():
    """⭐ Round 37 新增: 旧 self.stats_rx_row 容器也已删除."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'self.stats_rx_row = QHBoxLayout()' not in code_only, (
        "BUG: self.stats_rx_row 仍创建 (Round 37 应替换为 log_rx_row)"
    )


if __name__ == '__main__':
    test_no_duplicate_layout_adds()
    print("[PASS] test_no_duplicate_layout_adds (无重复 addLayout)")

    test_root_order_correct()
    print("[PASS] test_root_order_correct (root 顺序符合 Round 37)")

    test_rx_box_in_log_rx_row_not_tx_rx_row()
    print("[PASS] test_rx_box_in_log_rx_row_not_tx_rx_row (rx_box 在 log_rx_row)")

    test_old_tx_rx_row_removed()
    print("[PASS] test_old_tx_rx_row_removed (旧 tx_rx_row 容器移除)")

    test_old_stats_rx_row_removed()
    print("[PASS] test_old_stats_rx_row_removed (旧 stats_rx_row 容器移除)")

    print("\n所有 5 个 Round 34+37 测试通过!")
    print("布局重构: log + rx 同行横向, stats_label 单独一行, 无重复 addLayout.")