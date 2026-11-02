#!/usr/bin/env python3
"""验证 Round 39: 按钮重排 + _start 自动清空手动模式.

⭐ 2026-10-02 第三十九轮:
  用户要求 3 个改动:
    1. row3 (导航行): "发送并前进" 放在 "上一帧" 之前 (主操作靠左)
    2. row4 (管理行): "重发当前帧" 从 row3 移到 row4, 放在 "回到首帧" 之前
       (重发是行级操作, 与回到首帧/重新加载/清空加载同组)
    3. _start() 启动自动模式时, 若 manual_loaded=True, 先自动调 _manual_unload()
       用户选择: 自动清空手动模式后再启动 (无需手动点清空)
"""
from __future__ import annotations
import os
import sys

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
    def setMaximum(self, *a, **k): pass

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


def _get_row_block(row_name: str) -> str:
    """提取指定 row 的代码块 (从 'rowN = QHBoxLayout()' 到 'right_col.addLayout(rowN)')."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    pat = f'{row_name} = QHBoxLayout()'
    start = src.find(pat)
    assert start > 0, f"BUG: {row_name} 未创建"
    end_pat = f'right_col.addLayout({row_name})'
    end = src.find(end_pat, start)
    if end < 0:
        end = src.find(f'man.addLayout({row_name})', start)
    assert end > 0, f"BUG: {row_name} 未加入 right_col"
    return src[start:end]


def _find_addwidget(block: str, attr_name: str) -> int:
    """找 'addWidget(self.<attr_name>)' 在 block 中的位置. -1 = 没找到."""
    return block.find(f'addWidget(self.{attr_name})')


def _find_btn_decl(block: str, attr_name: str) -> int:
    """找 'self.<attr_name> = QPushButton' (按钮创建语句) 在 block 中的位置."""
    return block.find(f'self.{attr_name} = QPushButton')


# === Tests ===

def test_row3_send_next_before_prev():
    """验证 row3: '发送并前进' (send_next) 在 '上一帧' (prev) 之前."""
    block = _get_row_block('row3')
    # 按钮通过 for 循环 addWidget, 用 QPushButton 创建顺序判断
    send_next_idx = _find_btn_decl(block, 'send_next_btn')
    prev_idx = _find_btn_decl(block, 'prev_btn')
    assert send_next_idx >= 0, "BUG: send_next_btn 未在 row3 创建"
    assert prev_idx >= 0, "BUG: prev_btn 未在 row3 创建"
    assert send_next_idx < prev_idx, (
        f"BUG: send_next ({send_next_idx}) 必须在 prev ({prev_idx}) 之前 "
        "(Round 39: 主操作绿色按钮放最左)"
    )


def test_row3_resend_removed():
    """验证 row3 已不含 resend_btn (Round 39 移到 row4)."""
    block = _get_row_block('row3')
    assert _find_btn_decl(block, 'resend_btn') < 0, (
        "BUG: resend_btn 仍在 row3 (Round 39 应移到 row4)"
    )


def test_row4_resend_before_reset_ptr():
    """验证 row4: '重发当前帧' (resend) 在 '回到首帧' (reset_ptr) 之前."""
    block = _get_row_block('row4')
    resend_idx = _find_btn_decl(block, 'resend_btn')
    reset_ptr_idx = _find_btn_decl(block, 'reset_ptr_btn')
    assert resend_idx >= 0, "BUG: resend_btn 未在 row4 创建 (Round 39 应在 row4)"
    assert reset_ptr_idx >= 0, "BUG: reset_ptr_btn 未在 row4 创建"
    assert resend_idx < reset_ptr_idx, (
        f"BUG: resend ({resend_idx}) 必须在 reset_ptr ({reset_ptr_idx}) 之前 "
        "(Round 39: 重发放最左, 行级操作组)"
    )


def test_row4_buttons_order():
    """验证 row4 顺序: resend → reset_ptr → reload → unload."""
    block = _get_row_block('row4')
    resend_idx = _find_btn_decl(block, 'resend_btn')
    reset_idx = _find_btn_decl(block, 'reset_ptr_btn')
    reload_idx = _find_btn_decl(block, 'reload_btn')
    unload_idx = _find_btn_decl(block, 'unload_btn')
    assert resend_idx > 0, "BUG: resend_btn 不在 row4"
    assert reset_idx > 0, "BUG: reset_ptr_btn 不在 row4"
    assert reload_idx > 0, "BUG: reload_btn 不在 row4"
    assert unload_idx > 0, "BUG: unload_btn 不在 row4"
    assert resend_idx < reset_idx < reload_idx < unload_idx, (
        f"BUG: row4 顺序错 (期望 resend < reset < reload < unload), "
        f"got resend={resend_idx} reset={reset_idx} reload={reload_idx} unload={unload_idx}"
    )


def test_row3_goto_spin_and_btn_intact():
    """回归: row3 仍含 goto_spin + goto_btn (Round 39 不破跳转功能)."""
    block = _get_row_block('row3')
    assert 'self.goto_spin' in block, "BUG: row3 缺 goto_spin"
    assert 'self.goto_btn' in block, "BUG: row3 缺 goto_btn"


def test_start_auto_clears_manual():
    """验证 _start() 在 manual_loaded=True 时自动调 _manual_unload()."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    # 找 _start 函数
    start_idx = src.find('def _start(self):')
    assert start_idx >= 0, "BUG: _start 函数未找到"
    # 找 _start 函数体的开头 (下一行)
    next_n_def = src.find('\n    def ', start_idx + 1)
    if next_n_def < 0:
        next_n_def = start_idx + 3000  # 大约 3000 字符内
    block = src[start_idx:next_n_def]
    # 关键: manual_loaded 检查 + _manual_unload 调用
    assert 'manual_loaded' in block, (
        "BUG: _start 不引用 manual_loaded (Round 39 应自动清空手动模式)"
    )
    assert '_manual_unload' in block, (
        "BUG: _start 不调 _manual_unload (Round 39 应在 manual_loaded 时自动清空)"
    )


def test_start_unload_before_get_port():
    """验证 _start 中 manual_unload 在 _get_port 调用之前 (顺序正确)."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    start_idx = src.find('def _start(self):')
    next_n_def = src.find('\n    def ', start_idx + 1)
    if next_n_def < 0:
        next_n_def = start_idx + 3000
    block = src[start_idx:next_n_def]
    unload_idx = block.find('_manual_unload')
    get_port_idx = block.find('_get_port')
    assert unload_idx > 0, "BUG: _manual_unload 不在 _start 中"
    assert get_port_idx > 0, "BUG: _get_port 不在 _start 中"
    assert unload_idx < get_port_idx, (
        f"BUG: _manual_unload ({unload_idx}) 必须在 _get_port ({get_port_idx}) 之前 "
        "(顺序: 先清空手动模式, 再校验串口/CSV/波特率)"
    )


def test_round39_marker_comment():
    """验证源码有 Round 39 标记注释."""
    src = open('rtu_replay_gui.py', encoding='utf-8').read()
    assert '第三十九轮' in src, "BUG: 缺 Round 39 标记注释"
    assert '自动清空手动模式' in src or '自动清空' in src, (
        "BUG: 缺自动清空手动模式说明注释"
    )


if __name__ == '__main__':
    test_row3_send_next_before_prev()
    print("[PASS] test_row3_send_next_before_prev (row3: send_next 在 prev 前)")

    test_row3_resend_removed()
    print("[PASS] test_row3_resend_removed (row3 不含 resend)")

    test_row4_resend_before_reset_ptr()
    print("[PASS] test_row4_resend_before_reset_ptr (row4: resend 在 reset_ptr 前)")

    test_row4_buttons_order()
    print("[PASS] test_row4_buttons_order (row4 顺序: resend→reset→reload→unload)")

    test_row3_goto_spin_and_btn_intact()
    print("[PASS] test_row3_goto_spin_and_btn_intact (跳转功能回归)")

    test_start_auto_clears_manual()
    print("[PASS] test_start_auto_clears_manual (_start 自动清空 manual)")

    test_start_unload_before_get_port()
    print("[PASS] test_start_unload_before_get_port (unload 在 get_port 前)")

    test_round39_marker_comment()
    print("[PASS] test_round39_marker_comment (Round 39 注释存在)")

    print("\n所有 8 个 Round 39 测试通过!")
    print("row3: send_next → prev → goto (主操作靠左)")
    print("row4: resend → reset_ptr → reload → unload (行级操作组)")
    print("_start: manual_loaded=True 时自动调 _manual_unload()")