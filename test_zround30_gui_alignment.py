#!/usr/bin/env python3
"""验证 Round 30: GUI 视觉对齐 + 加载按钮配色.

⭐ 2026-10-02 第三十轮:
  1. 删除 "启用手动模式" 复选框, 改为状态指示标签 (manual_status_label).
     手动模式状态完全由 manual_loaded 驱动, 通过 _set_manual_state() 更新.
  2. 对齐 手动模式卡片 与 接收区卡片:
     - 接收区 去除 maxHeight=220, 加 minHeight=380 与手动模式卡片匹配.
  3. "📥 加载 (应用过滤)" 按钮改为深蓝色 (#003366 background, white text).
"""
from __future__ import annotations
import os
import sys
import inspect

# === Dummy PyQt5 (与之前测试一致) ===
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

class _DummyWidget:
    """能记录 setStyleSheet / setText / setEnabled / setMinimumHeight / setMaximumHeight."""
    def __init__(self, *args, **kwargs):
        self._style = ""
        self._text = ""
        self._enabled = True
        self._min_h = 0
        self._min_w = 0
        self._max_h = 16777215
        self._max_w = 16777215
    def setStyleSheet(self, qss): self._style = qss
    def styleSheet(self): return self._style
    def setText(self, t): self._text = t
    def text(self): return self._text
    def setEnabled(self, e): self._enabled = e
    def isEnabled(self): return self._enabled
    def setMinimumHeight(self, h): self._min_h = h
    def setMaximumHeight(self, h): self._max_h = h
    def setMinimumWidth(self, w): self._min_w = w
    def setMaximumWidth(self, w): self._max_w = w
    def minimumHeight(self): return self._min_h
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass

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
for _name in ('QApplication', 'QWidget', 'QHBoxLayout', 'QVBoxLayout',
              'QLabel', 'QPushButton', 'QPlainTextEdit', 'QComboBox',
              'QSpinBox', 'QDoubleSpinBox', 'QFileDialog', 'QMessageBox',
              'QMainWindow', 'QDialog', 'QCheckBox', 'QRadioButton',
              'QGroupBox', 'QFrame', 'QSizePolicy', 'QTextEdit', 'QLineEdit',
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
import rtu_replay_gui  # noqa: E402


# === Tests ===

def test_checkbox_removed():
    """验证 manual_enable_chk 复选框已删除."""
    # 源码不应再有 QCheckBox("启用手动模式") 或 manual_enable_chk 引用
    # 注释中提到 manual_enable_chk 是 OK 的 (Round 30 删改记录)
    src = inspect.getsource(rtu_replay_gui)
    # 过滤掉注释行
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert 'manual_enable_chk' not in code_only, (
        "BUG: 代码仍引用 manual_enable_chk (Round 30 应删除, 注释中允许)"
    )
    assert 'QCheckBox("启用手动模式")' not in code_only, (
        "BUG: 代码仍创建 '启用手动模式' 复选框"
    )


def test_status_label_created():
    """验证 manual_status_label 状态标签存在."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    assert 'self.manual_status_label = QLabel' in src, (
        "BUG: manual_status_label 未创建 (Round 30 应加)"
    )
    # 默认文本应是"未加载"
    assert '"⏸ 手动模式: 未加载"' in src or '"手动模式: 未加载"' in src, (
        "BUG: manual_status_label 初始文本缺失"
    )


def test_set_manual_state_defined():
    """验证 _set_manual_state 方法已定义."""
    assert hasattr(rtu_replay_gui.RTUReplayGUI, '_set_manual_state'), (
        "BUG: 缺 _set_manual_state 方法"
    )
    method = rtu_replay_gui.RTUReplayGUI._set_manual_state
    sig = inspect.signature(method)
    assert 'loaded' in sig.parameters, (
        f"_set_manual_state 应接 loaded 参数: {list(sig.parameters)}"
    )


def test_old_toggle_removed():
    """验证 _on_manual_toggle 旧方法已删除."""
    assert not hasattr(rtu_replay_gui.RTUReplayGUI, '_on_manual_toggle'), (
        "BUG: 旧 _on_manual_toggle 方法仍存在 (Round 30 应删除)"
    )


def test_load_btn_dark_blue():
    """验证 '加载' 按钮已改为深蓝色 (background #003366, color white)."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # 找 load_btn.setStyleSheet 块
    idx = src.find('self.load_btn.setStyleSheet')
    assert idx >= 0, "BUG: load_btn 没设样式 (Round 30 应深蓝)"
    block = src[idx:idx + 300]
    assert '003366' in block, (
        "BUG: load_btn 样式不含深蓝色 #003366"
    )
    assert 'white' in block, (
        "BUG: load_btn 文字色应白色 (深蓝底色需要)"
    )


def test_rx_box_min_height_aligned():
    """验证 接收区卡片 有 minHeight(380) 与手动模式卡片对齐."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    # rx_box 创建后应有 setMinimumHeight(380) 调用
    # 找 "接收区" 标记
    idx = src.find('rx_box = QGroupBox')
    assert idx >= 0, "BUG: rx_box 创建缺失"
    block = src[idx:idx + 500]
    assert 'setMinimumHeight(380)' in block, (
        "BUG: 接收区卡片没设 minHeight=380 (应与手动模式卡片对齐)"
    )
    # 不应有 maxHeight(220) (旧限制已去除)
    assert 'setMaximumHeight(220)' not in block, (
        "BUG: 接收区仍设 maxHeight=220 (应去除, 让卡片自然拉伸)"
    )


def test_rx_box_layout_stretch():
    """验证 接收区 RxPanel 用 stretch=1 加入, 确保文本框能拉伸占满剩余高度."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._build_ui)
    idx = src.find('rx_layout.addWidget(self.rx_panel')
    assert idx >= 0, "BUG: rx_panel 没加入布局"
    block = src[idx:idx + 80]
    assert 'stretch=1' in block or 'stretch = 1' in block, (
        f"BUG: RxPanel 加入布局时应用 stretch=1 占满高度, 当前: {block}"
    )


def test_status_color_changes():
    """验证 _set_manual_state(True/False) 切换状态标签颜色与文本."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._set_manual_state)
    # loaded=True 分支: 绿色
    assert '#006600' in src, "BUG: loaded=True 应绿色文字"
    assert '已加载' in src, "BUG: loaded=True 文本应含已加载"
    # loaded=False 分支: 灰色
    assert '#888' in src, "BUG: loaded=False 应灰色文字"
    assert '未加载' in src, "BUG: loaded=False 文本应含未加载"


def test_status_toggles_start_btn():
    """验证 _set_manual_state 切换 start_btn 的 enabled 状态."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._set_manual_state)
    assert 'self.start_btn.setEnabled(False)' in src, (
        "BUG: loaded=True 应禁用自动按钮 (避免冲突)"
    )
    assert 'self.start_btn.setEnabled(True)' in src, (
        "BUG: loaded=False 应启用自动按钮"
    )


if __name__ == '__main__':
    test_checkbox_removed()
    print("[PASS] test_checkbox_removed (manual_enable_chk 删除)")

    test_status_label_created()
    print("[PASS] test_status_label_created (状态标签替代)")

    test_set_manual_state_defined()
    print("[PASS] test_set_manual_state_defined")

    test_old_toggle_removed()
    print("[PASS] test_old_toggle_removed (_on_manual_toggle 删除)")

    test_load_btn_dark_blue()
    print("[PASS] test_load_btn_dark_blue (深蓝色 #003366)")

    test_rx_box_min_height_aligned()
    print("[PASS] test_rx_box_min_height_aligned (minHeight=380)")

    test_rx_box_layout_stretch()
    print("[PASS] test_rx_box_layout_stretch (RxPanel stretch=1)")

    test_status_color_changes()
    print("[PASS] test_status_color_changes (绿/灰切换)")

    test_status_toggles_start_btn()
    print("[PASS] test_status_toggles_start_btn (start_btn 联动)")

    print("\n所有 9 个 Round 30 测试通过!")
    print("删除 checkbox + 卡片对齐 + 加载深蓝, GUI 视觉一致性提升.")