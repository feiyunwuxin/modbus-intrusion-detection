#!/usr/bin/env python3
"""验证 Round 35: 默认数据集改回 IanArffDataset.csv (用户要求).

⭐ 2026-10-02 第三十五轮:
  旧默认: IanArffDatasetF32.csv (高精度, 2026-09-19 第二轮加, 防 time_diff 漂移)
  新默认: IanArffDataset.csv (原始 ARFF CSV, 整数秒时间字段, 用户要求)

  注意: H743 烧写/PC 推理脚本 (scripts/*.py, h743/scripts/*.py) 仍用 F32.csv
        是有意的 (H743 firmware 需要高精度 time_diff).
"""
from __future__ import annotations
import os
import sys

# Dummy PyQt5 (与之前测试一致)
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
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass

class _DummyLineEdit(_DummyWidget):
    def __init__(self, text="", *a, **k):
        super().__init__(*a, **k)
        self._text = text
    def setText(self, t): self._text = t
    def text(self): return self._text

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
for _name in ('QApplication', 'QWidget', 'QHBoxLayout', 'QVBoxLayout',
              'QLabel', 'QPushButton', 'QPlainTextEdit', 'QTextEdit',
              'QComboBox', 'QSpinBox', 'QDoubleSpinBox', 'QFileDialog',
              'QMessageBox', 'QMainWindow', 'QDialog', 'QCheckBox',
              'QRadioButton', 'QGroupBox', 'QFrame', 'QSizePolicy',
              'QLineEdit', 'QTabWidget', 'QGridLayout', 'QStatusBar',
              'QMenuBar', 'QAction', 'QListWidget', 'QTreeWidget',
              'QTableWidget', 'QHeaderView', 'QProgressBar', 'QScrollArea', 'QSlider'):
    setattr(_dummy_qtwidgets, _name, _DummyWidget)
setattr(_dummy_qtwidgets, 'QLineEdit', _DummyLineEdit)
_dummy_qtwidgets.WidgetWidth = 1
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

def test_default_csv_is_original():
    """验证 DEFAULT_CSV 改回 IanArffDataset.csv (无 F32)."""
    assert rtu_replay_gui.DEFAULT_CSV == r"C:\work\Claude\Issue\IanArffDataset.csv", (
        f"BUG: DEFAULT_CSV 应 IanArffDataset.csv, got '{rtu_replay_gui.DEFAULT_CSV}'"
    )


def test_default_csv_file_exists():
    """验证 默认数据集文件实际存在 (防路径错)."""
    import os
    assert os.path.exists(rtu_replay_gui.DEFAULT_CSV), (
        f"BUG: DEFAULT_CSV 文件不存在: {rtu_replay_gui.DEFAULT_CSV}"
    )


def test_default_csv_no_f32_in_constant():
    """验证 DEFAULT_CSV 常量字符串不含 F32 (Round 35 切换后)."""
    assert 'F32' not in rtu_replay_gui.DEFAULT_CSV, (
        "BUG: DEFAULT_CSV 仍含 F32"
    )


def test_default_csv_has_round35_comment():
    """验证源码有 Round 35 切换说明注释."""
    import inspect
    src = inspect.getsource(rtu_replay_gui)
    assert '第三十五轮' in src, "BUG: 缺 Round 35 切换说明注释"
    # DEFAULT_CSV 行附近
    csv_idx = src.find('DEFAULT_CSV = r"')
    block = src[max(0, csv_idx - 200):csv_idx + 200]
    assert 'IanArffDataset.csv' in block, (
        "BUG: DEFAULT_CSV 应指向 IanArffDataset.csv"
    )


if __name__ == '__main__':
    test_default_csv_is_original()
    print("[PASS] test_default_csv_is_original")

    test_default_csv_file_exists()
    print("[PASS] test_default_csv_file_exists")

    test_default_csv_no_f32_in_constant()
    print("[PASS] test_default_csv_no_f32_in_constant")

    test_default_csv_has_round35_comment()
    print("[PASS] test_default_csv_has_round35_comment")

    print("\n所有 4 个 Round 35 测试通过!")
    print("默认数据集 = IanArffDataset.csv (原始 ARFF CSV, 整数秒时间字段).")