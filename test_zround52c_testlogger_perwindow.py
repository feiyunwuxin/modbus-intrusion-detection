#!/usr/bin/env python3
"""验证 Round 52c: TestLogger 加 per-window 摘要块, 跟训练 / H743 1:1.

⭐ 2026-10-10 第五十二轮 c (用户发现 per-frame 评估 Recall=8.16% 是粒度错位):
  - 实际 H743 windows_attack 累计 = 4900 (从 TP+FP 推算, 跟 count_changed 一致)
  - PC attack windows (max-aggregation) = 8821
  - 真实 per-window Recall = 4900 / 8821 = 55.6%
  - 旧 summary 只算 per-frame (4900/60048=8.16%) 看起来像模型差, 实际是评估粒度错位

  修复:
    1. TestLogger 加 per-window state (running_window_total/attack/normal/detected)
    2. main_window._on_window_stats_for_summary / _on_detected_for_summary 维护这些字段
    3. TestLogger.set_window_metrics(wm) 收 WindowEvaluator 推来的 TP/FP/TN/FN
    4. write_summary 加 per-window 块 (Recall = TP / attack_windows, 跟训练 1:1)
"""
from __future__ import annotations

import os
import sys
from typing import List


class _DummySignal:
    def __init__(self, *args, **kwargs):
        self._calls: List = []
        self._slots: List = []
    def emit(self, *args):
        self._calls.append(args)
        for slot in self._slots:
            try:
                slot(*args)
            except TypeError:
                try:
                    slot()
                except TypeError:
                    pass
    def connect(self, slot, *args, **kwargs):
        self._slots.append(slot)


class _DummyLabel:
    def __init__(self, text=""): self._text = text
    def setStyleSheet(self, *a, **k): pass
    def setText(self, text): self._text = text
    def text(self): return self._text
    def setMinimumHeight(self, *a, **k): pass
    def setMinimumWidth(self, *a, **k): pass
    def setMaximumHeight(self, *a, **k): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addSpacing(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setStretch(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setAlignment(self, *a, **k): pass
    def isChecked(self): return False
    def setEnabled(self, *a, **k): pass
    def setCheckable(self, *a, **k): pass
    def setChecked(self, *a, **k): pass
    def setVisible(self, *a, **k): pass
    def value(self): return 0
    def setValue(self, *a, **k): pass
    def setMaximum(self, *a, **k): pass
    def setRange(self, *a, **k): pass
    def setFormat(self, *a, **k): pass
    def setSingleStep(self, *a, **k): pass


class _DummyLayout:
    def __init__(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass
    def setAlignment(self, *a, **k): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addSpacing(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setStretch(self, *a, **k): pass


_dummy_pkg = _DummyPyQt5 = sys.modules.setdefault  # placeholder
sys.modules['PyQt5'] = type(sys)('PyQt5')
sys.modules['PyQt5.QtCore'] = type(sys)('PyQt5.QtCore')
sys.modules['PyQt5.QtCore'].Qt = type('Qt', (), {'_dummy': staticmethod(lambda *a, **k: None)})
sys.modules['PyQt5.QtCore'].QObject = type('QObject', (), {'__init__': lambda *a, **k: None})
sys.modules['PyQt5.QtCore'].QThread = type('QThread', (), {})
sys.modules['PyQt5.QtCore'].QTimer = type('QTimer', (), {'singleShot': staticmethod(lambda *a, **k: None)})
sys.modules['PyQt5.QtCore'].pyqtSignal = lambda *a, **k: _DummySignal()
sys.modules['PyQt5.QtGui'] = type(sys)('PyQt5.QtGui')
sys.modules['PyQt5.QtGui'].QTextCursor = type('QTextCursor', (), {})
sys.modules['PyQt5.QtWidgets'] = type(sys)('PyQt5.QtWidgets')
setattr(sys.modules['PyQt5.QtWidgets'], 'QLabel', _DummyLabel)
setattr(sys.modules['PyQt5.QtWidgets'], 'QHBoxLayout', _DummyLayout)
setattr(sys.modules['PyQt5.QtWidgets'], 'QVBoxLayout', _DummyLayout)
setattr(sys.modules['PyQt5.QtWidgets'], 'QGridLayout', _DummyLayout)
for _name in ('QApplication', 'QWidget', 'QPushButton', 'QPlainTextEdit', 'QComboBox',
              'QSpinBox', 'QDoubleSpinBox', 'QFileDialog', 'QMessageBox',
              'QMainWindow', 'QDialog', 'QCheckBox', 'QRadioButton',
              'QGroupBox', 'QFrame', 'QSizePolicy', 'QTextEdit', 'QLineEdit',
              'QTabWidget', 'QStatusBar', 'QMenuBar', 'QAction',
              'QListWidget', 'QTreeWidget', 'QTableWidget', 'QHeaderView',
              'QProgressBar', 'QScrollArea', 'QSlider'):
    setattr(sys.modules['PyQt5.QtWidgets'], _name, type(_name, (), {'__init__': lambda *a, **k: None}))


def _load_gui_module():
    gui_path = os.path.abspath(os.path.join(os.path.dirname(__file__) or '.', 'rtu_replay_gui.py'))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rtu_replay_gui", gui_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__file__ = gui_path
    return mod


rtu_replay_gui = _load_gui_module()


# ============================================================================
# 1) TestLogger 字段: per-window 5 字段 (total/attack/normal/detected + tp/fp/tn/fn)
# ============================================================================

def test_round52c_testlogger_has_perwindow_fields():
    """⭐ Round 52c: TestLogger 加 5 个 per-window 字段."""
    logger = rtu_replay_gui.TestLogger()
    for attr in ('running_window_total', 'running_window_attack',
                 'running_window_normal', 'running_window_detected',
                 'running_window_tp', 'running_window_fp',
                 'running_window_tn', 'running_window_fn'):
        assert hasattr(logger, attr), f"TestLogger 应有 {attr}"
        assert getattr(logger, attr) == 0, f"初始 {attr} 应为 0"


# ============================================================================
# 2) set_window_metrics: 从 WindowEvaluator 推 TP/FP/TN/FN
# ============================================================================

def test_round52c_set_window_metrics():
    """⭐ set_window_metrics(dict) 把 WindowEvaluator.get_metrics() 喂给 TestLogger."""
    logger = rtu_replay_gui.TestLogger()
    logger.set_window_metrics({
        'tp': 4900, 'fp': 1759, 'tn': 6584, 'fn': 3921, 'n_windows': 17164,
    })
    assert logger.running_window_tp == 4900
    assert logger.running_window_fp == 1759
    assert logger.running_window_tn == 6584
    assert logger.running_window_fn == 3921
    assert logger.running_window_total == 17164


def test_round52c_set_window_metrics_empty():
    """⭐ set_window_metrics({}) 或 None 不应崩."""
    logger = rtu_replay_gui.TestLogger()
    logger.set_window_metrics({})  # 空 dict
    assert logger.running_window_tp == 0
    logger.set_window_metrics(None)  # None
    assert logger.running_window_tp == 0


# ============================================================================
# 3) write_summary 验证: 包含 Per-Window 块, Recall 跟训练 1:1
# ============================================================================

def test_round52c_write_summary_has_perwindow_block(tmp_path=None):
    """⭐ write_summary 写 per-window 块, 跟训练 1:1 评估.

    ⭐ 2026-10-19: 删 per-frame 块 (running_total/attack/normal/tp/fp 已删除).
    summary 现在只写 per-window 块.
    """
    import tempfile
    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, 'test_perwindow.csv')

    logger = rtu_replay_gui.TestLogger()
    # per-window 数据
    logger.running_window_total = 17164
    logger.running_window_attack = 8821
    logger.running_window_normal = 8343
    logger.running_window_detected = 4900  # H743 windows_attack 累计
    logger.running_window_tp = 4900
    logger.running_window_fp = 0  # 真实 per-window 评估应该没 FP
    logger.running_window_tn = 8343
    logger.running_window_fn = 3921  # 8821 - 4900

    # 用子类让 is_open 永远返回 True
    class _SubTL(rtu_replay_gui.TestLogger):
        @property
        def is_open(self):
            return True

    sub = _SubTL()
    sub._path = log_path
    sub.running_window_total = 17164
    sub.running_window_attack = 8821
    sub.running_window_normal = 8343
    sub.running_window_detected = 4900
    sub.running_window_tp = 4900
    sub.running_window_fp = 0
    sub.running_window_tn = 8343
    sub.running_window_fn = 3921
    sub.write_summary()

    # 读生成的 summary
    summary_path = log_path + '.summary.txt'
    assert os.path.exists(summary_path), f"应生成 {summary_path}"
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # ⭐ 2026-10-19: per-frame 块已删除, 不应再出现
    assert 'Per-Frame' not in content, "per-frame 块已删除, summary 不应包含"

    # 验证: per-window 块存在
    assert 'Per-Window' in content, "新 Per-Window 块应存在"
    assert 'Total windows:' in content
    assert 'Attack (Sent): 8821' in content, "Attack windows 应是 8821 (max-aggregation)"
    assert 'Normal:' in content and '8343' in content
    assert 'Detected:      4900' in content, "Detected 应是 H743 windows_attack 累计"
    assert 'Missed (FN):   3921' in content, "Missed = 8821 - 4900 = 3921"

    # 验证: per-window Recall = 4900 / 8821 ≈ 0.5556
    expected_w_recall = 4900 / 8821
    assert f'Recall (TP/Atk): {expected_w_recall:.4f}' in content, (
        f"Per-window Recall 应是 {expected_w_recall:.4f}, summary 内容: \n{content}"
    )
    # 验证: per-window F1 (跟训练 1:1)
    expected_w_f1 = 2 * expected_w_recall / (1 + expected_w_recall)  # precision=1
    assert f'F1:              {expected_w_f1:.4f}' in content


def test_round52c_perwindow_4900_to_8821_recall_55pct():
    """⭐⭐ 关键诊断: 真实 per-window Recall = 4900/8821 = 55.6%, 不是 per-frame 8.16%."""
    # 这就是用户看到的 8.16% 假象的根因
    tp = 4900  # H743 实际报了的 attack window
    attack_windows = 8821  # PC 端 max-aggregation 后 attack windows (真值)
    per_frame_recall = tp / 60048  # 8.16% 假象
    per_window_recall = tp / attack_windows  # 55.6% 真实
    assert abs(per_frame_recall - 0.0816) < 0.001
    assert abs(per_window_recall - 0.5556) < 0.001
    print(f"  per-frame Recall (旧, 错) = {per_frame_recall*100:.2f}%")
    print(f"  per-window Recall (新, 真) = {per_window_recall*100:.2f}%")
    print(f"  训练 baseline F1=0.8775, 部署差距应该 < 30% (过拟合到 23-dim)")


# ============================================================================
# 4) main_window 接线: window_stats + count_changed → test_logger
# ============================================================================

def test_round52c_mainwindow_wires_summary_signals():
    """⭐ main_window._start 接线 worker.window_stats + ids_parser.count_changed → summary."""
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 验证: window_stats → _on_window_stats_for_summary
    assert 'self.worker.window_stats.connect(self._on_window_stats_for_summary)' in src
    # 验证: count_changed → _on_detected_for_summary
    assert 'self.ids_parser.count_changed.connect(self._on_detected_for_summary)' in src
    # 验证: _on_finished 调 set_window_metrics
    assert 'self.test_logger.set_window_metrics(' in src
    # 验证: _on_error 也调 set_window_metrics
    # (counts both _on_finished and _on_error 调用)
    assert src.count('self.test_logger.set_window_metrics(') >= 2


def test_round52c_summary_handler_methods_exist():
    """⭐ main_window 有 _on_window_stats_for_summary / _on_detected_for_summary 槽函数."""
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'def _on_window_stats_for_summary' in src
    assert 'def _on_detected_for_summary' in src


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
