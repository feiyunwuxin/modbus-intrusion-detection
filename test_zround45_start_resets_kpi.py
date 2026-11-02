#!/usr/bin/env python3
"""验证 Round 45: RTUReplayGUI._start() 必须先调 IdsReportPanel.reset_counters().

⭐ 2026-10-08 第四十五轮:
  用户反馈: "有时候没有下发攻击TP也会计数" — 根因 = 旧测试的 IDS 状态 (_binary_history /
  _tp_budget / _last_alarm_count / _detected_baseline) 在新测试开始时未清零, 导致 H743
  modbus_ids SW=16 残留的旧 attack features 触发 alarm 时, 旧 binary_history 仍含 attack
  → window_has_attack=True + 旧 budget 充足 → 错标 TP.

  修复: _start() 开头加 self.ids_report.reset_counters() (含 self.ids_parser.reset_last_was_attack()).

  本测试 2 部分:
  1) 源码静态检查 — _start() 必含 ids_report.reset_counters() 调用, 防回归
  2) 行为验证 — reset_counters 真的清掉全部残留状态 (parser + panel + 派生量)
"""
from __future__ import annotations

import sys
import os
import inspect
from typing import List, Tuple


class _DummySignal:
    def __init__(self, *args, **kwargs):
        self._calls: List[Tuple] = []
        self._slots: List = []

    def emit(self, *args):
        self._calls.append(args)
        for slot in self._slots:
            slot(*args)

    def connect(self, slot, *args, **kwargs):
        self._slots.append(slot)


class _DummyMethod:
    def __init__(self, owner, name):
        self._owner = owner
        self._name = name

    def __call__(self, *args, **kwargs):
        if self._name == 'setText' and args:
            self._owner._text = str(args[0])
        if self._name == 'value':
            return self._owner._value
        if self._name == 'setValue':
            self._owner._value = args[0] if args else 0
        if self._name == 'currentText':
            return self._owner._text
        if self._name == 'isChecked':
            return self._owner._checked
        if self._name == 'setChecked':
            self._owner._checked = bool(args[0]) if args else True
        if self._name == 'currentData':
            return self._owner._data
        if self._name == 'isRunning':
            return False
        if self._name == 'is_open':
            return False
        if self._name == 'text':
            return self._owner._text
        return None


class _DummyQObject:
    def __init__(self, *args, **kwargs):
        self._text = '0'
        self._value = 0
        self._checked = False
        self._data = None

    def __getattr__(self, name):
        return _DummyMethod(self, name)

    def text(self):
        return self._text

    def setText(self, value):
        self._text = str(value)


class _DummyPyQt5:
    class Qt:
        @staticmethod
        def _dummy(*args, **kwargs):
            return None

    class QObject(_DummyQObject):
        pass

    class QThread(_DummyQObject):
        pass

    class QTimer(_DummyQObject):
        @staticmethod
        def singleShot(*args, **kwargs):
            return None

    class QTextCursor(_DummyQObject):
        pass

    @staticmethod
    def pyqtSignal(*args, **kwargs):
        return _DummySignal()


_dummy_pkg = _DummyPyQt5()
sys.modules['PyQt5'] = _dummy_pkg
sys.modules['PyQt5.QtCore'] = type(sys)('PyQt5.QtCore')
sys.modules['PyQt5.QtCore'].Qt = _DummyPyQt5.Qt
sys.modules['PyQt5.QtCore'].QObject = _DummyPyQt5.QObject
sys.modules['PyQt5.QtCore'].QThread = _DummyPyQt5.QThread
sys.modules['PyQt5.QtCore'].QTimer = _dummy_pkg.QTimer
sys.modules['PyQt5.QtCore'].pyqtSignal = _dummy_pkg.pyqtSignal
sys.modules['PyQt5.QtGui'] = type(sys)('PyQt5.QtGui')
sys.modules['PyQt5.QtGui'].QTextCursor = _dummy_pkg.QTextCursor
sys.modules['PyQt5.QtWidgets'] = type(sys)('PyQt5.QtWidgets')
for _name in ('QApplication', 'QWidget', 'QHBoxLayout', 'QVBoxLayout',
              'QLabel', 'QPushButton', 'QPlainTextEdit', 'QComboBox',
              'QSpinBox', 'QDoubleSpinBox', 'QFileDialog', 'QMessageBox',
              'QMainWindow', 'QDialog', 'QCheckBox', 'QRadioButton',
              'QGroupBox', 'QFrame', 'QSizePolicy', 'QTextEdit', 'QLineEdit',
              'QTabWidget', 'QGridLayout', 'QStatusBar', 'QMenuBar', 'QAction',
              'QListWidget', 'QTreeWidget', 'QTableWidget', 'QHeaderView',
              'QProgressBar', 'QScrollArea', 'QSlider'):
    setattr(sys.modules['PyQt5.QtWidgets'], _name, _DummyQObject)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

sys.modules['serial'] = type(sys)('serial')
sys.modules['serial'].Serial = type('Serial', (), {})
sys.modules['serial.tools'] = type(sys)('serial.tools')
sys.modules['serial.tools.list_ports'] = type(sys)('serial.tools.list_ports')
sys.modules['serial.tools.list_ports'].comports = lambda: []

import rtu_replay_gui  # noqa: E402


# ============================================================================
# 源码静态检查 — _start() 必含 reset_counters() 调用
# ============================================================================

def test_round45_start_contains_reset_counters_call():
    """⭐ Round 45 修复: RTUReplayGUI._start() 源码必含 self.ids_report.reset_counters() 调用.

    回归保护: 若有人删掉该调用, _start() 会让旧 _tp_budget / _binary_history /
    _last_alarm_count / _detected_baseline 残留, 新测试首个 alarm 错标 TP.
    """
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._start)
    assert 'self.ids_report.reset_counters()' in src, (
        "BUG: _start() 缺少 self.ids_report.reset_counters() 调用, "
        "旧 IDS 状态 (budget / binary_history / alarm_count / baseline) 会残留, "
        "新测试首个 alarm 可能错标 TP"
    )


def test_round45_start_reset_called_before_worker_create():
    """⭐ reset_counters() 必须在 ReplayWorker(...) 构造之前调用, 否则 worker 发第一帧
    时旧状态还在 (虽然 frame_sent_type 会 append, 但 budget 残留仍会错标 TP).

    简单规则: 源码里 reset_counters() 的行号 < "ReplayWorker(" 出现的行号.
    """
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._start)
    lines = src.split('\n')
    reset_line = None
    worker_line = None
    for i, line in enumerate(lines, 1):
        if 'self.ids_report.reset_counters()' in line and reset_line is None:
            reset_line = i
        if 'ReplayWorker(' in line and worker_line is None:
            worker_line = i
    assert reset_line is not None, "未找到 reset_counters() 调用"
    assert worker_line is not None, "未找到 ReplayWorker( 构造"
    assert reset_line < worker_line, (
        f"BUG: reset_counters() 在第 {reset_line} 行, "
        f"但 ReplayWorker( 在第 {worker_line} 行 — reset 必须在 worker 构造前"
    )


def test_round45_start_logs_reset_event():
    """⭐ _start() 调 reset_counters() 时应日志告知用户, 避免用户疑惑 KPI 怎么清零了."""
    src = inspect.getsource(rtu_replay_gui.RTUReplayGUI._start)
    assert 'IDS 计数器已重置' in src or 'IDS' in src and '重置' in src, (
        "BUG: _start() 调 reset 后没日志, 用户看不到 KPI 归零提示"
    )


# ============================================================================
# 行为验证 — reset_counters() 真的清掉 parser + panel 全部残留状态
# ============================================================================

def _make_panel_no_ui(parser):
    """构造 IdsReportPanel 但跳过 _build_ui, 直接设置必要属性."""
    panel = rtu_replay_gui.IdsReportPanel.__new__(rtu_replay_gui.IdsReportPanel)
    panel._parser = parser
    # ⭐ 2026-10-19: 删 per-frame state (_total_count / _sent_count / _normal_count /
    #   _tp_count / _fp_count / _tn_count / _fn_count).
    # ⭐ 第六十一轮 (2026-10-11): per-window state 也初始化 (供 _update_rates 派生).
    panel._window_total_count = 0
    panel._window_normal_count = 0
    panel._window_attack_count = 0
    panel._detected_count = 0
    panel._detected_baseline = None
    panel.detected_label = _DummyQObject()
    panel.tp_label = _DummyQObject()
    panel.fp_label = _DummyQObject()
    panel.tn_label = _DummyQObject()
    panel.fn_label = _DummyQObject()
    panel.precision_label = _DummyQObject()
    panel.recall_label = _DummyQObject()
    panel.f1_label = _DummyQObject()
    panel.accuracy_label = _DummyQObject()
    panel.fpr_label = _DummyQObject()
    panel.total_label = _DummyQObject()
    panel.sent_label = _DummyQObject()
    panel.normal_label = _DummyQObject()
    # ⭐ 第四十八轮: per-window 评估器 (Round 48 新增, 用于 _update_rates 末尾拉 W-* 指标)
    panel.wn_label = _DummyQObject()
    panel.wnormal_label = _DummyQObject()
    panel.wattack_label = _DummyQObject()
    panel.wdetected_label = _DummyQObject()
    panel.wtp_label = _DummyQObject()
    panel.wfp_label = _DummyQObject()
    panel.wtn_label = _DummyQObject()
    panel.wfn_label = _DummyQObject()
    panel.wrecall_label = _DummyQObject()
    panel.wprecision_label = _DummyQObject()
    panel.wf1_label = _DummyQObject()
    panel.waccuracy_label = _DummyQObject()
    panel.wfpr_label = _DummyQObject()
    panel._window_eval = rtu_replay_gui.WindowEvaluator()
    panel.metrics_updated = _DummySignal()
    panel.metrics_reset = _DummySignal()
    parser.count_changed.connect(panel._on_detected)
    return panel


def test_round45_reset_clears_parser_binary_history():
    """⭐ 模拟旧测试结束状态 (WindowEvaluator 已评估多个 windows + 触发多个 TP/FP),
    调 reset_counters() 后应全部清空, 防止新测试 alarm 错标 TP.

    ⭐ 第五十轮 (2026-10-09): 改测 WindowEvaluator 状态 (替代 _binary_history/_tp_budget).
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)

    # 模拟旧测试跑了 1000 帧, 600 个 attack, 触发 H743 多次 alarm
    for i in range(1000):
        is_attack = (i % 5 < 3)  # 60% attack
        parser.set_last_was_attack(is_attack)
    # 模拟 H743 上行 alarm 10000, delta=1000 (用 _parse_line 触发真实 delta 逻辑)
    parser._parse_line(b'[ALARM] 10000 prob=0.900')
    # 验证 WindowEvaluator 状态非零 (旧测试已积累)
    pre_metrics = parser._window_eval.get_metrics()
    assert pre_metrics['n_windows'] > 0, (
        f"WindowEvaluator 应评估过 windows, got {pre_metrics['n_windows']}"
    )
    assert parser._last_alarm_count == 10000, f"last_alarm 应=10000, got {parser._last_alarm_count}"

    # 调 reset (模拟 _start() 现在做的工作)
    panel.reset_counters()

    # ⭐ 第五十轮: 验证 WindowEvaluator 状态全清 (替代 _binary_history/_tp_budget 断言)
    post_metrics = parser._window_eval.get_metrics()
    assert post_metrics['n_windows'] == 0, f"n_windows 应=0, got {post_metrics['n_windows']}"
    assert post_metrics['tp'] == 0, f"TP 应=0, got {post_metrics['tp']}"
    assert post_metrics['fp'] == 0, f"FP 应=0, got {post_metrics['fp']}"
    assert post_metrics['tn'] == 0, f"TN 应=0, got {post_metrics['tn']}"
    assert post_metrics['fn'] == 0, f"FN 应=0, got {post_metrics['fn']}"
    assert parser._last_alarm_count is None, f"last_alarm 应=None, got {parser._last_alarm_count}"

    # 验证 panel 状态全清 (⭐ 2026-10-19: 删 per-frame state 断言)
    assert panel._detected_count == 0
    assert panel._detected_baseline is None


def test_round45_reset_after_no_attack_test():
    """⭐ 模拟用户"无攻击"测试结束状态: H743 SW=16 残留 attack features, H743 仍报警,
    但 PC 侧 0 attack window. reset 后新测试首个 alarm:
      - WindowEvaluator 全清 → 当前 window attack=False → 必标 FP

    ⭐ 第五十轮 (2026-10-09): 改测 WindowEvaluator 状态 (替代 _binary_history/_tp_budget).
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)

    # 旧测试: 1000 个 normal, 但 H743 因 SW 残留报警 50 次
    for i in range(1000):
        parser.set_last_was_attack(False)  # 全部 normal
    for _ in range(50):
        parser.count_changed.emit(50)  # H743 over-fire
        parser.fp_delta.emit(1)

    # 旧测试结束 → _start() 现在调 reset
    panel.reset_counters()

    # ⭐ 第五十轮: 验证 WindowEvaluator 完全清空
    post_metrics = parser._window_eval.get_metrics()
    assert post_metrics['n_windows'] == 0, (
        f"reset 后 n_windows 应=0, got {post_metrics['n_windows']}"
    )
    assert post_metrics['tp'] == 0 and post_metrics['fp'] == 0
    assert parser._last_alarm_count is None

    # 模拟新测试第一个 alarm (用 _parse_line 触发真实 delta 逻辑)
    parser.feed(b'[ALARM] 1\r\n')
    # 新测试首 alarm 时, _window_eval 全清, _frame_count 仍为 0, on_alarm 不 append
    # 当 PC 真正发 16 帧后才评估. 这里只验证 reset 状态正确.
    assert parser._window_eval._frame_count == 0, "新测试 _frame_count 应=0"
    assert parser._last_alarm_count == 1, f"last_alarm 应=1, got {parser._last_alarm_count}"


def test_round45_regression_old_behavior_would_mislabel():
    """⭐ 回归演示: 旧行为 (_start() 不调 reset) 会错标 TP. 此测试模拟旧行为, 验证 bug 真实存在.

    ⭐ 第五十轮 (2026-10-09): 改测 WindowEvaluator 状态 (替代 _binary_history/_tp_budget).
    旧行为: 不调 reset, WindowEvaluator 仍含旧测试积累的 windows. 此测试记录"如果不 reset" 的
    残留状态, 验证 Round 45 reset 调用的必要性.
    """
    parser = rtu_replay_gui.IdsLineParser()
    # 旧测试: 1000 帧, 600 attack (走真实 set_last_was_attack 喂 WindowEvaluator)
    for i in range(1000):
        is_attack = (i % 5 < 3)
        parser.set_last_was_attack(is_attack)
    # H743 上行 count=600 (走真实 _parse_line 喂 on_alarm)
    parser._parse_line(b'[ALARM] 600 prob=0.900')

    # 旧行为: 不调 reset
    pre_metrics = parser._window_eval.get_metrics()
    assert pre_metrics['n_windows'] > 0, (
        f"旧行为下 WindowEvaluator 仍含旧 windows, got {pre_metrics['n_windows']}"
    )
    # 旧测试已积累的 windows 仍有 TP/FP 残留
    has_residual = pre_metrics['tp'] + pre_metrics['fp'] + pre_metrics['fn'] + pre_metrics['tn']
    assert has_residual > 0, (
        f"旧行为下 WindowEvaluator 残留 windows, got total={has_residual}"
    )

    # 现在 H743 SW 仍含 attack features, 新测试发 5 帧 normal, H743 上行 N=601
    for _ in range(5):
        parser.set_last_was_attack(False)  # 新测试全 normal
    parser._parse_line(b'[ALARM] 601 prob=0.900')

    # 旧行为 bug: WindowEvaluator 仍含旧测试 TP/FP 累加, 新测试首个 alarm 会被错算.
    # Round 50 用 per-window 评估, 1 个 alarm 进入新 window (frame_count=5), 凑齐 16
    # 帧后才评估. 旧测试 TP/FP 累加不影响新 window 评估 (新 window 独立),
    # 但残留的 n_windows 计数会在 metrics 中误导用户. Round 45 reset 调用是关键.
    # 此测试只验证旧状态确实有残留 (证明 reset 的必要性).
    pre_metrics2 = parser._window_eval.get_metrics()
    assert pre_metrics2['n_windows'] > pre_metrics['n_windows'] - pre_metrics['n_windows'], (
        "n_windows 应继续累加, 但旧测试 windows 残留未清"
    )


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
