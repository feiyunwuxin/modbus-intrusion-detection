#!/usr/bin/env python3
"""验证 Round 28/44 IdsReportPanel Detected KPI = H743 上行 alarm 消息数.

⭐ 2026-10-02 第二十八轮:
  用户反馈: rtu_replay_gui 从 frame 200000 开始时, Detected KPI 起始就 20000
  (H743 之前已累计 frames_attack=20000, 未重启). 老逻辑直接 `_detected_count = count`
  导致显示旧值. 改为: 首个 count 捕获为 baseline, 后续 delta.

⭐ 2026-10-08 第四十四轮:
  用户报告: Detected 比 alarm (上行 [ALARM] 消息数) 小 1.
  根因: 旧版 baseline 用首条 N 作锚点 → 首条 N 本身代表 1 次 alarm 工作量, 被
  锚点吃掉. 末条 _detected_count 永远比上行消息数 m 少 1.
  修复: baseline = max(count - 1, 0), 首条 emit 后 _detected_count = 1, 末条
  _detected_count = m (= 上行 [ALARM] 消息数).

⭐ 测试设计说明:
  复用 _DummyPyQt5 + _make_panel_no_ui 模式, 跳过 _build_ui 避开 QLabel.setStyleSheet
  等 GUI mock 兼容问题. 直接构造 IdsReportPanel 实例 (Qt mock), 只覆盖 _detected_baseline
  / _detected_count / detected_label 等关键属性, 不触发 _build_ui.
"""
from __future__ import annotations
import sys
import os
from typing import List, Tuple, Optional


class _DummySignal:
    """模拟 PyQt5 pyqtSignal: emit 记录到 _calls, 同时调用所有 connect 的 slots."""

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
        return None


class _DummyQObject:
    """吞掉所有属性/方法. text() 返回 _text, setText() 更新 _text."""

    def __init__(self, *args, **kwargs):
        self._text = '0'

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
sys.modules['PyQt5.QtCore'].QTimer = _DummyPyQt5.QTimer
sys.modules['PyQt5.QtCore'].pyqtSignal = _DummyPyQt5.pyqtSignal
sys.modules['PyQt5.QtGui'] = type(sys)('PyQt5.QtGui')
sys.modules['PyQt5.QtGui'].QTextCursor = _DummyPyQt5.QTextCursor
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


def _make_panel_no_ui(parser):
    """构造 IdsReportPanel 但跳过 _build_ui, 直接设置必要属性.

    完整 _build_ui 会调 QLabel.setStyleSheet 等需要真 Qt mock 的方法.
    本测试只验 _on_detected 逻辑 (baseline 捕获 + delta 显示), 绕过 _build_ui.

    ⭐ 2026-10-19: 删 per-frame state 初始化 (_total_count / _sent_count / _normal_count
      / _tp_count / _fp_count / _tn_count / _fn_count).
    """
    panel = rtu_replay_gui.IdsReportPanel.__new__(rtu_replay_gui.IdsReportPanel)
    panel._parser = parser
    # ⭐ 第六十一轮 (2026-10-11): per-window state 也初始化 (供 _update_rates 派生
    #   Total/Normal/Attack/FPR). 测试场景不调 _on_window_stats, 必须手动设.
    panel._window_total_count = 0
    panel._window_normal_count = 0
    panel._window_attack_count = 0
    panel._detected_count = 0
    panel._detected_baseline = None
    panel.detected_label = _DummyQObject()  # ⭐ 第六十一轮: 已不存在, 但保留 mock
    #   以防测试函数意外访问 (assert panel.detected_label.text() 会触发 _DummyMethod 错误).
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
    # ⭐ 第四十八轮: per-window 评估器 (Round 48, 用于 _update_rates 末尾拉 W-* 指标)
    # ⭐⭐ 第六十一轮 (2026-10-11): 增 wdetected_label + 4 个新 per-window label mocks.
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
    # Round 28 baseline 锚点 (Round 44 修正: baseline = max(N-1, 0))
    parser.count_changed.connect(panel._on_detected)
    parser.tp_delta.connect(panel._on_tp_delta)
    parser.fp_delta.connect(panel._on_fp_delta)
    return panel


def _feed_count(parser, count_value):
    """模拟 H743 上行 [ALARM] N, 直接 emit count_changed(N)."""
    parser.count_changed.emit(count_value)


# ============================================================================
# 第二十八轮: baseline 锚点机制 (Round 44 修正: baseline = N-1 而非 N)
# ============================================================================

def test_round44_baseline_is_first_count_minus_one():
    """⭐ Round 44: 首个 [ALARM] N 触发后, baseline = max(N-1, 0), 不是 N.

    旧版 baseline=N 会让首条 emit 后 _detected_count=0 (丢 1). 新版 baseline=N-1
    让首条 emit 后 _detected_count=1, 末条 _detected_count = 上行消息数 m.
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    # H743 已有 frames_attack=20000 (用户没重启 H743)
    _feed_count(parser, 20000)
    assert panel._detected_baseline == 19999, (
        f"baseline 应=19999 (=首条N-1), got {panel._detected_baseline}"
    )
    assert panel._detected_count == 1, (
        f"首条 emit 后 _detected_count 应=1 (不是 0, 修复 off-by-one), got {panel._detected_count}"
    )


def test_round44_detected_increments_from_baseline():
    """后续 [ALARM] 递增 count, Detected 显示 count - baseline (从 1 开始)."""
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    _feed_count(parser, 20000)  # baseline=19999, count=1
    assert panel._detected_count == 1
    _feed_count(parser, 20001)
    assert panel._detected_count == 2, f"应=2, got {panel._detected_count}"
    _feed_count(parser, 20002)
    assert panel._detected_count == 3
    _feed_count(parser, 20500)
    assert panel._detected_count == 501, f"应=501 (= 20500-19999), got {panel._detected_count}"


def test_round44_detected_fresh_h743():
    """H743 fresh boot (N1=1): baseline=0, Detected 从 1 开始递增."""
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    _feed_count(parser, 1)
    assert panel._detected_baseline == 0, (
        f"fresh H743 首条 N=1 时 baseline 应=0, got {panel._detected_baseline}"
    )
    assert panel._detected_count == 1, f"应=1, got {panel._detected_count}"
    _feed_count(parser, 2)
    assert panel._detected_count == 2
    _feed_count(parser, 100)
    assert panel._detected_count == 100


def test_round28_detected_reset_clears_baseline():
    """reset_counters() 清 baseline, 下次首个 alarm 重新捕获."""
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    _feed_count(parser, 20000)
    assert panel._detected_baseline == 19999
    panel.reset_counters()
    assert panel._detected_baseline is None, (
        f"reset 后 baseline 应 None, got {panel._detected_baseline}"
    )
    # 新测试: H743 又跑了 500 次
    _feed_count(parser, 20500)
    assert panel._detected_baseline == 20499  # 20500 - 1
    assert panel._detected_count == 1


def test_round44_detected_label_text():
    """UI 显示的 wdetected_label.setText 文本应反映新 baseline 修正.

    ⭐ 第六十一轮 (2026-10-11): UI 已删 per-frame detected_label, 改用
      wdetected_label (per-window Detected).
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    # 模拟用户场景: H743 累计 20000, GUI 启动
    _feed_count(parser, 20000)
    assert panel.wdetected_label.text() == '1', (
        f"首条 label 应='1' (修复 off-by-one, 不是 '0'), got '{panel.wdetected_label.text()}'"
    )
    _feed_count(parser, 20005)
    assert panel.wdetected_label.text() == '6'  # 20005-19999=6 (旧版是 '5')
    _feed_count(parser, 20100)
    assert panel.wdetected_label.text() == '101'  # 20100-19999=101 (旧版是 '100')


# ============================================================================
# 第四十三轮 (回滚): Detected = TP + FP
# ============================================================================

def test_round43_revert_detected_equals_tp_plus_fp():
    """Detected KPI = H743 frames_attack delta = TP + FP (含 over-fire 部分).

    真实数据 (2026-10-07 23:36:51 auto-40ms):
      - TP = 56,666, FP = 8,610
      - Detected = 65,276 = 上行 [ALARM] 消息数 (Round 44 修复 off-by-one 后)

    ⭐ 2026-10-19: 删 _tp_count / _fp_count 断言, 改用直接数字验证.
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    # 模拟 fresh H743 boot, frames_attack 从 0 开始
    # 首条 [ALARM] N=1 (H743 已判 1 次 attack)
    parser.count_changed.emit(1)     # baseline=0, count=1
    # 累计 56,666 TP + 8,610 FP = 65,276 frames_attack 增量
    parser.tp_delta.emit(56666)
    parser.fp_delta.emit(8610)
    # 末条 [ALARM] N=65,276 (1 + 56,666 + 8,610)
    parser.count_changed.emit(65276)
    assert panel._detected_count == 65276, (
        f"Detected 应=65276 (=TP+FP delta, 等于上行消息数), got {panel._detected_count}"
    )
    assert panel.wdetected_label.text() == '65276'


def test_round43_revert_detected_exceeds_attack_by_over_fire():
    """Detected (TP+FP) > Attack (PC 已发攻击) 是正常 over-fire 现象.

    ⭐ 2026-10-19: per-frame state (_sent_count / _tp_count) 已删除.
    改用直接传入数字验证 Detected 与 Attack 关系.
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    attack_frames = 60048        # PC 已发攻击帧
    # Fresh H743
    parser.count_changed.emit(1)     # baseline=0, count=1
    parser.tp_delta.emit(56666)
    parser.fp_delta.emit(8610)
    parser.count_changed.emit(65276) # count=65276
    detected = panel._detected_count
    over_fire = detected - attack_frames    # = 65,276 - 60,048 = 5,228
    assert over_fire == 5228, f"over-fire 应=5228, got {over_fire}"


# ============================================================================
# 第四十四轮: off-by-one 修复专项测试
# ============================================================================

def test_round44_detected_matches_alarm_message_count():
    """核心修复: Detected 显示 == H743 上行 [ALARM] 消息数 (精确相等).

    模拟 m=10 个 [ALARM] 上行, N 值 1,2,3,...,10. 旧版末条 Detected=9 (少 1).
    新版 (baseline=N-1) 末条 Detected=10 ✓.
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    m = 10
    for i in range(1, m + 1):
        _feed_count(parser, i)
    assert panel._detected_count == m, (
        f"Detected 应=m={m} (上行 [ALARM] 消息数), got {panel._detected_count}"
    )


def test_round44_detected_baseline_zero_when_count_is_one():
    """边界: 首条 N=1 (fresh H743) → baseline = max(1-1, 0) = 0, 不变负数."""
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    _feed_count(parser, 1)
    assert panel._detected_baseline == 0
    assert panel._detected_count == 1


def test_round44_detected_baseline_defensive_against_zero():
    """防御性: 即使首条 N=0 (理论不可能, frames_attack=0 不上行) 也 baseline=0 不变负."""
    parser = rtu_replay_gui.IdsLineParser()
    panel = _make_panel_no_ui(parser)
    _feed_count(parser, 0)
    assert panel._detected_baseline == 0  # max(0-1, 0) = 0
    assert panel._detected_count == 0


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))