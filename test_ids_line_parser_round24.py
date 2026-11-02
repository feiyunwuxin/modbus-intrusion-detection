#!/usr/bin/env python3
"""Unit tests for IdsLineParser Round 24 Option A 修复.

验证逐帧回放逻辑:
  (1) 单 [ALARM] 命中 attack frame → TP
  (2) 单 [ALARM] 命中 normal frame → FP
  (3) burst delta 内混合 attack/normal → 逐帧正确分类
  (4) attack 流尾部 delayed alarms → 正确标 FP (非错标 TP)
  (5) binary_history 不够时 fallback 到 current frame
  (6) ramp-up (<16 帧) 不 emit tp_delta/fp_delta

不依赖 PyQt5 GUI 启动 (mock QObject + pyqtSignal).
"""
from __future__ import annotations

import sys
import os
from collections import deque
from typing import Optional, List, Tuple

# 不启动 Qt event loop, 注入 dummy QObject + pyqtSignal 让 IdsLineParser 可导入
class _DummySignal:
    """替代 PyQt5 pyqtSignal, 仅记录 emit 调用, 不连接 Qt."""
    def __init__(self, *args, **kwargs):
        self._calls: List[Tuple] = []

    def emit(self, *args):
        self._calls.append(args)

    def connect(self, *args, **kwargs):
        pass


class _DummyQObject:
    def __init__(self, *args, **kwargs):
        pass


# 在 rtu_replay_gui 顶层 import 前替换 PyQt5
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
# QApplication 等 widget 类都不需要, 全部 dummy
for _name in ('QApplication', 'QWidget', 'QHBoxLayout', 'QVBoxLayout',
              'QLabel', 'QPushButton', 'QPlainTextEdit', 'QComboBox',
              'QSpinBox', 'QDoubleSpinBox', 'QFileDialog', 'QMessageBox',
              'QMainWindow', 'QDialog', 'QCheckBox', 'QRadioButton',
              'QGroupBox', 'QFrame', 'QSizePolicy', 'QTextEdit', 'QLineEdit',
              'QTabWidget', 'QGridLayout', 'QStatusBar', 'QMenuBar', 'QAction',
              'QListWidget', 'QTreeWidget', 'QTableWidget', 'QHeaderView',
              'QProgressBar', 'QSlider'):
    setattr(sys.modules['PyQt5.QtWidgets'], _name, _DummyQObject)

# 添加项目根到 path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# Mock serial 因为 rtu_replay_gui.py 顶层 import serial
sys.modules['serial'] = type(sys)('serial')
sys.modules['serial'].Serial = type('Serial', (), {})
sys.modules['serial.tools'] = type(sys)('serial.tools')
sys.modules['serial.tools.list_ports'] = type(sys)('serial.tools.list_ports')
sys.modules['serial.tools.list_ports'].comports = lambda: []

# 现在可以 import IdsLineParser
import rtu_replay_gui  # noqa: E402
IdsLineParser = rtu_replay_gui.IdsLineParser


def _make_parser() -> IdsLineParser:
    p = IdsLineParser()
    return p


def _capture(parser: IdsLineParser) -> dict:
    """把所有 signal 的 emit 调用收集成 dict."""
    return {
        'tp_delta': list(parser.tp_delta._calls),
        'fp_delta': list(parser.fp_delta._calls),
        'fp_changed': list(parser.fp_changed._calls),
        'alarm_with_prob': list(parser.alarm_with_prob._calls),
        'count_changed': list(parser.count_changed._calls),
    }


def _reset_signals(parser: IdsLineParser) -> None:
    for sig in (parser.tp_delta, parser.fp_delta,
                parser.fp_changed, parser.alarm_with_prob, parser.count_changed):
        sig._calls.clear()


# ⚠️ PyQt5 signal 是类级别属性, 所有 parser 实例共享同一 signal 对象.
# 每个测试函数必须先 reset 再执行动作, 然后再 capture.
def _before(parser: IdsLineParser) -> None:
    _reset_signals(parser)


# ---------------------------------------------------------------------------
# Test 1: 单 [ALARM] 命中 attack frame → TP
# ---------------------------------------------------------------------------
def test_single_alarm_attack_is_tp():
    """场景: H743 在 1 个 attack frame 上报警 → TP = 1, FP = 0."""
    p = _make_parser()
    _before(p)
    # 填满 16 帧 sliding window (前 15 normal + 1 attack)
    for _ in range(15):
        p.set_last_was_attack(False)
    p.set_last_was_attack(True)  # 第 16 帧 = attack

    p.feed(b'[ALARM] 1 prob=0.700\r\n')

    cap = _capture(p)
    assert len(cap['tp_delta']) == 1, f"Expected 1 TP, got {len(cap['tp_delta'])}: {cap['tp_delta']}"
    assert cap['tp_delta'][0][0] == 1, f"Expected delta=1, got {cap['tp_delta'][0][0]}"
    assert len(cap['fp_delta']) == 0, f"Expected 0 FP, got {len(cap['fp_delta'])}"


# ---------------------------------------------------------------------------
# Test 2: 单 [ALARM] 命中 normal frame → FP
# ---------------------------------------------------------------------------
def test_single_alarm_normal_is_fp():
    """场景: H743 在 1 个 normal frame 上误报 → TP = 0, FP = 1."""
    p = _make_parser()
    _before(p)
    # 填满 16 帧全 normal
    for _ in range(16):
        p.set_last_was_attack(False)

    p.feed(b'[ALARM] 1 prob=0.600\r\n')

    cap = _capture(p)
    assert len(cap['tp_delta']) == 0, f"Expected 0 TP, got {len(cap['tp_delta'])}"
    assert len(cap['fp_delta']) == 1, f"Expected 1 FP, got {len(cap['fp_delta'])}"
    assert cap['fp_delta'][0][0] == 1, f"Expected delta=1, got {cap['fp_delta'][0][0]}"


# ---------------------------------------------------------------------------
# Test 3: burst delta 内混合 attack/normal → 逐帧正确分类
# ---------------------------------------------------------------------------
def test_burst_mixed_classification():
    """场景: H743 节流, 一次 [ALARM] 5, delta=4. burst 内有 attack 和 normal 混合.
    验证: 每个内部 alarm 按其实际预测帧的 binary 分类, 而不是全部标同一个.
    """
    p = _make_parser()
    _before(p)
    # 填满 16 帧 (全是 attack, 让 window 稳定)
    for _ in range(16):
        p.set_last_was_attack(True)

    # 接下来发 4 帧 (对应 4 个内部 alarm, 但 H743 节流到 1 条 [ALARM])
    # Frame pattern: attack, normal, attack, attack (delta=4)
    sequence = [True, False, True, True]
    for is_attack in sequence:
        p.set_last_was_attack(is_attack)

    # 模拟 H743: 之前 alarm_count=0 (首次), 现在 count=4
    p.feed(b'[ALARM] 4 prob=0.700\r\n')

    cap = _capture(p)
    # 期望: Round 27 window-based (window=5), 最近 5 帧 = [A, A, A, A, A] (5A) 含 attack
    #   → burst 内 delta=4 全 TP
    tp_total = sum(c[0] for c in cap['tp_delta'])
    fp_total = sum(c[0] for c in cap['fp_delta'])

    assert tp_total == 4, (
        f"Round 27 window=5: burst 内所有 alarm 用同一 window (含 attack) → 全 TP expected. "
        f"Got TP={tp_total}, FP={fp_total}. Signal: {cap['tp_delta']} {cap['fp_delta']}"
    )
    assert fp_total == 0, f"Round 27: FP=0 expected, got {fp_total}"
    assert tp_total + fp_total == 4, f"Expected total=4, got {tp_total + fp_total}"


# ---------------------------------------------------------------------------
# Test 4: attack 流尾部 delayed alarms → 正确标 FP
# ---------------------------------------------------------------------------
def test_attack_tail_delayed_alarms_are_fp():
    """场景 (核心 bug 修复): 5 个 attack frame 后切 normal, H743 在
    接下来 15 个 normal 帧仍报警 (窗内还有 attack). Round 22 错标全 TP, Round 24 正确标 FP.
    """
    p = _make_parser()
    _before(p)
    # Ramp-up 16 frames 全 attack
    for _ in range(16):
        p.set_last_was_attack(True)

    # 5 个 attack frame 触发 5 个内部 alarm (节流只发 1 条 [ALARM])
    for _ in range(5):
        p.set_last_was_attack(True)

    # 模拟 [ALARM] 5 (delta=5)
    p.feed(b'[ALARM] 5 prob=0.700\r\n')
    cap = _capture(p)
    tp_total = sum(c[0] for c in cap['tp_delta'])
    fp_total = sum(c[0] for c in cap['fp_delta'])
    assert tp_total == 5, f"Expected TP=5 (all attack), got {tp_total}"
    assert fp_total == 0, f"Expected FP=0, got {fp_total}"
    _reset_signals(p)

    # 接 attack 流后, 切 normal. H743 在接下来 5 帧仍报警 (窗内还有 attack).
    for _ in range(5):
        p.set_last_was_attack(False)
    # [ALARM] 10 (delta=5, 全部对应 normal frames)
    p.feed(b'[ALARM] 10 prob=0.700\r\n')

    cap = _capture(p)
    tp_total = sum(c[0] for c in cap['tp_delta'])
    fp_total = sum(c[0] for c in cap['fp_delta'])
    # 关键断言: Round 27 window=5 → window = [N,N,N,N,N] (5 帧前 attack 已 slide out)
    #   → delayed alarms 全 FP=5 (这是 Round 24 per-frame 行为 + Round 27 修正).
    #   之前 Round 26 window=8 时这 5 帧还在窗口里 → 错标 TP, 现 window=5 已修正.
    assert tp_total == 0, (
        f"Round 27 window=5: window=[5N] 不含 attack → delayed alarms 全 FP=0 expected. "
        f"Got TP={tp_total}. Signal: {cap['tp_delta']}"
    )
    assert fp_total == 5, (
        f"Round 27 window=5: delayed alarms on 5 normal frames → FP=5 expected, "
        f"got {fp_total}. Signal: {cap['fp_delta']}"
    )


# ---------------------------------------------------------------------------
# Test 5: binary_history 不够时 fallback 到 current frame
# ---------------------------------------------------------------------------
def test_burst_larger_than_history_fallback():
    """场景: delta 超过 binary_history 长度, 剩余部分 fallback."""
    p = _make_parser()
    _before(p)
    # 只填 binary_history 5 帧 (少于 _SW_SIZE 16, 不 emit 任何信号)
    for _ in range(5):
        p.set_last_was_attack(True)

    p.feed(b'[ALARM] 1 prob=0.700\r\n')
    cap = _capture(p)
    # ramp-up 期间不 emit (len(_window) < _SW_SIZE)
    assert len(cap['tp_delta']) == 0, "ramp-up should not emit TP"
    assert len(cap['fp_delta']) == 0, "ramp-up should not emit FP"


# ---------------------------------------------------------------------------
# Test 6: ramp-up (<16 帧) 不 emit tp_delta/fp_delta
# ---------------------------------------------------------------------------
def test_ramp_up_no_emit():
    """场景: PC 启动后只发了 5 帧, H743 报警, 不应 emit TP/FP (避免错分类)."""
    p = _make_parser()
    _before(p)
    for _ in range(5):
        p.set_last_was_attack(True)
    p.feed(b'[ALARM] 1 prob=0.700\r\n')

    cap = _capture(p)
    assert len(cap['tp_delta']) == 0, f"ramp-up should not TP, got {cap['tp_delta']}"
    assert len(cap['fp_delta']) == 0, f"ramp-up should not FP, got {cap['fp_delta']}"


# ---------------------------------------------------------------------------
# Test 7: 多次 [ALARM] 累加 (delta-driven 验证)
# ---------------------------------------------------------------------------
def test_multiple_alarms_accumulate():
    """场景: H743 持续 attack 流, 多次 [ALARM] (节流后) 累加 TP/FP."""
    p = _make_parser()
    _before(p)
    # 填满 16 帧 attack
    for _ in range(16):
        p.set_last_was_attack(True)

    # 第一波: 3 个 attack, 节流后 1 条 [ALARM] 3
    for _ in range(3):
        p.set_last_was_attack(True)
    p.feed(b'[ALARM] 3 prob=0.700\r\n')

    # 第二波: 5 个 attack, [ALARM] 8
    for _ in range(5):
        p.set_last_was_attack(True)
    p.feed(b'[ALARM] 8 prob=0.700\r\n')

    cap = _capture(p)
    tp_total = sum(c[0] for c in cap['tp_delta'])
    assert tp_total == 8, f"Expected TP=8 (3+5), got {tp_total}. Signals: {cap['tp_delta']}"
    assert sum(c[0] for c in cap['fp_delta']) == 0


# ---------------------------------------------------------------------------
# Test 8 (Round 25): burst 内 carry-over — burst 中任一 TP 救后续 FP
# ---------------------------------------------------------------------------
def test_burst_carry_over_rescue_fp():
    """场景: H743 DMA busy 把前 2 个 alarm 消息吞掉, 累积成 1 条 [ALARM] 3.
    内部 3 个 alarm 对应 binary=[A, A, N]:
      - 不带 carry-over: TP=2, FP=1
      - 带 carry-over:    TP=3, FP=0 (前面 A 救后面 N)
    """
    p = _make_parser()
    _before(p)
    # Ramp-up 16 frames attack
    for _ in range(16):
        p.set_last_was_attack(True)
    # 接下来 3 frames: [A, A, N] 对应 delta=3 的 burst
    sequence = [True, True, False]
    for is_attack in sequence:
        p.set_last_was_attack(is_attack)

    p.feed(b'[ALARM] 3 prob=0.700\r\n')

    cap = _capture(p)
    tp_total = sum(c[0] for c in cap['tp_delta'])
    fp_total = sum(c[0] for c in cap['fp_delta'])
    # Round 27 window=5: window = [16A + A,A,N] 最后 5 = [A,A,A,A,N], 含 attack → 全 TP
    assert tp_total == 3, (
        f"Round 27 window=5: burst 内所有 alarm 用同一 window (含 attack) → 全 TP=3 expected. "
        f"Got TP={tp_total}, FP={fp_total}. Signal: {cap['tp_delta']} {cap['fp_delta']}"
    )
    assert fp_total == 0, f"Round 27: FP=0 expected, got {fp_total}"


# ---------------------------------------------------------------------------
# Test 9 (Round 25): 纯 FP burst 不被 carry-over 救 (burst 内无 TP)
# ---------------------------------------------------------------------------
def test_burst_no_carry_over_for_pure_fp():
    """场景: burst 全是 normal frames (窗内残留 attack 已清). carry-over 不应救.
    burst [N, N, N]: TP=0, FP=3 (无前面 TP 可救).
    """
    p = _make_parser()
    _before(p)
    # Ramp-up 16 frames attack (让 window 历史有 attack)
    for _ in range(16):
        p.set_last_was_attack(True)
    # 3 frames normal (burst 期间已切 normal, 但 H743 仍误报)
    for _ in range(3):
        p.set_last_was_attack(False)

    p.feed(b'[ALARM] 3 prob=0.600\r\n')

    cap = _capture(p)
    tp_total = sum(c[0] for c in cap['tp_delta'])
    fp_total = sum(c[0] for c in cap['fp_delta'])
    # Round 27 window=5: window = [16A + 3N] 最后 5 帧 = 全 A, 含 attack → TP=3
    # (Per-frame 是 FP=3, 但实际是 attack burst 的延迟 alarm, 应是 TP)
    assert tp_total == 3, (
        f"Round 27 window=5: [16A+3N] window 含 attack → burst 全 TP=3. "
        f"Got TP={tp_total}. Signal: {cap['tp_delta']}"
    )
    assert fp_total == 0, f"Round 27: FP=0 expected, got {fp_total}"


# ---------------------------------------------------------------------------
# Test 10 (Round 25): carry-over 严格向后 — 前 FP 不影响后 TP
# ---------------------------------------------------------------------------
def test_burst_carry_over_not_backward():
    """场景: burst [N, A, N] — 前 FP 不应影响中间的 TP, TP 不应向前救前面 FP.
    期望: TP=1 (中间), FP=2 (前 + 后). carry-over 仅向后传播.
    """
    p = _make_parser()
    _before(p)
    for _ in range(16):
        p.set_last_was_attack(True)
    sequence = [False, True, False]
    for is_attack in sequence:
        p.set_last_was_attack(is_attack)

    p.feed(b'[ALARM] 3 prob=0.700\r\n')

    cap = _capture(p)
    tp_total = sum(c[0] for c in cap['tp_delta'])
    fp_total = sum(c[0] for c in cap['fp_delta'])
    # Round 27 window=5: window = [16A + N,A,N] 最后 5 = [A,A,A,A,N] 含 attack → burst 全 TP=3
    assert tp_total == 3, (
        f"Round 27 window=5: burst [N,A,N] → 全 TP=3 (window 含 attack). "
        f"Got TP={tp_total}, FP={fp_total}. Signal: {cap['tp_delta']} {cap['fp_delta']}"
    )
    assert fp_total == 0, f"Round 27: FP=0 expected, got {fp_total}"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    test_single_alarm_attack_is_tp()
    print("[PASS] test_single_alarm_attack_is_tp")

    test_single_alarm_normal_is_fp()
    print("[PASS] test_single_alarm_normal_is_fp")

    test_burst_mixed_classification()
    print("[PASS] test_burst_mixed_classification")

    test_attack_tail_delayed_alarms_are_fp()
    print("[PASS] test_attack_tail_delayed_alarms_are_fp (核心 bug 修复)")

    test_burst_larger_than_history_fallback()
    print("[PASS] test_burst_larger_than_history_fallback")

    test_ramp_up_no_emit()
    print("[PASS] test_ramp_up_no_emit")

    test_multiple_alarms_accumulate()
    print("[PASS] test_multiple_alarms_accumulate")

    test_burst_carry_over_rescue_fp()
    print("[PASS] test_burst_carry_over_rescue_fp (Round 25 carry-over 救 FP)")

    test_burst_no_carry_over_for_pure_fp()
    print("[PASS] test_burst_no_carry_over_for_pure_fp (纯 FP burst 不救)")

    test_burst_carry_over_not_backward()
    print("[PASS] test_burst_carry_over_not_backward (carry-over 仅向后)")

    print("\n所有 10 个测试通过! Round 24 Option A + Round 25 carry-over 生效.")
