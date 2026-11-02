#!/usr/bin/env python3
"""验证 Round 28 TP budget cap — 保证 TP_count ≤ Attack_count, Recall ≤ 100%.

⭐ 2026-10-02 第二十八轮:
  Round 26/27 window=5 分类仍有 +309 个误标 TP (test_20261001-234034.csv Recall 101.91%).
  根因: H743 SW=16 滑动窗口重叠, 1 次 attack 触发多次 [ALARM] 上行 (over-fire),
  window-based 分类对 burst 内所有 alarm 都标 TP → TP_count > Attack_count.

  修复: IdsLineParser._tp_budget — 每条 PC attack 帧 +1, 每个 TP 标消耗 1, 不足则 FP.
"""
from __future__ import annotations
import sys
import os
import csv
from collections import deque
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

class _DummyQObject:
    def __init__(self, *args, **kwargs):
        pass

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
              'QProgressBar', 'QSlider'):
    setattr(sys.modules['PyQt5.QtWidgets'], _name, _DummyQObject)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

sys.modules['serial'] = type(sys)('serial')
sys.modules['serial'].Serial = type('Serial', (), {})
sys.modules['serial.tools'] = type(sys)('serial.tools')
sys.modules['serial.tools.list_ports'] = type(sys)('serial.tools.list_ports')
sys.modules['serial.tools.list_ports'].comports = lambda: []

import rtu_replay_gui  # noqa: E402


def _drive(parser, binary_seq, alarm_at_idx_list, tp_calls, fp_calls):
    """用 binary 序列 + alarm 触发位置驱动 parser.

    ⭐ 第二十八轮: 修正原 test_window_based_f1 的 drive 逻辑 bug — 当多个 ALARM 行共享
    同一 frame_idx (CSV 中常 2-3 个 [ALARM] 同行), 老 next_alarm 单次推进会卡在重复
    索引上, 后续 alarm 全漏掉. 改为: 每帧处理所有 ≤ 当前 i 的 alarm, 自动跳过过期 alarm.
    """
    alarm_iter = iter(alarm_at_idx_list)
    next_alarm = next(alarm_iter, None)
    for i, b in enumerate(binary_seq):
        parser.set_last_was_attack(bool(b))
        # 处理所有到当前帧为止的 alarm (含同帧多个, 跳过过期)
        while next_alarm is not None and next_alarm <= i:
            parser.feed(b'[ALARM] 1 prob=0.700\r\n')
            next_alarm = next(alarm_iter, None)


def _totals(tp_calls, fp_calls):
    """统计 parser 已 emit 的 TP/FP 总数 (来自连接回调)."""
    tp = sum(c[0] for c in tp_calls)
    fp = sum(c[0] for c in fp_calls)
    return tp, fp


def _attach_recorders(parser):
    """挂回调记录 tp_delta/fp_delta emit, 避免读类级别 tp_delta._calls (被其他测试污染).

    ⭐ 第二十八轮: 真实 PyQt5 的 Signal 是 instance-level, 但 dummy 实现是 class-level.
    dummy 测试中所有 IdsLineParser 实例共享同一个 _DummySignal 对象, 直接读 ._calls 会跨
    测试污染. 改用 connect 回调记录每次 emit 到独立 list, 保证隔离.
    """
    tp_calls = []
    fp_calls = []
    parser.tp_delta.connect(lambda *a: tp_calls.append(a))
    parser.fp_delta.connect(lambda *a: fp_calls.append(a))
    return tp_calls, fp_calls


# === 单元测试: budget 机制 ===

def test_round28_budget_increments_on_attack_frame():
    """PC 发送 attack 帧 → budget +1."""
    parser = rtu_replay_gui.IdsLineParser()
    assert parser._tp_budget == 0
    parser.set_last_was_attack(True)
    assert parser._tp_budget == 1, f"attack 帧应+1 budget, got {parser._tp_budget}"
    parser.set_last_was_attack(True)
    assert parser._tp_budget == 2, f"两次 attack 应+2, got {parser._tp_budget}"


def test_round28_budget_unchanged_on_normal_frame():
    """PC 发送 normal 帧 → budget 不变."""
    parser = rtu_replay_gui.IdsLineParser()
    parser.set_last_was_attack(True)  # budget=1
    parser.set_last_was_attack(False)  # normal
    assert parser._tp_budget == 1, f"normal 帧不应消耗 budget, got {parser._tp_budget}"


def test_round28_budget_reset_clears():
    """reset_last_was_attack → budget 清零."""
    parser = rtu_replay_gui.IdsLineParser()
    for _ in range(5):
        parser.set_last_was_attack(True)
    assert parser._tp_budget == 5
    parser.reset_last_was_attack()
    assert parser._tp_budget == 0, f"reset 后 budget 应清零, got {parser._tp_budget}"


def test_round28_alarm_without_attack_is_fp():
    """无 attack 帧时 [ALARM] → FP (budget=0 不能给 TP)."""
    parser = rtu_replay_gui.IdsLineParser()
    tp_calls, fp_calls = _attach_recorders(parser)
    # 先填 SW_SIZE=16 个 normal 帧让 window 就绪
    for _ in range(16):
        parser.set_last_was_attack(False)
    assert parser._tp_budget == 0
    # 触发 alarm
    parser.feed(b'[ALARM] 1 prob=0.700\r\n')
    tp, fp = _totals(tp_calls, fp_calls)
    assert tp == 0, f"无 attack 时 alarm 应 FP, 但 TP={tp}"
    assert fp == 1, f"应有 1 FP, got {fp}"


def test_round28_alarm_with_attack_window_is_tp():
    """window=5 有 attack 且 budget>=1 → TP."""
    parser = rtu_replay_gui.IdsLineParser()
    tp_calls, fp_calls = _attach_recorders(parser)
    for _ in range(15):
        parser.set_last_was_attack(False)
    parser.set_last_was_attack(True)  # 第 16 帧 attack, budget=1
    parser.feed(b'[ALARM] 1 prob=0.700\r\n')
    tp, fp = _totals(tp_calls, fp_calls)
    assert tp == 1, f"应 1 TP, got {tp}"
    assert fp == 0


def test_round28_budget_caps_over_fire():
    """H743 over-fire: 1 attack 触发 N 个 alarm → 只有 1 TP, 余 N-1 个 FP.

    场景: PC 发送 1 个 attack 帧 (budget=1), H743 上行 5 个 alarm (over-fire).
    ├─ 不修复: 5 个 alarm 全标 TP (window 含 attack) → TP=5 > Attack=1.
    └─ 修复: 只有 1 TP (budget 耗尽), 4 FP → TP ≤ Attack ✓
    """
    parser = rtu_replay_gui.IdsLineParser()
    tp_calls, fp_calls = _attach_recorders(parser)
    # 15 normal + 1 attack (budget=1)
    for _ in range(15):
        parser.set_last_was_attack(False)
    parser.set_last_was_attack(True)  # budget=1, attack=1
    # 5 个 [ALARM] over-fire, delta=1 each
    for _ in range(5):
        parser.feed(b'[ALARM] 1 prob=0.700\r\n')
    tp, fp = _totals(tp_calls, fp_calls)
    attack_sent = 1
    assert tp == 1, f"over-fire 应只 1 TP (budget 限), got {tp}"
    assert fp == 4, f"应 4 FP (over-fire 部分), got {fp}"
    assert tp <= attack_sent, f"TP {tp} > Attack {attack_sent}, Recall 仍超 100%"


def test_round28_attack_count_equals_tp_count_invariant():
    """不变量: 任何测试中 TP_count ≤ Attack_count (PC 发送 attack 帧数)."""
    parser = rtu_replay_gui.IdsLineParser()
    tp_calls, fp_calls = _attach_recorders(parser)
    # 模拟: 100 个 attack 帧, 100 个 normal 帧交替, 但 alarm 在 attack 后立即爆发
    # H743 模拟: 每 attack 帧后立即发 1 alarm (delta=1)
    n_attacks = 0
    for i in range(200):
        if i % 3 == 0:
            parser.set_last_was_attack(True)
            n_attacks += 1
        else:
            parser.set_last_was_attack(False)
        # 模拟 H743 在 attack 后立即报 alarm (delta=1)
        if n_attacks > 0 and i % 3 == 1:
            parser.feed(b'[ALARM] 1 prob=0.700\r\n')
    tp, fp = _totals(tp_calls, fp_calls)
    assert tp <= n_attacks, (
        f"INVARIANT VIOLATED: TP {tp} > Attack {n_attacks} "
        f"→ Recall {tp/n_attacks*100:.2f}% > 100%"
    )


def test_round28_real_log_recall_capped():
    """用真实测试日志 test_20261001-234034.csv 验证 TP ≤ Attack.

    老逻辑跑这个测试 Recall=101.91% (TP=16513 vs Attack=16204, +309).
    新逻辑应保证 TP ≤ Attack.
    """
    path = 'logs/test_20261001-234034.csv'
    if not os.path.exists(path):
        import pytest
        pytest.skip(f'{path} 不存在')
    binary_seq = []
    alarm_at_idx_list = []
    with open(path, encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r['event'] == 'SEND':
                binary_seq.append(int(r['binary']) if r['binary'] else 0)
            elif r['event'] == 'ALARM':
                if binary_seq:
                    alarm_at_idx_list.append(len(binary_seq) - 1)
    parser = rtu_replay_gui.IdsLineParser()
    tp_calls, fp_calls = _attach_recorders(parser)
    _drive(parser, binary_seq, alarm_at_idx_list, tp_calls, fp_calls)
    tp, fp = _totals(tp_calls, fp_calls)
    attack_sent = sum(binary_seq)
    print(f"  test_20261001-234034.csv: Attack={attack_sent} TP={tp} FP={fp}")
    print(f"  Recall={tp/attack_sent*100:.2f}% (was 101.91%)")
    assert tp <= attack_sent, (
        f"TP {tp} > Attack {attack_sent}, Recall={tp/attack_sent*100:.2f}% 仍 > 100%"
    )
    assert tp > attack_sent * 0.5, (
        f"TP {tp} 太低 (Attack {attack_sent}, Recall {tp/attack_sent*100:.2f}%), "
        f"可能 budget 太严 — 验证 window 逻辑没坏"
    )


def test_round28_carry_over_removed():
    """验证 Round 25 carry-over (saw_tp) 已移除 — window-based 下不应有 batched FP→TP 转换.

    老 carry-over 逻辑: batched alarm 中任一 TP → 后续 FP 改判 TP.
    新逻辑: 纯 budget 限, 不做 carry-over.

    场景: budget 耗尽后再发一批 alarm, 不应继承之前 batch 的 TP.
    """
    parser = rtu_replay_gui.IdsLineParser()
    tp_calls, fp_calls = _attach_recorders(parser)
    for _ in range(15):
        parser.set_last_was_attack(False)
    parser.set_last_was_attack(True)  # budget=1
    parser.feed(b'[ALARM] 1 prob=0.700\r\n')  # TP, budget→0
    # 第二个 alarm (window 仍含 attack), 老 carry-over 会让它也 TP, 但 budget=0 → FP
    parser.feed(b'[ALARM] 1 prob=0.700\r\n')
    tp, fp = _totals(tp_calls, fp_calls)
    assert tp == 1, f"应有 1 TP (budget 限), got {tp}"
    assert fp == 1, f"第 2 个 alarm 应 FP (budget=0), got {fp}"


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))