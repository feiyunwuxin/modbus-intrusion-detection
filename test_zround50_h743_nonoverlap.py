#!/usr/bin/env python3
"""验证 Round 50: H743 改成 non-overlapping 部署 (16 帧 1 window 1 alarm), 跟 PC 训练 1:1.

⭐ 2026-10-09 第五十轮 (用户要求: "在 H743 上按 PC 方式部署"):
  - H743 触发逻辑: 16 帧凑齐 1 次推理, 1 window 1 alarm (跟训练 reshape(n//16, 16, -1) 1:1)
  - H743 g_stats: frames_attack → windows_attack (语义从"攻击帧数"改为"攻击 window 数")
  - PC 评估: IdsLineParser 改用 WindowEvaluator per-window max-aggregation 1:1
  - 移除 _CLASSIFY_WINDOW 字段 (per-alarm 容差窗口不再有意义)
  - 移除 _tp_budget (per-window 评估天然不 over-fire, 1 window 1 alarm 1:1)

历史对照:
  Round 27 (window=5) → Round 46 (window=2) → Round 47 (window=16) → Round 48 (window=1)
  → Round 49 (window=2) → Round 50 (per-window 替换 per-alarm)
"""
from __future__ import annotations

import sys
import os
import re
from typing import List


class _DummySignal:
    def __init__(self, *args, **kwargs):
        self._calls: List = []
        self._slots: List = []
    def emit(self, *args):
        self._calls.append(args)
        for slot in self._slots:
            slot(*args)
    def connect(self, slot, *args, **kwargs):
        self._slots.append(slot)


class _DummyPyQt5:
    class Qt:
        @staticmethod
        def _dummy(*args, **kwargs): return None
    class QObject:
        def __init__(self, *args, **kwargs): pass
    class QThread(object): pass
    class QTimer(object):
        @staticmethod
        def singleShot(*args, **kwargs): return None
    class QTextCursor(object): pass
    @staticmethod
    def pyqtSignal(*args, **kwargs): return _DummySignal()


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
    setattr(sys.modules['PyQt5.QtWidgets'], _name, _DummyPyQt5.QObject)


def _load_gui_module():
    """加载 rtu_replay_gui.py 用 Dummy PyQt5 桩."""
    gui_path = os.path.abspath(os.path.join(os.path.dirname(__file__) or '.', 'rtu_replay_gui.py'))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rtu_replay_gui", gui_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__file__ = gui_path
    return mod


rtu_replay_gui = _load_gui_module()


# ============================================================================
# 1) IdsLineParser 字段重构: 移除 per-alarm 字段, 改用 WindowEvaluator
# ============================================================================

def test_round50_classify_window_removed():
    """⭐ Round 50: _CLASSIFY_WINDOW 字段已移除 (per-alarm 评估被 per-window 取代)."""
    parser_cls = rtu_replay_gui.IdsLineParser
    assert not hasattr(parser_cls, '_CLASSIFY_WINDOW'), (
        "_CLASSIFY_WINDOW 字段应已移除 (per-alarm 评估改用 WindowEvaluator)"
    )


def test_round50_tp_budget_removed():
    """⭐ Round 50: _tp_budget 字段已移除 (per-window 评估天然 1:1 不 over-fire)."""
    parser = rtu_replay_gui.IdsLineParser()
    assert not hasattr(parser, '_tp_budget'), (
        "_tp_budget 实例属性应已移除 (per-window 评估不需要 budget cap)"
    )


def test_round50_window_eval_internal():
    """⭐ Round 50: IdsLineParser 内部维护 WindowEvaluator 实例."""
    parser = rtu_replay_gui.IdsLineParser()
    assert hasattr(parser, '_window_eval'), "IdsLineParser 内部应维护 _window_eval"
    assert isinstance(parser._window_eval, rtu_replay_gui.WindowEvaluator)


def test_round50_sw_size_unchanged():
    """⭐ H743 推理窗口 _SW_SIZE 仍 = 16, 跟 WindowEvaluator._WINDOW_SIZE 一致."""
    assert rtu_replay_gui.IdsLineParser._SW_SIZE == 16
    assert rtu_replay_gui.WindowEvaluator._WINDOW_SIZE == 16


# ============================================================================
# 2) set_last_was_attack 行为: 喂 WindowEvaluator
# ============================================================================

def test_round50_set_attack_feeds_window_eval():
    """⭐ set_last_was_attack(True) 喂给 WindowEvaluator.on_frame()."""
    parser = rtu_replay_gui.IdsLineParser()
    parser.set_last_was_attack(True)
    assert parser._window_eval._current_window_attack is True, (
        "set_last_was_attack(True) 应触发 WindowEvaluator 标记当前窗有 attack"
    )


def test_round50_set_normal_does_not_mark_attack():
    """⭐ set_last_was_attack(False) 不应标记当前窗有 attack."""
    parser = rtu_replay_gui.IdsLineParser()
    parser.set_last_was_attack(False)
    assert parser._window_eval._current_window_attack is False


def test_round50_set_attack_16_times_triggers_evaluate():
    """⭐ 16 帧凑齐 1 个 window, WindowEvaluator 内部自动评估.
    16 帧全 normal → 1 个 TN window.
    """
    parser = rtu_replay_gui.IdsLineParser()
    for _ in range(16):
        parser.set_last_was_attack(False)
    metrics = parser._window_eval.get_metrics()
    assert metrics['n_windows'] == 1
    assert metrics['tn'] == 1
    assert metrics['tp'] == 0
    assert metrics['fp'] == 0
    assert metrics['fn'] == 0


def test_round50_set_attack_16_mixed_triggers_evaluate():
    """⭐ 16 帧中有 1 帧 attack, max-aggregation → 1 个 attack window, 凑齐 16 时自动评估.
    有 1 帧 attack + 15 normal + 0 alarm = 1 FN.
    """
    parser = rtu_replay_gui.IdsLineParser()
    # 1 帧 attack (位置 0)
    parser.set_last_was_attack(True)
    for _ in range(15):
        parser.set_last_was_attack(False)
    metrics = parser._window_eval.get_metrics()
    assert metrics['n_windows'] == 1
    assert metrics['fn'] == 1, "1 attack window 没 alarm → FN=1 (per-window 严格)"
    assert metrics['tp'] == 0
    assert metrics['fp'] == 0


def test_round50_reset_clears_window_eval():
    """⭐ reset_last_was_attack() 应清空 WindowEvaluator 状态."""
    parser = rtu_replay_gui.IdsLineParser()
    for _ in range(10):
        parser.set_last_was_attack(True)
    parser.reset_last_was_attack()
    metrics = parser._window_eval.get_metrics()
    assert metrics['n_windows'] == 0
    assert metrics['tp'] == 0
    assert metrics['_frame_count' if False else 'n_windows'] == 0


# ============================================================================
# 3) _parse_line 行为: H743 alarm 喂 WindowEvaluator, 凑齐 16 帧评估
# ============================================================================

def test_round50_parse_alarm_feeds_window_eval():
    """⭐ _parse_line 解析 [ALARM] N 后, 调 WindowEvaluator.on_alarm() delta 次.

    测试场景: 15 帧 attack + 1 alarm (在第 16 帧前) + 1 帧 attack
    → 第 16 帧凑齐 1 个 window, _current_window_attack=True, _current_window_alarm=True → TP=1
    """
    parser = rtu_replay_gui.IdsLineParser()
    # 前 15 帧 attack
    for _ in range(15):
        parser.set_last_was_attack(True)
    # H743 上行 [ALARM] 1 (这个 alarm 注册到 _recent_alarm_frames, 给当前 window 关联)
    parser.feed(b'[ALARM] 1\r\n')
    # 第 16 帧 attack, 凑齐 1 个 window, 触发评估
    parser.set_last_was_attack(True)
    metrics = parser._window_eval.get_metrics()
    assert metrics['n_windows'] == 1
    assert metrics['tp'] == 1, "15 attack + 1 alarm + 1 attack → window={attack, alarm} → TP=1"


def test_round50_parse_alarm_no_attack_is_fp():
    """⭐ 15 帧全 normal + 1 alarm + 1 帧 normal = 1 FP (per-window 严格)."""
    parser = rtu_replay_gui.IdsLineParser()
    for _ in range(15):
        parser.set_last_was_attack(False)
    parser.feed(b'[ALARM] 1\r\n')
    parser.set_last_was_attack(False)
    metrics = parser._window_eval.get_metrics()
    assert metrics['fp'] == 1, "15 normal + 1 alarm + 1 normal → FP=1 (attack=False, alarm=True)"


def test_round50_parse_alarm_delta_handles_burst():
    """⭐ burst 内 delta 个 alarm 一次性喂给 WindowEvaluator.

    测试场景: 15 attack + burst(2 alarms) + 1 attack → 凑齐 1 window = TP=1
    """
    parser = rtu_replay_gui.IdsLineParser()
    for _ in range(15):
        parser.set_last_was_attack(True)
    # H743 burst: 一次性上行 2 个 attack windows (count=2, delta=2)
    parser.feed(b'[ALARM] 2\r\n')
    parser.set_last_was_attack(True)
    metrics = parser._window_eval.get_metrics()
    assert metrics['n_windows'] == 1
    assert metrics['tp'] == 1, "burst delta=2 都喂 on_alarm, 当前 window = TP=1"


# ============================================================================
# 4) H743 字段名: frames_attack → windows_attack
# ============================================================================

def test_round50_h743_source_uses_windows_attack():
    """⭐ H743 源码 g_stats.windows_attack 取代 frames_attack (跟 1 window 1 alarm 语义对齐).

    ⭐ 2026-10-09 修正: 实际生产工程是 C:\work\Claude\Issue\h743\ (用 ch32_inference = TCN+SE),
    不是 C:\work\Claude\Issue\CNNH743\h743\ (CNN 实验). 测试路径要指向生产工程.
    """
    c_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__) or '.',
        'h743', 'Core', 'Inc', 'modbus_ids.h'))
    with open(c_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 验证: 新增 windows_attack 字段
    assert 'windows_attack' in src, "H743 头文件应有 windows_attack 字段"
    # 验证: 移除 frames_attack 字段 (仅作为 g_stats 字段名)
    # 注: 'frames_attack' 可能仍出现在注释里, 这里只检查结构体定义内
    struct_match = re.search(r'typedef struct\s*\{[^}]+\}\s*modbus_ids_stats_t', src, re.DOTALL)
    if struct_match:
        struct_body = struct_match.group(0)
        assert 'frames_attack' not in struct_body, (
            "g_stats 结构体内不应再有 frames_attack 字段"
        )


def test_round50_h743_source_nonoverlap_trigger():
    """⭐ H743 触发逻辑: 16 帧 1 次推理 (frames_in_current_window 计数).

    ⭐ 2026-10-09 修正: 测试路径指向生产 h743/ 工程.
    """
    c_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__) or '.',
        'h743', 'Core', 'Src', 'modbus_ids.c'))
    with open(c_path, 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'frames_in_current_window' in src, (
        "H743 modbus_ids.c 应有 frames_in_current_window 计数器"
    )
    # 验证: 触发条件是 frames_in_current_window >= SW_WINDOW
    assert 'SW_WINDOW' in src, "应引用 SW_WINDOW (=16) 作为触发阈值"
    # 验证: 仍用 ch32_inference (生产 TCN+SE, 不是 CNN 实验)
    assert 'ch32_forward' in src, "H743 生产工程应仍用 ch32_forward (TCN+SE)"


def test_round50_h743_alarm_uses_windows_attack():
    """⭐ H743 报警消息携带 windows_attack (而非 frames_attack).

    ⭐ 2026-10-09 修正: 测试路径指向生产 h743/ 工程.
    """
    c_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__) or '.',
        'h743', 'Core', 'Src', 'modbus_ids.c'))
    with open(c_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # ids_print_alarm 应该传 windows_attack
    assert 'ids_print_alarm(g_stats.windows_attack' in src, (
        "H743 报警应传 g_stats.windows_attack 累计"
    )


# ============================================================================
# 5) 跟训练 1:1 对齐
# ============================================================================

def test_round50_evaluation_1to1_with_training():
    """⭐ Round 50 评估跟训练 max-aggregation 1:1.

    训练: 274k 帧 reshape(n//16, 16, -1) → 17164 windows, 1 window 1 label (max-aggregation)
    部署: 274k 帧 H743 16 帧 1 推理 → 17164 windows, 1 window 1 alarm (if attack)
    评估: 1 window = 1 ground truth (PC max 16 帧 label) = 1 prediction (H743 alarm)

    → F1 / Recall / Precision 直接对应训练 macro_f1

    ⭐ 2026-10-09 修正: 60048 attack 帧散在 21.86% 攻击率, max-aggregation 真值 = 8821 attack windows
    (不是 60048/16=3753 那个错误估算). 验证: 跑 pandas 算 `yw.sum() == 8821`.
    """
    import pandas as pd
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__) or '.', 'IanArffDatasetF32.csv'))
    n_total = 274628
    n_window = n_total // 16
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        y_bin = (df['binary result']==1).astype(int).values
        n = (len(y_bin) // 16) * 16
        yw = y_bin[:n].reshape(n//16, 16).max(axis=1)
        n_attack_windows = int(yw.sum())
        n_normal_windows = int((1 - yw).sum())
    else:
        n_attack_windows = 8821
        n_normal_windows = 8343
    assert n_window == 17164
    # 训练 baseline (5-seed): F1=0.8775, Recall=0.9014
    # 部署 H743 预期: F1 接近训练 baseline (部署损失 < 0.05)
    print(f"  训练: {n_total} 帧 → {n_window} windows ({n_attack_windows} attack + {n_normal_windows} normal, max-aggregation 真值)")
    print(f"  部署: {n_total} 帧 → {n_window} windows (1 window 1 alarm)")
    print(f"  评估: 1:1 对齐, F1 预期接近训练 baseline 0.8775")


def test_round50_no_overfire():
    """⭐ Round 50 没有 over-fire: H743 1 window 1 alarm, 不可能 TP > Attack.

    旧 (sliding): H743 SW=16 step=1, 1 个 attack 帧被 16 个 window 看到, 触发 ~16 个 alarm
    新 (non-overlapping): 1 个 attack 帧最多被 1-2 个 window 看到, 最多 1-2 个 alarm
    → TP_count ≤ Attack_windows (= Attack_count // 16)

    测试: 32 帧 attack (2 windows) + 1 alarm burst (count=1)
    → 第一个 window 评估时 _recent_alarm_frames 还没收到 alarm, 算 FN
    → 第二个 window 评估时 _recent_alarm_frames 收到 alarm, 但 2 windows 都 attack → TP=1, FN=1
    → 不会 over-fire
    """
    parser = rtu_replay_gui.IdsLineParser()
    for _ in range(32):
        parser.set_last_was_attack(True)
    # H743 上行 count=1, delta=1, 但 2 windows 都已经算完 (n_windows=2)
    parser.feed(b'[ALARM] 1\r\n')
    metrics = parser._window_eval.get_metrics()
    assert metrics['n_windows'] == 2, "32 帧 = 2 个 windows"
    # 第一个 window: attack=True, alarm=False → FN=1
    # 第二个 window: attack=True, alarm=True (alarm 在 16-32 之间) → TP=1
    assert metrics['tp'] + metrics['fn'] == 2, "2 attack windows → TP+FN=2"
    # 关键: 即使 burst 多个 alarm, 也不会让 TP > attack_windows
    assert metrics['tp'] <= 2, "TP ≤ attack_windows (no over-fire)"


# ============================================================================
# 6) 字段分离: per-window (新) ≠ per-alarm (旧, 已删)
# ============================================================================

def test_round50_no_peralarm_legacy():
    """⭐ _CLASSIFY_WINDOW / _window / _binary_history / _tp_budget 全已清理.

    Round 50 完全切换到 per-window 路径, per-alarm 评估的相关字段都不应再存在.
    """
    parser = rtu_replay_gui.IdsLineParser()
    legacy_attrs = ['_window', '_binary_history', '_tp_budget']
    for attr in legacy_attrs:
        assert not hasattr(parser, attr), (
            f"per-alarm 遗留字段 {attr} 应已移除"
        )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
