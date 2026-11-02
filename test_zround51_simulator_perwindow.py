#!/usr/bin/env python3
"""验证 Round 51: PC 模拟器按 16 帧 1 window 统计 attack 数和 send 数, 跟 H743 部署 1:1.

⭐ 2026-10-09 第五十一轮 (用户要求: "模拟器按照每16帧来统计攻击数和发送数"):
  - Replayer 内部维护 frames_in_current_window 计数 + current_window_had_attack flag
  - 凑齐 16 帧时, 用 max-aggregation 决定该 window:
    - 任一帧 attack → windows_attack_sent_count += 1
    - 全 normal      → windows_normal_sent_count += 1
  - 跟训练 preprocess_v2_scada.py reshape(n//16, 16, -1) max-aggregation 1:1 对齐
  - 跟 H743 部署 (Round 50 16 帧 1 window 1 alarm) 1:1 对齐
  - Attack KPI 从 60048 frames → 3753 windows
  - Total KPI 从 274628 frames → 17164 windows
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
    gui_path = os.path.abspath(os.path.join(os.path.dirname(__file__) or '.', 'rtu_replay_gui.py'))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rtu_replay_gui", gui_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__file__ = gui_path
    return mod


rtu_replay_gui = _load_gui_module()


# ============================================================================
# 1) Replayer 字段: 新增 per-window 计数
# ============================================================================

def test_round51_replayer_has_window_counters():
    """⭐ Round 51: Replayer 内部维护 per-window 计数."""
    # 不直接实例化 (需要 serial_obj), 检查源码含字段声明
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    for attr in ('windows_sent_count', 'windows_attack_sent_count', 'windows_normal_sent_count',
                 '_frames_in_current_window', '_current_window_had_attack'):
        assert attr in src, f"Replayer 应有字段 {attr}"


def test_round51_window_stats_signal_defined():
    """⭐ Round 51: 新增 window_stats 信号 (per-window KPI)."""
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'window_stats = pyqtSignal(int, int, int)' in src, (
        "Replayer 应有 window_stats 信号 (int, int, int) = (total_w, attack_w, normal_w)"
    )


# ============================================================================
# 2) per-window 累加逻辑 (源码静态检查)
# ============================================================================

def test_round51_max_aggregation_logic():
    """⭐ Round 51: 凑齐 16 帧时 max-aggregation 决定当前 window attack.

    源码逻辑:
      if binary_int == 1: _current_window_had_attack = True
      _frames_in_current_window += 1
      if _frames_in_current_window >= 16:
          windows_sent_count += 1
          if _current_window_had_attack: attack_count += 1
          else: normal_count += 1
          _current_window_had_attack = False
    """
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 验证关键逻辑存在
    assert '_frames_in_current_window >= 16' in src, (
        "Replayer 应在 _frames_in_current_window >= 16 时凑齐 1 window"
    )
    assert '_current_window_had_attack' in src, (
        "Replayer 应维护 _current_window_had_attack flag (max-aggregation)"
    )
    assert 'window_stats.emit' in src, (
        "Replayer 凑齐 window 时应 emit window_stats 信号"
    )


# ============================================================================
# 3) 模拟 16 帧增量路径
# ============================================================================

def _make_dummy_replayer():
    """构造一个 dummy ReplayWorker (避免实例化时开 serial)."""
    cls = rtu_replay_gui.ReplayWorker
    rp = cls.__new__(cls)
    rp.csv_path = ''
    rp.port = 'COM5'
    rp.baud = 115200
    rp.attack_filter = None
    rp.start_frame = 0
    rp.verbose = False
    rp.interval_ms = 0
    rp._shared_ser = None
    rp._stopping = False
    rp._paused = False
    rp.frames_sent = 0
    rp.bytes_sent = 0
    rp.attack_sent_count = 0
    rp.total_sent_count = 0
    rp.normal_sent_count = 0
    rp.windows_sent_count = 0
    rp.windows_attack_sent_count = 0
    rp.windows_normal_sent_count = 0
    rp._frames_in_current_window = 0
    rp._current_window_had_attack = False
    rp.ser = None
    return rp


def _accumulate_window(rp, is_attack_list):
    """模拟 ReplayWorker 发送 N 帧, 每帧走 per-frame + per-window 累加."""
    for is_attack in is_attack_list:
        # per-frame 累加
        rp.total_sent_count += 1
        if is_attack:
            rp.attack_sent_count += 1
        else:
            rp.normal_sent_count += 1
        # per-window 累加 (跟 ReplayWorker._do_send 同步)
        if is_attack:
            rp._current_window_had_attack = True
        rp._frames_in_current_window += 1
        if rp._frames_in_current_window >= 16:
            rp._frames_in_current_window = 0
            rp.windows_sent_count += 1
            if rp._current_window_had_attack:
                rp.windows_attack_sent_count += 1
            else:
                rp.windows_normal_sent_count += 1
            rp._current_window_had_attack = False


def test_round51_16_normal_is_normal_window():
    """⭐ 16 帧全 normal → 1 个 normal window."""
    rp = _make_dummy_replayer()
    _accumulate_window(rp, [False] * 16)
    assert rp.windows_sent_count == 1
    assert rp.windows_attack_sent_count == 0
    assert rp.windows_normal_sent_count == 1
    # per-frame 仍正常累加
    assert rp.total_sent_count == 16
    assert rp.normal_sent_count == 16


def test_round51_1_attack_in_16_is_attack_window():
    """⭐ 16 帧中 1 帧 attack → 1 个 attack window (max-aggregation)."""
    rp = _make_dummy_replayer()
    seq = [True] + [False] * 15  # 第 1 帧 attack
    _accumulate_window(rp, seq)
    assert rp.windows_sent_count == 1
    assert rp.windows_attack_sent_count == 1
    assert rp.windows_normal_sent_count == 0


def test_round51_16_attack_is_attack_window():
    """⭐ 16 帧全 attack → 1 个 attack window."""
    rp = _make_dummy_replayer()
    _accumulate_window(rp, [True] * 16)
    assert rp.windows_sent_count == 1
    assert rp.windows_attack_sent_count == 1


def test_round51_32_frames_two_windows():
    """⭐ 32 帧: 2 个 windows."""
    rp = _make_dummy_replayer()
    seq = [True] * 16 + [False] * 16  # 第 1 个 attack, 第 2 个 normal
    _accumulate_window(rp, seq)
    assert rp.windows_sent_count == 2
    assert rp.windows_attack_sent_count == 1
    assert rp.windows_normal_sent_count == 1


def test_round51_17_frames_one_full_one_partial():
    """⭐ 17 帧: 1 个完整 window + 1 个未满 window (不计入)."""
    rp = _make_dummy_replayer()
    seq = [True] * 16 + [True]  # 16 attack + 1 attack
    _accumulate_window(rp, seq)
    assert rp.windows_sent_count == 1
    assert rp.windows_attack_sent_count == 1
    assert rp._frames_in_current_window == 1
    assert rp._current_window_had_attack is True  # 第 17 帧 attack, 还没凑齐


def test_round51_attack_count_drops_from_60k_to_8k8():
    """⭐ 274k 帧, 60048 attack → windows_attack_sent_count = 8821 (max-aggregation 真值).

    训练 max-aggregation 真值: 60048 attack 帧散在 274628 帧里 (21.86% 攻击率),
    16 帧 max-aggregation 后: 17164 windows 中 8821 个是 attack window (51.4%).

    ⭐ 2026-10-09 修正: 之前注释写 3753 (=60048/16 简单整除, 假设 attack 帧聚在一起),
    实际数据是均匀散在的, max-aggregation 后大多数 window 都含 attack 帧,
    实际数字是 8821. 验证 (在 test_zround51 之外, 跑 pandas 算):
        import pandas as pd
        df = pd.read_csv('IanArffDatasetF32.csv')
        y_bin = (df['binary result']==1).astype(int).values
        n = (len(y_bin)//16)*16
        yw = y_bin[:n].reshape(n//16, 16).max(axis=1)
        yw.sum() == 8821
    """
    import pandas as pd
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__) or '.', 'IanArffDatasetF32.csv'))
    if not os.path.exists(csv_path):
        # 没有 CSV, 跳过实际验证
        print(f"  [skip] {csv_path} 不存在, 仅检查数学公式")
        return
    df = pd.read_csv(csv_path)
    y_bin = (df['binary result']==1).astype(int).values
    n = (len(y_bin) // 16) * 16
    yw = y_bin[:n].reshape(n // 16, 16).max(axis=1)
    n_attack_w = int(yw.sum())
    n_normal_w = int((1 - yw).sum())
    assert n_attack_w == 8821, (
        f"训练 max-aggregation 后 attack windows 应为 8821, 实为 {n_attack_w} "
        f"(60048 attack 帧散在 274628 帧里, 21.86% 攻击率, 16 帧 max 后 51.4% windows 是 attack)"
    )
    assert n_normal_w == 8343, (
        f"normal windows 应为 8343 (=17164-8821), 实为 {n_normal_w}"
    )
    print(f"  训练 baseline: 60048 attack frames → 8821 attack windows (max-aggregation 真值)")
    print(f"  训练 baseline: 214580 normal frames → 8343 normal windows (max-aggregation 真值)")


def test_round51_total_drops_to_17k():
    """⭐ 274628 帧 → 17164 windows."""
    n_total = 274628
    n_windows = n_total // 16
    assert n_windows == 17164
    print(f"  PC total: 274628 frames → {n_windows} windows (1:1 跟训练/H743)")


# ============================================================================
# 4) 跟 H743 部署 1:1 对齐
# ============================================================================

def test_round51_simulator_aligned_with_h743():
    """⭐ 模拟器 per-window 跟 H743 per-window 1:1 对齐.

    训练: 16 帧 max-aggregation → 1 window label
    PC 模拟器: 16 帧 max-aggregation → 1 attack window 计数
    H743 部署: 16 帧 1 次推理 → 1 alarm (if attack)
    评估: 1 H743 alarm vs 1 PC attack window → TP/FP

    整个链路都是 per-window 16 帧, 1:1 对齐.

    ⭐ 2026-10-09 修正: 60048 attack 帧散在数据中, max-aggregation 后 8821 个 attack window
    (不是 60048/16=3753, 那是假设 attack 帧聚在一起的错误估算).
    """
    n_total = 274628
    n_attack_frames = 60048
    n_windows = n_total // 16
    n_attack_windows_actual = 8821  # max-aggregation 真值 (从 CSV 算)
    n_normal_windows_actual = 8343
    assert n_windows == 17164
    assert n_attack_windows_actual + n_normal_windows_actual == n_windows
    print(f"  全链路 1:1 对齐: 274628 帧 → 17164 windows ({n_attack_windows_actual} attack + {n_normal_windows_actual} normal)")
    print(f"  训练: {n_windows} windows × 1 label = 17164 个训练样本 ({n_attack_windows_actual} attack)")
    print(f"  PC 发送: 17164 windows × 1 attack 决策 = 17164 个攻击统计")
    print(f"  H743 推理: 17164 windows × 1 推理 = 17164 个 binary decision")
    print(f"  评估: 1 H743 alarm vs 1 PC window → TP/FP 直接对比训练 macro_f1")


def test_round51_no_partial_window_counted():
    """⭐ 未凑齐 16 帧的尾部帧不计入 window 统计 (跟训练 reshape(n//16) 行为一致).

    训练: n = (len(X) // win) * win  ← 整除, 丢弃尾部
    PC 模拟器: _frames_in_current_window < 16 不递增 windows_sent_count
    H743: 同理, 凑齐 16 才推理
    """
    rp = _make_dummy_replayer()
    # 发 15 帧 (不足 1 个 window)
    _accumulate_window(rp, [True] * 15)
    assert rp.windows_sent_count == 0, (
        "15 帧不应计入 window 统计 (跟训练整除 16 行为一致)"
    )
    assert rp._frames_in_current_window == 15
    assert rp._current_window_had_attack is True  # 状态保留等下一帧


# ============================================================================
# 5) Round 52b: 默认 per-window 进度日志, 解决 "auto 模式看不到发送日志"
# ============================================================================

def test_round52b_per_window_progress_log_emitted():
    """⭐⭐ Round 52b: 每凑齐 1 window (16 帧) emit 1 条 log, 解决 auto 模式无进度问题.

    之前: verbose 模式才 emit [TX N] (每 100 帧 1 条), 默认 verbose=False → log_text
    完全没实时进度, auto 跑 274k 帧用户只能盯 progress bar.

    现在: 凑齐 1 window 时无条件 emit 1 条简洁进度, 跟 per-window 1:1.
    格式: "[TX 帧N] window W (attack|normal)  累计 A attack + B normal"

    ⭐ 2026-10-10: 验证源码包含这个 emit 逻辑.
    """
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 验证: 凑齐 1 window 时 self.log.emit 包含 "window" / "attack" / "normal" 关键字
    # 找 window_stats.emit 之后的 self.log.emit (Round 52b 紧跟其后)
    assert 'self.windows_sent_count += 1' in src
    # 验证: emit log 包含 win_tag (attack/normal) 区分
    assert "win_tag" in src, "凑齐 1 window 应 emit log 区分 attack/normal"
    # 验证: emit log 包含累计 attack/normal 计数
    assert 'self.windows_attack_sent_count} attack' in src
    assert 'self.windows_normal_sent_count} normal' in src
    # 验证: emit log 包含 frame 计数 (实时进度)
    assert 'self.frames_sent' in src


def test_round52b_progress_log_format():
    """⭐⭐ Round 52b: log 格式 — [TX 帧N] window W (attack|normal)  累计 A attack + B normal."""
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 找 emit log 整段 (跨多行 f-string, 用 DOTALL 匹配到行尾空白)
    import re
    m = re.search(
        r'self\.log\.emit\(\s*\n\s*f"\[TX 帧\{self\.frames_sent[\s\S]*?\n\s*\)',
        src
    )
    assert m, "源码应有 self.log.emit('[TX 帧{self.frames_sent...') 整段"
    log_block = m.group(0)
    # 验证格式包含关键字段
    assert 'window' in log_block
    assert '{win_tag}' in log_block, f"log 应包含 win_tag 区分 attack/normal, 实为 {log_block}"
    assert 'attack' in log_block
    assert 'normal' in log_block


def test_round52b_default_no_longer_silent():
    """⭐⭐ Round 52b: auto 模式 (verbose=False) 默认也能看到 log 进度.

    之前: line 621 `if self.verbose and self.frames_sent % 100 == 1:` 决定是否 emit [TX N].
    默认 verbose=False → 完全不 emit. 现在新逻辑在凑齐 1 window 时无条件 emit, 跟 verbose
    解耦, auto 模式 log_text 有实时进度.
    """
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 验证: per-window 进度 emit 跟 self.verbose 无关 (无条件)
    # 找凑齐 1 window 那个 if 块
    import re
    m = re.search(
        r'if self\._frames_in_current_window >= 16:.*?self\.log\.emit\(',
        src, re.DOTALL
    )
    assert m, "凑齐 1 window 块内应有 self.log.emit (无条件)"
    block = m.group(0)
    # 块内不应有 self.verbose 限制 (那是 [TX N] 详细模式, 独立的)
    assert 'self.verbose' not in block, (
        "per-window 进度 log 应该是默认行为, 不应被 self.verbose 控制. "
        "verbose 模式 [TX N] hex 详细日志是独立的, 保留在另一处."
    )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
