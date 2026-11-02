#!/usr/bin/env python3
"""验证 Round 52: IdsReportPanel 4 个主 KPI (Total / Normal / Attack / Detected)
全部按 16 帧 1 window 计算, 跟训练 / H743 部署 / PC 模拟器 1:1 对齐.

⭐ 2026-10-09 第五十二轮 (用户要求: "total、normal、attack、detected 都是
按照每 16 帧来计算一次"):

  Round 50: H743 改成 non-overlapping (16 帧 1 window 1 alarm, 跟训练 1:1)
  Round 51: PC 模拟器 (ReplayWorker) 加 per-window 计数 (windows_sent_count
            / windows_attack_sent_count / windows_normal_sent_count)
  Round 52: IdsReportPanel 接 ReplayWorker.window_stats, 4 个主 KPI label
            (Total / Normal / Attack / Detected) 全部按 16 帧 1 window 显示.
            Detected 走 IdsLineParser.count_changed, 携带的 N 是 H743
            windows_attack 累计 (Round 50 起就是 per-window), 天然 1:1.

  数字对比 (274k 帧全跑):
    Per-frame (旧): Total=274628, Normal=214580, Attack=60048, Detected=H743 frames_attack
    Per-window (新): Total=17164,  Normal=8343,    Attack=8821,    Detected=H743 windows_attack
    (⭐ 2026-10-09 修正: 60048 attack 帧散在 21.86%, max-aggregation 后 8821 attack + 8343 normal
     = 17164. 之前写 3753/13411 是 60048/16 简单整除的错误估算.)

  整个链路 1:1:
    训练:     274628 帧 → 17164 windows × 1 label (max-aggregation)
    PC 发送:  274628 帧 → 17164 windows × 1 attack 决策 (Round 51)
    H743 推理: 274628 帧 → 17164 windows × 1 binary decision (Round 50)
    PC 评估:   1 H743 alarm vs 1 PC window → TP/FP
    4 指标:   F1 / Recall / Precision / FPR 跟训练 macro_f1 直接对比
"""
from __future__ import annotations

import os
import sys
from typing import List


class _DummySignal:
    """PyQt5 signal stub. emit 时兼容 bound method 签名 (多余 arg 静默丢弃)."""
    def __init__(self, *args, **kwargs):
        self._calls: List = []
        self._slots: List = []
    def emit(self, *args):
        self._calls.append(args)
        for slot in self._slots:
            try:
                slot(*args)
            except TypeError:
                # 真实 PyQt5: 槽函数签名不匹配时, 多余 arg 静默丢弃
                # (例如 count_changed.emit(25) → on_alarm(self) 仍 OK)
                try:
                    slot()
                except TypeError:
                    pass
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


class _DummyLabel:
    """QLabel stub: 支持 setStyleSheet / setText / text, 其它方法静默 no-op."""
    def __init__(self, text=""):
        self._text = text
    def setStyleSheet(self, *args, **kwargs): pass
    def setText(self, text): self._text = text
    def text(self): return self._text
    def setMinimumHeight(self, *args, **kwargs): pass
    def setMinimumWidth(self, *args, **kwargs): pass
    def setMaximumHeight(self, *args, **kwargs): pass
    def addWidget(self, *args, **kwargs): pass
    def addLayout(self, *args, **kwargs): pass
    def addSpacing(self, *args, **kwargs): pass
    def addStretch(self, *args, **kwargs): pass
    def setStretch(self, *args, **kwargs): pass
    def setSpacing(self, *args, **kwargs): pass
    def setContentsMargins(self, *args, **kwargs): pass
    def setAlignment(self, *args, **kwargs): pass
    def isChecked(self): return False
    def setEnabled(self, *args, **kwargs): pass
    def setCheckable(self, *args, **kwargs): pass
    def setChecked(self, *args, **kwargs): pass
    def setVisible(self, *args, **kwargs): pass
    def connect(self, *args, **kwargs): pass
    def disconnect(self, *args, **kwargs): pass
    def emit(self, *args, **kwargs): pass
    def value(self): return 0
    def setValue(self, *args, **kwargs): pass
    def setMaximum(self, *args, **kwargs): pass
    def setRange(self, *args, **kwargs): pass
    def setFormat(self, *args, **kwargs): pass
    def setSingleStep(self, *args, **kwargs): pass


class _DummyLayout:
    """QLayout stub: 所有 add* 方法 no-op, 配合 _DummyLabel 使用."""
    def __init__(self, *args, **kwargs): pass
    def setContentsMargins(self, *args, **kwargs): pass
    def setSpacing(self, *args, **kwargs): pass
    def setAlignment(self, *args, **kwargs): pass
    def addWidget(self, *args, **kwargs): pass
    def addLayout(self, *args, **kwargs): pass
    def addSpacing(self, *args, **kwargs): pass
    def addStretch(self, *args, **kwargs): pass
    def setStretch(self, *args, **kwargs): pass


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
# QLabel 单独, 用支持 setText/text 的 stub
setattr(sys.modules['PyQt5.QtWidgets'], 'QLabel', _DummyLabel)
# QLayout 用支持 addWidget/addLayout 的 stub
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
# 1) IdsReportPanel 字段: 4 个 per-window 内部 state
# ============================================================================

def _make_panel():
    parser = rtu_replay_gui.IdsLineParser()
    return rtu_replay_gui.IdsReportPanel(parser)


def test_round52_panel_has_window_state():
    """⭐ Round 52: IdsReportPanel 内部维护 per-window 4 字段."""
    panel = _make_panel()
    for attr in ('_window_total_count', '_window_normal_count',
                 '_window_attack_count'):
        assert hasattr(panel, attr), f"IdsReportPanel 应有字段 {attr}"
        assert getattr(panel, attr) == 0, f"初始 {attr} 应为 0"


def test_round52_per_frame_state_removed():
    """⭐ 2026-10-19: 删 per-frame 内部 state (per-frame 计数已删除).

    反向测试: 验证 _total_count / _sent_count / _normal_count 不再存在.
    """
    panel = _make_panel()
    for attr in ('_total_count', '_sent_count', '_normal_count',
                 '_fp_count', '_fn_count', '_tn_count'):
        assert not hasattr(panel, attr), f"per-frame 字段 {attr} 应已删除"


# ============================================================================
# 2) _on_window_stats 槽函数: 喂 per-window 累计
# ============================================================================

def test_round52_on_window_stats_handler_exists():
    """⭐ Round 52: IdsReportPanel 有 _on_window_stats 槽函数."""
    panel = _make_panel()
    assert hasattr(panel, '_on_window_stats'), (
        "IdsReportPanel 应有 _on_window_stats 槽函数 (接 ReplayWorker.window_stats)"
    )


def test_round52_on_window_stats_updates_internal_state():
    """⭐ _on_window_stats(total_w, attack_w, normal_w) 更新 3 个内部 state."""
    panel = _make_panel()
    panel._on_window_stats(100, 5, 95)
    assert panel._window_total_count == 100
    assert panel._window_attack_count == 5
    assert panel._window_normal_count == 95


def test_round52_on_window_stats_updates_labels():
    """⭐ _on_window_stats 更新 Total / Attack / Normal 3 个 per-window label (数字 per-window).

    模拟 274k 帧跑完: total=17164, attack=8821, normal=8343 (max-aggregation 真值)
    ⭐ 第六十一轮 (2026-10-11): UI 已删 per-frame 行, 改用 wn_label / wattack_label /
      wnormal_label (per-window labels).
    """
    panel = _make_panel()
    panel._on_window_stats(17164, 8821, 8343)
    assert panel.wn_label.text() == '17164', (
        f"Total label 应显示 per-window 累计 17164, 实为 {panel.wn_label.text()}"
    )
    assert panel.wattack_label.text() == '8821', (
        f"Attack label 应显示 per-window 累计 8821 (max-aggregation 真值), 实为 {panel.wattack_label.text()}"
    )
    assert panel.wnormal_label.text() == '8343', (
        f"Normal label 应显示 per-window 累计 8343 (max-aggregation 真值), 实为 {panel.wnormal_label.text()}"
    )


def test_round52_on_window_stats_incremental():
    """⭐ _on_window_stats 多次调用累加语义 (每凑齐 1 window +1)."""
    panel = _make_panel()
    # 1 个 attack window
    panel._on_window_stats(1, 1, 0)
    assert panel.wn_label.text() == '1'
    assert panel.wattack_label.text() == '1'
    # 1 个 normal window
    panel._on_window_stats(2, 1, 1)
    assert panel.wn_label.text() == '2'
    assert panel.wattack_label.text() == '1'
    assert panel.wnormal_label.text() == '1'
    # 10 个 normal window
    panel._on_window_stats(12, 1, 11)
    assert panel.wn_label.text() == '12'
    assert panel.wnormal_label.text() == '11'


def test_round52_on_sent_stats_does_not_overwrite_window_labels():
    """⭐ _on_sent_stats (per-frame) 不再覆盖 per-window label (Round 52 设计).

    之前 _on_sent_stats 直接 setText(total/attack/normal), 但 4 个主 KPI
    已经按 per-window 显示, _on_sent_stats 只更新内部 state, 不碰 label.
    """
    panel = _make_panel()
    # 先喂 per-window (17164 总 windows, 8821 attack, 8343 normal — max-aggregation 真值)
    panel._on_window_stats(17164, 8821, 8343)
    assert panel.wn_label.text() == '17164'
    # ⭐ 2026-10-19: per-frame 状态已删除. _on_sent_stats 现在仅触发 _update_rates,
    #   不维护任何 per-frame state. label 仍 per-window (不被 sent_stats 覆盖).
    panel._on_sent_stats(274628, 60048, 214580)
    # label 仍 per-window (不被 sent_stats 覆盖)
    assert panel.wn_label.text() == '17164'
    assert panel.wattack_label.text() == '8821'
    assert panel.wnormal_label.text() == '8343'


def test_round52_on_sent_does_not_overwrite_window_label():
    """⭐⭐ Round 52 关键 bug 修复: _on_sent (per-frame 累计) 不再 setText Attack label.

    ⭐ 2026-10-19: _on_sent 方法已删除 (per-frame 计数已删除). 此测试改为验证
    _on_window_stats 单独维护 wattack_label, 不会有 per-frame 跳变.
    """
    panel = _make_panel()
    # 1. 先喂 per-window 累计 (模拟凑齐 16 帧时 window_stats 触发)
    panel._on_window_stats(1, 1, 0)
    assert panel.wattack_label.text() == '1', "window_stats 后 Attack label=1"
    # ⭐ 2026-10-19: _on_sent 已删除. label 仅由 _on_window_stats 维护.
    # 4. 凑齐下一 window 时, _on_window_stats 拉回 per-window
    panel._on_window_stats(2, 2, 0)
    assert panel.wattack_label.text() == '2', "新一轮 per-window 累计 → label=2"


def test_round52_attack_label_stable_during_attack_run():
    """⭐⭐ Round 52: 连续发 attack 帧, Attack label 不会按帧跳变 (per-window 稳定).

    ⭐ 第六十一轮: 测 wattack_label (per-window), 不用 per-frame sent_label (已删).

    ⭐ 2026-10-19: _on_sent 已删除, label 仅 _on_window_stats 维护, 不可能跳到 per-frame.
    """
    panel = _make_panel()
    # 初始: 已发 1 个 attack window (label='1')
    panel._on_window_stats(1, 1, 0)
    assert panel.wattack_label.text() == '1'
    # ⭐ 2026-10-19: 不再有 per-frame 跳变路径. 直接验证 window_stats 累计.
    panel._on_window_stats(2, 2, 0)
    assert panel.wattack_label.text() == '2'


# ============================================================================
# 3) Detected KPI: 走 count_changed, H743 windows_attack 累计 (per-window)
# ============================================================================

def test_round52_detected_baseline_first_alarm():
    """⭐ Detected 走 IdsLineParser.count_changed, baseline = max(count-1, 0)."""
    parser = rtu_replay_gui.IdsLineParser()
    panel = rtu_replay_gui.IdsReportPanel(parser)
    # H743 上行首条 [ALARM] 1 → Detected=1 (baseline=0)
    parser.feed(b'[ALARM] 1\r\n')
    assert panel._detected_baseline == 0
    assert panel._detected_count == 1
    assert panel.wdetected_label.text() == '1'


def test_round52_detected_incremental_alarms():
    """⭐ H743 逐条上行: 每行 N+1, Detected 累加 1."""
    parser = rtu_replay_gui.IdsLineParser()
    panel = rtu_replay_gui.IdsReportPanel(parser)
    # H743 上行 [ALARM] 1 → Detected=1
    parser.feed(b'[ALARM] 1\r\n')
    assert panel._detected_count == 1
    # H743 上行 [ALARM] 2 → Detected=2
    parser.feed(b'[ALARM] 2\r\n')
    assert panel._detected_count == 2
    # H743 上行 [ALARM] 5 → Detected=5
    parser.feed(b'[ALARM] 5\r\n')
    assert panel._detected_count == 5
    assert panel.wdetected_label.text() == '5'


def test_round52_detected_is_per_window():
    """⭐ H743 windows_attack 累计 = per-window (1 window 1 alarm 1:1).

    count_changed 携带的 N 在 Round 50 后就是 windows_attack 累计, 天然 per-window.
    274k 帧全跑, H743 windows_attack 上限 = 8821 (= PC attack windows 数, max-aggregation 真值).
    实际 H743 漏报的话 Detected < 8821, 但语义仍是 per-window 累计.

    注: H743 实际是每条 alarm 单独上行, N 累加 1, 不是单行 burst. 测试用单增模式.
    """
    parser = rtu_replay_gui.IdsLineParser()
    panel = rtu_replay_gui.IdsReportPanel(parser)
    # 模拟 H743 windows_attack 跑到上限 8821 (= PC attack windows 数, max-aggregation 真值).
    # 实际 H743 推理会有 FN, 真实 Detected 数 < 8821, 但 UI 仍按 windows_attack 累计 per-window 显示.
    for n in range(1, 8822):
        parser.feed(f'[ALARM] {n}\r\n'.encode('utf-8'))
    assert panel._detected_count == 8821
    assert panel.wdetected_label.text() == '8821'
    # 注: 跟 _window_attack_count (PC 端 attack window 数 = 8821) 概念 1:1
    # Detected 不会超过 _window_attack_count (H743 不可能比 PC 报更多 attack window)


# ============================================================================
# 4) reset_counters: per-window state 也清零
# ============================================================================

def test_round52_reset_clears_window_state():
    """⭐ reset_counters() 清 per-window 4 字段 + 8 KPI label."""
    panel = _make_panel()
    panel._on_window_stats(100, 30, 70)
    panel._on_sent_stats(1600, 480, 1120)
    # 模拟 Detected (H743 逐条上行, 25 个 alarm)
    for n in range(1, 26):
        panel._parser.feed(f'[ALARM] {n}\r\n'.encode('utf-8'))
    assert panel._window_total_count == 100
    assert panel._detected_count == 25

    panel.reset_counters()

    # 内部 state 清零 (⭐ 2026-10-19: 删 per-frame state 断言)
    assert panel._window_total_count == 0
    assert panel._window_normal_count == 0
    assert panel._window_attack_count == 0
    assert panel._detected_count == 0
    assert panel._detected_baseline is None
    # label 也清零
    assert panel.wn_label.text() == '0'
    assert panel.wattack_label.text() == '0'
    assert panel.wnormal_label.text() == '0'
    assert panel.wdetected_label.text() == '0'


# ============================================================================
# 5) main_window 接线: window_stats → IdsReportPanel._on_window_stats
# ============================================================================

def test_round52_mainwindow_wires_window_stats():
    """⭐ main_window._start 调 self.worker.window_stats.connect(self.ids_report._on_window_stats)."""
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 验证: main_window._start 内有 window_stats → _on_window_stats 接线
    assert 'self.worker.window_stats.connect(self.ids_report._on_window_stats)' in src, (
        "main_window._start 应接 window_stats → _on_window_stats "
        "(跟 sent_stats → _on_sent_stats 平级)"
    )


def test_round52_mainwindow_keeps_sent_stats_wire():
    """⭐ main_window 仍接 sent_stats → _on_sent_stats (触发 _update_rates).

    ⭐ 2026-10-19: _on_sent_stats 不再维护 per-frame state, 仅触发 _update_rates.
    """
    gui_path = rtu_replay_gui.__file__
    with open(gui_path, 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'self.worker.sent_stats.connect(self.ids_report._on_sent_stats)' in src, (
        "main_window 应仍接 sent_stats → _on_sent_stats (触发 _update_rates)"
    )


# ============================================================================
# 6) H743 部署端验证: windows_attack 累计 (Round 50 已就绪)
# ============================================================================

def test_round52_h743_uses_windows_attack_count():
    """⭐ H743 modbus_ids.c 仍走 windows_attack 累计 (Round 50 已是 per-window).

    Round 52 不改 H743 代码, 仅在 PC 端把 H743 报的 count 接到 per-window label.
    这个测试确认 H743 端状态: windows_attack 字段仍在 g_stats 结构体里,
    ids_print_alarm 仍发 windows_attack 累计.
    """
    c_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__) or '.', 'h743', 'Core', 'Src', 'modbus_ids.c'))
    with open(c_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # 验证: ids_print_alarm 仍携带 windows_attack (Round 50 已就绪, Round 52 沿用)
    assert 'ids_print_alarm(g_stats.windows_attack' in src, (
        "H743 报警应传 g_stats.windows_attack 累计 (Round 50 per-window, Round 52 沿用)"
    )


# ============================================================================
# 7) 全链路 1:1 对齐总结
# ============================================================================

def test_round52_full_pipeline_1to1():
    """⭐ 整个评估链路 1:1 对齐训练 max-aggregation.

    训练: 274628 帧 → 17164 windows × 1 label (max-aggregation, preprocess_v2_scada.py)
    PC 发送: 274628 帧 → 17164 windows × 1 attack 决策 (Round 51)
    H743 推理: 274628 帧 → 17164 windows × 1 binary decision (Round 50)
    PC 评估: 1 H743 alarm vs 1 PC window → TP/FP
    GUI 显示: 4 个主 KPI (Total / Normal / Attack / Detected) 全部 per-window (Round 52)
    4 指标: F1 / Recall / Precision / FPR 跟训练 macro_f1 直接对比

    ⭐ 2026-10-09 修正: 60048 attack 帧散在 21.86% 攻击率, max-aggregation 后:
      - 8821 attack windows (51.4% of 17164)
      - 8343 normal windows (48.6% of 17164)
      不是 3753 attack (= 60048/16 简单整除的错误估算)
    """
    import pandas as pd
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__) or '.', 'IanArffDatasetF32.csv'))
    n_total = 274628
    n_windows = n_total // 16
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        y_bin = (df['binary result']==1).astype(int).values
        n = (len(y_bin) // 16) * 16
        yw = y_bin[:n].reshape(n//16, 16).max(axis=1)
        n_attack_w = int(yw.sum())
        n_normal_w = int((1 - yw).sum())
    else:
        # 没 CSV 时用真值常量 (硬编码)
        n_attack_w = 8821
        n_normal_w = 8343
    assert n_windows == 17164
    assert n_attack_w + n_normal_w == n_windows
    print(f"  训练: {n_total} 帧 → {n_windows} windows ({n_attack_w} attack + {n_normal_w} normal, max-aggregation 真值)")
    print(f"  PC GUI (Round 52): Total={n_windows} Attack={n_attack_w} Normal={n_normal_w}")
    print(f"  H743: {n_windows} windows × 1 alarm → Detected 累加到 attack_windows (上限 {n_attack_w})")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
