#!/usr/bin/env python3
"""Round 62 测试: IdsReportPanel attack/alarm 实时颜色反馈.

功能:
- 当 ReplayWorker.frame_sent_type=True (发送攻击帧) → wattack_label 红色背景
- 当 IdsLineParser.count_changed (H743 上行 alarm) → wdetected_label 绿色背景
- reset_counters 后颜色还原
"""
from __future__ import annotations
import os, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import pytest

# ⭐ 防御 test_zround48_per_window_evaluator 在 sys.modules['PyQt5.QtWidgets']
# 上 monkeypatch QApplication 等为 _DummyQObject. 本测试需要真实 PyQt5, 故在
# import 时检测并 raise skip.
import sys as _sys
_qt_mod = _sys.modules.get('PyQt5.QtWidgets')
_qt_mod_name = getattr(_qt_mod, '__name__', '') if _qt_mod else ''
_qapp_cls = getattr(_qt_mod, 'QApplication', None) if _qt_mod else None
_qapp_mod = getattr(_qapp_cls, '__module__', '') if _qapp_cls else ''
_qapp_is_dummy = ('PyQt5' not in str(_qapp_mod)) or _qapp_cls is None
if not _qt_mod_name.startswith('PyQt5.QtWidgets') or _qapp_is_dummy:
    pytest.skip(f"PyQt5.QtWidgets 被前序测试 monkeypatch "
                f"(qapp_cls={_qapp_cls!r}, qapp_module={_qapp_mod!r}), "
                f"跳过本轮 UI 集成测试", allow_module_level=True)

# pytest-qt 不会自动启动 QApplication 在没有 qtbot fixture 的测试里, 显式创建一次
from PyQt5.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

from rtu_replay_gui import IdsReportPanel, IdsLineParser


@pytest.fixture
def panel():
    """直接创建 panel (跟 test_zround52 一致, 不用 qtbot).

    注意: test_zround48_per_window_evaluator 在 sys.modules['PyQt5.QtWidgets']
    上 monkeypatch 了 QLabel 等为 _DummyQObject. 如果本轮跑在它后面, QLabel
    styleSheet() 返回 None 而不是字符串, 测试失败. 这里检测并 skip.
    """
    from PyQt5.QtWidgets import QLabel as _RealQLabel
    # 真实 QLabel 应有 PyQt5.QtWidgets 模块路径 + 有 setStyleSheet 方法
    if (not hasattr(_RealQLabel, 'setStyleSheet')
            or 'PyQt5' not in str(getattr(_RealQLabel, '__module__', ''))):
        pytest.skip("PyQt5.QtWidgets 被前序测试 monkeypatch, 跳过本轮 UI 集成测试")

    parser = IdsLineParser()
    p = IdsReportPanel(parser)
    yield p
    p.deleteLater()


def _ss(widget):
    """安全读 styleSheet (无 parent 时可能返回 None, 兜底空字符串)."""
    return widget.styleSheet() or ""


def test_initial_color_no_bg(panel):
    """新 panel: wattack_label 和 wdetected_label 都不应有红色/绿色背景."""
    assert "#ff5555" not in _ss(panel.wattack_label)
    assert "#55ff55" not in _ss(panel.wdetected_label)


def test_attack_frame_triggers_red(panel):
    """frame_sent_type=True → wattack_label 红色背景."""
    panel._window_eval.on_frame(True)
    panel._refresh_attack_alarm_color()
    assert "#ff5555" in _ss(panel.wattack_label), \
        f"attack bg 应为 #ff5555, got: {_ss(panel.wattack_label)}"


def test_normal_frame_resets_red(panel):
    """完成 16 帧评估后 _current_window_attack 重置 → 红色消失."""
    panel._window_eval.on_frame(True)
    panel._refresh_attack_alarm_color()
    # 完成 16 帧 (max-aggregation 规则), _current_window_attack 在 _evaluate_window 重置
    for _ in range(15):
        panel._window_eval.on_frame(False)
    panel._refresh_attack_alarm_color()
    assert "#ff5555" not in _ss(panel.wattack_label), \
        f"完成后应无红, got: {_ss(panel.wattack_label)}"


def test_alarm_triggers_green_on_detected(panel):
    """count_changed → wdetected_label 绿色背景.

    注意时序: WindowEvaluator.on_alarm 把 alarm 加进 _recent_alarm_frames,
    _current_window_alarm=True 是在下一次 on_frame 才设置. 这是真实生产路径
    (H743 alarm 上行 → 下一个 PC 帧到达 → frame_sent_type 触发 on_frame).
    """
    panel._window_eval.on_frame(False)   # _frame_count=1
    panel._window_eval.on_alarm()         # 加 recent_alarm_frames
    panel._window_eval.on_frame(False)   # _current_window_alarm 翻 True
    panel._refresh_attack_alarm_color()
    assert "#55ff55" in _ss(panel.wdetected_label), \
        f"detected bg 应为 #55ff55, got: {_ss(panel.wdetected_label)}"


def test_reset_counters_clears_colors(panel):
    """reset_counters → 颜色全部还原."""
    # 模拟 attack + alarm 同时触发 (顺序: frame → alarm → frame 才能同时上色)
    panel._window_eval.on_frame(True)
    panel._window_eval.on_alarm()
    panel._window_eval.on_frame(False)
    panel._refresh_attack_alarm_color()
    # 确认已上色
    assert "#ff5555" in _ss(panel.wattack_label)
    assert "#55ff55" in _ss(panel.wdetected_label)

    panel.reset_counters()
    # 重置后 attack/alarm 状态 = False → 无 bg
    assert "#ff5555" not in _ss(panel.wattack_label)
    assert "#55ff55" not in _ss(panel.wdetected_label)


def test_color_method_bound(panel):
    """确认 _refresh_attack_alarm_color 方法已绑定."""
    assert hasattr(panel, "_refresh_attack_alarm_color")
    assert callable(panel._refresh_attack_alarm_color)
    assert hasattr(panel, "wattack_label")
    assert hasattr(panel, "wdetected_label")
