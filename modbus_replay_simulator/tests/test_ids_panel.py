"""Tests for the IdsPanel widget.

Pure PyQt5 tests — no torch / numpy / IDS worker involved. The panel
just renders state and emits signals; we drive the public slots with
hand-crafted payloads and assert the UI updates as documented.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from modbus_replay.gui.widgets import IdsPanel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    a = QApplication.instance() or QApplication([])
    return a


def _wait_for(signal, timeout=2000):
    """Block until signal emitted, or timeout (ms). Returns list of arg-tuples."""
    loop = QEventLoop()
    captured: list[tuple] = []

    def slot(*args):
        captured.append(args)
        loop.quit()

    signal.connect(slot)
    QTimer.singleShot(timeout, loop.quit)
    loop.exec()
    signal.disconnect(slot)
    return captured


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


def test_ids_panel_default_state(qapp):
    panel = IdsPanel()
    # Checkbox unchecked → IDS not enabled.
    assert panel.is_enabled() is False
    # No files picked yet.
    assert panel.current_model_path() is None
    assert panel.current_scaler_path() is None
    assert panel.current_threshold_path() is None
    assert panel.current_csv_log_path() is None
    assert panel.csv_log_enabled() is False
    # KPI labels at zero.
    assert "0" in panel.scored_label.text()
    assert "0" in panel.alarm_label.text()
    # Table is empty.
    assert panel.alarm_table.rowCount() == 0
    # File pickers are disabled until the user ticks the enable box.
    assert panel.model_btn.isEnabled() is False


def test_ids_panel_enable_checkbox_emits_signal(qapp):
    panel = IdsPanel()
    emissions: list[bool] = []
    panel.enabled_changed.connect(emissions.append)
    panel.enable_chk.setChecked(True)
    panel.enable_chk.setChecked(False)
    assert emissions == [True, False]


def test_ids_panel_enable_checkbox_enables_pickers(qapp):
    panel = IdsPanel()
    assert panel.model_btn.isEnabled() is False
    panel.enable_chk.setChecked(True)
    assert panel.model_btn.isEnabled() is True
    assert panel.threshold_spin.isEnabled() is True
    assert panel.csv_chk.isEnabled() is True


# ---------------------------------------------------------------------------
# KPI updates
# ---------------------------------------------------------------------------


def test_ids_panel_kpi_updates_on_frame_scored(qapp):
    panel = IdsPanel()
    panel.on_frame_scored(42, 0.91, True, 4, 3, 1, 21000)
    assert "1" in panel.scored_label.text()
    assert "1" in panel.alarm_label.text()
    assert "100.00%" in panel.rate_label.text()


def test_ids_panel_kpi_alarm_rate_under_threshold(qapp):
    panel = IdsPanel()
    # 3 frames, only 1 is alarm → 33.33%.
    panel.on_frame_scored(1, 0.10, False, 4, 3, 0, 100)
    panel.on_frame_scored(2, 0.20, False, 4, 3, 0, 200)
    panel.on_frame_scored(3, 0.95, True, 4, 3, 0, 300)
    assert "3" in panel.scored_label.text()
    assert "1" in panel.alarm_label.text()
    assert "33.33%" in panel.rate_label.text()


def test_ids_panel_alarm_log_only_keeps_alarms(qapp):
    panel = IdsPanel()
    for i, alarm in enumerate([True, False, True, False, True]):
        panel.on_frame_scored(i, 0.9, alarm, 4, 3, 0, i * 100)
    assert panel.alarm_table.rowCount() == 3
    # Verify the kept rows are the alarm ones (idx 0, 2, 4).
    assert panel.alarm_table.item(0, 0).text() == "0"
    assert panel.alarm_table.item(1, 0).text() == "2"
    assert panel.alarm_table.item(2, 0).text() == "4"


def test_ids_panel_alarm_log_capped_at_200(qapp):
    panel = IdsPanel()
    for i in range(300):
        panel.on_frame_scored(i, 0.95, True, 4, 3, 0, i * 100)
    assert panel.alarm_table.rowCount() == 200
    # The oldest (i=0) was dropped; the most-recent (i=299) is the
    # last row.
    assert panel.alarm_table.item(0, 0).text() == "100"
    assert panel.alarm_table.item(199, 0).text() == "299"


def test_ids_panel_alarm_log_renders_direction(qapp):
    panel = IdsPanel()
    panel.on_frame_scored(0, 0.9, True, 4, 3, 0, 100)  # direction=0 → cmd
    panel.on_frame_scored(1, 0.9, True, 4, 3, 1, 200)  # direction=1 → resp
    assert panel.alarm_table.item(0, 4).text() == "cmd"
    assert panel.alarm_table.item(1, 4).text() == "resp"


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_ids_panel_reset_clears_state(qapp):
    panel = IdsPanel()
    panel.on_frame_scored(0, 0.95, True, 4, 3, 1, 100)
    panel.on_frame_scored(1, 0.10, False, 4, 3, 0, 200)
    assert panel.alarm_table.rowCount() == 1
    panel.on_started_scoring()
    panel.reset()
    assert "0" in panel.scored_label.text()
    assert "0" in panel.alarm_label.text()
    assert panel.alarm_table.rowCount() == 0
    assert "0/16" in panel.status_label.text()


def test_ids_panel_reset_preserves_user_paths(qapp):
    """File picks survive reset() — the user shouldn't have to re-pick
    between runs."""
    panel = IdsPanel()
    panel._model_path = r"C:\fake\model.pt"
    panel._scaler_path = r"C:\fake\scaler.joblib"
    panel._threshold_path = r"C:\fake\thr.json"
    panel._csv_log_path = r"C:\fake\ids_log.csv"
    panel.reset()
    assert panel.current_model_path() == r"C:\fake\model.pt"
    assert panel.current_scaler_path() == r"C:\fake\scaler.joblib"
    assert panel.current_threshold_path() == r"C:\fake\thr.json"
    assert panel.current_csv_log_path() == r"C:\fake\ids_log.csv"


# ---------------------------------------------------------------------------
# Status label
# ---------------------------------------------------------------------------


def test_ids_panel_started_scoring_flips_window_label(qapp):
    panel = IdsPanel()
    assert "0/16" in panel.status_label.text()
    panel.on_started_scoring()
    assert "16/16" in panel.status_label.text()


def test_ids_panel_run_summary_appends_finished_text(qapp):
    panel = IdsPanel()
    panel.on_run_summary(100, 50, tp=80, fp=20, tn=900, fn=0)
    assert "run finished" in panel.status_label.text()
    assert "100" in panel.status_label.text()
    assert "50" in panel.status_label.text()


def test_ids_panel_error_appends_error_text(qapp):
    panel = IdsPanel()
    panel.on_error("torch missing")
    assert "torch missing" in panel.status_label.text()


# ---------------------------------------------------------------------------
# Binary classification metrics (Accuracy / Precision / Recall / F1)
# ---------------------------------------------------------------------------


def test_ids_panel_metrics_default_dashes_when_no_ground_truth(qapp):
    """Without a metrics update the labels stay at the — sentinel."""
    panel = IdsPanel()
    assert "—" in panel.accuracy_label.text()
    assert "—" in panel.precision_label.text()
    assert "—" in panel.recall_label.text()
    assert "—" in panel.f1_label.text()


def test_ids_panel_metrics_updates_from_metrics_updated(qapp):
    """Confusion matrix update recomputes all 4 Binary labels."""
    panel = IdsPanel()
    # 100 labels total: 80 attack correctly flagged (TP=80), 20 attack
    # missed (FN=20), 850 normal correctly silent (TN=850), 50 normal
    # false-flagged (FP=50). Acc = (80+850)/1000 = 0.93; P = 80/130 ≈
    # 0.6154; R = 80/100 = 0.80; F1 = 2*0.6154*0.8 / (0.6154+0.8) ≈ 0.6957.
    panel.on_metrics_updated(tp=80, fp=50, tn=850, fn=20)
    assert "0.9300" in panel.accuracy_label.text()
    assert "0.6154" in panel.precision_label.text()
    assert "0.8000" in panel.recall_label.text()
    assert "0.6957" in panel.f1_label.text()


def test_ids_panel_metrics_handles_zero_positive_predictions(qapp):
    """When tp+fp=0, precision is defined as 0.0 (no division-by-zero)."""
    panel = IdsPanel()
    panel.on_metrics_updated(tp=0, fp=0, tn=10, fn=0)
    # Acc = 1.0; P = 0.0/0 → 0.0; R = 0.0/0 → 0.0; F1 = 0.0.
    assert "1.0000" in panel.accuracy_label.text()
    assert "0.0000" in panel.precision_label.text()
    assert "0.0000" in panel.recall_label.text()
    assert "0.0000" in panel.f1_label.text()


def test_ids_panel_metrics_perfect_classifier(qapp):
    """tp = total positives, fp = fn = 0 → P = R = F1 = 1.0."""
    panel = IdsPanel()
    panel.on_metrics_updated(tp=100, fp=0, tn=900, fn=0)
    assert "1.0000" in panel.accuracy_label.text()
    assert "1.0000" in panel.precision_label.text()
    assert "1.0000" in panel.recall_label.text()
    assert "1.0000" in panel.f1_label.text()


def test_ids_panel_metrics_reset_returns_to_dashes(qapp):
    panel = IdsPanel()
    panel.on_metrics_updated(tp=10, fp=5, tn=100, fn=2)
    assert "1.0000" not in panel.accuracy_label.text() or "—" not in panel.accuracy_label.text()
    panel.on_started_scoring()
    panel.reset()
    assert "—" in panel.accuracy_label.text()
    assert "—" in panel.precision_label.text()
    assert "—" in panel.recall_label.text()
    assert "—" in panel.f1_label.text()


def test_ids_panel_run_summary_stamps_confusion_matrix(qapp):
    """End-of-run run_summary should also refresh the metric labels."""
    panel = IdsPanel()
    panel.on_run_summary(100, 60, tp=50, fp=10, tn=40, fn=0)
    # Acc = 90/100 = 0.9; P = 50/60; R = 50/50 = 1.0; F1 = 2*P*1/(P+1).
    assert "0.9000" in panel.accuracy_label.text()
    assert "1.0000" in panel.recall_label.text()


# ---------------------------------------------------------------------------
# Threshold spinbox + CSV log gating
# ---------------------------------------------------------------------------


def test_ids_panel_threshold_spinbox_bounds(qapp):
    panel = IdsPanel()
    panel.threshold_spin.setValue(0.123)
    assert panel.current_threshold() == pytest.approx(0.123, abs=1e-4)
    panel.threshold_spin.setValue(0.0)
    assert panel.current_threshold() == 0.0
    panel.threshold_spin.setValue(1.0)
    assert panel.current_threshold() == 1.0


def test_ids_panel_csv_log_enabled_requires_both(qapp):
    panel = IdsPanel()
    # Only checkbox ticked, no path → not enabled.
    panel.csv_chk.setChecked(True)
    assert panel.csv_log_enabled() is False
    # Now provide a path → enabled.
    panel._csv_log_path = r"C:\fake\ids_log.csv"
    assert panel.csv_log_enabled() is True
    # Untick checkbox → disabled again.
    panel.csv_chk.setChecked(False)
    assert panel.csv_log_enabled() is False


def test_ids_panel_set_file_pickers_enabled_locked_during_run(qapp):
    panel = IdsPanel()
    panel.enable_chk.setChecked(True)
    assert panel.model_btn.isEnabled() is True
    panel.set_file_pickers_enabled(False)
    assert panel.model_btn.isEnabled() is False
    # Enable checkbox stays live.
    assert panel.enable_chk.isEnabled() is True
    panel.set_file_pickers_enabled(True)
    assert panel.model_btn.isEnabled() is True


def test_ids_panel_set_file_pickers_disabled_when_ids_off(qapp):
    """set_file_pickers_enabled is a no-op when the IDS is disabled —
    the pickers are already off."""
    panel = IdsPanel()
    assert panel.model_btn.isEnabled() is False
    panel.set_file_pickers_enabled(True)
    assert panel.model_btn.isEnabled() is False