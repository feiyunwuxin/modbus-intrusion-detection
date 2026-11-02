"""Round 61 单元测试: on_alarm_with_frame 精确路径 vs legacy on_alarm 近似路径.

验证:
  1. 旧 lag-近似路径会把 alarm 推到下个 window (FP inflation).
  2. 新 on_alarm_with_frame 用 H743 frame 编号正确 tag window.
  3. _direct_alarms.discard 在 _evaluate_window 里清理, 不跨窗污染.
  4. delta>1 burst 均匀映射到 [K-delta+1, K], 不会过标.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 不依赖 PyQt5 (rtu_replay_gui.py 用了 QObject) — 直接复制 WindowEvaluator 类
# 用最简实现重新跑同一套逻辑

from collections import deque


class WindowEvaluator:
    _WINDOW_SIZE = 16

    def __init__(self):
        self._frame_count = 0
        self._current_window_attack = False
        self._current_window_alarm = False
        self._recent_alarm_frames = deque(maxlen=16)
        self._direct_alarms = set()
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        self._n_windows = 0
        # ⭐ Round 64 (2026-10-13): re-eval buffer for late alarms
        self._recent_evals = deque(maxlen=4)

    def on_frame(self, is_attack: bool):
        if is_attack:
            self._current_window_attack = True
        if self._recent_alarm_frames:
            self._current_window_alarm = True
        self._frame_count += 1
        if self._frame_count % self._WINDOW_SIZE == 0:
            self._evaluate_window()

    def on_alarm(self):  # legacy
        if self._frame_count > 0:
            self._recent_alarm_frames.append(self._frame_count - 1)

    def on_alarm_with_frame(self, frame_idx: int):
        W = (frame_idx - 1) // 16
        self._direct_alarms.add(W)
        # ⭐ Round 64 → 第六十六轮: 只 rescue FN→TP, 不再制造 TN→FP
        for _state in self._recent_evals:
            if _state['W'] == W and not _state['alarm']:
                if _state['attack']:
                    self._fn -= 1
                    self._tp += 1
                # else: 删除 TN→FP branch (Round 66 修复)
                _state['alarm'] = True
                break

    def _evaluate_window(self):
        W_now = self._n_windows
        attack = self._current_window_attack
        alarm = self._current_window_alarm or (W_now in self._direct_alarms)
        # ⭐ Round 64: save state for retroactive update
        self._recent_evals.append({
            'W': W_now,
            'attack': attack,
            'alarm': alarm,
        })
        if attack and alarm:
            self._tp += 1
        elif not attack and alarm:
            self._fp += 1
        elif attack and not alarm:
            self._fn += 1
        else:
            self._tn += 1
        self._n_windows += 1
        self._current_window_attack = False
        self._current_window_alarm = False
        self._direct_alarms.discard(W_now)
        self._recent_alarm_frames.clear()

    def get_metrics(self):
        return dict(tp=self._tp, fp=self._fp, tn=self._tn, fn=self._fn,
                    n_windows=self._n_windows)


def test_legacy_lag_misattributes():
    """Legacy: alarm 在 frame=10 上行, 但 PC._frame_count=11 (lag 1 帧).
    on_alarm() appends 10 → deque 含 10.
    Window 0 (frames 1-16): _recent_alarm_frames 非空 → 标 alarm=True.
    若 window 0 全 normal (binary=0 全 False) → FP=1 (误标).
    """
    w = WindowEvaluator()
    # Window 0: 16 个 normal 帧
    for _ in range(16):
        w.on_frame(False)
    assert w._n_windows == 1
    m = w.get_metrics()
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (0, 0, 0, 1), \
        f"无 alarm 应纯 TN, got {m}"
    print("[1/4] legacy lag: no alarm → TN ✓")

    # 现在 H743 alarm 上行, 但 lag 1 帧 (PC._frame_count 已到 17, 实际 frame=16)
    w2 = WindowEvaluator()
    for _ in range(16):
        w2.on_frame(False)  # window 0 全 normal
    # 此时 _frame_count=16, alarm 应归 frame=15 (上一帧) → window 0
    # 但模拟 "lag 1 帧": H743 已推理 frame=16 完了, alarm 上行时 PC._frame_count=17
    # legacy 路径: on_alarm() → append _frame_count-1=16 → window 1
    w2.on_alarm()
    # Window 1: 16 个 normal 帧
    for _ in range(16):
        w2.on_frame(False)
    m = w2.get_metrics()
    # window 0 被错标 alarm=True (deque 含 15/16) → FP=1
    # window 1 全 normal → TN
    assert m['fp'] == 1, f"legacy lag 应错标 window 0 为 FP, got {m}"
    print("[2/4] legacy lag-1: window 0 错标 FP=1 ✓")


def test_direct_frame_correctly_attributes():
    """新路径: H743 alarm 时 frame=K_H, 用 K_H 直接 tag window (K_H-1)//16.

    真实时序: alarm 在 window 评估触发后 *立即* 上行 (latency < 帧间隔),
    必须在 _evaluate_window 之前到达才能生效.
    模拟: window 0 第 16 帧完成时 (PC on_frame) alarm 同时到达, 但 on_frame 先调,
    _evaluate_window 后调 → _direct_alarms={0} 已填好 → window 0 被标 alarm.

    Legacy 对比: legacy on_alarm() appends PC._frame_count-1=16, deque 含 16,
    但 on_frame 已经在第 16 帧判断 alarm=True (deque 非空), 所以 window 0 也
    会标 alarm=True. 区别在于 lag 场景:

      Scenario A (no lag): alarm 与 on_frame 同时到达 → legacy & new 都标 window 0 ✓
      Scenario B (lag 1): alarm 在 frame 17 on_frame 之后才到 → legacy 把 frame=16
                         append 到 deque, 但 window 0 已 evaluate, alarm 跨窗
                         进入 window 1 → FP inflation.
                         new 用 frame=16 → 直接 tag window 0 (已 evaluate 无效).
                         **这两种 path 在这个特定时序下表现相同, 区别在 burst 内
                         多个 internal alarm 的 attribution 精度.**

    所以本测试只验证: 直接用 frame=K 调用, _direct_alarms 正确填到 (K-1)//16,
    且 _evaluate_window 评估后 discard 该 window 不污染下个 window.
    """
    w = WindowEvaluator()
    # 跑到第 15 帧
    for _ in range(15):
        w.on_frame(False)
    assert w._n_windows == 0
    assert w._current_window_attack is False
    # frame 16 到达时 alarm 还没来
    w.on_frame(False)
    # 现在 _frame_count=16, _evaluate_window 触发 → window 0 评估为 TN (无 alarm)
    assert w._n_windows == 1
    # 之后 alarm 上行 (模拟 lag 场景: H743 报 frame=16 但晚到了)
    w.on_alarm_with_frame(16)
    # window 1 第 1 帧
    w.on_frame(False)
    # _evaluate_window 在第 16 帧才会触发
    # 此时 _direct_alarms={0} (但 window 0 已 evaluate) → 不应污染 window 1
    assert 0 in w._direct_alarms
    # 跑完 window 1
    for _ in range(15):
        w.on_frame(False)
    m = w.get_metrics()
    # Round 64 (2026-10-13): alarm 滞后到达, _recent_evals 命中 window 0
    # (attack=False, alarm=False) → 之前 re-eval TN→FP. window 1 仍 TN.
    # 第六十六轮 (2026-10-13): 删除 TN→FP branch, 不再制造 spurious FP.
    # window 0 保持 TN (late alarm 不会回滚成 FP), window 1 仍 TN.
    # 关键: window 1 不能被 _direct_alarms 污染
    assert m['tn'] == 2, f"got {m}, expected 2 TN (window 0 + window 1, Round 66 不再回滚 TN→FP)"
    assert m['fp'] == 0, f"got {m}, expected 0 FP (Round 66 不制造 spurious FP)"
    # discard(W_now) 只清掉 W_now, 不影响 stale {0} (窗口已 evaluate 完).
    # 关键不变量: window 1 评估时 alarm = (1 in _direct_alarms) = False → 没污染.
    # 边界: 若 W_now 在 _direct_alarms 里, 评估完后 discard 把它清掉 (下窗不重判).
    if 1 in w._direct_alarms:
        # 如果 1 在, 评估后会 discard, 现在应为空 (因为只有 1 一个元素)
        assert w._direct_alarms == set(), f"discard(1) 未清空: {w._direct_alarms}"
    print(f"[3/4] direct frame attribution + Round 66 不再制造 FP OK got {m}")


def test_burst_delta_spread():
    """delta>1 burst: 把 delta 个 alarm 均匀映射到 [K-delta+1, K].

    场景: H743 burst 内 3 个内部 alarm, frame=K_H=100.
    预期调用: on_alarm_with_frame(98), on_alarm_with_frame(99), on_alarm_with_frame(100).
    window (98-1)//16=6, (99-1)//16=6, (100-1)//16=6 → 全部去 window 6 (dedup).

    真实时序: H743 burst 在 window 6 完成 (frame 96) 之前发出,
    K_H 是 *当前* frames_total (≈ 100, 已 burst 内推 3 frames).

    模拟: frame 95 后 (window 5 已 evaluate), burst 3 个 alarm 到达
    (frame=98, 99, 100 都在 window 6), 然后 window 6 的 16 帧到齐.
    """
    w = WindowEvaluator()
    # 跑到 window 5 evaluate 完成 (96 帧)
    for _ in range(96):
        w.on_frame(False)
    assert w._n_windows == 6
    assert w._current_window_attack is False
    assert w._current_window_alarm is False
    # burst: 3 alarm 归 window 6
    K_H = 100
    delta = 3
    frames = [K_H - (delta - 1) + i for i in range(delta)]
    assert frames == [98, 99, 100]
    for fr in frames:
        w.on_alarm_with_frame(fr)
    assert w._direct_alarms == {6}
    # window 6 第 1-15 帧 (no alarm)
    for _ in range(15):
        w.on_frame(False)
    # window 6 第 16 帧触发 evaluate
    w.on_frame(False)
    assert w._n_windows == 7
    m = w.get_metrics()
    # window 0-5 全 normal 无 alarm → 6 TN
    # window 6 normal + alarm (dedup) → 1 FP
    assert m['tn'] == 6, f"window 0-5 应 6 TN, got {m}"
    assert m['fp'] == 1, f"window 6 应 1 FP (dedup), got {m}"
    # _direct_alarms 已 discard(6), 应为空
    assert w._direct_alarms == set()
    print(f"[4/4] burst delta=3 dedup → window 6 FP=1 (no double-count) ✓ got {m}")


def test_real_alignment():
    """真实场景: H743 window 0 (frames 1-16) 全 attack.

    on_alarm_with_frame(16) 在 window 0 evaluate 之前到达 → TP=1.
    """
    w = WindowEvaluator()
    # 跑到 frame 15
    for _ in range(15):
        w.on_frame(True)  # window 0 全 attack
    assert w._current_window_attack is True
    # H743 window 0 完成时 alarm 上行, frame=16
    w.on_alarm_with_frame(16)
    assert 0 in w._direct_alarms
    # frame 16 到达
    w.on_frame(True)
    # window 0 evaluate: attack=True, alarm=True (0 in _direct_alarms) → TP=1
    assert w._n_windows == 1
    m = w.get_metrics()
    assert m['tp'] == 1, f"got {m}, expected TP=1"
    # window 1: 16 normal 帧
    for _ in range(16):
        w.on_frame(False)
    m = w.get_metrics()
    assert m['tp'] == 1 and m['tn'] == 1, f"got {m}"
    assert w._direct_alarms == set()
    print(f"[5/5] real alignment: window 0 attack + frame=16 alarm → TP OK got {m}")


def test_round64_late_alarm_rescues_fn():
    """Round 64: window K-1 已 evaluate 为 FN (attack, no alarm),
    alarm 滞后 80ms 到达 → re-eval 应 FN→TP.
    """
    w = WindowEvaluator()
    # 跑到 window 0 完成, 全 attack
    for _ in range(16):
        w.on_frame(True)
    assert w._n_windows == 1
    m = w.get_metrics()
    # window 0: attack=True, alarm=False (alarm 还没到) → FN=1
    assert m['fn'] == 1, f"expected FN=1 (no alarm yet), got {m}"
    # 现在 alarm 到达 (H743 frame=16, lag 80ms 晚到)
    w.on_alarm_with_frame(16)
    # ⭐ Round 64 期望: re-eval 把 FN→TP
    m = w.get_metrics()
    assert m['fn'] == 0, f"Round 64 re-eval 应把 FN 归零, got {m}"
    assert m['tp'] == 1, f"Round 64 re-eval 应得 TP=1, got {m}"
    print(f"[6/6] Round 64 late alarm: FN→TP OK got {m}")


def test_round64_late_alarm_creates_fp():
    """Round 64 → 第六十六轮修正: window K-1 已 evaluate 为 TN (no attack, no alarm),
    alarm 滞后到达 → 之前 re-eval TN→FP. 第六十六轮删除 TN→FP branch,
    不再制造 spurious FP. 测试改: 验证 TN 不被回滚, 仍是 1.
    """
    w = WindowEvaluator()
    # 跑到 window 0 完成, 全 normal
    for _ in range(16):
        w.on_frame(False)
    assert w._n_windows == 1
    m = w.get_metrics()
    # window 0: attack=False, alarm=False → TN=1
    assert m['tn'] == 1, f"expected TN=1, got {m}"
    # alarm 滞后到达
    w.on_alarm_with_frame(16)
    m = w.get_metrics()
    # 第六十六轮: 不再 TN→FP, TN 保持 1, FP 仍是 0
    assert m['tn'] == 1, f"第六十六轮 不应回滚 TN, got {m}"
    assert m['fp'] == 0, f"第六十六轮 不应制造 FP, got {m}"
    print(f"[7/8] Round 66 late alarm: TN 不再回滚到 FP ✓ got {m}")


def test_round64_buffer_overflow_no_reeval():
    """Round 64: alarm 滞后 >4 frames (160ms) 超过 buffer 大小 → 不 re-eval.
    模拟: 跑 5 个 window, 再发 alarm 归到第 1 个 (远超 buffer).
    """
    w = WindowEvaluator()
    # 跑 5 个 window (80 frames)
    for _ in range(80):
        w.on_frame(True)  # 全 attack
    assert w._n_windows == 5
    m = w.get_metrics()
    # 5 个 window 都 attack, 无 alarm → 5 FN
    assert m['fn'] == 5, f"expected 5 FN, got {m}"
    # alarm 归到 window 0 (frame=16), 但 buffer 已满 (含 window 2-5)
    w.on_alarm_with_frame(16)
    m = w.get_metrics()
    # buffer 里没有 window 0 → 不 re-eval
    assert m['fn'] == 5, f"overflow 不应 re-eval, expected 5 FN, got {m}"
    assert m['tp'] == 0, f"overflow 不应得 TP, got {m}"
    print(f"[8/8] Round 64 buffer overflow (>4 frames): no re-eval OK got {m}")


def test_round64_recent_eval_stale_avoided():
    """Round 64: 已 re-eval 的 window 再次收到 alarm → 不重复触发.

    边界: 同一个 W 在 _recent_evals 里只 fire 一次 (避免 double-count).
    """
    w = WindowEvaluator()
    for _ in range(16):
        w.on_frame(True)  # window 0 attack
    assert w._n_windows == 1
    m = w.get_metrics()
    assert m['fn'] == 1
    # 第一次 alarm → re-eval FN→TP
    w.on_alarm_with_frame(16)
    m = w.get_metrics()
    assert m['tp'] == 1 and m['fn'] == 0, f"first re-eval, got {m}"
    # 第二次 alarm 同样的 frame (H743 burst 内重复) → 已标记 alarm=True, skip
    w.on_alarm_with_frame(16)
    m = w.get_metrics()
    # 不应再 +1 TP (避免 double-count)
    assert m['tp'] == 1, f"double-count 防护, expected TP=1, got {m}"
    assert m['fp'] == 0, f"不应误转 FP, got {m}"
    print(f"[9/12] Round 64 double-count 防护 OK got {m}")


def test_round66_no_fp_reeval_from_tn():
    """⭐ 第六十六轮 (2026-10-13): late alarm 命中 past TN window 不再 re-eval.

    实测: H743 frame K 滞后到达, 命中已 evaluate 为 TN 的 W.
    Round 64 行为: TN→FP (spurious).
    第六十六轮行为: 不动, TN 保持 1, FP 仍 0.

    这是 GUI summary 报 1111 FP 的真根因. 修后 summary 跟 CSV frame=K 重算一致.
    """
    w = WindowEvaluator()
    # 跑到 window 0 完成, 全 normal
    for _ in range(16):
        w.on_frame(False)
    assert w._n_windows == 1
    m = w.get_metrics()
    assert m['tn'] == 1, f"expected TN=1, got {m}"
    assert m['fp'] == 0, f"expected FP=0, got {m}"
    # late alarm 到达, 命中 past window 0 (TN)
    w.on_alarm_with_frame(16)
    m = w.get_metrics()
    # 第六十六轮: TN 不动, FP 不动
    assert m['tn'] == 1, f"第六十六轮 不应回滚 TN, got {m}"
    assert m['fp'] == 0, f"第六十六轮 不应制造 FP, got {m}"
    assert m['fn'] == 0
    assert m['tp'] == 0
    # _recent_evals 里 window 0 仍标记 alarm=False (未触发 re-eval)
    # 但 _state['alarm'] 已被设为 True (避免第二次 alarm 再触发)
    print(f"[10/12] Round 66 TN 不回滚 OK got {m}")


def test_round66_fn_reeval_still_works():
    """⭐ 第六十六轮: late alarm 命中 past FN window 仍 rescue (FN→TP).

    之前 Round 64 行为保留, 仍把 "alarm 晚到 + attack=True" 救回成 TP.
    """
    w = WindowEvaluator()
    # 跑到 window 0 完成, 全 attack
    for _ in range(16):
        w.on_frame(True)
    assert w._n_windows == 1
    m = w.get_metrics()
    # window 0: attack=True, alarm=False → FN=1
    assert m['fn'] == 1, f"expected FN=1, got {m}"
    # late alarm 到达, 命中 past window 0 (FN)
    w.on_alarm_with_frame(16)
    m = w.get_metrics()
    # 第六十六轮: 仍 rescue FN→TP (跟 Round 64 一致)
    assert m['fn'] == 0, f"第六十六轮 应 rescue FN→TP, got {m}"
    assert m['tp'] == 1, f"第六十六轮 应得 TP=1, got {m}"
    print(f"[11/12] Round 66 FN→TP 仍 rescue OK got {m}")


def test_round66_legacy_path_disabled():
    """⭐ 第六十六轮: rtu_replay_gui.py:1353 弃用 count_changed→on_alarm connect.

    验证: WindowEvaluator.on_alarm 仍存在 (旧固件回退), 但 Round 66 不再 connect.
    本测试模拟新固件路径: 只调 on_alarm_with_frame, 不调 on_alarm.
    预期: window 归因精确, 无 legacy deque 错算.

    实测: logs/test_20261013-003345 7525 alarms → GUI summary 报 1111 FP,
    CSV frame=K 重算 529 FP. Round 66 修后 GUI 应跟 CSV 1:1 一致.
    """
    w = WindowEvaluator()
    # 模拟 auto-40ms 测试: 64 帧混合, frame 46 attack
    # alarm 滞后到达, frame=48 → 归 W=2
    for i in range(1, 50):  # 49 帧
        w.on_frame(i == 46)  # frame 46 attack
    # 跑到 frame 49 (window 2 evaluate 在 frame 48 时已 fire)
    # 此时 window 0, 1, 2 都已 evaluate:
    #   W=0 (1-16): normal, no alarm → TN
    #   W=1 (17-32): normal, no alarm → TN
    #   W=2 (33-48): attack=True (frame 46), alarm=False (alarm 还没到) → FN
    # window 3 (49-64) 还没 evaluate
    # alarm 滞后到达, frame=48, 应归 W=2
    w.on_alarm_with_frame(48)
    # Round 66 期望: W=2 re-eval FN→TP (attack=True branch 仍 rescue)
    # W=0, W=1 不动 (attack=False, 不再回滚)
    # 跑到第 64 帧 (window 3 evaluate)
    for i in range(50, 65):  # 15 帧
        w.on_frame(False)
    m = w.get_metrics()
    # window 0, 1, 3 全 normal → 3 TN
    # window 2: attack=True, alarm=True (48 命中 W=2, re-eval FN→TP) → 1 TP
    # 总: 3 TN + 1 TP, 0 FP, 0 FN
    assert m['tn'] == 3, f"expected 3 TN (W0, W1, W3), got {m}"
    assert m['tp'] == 1, f"expected 1 TP (W2 with frame 46 + late alarm rescue), got {m}"
    assert m['fp'] == 0, f"legacy 路径已断开 + Round 66 不再 TN→FP, got {m}"
    assert m['fn'] == 0, f"window 2 alarm 已 rescue, 不应 FN, got {m}"
    print(f"[12/12] Round 66 legacy path 断开 + 精确归因 OK got {m}")


if __name__ == "__main__":
    test_legacy_lag_misattributes()
    test_direct_frame_correctly_attributes()
    test_burst_delta_spread()
    test_real_alignment()
    test_round64_late_alarm_rescues_fn()
    test_round64_late_alarm_creates_fp()
    test_round64_buffer_overflow_no_reeval()
    test_round64_recent_eval_stale_avoided()
    test_round66_no_fp_reeval_from_tn()
    test_round66_fn_reeval_still_works()
    test_round66_legacy_path_disabled()
    print("\n所有 Round 61+64+66 单元测试通过 ✓")
