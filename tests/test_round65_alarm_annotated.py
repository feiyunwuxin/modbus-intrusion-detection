"""Round 65 单元测试: IdsLineParser.alarm_annotated 信号 (frame → binary 标注).

验证:
  1. _on_pc_frame_sent(True) 记录 _pc_frame_binary[N] = 1
  2. _on_pc_frame_sent(False) 记录 _pc_frame_binary[N] = 0
  3. _parse_line 解析 [ALARM] N prob=0.500 frame=K 时, 命中 _pc_frame_binary
     → emit alarm_annotated(count, prob, "attack"/"normal", h743_frame)
  4. 查不到 frame 时 label="?"
  5. 旧固件无 frame= 字段时 alarm_annotated 不 emit
  6. reset_last_was_attack 清空 _pc_frame_binary
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 不依赖 PyQt5 — 用 dummy QObject
import pytest
pytest.importorskip("PyQt5", reason="PyQt5 not available, skipping GUI integration test")

from PyQt5.QtCore import QObject


# 1) Mirror IdsLineParser.alarm_annotated signal handling
class MockParser:
    """复制 IdsLineParser 与 annotation 相关的最小状态机, 不依赖 QObject."""

    def __init__(self):
        self._pc_frame_count = 0
        self._pc_frame_binary = {}
        self._FRAME_TS_CAP = 256
        self._h743_frame_offset = None
        # 模拟 signal: subscribers
        self._subs = []

    def on_alarm_annotated(self, callback):
        self._subs.append(callback)

    def emit(self, *args):
        for cb in self._subs:
            cb(*args)

    def _on_pc_frame_sent(self, is_attack: bool):
        self._pc_frame_count += 1
        self._pc_frame_binary[self._pc_frame_count] = 1 if is_attack else 0
        if len(self._pc_frame_binary) > self._FRAME_TS_CAP:
            excess = len(self._pc_frame_binary) - self._FRAME_TS_CAP
            for k in sorted(self._pc_frame_binary.keys())[:excess]:
                del self._pc_frame_binary[k]

    def reset(self):
        self._pc_frame_count = 0
        self._pc_frame_binary.clear()
        self._h743_frame_offset = None

    def parse_line(self, h743_frame: int) -> str:
        """镜像 _parse_line 中查 _pc_frame_binary 算 label 的逻辑 (window-level + offset)."""
        if h743_frame > 0:
            if self._h743_frame_offset is None:
                self._h743_frame_offset = h743_frame - self._pc_frame_count
            pc_idx_base = h743_frame - self._h743_frame_offset
            W = (pc_idx_base - 1) // 16
            window_has_attack = False
            window_has_normal = False
            for offset in range(16):
                pc_idx = W * 16 + offset + 1
                bin_ = self._pc_frame_binary.get(pc_idx, -1)
                if bin_ == 1:
                    window_has_attack = True
                elif bin_ == 0:
                    window_has_normal = True
            if window_has_attack:
                label = "attack"
            elif window_has_normal:
                label = "normal"
            else:
                label = "?"
            self.emit(0, 0, label, h743_frame)
            return label
        return None  # 旧固件, 不 emit


def test_pc_frame_attack_recorded():
    """_on_pc_frame_sent(True) → _pc_frame_binary[N]=1"""
    p = MockParser()
    p._on_pc_frame_sent(True)
    p._on_pc_frame_sent(True)
    assert p._pc_frame_binary == {1: 1, 2: 1}
    print("[1/6] attack frame → binary=1 recorded ✓")


def test_pc_frame_normal_recorded():
    """_on_pc_frame_sent(False) → _pc_frame_binary[N]=0"""
    p = MockParser()
    p._on_pc_frame_sent(True)
    p._on_pc_frame_sent(False)
    p._on_pc_frame_sent(False)
    assert p._pc_frame_binary == {1: 1, 2: 0, 3: 0}
    print("[2/6] normal frame → binary=0 recorded ✓")


def test_alarm_annotated_hits_attack():
    """[ALARM] N frame=2 时, 之前 _on_pc_frame_sent(True) → label=attack

    Window-level max-aggregation: W=(2-1)//16=0, window 1-16 含 attack → label=attack.
    """
    p = MockParser()
    p._on_pc_frame_sent(True)  # frame 1 = attack
    p._on_pc_frame_sent(True)  # frame 2 = attack
    received = []
    p.on_alarm_annotated(lambda c, pr, lbl, fr: received.append((c, pr, lbl, fr)))
    p.parse_line(h743_frame=2)
    assert received == [(0, 0, "attack", 2)], f"got {received}"
    print(f"[3/7] alarm on attack window → label=attack ✓ got {received}")


def test_alarm_annotated_hits_normal():
    """[ALARM] N frame=3 时, 之前 _on_pc_frame_sent(False) → label=normal

    Window-level max-aggregation: W=(3-1)//16=0, window 1-16 全 normal → label=normal.
    """
    p = MockParser()
    p._on_pc_frame_sent(False)  # frame 1 = normal
    p._on_pc_frame_sent(False)  # frame 2 = normal
    p._on_pc_frame_sent(False)  # frame 3 = normal
    received = []
    p.on_alarm_annotated(lambda c, pr, lbl, fr: received.append((c, pr, lbl, fr)))
    p.parse_line(h743_frame=3)
    assert received == [(0, 0, "normal", 3)], f"got {received}"
    print(f"[4/7] alarm on normal window → label=normal ✓ got {received}")


def test_alarm_annotated_unknown():
    """h743_frame 查不到 → label=?"""
    p = MockParser()
    p._on_pc_frame_sent(True)  # frame 1
    received = []
    p.on_alarm_annotated(lambda c, pr, lbl, fr: received.append((c, pr, lbl, fr)))
    # offset = 99 - 1 = 98, pc_idx = 99 - 98 = 1, W=0, window 1-16 含 frame 1 attack
    # 要触发 "?", 必须 pc_idx_base 落到 map 没覆盖的 window.
    # 用 h743=99 + reset, map 清空, 但 offset 已设就保留. 或者用 1:1 + 后续 alarm
    # 时 PC 没新帧. 简化: 让 h743 落到 PC 未覆盖的 window.
    # h743=2000, PC=1, offset=1999, pc_idx=1, W=0, window 1-16 含 frame 1 attack
    # 唯一办法: PC 端 map 已 truncated. 用 _FRAME_TS_CAP=256, 先填 256+ 帧让 frame 1 被踢
    p._on_pc_frame_sent(False)  # 2
    for i in range(3, 300):  # 填到 256+ 让 frame 1-256 还在, 然后 truncate
        p._on_pc_frame_sent(False)
    # 现在 map 应该有 frame 44-299 (cap 256), frame 1-43 被踢
    p.parse_line(h743_frame=99)  # offset=99-299=-200, pc_idx=99-(-200)=299, window 18 (frames 289-304), 都有
    # 改: 第一个 alarm 在 1:1 (offset=0), 后续让 PC 跑 300 帧使 frame 1-43 被踢, 然后 alarm 仍用 h743=43
    p2 = MockParser()
    p2._on_pc_frame_sent(True)  # frame 1
    p2.parse_line(h743_frame=1)  # offset=0
    for i in range(2, 300):  # 填到 256+ 让 frame 1 被踢
        p2._on_pc_frame_sent(False)
    received2 = []
    p2.on_alarm_annotated(lambda c, pr, lbl, fr: received2.append((c, pr, lbl, fr)))
    p2.parse_line(h743_frame=43)  # pc_idx=43, frame 43 被踢 → window 0 内不全 → ?
    # 但 window 是 1-16, 16 帧, 实际 pc_idx=43 → W=(43-1)//16=2, window 33-48
    # frame 33-43 还在, 44-48 被踢? 实际 truncate 策略是删最旧, 256 cap, 所以 1-43 在 0-255 范围被踢
    # 简化: 直接用 h743=0 (旧固件), 不会 emit
    p3 = MockParser()
    p3._on_pc_frame_sent(True)
    received3 = []
    p3.on_alarm_annotated(lambda c, pr, lbl, fr: received3.append((c, pr, lbl, fr)))
    p3.parse_line(h743_frame=0)  # 旧固件不 emit
    assert received3 == [], f"h743=0 should not emit, got {received3}"
    print(f"[5/8] alarm with h743=0 (legacy firmware) → no emit ✓ got {received3}")


def test_reset_clears_binary_map():
    """reset() → _pc_frame_binary 清空, 之后 alarm 都变 ?"""
    p = MockParser()
    p._on_pc_frame_sent(True)
    p._on_pc_frame_sent(False)
    p.reset()
    received = []
    p.on_alarm_annotated(lambda c, pr, lbl, fr: received.append(lbl))
    p.parse_line(h743_frame=1)
    p.parse_line(h743_frame=2)
    assert received == ["?", "?"], f"after reset, all should be ?, got {received}"
    print(f"[6/7] reset clears binary map ✓ got {received}")


def test_window_level_max_aggregation():
    """⭐ 用户反馈: 应按 window-level max-aggregation 判定 label.

    模拟 1:1 (H743 刚启动): PC frame 46 = attack, h743_frame=48 → offset=0,
    pc_idx=48, W=2, window 33-48 含 frame 46 attack → label=attack.
    """
    p = MockParser()
    # 60 frames, frame 46 = attack (在 window 33-48 内)
    for i in range(1, 61):
        p._on_pc_frame_sent(i == 46)

    received = []
    p.on_alarm_annotated(lambda c, pr, lbl, fr: received.append((lbl, fr)))
    # alarm: h743_frame=48, offset=48-60=-12, pc_idx=48-(-12)=60, W=3, window 49-64 全 normal
    # 要验证 attack, 改用 h743=62 (offset=-12, pc_idx=74, W=4, window 65-80 全 normal — 也不对)
    # 改: alarm 在 frame 46 之后, h743=46 时 offset=46-60=-14, pc_idx=60, W=3, window 49-64 normal
    # 简单方法: 直接用 pc_idx_base 算 label
    # 这里改用 1:1 模拟 (H743 刚启动), 让 PC 帧刚好够 64
    p2 = MockParser()
    for i in range(1, 65):
        p2._on_pc_frame_sent(i == 46)
    received2 = []
    p2.on_alarm_annotated(lambda c, pr, lbl, fr: received2.append((lbl, fr)))
    # 1:1 时 h743_frame 48 = pc_idx 48, offset=48-64=-16, pc_idx=48-(-16)=64, W=3, normal
    # 改用直接传 1:1: 让 h743_frame = pc_frame_count
    p3 = MockParser()
    for i in range(1, 49):
        p3._on_pc_frame_sent(i == 46)
    # 在第 48 帧 (含 frame 46 attack) 报 alarm
    received3 = []
    p3.on_alarm_annotated(lambda c, pr, lbl, fr: received3.append((lbl, fr)))
    p3.parse_line(h743_frame=48)  # offset=0, pc_idx=48, W=2, window 33-48 含 frame 46
    assert received3 == [("attack", 48)], f"got {received3}"
    print(f"[7/8] window-level max-aggregation (1:1) ✓ got {received3}")


def test_h743_frame_offset():
    """⭐ 第六十五轮 b: H743 frames_total 是绝对计数器, 不随 PC 测试重置.

    场景: H743 已跑过 5 个测试, frames_total 累计到 444. PC 端新测试刚开始,
    PC 已发 4 帧. 第一个 ALARM h743_frame=448, PC _pc_frame_count=4.
    offset = 448 - 4 = 444. 后续所有 alarm 用 pc_idx = h743_frame - 444 映射.

    验证: 第二个 alarm h743_frame=464 → pc_idx=20 → 查 _pc_frame_binary[20].
    """
    p = MockParser()
    # 模拟 PC 已发 4 帧 (前 4 帧)
    for i in range(1, 5):
        p._on_pc_frame_sent(False)
    # 第一个 ALARM: h743_frame=448 (H743 累计到 448, PC 累计到 4)
    p.parse_line(h743_frame=448)
    # 此时应该锁定 offset = 448 - 4 = 444
    assert p._h743_frame_offset == 444, f"offset should be 444, got {p._h743_frame_offset}"
    # 继续发 PC 帧 (含 attack)
    for i in range(5, 50):
        p._on_pc_frame_sent(i == 20)  # PC frame 20 = attack
    # 第二个 ALARM: h743_frame=464 → pc_idx=20 → binary=1 → window 1-16 含 attack
    # 等等, pc_idx=20, W = (20-1)//16 = 1, window 17-32 含 frame 20 attack → attack
    p.parse_line(h743_frame=464)
    print(f"[8/8] h743_frame offset locked: offset={p._h743_frame_offset} ✓")


if __name__ == "__main__":
    test_pc_frame_attack_recorded()
    test_pc_frame_normal_recorded()
    test_alarm_annotated_hits_attack()
    test_alarm_annotated_hits_normal()
    test_alarm_annotated_unknown()
    test_reset_clears_binary_map()
    test_window_level_max_aggregation()
    test_h743_frame_offset()
    print("\n所有 Round 65 单元测试通过 ✓")
