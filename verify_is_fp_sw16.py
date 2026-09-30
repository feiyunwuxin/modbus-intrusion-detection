#!/usr/bin/env python3
"""验证 IdsLineParser 第二十二轮滑动窗口 is_fp 逻辑.

修复前: is_fp 只看上一帧 binary → attack burst 内除首帧外 15 个 alarm 全误标 FP
修复后: is_fp 看最近 16 帧滑动窗口 → 任一帧 attack → TP, 全 normal → FP

测试场景:
  1. ramp-up 期 (< 16 帧): 不判定 TP/FP
  2. attack burst (连续 attack): 整个 burst 期间所有 ALARM 全部标 TP
  3. 全 normal 期间 ALARM: 标 FP
  4. 混合场景: ramp-up + normal + burst + normal
  5. 验证 2026-09-29 auto-40ms 真实日志: F1 从 25.13% → ~51%

不依赖 PyQt5, 用纯 Python 实现 IdsLineParser 镜像.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import re
from collections import deque
from typing import Optional


class IdsLineParserMirror:
    """镜像 rtu_replay_gui.py IdsLineParser 第二十二轮修复版 (无 PyQt5 依赖)."""

    _SW_SIZE = 16
    _ALARM_RE = re.compile(rb'\[ALARM\]\s+(\d+)')
    _PROB_RE  = re.compile(rb'prob=0\.(\d{1,3})')

    def __init__(self):
        self._buf = bytearray()
        self._window: "deque[bool]" = deque(maxlen=self._SW_SIZE)
        # 模拟信号回调
        self.fp_history = []  # [(count, is_fp, prob_x1000), ...]
        self.count_history = []  # [count, ...]

    def set_last_was_attack(self, val: Optional[bool]) -> None:
        if val is None:
            self._window.clear()
            return
        self._window.append(bool(val))

    def reset_last_was_attack(self) -> None:
        self._window.clear()

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)
        while True:
            idx = self._buf.find(b'\n')
            if idx < 0:
                if len(self._buf) > 8192:
                    del self._buf[:len(self._buf) - 4096]
                return
            line = bytes(self._buf[:idx]).strip(b'\r').strip()
            del self._buf[:idx + 1]
            if not line:
                continue
            self._parse_line(line)

    def _parse_line(self, line: bytes) -> None:
        m = self._ALARM_RE.search(line)
        if not m:
            return
        count = int(m.group(1))
        prob_match = self._PROB_RE.search(line)
        prob_x1000 = int(prob_match.group(1)) if prob_match else -1
        self.count_history.append(count)
        is_fp: Optional[bool] = None
        if len(self._window) >= self._SW_SIZE:
            is_fp = not any(self._window)
            self.fp_history.append((count, is_fp, prob_x1000))


# ════════════════════════════════════════════════════════════════════
# 测试场景
# ════════════════════════════════════════════════════════════════════
print('=' * 70)
print('测试 1: ramp-up 期 (< 16 帧) 不判定 TP/FP')
print('=' * 70)
p = IdsLineParserMirror()
# 推 5 帧 normal, 然后 1 个 alarm
for _ in range(5):
    p.set_last_was_attack(False)
p.feed(b'[ALARM] 1 prob=0.700\n')
assert len(p.fp_history) == 0, f'期望 0 个 fp_history (ramp-up), 实际 {len(p.fp_history)}'
print(f'  5 帧 normal + 1 alarm → fp_history={len(p.fp_history)} ✓ (期望 0, ramp-up 不判定)')

# 继续推帧直到填满 16 帧, 然后给 alarm
for _ in range(15):
    p.set_last_was_attack(False)
assert len(p._window) == 16
print(f'  窗口已填满: len(_window)={len(p._window)}')
p.feed(b'[ALARM] 2 prob=0.500\n')
assert len(p.fp_history) == 1
assert p.fp_history[0] == (2, True, 500), f'期望 (2, True, 500), 实际 {p.fp_history[0]}'
print(f'  16 帧 normal + 1 alarm → fp_history={p.fp_history} ✓ (FP, 全 normal)')


print()
print('=' * 70)
print('测试 2: attack burst 内所有 ALARM 全部标 TP (修复核心)')
print('=' * 70)
p = IdsLineParserMirror()
# 填满窗口用 normal
for _ in range(16):
    p.set_last_was_attack(False)
# 现在推 5 帧 attack (burst), 每帧后跟 1 个 ALARM
for i in range(5):
    p.set_last_was_attack(True)
    p.feed(f'[ALARM] {i+1} prob=0.850\n'.encode())

# 期望: 5 个 ALARM 全部标 TP (修复前只有第 1 个是 TP)
assert len(p.fp_history) == 5, f'期望 5 个 fp_history, 实际 {len(p.fp_history)}'
for i, (cnt, is_fp, prob) in enumerate(p.fp_history):
    assert (cnt, is_fp, prob) == (i+1, False, 850), f'第 {i+1} 个 ALARM: 期望 TP, 实际 is_fp={is_fp}'
print(f'  burst 5 帧 attack + 5 alarm → 全部 5 个 TP ✓')
print(f'    修复前: 仅第 1 个 TP, 其余 4 个 FP → 错误!')
print(f'    修复后: 5/5 TP ✓')


print()
print('=' * 70)
print('测试 3: 全 normal 期间 ALARM 标 FP')
print('=' * 70)
p = IdsLineParserMirror()
# 填满窗口用 normal + 持续推 normal
for _ in range(30):
    p.set_last_was_attack(False)
# 3 个 ALARM 都应标 FP
for i in range(3):
    p.feed(f'[ALARM] {i+1} prob=0.450\n'.encode())
assert len(p.fp_history) == 3
for cnt, is_fp, prob in p.fp_history:
    assert is_fp is True, f'ALARM {cnt} 应为 FP (窗口全 normal), 实际 is_fp={is_fp}'
print(f'  30 帧 normal + 3 alarm → 全部 FP ✓')


print()
print('=' * 70)
print('测试 4: 混合场景 (ramp-up → normal → burst → normal)')
print('=' * 70)
p = IdsLineParserMirror()
# 推 5 帧 normal (ramp-up) + 1 alarm (不判定)
for _ in range(5):
    p.set_last_was_attack(False)
p.feed(b'[ALARM] 1 prob=0.500\n')
assert len(p.fp_history) == 0
print(f'  ramp-up 5 帧 + 1 alarm: 不判定 ✓')
# 继续推 11 帧 normal, 窗口填满 (5+11=16)
for _ in range(11):
    p.set_last_was_attack(False)
assert len(p._window) == 16
# burst: 推 3 帧 attack + 3 alarm
for i in range(3):
    p.set_last_was_attack(True)
    p.feed(f'[ALARM] {i+2} prob=0.900\n'.encode())
# 期望: 3 alarm 全 TP
assert len(p.fp_history) == 3
for cnt, is_fp, prob in p.fp_history:
    assert is_fp is False
print(f'  burst 3 帧 attack + 3 alarm: 全部 TP ✓')
# 推 5 帧 normal (滑动窗口逐渐被替换, 但 burst 中至少 1 帧 attack 还在窗内)
# 实际情况: 16 - 5 = 11 帧留原, 5 帧新 push → 总 16 帧
# 5 帧 normal 替换了 burst 中最早 5 帧, 还剩 burst 中 1 帧 attack 在窗内
# 但窗口最大 16, 我们推 5 次会替换最早 5 个. 现在窗内:
# [normal x 5, normal x 8, attack x 3] → 替换后 [normal x 13, attack x 3]
# 但我们要确保 burst 内 attack 在窗里
for _ in range(5):
    p.set_last_was_attack(False)
assert any(p._window), f'期望窗内还有 attack (burst 后 5 帧内), 实际 {list(p._window)}'
print(f'  → burst 后 5 帧 normal, 窗内仍有 attack ✓')
# 再给 1 alarm: 期望 TP
p.feed(b'[ALARM] 5 prob=0.800\n')
assert p.fp_history[-1][1] is False, 'burst 后 alarm 应仍为 TP'
print(f'  burst 后 5 帧 normal + 1 alarm: TP ✓')
# 推 16 帧 normal 完全清掉 attack 痕迹
for _ in range(16):
    p.set_last_was_attack(False)
assert not any(p._window), f'期望窗内全 normal, 实际 {list(p._window)}'
# 1 alarm: FP
p.feed(b'[ALARM] 6 prob=0.400\n')
assert p.fp_history[-1][1] is True, '纯 normal 期 alarm 应 FP'
print(f'  纯 normal 期间 alarm: FP ✓')


print()
print('=' * 70)
print('测试 5: 验证 2026-09-29 auto-40ms 真实日志')
print('=' * 70)
import csv
from collections import Counter

csv_log = r'logs/test_20260929-235212_auto-40ms.csv'
events = []
with open(csv_log, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        events.append(row)

# 用镜像实现重放日志
p = IdsLineParserMirror()
sends = [e for e in events if e['event'] == 'SEND']
alarms_raw = [e for e in events if e['event'] == 'ALARM']

# 时间序列重放
all_events = sorted(events, key=lambda x: int(x['t_event_ms']))
for e in all_events:
    if e['event'] == 'SEND':
        is_attack = (e['binary'] == '1')
        p.set_last_was_attack(is_attack)
    elif e['event'] == 'ALARM':
        p.feed(f'[ALARM] {e["alarm_count"]} prob=0.{e["prob_x1000"]}\n'.encode())

# 对比新旧评估
total_attack = sum(1 for e in sends if e['binary'] == '1')
total_normal = sum(1 for e in sends if e['binary'] == '0')

# 修复后指标
fp_count = sum(1 for _, is_fp, _ in p.fp_history if is_fp)
tp_count = sum(1 for _, is_fp, _ in p.fp_history if not is_fp)
recall = tp_count / total_attack if total_attack else 0
prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else 0
f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
acc = (tp_count + (total_normal - fp_count)) / (tp_count + fp_count + (total_attack - tp_count) + (total_normal - fp_count))
fpr = fp_count / total_normal if total_normal else 0

print(f'  修复后评估 (基于滑动窗口 16 帧):')
print(f'    TP: {tp_count}, FP: {fp_count}')
print(f'    Recall:    {recall*100:.2f}%')
print(f'    Precision: {prec*100:.2f}%')
print(f'    F1:        {f1*100:.2f}%')
print(f'    FPR:       {fpr*100:.2f}%')
print(f'    Accuracy:  {acc*100:.2f}%')
print()
print(f'  修复前 (日志 summary):')
print(f'    TP: 469, FP: 672')
print(f'    Recall: 18.09%, Precision: 41.10%, F1: 25.13%')
print()
print(f'  改善幅度:')
print(f'    Recall:    18.09% → {recall*100:.2f}% (Δ +{recall*100-18.09:.2f}%)')
print(f'    Precision: 41.10% → {prec*100:.2f}% (Δ +{prec*100-41.10:.2f}%)')
print(f'    F1:        25.13% → {f1*100:.2f}% (Δ +{f1*100-25.13:.2f}%)')

# 期望 F1 ~ 51% (与离线分析一致)
assert 0.50 <= f1 <= 0.55, f'F1={f1:.4f} 不在预期范围 [0.50, 0.55], 修复可能有问题'
print()
print('  ✓ F1 在预期范围 [0.50, 0.55], 与离线分析一致')

print()
print('=' * 70)
print('全部测试通过 ✓')
print('=' * 70)