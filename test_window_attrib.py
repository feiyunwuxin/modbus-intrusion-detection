"""
test_window_attrib.py -- 回归测试: 用 GUI 真实的 WindowEvaluator 回放 2026-10-21 日志.

  期望:
    原公式 W=(frame-1)//16            -> 复现 GUI summary (F1 0.8109 / FPR 0.0469)
    新公式 (frame%16!=0 时 W-1)        -> 复现仿真真值   (F1 0.8813 / FPR 0.0227)

用法:  python test_window_attrib.py
"""
import csv
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOG = os.path.join(HERE, "logs", "test_20261021-003650_auto-40ms.csv")
NPZ = os.path.join(HERE, "fullsim_f32.npz")

# 直接从源文件里抠出 WindowEvaluator, 避免 import PyQt5
src = open(os.path.join(HERE, "rtu_replay_gui.py"), encoding="utf-8").read()
start = src.index("class WindowEvaluator:")
end = src.index("\nclass ", start + 10)
ns = {}
exec(compile(src[start:end], "rtu_replay_gui.py", "exec"), ns)
WindowEvaluator = ns["WindowEvaluator"]

d = np.load(NPZ)
probs, atk = d["probs"], d["atk"].astype(bool)
truth_det = set(np.nonzero(probs >= 0.5)[0].tolist())

rows = list(csv.DictReader(open(LOG, encoding="utf-8")))

TRUTH = dict(tp=7098, fp=189, fn=1723, tn=8154)
GUI_MEASURED = dict(tp=6282, fp=391, fn=2539, tn=7952)


def replay(fixed: bool) -> dict:
    """回放日志. fixed=True 时用修正后的归窗公式 (等价于固件修好后的行为)."""
    ev = WindowEvaluator()
    # 若要模拟"原公式", 就把修正那两行去掉 —— 这里直接 monkey-patch 归窗
    if not fixed:
        orig = ev.on_alarm_with_frame

        def legacy(frame_idx: int) -> None:
            W = (frame_idx - 1) // 16
            ev._direct_alarms.add(W)
            for _state in ev._recent_evals:
                if _state["W"] == W and not _state["alarm"]:
                    if _state["attack"]:
                        ev._fn -= 1; ev._tp += 1
                    else:
                        ev._tn -= 1; ev._fp += 1
                    _state["alarm"] = True
                    break
        ev.on_alarm_with_frame = legacy

    for r in rows:
        if r["event"] == "SEND":
            ev.on_frame(r["binary"] == "1")
        elif r["event"] == "ALARM":
            ev.on_alarm_with_frame(int(r["frame_idx"]))
    return ev.get_metrics()


print("=" * 92)
print("回放 logs/test_20261021-003650_auto-40ms.csv (274628 帧 / 17164 窗 / 7287 alarm)")
print("=" * 92)
for fixed in (False, True):
    m = replay(fixed)
    tag = "新公式 (已修正)" if fixed else "原公式 (frame-1)//16"
    print()
    print("  [%s]" % tag)
    print("    n_windows=%d  TP=%5d FP=%4d FN=%5d TN=%5d" % (m["n_windows"], m["tp"], m["fp"], m["fn"], m["tn"]))
    print("    Precision=%.4f Recall=%.4f F1=%.4f FPR=%.4f Accuracy=%.4f"
          % (m["precision"], m["recall"], m["f1"], m["fp"] / max(m["fp"] + m["tn"], 1), m["accuracy"]))
    exp = TRUTH if fixed else GUI_MEASURED
    ok = all(m[k] == v for k, v in exp.items())
    print("    期望 TP=%d FP=%d FN=%d TN=%d  ->  %s"
          % (exp["tp"], exp["fp"], exp["fn"], exp["tn"], "✅ 逐项一致" if ok else "❌ 不一致"))

print()
print("=" * 92)
print("结论")
print("=" * 92)
print("  原公式复现 GUI 实测 summary  -> 证明 GUI 的记账逻辑本身没别的 bug,")
print("                                全部偏差都来自 (frame-1)//16 这一行归窗公式")
print("  新公式复现仿真真值           -> 证明修正后 GUI summary 与模型实际表现一致")
