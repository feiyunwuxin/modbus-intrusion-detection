"""
h743_windows.py -- 从 IanArffDataset.csv 重建 23-dim model 输入 window.

两套输入:
  x_pc : PC 视角 (直读 CSV, 跟 ids_pipeline.featurize_one 一致)
  x_h7 : H743 视角 (PC 编码 -> MCU raw17_decode -> feature_extractor -> sliding_window)

window 语义 = H743 的 non-overlap: 每 16 帧一个 window, 窗口 = 最近 16 帧.
"""
import csv
from collections import deque

import joblib
import numpy as np

from h743_view_sim import (CSV_PATH, CLIP_VAL, KEEP_18, KEEP_23, MCU_UNUSUAL,
                           SCALER, START, St, build23, featurize_pc, mcu_decode,
                           pc_encode)


def build_windows(start=START, n=74628, phase=0):
    """phase: 窗口结束帧在 (start + k) 上的同余相位.
    phase=0 → 部署端 (H743 从 replay 起点每 16 帧一个窗);
    phase=5 → 官方训练管线的 test 网格 (test split 起点 219701 % 16 = 5)."""
    sc = joblib.load(SCALER)
    mu18 = np.asarray(sc.center_, dtype=np.float32)[np.asarray(KEEP_18)]
    sg18 = np.asarray(sc.scale_, dtype=np.float32)[np.asarray(KEEP_18)]

    st = St()
    lru = {}
    last_ts = None
    first = True
    win_pc = deque(maxlen=16)
    win_h7 = deque(maxlen=16)
    wins = []
    labels = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if i <= start:
                continue
            if i > start + n:
                break
            fidx = i - start

            raw_pc = featurize_pc(st, row)
            win_pc.append((raw_pc - mu18) / sg18)

            dec = mcu_decode(pc_encode(row))
            ts = dec["time"]
            dt = 0.0 if first else float(np.float32(ts - last_ts))
            first = False
            last_ts = ts
            k = (dec["address"], dec["function"])
            tsl = 0.0 if k not in lru else float(np.float32(ts - lru[k]))
            lru[k] = ts
            raw_h7 = np.array([
                dec["address"], dec["function"], dec["length"],
                dec["gain"], dec["reset"], dec["deadband"], dec["cycle"], dec["rate"],
                dec["system_mode"], dec["control_scheme"], dec["pump"], dec["solenoid"],
                min(100.0, max(0.0, dec["pressure"])), dec["crc"], dt, tsl,
                1.0 if dec["function"] in MCU_UNUSUAL else 0.0,
                float(dec["is_response"])], dtype=np.float32)
            win_h7.append((raw_h7 - mu18) / sg18)
            labels.append(int(float(row.get("binary result") or 0)))

            if len(win_pc) < 16:
                continue
            if fidx % 16 == phase:
                wins.append((fidx, build23(np.stack(list(win_pc), 0)),
                             build23(np.stack(list(win_h7), 0))))
    return wins, labels


def window_labels(wins, labels, W=16):
    """per-window 攻击标签 (16 帧任一帧 attack)."""
    lab = {i + 1: v for i, v in enumerate(labels)}
    out = []
    for (fidx, _, _) in wins:
        s = fidx - W + 1
        out.append(any(lab.get(f, 0) for f in range(s, fidx + 1)))
    return out


if __name__ == "__main__":
    wins, labels = build_windows()
    atk = window_labels(wins, labels)
    print("windows=%d attack=%d normal=%d" % (len(wins), sum(atk), len(atk) - sum(atk)))
