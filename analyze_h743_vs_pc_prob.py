"""
analyze_h743_vs_pc_prob.py -- 同一 window 上 H743 prob vs PC sim prob 直接比对。

用已保存的 parity_pysim_probs.npz (PC sim 每帧 sliding prob) + H743 csv (alarm prob),
回答核心问题: H743 报出来的 prob 跟 PC sim 在同一 window 算出来的 prob 是否一致?

不一致 → H743 输入数据流 (sw_buf / feature_extractor / scaler) 跟 PC 不一样
一致   → 只是 alarm 上报 / 归并 / 阈值逻辑的差异
"""
import csv
import json
import statistics
from pathlib import Path

import numpy as np

W = 16
NPZ = "parity_pysim_probs.npz"
H743_CSV = r"C:\work\Claude\Issue\logs\test_20261020-215840_auto-40ms.csv"
OUT_JSON = "analyze_h743_vs_pc_prob.json"

print(f"[1] 加载 {NPZ}")
d = np.load(NPZ)
frames = d["frames"]          # (N,) int64   PC sim 推理的 frame_idx (1-based, 相对 START_FRAME)
probs = d["probs"]            # (N,) float32
label_frames = d["label_frames"]
labels = d["labels"]          # 1 = attack
N = len(frames)
print(f"    {N} inferences, frame {frames[0]}..{frames[-1]}")

# frame → prob 查表 (frame_idx → prob)
pc_by_frame = dict(zip(frames.tolist(), probs.tolist()))
# frame → label
lab_by_frame = dict(zip(label_frames.tolist(), labels.tolist()))

n_windows = (int(frames[-1])) // W
print(f"    windows: {n_windows}")

# window label (max-aggregation, 跟训练 yw 一致)
win_attack = []
for w in range(n_windows):
    s = w * W + 1
    e = s + W - 1
    win_attack.append(any(lab_by_frame.get(f, 0) == 1 for f in range(s, e + 1)))
n_atk = sum(win_attack)
n_norm = n_windows - n_atk
print(f"    attack windows={n_atk}  normal windows={n_norm}")

# ---------------------------------------------------------------------------
# 读 H743 alarms
# ---------------------------------------------------------------------------
h743 = {}   # frame_idx → prob
with open(H743_CSV, encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        if r["event"] == "ALARM" and r["frame_idx"] and r["prob_x1000"]:
            h743[int(r["frame_idx"])] = int(r["prob_x1000"]) / 1000.0
print(f"    H743 ALARM rows: {len(h743)}")

# ---------------------------------------------------------------------------
# 对齐策略: H743 上报的 frame=N 归 window (N-1)//16
# PC sim 对应: 该 window 结束帧 16W+16 的 sliding prob (末帧)
#            + 窗口内 16 帧 sliding prob 的 max
# ---------------------------------------------------------------------------
rows = []
for f_idx, h7_p in h743.items():
    w = (f_idx - 1) // W
    if w >= n_windows:
        continue
    # PC sim 在该 window 内的各 sliding prob
    vals = [pc_by_frame.get(f) for f in range(w * W + 1, (w + 1) * W + 1)
            if f in pc_by_frame]
    if not vals:
        continue
    pc_last = pc_by_frame.get((w + 1) * W)       # 末帧
    pc_max = max(vals)
    rows.append({
        "window": w,
        "h743_frame": f_idx,
        "h743_prob": h7_p,
        "pc_last": pc_last,
        "pc_max": pc_max,
        "pc_min": min(vals),
        "label": "atk" if win_attack[w] else "norm",
    })

print()
print(f"[2] 同一 window 的 prob 比对 (n={len(rows)})")

h7 = np.array([r["h743_prob"] for r in rows])
pc_last = np.array([r["pc_last"] if r["pc_last"] is not None else np.nan for r in rows])
pc_max = np.array([r["pc_max"] for r in rows])
pc_min = np.array([r["pc_min"] for r in rows])

for label, name in [("atk", "attack windows"), ("norm", "normal windows")]:
    m = np.array([r["label"] == label for r in rows])
    if m.sum() == 0:
        continue
    print(f"  -- {name} (n={m.sum()}) --")
    print(f"     H743 prob : mean={h7[m].mean():.4f}  median={np.median(h7[m]):.4f}  "
          f"min={h7[m].min():.4f}  max={h7[m].max():.4f}")
    print(f"     PC  last  : mean={np.nanmean(pc_last[m]):.4f}  median={np.nanmedian(pc_last[m]):.4f}")
    print(f"     PC  max   : mean={pc_max[m].mean():.4f}  median={np.median(pc_max[m]):.4f}")
    print(f"     PC  min   : mean={pc_min[m].mean():.4f}  median={np.median(pc_min[m]):.4f}")
    d_last = h7[m] - pc_last[m]
    print(f"     diff (H743 - PC_last): mean={np.nanmean(d_last):+.4f}  "
          f"median={np.nanmedian(d_last):+.4f}  |d|>0.1 占比={np.mean(np.abs(d_last)>0.1):.3f}")
    d_max = h7[m] - pc_max[m]
    print(f"     diff (H743 - PC_max) : mean={np.nanmean(d_max):+.4f}  "
          f"median={np.nanmedian(d_max):+.4f}  |d|>0.1 占比={np.mean(np.abs(d_max)>0.1):.3f}")

# ---------------------------------------------------------------------------
# 整体: PC sim 在每个 non-overlap window 末帧的 prob vs ground truth
# 用多种归并策略重算 PC 的 TP/FP/FN, 看哪一 strat 能复现 H743 summary
# ---------------------------------------------------------------------------
print()
print(f"[3] 用不同归并策略重算 PC sim 指标, 看哪一 strat 接近 H743 summary")
print(f"    (H743 summary: TP=1482, FP=58, FN=933, Recall=61.37%, FPR=2.58%)")

strats = {}
# A: 末帧 sliding prob
strats["A_last_frame"] = {}
# B: window 内 max
strats["B_window_max"] = {}
# C: 只看 H743 上报 frame 严格同窗 (frame == 16W+16)
strats["C_exact_end_frame"] = {}

for w in range(n_windows):
    vals = [pc_by_frame.get(f) for f in range(w * W + 1, (w + 1) * W + 1) if f in pc_by_frame]
    if not vals:
        continue
    strats["A_last_frame"][w] = pc_by_frame.get((w + 1) * W)
    strats["B_window_max"][w] = max(vals)
    strats["C_exact_end_frame"][w] = pc_by_frame.get((w + 1) * W)

for name, tab in strats.items():
    for thr in (0.36, 0.5):
        tp = fp = fn = 0
        for w in range(n_windows):
            p = tab.get(w)
            if p is None:
                continue
            det = p >= thr
            if win_attack[w]:
                if det:
                    tp += 1
                else:
                    fn += 1
            else:
                if det:
                    fp += 1
        R = tp / max(n_atk, 1)
        P = tp / max(tp + fp, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        FPR = fp / max(n_norm, 1)
        print(f"    {name:20s} thr={thr:.2f}  TP={tp:5d} FP={fp:5d} FN={fn:5d}  "
              f"P={P:.4f} R={R:.4f} F1={F1:.4f} FPR={FPR:.4f}")

# ---------------------------------------------------------------------------
# 反向: H743 报的 1540 window 里, PC 在同 window 的 prob 是多少
# 如果 PC 在这些 window 也都 ≥ thr, 说明 H743 只是报得少 (输入更钝)
# ---------------------------------------------------------------------------
print()
print(f"[4] H743 报出的 {len(rows)} 个 window, PC sim 在同 window 的判定")
for thr in (0.36, 0.5):
    same = sum(1 for r in rows if r["pc_last"] is not None and r["pc_last"] >= thr)
    print(f"    thr={thr:.2f}:  H743 报 & PC 末帧也报 = {same}/{len(rows)}  "
          f"({same/max(len(rows),1):.3f})")
    same_m = sum(1 for r in rows if r["pc_max"] >= thr)
    print(f"    thr={thr:.2f}:  H743 报 & PC 窗口max也报 = {same_m}/{len(rows)}  "
          f"({same_m/max(len(rows),1):.3f})")

# ---------------------------------------------------------------------------
# PC sim 报但 H743 没报 的 window (H743 漏报) 的 PC prob 分布
# ---------------------------------------------------------------------------
h743_windows = set((f - 1) // W for f in h743.keys())
missed = []
for w in range(n_windows):
    if w in h743_windows:
        continue
    p = pc_by_frame.get((w + 1) * W)
    if p is None:
        continue
    missed.append((w, p, win_attack[w]))

atk_missed = [(w, p) for w, p, a in missed if a]
print()
print(f"[5] H743 没报但 PC 末帧 prob ≥ 0.5 的 window")
hi = [(w, p) for w, p in atk_missed if p >= 0.5]
print(f"    attack windows 里 PC≥0.5 但 H743 没报: {len(hi)}")
if hi:
    ps = np.array([p for _, p in hi])
    print(f"    它们的 PC prob: median={np.median(ps):.4f}  max={ps.max():.4f}")
    print(f"    前 10 个: {[(w, round(p,3)) for w,p in hi[:10]]}")

out = {
    "n_h743_alarms": len(rows),
    "attack_windows": n_atk,
    "normal_windows": n_norm,
    "h743_atk_alarms": int(sum(1 for r in rows if r["label"] == "atk")),
    "h743_norm_alarms": int(sum(1 for r in rows if r["label"] == "norm")),
    "h743_prob_mean": float(h7.mean()),
    "pc_last_mean": float(np.nanmean(pc_last)),
    "pc_max_mean": float(pc_max.mean()),
    "diff_last_median": float(np.nanmedian(h7 - pc_last)),
    "diff_max_median": float(np.median(h7 - pc_max)),
    "pct_absdiff_gt_0_1_last": float(np.mean(np.abs(h7 - pc_last) > 0.1)),
    "h743_reported_pc_also_reported_at_0_5": int(
        sum(1 for r in rows if r["pc_last"] is not None and r["pc_last"] >= 0.5)),
    "atk_missed_by_h743_but_pc_high": len(hi),
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[6] saved {OUT_JSON}")
