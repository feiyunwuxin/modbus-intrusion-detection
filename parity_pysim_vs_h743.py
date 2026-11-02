"""
parity_pysim_vs_h743.py -- 跑 PC sim (PyTorch) 跟 H743 实测 log 对比。

目的: 把当前 933 FN (Recall 61.37% 跟 PC sim 95% 差 33.6pp) 拆分
      - 真 FN: attack window 但 PC PyTorch 也给 prob < 0.5 → 数据 / label 问题
      - 假 FN: attack window 但 PC PyTorch 给 prob ≥ 0.5 → H743 推理出问题

跟 parity_ch32_vs_onnx.py (controlled input) 不同,这里跑真实数据。
"""
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import torch

# 跟 ids_pipeline.py 一致
SW_WINDOW = 16
CLIP_VAL = 10.0
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]
UNUSUAL_FCS = frozenset({136, 171, 139, 133, 137, 138, 140})

CSV_PATH = r"C:\work\Claude\Issue\IanArffDataset.csv"
PT_PATH = "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"
SCALER_PATH = "scaler_binary_v2_scada.joblib"
H743_CSV = r"C:\work\Claude\Issue\logs\test_20261020-215840_auto-40ms.csv"
START_FRAME = 200000   # 跟 auto-40ms 测试对齐
N_FRAMES = 74628        # 跟 csv frame range 一致
W = 16

# ---------------------------------------------------------------------------
# Feature extractor (跟 ids_pipeline.featurize_one bit-perfect)
# ---------------------------------------------------------------------------
def _num(v, default=0.0):
    if v is None or v == "?":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _int(v, default=0):
    if v is None or v == "?":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


class FeatState:
    def __init__(self):
        self.last_seen = {}
        self.prev_time = None


def featurize_one(state, row):
    t = _num(row.get("time"))
    dt = 0.0 if state.prev_time is None else t - state.prev_time
    addr_i = _int(row.get("address"))
    fc_i = _int(row.get("function"))
    k = (addr_i, fc_i)
    tsl = 0.0 if k not in state.last_seen else t - state.last_seen[k]
    state.prev_time = t
    state.last_seen[k] = t

    is_unusual_fc = 1.0 if fc_i in UNUSUAL_FCS else 0.0
    is_response = float(_int(row.get("command response")))
    pressure = max(0.0, min(100.0, _num(row.get("pressure measurement"))))

    # Order MUST match ids_pipeline.featurize_one (跟 build_18_raw_from_csv 一致)
    vec = np.array([
        addr_i,                                              # 0  address
        fc_i,                                                # 1  function
        _int(row.get("length")),                              # 2  length
        _num(row.get("gain")),                                # 3  gain
        _num(row.get("reset rate")),                          # 4  reset rate
        _num(row.get("deadband")),                            # 5  deadband
        _num(row.get("cycle time")),                          # 6  cycle time
        _num(row.get("rate")),                                # 7  rate
        _int(row.get("system mode")),                         # 8  system mode
        _int(row.get("control scheme")),                      # 9  control scheme
        _int(row.get("pump")),                                # 10 pump
        _int(row.get("solenoid")),                            # 11 solenoid
        pressure,                                             # 12 pressure
        _num(row.get("crc rate")),                            # 13 crc rate
        dt,                                                   # 14 dt (time_diff)
        tsl,                                                  # 15 tsl
        is_unusual_fc,                                        # 16 is_unusual_fc
        is_response,                                          # 17 is_response
    ], dtype=np.float32)
    return vec  # 18-dim


def build_window_input(win18_stack):
    """win18_stack: (16, 18) np array normalized."""
    press_sum = float(win18_stack[:, 12].sum())  # pressure idx
    crc_max = float(win18_stack[:, 13].max())    # crc rate idx
    resp_signed = float(win18_stack[:, 17].sum()) # is_response idx
    unusual_sign = float(win18_stack[:, 16].sum()) # is_unusual_fc idx

    seen = set()
    n_distinct = 0
    for v in win18_stack[:, 2]:  # length idx
        k = int(round(float(v)))
        if k not in seen:
            seen.add(k)
            n_distinct += 1

    press_mean = press_sum / SW_WINDOW
    balance = 1.0 - resp_signed / 8.0
    crc_mean = float(win18_stack[:, 12].mean())
    cmd_count = float(SW_WINDOW - unusual_sign - resp_signed)

    frame27 = np.zeros((SW_WINDOW, 27), dtype=np.float32)
    for c18 in range(18):
        frame27[:, KEEP_18[c18]] = win18_stack[:, c18]
    frame27[:, 19] = press_mean
    frame27[:, 20] = crc_mean
    frame27[:, 21] = crc_max
    frame27[:, 22] = cmd_count
    frame27[:, 23] = resp_signed
    frame27[:, 24] = balance
    frame27[:, 25] = n_distinct
    frame27[:, 26] = unusual_sign

    sliced = frame27[:, KEEP_23]
    clipped = np.clip(sliced, -CLIP_VAL, CLIP_VAL)
    return clipped.T.astype(np.float32)  # (23, 16)


# ---------------------------------------------------------------------------
# 加载 artifacts
# ---------------------------------------------------------------------------
print(f"[1] 加载 scaler: {SCALER_PATH}")
sc = joblib.load(SCALER_PATH)
center = np.asarray(sc.center_, dtype=np.float32)
scale = np.asarray(sc.scale_, dtype=np.float32)
assert center.shape == (19,) and scale.shape == (19,)
keep18 = np.asarray(KEEP_18, dtype=np.int64)
mu_18 = center[keep18]
sg_18 = scale[keep18]
print(f"    mu_18 shape={mu_18.shape}  sg_18 shape={sg_18.shape}")

print(f"[2] 加载模型: {PT_PATH}")
ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
arch_kwargs = dict(
    n_features=ckpt["n_features"],
    window=ckpt["window"],
    n_blocks=ckpt["n_blocks"],
    channels=ckpt["channels"],
    kernel_size=ckpt["kernel_size"],
    dilations=tuple(ckpt["dilations"]),
    se_reduction=ckpt["se_reduction"],
    dropout=ckpt["dropout"],
)
print(f"    arch: {arch_kwargs}")

# 用 ids_worker_torch 里的 build_tcn_se_model
sys.path.insert(0, str(Path("modbus_replay_simulator").resolve()))
from modbus_replay.ids_worker_torch import build_tcn_se_model
# build_tcn_se_model 用 in_ch 不是 n_features
model_kwargs = dict(
    in_ch=arch_kwargs["n_features"],
    n_blocks=arch_kwargs["n_blocks"],
    channels=arch_kwargs["channels"],
    kernel_size=arch_kwargs["kernel_size"],
    dilations=arch_kwargs["dilations"],
    dropout=arch_kwargs["dropout"],
    se_reduction=arch_kwargs["se_reduction"],
)
model = build_tcn_se_model(**model_kwargs)
model.load_state_dict(sd)
model.eval()
print(f"    ok, params={sum(t.numel() for t in sd.values())}")

# ---------------------------------------------------------------------------
# 加载 csv → 取 START_FRAME 之后 N_FRAMES 行
# ---------------------------------------------------------------------------
print(f"[3] 读 csv (从 START_FRAME={START_FRAME} 取 {N_FRAMES} 行)")
state = FeatState()
window = deque(maxlen=SW_WINDOW)
probs_per_frame = []   # (frame_idx, prob)
labels_per_frame = []  # (frame_idx, is_attack)

t0 = time.time()
with open(CSV_PATH, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for i, row in enumerate(rdr, start=1):
        if i <= START_FRAME:
            continue
        if i > START_FRAME + N_FRAMES:
            break
        # 18-dim raw
        feat18 = featurize_one(state, row)
        norm18 = (feat18 - mu_18) / sg_18
        window.append(norm18)

        is_attack = int(_num(row.get("binary result"))) == 1
        labels_per_frame.append((i - START_FRAME, is_attack))

        if len(window) < SW_WINDOW:
            continue

        win18_stack = np.stack(list(window), axis=0)  # (16, 18)
        x23 = build_window_input(win18_stack)         # (23, 16)
        xt = torch.from_numpy(x23).float().unsqueeze(0)
        with torch.no_grad():
            logit = float(model(xt).item())
        prob = 1.0 / (1.0 + math.exp(-logit))
        probs_per_frame.append((i - START_FRAME, prob))

elapsed = time.time() - t0
print(f"    done {len(probs_per_frame)} inferences in {elapsed:.1f}s ({len(probs_per_frame)/elapsed:.0f} inf/s)")

# ---------------------------------------------------------------------------
# 跟 H743 csv 对比 (用 window=16 max-aggregation 跟 summary 一致)
# ---------------------------------------------------------------------------
print()
print(f"[4] per-window max-aggregation 跟 H743 summary 对比")
n_windows = N_FRAMES // W
window_attack = []
for w in range(n_windows):
    s = w * W + 1
    e = s + W - 1
    is_atk = any(lab for f, lab in labels_per_frame if s <= f <= e)
    window_attack.append(is_atk)
n_atk = sum(window_attack)
n_norm = n_windows - n_atk
print(f"    Attack windows={n_atk}  Normal windows={n_norm}")

# 读 H743 csv 的 alarm (保留原始行,后面按 window 归并)
h743_rows = []
with open(H743_CSV, encoding="utf-8", errors="replace") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        if r["event"] == "ALARM" and r["frame_idx"] and r["prob_x1000"]:
            h743_rows.append(r)
print(f"    H743 ALARM rows: {len(h743_rows)}")

# ---------------------------------------------------------------------------
# 跟 H743 csv 对比
#
# ⚠ 关键: PC sim / H743 都是 SLIDING window (每帧推理一次, deque maxlen=16),
#    74613 次推理。评估 ground truth 是 non-overlap 16 帧 window
#    (yw = max(y_bin in window=16), 跟训练一致)。
#    ⇒ 一个 window 被"检出" = 该 window 内 16 帧的 sliding prob 全部取 max ≥ thr。
#    之前用"只取末帧 prob"是错的, 会同时高估 FP 和 FN。
# ---------------------------------------------------------------------------
def window_max_prob(prob_by_frame, w_idx, W):
    """该 non-overlap window 内所有 sliding-window 推理的 prob 最大值。"""
    s = w_idx * W + 1
    e = s + W - 1
    vals = [p for f, p in prob_by_frame if s <= f <= e]
    return max(vals) if vals else None


prob_by_frame = probs_per_frame

# PC sim 用 max-aggregation
tp_pc = []
fn_pc = []
fp_pc = []
pc_win_max = {}
for w_idx in range(n_windows):
    mp = window_max_prob(prob_by_frame, w_idx, W)
    if mp is None:
        continue
    pc_win_max[w_idx] = mp
    if window_attack[w_idx]:
        (tp_pc if mp >= 0.5 else fn_pc).append(mp)
    else:
        if mp >= 0.5:
            fp_pc.append(mp)

print()
print(f"[5] PC sim 结果 (thr=0.5, sliding-window max-aggregation)")
print(f"    TP (atk, PC≥0.5):  {len(tp_pc):5d}  / attack windows {n_atk}  Recall={len(tp_pc)/max(n_atk,1):.4f}")
print(f"    FN (atk, PC<0.5):  {len(fn_pc):5d}")
print(f"    FP (norm, PC≥0.5): {len(fp_pc):5d}  / normal windows {n_norm}  FPR={len(fp_pc)/max(n_norm,1):.4f}")

# 跟 H743 真阳/假阳/假阴 对比
# H743 alarm 也是 per-frame sliding → 同样要按 window 归并
h743_win_alarm = set()
h743_win_max = {}
for r in h743_rows:
    f_idx = int(r["frame_idx"])
    w_idx = (f_idx - 1) // W
    h743_win_alarm.add(w_idx)
    p = int(r["prob_x1000"]) / 1000.0
    h743_win_max[w_idx] = max(h743_win_max.get(w_idx, 0.0), p)

h743_tp_w = {w for w in h743_win_alarm if w < n_windows and window_attack[w]}
h743_fn_w = [w for w in range(n_windows) if window_attack[w] and w not in h743_tp_w]
h743_fp_w = [w for w in h743_win_alarm if w < n_windows and not window_attack[w]]

print()
print(f"[6] H743 跟 PC sim 真因拆分")
print(f"    H743 检出 windows: {len(h743_tp_w) + len(h743_fp_w)}  "
      f"(TP={len(h743_tp_w)}  FP={len(h743_fp_w)})")
print(f"    H743 FN windows: {len(h743_fn_w)}")
fn_with_pc_high = sum(1 for w in h743_fn_w if pc_win_max.get(w, 0.0) >= 0.5)
fn_with_pc_low = sum(1 for w in h743_fn_w if pc_win_max.get(w, 0.0) < 0.5)
print(f"      其中 PC PyTorch window-max prob ≥ 0.5: {fn_with_pc_high}  → H743 推理/数据流问题")
print(f"      其中 PC PyTorch window-max prob <  0.5: {fn_with_pc_low}   → 数据/label 本身难,H743 跟 PC 一致")
print()
print(f"    H743 FP windows: {len(h743_fp_w)}")
fp_with_pc_high = sum(1 for w in h743_fp_w if pc_win_max.get(w, 0.0) >= 0.5)
fp_with_pc_low = sum(1 for w in h743_fp_w if pc_win_max.get(w, 0.0) < 0.5)
print(f"      其中 PC PyTorch window-max prob ≥ 0.5: {fp_with_pc_high}  → PC 跟 H743 都误报 (模型层 FP)")
print(f"      其中 PC PyTorch window-max prob <  0.5: {fp_with_pc_low}   → PC 不报 H743 报 (H743 数据流 drift)")

# H743 summary 的原始 FP=58 / TP=1482 是 PC 端 per-window 归并后的数,对照一下
print()
print(f"[7] 对照 H743 summary (TP=1482, FP=58, FN=933):")
print(f"    本脚本按 frame→window 归并得到 TP={len(h743_tp_w)} FP={len(h743_fp_w)}")
print(f"    (差异来源: summary 用的是 PC 端 WindowEvaluator 的 lag 补偿逻辑)")

# 10 个 H743 FN window 的 PC prob 详情
print()
print(f"[8] 10 个 H743 FN window 的 PC window-max prob 详情:")
for w_idx in h743_fn_w[:10]:
    f_start = w_idx * W + 1
    f_end = (w_idx + 1) * W
    mp = pc_win_max.get(w_idx, None)
    atk_count = sum(1 for f, lab in labels_per_frame if f_start <= f <= f_end and lab)
    print(f"    w#{w_idx:4d}  frames {f_start:5d}..{f_end:5d}  attack_frames={atk_count}/16  "
          f"PC win-max prob={mp:.3f}" if mp is not None else f"    w#{w_idx}  no PC prob")

# 写出去供后续比对
print()
print(f"[8] 保存对比结果到 parity_pysim_vs_h743.json")
import json
out = {
    "csv": CSV_PATH,
    "start_frame": START_FRAME,
    "n_frames": N_FRAMES,
    "pc_inferences": len(probs_per_frame),
    "elapsed_s": elapsed,
    "attack_windows": n_atk,
    "normal_windows": n_norm,
    "pc": {
        "TP": len(tp_pc),
        "FN": len(fn_pc),
        "FP": len(fp_pc),
        "recall_at_0_5": len(tp_pc) / max(n_atk, 1),
        "fpr_at_0_5": len(fp_pc) / max(n_norm, 1),
    },
    "h743": {
        "alarm_rows": len(h743_rows),
        "detected_windows": len(h743_tp_w) + len(h743_fp_w),
        "TP_windows": len(h743_tp_w),
        "FP_windows": len(h743_fp_w),
        "FN_windows": len(h743_fn_w),
        "FN_with_PC_high_prob": fn_with_pc_high,
        "FN_with_PC_low_prob": fn_with_pc_low,
        "FP_with_PC_high_prob": fp_with_pc_high,
        "FP_with_PC_low_prob": fp_with_pc_low,
    },
}
with open("parity_pysim_vs_h743.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"    saved")

# 存 per-frame prob 供后续分析 (免得重跑 213s)
np.savez_compressed(
    "parity_pysim_probs.npz",
    frames=np.array([f for f, _ in probs_per_frame], dtype=np.int64),
    probs=np.array([p for _, p in probs_per_frame], dtype=np.float32),
    label_frames=np.array([f for f, _ in labels_per_frame], dtype=np.int64),
    labels=np.array([l for _, l in labels_per_frame], dtype=np.int8),
)
print(f"    saved parity_pysim_probs.npz")
