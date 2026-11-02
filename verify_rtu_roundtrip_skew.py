"""
verify_rtu_roundtrip_skew.py -- 验证 train-serving skew 假设。

假设: H743 的 18-dim raw 跟 PC sim 的 18-dim raw 不同, 因为
      PC sim (ids_worker.featurize_one) 直接读 CSV 列 (float64 原值),
      而 H743 从 RTU 帧解出 (rtu_replay_gui.build_raw17_frame 编码 →
      modbus_rtu.c raw17_decode 解码), length=16 时 10 个特征被量化成 uint8.

本脚本在 PC 侧模拟两种 18-dim:
  A) "PC 视角"  = 直接读 CSV 列 (现状, 跟 ids_worker 一致)
  B) "H743 视角" = 先过 RTU round-trip (编码 + 解码), 再喂模型
然后:
  1. 比对两种 18-dim 的逐通道差异
  2. 跑同一个模型, 比对两种 prob
  3. 跟 H743 实测 1546 个 alarm prob 对比 → 看哪一版更接近
"""
import csv
import math
import struct
import sys
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import torch

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
START_FRAME = 200000
N_FRAMES = 74628
W = 16

RAW17_FEAT_COLS = (
    'setpoint', 'gain', 'reset rate', 'deadband',
    'cycle time', 'rate', 'system mode', 'control scheme',
    'pump', 'solenoid', 'pressure measurement',
)


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


def _safe_uint(v, default=0, vmin=0, vmax=255):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(vmin, min(vmax, int(f)))


# ---------------------------------------------------------------------------
# RTU round-trip 模拟 (严格照抄 rtu_replay_gui.build_raw17_frame + MCU modbus_rtu.c)
# ---------------------------------------------------------------------------
def rtu_roundtrip_raw(row):
    """模拟 PC 编码 → H743 解码, 返回 H743 视角的 18-dim raw (未归一化)."""
    length = _safe_uint(row.get('length'), default=16, vmin=0, vmax=255)
    feat_vals = [_num(row.get(c)) for c in RAW17_FEAT_COLS]
    crc_raw = _num(row.get('crc rate'))
    crc_u16 = max(0, min(65535, int(round(crc_raw))))

    # --- 编码 (build_raw17_frame) ---
    if length == 16:
        body = bytearray()
        for v in feat_vals[:10]:
            body.append(max(0, min(255, int(round(v)))))
        body += struct.pack('>f', float(feat_vals[10]))
        body += struct.pack('>H', crc_u16)
    elif length == 46:
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>f', float(v))
        body += struct.pack('>H', crc_u16)
    elif length == 90:
        body = bytearray()
        for v in feat_vals:
            body += struct.pack('>d', float(v))
        body += struct.pack('>H', crc_u16)
    else:
        body = bytearray(11)
        body += struct.pack('>H', crc_u16)

    # --- 解码 (MCU modbus_rtu.c raw17_decode) ---
    out = dict.fromkeys(
        ('setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate',
         'system_mode', 'control_scheme', 'pump', 'solenoid', 'pressure', 'crc'), 0.0)
    if length == 16:
        out['gain'] = float(body[1])
        out['reset'] = float(body[2])
        out['deadband'] = float(body[3])
        out['cycle'] = float(body[4])
        out['rate'] = float(body[5])
        out['system_mode'] = float(body[6])
        out['control_scheme'] = float(body[7])
        out['pump'] = float(body[8])
        out['solenoid'] = float(body[9])
        out['pressure'] = struct.unpack_from('>f', body, 10)[0]
        out['crc'] = float(struct.unpack_from('>H', body, 14)[0])
    elif length == 46:
        out['gain'] = struct.unpack_from('>f', body, 4)[0]
        out['reset'] = struct.unpack_from('>f', body, 8)[0]
        out['deadband'] = struct.unpack_from('>f', body, 12)[0]
        out['cycle'] = struct.unpack_from('>f', body, 16)[0]
        out['rate'] = struct.unpack_from('>f', body, 20)[0]
        out['system_mode'] = struct.unpack_from('>f', body, 24)[0]
        out['control_scheme'] = struct.unpack_from('>f', body, 28)[0]
        out['pump'] = struct.unpack_from('>f', body, 32)[0]
        out['solenoid'] = struct.unpack_from('>f', body, 36)[0]
        out['pressure'] = struct.unpack_from('>f', body, 40)[0]
        out['crc'] = float(struct.unpack_from('>H', body, 44)[0])
    elif length == 90:
        out['gain'] = struct.unpack_from('>d', body, 8)[0]
        out['reset'] = struct.unpack_from('>d', body, 16)[0]
        out['deadband'] = struct.unpack_from('>d', body, 24)[0]
        out['cycle'] = struct.unpack_from('>d', body, 32)[0]
        out['rate'] = struct.unpack_from('>d', body, 40)[0]
        out['system_mode'] = struct.unpack_from('>d', body, 48)[0]
        out['control_scheme'] = struct.unpack_from('>d', body, 56)[0]
        out['pump'] = struct.unpack_from('>d', body, 64)[0]
        out['solenoid'] = struct.unpack_from('>d', body, 72)[0]
        out['pressure'] = struct.unpack_from('>d', body, 80)[0]
        out['crc'] = float(struct.unpack_from('>H', body, 88)[0])
    else:
        out['crc'] = float(struct.unpack_from('>H', body, 11)[0])

    return out


class FeatState:
    def __init__(self):
        self.last_seen = {}
        self.prev_time = None


def featurize_one(state, row, use_rtu):
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
    length = _int(row.get("length"))
    crc = _num(row.get("crc rate"))

    if use_rtu:
        rt = rtu_roundtrip_raw(row)
        gain = rt['gain']
        reset = rt['reset']
        deadband = rt['deadband']
        cycle = rt['cycle']
        rate = rt['rate']
        system_mode = rt['system_mode']
        control_scheme = rt['control_scheme']
        pump = rt['pump']
        solenoid = rt['solenoid']
        pressure = rt['pressure']
        crc = rt['crc']
    else:
        gain = _num(row.get("gain"))
        reset = _num(row.get("reset rate"))
        deadband = _num(row.get("deadband"))
        cycle = _num(row.get("cycle time"))
        rate = _num(row.get("rate"))
        system_mode = _num(row.get("system mode"))
        control_scheme = _num(row.get("control scheme"))
        pump = _num(row.get("pump"))
        solenoid = _num(row.get("solenoid"))

    return np.array([
        addr_i, fc_i, length,
        gain, reset, deadband, cycle, rate,
        system_mode, control_scheme, pump, solenoid,
        pressure, crc, dt, tsl, is_unusual_fc, is_response,
    ], dtype=np.float32)


def build_window_input(win18_stack):
    press_sum = float(win18_stack[:, 12].sum())
    crc_max = float(win18_stack[:, 13].max())
    resp_signed = float(win18_stack[:, 17].sum())
    unusual_sign = float(win18_stack[:, 16].sum())
    seen = set()
    n_distinct = 0
    for v in win18_stack[:, 2]:
        k = int(round(float(v)))
        if k not in seen:
            seen.add(k)
            n_distinct += 1
    press_mean = press_sum / SW_WINDOW
    balance = 1.0 - resp_signed / 8.0
    crc_mean = float(win18_stack[:, 13].mean())
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
    return np.clip(sliced, -CLIP_VAL, CLIP_VAL).T.astype(np.float32)


# ---------------------------------------------------------------------------
print("[1] 加载 artifacts")
sc = joblib.load(SCALER_PATH)
center = np.asarray(sc.center_, dtype=np.float32)
scale = np.asarray(sc.scale_, dtype=np.float32)
keep18 = np.asarray(KEEP_18, dtype=np.int64)
mu_18, sg_18 = center[keep18], scale[keep18]

ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
sys.path.insert(0, str(Path("modbus_replay_simulator").resolve()))
from modbus_replay.ids_worker_torch import build_tcn_se_model
model = build_tcn_se_model(
    in_ch=ckpt["n_features"], n_blocks=ckpt["n_blocks"], channels=ckpt["channels"],
    kernel_size=ckpt["kernel_size"], dilations=tuple(ckpt["dilations"]),
    dropout=ckpt["dropout"], se_reduction=ckpt["se_reduction"])
model.load_state_dict(sd)
model.eval()

# ---------------------------------------------------------------------------
print("[2] 读 csv, 同时算 PC 视角 / H743 视角 两套 18-dim")
state_pc = FeatState()
state_h7 = FeatState()
win_pc = deque(maxlen=SW_WINDOW)
win_h7 = deque(maxlen=SW_WINDOW)
probs_pc = []
probs_h7 = []
raw_pc_ch = []
raw_h7_ch = []
labels = []
diff_acc = np.zeros(18, dtype=np.int64)

t0 = time.time()
with open(CSV_PATH, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for i, row in enumerate(rdr, start=1):
        if i <= START_FRAME:
            continue
        if i > START_FRAME + N_FRAMES:
            break
        fidx = i - START_FRAME

        raw_pc = featurize_one(state_pc, row, use_rtu=False)
        raw_h7 = featurize_one(state_h7, row, use_rtu=True)
        diff_acc += (raw_pc != raw_h7)

        norm_pc = (raw_pc - mu_18) / sg_18
        norm_h7 = (raw_h7 - mu_18) / sg_18
        win_pc.append(norm_pc)
        win_h7.append(norm_h7)
        labels.append(int(_num(row.get("binary result"))) == 1)

        if len(win_pc) < SW_WINDOW:
            continue
        for tag, win, dest in (("pc", win_pc, probs_pc), ("h7", win_h7, probs_h7)):
            x = build_window_input(np.stack(list(win), axis=0))
            with torch.no_grad():
                logit = float(model(torch.from_numpy(x).float().unsqueeze(0)).item())
            dest.append((fidx, 1.0 / (1.0 + math.exp(-logit))))

print(f"    done in {time.time()-t0:.1f}s  ({len(probs_pc)} inferences x2)")

print()
print("[3] 18-dim raw 逐通道差异 (PC 直读 CSV vs H743 RTU round-trip)")
NAMES = ['address', 'function', 'length', 'gain', 'reset_rate', 'deadband',
         'cycle_time', 'rate', 'system_mode', 'control_scheme', 'pump', 'solenoid',
         'pressure', 'crc_rate', 'time_diff', 'time_since', 'is_unusual_fc', 'is_response']
for c in range(18):
    pct = diff_acc[c] / N_FRAMES * 100
    flag = '  <<<< 差异' if pct > 0.5 else ''
    print(f"  ch{c:2d} {NAMES[c]:18s} 不同帧数 {diff_acc[c]:7d} ({pct:6.2f}%){flag}")

# ---------------------------------------------------------------------------
print()
print("[4] prob 对比 + 跟 H743 实测比对")
pc_by_frame = dict(probs_pc)
h7_by_frame = dict(probs_h7)

h743 = {}
with open(H743_CSV, encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        if r["event"] == "ALARM" and r["frame_idx"] and r["prob_x1000"]:
            h743[int(r["frame_idx"])] = int(r["prob_x1000"]) / 1000.0

h7_p, pc_p = [], []
for f_idx, h743_p in h743.items():
    w = (f_idx - 1) // W
    pcv = pc_by_frame.get((w + 1) * W)
    h7v = h7_by_frame.get((w + 1) * W)
    if pcv is None or h7v is None:
        continue
    pc_p.append(pcv)
    h7_p.append(h7v)

h7_p = np.array(h7_p)
pc_p = np.array(pc_p)
h743_a = np.array([h743[f] for f in h743 if (f - 1) // W * W + W in pc_by_frame])

print(f"  H743 实测 alarm prob : mean={h743_a.mean():.4f}  median={np.median(h743_a):.4f}")
print(f"  PC 视角   同 window  : mean={pc_p.mean():.4f}  median={np.median(pc_p):.4f}  "
      f"diff vs H743 median={np.median(h743_a)-np.median(pc_p):+.4f}")
print(f"  H743视角  同 window  : mean={h7_p.mean():.4f}  median={np.median(h7_p):.4f}  "
      f"diff vs H743 median={np.median(h743_a)-np.median(h7_p):+.4f}")

# MAE
mae_pc = np.mean(np.abs(h743_a - pc_p))
mae_h7 = np.mean(np.abs(h743_a - h7_p))
print()
print(f"  MAE (H743实测 vs PC视角)    = {mae_pc:.4f}")
print(f"  MAE (H743实测 vs H743视角)  = {mae_h7:.4f}   <-- 更小则假设成立")

# ---------------------------------------------------------------------------
print()
print("[5] 两版 PC sim 指标对比 (末帧归并, thr=0.5)")
label_by_frame = {fidx: lab for (fidx, _), lab in zip(probs_pc, labels)}
n_windows = N_FRAMES // W
win_atk = []
for w in range(n_windows):
    s = w * W + 1
    win_atk.append(any(label_by_frame.get(f, 0) for f in range(s, s + W)))
n_atk = sum(win_atk)
n_norm = n_windows - n_atk
for tag, by_frame in (("PC 视角 (直读 CSV)", pc_by_frame),
                       ("H743 视角 (RTU round-trip)", h7_by_frame)):
    tp = fp = fn = 0
    for w in range(n_windows):
        p = by_frame.get((w + 1) * W)
        if p is None:
            continue
        if win_atk[w]:
            if p >= 0.5:
                tp += 1
            else:
                fn += 1
        elif p >= 0.5:
            fp += 1
    R = tp / max(n_atk, 1)
    P = tp / max(tp + fp, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    print(f"  {tag:28s} TP={tp:5d} FP={fp:5d} FN={fn:5d}  P={P:.4f} R={R:.4f} "
          f"F1={F1:.4f} FPR={fp/max(n_norm,1):.4f}")
print(f"  (H743 实测: TP=1482 FP=58 FN=933 P=0.9623 R=0.6137 F1=0.7494 FPR=0.0258)")

np.savez_compressed("rtu_skew_probs.npz",
                    pc_frames=np.array([f for f, _ in probs_pc]),
                    pc_probs=np.array([p for _, p in probs_pc]),
                    h7_frames=np.array([f for f, _ in probs_h7]),
                    h7_probs=np.array([p for _, p in probs_h7]))
print("\n[6] saved rtu_skew_probs.npz")
