"""
h743_view_sim.py -- 精确复现 H743 端数据流, 跟 H743 实测 prob 对比.

链路: CSV row -> rtu_replay_gui.build_raw17_frame (字节) -> MCU raw17_decode
      -> MCU feature_extractor_update (18-dim raw + RobustScaler)
      -> MCU sliding_window_push/get (23-dim)
      -> 部署 header 权重 forward (等价 ch32_forward)
"""
import csv
import math
import os
import re
import struct
import time
from collections import deque

import joblib
import numpy as np
import torch
import torch.nn.functional as F

SW_WINDOW = 16
CLIP_VAL = 10.0
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]
MCU_UNUSUAL = {133, 136, 137, 138, 139, 140, 171}
ATTACK_LEN_THRESHOLD = 16
ATTACK_BODY_SIZE = 13

CSV_PATH = os.environ.get("H743_CSV", "IanArffDataset.csv")
PT_PATH = "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"
SCALER = "scaler_binary_v2_scada.joblib"
HDR = "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h"
H743_CSV = r"logs\test_20261020-215840_auto-40ms.csv"
START, N = 200000, 74628

RAW17_COLS = ('setpoint', 'gain', 'reset rate', 'deadband', 'cycle time', 'rate',
              'system mode', 'control scheme', 'pump', 'solenoid', 'pressure measurement')


def _f(v, d=0.0):
    if v is None or v == "?":
        return d
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _i(v, d=0):
    if v is None or v == "?":
        return d
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return d


def _u(v, d=0, lo=0, hi=255):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return d
    if f != f:
        return d
    return max(lo, min(hi, int(f)))


def pc_encode(row):
    """rtu_replay_gui.build_raw17_frame 的字节级复刻."""
    slave = _u(row.get('address'), 1, 1, 247)
    func = _u(row.get('function'), 3, 0, 255)
    length = _u(row.get('length'), 16, 0, 255)
    is_resp = 1 if _u(row.get('command response'), 0, 0, 1) == 1 else 0
    t = _f(row.get('time'), 0.0)
    crc_u16 = max(0, min(65535, int(round(_f(row.get('crc rate'))))))
    fv = [_f(row.get(c)) for c in RAW17_COLS]
    if length == 16:
        body = bytearray()
        for v in fv[:10]:
            body.append(max(0, min(255, int(round(v)))))
        body += struct.pack('>f', float(fv[10]))
        body += struct.pack('>H', crc_u16)
    elif length == 46:
        body = bytearray()
        for v in fv:
            body += struct.pack('>f', float(v))
        body += struct.pack('>H', crc_u16)
    elif length == 90:
        body = bytearray()
        for v in fv:
            body += struct.pack('>d', float(v))
        body += struct.pack('>H', crc_u16)
    else:
        body = bytearray(11)
        body += struct.pack('>H', crc_u16)
    return (bytes([slave, func, length]) + bytes(body)
            + bytes([is_resp]) + struct.pack('>d', t))


def mcu_decode(buf):
    """MCU modbus_rtu.c raw17_decode 的 Python 复刻."""
    address, function, length = buf[0], buf[1], buf[2]
    body_size = ATTACK_BODY_SIZE if length < ATTACK_LEN_THRESHOLD else length
    body = buf[3:3 + body_size]
    tail = buf[3 + body_size:]
    keys = ('setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate', 'system_mode',
            'control_scheme', 'pump', 'solenoid', 'pressure')
    z = dict.fromkeys(keys, 0.0)
    crc = 0.0
    if length < ATTACK_LEN_THRESHOLD:
        crc = float(struct.unpack_from('>H', body, ATTACK_BODY_SIZE - 2)[0])
    elif length == 16:
        names = ('setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate',
                 'system_mode', 'control_scheme', 'pump', 'solenoid')
        for n, k in enumerate(names):
            z[k] = float(body[n])
        z['pressure'] = struct.unpack_from('>f', body, 10)[0]
        crc = float(struct.unpack_from('>H', body, 14)[0])
    elif length == 46:
        names = ('setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate',
                 'system_mode', 'control_scheme', 'pump', 'solenoid', 'pressure')
        for n, k in enumerate(names):
            z[k] = struct.unpack_from('>f', body, 4 * n)[0]
        crc = float(struct.unpack_from('>H', body, 44)[0])
    elif length == 90:
        names = ('setpoint', 'gain', 'reset', 'deadband', 'cycle', 'rate',
                 'system_mode', 'control_scheme', 'pump', 'solenoid', 'pressure')
        for n, k in enumerate(names):
            z[k] = struct.unpack_from('>d', body, 8 * n)[0]
        crc = float(struct.unpack_from('>H', body, 88)[0])
    else:
        crc = float(struct.unpack_from('>H', body, length - 2)[0])
    is_response = tail[0]
    t = struct.unpack_from('>d', tail, 1)[0]
    out = dict(address=address, function=function, length=length, crc=crc,
               is_response=is_response, time=t)
    out.update(z)
    return out


def build23(w18):
    """MCU sliding_window_get 的 23-dim 组装 (在归一化空间)."""
    press_sum = float(w18[:, 12].sum())
    crc_max = float(w18[:, 13].max())
    resp = float(w18[:, 17].sum())
    unu = float(w18[:, 16].sum())
    seen = set()
    nd = 0
    for v in w18[:, 2]:
        kk = int(round(float(v)))
        if kk not in seen:
            seen.add(kk)
            nd += 1
    f27 = np.zeros((SW_WINDOW, 27), dtype=np.float32)
    for c18 in range(18):
        f27[:, KEEP_18[c18]] = w18[:, c18]
    f27[:, 19] = press_sum / SW_WINDOW
    f27[:, 20] = float(w18[:, 13].mean())
    f27[:, 21] = crc_max
    f27[:, 22] = float(SW_WINDOW - unu - resp)
    f27[:, 23] = resp
    f24 = 1.0 - resp / 8.0
    f27[:, 24] = f24
    f27[:, 25] = nd
    f27[:, 26] = unu
    return np.clip(f27[:, KEEP_23], -CLIP_VAL, CLIP_VAL).T.astype(np.float32)


def load_header_weights():
    txt = open(HDR, encoding="utf-8", errors="replace").read()
    A = {}
    for m in re.finditer(r"static const float\s+(\w+)\[(\d+)\]\s*=\s*\{(.*?)\};", txt, re.S):
        vals = [float(x[:-1]) for x in re.findall(r"[-+0-9.eE]+f", m.group(3))]
        assert len(vals) == int(m.group(2)), (m.group(1), len(vals))
        A[m.group(1)] = np.array(vals, dtype=np.float32)
    return A


def verify_weights(A):
    ck = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]

    def fold(w, b, mean, var, gamma, beta, eps=1e-5):
        scale = gamma / torch.sqrt(var + eps)
        return w * scale.view(-1, 1, 1), (b - mean) * scale + beta

    md = {}
    for bi in (0, 1, 2):
        w1, b1 = fold(sd["tcn.%d.conv1.weight" % bi], sd["tcn.%d.conv1.bias" % bi],
                      sd["tcn.%d.bn1.running_mean" % bi], sd["tcn.%d.bn1.running_var" % bi],
                      sd["tcn.%d.bn1.weight" % bi], sd["tcn.%d.bn1.bias" % bi])
        w2, b2 = fold(sd["tcn.%d.conv2.weight" % bi], sd["tcn.%d.conv2.bias" % bi],
                      sd["tcn.%d.bn2.running_mean" % bi], sd["tcn.%d.bn2.running_var" % bi],
                      sd["tcn.%d.bn2.weight" % bi], sd["tcn.%d.bn2.bias" % bi])
        for tag, arr, ref in (
                ("c1w", w1, "v4se23_ch32_s42_w_tcn_%d_conv1_weight" % bi),
                ("c1b", b1, "v4se23_ch32_s42_b_tcn_%d_conv1_bias" % bi),
                ("c2w", w2, "v4se23_ch32_s42_w_tcn_%d_conv2_weight" % bi),
                ("c2b", b2, "v4se23_ch32_s42_b_tcn_%d_conv2_bias" % bi)):
            md["b%d_%s" % (bi, tag)] = float(np.abs(arr.numpy().ravel() - A[ref]).max())
        for tag, key, ref in (
                ("se1w", "tcn.%d.se.fc1.weight" % bi, "v4se23_ch32_s42_w_tcn_%d_se_fc1_weight" % bi),
                ("se1b", "tcn.%d.se.fc1.bias" % bi, "v4se23_ch32_s42_b_tcn_%d_se_fc1_bias" % bi),
                ("se2w", "tcn.%d.se.fc2.weight" % bi, "v4se23_ch32_s42_w_tcn_%d_se_fc2_weight" % bi),
                ("se2b", "tcn.%d.se.fc2.bias" % bi, "v4se23_ch32_s42_b_tcn_%d_se_fc2_bias" % bi)):
            md["b%d_%s" % (bi, tag)] = float(np.abs(sd[key].numpy().ravel() - A[ref]).max())
    md["resw"] = float(np.abs(sd["tcn.0.residual.weight"].numpy().ravel()
                              - A["v4se23_ch32_s42_w_tcn_0_residual_weight"]).max())
    md["resb"] = float(np.abs(sd["tcn.0.residual.bias"].numpy().ravel()
                              - A["v4se23_ch32_s42_b_tcn_0_residual_bias"]).max())
    for tag, key, ref in (
            ("fc1w", "fc1.weight", "v4se23_ch32_s42_fc1_weight"),
            ("fc1b", "fc1.bias", "v4se23_ch32_s42_b_fc1_bias"),
            ("fc2w", "fc2.weight", "v4se23_ch32_s42_fc2_weight"),
            ("fc2b", "fc2.bias", "v4se23_ch32_s42_b_fc2_bias")):
        md[tag] = float(np.abs(sd[key].numpy().ravel() - A[ref]).max())
    print("[权重校验] 部署 header vs .pt (BN-folded) 最大绝对差:")
    worst = max(md.values())
    for k in sorted(md, key=md.get, reverse=True)[:6]:
        print("    %-10s %.3e" % (k, md[k]))
    print("    最大 %.3e  ->  bit-perfect? %s" % (worst, worst < 1e-6))
    return A


def deployed_forward(A, x23):
    """等价 MCU ch32_forward: conv1d_same(dilation) -> SE -> +residual -> relu, GAP -> FC."""
    x = torch.from_numpy(np.ascontiguousarray(x23, dtype=np.float32)).unsqueeze(0)
    W = {k: torch.from_numpy(v.copy()) for k, v in A.items()}

    def conv_same(w, b, dil):
        k = w.shape[2]
        pad = dil * (k - 1) // 2
        xp = F.pad(x, (pad, pad))
        y = F.conv1d(xp, w, b, dilation=dil)
        return y[:, :, :x.shape[2]]

    h = x
    for bi, dil in ((0, 1), (1, 2), (2, 4)):
        cin = 23 if bi == 0 else 32
        c1w = W["v4se23_ch32_s42_w_tcn_%d_conv1_weight" % bi].view(32, cin, 3)
        c1b = W["v4se23_ch32_s42_b_tcn_%d_conv1_bias" % bi]
        c2w = W["v4se23_ch32_s42_w_tcn_%d_conv2_weight" % bi].view(32, 32, 3)
        c2b = W["v4se23_ch32_s42_b_tcn_%d_conv2_bias" % bi]
        h1 = F.relu(conv_same(c1w, c1b, dil))
        h2 = F.relu(conv_same(c2w, c2b, dil))
        if bi == 0:
            rw = W["v4se23_ch32_s42_w_tcn_0_residual_weight"].view(32, 23, 1)
            rb = W["v4se23_ch32_s42_b_tcn_0_residual_bias"]
            r = F.conv1d(x, rw, rb)
        else:
            r = h
        h = F.relu(h2 + r)
        s = h.mean(dim=2)
        s1 = F.relu(F.linear(s, W["v4se23_ch32_s42_w_tcn_%d_se_fc1_weight" % bi],
                             W["v4se23_ch32_s42_b_tcn_%d_se_fc1_bias" % bi]))
        s2 = torch.sigmoid(F.linear(s1, W["v4se23_ch32_s42_w_tcn_%d_se_fc2_weight" % bi],
                                    W["v4se23_ch32_s42_b_tcn_%d_se_fc2_bias" % bi]))
        h = h * s2.unsqueeze(2)
    g = h.mean(dim=2)
    f1 = F.relu(F.linear(g, W["v4se23_ch32_s42_fc1_weight"], W["v4se23_ch32_s42_b_fc1_bias"]))
    logit = F.linear(f1, W["v4se23_ch32_s42_fc2_weight"], W["v4se23_ch32_s42_b_fc2_bias"]).squeeze()
    return float(logit)


class St:
    def __init__(self):
        self.last = {}
        self.prev = None


def featurize_pc(st, row):
    t = _f(row.get("time"))
    dt = 0.0 if st.prev is None else t - st.prev
    a = _i(row.get("address"))
    fc = _i(row.get("function"))
    k = (a, fc)
    tsl = 0.0 if k not in st.last else t - st.last[k]
    st.prev = t
    st.last[k] = t
    pres = max(0.0, min(100.0, _f(row.get("pressure measurement"))))
    return np.array([a, fc, _i(row.get("length")), _f(row.get("gain")), _f(row.get("reset rate")),
                     _f(row.get("deadband")), _f(row.get("cycle time")), _f(row.get("rate")),
                     _i(row.get("system mode")), _i(row.get("control scheme")), _i(row.get("pump")),
                     _i(row.get("solenoid")), pres, _f(row.get("crc rate")), dt, tsl,
                     1.0 if fc in MCU_UNUSUAL else 0.0,
                     float(_i(row.get("command response")))], dtype=np.float32)


def main():
    print("[1] 加载 scaler / 部署权重")
    sc = joblib.load(SCALER)
    center = np.asarray(sc.center_, dtype=np.float32)
    scale = np.asarray(sc.scale_, dtype=np.float32)
    mu18 = center[np.asarray(KEEP_18)]
    sg18 = scale[np.asarray(KEEP_18)]
    A = verify_weights(load_header_weights())

    print("[2] 读 CSV, 双路 18-dim")
    st_pc = St()
    lru = {}
    last_ts = None
    first = True
    win_pc = deque(maxlen=SW_WINDOW)
    win_h7 = deque(maxlen=SW_WINDOW)
    diff_cnt = np.zeros(18, dtype=np.int64)
    length_hist = {}
    wins = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if i <= START:
                continue
            if i > START + N:
                break
            fidx = i - START
            ln = _i(row.get("length"))
            length_hist[ln] = length_hist.get(ln, 0) + 1

            raw_pc = featurize_pc(st_pc, row)
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
            diff_cnt += (raw_pc != raw_h7)

            if len(win_pc) < SW_WINDOW:
                continue
            if fidx % SW_WINDOW == 0:
                wins.append((fidx, build23(np.stack(list(win_pc), 0)),
                             build23(np.stack(list(win_h7), 0))))

    print("    windows: %d" % len(wins))
    print("[3] 18-dim 逐通道差异 (PC 直读 CSV vs H743 全链路)")
    names = ['address', 'function', 'length', 'gain', 'reset_rate', 'deadband',
             'cycle_time', 'rate', 'system_mode', 'control_scheme', 'pump', 'solenoid',
             'pressure', 'crc_rate', 'time_diff', 'time_since', 'is_unusual_fc', 'is_response']
    for c in range(18):
        pct = diff_cnt[c] / N * 100
        print("    ch%2d %-16s %7d (%6.2f)%s" % (c, names[c], diff_cnt[c], pct,
                                                 "   <<<<" if pct > 1 else ""))
    print("    length 分布:", dict(sorted(length_hist.items())))

    print("[4] 部署权重 forward (H743 视角 vs PC 视角)")
    t0 = time.time()
    pc_p, h7_p = [], []
    for (fidx, x23pc, x23h7) in wins:
        pc_p.append(1.0 / (1.0 + math.exp(-deployed_forward(A, x23pc))))
        h7_p.append(1.0 / (1.0 + math.exp(-deployed_forward(A, x23h7))))
    pc_p = np.array(pc_p)
    h7_p = np.array(h7_p)
    frame_end = np.array([w[0] for w in wins])
    print("    %d forwards in %.1fs" % (len(wins) * 2, time.time() - t0))
    np.savez_compressed("h743view_probs.npz", frames=frame_end, pc=pc_p, h7=h7_p)

    h743 = {}
    with open(H743_CSV, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["event"] == "ALARM" and r["frame_idx"] and r["prob_x1000"]:
                h743[int(r["frame_idx"])] = int(r["prob_x1000"]) / 1000.0
    print("    H743 ALARM rows: %d" % len(h743))

    fe2p = dict(zip(frame_end.tolist(), pc_p.tolist()))
    fe2h = dict(zip(frame_end.tolist(), h7_p.tolist()))
    hp, pp, h7p, fr = [], [], [], []
    for fr_i, pv in h743.items():
        cand = [f for f in frame_end.tolist() if fr_i - 15 <= f <= fr_i]
        if not cand:
            continue
        f = cand[-1]
        fr.append(f)
        pp.append(fe2p.get(f))
        h7p.append(fe2h.get(f))
        hp.append(pv)
    hp = np.array(hp)
    pp = np.array(pp)
    h7p = np.array(h7p)
    m = ~np.isnan(pp)
    print()
    print("    [H743 实测 ] prob: mean=%.4f median=%.4f  n=%d" % (hp.mean(), np.median(hp), len(hp)))
    print("    [PC 视角  ] prob: mean=%.4f median=%.4f  MAE=%.4f  corr=%.4f"
          % (pp[m].mean(), np.median(pp[m]), np.mean(np.abs(hp[m] - pp[m])),
             np.corrcoef(hp[m], pp[m])[0, 1]))
    print("    [H743 视角] prob: mean=%.4f median=%.4f  MAE=%.4f  corr=%.4f"
          % (h7p[m].mean(), np.median(h7p[m]), np.mean(np.abs(hp[m] - h7p[m])),
             np.corrcoef(hp[m], h7p[m])[0, 1]))

    # 阈值扫描: 哪种视角能复现 H743 的 TP/FP?
    lab = np.load("gui_labels.npz")
    gl = dict(zip(lab["frames"].tolist(), lab["gui"].tolist()))
    n_win = len(wins)
    atk = [any(gl.get(f, 0) for f in range(w * 16 + 1, w * 16 + 17)) for w in range(n_win)]
    n_atk = sum(atk)
    n_norm = n_win - n_atk
    print()
    print("    窗口标签: attack=%d normal=%d (summary: 2415/2249)" % (n_atk, n_norm))
    for name, arr in (("PC 视角", pc_p), ("H743 视角", h7_p)):
        for thr in (0.36, 0.5, 0.7):
            tp = fp = fn = 0
            for w in range(n_win):
                det = arr[w] >= thr
                if atk[w]:
                    if det:
                        tp += 1
                    else:
                        fn += 1
                elif det:
                    fp += 1
            P = tp / max(tp + fp, 1)
            R = tp / max(n_atk, 1)
            F1 = 2 * P * R / max(P + R, 1e-9)
            print("      %-8s thr=%.2f  TP=%5d FP=%5d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
                  % (name, thr, tp, fp, fn, P, R, F1, fp / max(n_norm, 1)))
    print("    H743 实测: TP=1482 FP=58 FN=933 P=0.9623 R=0.6137 F1=0.7494 FPR=0.0258")


if __name__ == "__main__":
    main()
