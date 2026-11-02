"""
conv_sem_test.py -- 验证 H743 C kernel 的 dilated conv 语义是否等价于训练模型.

对比三组 prob (同一批 23-dim window 输入):
  A) PyTorch .pt 模型 (= 训练时的 forward, 对称 same padding)
  B) C 语义 forward (t_in = t + (k - pad)*dilation, 因果对齐)  == H743 实际跑的
  C) H743 实测上报 prob
"""
import csv
import math
import re
import struct
import sys
from collections import deque

import joblib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from h743_view_sim import (SW_WINDOW, CLIP_VAL, KEEP_18, KEEP_23, MCU_UNUSUAL,
                           build23, load_header_weights, mcu_decode, pc_encode,
                           _f, _i, St, featurize_pc, CSV_PATH, PT_PATH, SCALER,
                           HDR, H743_CSV, START, N)


def conv_c(inp, w, b, dil):
    """C 语义 conv1d_same (MCU ch32_inference.c 逐行复刻)."""
    O, C, K = w.shape
    T = inp.shape[1]
    pad = (K - 1) * dil // 2
    out = np.repeat(np.asarray(b, dtype=np.float64)[:, None], T, axis=1)
    for k in range(K):
        idx = np.arange(T) + (k - pad) * dil
        valid = (idx >= 0) & (idx < T)
        ic = np.clip(idx, 0, T - 1)
        out += np.einsum('oc,ct->ot', np.asarray(w[:, :, k], dtype=np.float64),
                         np.asarray(inp[:, ic], dtype=np.float64) * valid[None, :])
    return out


def forward_c_sem(A, x23):
    """完全复刻 MCU ch32_forward (C 语义)."""
    x = np.asarray(x23, dtype=np.float32)
    T = SW_WINDOW

    def get(n):
        return np.asarray(A[n], dtype=np.float32)

    # Block 0
    res = conv_c(x, get("v4se23_ch32_s42_w_tcn_0_residual_weight").reshape(32, 23, 1),
                 get("v4se23_ch32_s42_b_tcn_0_residual_bias"), 1)
    tmp = conv_c(x, get("v4se23_ch32_s42_w_tcn_0_conv1_weight").reshape(32, 23, 3),
                 get("v4se23_ch32_s42_b_tcn_0_conv1_bias"), 1)
    tmp = np.maximum(tmp, 0)
    cur = conv_c(tmp, get("v4se23_ch32_s42_w_tcn_0_conv2_weight").reshape(32, 32, 3),
                 get("v4se23_ch32_s42_b_tcn_0_conv2_bias"), 1)
    cur = np.maximum(cur, 0)
    cur = se_block(cur, A, 0)
    cur = np.maximum(cur + res, 0)

    for bi, dil in ((1, 2), (2, 4)):
        res = cur.copy()
        tmp = conv_c(cur, get("v4se23_ch32_s42_w_tcn_%d_conv1_weight" % bi).reshape(32, 32, 3),
                     get("v4se23_ch32_s42_b_tcn_%d_conv1_bias" % bi), dil)
        tmp = np.maximum(tmp, 0)
        cur = conv_c(tmp, get("v4se23_ch32_s42_w_tcn_%d_conv2_weight" % bi).reshape(32, 32, 3),
                     get("v4se23_ch32_s42_b_tcn_%d_conv2_bias" % bi), dil)
        cur = np.maximum(cur, 0)
        cur = se_block(cur, A, bi)
        cur = np.maximum(cur + res, 0)

    z = cur.sum(axis=1) / T
    w1 = get("v4se23_ch32_s42_fc1_weight").reshape(32, 32)
    b1 = get("v4se23_ch32_s42_b_fc1_bias")
    w2 = get("v4se23_ch32_s42_fc2_weight").reshape(1, 32)
    b2 = get("v4se23_ch32_s42_b_fc2_bias")
    h = np.maximum(w1 @ z + b1, 0)
    logit = float((w2 @ h + b2)[0])
    return 1.0 / (1.0 + math.exp(-logit))


def se_block(x, A, bi):
    T = x.shape[1]
    z = x.sum(axis=1) / T
    w1 = np.asarray(A["v4se23_ch32_s42_w_tcn_%d_se_fc1_weight" % bi], dtype=np.float32).reshape(4, 32)
    b1 = np.asarray(A["v4se23_ch32_s42_b_tcn_%d_se_fc1_bias" % bi], dtype=np.float32)
    w2 = np.asarray(A["v4se23_ch32_s42_w_tcn_%d_se_fc2_weight" % bi], dtype=np.float32).reshape(32, 4)
    b2 = np.asarray(A["v4se23_ch32_s42_b_tcn_%d_se_fc2_bias" % bi], dtype=np.float32)
    h = np.maximum(w1 @ z + b1, 0)
    s = 1.0 / (1.0 + np.exp(-(w2 @ h + b2)))
    return x * s[:, None]


def forward_torch(A, x23):
    """用 header (BN-folded) 权重 + PyTorch 对称 same padding (训练时的 forward)."""
    x = torch.from_numpy(np.ascontiguousarray(x23, dtype=np.float32)).unsqueeze(0)
    W = {k: torch.from_numpy(v.copy()) for k, v in A.items()}
    h = x
    for bi, dil in ((0, 1), (1, 2), (2, 4)):
        cin = 23 if bi == 0 else 32
        c1 = torch.nn.functional.conv1d(
            h, W["v4se23_ch32_s42_w_tcn_%d_conv1_weight" % bi].view(32, cin, 3),
            W["v4se23_ch32_s42_b_tcn_%d_conv1_bias" % bi],
            padding=(3 - 1) * dil // 2, dilation=dil)
        c1 = torch.relu(c1)
        c2 = torch.nn.functional.conv1d(
            c1, W["v4se23_ch32_s42_w_tcn_%d_conv2_weight" % bi].view(32, 32, 3),
            W["v4se23_ch32_s42_b_tcn_%d_conv2_bias" % bi],
            padding=(3 - 1) * dil // 2, dilation=dil)
        cur = torch.relu(c2)
        s = cur.mean(dim=2)
        s1 = torch.relu(torch.nn.functional.linear(
            s, W["v4se23_ch32_s42_w_tcn_%d_se_fc1_weight" % bi].view(4,32),
            W["v4se23_ch32_s42_b_tcn_%d_se_fc1_bias" % bi]))
        s2 = torch.sigmoid(torch.nn.functional.linear(
            s1, W["v4se23_ch32_s42_w_tcn_%d_se_fc2_weight" % bi].view(32,4),
            W["v4se23_ch32_s42_b_tcn_%d_se_fc2_bias" % bi]))
        cur = cur * s2.unsqueeze(2)
        if bi == 0:
            r = torch.nn.functional.conv1d(
                h, W["v4se23_ch32_s42_w_tcn_0_residual_weight"].view(32, 23, 1),
                W["v4se23_ch32_s42_b_tcn_0_residual_bias"])
        else:
            r = h
        h = torch.relu(cur + r)
    g = h.mean(dim=2)
    f1 = torch.relu(torch.nn.functional.linear(
        g, W["v4se23_ch32_s42_fc1_weight"].view(32,32), W["v4se23_ch32_s42_b_fc1_bias"]))
    logit = torch.nn.functional.linear(
        f1, W["v4se23_ch32_s42_fc2_weight"].view(1,32), W["v4se23_ch32_s42_b_fc2_bias"]).squeeze()
    return 1.0 / (1.0 + math.exp(-float(logit)))


def main():
    sc = joblib.load(SCALER)
    center = np.asarray(sc.center_, dtype=np.float32)
    scale = np.asarray(sc.scale_, dtype=np.float32)
    mu18 = center[np.asarray(KEEP_18)]
    sg18 = scale[np.asarray(KEEP_18)]
    A = load_header_weights()

    st_pc = St()
    lru = {}
    last_ts = None
    first = True
    win_pc = deque(maxlen=SW_WINDOW)
    win_h7 = deque(maxlen=SW_WINDOW)
    wins = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if i <= START:
                continue
            if i > START + N:
                break
            fidx = i - START
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
            raw_h7 = np.array([dec["address"], dec["function"], dec["length"],
                               dec["gain"], dec["reset"], dec["deadband"], dec["cycle"],
                               dec["rate"], dec["system_mode"], dec["control_scheme"],
                               dec["pump"], dec["solenoid"],
                               min(100.0, max(0.0, dec["pressure"])), dec["crc"],
                               dt, tsl,
                               1.0 if dec["function"] in MCU_UNUSUAL else 0.0,
                               float(dec["is_response"])], dtype=np.float32)
            win_h7.append((raw_h7 - mu18) / sg18)
            if len(win_pc) < SW_WINDOW:
                continue
            if fidx % SW_WINDOW == 0:
                wins.append((fidx, build23(np.stack(list(win_pc), 0)),
                             build23(np.stack(list(win_h7), 0))))

    print("windows: %d" % len(wins))
    # 抽 300 个 window (均匀) 做对比, 全跑太慢
    idx = np.linspace(0, len(wins) - 1, 300).astype(int)
    cs, ts_, tsc = [], [], []
    for j in idx:
        fidx, xpc, xh7 = wins[j]
        cs.append(forward_c_sem(A, xh7))
        ts_.append(forward_torch(A, xpc))
        tsc.append(forward_torch(A, xh7))
    cs = np.array(cs)
    ts_ = np.array(ts_)
    tsc = np.array(tsc)
    print()
    print("  C语义 (H743 实际跑)      vs 训练模型(PyTorch对称same):")
    print("    prob 均值 C=%.4f  torch=%.4f" % (cs.mean(), ts_.mean()))
    print("    |diff| mean=%.4f  max=%.4f  一致(>0.99)=%.3f"
          % (np.abs(cs - ts_).mean(), np.abs(cs - ts_).max(),
             np.mean(np.abs(cs - ts_) < 0.01)))
    print("  C语义 vs PyTorch(同一 H743 输入) MAE=%.6f" % np.mean(np.abs(cs - tsc)))

    # 全部 window: C 语义 prob
    allc = np.zeros(len(wins))
    for j, (fidx, xpc, xh7) in enumerate(wins):
        allc[j] = forward_c_sem(A, xh7)
    frame_end = np.array([w[0] for w in wins])

    h743 = {}
    with open(H743_CSV, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["event"] == "ALARM" and r["frame_idx"] and r["prob_x1000"]:
                h743[int(r["frame_idx"])] = int(r["prob_x1000"]) / 1000.0
    fe2c = dict(zip(frame_end.tolist(), allc.tolist()))
    hp, cp, fr = [], [], []
    for fr_i, pv in h743.items():
        cand = [f for f in frame_end.tolist() if fr_i - 15 <= f <= fr_i]
        if not cand:
            continue
        f = cand[-1]
        fr.append(f)
        cp.append(fe2c.get(f))
        hp.append(pv)
    hp = np.array(hp)
    cp = np.array(cp)
    m = ~np.isnan(cp)
    print()
    print("  [H743 实测 prob ] n=%d  mean=%.4f median=%.4f" % (len(hp), hp.mean(), np.median(hp)))
    print("  [C语义复现 prob ] n=%d  mean=%.4f median=%.4f  MAE=%.4f  corr=%.4f"
          % (m.sum(), cp[m].mean(), np.median(cp[m]),
             np.mean(np.abs(hp[m] - cp[m])), np.corrcoef(hp[m], cp[m])[0, 1]))
    np.savez_compressed("csem_probs.npz", frames=frame_end, csem=allc)


if __name__ == "__main__":
    main()
