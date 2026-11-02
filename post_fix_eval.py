"""
post_fix_eval.py -- 修复 conv 对齐后, H743 的预期 per-window 指标 + 阈值扫描.

用 C 源码解析出的 t_in 公式 (即修复后的 C kernel) + H743 视角输入.
"""
import csv
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from parity_fp32_c_vs_onnx import (parse_h, c_forward, load_t_in_formula,
                                   TEMPERATURE)
from h743_windows import build_windows, window_labels
from conv_sem_test import forward_c_sem  # 旧 (因果) 语义, 保留做对照

H743_CSV = r"logs\test_20261020-215840_auto-40ms.csv"

raw = parse_h("MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h")
W = {
    "w_tcn_0_conv1": raw["v4se23_ch32_s42_w_tcn_0_conv1_weight"],
    "b_tcn_0_conv1": raw["v4se23_ch32_s42_b_tcn_0_conv1_bias"],
    "w_tcn_0_conv2": raw["v4se23_ch32_s42_w_tcn_0_conv2_weight"],
    "b_tcn_0_conv2": raw["v4se23_ch32_s42_b_tcn_0_conv2_bias"],
    "w_tcn_0_residual": raw["v4se23_ch32_s42_w_tcn_0_residual_weight"],
    "b_tcn_0_residual": raw["v4se23_ch32_s42_b_tcn_0_residual_bias"],
    "fc1": raw["v4se23_ch32_s42_fc1_weight"],
    "b_fc1": raw["v4se23_ch32_s42_b_fc1_bias"],
    "fc2": raw["v4se23_ch32_s42_fc2_weight"],
    "b_fc2": raw["v4se23_ch32_s42_b_fc2_bias"],
}
for bi in (0, 1, 2):
    W["w_tcn_%d_conv1" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_conv1_weight" % bi]
    W["b_tcn_%d_conv1" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_conv1_bias" % bi]
    W["w_tcn_%d_conv2" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_conv2_weight" % bi]
    W["b_tcn_%d_conv2" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_conv2_bias" % bi]
    W["w_tcn_%d_se_fc1" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_se_fc1_weight" % bi]
    W["b_tcn_%d_se_fc1" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_se_fc1_bias" % bi]
    W["w_tcn_%d_se_fc2" % bi] = raw["v4se23_ch32_s42_w_tcn_%d_se_fc2_weight" % bi]
    W["b_tcn_%d_se_fc2" % bi] = raw["v4se23_ch32_s42_b_tcn_%d_se_fc2_bias" % bi]

wins, labels = build_windows(n=74628)
atk = window_labels(wins, labels)
n_win = len(wins)
n_atk = sum(atk)
n_norm = n_win - n_atk
frame_end = np.array([w[0] for w in wins])

print("[计算] %d 个 window 的修复后 prob ..." % n_win)
fixed = np.array([1.0 / (1.0 + math.exp(-c_forward(W, xh7))) for (_, _, xh7) in wins])
old = np.array([forward_c_sem(raw, xh7) for (_, _, xh7) in wins])
np.savez_compressed("post_fix_probs.npz", frames=frame_end, fixed=fixed, old=old,
                    atk=np.array(atk, dtype=np.int8))

print("windows=%d  attack=%d  normal=%d" % (n_win, n_atk, n_norm))
print("修复后 prob: mean=%.4f median=%.4f   (修复前 mean=%.4f median=%.4f)"
      % (fixed.mean(), np.median(fixed), old.mean(), np.median(old)))


def metrics(probs, thr, tag, quiet=False):
    tp = fp = fn = 0
    for w in range(n_win):
        det = probs[w] >= thr
        if atk[w]:
            tp += det
            fn += (not det)
        else:
            fp += det
    P = tp / max(tp + fp, 1)
    R = tp / max(n_atk, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    FPR = fp / max(n_norm, 1)
    if not quiet:
        print("  %-26s thr=%.2f TP=%5d FP=%5d FN=%5d P=%.4f R=%.4f F1=%.4f FPR=%.4f Acc=%.4f"
              % (tag, thr, tp, fp, fn, P, R, F1, FPR, (tp + n_norm - fp) / n_win))
    return F1, thr, tp, fp, fn, P, R, FPR


print()
print("修复前 (H743 当前固件, 因果 conv):")
for thr in (0.36, 0.5):
    metrics(old, thr, "修复前")
print()
print("H743 实测 summary:      TP=1482 FP=58  FN=933 P=0.9623 R=0.6137 F1=0.7494 FPR=0.0258")
print()
print("修复后 (对称 conv == 训练模型):")
best = None
for thr in (0.3, 0.36, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
    r = metrics(fixed, thr, "修复后")
    if best is None or r[0] > best[0]:
        best = r
print()
print("  最优阈值: thr=%.2f  F1=%.4f  (Recall=%.4f  FPR=%.4f)" % (best[1], best[0], best[6], best[7]))
