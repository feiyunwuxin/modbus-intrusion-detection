"""
analyze_fpr_train_vs_deploy.py -- 回答 "训练时 FPR 很小, 为什么部署实测 16.76%?"

把两套数据摆在一起:
  (A) 官方训练管线: X_val/X_test npy + 部署的 .pt 模型  -> 训练期报告的 val/test 指标
  (B) 部署回放管线: 同一份 .pt, 走 H743 23-dim 特征链路   -> 部署实测指标

并按全局帧号把回放区间切成 val 段和 test 段, 逐段算 FPR。
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from parity_fp32_c_vs_onnx import c_forward, parse_h
from h743_windows import build_windows, window_labels

BASE = r"C:\work\Claude\Issue"
PT = os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt")
H_PATH = "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h"
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]

# 全局帧号边界 (1-indexed), 来自 retrain_tcn_v4_se_window16_train_log:
#   train=(192239,17) val=(27462,17) test=(54927,17)  总 274628
N_TRAIN, N_VAL, N_TEST = 192239, 27462, 54927
VAL_LO, VAL_HI = N_TRAIN + 1, N_TRAIN + N_VAL                 # 192240 .. 219701
TEST_LO, TEST_HI = VAL_HI + 1, N_TRAIN + N_VAL + N_TEST        # 219702 .. 274628


def metrics(probs, y, thr):
    det = probs >= thr
    tp = int((det & (y == 1)).sum())
    fp = int((det & (y == 0)).sum())
    fn = int((~det & (y == 1)).sum())
    tn = int((~det & (y == 0)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    FPR = fp / max(fp + tn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=P, R=R, F1=F1, FPR=FPR,
                Acc=(tp + tn) / max(len(y), 1))


def show(tag, m, thr):
    print("  %-34s thr=%.2f  TP=%5d FP=%5d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (tag, thr, m["TP"], m["FP"], m["FN"], m["P"], m["R"], m["F1"], m["FPR"]))


# ---------------------------------------------------------------- (A) 官方管线
print("=" * 108)
print("(A) 官方训练管线: X_val/X_test npy + 部署 .pt (PyTorch, BN 展开前)")
print("=" * 108)

ck = torch.load(PT, map_location="cpu", weights_only=False)
sys.path.insert(0, "modbus_replay_simulator")
from modbus_replay.ids_worker_torch import build_tcn_se_model
model = build_tcn_se_model(
    in_ch=ck["n_features"], n_blocks=ck["n_blocks"], channels=ck["channels"],
    kernel_size=ck["kernel_size"], dilations=tuple(ck["dilations"]),
    dropout=ck["dropout"], se_reduction=ck["se_reduction"])
model.load_state_dict(ck["state_dict"])
model.eval()

official = {}
for split in ("val", "test"):
    X = np.load(os.path.join(BASE, f"X_{split}_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    y = np.load(os.path.join(BASE, f"y_{split}_binary_v2_scada_window16.npy")).astype(np.int64)
    X = np.clip(X, -10.0, 10.0).transpose(0, 2, 1)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()
    official[split] = (probs, y)
    for thr in (0.48, 0.5):
        show(f"官方 {split} 集 (n={len(y)})", metrics(probs, y, thr), thr)
    print()

# ---------------------------------------------------------------- (B) 回放管线
print("=" * 108)
print("(B) 部署回放管线: 同一 .pt, H743 23-dim 特征链路, 帧 200001..274628")
print("=" * 108)

wins, labels = build_windows(n=74628)
atk = np.array(window_labels(wins, labels), dtype=np.int64)
frame_end = np.array([w[0] for w in wins], dtype=np.int64)   # 回放内帧号
global_end = frame_end + 200000                            # 全局帧号

raw = parse_h(H_PATH)
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
    for nm, key in (("conv1", "conv1"), ("conv2", "conv2"), ("se_fc1", "se_fc1"), ("se_fc2", "se_fc2")):
        W["w_tcn_%d_%s" % (bi, nm)] = raw[f"v4se23_ch32_s42_w_tcn_{bi}_{nm}_weight"]
        W["b_tcn_%d_%s" % (bi, nm)] = raw[f"v4se23_ch32_s42_b_tcn_{bi}_{nm}_bias"]

print("[计算] %d window ..." % len(wins))
replay_probs = np.array([1.0 / (1.0 + math.exp(-c_forward(W, w[2]))) for w in wins])
np.savez_compressed("replay_vs_official.npz",
                    replay_probs=replay_probs, atk=atk, global_end=global_end)

print()
for thr in (0.48, 0.5):
    show("回放 全区间 (200001..274628)", metrics(replay_probs, atk, thr), thr)

# 按全局帧号切 val 段 / test 段
m_val = (global_end <= VAL_HI)
m_test = (global_end >= TEST_LO)
print()
for name, m in (("回放∩val 段 (200001..219701)", m_val), ("回放∩test 段 (219702..274628)", m_test)):
    for thr in (0.5,):
        show(f"{name} (n={m.sum()})", metrics(replay_probs[m], atk[m], thr), thr)

# ---------------------------------------------------------------- 对比
print()
print("=" * 108)
print("对比: 同一段 test 帧, 官方管线 vs 部署回放管线")
print("=" * 108)
po, yo = official["test"]
ro, yo2 = replay_probs[m_test], atk[m_test]
for thr in (0.48, 0.5):
    a = metrics(po, yo, thr)
    b = metrics(ro, yo2, thr)
    print("  thr=%.2f" % thr)
    print("    官方 test 集 : F1=%.4f R=%.4f FPR=%.4f  (n=%d)" % (a["F1"], a["R"], a["FPR"], len(yo)))
    print("    回放 test 段 : F1=%.4f R=%.4f FPR=%.4f  (n=%d)" % (b["F1"], b["R"], b["FPR"], len(yo2)))
    print("    prob 差异: MAE=%.5f  corr=%.4f" % (np.abs(po - ro).mean(), np.corrcoef(po, ro)[0, 1]))
    print("    标签一致率: %.4f  (官方 y=%d 个, 回放 y=%d 个)"
          % ((yo == yo2).mean(), int(yo.sum()), int(yo2.sum())))
    print()
