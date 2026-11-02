#!/usr/bin/env python3
"""Cross-window evaluation — 分离 "模型能力" vs "评估颗粒度"。

方法: 取 W=16 训好的 5 个 .pt, 把 test 数据切成 W=4 窗口, 跑推理,
       然后对每 4 个连续 W=4 窗口的预测取 max → 得到 1 个 W=16 等效预测.
       跟原 W=16 的 5-seed ensemble F1 对比.

如果 F1 几乎一致 → W=4 vs W=16 的差异主要是评估颗粒度混淆
如果 F1 仍低很多 → 是真实模型能力差异 (W=4 训不出同样的能力)
"""
import os, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# 复用 sweep_tcn_se_ch32_window_4_8.py 的模型定义
import sys
sys.path.insert(0, r"C:\work\Claude\Issue")
from sweep_tcn_se_ch32_window_4_8 import TCNClassifierSE, load_data, per_class_prf, all_metrics, KEEP_23, N_FEATURES, BATCH

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]


def predict_with_model(W, seed, ckpt_path):
    """加载 .pt, 在 W 窗口的 test 上跑推理, 返回 (probs, labels)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TCNClassifierSE(in_ch=N_FEATURES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    Xte, yte = load_data(W)[4:6]
    loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    probs, lbls = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
            lbls.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(lbls)


def maxpool_aggregate(probs_w4, n_orig):
    """每 4 个连续 W=4 窗口的预测取 max → 等效 1 个 W=16 预测."""
    n_w4 = len(probs_w4)
    n_w16 = n_w4 // 4
    return probs_w4[:n_w16 * 4].reshape(n_w16, 4).max(axis=1)


def main():
    print("="*72)
    print("Cross-window evaluation — 同一模型跨窗口的 F1 表现")
    print("="*72)

    # ========== Test 1: W=16 模型在 native W=16 test (baseline) ==========
    print("\n[1] W=16 模型 × native W=16 test (baseline, 跟 rerun_4metrics_5seed.json 一致)")
    probs_all = []
    lbls_ref = None
    for seed in SEEDS:
        ckpt = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  missing: {ckpt}"); continue
        probs, lbls = predict_with_model(16, seed, ckpt)
        probs_all.append(probs); lbls_ref = lbls
    ens_prob = np.mean(probs_all, axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    m = all_metrics(lbls_ref, ens_pred)
    print(f"  5-seed ensemble: F1m={m['F1-Score']:.4f} BinF1={m['F1-Score']:.4f} "
          f"P={m['Precision']:.4f} R={m['Recall']:.4f} Acc={m['Accuracy']:.4f}  "
          f"PR-AUC={average_precision_score(lbls_ref, ens_prob):.4f}")
    f1_w16_native = m["F1-Score"]

    # ========== Test 2: W=4 模型在 native W=4 test (baseline) ==========
    print("\n[2] W=4 模型 × native W=4 test (baseline)")
    probs_all = []
    lbls_ref = None
    for seed in SEEDS:
        ckpt = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window4_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  missing: {ckpt}"); continue
        probs, lbls = predict_with_model(4, seed, ckpt)
        probs_all.append(probs); lbls_ref = lbls
    ens_prob = np.mean(probs_all, axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    m = all_metrics(lbls_ref, ens_pred)
    print(f"  5-seed ensemble: F1m={m['F1-Score']:.4f} BinF1={m['F1-Score']:.4f} "
          f"P={m['Precision']:.4f} R={m['Recall']:.4f} Acc={m['Accuracy']:.4f}  "
          f"PR-AUC={average_precision_score(lbls_ref, ens_prob):.4f}")
    f1_w4_native = m["F1-Score"]

    # ========== Test 3: W=16 模型跑在 W=4 test, max-pool 4 个连续窗口 ==========
    print("\n[3] W=16 模型 × W=4 test (max-pool 4 个连续窗口 → W=16 等效预测)")
    print("    关键: 如果 F1 ≈ Test1, 差异是评估颗粒度; 如果 F1 ≈ Test2, 是真模型能力")
    probs_all = []
    lbls_ref = None
    n_orig = 54927
    for seed in SEEDS:
        ckpt = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  missing: {ckpt}"); continue
        probs, _ = predict_with_model(4, seed, ckpt)
        probs_pooled = maxpool_aggregate(probs, n_orig)
        probs_all.append(probs_pooled)

    # 重建对应的 W=16 labels
    y_flat = np.load(os.path.join(BASE, "y_test_binary_v2_scada.npy")).astype(int)
    n_use = (n_orig // 16) * 16
    y_w16 = y_flat[:n_use].reshape(-1, 16).max(axis=1)

    ens_prob = np.mean(probs_all, axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    m = all_metrics(y_w16, ens_pred)
    print(f"  5-seed ensemble (W=16 model, max-pool over W=4): F1m={m['F1-Score']:.4f} "
          f"BinF1={m['F1-Score']:.4f} P={m['Precision']:.4f} R={m['Recall']:.4f} "
          f"Acc={m['Accuracy']:.4f} PR-AUC={average_precision_score(y_w16, ens_prob):.4f}")
    f1_w16_on_w4_pooled = m["F1-Score"]

    # ========== Test 4: W=4 模型跑在 W=16 test (UPSCALE — 单预测重复 4 次) ==========
    print("\n[4] W=4 模型 × W=16 test (W=4 模型只看 4 帧, 在 16 帧序列上每 4 帧切一次取平均)")
    print("    验证 W=4 模型真在长序列上的表现")
    probs_all = []
    n_orig = 54927
    n_w16 = n_orig // 16
    for seed in SEEDS:
        ckpt = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window4_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  missing: {ckpt}"); continue
        # 用 W=16 test 数据, 切成 4 个非重叠 W=4 子窗口 (取前 4, 中间 4, 后 4, 倒数第二个 4)
        Xte16, yte16 = load_data(16)[4:6]
        Xte4_full, _ = load_data(4)[4:6]  # 全 W=4 测试集
        # 每个 W=16 窗口对应 Xte16[i], 我们把它切成 4 个 W=4 子窗口
        # Xte16 shape: (n_w16, 16, 23), 切 (n_w16, 4, 4, 23) — 4 个连续 W=4
        n_w16 = Xte16.shape[0]
        Xte16_4sub = Xte16.reshape(n_w16, 4, 4, 23)  # (n_w16, 4 chunks, 4, 23)
        # flatten 成 (n_w16 * 4, 4, 23) — 但 channel-first
        Xte16_4sub = Xte16_4sub.transpose(0, 1, 3, 2)  # (n_w16, 4, 23, 4)
        Xte16_4sub = Xte16_4sub.reshape(-1, 23, 4)  # (n_w16*4, 23, 4)
        yte_dummy = np.zeros(len(Xte16_4sub), dtype=np.int64)
        loader = DataLoader(TensorDataset(torch.from_numpy(Xte16_4sub), torch.from_numpy(yte_dummy)),
                            batch_size=BATCH, shuffle=False, num_workers=0)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TCNClassifierSE(in_ch=N_FEATURES)
        model.load_state_dict(ck["state_dict"]); model.eval()
        probs_sub = []
        with torch.no_grad():
            for xb, _ in loader:
                probs_sub.append(torch.sigmoid(model(xb)).numpy())
        probs_sub = np.concatenate(probs_sub).reshape(n_w16, 4)  # (n_w16, 4 sub-predictions)
        # 聚合: 平均 4 个子预测
        probs_agg = probs_sub.mean(axis=1)
        probs_all.append(probs_agg)

    ens_prob = np.mean(probs_all, axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    m = all_metrics(yte16, ens_pred)
    print(f"  5-seed ensemble (W=4 model, 4-sub mean-pool over W=16): F1m={m['F1-Score']:.4f} "
          f"BinF1={m['F1-Score']:.4f} P={m['Precision']:.4f} R={m['Recall']:.4f} "
          f"Acc={m['Accuracy']:.4f} PR-AUC={average_precision_score(yte16, ens_prob):.4f}")
    f1_w4_on_w16_pooled = m["F1-Score"]

    # ========== 结论 ==========
    print("\n" + "="*72)
    print("分离分析结论")
    print("="*72)
    print(f"  [1] W=16 model × W=16 native test      → F1 = {f1_w16_native:.4f}  (SOTA baseline)")
    print(f"  [2] W=4  model × W=4  native test      → F1 = {f1_w4_native:.4f}  (本次 sweep)")
    print(f"  [3] W=16 model × W=4 test, max-pool    → F1 = {f1_w16_on_w4_pooled:.4f}  (key experiment)")
    print(f"  [4] W=4  model × W=16 test, 4-mean    → F1 = {f1_w4_on_w16_pooled:.4f}  (reverse)")
    print()
    print(f"  [3] - [1] = {f1_w16_on_w4_pooled - f1_w16_native:+.4f}  (≤0 说明 W=16 模型 max-pool 后基本不变)")
    print(f"  [2] - [4] = {f1_w4_native - f1_w4_on_w16_pooled:+.4f}  (≤0 说明 W=4 模型只看 4 帧的固有上限)")

    if abs(f1_w16_on_w4_pooled - f1_w16_native) < 0.02:
        print("\n  ✓ [3] 接近 [1] → W=4 的差距主要来自评估颗粒度, 模型本身仍能训出同等能力")
    else:
        print(f"\n  ✗ [3] 与 [1] 差 {(f1_w16_native - f1_w16_on_w4_pooled):.4f} → 训练数据量/感受野本身限制")

    if abs(f1_w4_on_w16_pooled - f1_w4_native) < 0.02:
        print("  ✓ [4] 接近 [2] → W=4 模型在长序列上同样弱, 是真实能力差距")
    else:
        print(f"  ✗ [4] 与 [2] 差 {(f1_w4_native - f1_w4_on_w16_pooled):.4f} → W=4 训过的模型在长序列可拼")


if __name__ == "__main__":
    main()
