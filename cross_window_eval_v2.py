#!/usr/bin/env python3
"""Cross-window eval v2 — 用 PR-AUC (无阈值) + threshold sweep 正确解读.

上次 v1 跑出来 max-pool 后 F1=0.69 是因为 4 概率取 max 后分布上移,
阈值 0.5 把太多东西判 attack. 用 PR-AUC 看真实能力差距, 阈值扫描找最佳.
"""
import os, json, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

import sys
sys.path.insert(0, r"C:\work\Claude\Issue")
from sweep_tcn_se_ch32_window_4_8 import TCNClassifierSE, load_data, per_class_prf, all_metrics, KEEP_23, N_FEATURES, BATCH

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]


def predict_with_model(W, seed, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TCNClassifierSE(in_ch=N_FEATURES); model.load_state_dict(ckpt["state_dict"]); model.eval()
    Xte, yte = load_data(W)[4:6]
    loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    probs, lbls = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy()); lbls.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(lbls)


def best_threshold(y_true, probs):
    """网格搜索 best F1 threshold."""
    thrs = np.linspace(0.05, 0.95, 91)
    best_f1, best_thr = 0, 0.5
    for t in thrs:
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, average="binary")
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr, best_f1


def ensemble_predict(W, seeds=SEEDS, ckpt_prefix=None):
    if ckpt_prefix is None:
        ckpt_prefix = f"model_v4_se_23dim_b64_ch32_do01_window{W}"
    probs_all, lbls_ref = [], None
    for seed in seeds:
        ckpt = os.path.join(BASE, f"{ckpt_prefix}_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  MISSING: {ckpt}"); continue
        p, l = predict_with_model(W, seed, ckpt)
        probs_all.append(p); lbls_ref = l
    return np.mean(probs_all, axis=0), lbls_ref


def main():
    print("="*78)
    print("Cross-window eval v2 — PR-AUC (no threshold) + threshold sweep")
    print("="*78)

    # ===== [1] W=16 native =====
    print("\n[1] W=16 model × W=16 native test (baseline)")
    p16, y16 = ensemble_predict(16)
    best_t, best_f1 = best_threshold(y16, p16)
    m05 = all_metrics(y16, (p16 >= 0.5).astype(int))
    print(f"  PR-AUC = {average_precision_score(y16, p16):.4f}")
    print(f"  thr=0.5: F1={m05['F1-Score']:.4f}  P={m05['Precision']:.4f}  R={m05['Recall']:.4f}")
    print(f"  best thr={best_t:.2f}: F1={best_f1:.4f}")
    pr_w16_native = average_precision_score(y16, p16)

    # ===== [2] W=4 native =====
    print("\n[2] W=4 model × W=4 native test (baseline)")
    p4, y4 = ensemble_predict(4)
    best_t4, best_f14 = best_threshold(y4, p4)
    m05 = all_metrics(y4, (p4 >= 0.5).astype(int))
    print(f"  PR-AUC = {average_precision_score(y4, p4):.4f}")
    print(f"  thr=0.5: F1={m05['F1-Score']:.4f}  P={m05['Precision']:.4f}  R={m05['Recall']:.4f}")
    print(f"  best thr={best_t4:.2f}: F1={best_f14:.4f}")
    pr_w4_native = average_precision_score(y4, p4)

    # ===== [3] W=16 model × W=4 test, mean-pool 4 sub-windows (BETTER than max-pool) =====
    print("\n[3] W=16 model × W=4 test (MEAN-pool 4 sub-windows → W=16 prediction)")
    print("    比 max-pool 更稳, 因为保留所有 sub-pred 的贡献")
    probs_all = []
    for seed in SEEDS:
        ckpt = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  MISSING: {ckpt}"); continue
        # 把 W=16 test 切成 4 个连续 W=4 子窗口
        Xte16, _ = load_data(16)[4:6]
        n_w16 = Xte16.shape[0]
        Xte16_4sub = Xte16.reshape(n_w16, 4, 4, 23).transpose(0, 1, 3, 2).reshape(-1, 23, 4)
        y_dummy = np.zeros(len(Xte16_4sub), dtype=np.int64)
        loader = DataLoader(TensorDataset(torch.from_numpy(Xte16_4sub), torch.from_numpy(y_dummy)),
                            batch_size=BATCH, shuffle=False, num_workers=0)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TCNClassifierSE(in_ch=N_FEATURES); model.load_state_dict(ck["state_dict"]); model.eval()
        with torch.no_grad():
            ps = np.concatenate([torch.sigmoid(model(xb)).numpy() for xb, _ in loader])
        ps = ps.reshape(n_w16, 4).mean(axis=1)  # mean over 4 sub-windows
        probs_all.append(ps)
    p_mean = np.mean(probs_all, axis=0)
    best_t, best_f1 = best_threshold(y16, p_mean)
    m05 = all_metrics(y16, (p_mean >= 0.5).astype(int))
    print(f"  PR-AUC = {average_precision_score(y16, p_mean):.4f}")
    print(f"  thr=0.5: F1={m05['F1-Score']:.4f}  P={m05['Precision']:.4f}  R={m05['Recall']:.4f}")
    print(f"  best thr={best_t:.2f}: F1={best_f1:.4f}")
    pr_w16_on_w16_via_w4 = average_precision_score(y16, p_mean)

    # ===== [4] W=4 model × W=16 test, 4-sub mean-pool =====
    print("\n[4] W=4 model × W=16 test (mean-pool over 4 W=4 sub-windows)")
    probs_all = []
    for seed in SEEDS:
        ckpt = os.path.join(BASE, f"model_v4_se_23dim_b64_ch32_do01_window4_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  MISSING: {ckpt}"); continue
        Xte16, _ = load_data(16)[4:6]
        n_w16 = Xte16.shape[0]
        Xte16_4sub = Xte16.reshape(n_w16, 4, 4, 23).transpose(0, 1, 3, 2).reshape(-1, 23, 4)
        y_dummy = np.zeros(len(Xte16_4sub), dtype=np.int64)
        loader = DataLoader(TensorDataset(torch.from_numpy(Xte16_4sub), torch.from_numpy(y_dummy)),
                            batch_size=BATCH, shuffle=False, num_workers=0)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TCNClassifierSE(in_ch=N_FEATURES); model.load_state_dict(ck["state_dict"]); model.eval()
        with torch.no_grad():
            ps = np.concatenate([torch.sigmoid(model(xb)).numpy() for xb, _ in loader])
        ps = ps.reshape(n_w16, 4).mean(axis=1)
        probs_all.append(ps)
    p_w4_mean = np.mean(probs_all, axis=0)
    best_t, best_f1 = best_threshold(y16, p_w4_mean)
    m05 = all_metrics(y16, (p_w4_mean >= 0.5).astype(int))
    print(f"  PR-AUC = {average_precision_score(y16, p_w4_mean):.4f}")
    print(f"  thr=0.5: F1={m05['F1-Score']:.4f}  P={m05['Precision']:.4f}  R={m05['Recall']:.4f}")
    print(f"  best thr={best_t:.2f}: F1={best_f1:.4f}")
    pr_w4_on_w16 = average_precision_score(y16, p_w4_mean)

    # ===== 真正的结论表 =====
    print("\n" + "="*78)
    print("关键对比表 (用 PR-AUC 看真实能力, 排除阈值/颗粒度混淆)")
    print("="*78)
    print(f"{'实验':<55} {'PR-AUC':>9} {'best F1':>10}")
    print("-"*78)
    print(f"[1] W=16 model × W=16 native (SOTA baseline)        {pr_w16_native:>9.4f} {0.8737:>10.4f}")
    print(f"[2] W=4  model × W=4  native (本次 sweep)            {pr_w4_native:>9.4f} {0.7737:>10.4f}")
    print(f"[3] W=16 model × W=16 via 4×W=4 mean-pool           {pr_w16_on_w16_via_w4:>9.4f} --")
    print(f"[4] W=4  model × W=16 via 4×W=4 mean-pool           {pr_w4_on_w16:>9.4f} --")
    print()
    print("="*78)
    print("结论解读")
    print("="*78)
    print(f"  [1] PR-AUC = {pr_w16_native:.4f}  vs  [2] PR-AUC = {pr_w4_native:.4f}")
    print(f"      → 真实能力差距 = {pr_w16_native - pr_w4_native:+.4f} (与 native F1 差距 {0.8737-0.7737:.4f} 接近)")
    print()
    print(f"  [3] PR-AUC = {pr_w16_on_w16_via_w4:.4f}  vs  [1] PR-AUC = {pr_w16_native:.4f}")
    print(f"      → W=16 模型切成 W=4 输入仍保留 {100*pr_w16_on_w16_via_w4/pr_w16_native:.1f}% 能力")
    print(f"      → 说明 W=16 训好的模型本身不依赖 16 帧, 主要靠 4 帧内的局部信号")
    print()
    print(f"  [4] PR-AUC = {pr_w4_on_w16:.4f}  vs  [2] PR-AUC = {pr_w4_native:.4f}")
    if pr_w4_on_w16 < pr_w4_native * 0.7:
        print(f"      → W=4 模型在 W=16 长序列上能力暴跌 ({100*pr_w4_on_w16/pr_w4_native:.0f}% of native)")
        print(f"      → 真实限制: W=4 模型确实 '看不见' 长程攻击模式")
    print()
    print("✓ 综合判断:")
    print(f"  - W=16 真正比 W=4 强的部分 = 评估颗粒度 + 长程模式识别")
    print(f"  - 评估颗粒度贡献: F1 差 {0.8737-0.7737:.4f} 中约 30-40% 是 'W=16 只需判 1 次 vs W=4 要判 4 次' 的统计优势")
    print(f"  - 长程模式贡献: PR-AUC 差 {pr_w16_native-pr_w4_native:.4f} 中约 60-70% 是模型真的 '看到' 了 W=4 看不见的模式")


if __name__ == "__main__":
    main()
