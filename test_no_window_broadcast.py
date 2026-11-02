#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_no_window_broadcast.py
================================================================
"去掉滑窗,检测模型的正确性" — Phase A: Python 端隔离测试.

目标: 不依赖 sliding_window / feature_extractor,直接给训练好的 .pt 模型喂
broadcast 当前帧 23 维特征 × 16 时间步, 看输出是否符合预期,定位 OOD/报警
问题的根因 (模型 vs 滑窗 vs 特征).

模型架构 (from retrain_tcn_23dim_b64_ch32_do01_savept.py):
  TCNClassifierSE(in_ch=23) — Conv1d k=3 dilations=[1,2,4] × 3 + SE + GAP + FC.
  输入形状 (batch, 23, 16), channels-first.

7 个场景 (5-seed 集成 + 单 seed 对照):

  # | 场景         | 输入 (23,)               | 期望         | 验证目标
  --|--------------|---------------------------|--------------|---------------------
  1 | zeros_all    | 全 0                      | 概率 ∈ [0,1] | 极端 baseline
  2 | ones_05      | 全 0.5                    | 概率 ∈ [0,1] | 常量响应
  3 | clip_pos     | 全 +10                    | 概率 ∈ [0,1] | σ-clip 上限
  4 | clip_neg     | 全 -10                    | 概率 ∈ [0,1] | σ-clip 下限
  5 | attack_like  | fc=24 (unusual), crc=RAW_MAX, is_resp=1 | prob > 0.5 | 攻击特征组合
  6 | real_normal  | X_test[y==0][0][-1][KEEP_23]            | prob < 0.5 | 训练分布 normal
  7 | real_attack  | X_test[y==1][0][-1][KEEP_23]            | prob > 0.5 | 训练分布 attack

附加对照 (场景 6/7): broadcast (单帧复制 16 份) vs 真 16 帧窗口,量化滑窗贡献 Δ.

用法:
  python test_no_window_broadcast.py
  python test_no_window_broadcast.py --save-csv results.csv
"""
import sys, os, time, argparse, csv
import numpy as np
import torch

BASE = r"C:\work\Claude\Issue"
sys.path.insert(0, BASE)
from retrain_tcn_23dim_b64_ch32_do01_savept import (
    TCNClassifierSE, SEEDS, N_FEATURES, WINDOW,
    CHANNELS, DROPOUT, KEEP_23,
)

CLIP_VAL = 10.0  # 与训练管线一致


# ── 模型加载 ───────────────────────────────────────────────────────────
def load_models(seeds=SEEDS):
    models = []
    for s in seeds:
        path = f"{BASE}/model_v4_se_23dim_b64_ch32_do01_window16_s{s}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = TCNClassifierSE(in_ch=N_FEATURES)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append((s, m))
    return models


# ── 输入构造 ───────────────────────────────────────────────────────────
def make_scenarios(X_test, y_test):
    """返回 [(name, feat23), ...] — 每个 feat23 是已归一化的 23 维向量."""
    scenarios = []

    # 场景 1-4: 极端 + 常量 baseline
    scenarios.append(("zeros_all", np.zeros(23, dtype=np.float32)))
    scenarios.append(("ones_05",  np.full(23, 0.5, dtype=np.float32)))
    scenarios.append(("clip_pos", np.full(23, +CLIP_VAL, dtype=np.float32)))
    scenarios.append(("clip_neg", np.full(23, -CLIP_VAL, dtype=np.float32)))

    # 场景 5: 攻击特征组合 (训练管线归一化后)
    # f[1]=function=24 (unusual_fc), f[12]=crc_rate=RAW_MAX (≈17990), f[16]=is_response=1
    # 注意: 这是 RAW 输入空间,会经 RobustScaler → 归一化后送模型. 但场景 5 是合成
    # 测试, 跳过 scaler 直接用归一化域构造 "肯定攻击" 模式.
    # 用一种更直接的方式: 取 X_test[y==1] 平均特征作为 "典型攻击" 模式
    att_mean = X_test[y_test == 1].mean(axis=(0, 1))  # (23,)
    scenarios.append(("attack_like", att_mean.astype(np.float32)))

    # 场景 6: 训练分布内的 normal 末帧 (X_test 是 (N, 16, 23) 形态, channels-first 视 T 维度)
    # 注意: 原 X_test 是 channels-LAST (N, 16, 27), retrain 脚本 line 138 transpose (0,2,1).
    # 我们从原始 .npy 加载后, 取 KEEP_23 子集, 不 transpose, 直接用末帧 (t=15).
    nrm_end = X_test[y_test == 0][0, -1, :].astype(np.float32)  # (23,) already KEEP_23? 详见下
    scenarios.append(("real_normal", nrm_end))

    # 场景 7: 训练分布内的 attack 末帧
    atk_end = X_test[y_test == 1][0, -1, :].astype(np.float32)
    scenarios.append(("real_attack", atk_end))

    return scenarios


# ── broadcast (1, 23) → (1, 23, 16) ────────────────────────────────────
def to_window_input(feat23: np.ndarray) -> np.ndarray:
    """单帧 23 维 → channels-first (1, 23, 16), 复制 16 份. clip ±CLIP_VAL 保持训练分布."""
    return np.clip(np.tile(feat23, (WINDOW, 1)).T[np.newaxis, :, :], -CLIP_VAL, CLIP_VAL).astype(np.float32)


# ── 推理 ──────────────────────────────────────────────────────────────
def infer_one(models, x: np.ndarray) -> tuple:
    """x: (1, 23, 16) FP32 → (probs (5,), logits (5,)) — 5-seed 一一对应."""
    probs, logits = [], []
    with torch.no_grad():
        for seed, m in models:
            logit = m(torch.from_numpy(x)).item()
            probs.append(1.0 / (1.0 + np.exp(-logit)))
            logits.append(logit)
    return (np.array(probs, dtype=np.float32),
            np.array(logits, dtype=np.float32))


def infer_window16(models, win: np.ndarray) -> tuple:
    """win: (16, 23) channels-LAST FP32 (原始 .npy 形态) → (1, 23, 16) channels-first."""
    x = win.T[np.newaxis, :, :].astype(np.float32)
    return infer_one(models, x)


# ── 主函数 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-csv", default=None,
                    help="把结果落盘到 CSV (per-seed + ensemble)")
    ap.add_argument("--no-window16-compare", action="store_true",
                    help="跳过场景 6/7 的 broadcast vs window16 对照")
    args = ap.parse_args()

    print("=" * 72)
    print("Phase A: 23-dim TCN+SE broadcast 测试 (无滑窗)")
    print("=" * 72)
    t0 = time.time()

    # 加载 5 个 .pt
    models = load_models()
    print(f"[load] 加载 {len(models)} 个 .pt 模型: seeds={[s for s, _ in models]}")
    print(f"[load] 耗时 {time.time()-t0:.1f}s")

    # 加载测试数据 (原始 channels-last, NOT transposed)
    X = np.load(f"{BASE}/X_test_binary_v2_scada_window16.npy").astype(np.float32)  # (3432, 16, 27)
    y = np.load(f"{BASE}/y_test_binary_v2_scada_window16.npy").astype(np.int64)    # (3432,)
    print(f"[data] X_test shape={X.shape}, y_test pos_rate={y.mean():.4f}")
    # 取 KEEP_23 子集 (drop length/setpoint/crc_mean_w/cmd_count_w)
    X23 = X[:, :, KEEP_23]  # (N, 16, 23)
    print(f"[data] X23 (KEEP_23) shape={X23.shape}")

    # 构造 7 个场景
    scenarios = make_scenarios(X23, y)

    # 推理循环
    rows = []
    print()
    print("─" * 72)
    print(f"{'scenario':<18} | {'seed=42':<22} | {'seed=123':<22} | {'seed=456':<22} | "
          f"{'seed=789':<22} | {'seed=1024':<22} | {'ens_mean±std':<14}")
    print("─" * 72)
    for name, feat23 in scenarios:
        # clip 已在 to_window_input 内, 但 scenarios 1-4 是合成极端值, 部分会被 clip 截断
        # (这是预期的: clip_pos=[+10]+10, clip_neg=[-10]+10 等会原样; zeros_all 不会被截断)
        x = to_window_input(feat23)  # (1, 23, 16)
        probs, logits = infer_one(models, x)  # (5,)
        mean = float(probs.mean())
        std = float(probs.std())
        per_seed = " | ".join(f"p={p:.3f}/L={logits[i]:+6.2f}"
                              for i, p in enumerate(probs))
        print(f"{name:<18} | {per_seed} | {mean:.4f}+/-{std:.4f}")
        rows.append({"scenario": name, "seed_42": float(probs[0]),
                     "seed_123": float(probs[1]), "seed_456": float(probs[2]),
                     "seed_789": float(probs[3]), "seed_1024": float(probs[4]),
                     "ens_mean": mean, "ens_std": std})

    # 场景 6/7: broadcast vs 真 16 帧窗口对照
    if not args.no_window16_compare:
        print()
        print("─" * 72)
        print("场景 6/7 对照: broadcast (单帧 ×16) vs 真 16 帧窗口")
        print("─" * 72)
        nrm_window = X23[y == 0][0]  # (16, 23)
        atk_window = X23[y == 1][0]  # (16, 23)

        nrm_bc = to_window_input(nrm_window[-1])  # 末帧 broadcast
        atk_bc = to_window_input(atk_window[-1])
        nrm_win_p, _ = infer_window16(models, nrm_window)
        atk_win_p, _ = infer_window16(models, atk_window)
        nrm_bc_p, _ = infer_one(models, nrm_bc)
        atk_bc_p, _ = infer_one(models, atk_bc)

        d_nrm = abs(float(nrm_bc_p.mean()) - float(nrm_win_p.mean()))
        d_atk = abs(float(atk_bc_p.mean()) - float(atk_win_p.mean()))
        print(f"{'case':<18} | {'window16 (ens_mean)':<22} | {'broadcast (ens_mean)':<22} | {'|diff|':<8}")
        print("-" * 72)
        print(f"{'real_normal':<18} | {nrm_win_p.mean():<22.4f} | {nrm_bc_p.mean():<22.4f} | {d_nrm:<8.4f}")
        print(f"{'real_attack':<18} | {atk_win_p.mean():<22.4f} | {atk_bc_p.mean():<22.4f} | {d_atk:<8.4f}")

        rows.append({"scenario": "real_normal_window16", "ens_mean": float(nrm_win_p.mean()),
                     "ens_std": float(nrm_win_p.std())})
        rows.append({"scenario": "real_attack_window16", "ens_mean": float(atk_win_p.mean()),
                     "ens_std": float(atk_win_p.std())})

    # 多样本评估: 测试 broadcast vs window16 在 FULL test set (3432) 上的 F1 / Acc
    print()
    print("=" * 72)
    print(f"Full test set ({len(y)} samples) 评估: broadcast vs window16")
    print("=" * 72)

    # broadcast: 取每个窗口的末帧, 复制 16 份
    nrm_windows_all = X23  # (3432, 16, 23)
    bc_input = np.array([to_window_input(w[-1])[0] for w in nrm_windows_all])  # (3432, 23, 16)
    # window16: 转置到 channels-first
    win_input = np.clip(nrm_windows_all, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)  # (3432, 23, 16)
    # ⭐ 注意: 训练管线 retrain_tcn_23dim_b64_ch32_do01_savept.py:138 在 forward 前 np.clip(-CLIP_VAL, CLIP_VAL).
    # 不 clip → BN 收到超出训练分布的值 → 模型饱和输出 1.0. broadcast 路径在 to_window_input 内已 clip.

    # 批量推理
    def batch_infer(arr):
        """arr: (N, 23, 16) FP32 → (5, N) probs — 5 seeds × N samples."""
        with torch.no_grad():
            x = torch.from_numpy(arr.astype(np.float32))
            logits_per_seed = []
            for seed, m in models:
                logits = m(x).numpy()  # (N,)
                logits_per_seed.append(logits)
        logits_all = np.array(logits_per_seed)  # (5, N)
        probs_all = 1.0 / (1.0 + np.exp(-logits_all))  # (5, N)
        return probs_all

    bc_p = batch_infer(bc_input)   # (5, 3432)
    win_p = batch_infer(win_input)  # (5, 3432)

    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score
    print(f"\n{'config':<26} | {'Acc':<8} | {'P':<8} | {'R':<8} | {'F1':<8} | "
          f"{'PR-AUC':<8} | {'nrm_prob':<14} | {'atk_prob':<14} | {'sep':<8}")
    print("-" * 116)
    for name, probs_all in [("broadcast 末帧 ×16", bc_p), ("window16 全窗口", win_p)]:
        # 5-seed ensemble
        ens = probs_all.mean(axis=0)  # (3432,)
        y_pred = (ens >= 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        p = precision_score(y, y_pred, zero_division=0)
        r = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred)
        pr_auc = average_precision_score(y, ens)
        nrm_mean = ens[y == 0].mean()
        atk_mean = ens[y == 1].mean()
        sep = atk_mean - nrm_mean
        print(f"{name:<26} | {acc:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} | "
              f"{pr_auc:.4f} | {nrm_mean:.4f}+/-{ens[y==0].std():.4f} | "
              f"{atk_mean:.4f}+/-{ens[y==1].std():.4f} | {sep:+.4f}")
        # per-seed F1
        f1_per_seed = []
        for s in range(5):
            ps = probs_all[s]
            yp = (ps >= 0.5).astype(int)
            f1_per_seed.append(f1_score(y, yp))
        print(f"  per-seed F1: " + ", ".join(f"s{i}={f:.4f}" for i, f in enumerate(f1_per_seed)))

    # 落盘
    if args.save_csv:
        keys = list({k for r in rows for k in r.keys()})
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[save] 写入 {args.save_csv} ({len(rows)} rows)")

    # 自动判定结果 (按 plan 的分类表)
    print()
    print("=" * 72)
    print("判定")
    print("=" * 72)
    by_name = {r["scenario"]: r for r in rows}
    nrm_mean = by_name.get("real_normal", {}).get("ens_mean", float("nan"))
    atk_mean = by_name.get("real_attack", {}).get("ens_mean", float("nan"))
    nrm_std = by_name.get("real_normal", {}).get("ens_std", 1.0)
    atk_std = by_name.get("real_attack", {}).get("ens_std", 1.0)
    nrm_win_mean = by_name.get("real_normal_window16", {}).get("ens_mean", None)
    atk_win_mean = by_name.get("real_attack_window16", {}).get("ens_mean", None)

    def verdict():
        if nrm_mean < 0.5 and atk_mean > 0.5:
            if nrm_win_mean is not None and atk_win_mean is not None:
                delta = (abs(nrm_mean - nrm_win_mean) + abs(atk_mean - atk_win_mean)) / 2
            else:
                delta = float("nan")
            if delta < 0.1:
                return ("PASS", f"模型对单帧敏感, broadcast 工作 (Δ={delta:.3f}) → "
                                "可触发 Phase B 改 H743 sliding_window_push")
            else:
                return ("FAIL-window-dependent",
                        f"模型工作但高度依赖滑窗 (Δ={delta:.3f}) → 排查其它层")
        if 0.4 < nrm_mean < 0.6 and 0.4 < atk_mean < 0.6:
            return ("FAIL-degenerate",
                    "模型在 broadcast 下完全退化 (~0.5) → 重新训练 window=1 模型")
        if nrm_mean > 0.5 and atk_mean < 0.5:
            return ("FAIL-reversed", "模型输出颠倒 (normal>attack) → 检查 .pt 加载 / label")
        return ("AMBIGUOUS",
                f"nrm={nrm_mean:.3f} atk={atk_mean:.3f} — 需要人工诊断")

    label, msg = verdict()
    if label == "PASS":
        icon = "[OK]"
    elif "FAIL" in label:
        icon = "[FAIL]"
    else:
        icon = "[??]"
    print(f"{icon} {label}: {msg}")
    print(f"   real_normal broadcast = {nrm_mean:.4f} +/- {nrm_std:.4f}")
    print(f"   real_attack broadcast = {atk_mean:.4f} +/- {atk_std:.4f}")
    if nrm_win_mean is not None:
        print(f"   real_normal window16  = {nrm_win_mean:.4f}")
        print(f"   real_attack window16  = {atk_win_mean:.4f}")
    print()
    print(f"[done] 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
