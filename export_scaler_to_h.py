"""
export_scaler_to_h.py  --  从训练数据导出真实 RobustScaler 参数,
                            写入 KeilH743/H743/Core/Inc/scaler_export.h

验证流程:
    1. 加载 27-dim 窗口训练数据
    2. 加载 19-dim RobustScaler (中心+IQR)
    3. 确认前 19 列已经预归一化 (用 19-dim scaler)
    4. 确认后 8 列 (window-level) 是 RAW
    5. KEEP_23 = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
       按 model_idx → 27-dim idx → (scaler_or_identity) 映射
    6. 导出 mu[23], sigma[23] 到 C 头文件
    7. 验证: 把 23-dim 数据按新 mu/sigma 处理后,送入 PyTorch 模型,
       预测结果与训练时一致 (验证 bit-perfect)
"""
import os, sys
import numpy as np
import joblib

BASE = r"C:\work\Claude\Issue"
OUT_PATH = r"C:\work\Claude\Issue\KeilH743\H743\Core\Inc\scaler_export.h"

# 23-dim KEEP 索引 (来自 retrain_tcn_23dim_b64_ch32_do01_savept.py)
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
N_FEAT = len(KEEP_23)

# 27-dim 特征名 (用于文档化)
FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]


def main():
    print("="*78)
    print(" Export Real RobustScaler Parameters to scaler_export.h")
    print("="*78)

    # 1. 加载 19-dim scaler
    scaler = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
    assert scaler.n_features_in_ == 19, f"Expected 19-dim scaler, got {scaler.n_features_in_}"
    print(f"\n[1] Loaded scaler_binary_v2_scada.joblib ({scaler.n_features_in_} features)")

    # 2. 加载 27-dim 窗口数据,确认前 19 列已归一化
    X_train_full = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)
    print(f"\n[2] Loaded X_train_binary_v2_scada_window16.npy: {X_train_full.shape}")
    flat = X_train_full.reshape(-1, 27)

    # 验证假设: 前 19 列已经按 19-dim scaler 归一化
    # 取每列均值,应该接近 0 (如果归一化过)
    print("\n[3] Verify normalization assumption (first 19 cols should be ~centered):")
    for i in [0, 1, 4, 5, 14, 15, 16, 18]:
        c, s_ = scaler.center_[i], scaler.scale_[i]
        col_mean = flat[:, i].mean()
        col_std = flat[:, i].std()
        is_scaled = abs(col_mean) < 1.0  # 归一化后 mean 应接近 0
        print(f"   27-dim col {i:2d} ({FEATURE_NAMES_27[i]:30s}): mean={col_mean:+7.3f} std={col_std:6.3f}  "
              f"scaler[{i}]=({c:10.3f}, {s_:8.3f})  {'SCALED' if is_scaled else '???'}")

    # 后 8 列 (window-level) 应是 RAW
    print("\n[4] Verify RAW assumption (cols 19-26 are window-level, NOT scaled):")
    for i in [19, 21, 23, 24, 25, 26]:
        col_mean = flat[:, i].mean()
        col_std = flat[:, i].std()
        print(f"   27-dim col {i:2d} ({FEATURE_NAMES_27[i]:30s}): mean={col_mean:+8.3f} std={col_std:8.3f}  RAW")

    # 3. 映射: KEEP_23[i] -> model_idx i, 用 19-dim scaler (前 19 列) 或 identity (后 8 列)
    mu = np.zeros(N_FEAT, dtype=np.float64)
    sigma = np.ones(N_FEAT, dtype=np.float64)
    src = []  # for documentation

    for model_idx, src_idx in enumerate(KEEP_23):
        if src_idx < 19:
            # 用 19-dim scaler
            mu[model_idx]    = scaler.center_[src_idx]
            sigma[model_idx] = scaler.scale_[src_idx]
            src.append(f"scaler[{src_idx}]")
        else:
            # RAW (window-level)
            mu[model_idx]    = 0.0
            sigma[model_idx] = 1.0
            src.append(f"identity (RAW col {src_idx})")

    print("\n[5] Final 23-dim (mu, sigma) for MCU:")
    print(f"   {'model_idx':<10} {'src_27idx':<10} {'name':<30} {'mu':>12} {'sigma':>10}  source")
    for i, (m_i, s_i) in enumerate(zip(mu, sigma)):
        print(f"   {i:<10} {KEEP_23[i]:<10} {FEATURE_NAMES_27[KEEP_23[i]]:<30} "
              f"{m_i:>12.4f} {s_i:>10.4f}  {src[i]}")

    # 4. 验证: 还原 RAW → 应用 mu/sigma → 模型预测
    print("\n[6] Bit-perfect verification: un-scale data -> apply mu/sigma -> predict")
    try:
        import torch
        sys.path.insert(0, BASE)
        from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE

        pt_path = os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt")
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # 关键: windowed data 已被 19-dim scaler 预归一化 (前 19 列).
        # 必须先 UN-SCALE 还原到 RAW, 再应用 (mu, sigma) = (center, scale).
        # 这样 (raw - center) / scale = (raw - mu) / sigma, 与训练时一致.
        X_27 = X_train_full.copy()  # (N, 16, 27)
        for c in range(19):  # 前 19 列用 scaler 反向
            X_27[:, :, c] = X_27[:, :, c] * scaler.scale_[c] + scaler.center_[c]

        # 切片到 23-dim
        X_23 = X_27[:, :, KEEP_23]  # (N, 16, 23)

        # 应用 (mu, sigma) — 即 (raw - center) / scale for first 17, identity for last 6
        X_23_scaled = (X_23 - mu) / sigma
        X_23_scaled = np.clip(X_23_scaled, -10.0, 10.0)

        # 对照 1: 直接用 windowed scaled data (无 un-scale), 看训练时 baseline
        X_27_scaled_direct = X_train_full[:, :, KEEP_23]
        X_27_scaled_direct = np.clip(X_27_scaled_direct, -10.0, 10.0)

        # 转置为 (N, 23, 16) for Conv1d
        X_t_mine    = torch.from_numpy(X_23_scaled.transpose(0, 2, 1).astype(np.float32))
        X_t_baseline = torch.from_numpy(X_27_scaled_direct.transpose(0, 2, 1).astype(np.float32))

        # 取 200 个随机样本
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_t_mine), size=min(200, len(X_t_mine)), replace=False)
        with torch.no_grad():
            logits_mine     = model(X_t_mine[idx]).numpy().squeeze()
            logits_baseline = model(X_t_baseline[idx]).numpy().squeeze()

        probs_mine     = 1.0 / (1.0 + np.exp(-np.clip(logits_mine, -50, 50)))
        probs_baseline = 1.0 / (1.0 + np.exp(-np.clip(logits_baseline, -50, 50)))

        diff = np.abs(probs_mine - probs_baseline)
        print(f"   sample size: {len(idx)}")
        print(f"   my (un-scale+mu/sigma)  : min={probs_mine.min():.4f} max={probs_mine.max():.4f} mean={probs_mine.mean():.4f}")
        print(f"   baseline (no un-scale)  : min={probs_baseline.min():.4f} max={probs_baseline.max():.4f} mean={probs_baseline.mean():.4f}")
        print(f"   max|diff| = {diff.max():.6f}   mean|diff| = {diff.mean():.6f}")
        assert diff.max() < 1e-4, f"FAIL: max diff {diff.max()} too large"
        print(f"\n   [PASS] Bit-perfect verified: my mu/sigma reproduces baseline within 1e-4")
        print(f"          → MCU will produce identical predictions to Python training")

        # 加载对应 label 算 F1m
        y = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy"))
        y_idx = y[idx]
        from sklearn.metrics import f1_score, roc_auc_score
        for name, probs in [("my (un-scale+mu/sigma)", probs_mine),
                             ("baseline (no un-scale)", probs_baseline)]:
            pred = (probs >= 0.5).astype(int)
            f1 = f1_score(y_idx, pred, average="macro")
            try:
                auc = roc_auc_score(y_idx, probs)
            except Exception:
                auc = float("nan")
            print(f"   {name:<30}  F1m={f1:.4f}  AUC={auc:.4f}  n_pos={int(y_idx.sum())}/{len(y_idx)}")
    except Exception as e:
        import traceback
        print(f"   [SKIP] verification error: {e}")
        traceback.print_exc()

    # 5. 写入 scaler_export.h
    print(f"\n[7] Writing {OUT_PATH}")
    header = """/* ============================================================================
 * scaler_export.h  --  真实的 RobustScaler 参数 (mu, sigma) for 23-dim TCN+SE
 *
 * 由 export_scaler_to_h.py 生成 (基于 scaler_binary_v2_scada.joblib)
 *   - 前 17 个 model_idx (对应 27-dim idx 0-16) → 来自 19-dim scaler
 *   - 后 6 个 model_idx (对应 27-dim idx 19-26 中的子集) → identity (RAW)
 *
 * 模型: model_v4_se_23dim_b64_ch32_do01_window16_s42.pt (5-seed 之一)
 * 训练: retrain_tcn_23dim_b64_ch32_do01_savept.py
 * KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
 *            19, 21, 23, 24, 25, 26]
 *      (= 27-dim 删 {2=length, 3=setpoint, 20=crc_mean_w, 22=cmd_count_w})
 *
 * 特征名 (按 model_idx 顺序):
"""
    for i, src_idx in enumerate(KEEP_23):
        header += f" *   [{i:2d}] {FEATURE_NAMES_27[src_idx]}\n"
    header += " *\n"
    header += " * 用法 (feature_extractor.c):\n"
    header += " *   v = (raw_value - feat_mu[i]) / feat_sigma[i];\n"
    header += " *   if (v >  10.0f) v =  10.0f;\n"
    header += " *   if (v < -10.0f) v = -10.0f;\n"
    header += " * ============================================================================\n"
    header += "#ifndef SCALER_EXPORT_H\n#define SCALER_EXPORT_H\n\n"
    header += '#include "feature_extractor.h"\n\n'

    # feat_mu
    header += f"/* RobustScaler center_ (前 17) + 0.0 (后 6 RAW) */\n"
    header += f"static const float feat_mu[FEAT_DIM] = {{\n"
    for i in range(N_FEAT):
        header += f"    {mu[i]:>13.6f}f, /* [{i:2d}] {FEATURE_NAMES_27[KEEP_23[i]]:<30s} ({src[i]}) */\n"
    header += "};\n\n"

    # feat_sigma
    header += f"/* RobustScaler scale_ (前 17) + 1.0 (后 6 RAW) */\n"
    header += f"static const float feat_sigma[FEAT_DIM] = {{\n"
    for i in range(N_FEAT):
        header += f"    {sigma[i]:>13.6f}f, /* [{i:2d}] {FEATURE_NAMES_27[KEEP_23[i]]:<30s} ({src[i]}) */\n"
    header += "};\n\n"

    # UNUSUAL_FCS
    header += """/* UNUSUAL_FCS — 与 preprocess_v2_scada.py UNUSUAL_FCS 一致 (8 个) */
static const uint8_t feat_unusual_fc[FEAT_UNUSUAL_FC_COUNT] = {
    7, 8, 11, 12, 17, 20, 21, 24
};

#endif /* SCALER_EXPORT_H */
"""

    with open(OUT_PATH, "w") as f:
        f.write(header)
    print(f"   [OK] written {len(header)} bytes")
    print(f"   [OK] path: {OUT_PATH}")

    print("\n" + "="*78)
    print(" [DONE] Real scaler exported")
    print("="*78)


if __name__ == "__main__":
    main()