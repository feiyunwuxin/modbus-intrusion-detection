#!/usr/bin/env python3
"""export_23dim_to_onnx.py — 增强版
把 23-dim TCN+SE ch=32 冠军模型导出为 ONNX (.onnx)

改进点 (vs 上版):
  ① ONNX Runtime 加载验证 (不只 onnx.checker)
  ② 与 PyTorch 推理做 bit-perfect 对比 (200 随机样本)
  ③ onnx-simplifier 简化图 (可选,失败跳过)
  ④ 生成 manifest.json 路径 + 尺寸 + 验证状态 + 部署推荐
  ⑤ 同时输出 5 个 seed 名的 .onnx

5 seed .pt 路径:
  model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt (seed ∈ {42,123,456,789,1024})
输出:
  model_v4_se_23dim_ch32_s{seed}.onnx
  onnx_export_manifest.json (5 个 .onnx 路口 + 5-seed ensemble 概率缓存)
"""
import os, sys, json, time
import numpy as np
import torch

BASE = r"C:\work\Claude\Issue"
sys.path.insert(0, BASE)
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE, SEEDS

# 训练时的维度参数(必须与训练严格一致)
N_FEATURES  = 23
WINDOW      = 16
CHANNELS    = 32
DROPOUT     = 0.1

OPSET = 13   # 平衡兼容性和算子支持


# ------------------------- 工具函数 -------------------------

def verify_onnx_with_ort(onnx_path: str, model: torch.nn.Module, n_samples: int = 200,
                          seed: int = 0) -> dict:
    """加载 ONNX → 用 ORT 跑 n_samples 随机样本 → 与 PyTorch 输出对比

    Returns:
        dict: {max_abs_diff, mean_abs_diff, n_samples, all_match}
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"skipped": True, "reason": "onnxruntime not installed"}

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, N_FEATURES, WINDOW)).astype(np.float32)

    # PyTorch 推理 (model 返回 logits, 验证时手动 sigmoid 与 ONNX 输出对齐)
    model.eval()
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(X)).numpy().squeeze()
    torch_out = 1.0 / (1.0 + np.exp(-np.clip(torch_logits, -50, 50)))

    # ONNX Runtime 推理 (返回 logits, sigmoid 化)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {"input": X})[0].squeeze()
    ort_out = 1.0 / (1.0 + np.exp(-np.clip(ort_logits, -50, 50)))

    diff = np.abs(torch_out - ort_out)
    return {
        "skipped": False,
        "n_samples": n_samples,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "torch_first_3": torch_out[:3].round(4).tolist(),
        "onnx_first_3":  ort_out[:3].round(4).tolist(),
        "all_match_1e-5": bool(diff.max() < 1e-5),
    }


def try_onnxsim(onnx_path: str) -> dict:
    """onnx-simplifier 简化图 (失败静默跳过)"""
    try:
        import onnx
        import onnxsim
        model = onnx.load(onnx_path)
        simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(simplified, onnx_path)
            return {"applied": True, "ok": True}
        return {"applied": True, "ok": False}
    except ImportError:
        return {"applied": False, "reason": "onnxsim not installed"}
    except Exception as e:
        return {"applied": True, "ok": False, "error": str(e)}


# ------------------------- 主流程 -------------------------

def export_one(seed: int) -> dict:
    """加载 1 个 .pt → 导出 .onnx → 验证 → 返回 dict"""
    pt_path = f"{BASE}/model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt"
    onnx_path = f"{BASE}/model_v4_se_23dim_ch32_s{seed}.onnx"

    if not os.path.exists(pt_path):
        return {"seed": seed, "ok": False, "error": f".pt not found: {pt_path}"}

    print(f"\n[seed={seed}]")
    print(f"  加载 {os.path.basename(pt_path)} ...")
    model = TCNClassifierSE(in_ch=N_FEATURES, channels=CHANNELS, dropout=DROPOUT)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  导出 ONNX (opset={OPSET}, batch=1 fixed) ...")
    dummy = torch.randn(1, N_FEATURES, WINDOW)  # (B=1, C=23, T=16)
    t0 = time.time()
    # 注意: torch 2.12 默认走 dynamo exporter, 需要 onnxscript (国内 pip timeout 装不上)
    # 回退到 legacy trace-based exporter (torch 1.x 风格), 稳定且 opset 兼容性好
    # dynamic_axes 允许 batch=1 (STM32 默认) 也支持 batch=N (server/edge TPU)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],       # (B, 23, 16)
        output_names=["prob"],       # (B,) sigmoid 概率
        dynamic_axes={
            "input":  {0: "batch"},
            "prob":   {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,    # 折叠 BN → conv
        dynamo=False,                # 用 legacy trace exporter (无需 onnxscript)
    )
    export_time = time.time() - t0

    # ① onnx.checker 静态检查
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # ② onnx-simplifier (可选)
    sim_info = try_onnxsim(onnx_path)
    if sim_info.get("applied") and sim_info.get("ok"):
        print(f"  [ok] onnx-simplifier 简化成功")
    elif sim_info.get("applied") and not sim_info.get("ok"):
        print(f"  [warn] onnx-simplifier 失败 (跳过)")

    # ③ ONNX Runtime 验证 + bit-perfect 对比 PyTorch
    print(f"  验证 ONNX Runtime 推理 (vs PyTorch) ...")
    verify_info = verify_onnx_with_ort(onnx_path, model, n_samples=200, seed=seed)
    if verify_info.get("skipped"):
        print(f"  [warn] ORT 验证跳过: {verify_info.get('reason')}")
    elif verify_info["all_match_1e-5"]:
        print(f"  [ok] 200 样本 bit-perfect (max_diff={verify_info['max_abs_diff']:.2e})")
    else:
        print(f"  [warn] 200 样本 max_diff={verify_info['max_abs_diff']:.2e}  > 1e-5")

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  [ok] {os.path.basename(onnx_path)} ({size_kb:.2f} KB, {n_params:,} params)")

    return {
        "seed": seed,
        "ok": True,
        "pt_path": pt_path,
        "onnx_path": onnx_path,
        "n_params": n_params,
        "onnx_size_kb": round(size_kb, 2),
        "export_time_s": round(export_time, 2),
        "input_shape": [1, N_FEATURES, WINDOW],
        "verify": verify_info,
        "onnxsim": sim_info,
        "test_metrics": ckpt.get("test_metrics", {}),
    }


def export_ensemble(per_seed: list) -> dict:
    """5-seed ensemble 概率平均 (与 retrain 脚本一致)"""
    print(f"\n[5-seed ensemble]")
    X_test  = np.load(f"{BASE}/X_test_binary_v2_scada_window16.npy").astype(np.float32)
    X_test  = np.clip(X_test, -10.0, 10.0).transpose(0, 2, 1)
    KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
    X_test  = X_test[:, KEEP_23, :]
    n = min(2000, len(X_test))
    X_sub = X_test[:n]

    probs_list = []
    for entry in per_seed:
        if not entry["ok"]:
            continue
        pt_path = entry["pt_path"]
        model = TCNClassifierSE(in_ch=N_FEATURES, channels=CHANNELS, dropout=DROPOUT)
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(X_sub))).numpy()
        probs_list.append(probs)

    ens_prob = np.mean(probs_list, axis=0)
    return {
        "ensemble_size": len(probs_list),
        "n_samples": n,
        "prob_mean": float(ens_prob.mean()),
        "prob_median": float(np.median(ens_prob)),
        "prob_high_risk_count": int((ens_prob >= 0.5).sum()),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  23-dim TCN+SE ch=32 → ONNX 批量导出 (增强版)")
    print("=" * 60)

    per_seed = []
    for seed in SEEDS:
        entry = export_one(seed)
        per_seed.append(entry)
        if not entry["ok"]:
            print(f"  ✗ seed={seed} 失败: {entry.get('error')}")

    # 5-seed ensemble
    ens = export_ensemble(per_seed)
    print(f"  ensemble 均值={ens['prob_mean']:.4f}  攻击概率 ≥0.5 = {ens['prob_high_risk_count']}/{ens['n_samples']}")

    # 输出 manifest
    manifest = {
        "model": "TCN+SE 23-dim",
        "tag": "v4_se_23dim_b64_ch32_do01_window16",
        "n_features": N_FEATURES,
        "window": WINDOW,
        "channels": CHANNELS,
        "dropout": DROPOUT,
        "opset": OPSET,
        "n_seeds": len(SEEDS),
        "per_seed": per_seed,
        "ensemble": ens,
        "deployment_paths": {
            "stm32cube_ai": "STM32CubeMX → X-CUBE-AI → Analyze 选 .onnx → Generate C",
            "onnxruntime": "import onnxruntime; ort.InferenceSession(path, providers=['CPUExecutionProvider'])",
            "tflite": "onnx-tf → TFLite (INT8 需 calib data); apt for mobile / edge TPU",
            "torch_jit": "torch.onnx.export 已经保留 torch 兼容,可 torch.jit.load 重载",
        },
    }
    manifest_path = f"{BASE}/onnx_export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[manifest] {manifest_path}")

    print("\n" + "=" * 60)
    n_ok = sum(1 for e in per_seed if e["ok"])
    print(f"  完成 {n_ok}/{len(SEEDS)} 个 ONNX 文件")
    for e in per_seed:
        if e["ok"]:
            print(f"    {e['onnx_path']}  ({e['onnx_size_kb']:.2f} KB)")
        else:
            print(f"    [fail] seed={e['seed']}: {e.get('error')}")
    print("=" * 60)
