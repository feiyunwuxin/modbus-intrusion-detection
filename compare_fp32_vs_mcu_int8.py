#!/usr/bin/env python3
"""Compare FP32 PyTorch reference model vs MCU Hybrid INT8 .h header.

Loads:
  - model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt  (FP32 reference)
  - model_v4_se_23dim_ch32_hybrid_s{seed}.h              (MCU deployed)

Runs both on the same N test samples, compares probabilities and verdict
agreement. Should be near bit-perfect (< 1e-5 max diff) since C code
dequantizes INT8 to FP32 at init time.

Usage:
  python compare_fp32_vs_mcu_int8.py --n 200 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--data", type=str,
                        default="X_test_binary_v2_scada_window16.npy",
                        help="Test data .npy file. Accepted shapes: "
                             "(N, 16, 27) 27-dim windowed, "
                             "(N, 23) flat 23-dim, or "
                             "(N, 23, 16) ready-made.")
    parser.add_argument("--labels", type=str,
                        default="y_test_binary_v2_scada_window16.npy",
                        help="Test labels .npy file")
    args = parser.parse_args()

    repo = Path(__file__).parent
    fp32_pt = repo / f"model_v4_se_23dim_b64_ch32_do01_window16_s{args.seed}.pt"
    mcu_h = repo / "KeilH743" / "H743" / "Core" / "Inc" / \
            f"model_v4_se_23dim_ch32_hybrid_s{args.seed}.h"

    if not fp32_pt.exists():
        sys.exit(f"FP32 model not found: {fp32_pt}")
    if not mcu_h.exists():
        sys.exit(f"MCU header not found: {mcu_h}")

    # Load test data
    X_path = repo / args.data
    y_path = repo / args.labels
    if not X_path.exists() or not y_path.exists():
        sys.exit(f"Test data not found: {X_path} / {y_path}")
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    window = 16
    # Build (N, 23, 16) windows
    if X.ndim == 3 and X.shape[-1] == 27:
        # 27-dim windowed data: extract MCU's 23-dim feature indices,
        # then transpose (N, 16, 23) -> (N, 23, 16).
        MCU_IDX_27 = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 21, 23, 24, 25, 26)
        Xw = X[:, :, MCU_IDX_27].transpose(0, 2, 1)
        yw = y
    elif X.ndim == 2 and X.shape[1] == 23:
        n_windows = (len(X) // window) * window
        Xw = X[:n_windows].reshape(-1, window, 23).transpose(0, 2, 1)
        yw = (y[:n_windows].reshape(-1, window).sum(axis=1) >= 1).astype(np.int64)
    else:
        Xw = X
        yw = y

    Xw = Xw[:args.n]
    yw = yw[:args.n]
    print(f"Test set: N={len(Xw)} windows of shape {Xw.shape[1:]}")

    # Load FP32 model
    import torch
    from quantize_v4se_23dim_ch32_to_h import TCNClassifierSE
    ckpt = torch.load(fp32_pt, map_location="cpu", weights_only=False)
    fp32_model = TCNClassifierSE(in_ch=23, channels=32, n_blocks=3,
                                  dilations=(1, 2, 4), dropout=0.1)
    fp32_model.load_state_dict(ckpt["state_dict"])
    fp32_model.eval()

    # Load MCU wrapper
    sys.path.insert(0, str(repo))
    from ids.model_loader import _McuHeaderWrapper
    mcu_wrapper = _McuHeaderWrapper(mcu_h, window_size=16)

    # Run inference
    fp32_probs = []
    mcu_probs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(Xw)):
            x = torch.from_numpy(Xw[i:i+1]).float()
            logit = fp32_model(x).squeeze().item()
            fp32_probs.append(1.0 / (1.0 + np.exp(-logit)))
    fp32_time = time.time() - t0

    t0 = time.time()
    for i in range(len(Xw)):
        # Pass the window directly to MCU wrapper: shape (23, 16) -> (16, 23)
        window_feats = Xw[i].T  # (16, 23) for wrapper.infer's 2D path
        label, prob = mcu_wrapper.infer(window_feats)
        mcu_probs.append(prob)
    mcu_time = time.time() - t0

    fp32_probs = np.array(fp32_probs)
    mcu_probs = np.array(mcu_probs)
    fp32_preds = (fp32_probs >= 0.5).astype(int)
    mcu_preds = (mcu_probs >= 0.49).astype(int)

    # Metrics
    max_diff = float(np.max(np.abs(fp32_probs - mcu_probs)))
    mean_diff = float(np.mean(np.abs(fp32_probs - mcu_probs)))
    verdict_agreement = float(np.mean(fp32_preds == mcu_preds))

    def acc_f1(p, y):
        from sklearn.metrics import accuracy_score, f1_score
        return float(accuracy_score(y, p)), float(f1_score(y, p, zero_division=0))

    fp32_acc, fp32_f1 = acc_f1(fp32_preds, yw)
    mcu_acc, mcu_f1 = acc_f1(mcu_preds, yw)

    report = {
        "n_samples": int(len(Xw)),
        "seed": args.seed,
        "fp32_model": str(fp32_pt),
        "mcu_header": str(mcu_h),
        "max_prob_diff": max_diff,
        "mean_prob_diff": mean_diff,
        "verdict_agreement": verdict_agreement,
        "fp32_threshold": 0.5,
        "mcu_threshold": 0.49,
        "fp32_accuracy": fp32_acc,
        "fp32_f1": fp32_f1,
        "mcu_accuracy": mcu_acc,
        "mcu_f1": mcu_f1,
        "fp32_inference_seconds": fp32_time,
        "mcu_inference_seconds": mcu_time,
        "fp32_ms_per_sample": fp32_time / len(Xw) * 1000,
        "mcu_ms_per_sample": mcu_time / len(Xw) * 1000,
    }

    out_json = repo / f"compare_fp32_vs_mcu_int8_v4se_s{args.seed}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(json.dumps(report, indent=2))

    # Markdown summary
    md = f"""# FP32 vs MCU Hybrid INT8 Comparison (seed={args.seed})

**N samples**: {len(Xw)}

| Metric | FP32 (.pt) | MCU (.h) |
|--------|-----------|----------|
| Threshold | 0.50 | 0.49 |
| Accuracy | {fp32_acc:.4f} | {mcu_acc:.4f} |
| F1 | {fp32_f1:.4f} | {mcu_f1:.4f} |
| Inference (ms/sample) | {report['fp32_ms_per_sample']:.2f} | {report['mcu_ms_per_sample']:.2f} |

## Numerical agreement

- Max prob diff: **{max_diff:.2e}** ({"PASS" if max_diff < 1e-5 else "FAIL"} < 1e-5)
- Mean prob diff: {mean_diff:.2e}
- Verdict agreement: **{verdict_agreement*100:.2f}%**
"""
    out_md = repo / f"compare_fp32_vs_mcu_int8_v4se_s{args.seed}.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()