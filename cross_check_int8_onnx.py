"""
cross_check_int8_onnx.py  --  L4 交叉验证: 用 **第三方量化器** (ONNX Runtime static
                             quantization) 做一份独立的 INT8 模型, 跟本工程
                             CMSIS-NN 路线的 INT8 结果对拍.

为什么需要:
  本工程没有 host C 编译器, PC 侧的 INT8 数字全部来自 numpy 镜像 (自证),
  唯一能证明 "INT8 量化本身没搞错" 的旁证, 就是换一个 **完全独立的量化实现**
  看概率分布是否落在同一区间.

⚠️ 预期不是 bit-exact: ORT 的量化器 (per-tensor 激活 / 自己的 requant 舍入 /
   自己的 calibration 范围) 跟本工程 (非对称激活 + CMSIS double-rounding +
   手选 input range) 设计不同. 所以这里比的是:
   1. 概率分布 (MAE / 相关系数 / 决策翻转率)
   2. 分区域 4 指标 (看是不是同样在部署区 F1 ≈ 0.86)
   如果 ORT 版和我方版都落在 FP32 附近, 就说明 "INT8 PTQ 对这个模型是无损的"
   这个结论跟具体量化器无关.

用法: python cross_check_int8_onnx.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("H743_CSV", "IanArffDatasetF32.csv")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

DEPLOY_FROM = 200001
TEST_FROM = 219701
ONNX_FP32 = os.path.join(HERE, "model_v4_se_23dim_ch32_s42.onnx")
ONNX_INT8 = os.path.join(HERE, "model_v4_se_23dim_ch32_s42_ortqdq.onnx")
OUT_JSON = os.path.join(HERE, "int8_crosscheck_onnx.json")
N_CALIB = 500


def metrics(det, atk):
    tp = int((det & atk).sum())
    fp = int((det & ~atk).sum())
    fn = int((~det & atk).sum())
    n_norm = int((~atk).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, P=P, R=R,
                F1=2 * P * R / max(P + R, 1e-9), FPR=fp / max(n_norm, 1))


def zone_report(probs, atk, frames, thr=0.5):
    out = {}
    for name, mask in (("all", np.ones(len(probs), dtype=bool)),
                       ("deploy", frames >= DEPLOY_FROM),
                       ("test", frames >= TEST_FROM),
                       ("train", frames < DEPLOY_FROM)):
        out[name] = metrics(probs[mask] >= thr, atk[mask])
    return out


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import (QuantFormat, QuantType, quantize_static,
                                           CalibrationDataReader)

    from h743_windows import build_windows, window_labels

    print("[1] 构建窗口 ...", flush=True)
    t0 = time.time()
    wins, labels = build_windows(start=0, n=274628, phase=0)
    X = np.stack([w[2] for w in wins], axis=0).astype(np.float32)   # (N, 23, 16)
    frames = np.array([w[0] for w in wins])
    atk = np.array(window_labels(wins, labels), dtype=bool)
    print("    %d 窗, 用时 %.1fs" % (len(X), time.time() - t0), flush=True)

    # ---------------- FP32 ONNX (确认是部署真值) ----------------
    print("[2] FP32 ONNX 推理 (对照 fullsim_f32.npz) ...", flush=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_FP32, so, providers=["CPUExecutionProvider"])

    def run(sess, X, bs=512):
        out = []
        for i in range(0, len(X), bs):
            out.append(sess.run(None, {"input": X[i:i + bs]})[0])
        return np.concatenate(out).astype(np.float64).ravel()

    def sigmoid(z):
        # ONNX 图里最后的 Sigmoid 只用于 SE, 末端输出是 **logit** (跟 C kernel 一致:
        # C 里 logit/=T 之后才做 sigmoid). 这里补上 sigmoid 才能跟 prob 比.
        z = np.asarray(z, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-z))

    p_onnx32 = sigmoid(run(sess, X))
    d32 = np.load(os.path.join(HERE, "fullsim_f32.npz"))
    p_ref32 = d32["probs"].astype(np.float64)
    mae_onnx = float(np.abs(p_onnx32 - p_ref32).max())
    print("    ONNX-FP32 vs fullsim_f32.npz: max|Δ|=%.3e  (输入 layout 一致性检查)" % mae_onnx)

    # ---------------- ORT 静态量化 ----------------
    print("[3] ORT 静态量化 (QDQ, per-channel weight, 激活 per-tensor) ...", flush=True)
    rng = np.random.default_rng(42)
    train_idx = np.where(frames < DEPLOY_FROM)[0]
    cal_idx = rng.choice(train_idx, size=min(N_CALIB, len(train_idx)), replace=False)

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data
            self.i = 0

        def get_next(self):
            if self.i >= len(self.data):
                return None
            b = self.data[self.i:self.i + 1]
            self.i += 1
            return {"input": b}

        def rewind(self):
            self.i = 0

    quantize_static(
        model_input=ONNX_FP32,
        model_output=ONNX_INT8,
        calibration_data_reader=Reader(X[cal_idx]),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    print("    写出 %s" % os.path.basename(ONNX_INT8))

    # ---------------- 量化模型推理 ----------------
    print("[4] INT8 ONNX 推理 ...", flush=True)
    sess_q = ort.InferenceSession(ONNX_INT8, so, providers=["CPUExecutionProvider"])
    p_onnx8 = sigmoid(run(sess_q, X))

    d8 = np.load(os.path.join(HERE, "fullsim_int8.npz"))
    p_mine8 = d8["probs"].astype(np.float64)

    print("\n[5] 结果对比 (全部 17164 窗)")
    for nm, a, b in (("ORT-INT8  vs  ONNX-FP32", p_onnx8, p_onnx32),
                     ("本工程INT8 vs  ONNX-FP32", p_mine8, p_onnx32),
                     ("本工程INT8 vs  ORT-INT8 ", p_mine8, p_onnx8)):
        mae = float(np.abs(a - b).mean())
        mx = float(np.abs(a - b).max())
        r = float(np.corrcoef(a, b)[0, 1])
        flip = float(((a >= 0.5) != (b >= 0.5)).mean())
        print("    %s: MAE=%.4f max|Δ|=%.4f r=%.6f 决策翻转=%.2f%%"
              % (nm, mae, mx, r, 100 * flip))

    rep = {
        "onnx_fp32_vs_fullsim_f32_maxabs": mae_onnx,
        "pairs": {
            "ort_int8_vs_onnx_fp32": {
                "mae": float(np.abs(p_onnx8 - p_onnx32).mean()),
                "maxabs": float(np.abs(p_onnx8 - p_onnx32).max()),
                "r": float(np.corrcoef(p_onnx8, p_onnx32)[0, 1]),
                "flip_rate": float(((p_onnx8 >= 0.5) != (p_onnx32 >= 0.5)).mean()),
            },
            "mine_int8_vs_onnx_fp32": {
                "mae": float(np.abs(p_mine8 - p_onnx32).mean()),
                "maxabs": float(np.abs(p_mine8 - p_onnx32).max()),
                "r": float(np.corrcoef(p_mine8, p_onnx32)[0, 1]),
                "flip_rate": float(((p_mine8 >= 0.5) != (p_onnx32 >= 0.5)).mean()),
            },
            "mine_int8_vs_ort_int8": {
                "mae": float(np.abs(p_mine8 - p_onnx8).mean()),
                "maxabs": float(np.abs(p_mine8 - p_onnx8).max()),
                "r": float(np.corrcoef(p_mine8, p_onnx8)[0, 1]),
                "flip_rate": float(((p_mine8 >= 0.5) != (p_onnx8 >= 0.5)).mean()),
            },
        },
        "zones": {
            "onnx_fp32": zone_report(p_onnx32, atk, frames),
            "ort_int8": zone_report(p_onnx8, atk, frames),
            "mine_int8": zone_report(p_mine8, atk, frames),
        },
    }
    print("\n[6] 分区域指标 (thr=0.5)")
    print("    %-10s %-28s %-28s %-28s" % ("区域", "ONNX-FP32", "ORT-INT8", "本工程INT8"))
    for z in ("all", "deploy", "test", "train"):
        cells = []
        for k in ("onnx_fp32", "ort_int8", "mine_int8"):
            m = rep["zones"][k][z]
            cells.append("F1=%.4f FPR=%.4f R=%.4f" % (m["F1"], m["FPR"], m["R"]))
        print("    %-10s %-28s %-28s %-28s" % (z, cells[0], cells[1], cells[2]))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(HERE, "fullsim_onnx_ort_int8.npz"),
             probs=p_onnx8.astype(np.float32), frames=frames)
    print("\n[write] %s" % OUT_JSON)


if __name__ == "__main__":
    main()
