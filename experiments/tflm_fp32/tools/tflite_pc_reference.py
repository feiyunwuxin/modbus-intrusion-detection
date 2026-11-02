#!/usr/bin/env python3
"""
G0b: PC 侧 TFLite 参考 —— 与 ONNX Runtime 逐窗对拍 + 产出 MCU 上机自检基线.

产出:
  1. experiments/tflm_fp32/artifacts/tflm_selftest_ref.json
     5 个 controlled input (ZEROS/ONES/HALF/NEG1/TWOS) 的**精确** logit
     (float.hex(), 供 MCU 上 'T' 命令逐位核对)
  2. experiments/tflm_fp32/artifacts/tflm_pc_report.json
     逐窗 parity (TFLite vs ORT vs 手写 C kernel) + 部署区 4 指标

设计原则 (沿用手写 kernel 路线的教训):
  * 输入窗口用 **MCU 视角** (h743_windows 的 x_h7 = PC 编码 -> MCU 解码 -> 特征 ->
    滑窗), 因为 MCU 上 TFLM 吃的就是这个布局; ORT 也吃同一份数组, 保证对拍公平.
  * TFLite 解释器**不挂任何 delegate** (只用 builtin reference kernel),
    跟 TFLM 的 reference impl 语义最接近.
  * 判据用 **logit** (ONNX 输出其实是 logit, Sigmoid 只在 SE 内部),
    阈值判定再自己补 sigmoid, 与 MCU ch32_forward 的语义一致.
  * 窗口缓存成 npz, 避免每次重扫 27 万行 CSV.

用法:
    python experiments/tflm_fp32/tools/tflite_pc_reference.py
    python experiments/tflm_fp32/tools/tflite_pc_reference.py --rebuild-windows
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ART = os.path.join(ROOT, "experiments", "tflm_fp32", "artifacts")

TFLITE = os.path.join(ART, "model_tflm_fp32_s42.tflite")
ONNX = os.path.join(ROOT, "model_v4_se_23dim_ch32_s42.onnx")
CSV = "IanArffDatasetF32.csv"          # 亚秒 time, 训练/部署同源 (根因2)
N_FRAMES = 274628
W = 16
DEPLOY_START_ROW = 200000              # 部署区起始行 (frame 200001)
G0 = DEPLOY_START_ROW // W

# 5 个 controlled input: 整个 23x16 窗口填同一个常数
CONTROLLED = {
    "ZEROS": 0.0,
    "ONES": 1.0,
    "HALF": 0.5,
    "NEG1": -1.0,
    "TWOS": 2.0,
}


def log(msg):
    print(msg, flush=True)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ── 1. 窗口 (MCU 视角) ──────────────────────────────────────────────────── #
def build_mcu_windows(cache_path, rebuild=False):
    """返回 (frames[N], X[N,23,16] float32, atk[N] bool)."""
    if os.path.exists(cache_path) and not rebuild:
        d = np.load(cache_path)
        log("[win] 用缓存 %s (N=%d)" % (os.path.basename(cache_path), len(d["frames"])))
        return d["frames"], d["X"], d["atk"].astype(bool)

    os.environ["H743_CSV"] = CSV       # 必须在 import h743_windows 之前
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from h743_windows import build_windows, window_labels

    log("[win] 从 %s 重建 %d 帧 (这一段最慢, 结果会缓存) ..." % (CSV, N_FRAMES))
    t0 = time.time()
    wins, labels = build_windows(start=0, n=N_FRAMES, phase=0)
    atk = np.array(window_labels(wins, labels), dtype=bool)
    frames = np.array([w[0] for w in wins], dtype=np.int64)
    X = np.stack([w[2] for w in wins], 0).astype(np.float32)   # x_h7 = MCU 视角
    log("[win] %d 窗, 用时 %.1fs" % (len(frames), time.time() - t0))
    np.savez_compressed(cache_path, frames=frames, X=X, atk=atk)
    log("[win] 缓存 -> %s" % cache_path)
    return frames, X, atk


# ── 2. TFLite (无 delegate) ────────────────────────────────────────────── #
def make_interpreter(path, op_resolver="BUILTIN_REF"):
    """建解释器.

    op_resolver:
      BUILTIN_REF  -> TFLite 的 reference kernel. **跟 TFLM 的 float 路径同源**
                     (micro/kernels/internal/reference/*.{h,cc} 是 TFLite reference
                      kernel 的副本: conv.h / FullyConnected / Add / Logistic / Mul),
                     所以这是跟 MCU 最可比的一种.
      BUILTIN_NO_XNNPACK -> 优化 kernel, 但不挂 XNNPACK delegate (次要参考).
      AUTO -> 默认 (会挂 XNNPACK delegate, **不采用**: XNNPACK 跟 TFLM 的
             reference impl 不是同一套算子, 对拍结论会被 delegate 污染).
    """
    import tensorflow as tf
    resolver = {
        "BUILTIN_REF": tf.lite.experimental.OpResolverType.BUILTIN_REF,
        "BUILTIN_NO_XNNPACK": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        "AUTO": tf.lite.experimental.OpResolverType.AUTO,
    }[op_resolver]
    interp = tf.lite.Interpreter(model_path=path,
                                 experimental_op_resolver_type=resolver)
    n_del = len(getattr(interp, "_delegates", []) or [])
    if op_resolver != "AUTO" and n_del != 0:
        raise RuntimeError("TFLite 解释器挂了 %d 个 delegate, 与 TFLM 语义不可比" % n_del)
    interp.allocate_tensors()
    return interp


def tflite_run(interp, X):
    """X: (N,23,16) -> logits (N,)"""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    assert list(inp["shape"]) == [1, 23, 16], inp["shape"]
    assert list(out["shape"]) == [1], out["shape"]
    logits = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        interp.set_tensor(inp["index"], X[i][None, ...].astype(np.float32))
        interp.invoke()
        logits[i] = float(interp.get_tensor(out["index"])[0])
    return logits


def ort_run(path, X, batch=256):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].astype(np.float32)
        outs.append(sess.run(None, {name: xb})[0].reshape(-1))
    return np.concatenate(outs).astype(np.float64)


# ── 3. 指标 ────────────────────────────────────────────────────────────── #
def metrics(pred, g):
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    tn = int(((pred == 0) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    R = tp / (tp + fn) if tp + fn else 0.0
    FPR = fp / (fp + tn) if fp + tn else 0.0
    P = tp / (tp + fp) if tp + fp else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=P, recall=R, f1=F1, fpr=FPR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=TFLITE)
    ap.add_argument("--onnx", default=ONNX)
    ap.add_argument("--cache", default=os.path.join(ART, "windows_f32_mcu.npz"))
    ap.add_argument("--rebuild-windows", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个窗口 (0=全部)")
    ap.add_argument("--c-npz", default=os.path.join(ROOT, "fullsim_f32.npz"),
                    help="手写 C kernel 的全量结果, 用于三方对拍")
    ap.add_argument("--skip-ort", action="store_true")
    ap.add_argument("--skip-opt", action="store_true", help="跳过优化 kernel 二次参考")
    ap.add_argument("--op-resolver", default="BUILTIN_REF",
                    choices=["BUILTIN_REF", "BUILTIN_NO_XNNPACK", "AUTO"],
                    help="BUILTIN_REF = TFLite reference kernel, 跟 TFLM float 路径同源")
    args = ap.parse_args()

    log("=" * 78)
    log("G0b: PC 侧 TFLite 参考 (无 delegate) vs ONNX Runtime vs 手写 C kernel")
    log("=" * 78)

    # ---- TFLM 兼容性静态检查 (先过这一关, 再谈数值) ---- #
    sys.path.insert(0, HERE)
    from tflm_compat_check import check_tflm_compat
    blob = open(args.tflite, "rb").read()
    compat = check_tflm_compat(blob)
    if compat["errors"]:
        log("[compat] [FAIL] 不兼容 TFLM:")
        for e in compat["errors"]:
            log("    - %s" % e)
        return 1
    log("[compat] [OK] TFLM 兼容性静态检查全过 (%d op, 集合 %s)"
        % (compat["n_operators"], ",".join(compat["op_set"])))

    frames, X, atk = build_mcu_windows(args.cache, args.rebuild_windows)
    if args.limit:
        frames, X, atk = frames[:args.limit], X[:args.limit], atk[:args.limit]

    # ---- controlled input 自检基线 ---- #
    interp = make_interpreter(args.tflite, args.op_resolver)
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    selftest = {}
    for name, val in CONTROLLED.items():
        buf = np.full((1, 23, 16), val, dtype=np.float32)
        interp.set_tensor(in_d["index"], buf)
        interp.invoke()
        lg = float(interp.get_tensor(out_d["index"])[0])
        lg32 = float(np.float32(lg))
        selftest[name] = {
            "fill_value": val,
            "logit": lg,
            "logit_hex": float(lg).hex(),
            "prob": float(sigmoid(lg)),
            "logit_f32_hex": lg32.hex(),
        }
        log("[selftest] %-6s fill=%-5s logit=%+.8f (f32 %s) prob=%.8f"
            % (name, val, lg, lg32.hex(), sigmoid(lg)))

    selftest_doc = {
        "model": os.path.basename(args.tflite),
        "op_resolver": args.op_resolver,
        "input_shape": [1, 23, 16],
        "output": "logit (1-D, 1 element)",
        "note": "logit_hex 是 float64 的精确表示; logit_f32_hex 是 float32 的精确表示. "
                "MCU 上按 float32 计算, 对拍请用 logit_f32_hex (或 1e-5 容差).",
        "cases": selftest,
    }
    p_sel = os.path.join(ART, "tflm_selftest_ref.json")
    with open(p_sel, "w", encoding="utf-8") as f:
        json.dump(selftest_doc, f, indent=2, ensure_ascii=False)
    log("[selftest] -> %s" % p_sel)

    # ---- 逐窗推理 ---- #
    log("\n[run] TFLite 跑 %d 窗 ..." % len(X))
    t0 = time.time()
    lg_tfl = tflite_run(interp, X)
    log("[run] TFLite 用时 %.1fs (%.2f ms/窗)" % (time.time() - t0,
                                                1000 * (time.time() - t0) / max(len(X), 1)))

    report = {"n_windows": int(len(X)), "model": os.path.basename(args.tflite)}

    if not args.skip_ort:
        log("[run] ONNX Runtime 跑 %d 窗 ..." % len(X))
        t0 = time.time()
        lg_ort = ort_run(args.onnx, X)
        log("[run] ORT 用时 %.1fs" % (time.time() - t0))

        d = np.abs(lg_tfl - lg_ort)
        flips = int(((sigmoid(lg_tfl) >= 0.5) != (sigmoid(lg_ort) >= 0.5)).sum())
        rel = d / np.maximum(np.abs(lg_ort), 1e-9)
        log("\n[parity] TFLite vs ORT")
        log("    logit 最大绝对误差 = %.3e   (logit 量级 %.1f ~ %.1f)"
            % (d.max(), lg_ort.min(), lg_ort.max()))
        log("    logit 中位绝对误差 = %.3e   最大相对误差 = %.3e" % (np.median(d), rel.max()))
        log("    决策翻转 (thr=0.5) = %d / %d" % (flips, len(X)))
        report["parity_tflite_vs_ort"] = {
            "max_abs_err_logit": float(d.max()),
            "median_abs_err_logit": float(np.median(d)),
            "max_rel_err": float(rel.max()),
            "decision_flips": flips,
            "n": int(len(X)),
        }
        np.savez(os.path.join(ART, "pc_parity_logits.npz"),
                 frames=frames, lg_tflite=lg_tfl, lg_ort=lg_ort)

    # ---- 次要参考: 优化 kernel (不挂 XNNPACK) 与 reference kernel 的差 ---- #
    if args.op_resolver == "BUILTIN_REF" and not args.skip_opt:
        log("\n[ref] 跑优化 kernel 版 (BUILTIN_WITHOUT_DEFAULT_DELEGATES) 对比 ...")
        interp2 = make_interpreter(args.tflite, "BUILTIN_NO_XNNPACK")
        lg_opt = tflite_run(interp2, X)
        d2 = np.abs(lg_tfl - lg_opt)
        flips2 = int(((sigmoid(lg_tfl) >= 0.5) != (sigmoid(lg_opt) >= 0.5)).sum())
        log("    reference vs optimized: logit 最大绝对误差 = %.3e, 决策翻转 = %d"
            % (d2.max(), flips2))
        report["parity_ref_vs_optimized_kernel"] = {
            "max_abs_err_logit": float(d2.max()),
            "decision_flips": flips2,
        }

    # ---- 与手写 C kernel (Route E 部署真值) 对拍 ---- #
    if os.path.exists(args.c_npz):
        cz = np.load(args.c_npz)
        cframes, cprobs = cz["frames"], cz["probs"]
        cmap = {int(f): float(p) for f, p in zip(cframes, cprobs)}
        sel = np.array([int(f) in cmap for f in frames])
        if sel.any():
            p_c = np.array([cmap[int(f)] for f in frames[sel]])
            p_t = sigmoid(lg_tfl[sel])
            d = np.abs(p_t - p_c)
            flips = int(((p_t >= 0.5) != (p_c >= 0.5)).sum())
            log("\n[parity] TFLite vs 手写 C kernel (fullsim_f32.npz, %d 窗可对齐)" % sel.sum())
            log("    prob 最大绝对误差 = %.3e   中位 = %.3e" % (d.max(), np.median(d)))
            log("    决策翻转 = %d" % flips)
            report["parity_tflite_vs_c_kernel"] = {
                "n": int(sel.sum()),
                "max_abs_err_prob": float(d.max()),
                "median_abs_err_prob": float(np.median(d)),
                "decision_flips": flips,
            }

    # ---- 部署区 4 指标 (window-level, 与 h743_error_analysis.py 同口径) ---- #
    g = frames // W - 1                       # 全局窗号
    dep = g >= G0
    gt = atk[dep].astype(int)
    m = metrics((sigmoid(lg_tfl[dep]) >= 0.5).astype(int), gt)
    log("\n[deploy] TFLite(PC) 部署区 (frame >= %d, %d 窗):" % (DEPLOY_START_ROW + 1, dep.sum()))
    log("    TP %d FP %d TN %d FN %d | P %.4f R %.4f F1 %.4f FPR %.4f"
        % (m["tp"], m["fp"], m["tn"], m["fn"], m["precision"], m["recall"], m["f1"], m["fpr"]))
    report["deploy_zone_tflite_pc"] = m

    if os.path.exists(args.c_npz):
        cz = np.load(args.c_npz)
        cmap = {int(f): float(p) for f, p in zip(cz["frames"], cz["probs"])}
        pc = np.array([cmap.get(int(f), np.nan) for f in frames[dep]])
        ok = ~np.isnan(pc)
        mc = metrics((pc[ok] >= 0.5).astype(int), gt[ok])
        log("[deploy] 手写 C kernel(PC) 同口径: TP %d FP %d TN %d FN %d | P %.4f R %.4f F1 %.4f FPR %.4f"
            % (mc["tp"], mc["fp"], mc["tn"], mc["fn"], mc["precision"], mc["recall"], mc["f1"], mc["fpr"]))
        report["deploy_zone_c_kernel_pc"] = mc

    p_rep = os.path.join(ART, "tflm_pc_report.json")
    with open(p_rep, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log("\n[report] -> %s" % p_rep)

    # ---- 闸门判定 ---- #
    gate_ok = True
    pv = report.get("parity_tflite_vs_ort")
    if pv:
        if pv["max_abs_err_logit"] > 1e-4:
            log("[GATE] [FAIL] logit 最大绝对误差 %.3e > 1e-4" % pv["max_abs_err_logit"])
            gate_ok = False
        if pv["decision_flips"] != 0:
            log("[GATE] [FAIL] 决策翻转 %d != 0" % pv["decision_flips"])
            gate_ok = False
    pc = report.get("parity_tflite_vs_c_kernel")
    if pc and pc["decision_flips"] != 0:
        log("[GATE] [FAIL] vs C kernel 决策翻转 %d != 0" % pc["decision_flips"])
        gate_ok = False
    if gate_ok:
        log("\n[GATE] [OK] G0b 通过")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
