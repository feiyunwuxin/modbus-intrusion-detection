"""
sweep_input_range.py -- 量化旋钮扫描: 模型输入范围 x 激活范围 (截断分位数).

两个旋钮:
  1. 输入量化范围: 对称 +-V / 非对称 [lo, hi]
     特征本身是标准化后的值 (大多在 +-2), 用 +-10 会浪费 4 倍分辨率;
     但裁太紧会丢信息 (sym6 就崩了), 所以要扫.
  2. 激活范围的上界: 真实 max vs 分位数 (p99 / p99.9 / p99.99)
     max 往往被离群点 (攻击窗口的大激活) 拉高, 截断后分辨率提升, 代价是离群点被 clip.

**选择只看 train 区 (frame<200001) 的 F1**; 部署区 (frame>=200001) 只在最后
报告, 绝不参与选择 (否则就是拿部署区调参, 数字不可信).

用法:  python sweep_input_range.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RANGES_JSON = os.path.join(HERE, "int8_qparams_base.json")
TMP_HDR = os.path.join(HERE, "i8_sweep_tmp.h")
TMP_JSON = os.path.join(HERE, "i8_sweep_tmp.json")
OUT_JSON = os.path.join(HERE, "quant_sweep.json")
FINAL_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h")
FINAL_JSON = os.path.join(HERE, "int8_qparams_s42.json")
DEPLOY_FROM = 200001


def metrics(det, atk):
    tp = int((det & atk).sum())
    fp = int((det & ~atk).sum())
    fn = int((~det & atk).sum())
    n_norm = int((~atk).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, P=P, R=R,
                F1=2 * P * R / max(P + R, 1e-9), FPR=fp / max(n_norm, 1))


def gen_header(args):
    cmd = [sys.executable, os.path.join(HERE, "quantize_fp32_to_int8.py"),
           "--from-json", RANGES_JSON, "--out", TMP_HDR, "--json", TMP_JSON] + args
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    os.environ["H743_CSV"] = "IanArffDatasetF32.csv"
    from h743_windows import build_windows, window_labels
    from sim_int8_kernel import Int8Model

    print("[sweep] 构建窗口 ...", flush=True)
    t0 = time.time()
    wins, labels = build_windows(start=0, n=274628, phase=0)
    X = np.stack([w[2] for w in wins], axis=0).astype(np.float32)
    frames = np.array([w[0] for w in wins])
    atk = np.array(window_labels(wins, labels), dtype=bool)
    train = frames < DEPLOY_FROM
    print("[sweep] %d 窗 (train %d), 用时 %.1fs" % (len(X), train.sum(), time.time() - t0), flush=True)

    Xtr = X[train]
    lo_obs, hi_obs = float(Xtr.min()), float(Xtr.max())
    print("[sweep] train 区输入范围: [%.3f, %.3f]" % (lo_obs, hi_obs))

    input_cands = [
        ("sym10", ["--clip", "10"]),
        ("asym-10_10", ["--ilo", "-10", "--ihi", "10"]),
        ("asym-obs", ["--ilo", repr(lo_obs), "--ihi", repr(hi_obs)]),
    ]
    act_cands = [(None, []), ("99.99", ["--act-pct", "99.99"]),
                 ("99.9", ["--act-pct", "99.9"]), ("99.0", ["--act-pct", "99.0"])]

    d32 = np.load(os.path.join(HERE, "fullsim_f32.npz"))
    p32 = d32["probs"]
    base = metrics(p32[train] >= 0.5, atk[train])
    print("\nFP32 基线 (train 区, thr=0.5): F1=%.4f FPR=%.4f R=%.4f P=%.4f"
          % (base["F1"], base["FPR"], base["R"], base["P"]))
    print("-" * 100)
    print("  %-22s %-30s %-30s" % ("候选 (输入/激活)", "train 区 (用于选择)", "部署区 (仅参考)"))
    print("  %-22s %-30s %-30s" % ("", "F1     FPR    R      P", "F1     FPR    R      P"))

    rows = []
    for iname, iargs in input_cands:
        for aname, aargs in act_cands:
            args = iargs + aargs
            gen_header(args)
            m = Int8Model(TMP_HDR)
            probs = np.empty(len(X), dtype=np.float32)
            for i in range(0, len(X), 1024):
                probs[i:i + 1024] = m.forward(X[i:i + 1024])
            tr = metrics(probs[train] >= 0.5, atk[train])
            dp = metrics(probs[~train] >= 0.5, atk[~train])
            rows.append((iname, aname, args, tr, dp))
            print("  %-22s %.4f %.4f %.4f %.4f   %.4f %.4f %.4f %.4f"
                  % ("%s/%s" % (iname, aname or "max"),
                     tr["F1"], tr["FPR"], tr["R"], tr["P"],
                     dp["F1"], dp["FPR"], dp["R"], dp["P"]), flush=True)

    rows.sort(key=lambda r: (-r[3]["F1"], r[3]["FPR"]))
    best = rows[0]
    print("-" * 100)
    print("==> 选中 (train 区 F1 最优): %s/%s   train F1=%.4f (FP32 %.4f, Δ=%+.4f)  train FPR=%.4f (FP32 %.4f)"
          % (best[0], best[1] or "max", best[3]["F1"], base["F1"],
             best[3]["F1"] - base["F1"], best[3]["FPR"], base["FPR"]))
    print("==> 该候选在部署区: F1=%.4f FPR=%.4f (FP32: F1=%.4f FPR=%.4f)"
          % (best[4]["F1"], best[4]["FPR"], metrics(p32[~train] >= 0.5, atk[~train])["F1"],
             metrics(p32[~train] >= 0.5, atk[~train])["FPR"]))

    gen_header(best[2])
    shutil.copyfile(TMP_HDR, FINAL_HDR)
    shutil.copyfile(TMP_JSON, FINAL_JSON)
    print("==> 正式 header 已更新: %s" % FINAL_HDR)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_input": best[0], "best_act_pct": best[1], "args": best[2],
                   "train": best[3], "deploy": best[4], "fp32_train": base,
                   "all": [{"input": r[0], "act_pct": r[1], "args": r[2],
                            "train": r[3], "deploy": r[4]} for r in rows]},
                  f, indent=2, ensure_ascii=False)
    print("==> 扫描结果写入 %s" % OUT_JSON)


if __name__ == "__main__":
    main()
