"""
exp_2x2_attrib.py -- 2x2 误差归因: 权重量化 x 激活量化.

目的: 回答 "INT8 掉 FPR 到底是权重量化造成的, 还是激活量化造成的?"

  权重 = FP32 (Route E 部署真值)      |  权重 = INT8 per-channel
  --------------------------------------+---------------------------------------
  激活 = FP32   ->  基线 (真值)        |  激活 = FP32  ->  Hybrid-INT8
  激活 = INT8   ->  只量化激活         |  激活 = INT8  ->  双量化

四个 cell 用 **同一个** sim_hybrid_kernel 框架跑 (只换权重头和 act_q),
所以差异只能来自量化本身, 不是框架差异.

优先复用 compare_fp32_vs_hybrid.py 已经算好的 npz; 缺哪个就现算哪个.

口径: 部署区 frame>=200001 (对外报这个), thr=0.5, Binary 指标.
     train 区只用来确认结论方向一致, 不做任何标定.

用法: python exp_2x2_attrib.py [--recompute]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compare_fp32_vs_hybrid import metrics
from sim_hybrid_kernel import run_all

WIN_NPZ = os.path.join(HERE, "experiments/tflm_fp32/artifacts/windows_f32_mcu.npz")
OUT_JSON = os.path.join(HERE, "attrib_2x2.json")

# cell -> npz 文件 (probs 键)
CELLS = [
    ("W=FP32 A=FP32 (基线/真值)", ("fp32", "fp32"), "fullsim_f32.npz"),
    ("W=INT8 A=FP32 (Hybrid)", ("int8", "fp32"), "fullsim_hybrid.npz"),
    ("W=FP32 A=INT8 (只量化激活)", ("fp32", "int8"), "fullsim_actonly.npz"),
    ("W=INT8 A=INT8 (双量化)", ("int8", "int8"), "fullsim_int8.npz"),
]


def main():
    d = np.load(WIN_NPZ)
    X = d["X"].astype(np.float32)
    frames = d["frames"]
    atk = d["atk"]

    dep = frames >= 200001
    tr = ~dep

    probs = {}
    for tag, (wm, am), fn in CELLS:
        path = os.path.join(HERE, fn)
        if os.path.exists(path):
            probs[tag] = np.load(path)["probs"]
            print("[用缓存] %-32s <- %s" % (tag, fn))
        else:
            print("[现算]   %-32s ..." % tag, flush=True)
            probs[tag] = run_all(X, weights_mode=wm, act_mode=am, verbose=False)
            np.savez(path, probs=probs[tag], atk=atk, frames=frames,
                     variant="%s_w_%s_a" % (wm, am))

    res = {}
    for tag, _, _ in CELLS:
        res[tag] = {
            "deploy": metrics(probs[tag][dep] >= 0.5, atk[dep]),
            "train": metrics(probs[tag][tr] >= 0.5, atk[tr]),
        }

    base = CELLS[0][0]
    pb = probs[base]
    flips = {}
    for tag, _, _ in CELLS[1:]:
        flips[tag] = int(((probs[tag][dep] >= 0.5) != (pb[dep] >= 0.5)).sum())

    print("\n" + "=" * 96)
    print("[2x2] 部署区 (frame>=200001, n=%d)  thr=0.5" % int(dep.sum()))
    print("=" * 96)
    print("%-32s %8s %8s %8s %8s %8s %8s"
          % ("cell", "F1", "FPR", "P", "R", "FP", "翻FP32"))
    for tag, _, _ in CELLS:
        m = res[tag]["deploy"]
        print("%-32s %8.4f %8.4f %8.4f %8.4f %8d %8d"
              % (tag, m["F1"], m["FPR"], m["P"], m["R"], m["FP"], flips.get(tag, 0)))

    print("\n[train 区 (frame<200001) 方向确认]")
    print("%-32s %8s %8s" % ("cell", "F1", "FPR"))
    for tag, _, _ in CELLS:
        m = res[tag]["train"]
        print("%-32s %8.4f %8.4f" % (tag, m["F1"], m["FPR"]))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"metrics": res, "flips_vs_baseline": flips,
                   "zones": {"deploy_n": int(dep.sum()), "train_n": int(tr.sum())}},
                  f, indent=2, ensure_ascii=False)
    print("\n已写入 %s" % OUT_JSON)


if __name__ == "__main__":
    main()
