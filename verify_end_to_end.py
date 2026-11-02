"""
verify_end_to_end.py -- 端到端验证: 修好 conv + 恢复亚秒 time 后的 H743 指标.

跑 4 种组合 (数据集 × 窗口相位), 全部用修复后的部署 C kernel:
  数据集: IanArffDataset.csv (整数秒, 当前默认)  vs  IanArffDatasetF32.csv (亚秒, 训练同源)
  相位:   phase=0 (H743 部署网格)                  vs  phase=5 (官方训练 test 网格)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(csv_name, phase):
    tag = "%s_p%d" % (csv_name.split(".")[0], phase)
    env = dict(os.environ, H743_CSV=csv_name, E2E_PHASE=str(phase), E2E_TAG=tag,
               PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, os.path.join(HERE, "_e2e_worker.py")],
                       env=env, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-3000:])
        raise SystemExit(1)
    return np.load(os.path.join(HERE, "e2e_%s.npz" % tag))


def metrics(probs, atk, thr):
    det = probs >= thr
    tp = int((det & atk).sum()); fp = int((det & ~atk).sum())
    fn = int((~det & atk).sum()); tn = int((~det & ~atk).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, P=P, R=R, F1=2 * P * R / max(P + R, 1e-9),
                FPR=fp / max(fp + tn, 1), Acc=(tp + tn) / len(atk))


print("=" * 108)
print("修复 conv 对齐后, H743 端到端 (窗口 200001..274628, thr=0.5)")
print("=" * 108)
store = {}
for csv_name in ("IanArffDataset.csv", "IanArffDatasetF32.csv"):
    for phase in (0, 5):
        npz = run_one(csv_name, phase)
        store[(csv_name, phase)] = (npz["probs"], npz["atk"])
        m = metrics(npz["probs"], npz["atk"], 0.5)
        name = "亚秒 F32" if "F32" in csv_name else "整数秒   "
        print("  %-10s phase=%d  n=%5d  TP=%5d FP=%5d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
              % (name, phase, len(npz["atk"]), m["TP"], m["FP"], m["FN"],
                 m["P"], m["R"], m["F1"], m["FPR"]))

print()
print("=" * 108)
print("阈值扫描 (亚秒 F32, phase=0 = H743 实际部署网格)")
print("=" * 108)
probs, atk = store[("IanArffDatasetF32.csv", 0)]
for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95):
    m = metrics(probs, atk, thr)
    print("  thr=%.2f  TP=%5d FP=%5d FN=%5d  P=%.4f R=%.4f F1=%.4f FPR=%.4f"
          % (thr, m["TP"], m["FP"], m["FN"], m["P"], m["R"], m["F1"], m["FPR"]))
print()
print("参考: 官方训练管线在同一 test 网格 (phase=5) 上: F1=0.8686 R=0.7983 FPR=0.0443")
