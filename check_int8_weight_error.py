"""
check_int8_weight_error.py  --  L0 权重自检

对比 INT8 header 里的量化权重 (按 per-output-channel 反量化回 FP32) 与
FP32 header 里的原始权重:
  1. 逐张量的 max |w_int8_dequant - w_fp32|
  2. 逐张量的相对误差 (相对该张量的 max|w|)
  3. 是否有系统性偏置 (mean signed error 应接近 0)
  4. 余弦相似度 (应 ≈ 1)

⚠️ 布局: FP32 header 是 [out][in][k], INT8 header (CMSIS-NN filter 布局) 是
   [out][k][in] -> 比较前必须转置, 否则是拿错位的数比 (会假阳性).

判据: per-output-channel 对称量化, scale[oc] = max|w[oc]|/127,
      理论上误差 <= 0.5 LSB -> 相对误差 <= 0.5/127 = 0.39%.

用法: python check_int8_weight_error.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from int8_math import parse_int8_header                     # noqa: E402
from quantize_fp32_to_int8 import parse_fp32_header, FP32_HDR   # noqa: E402

INT8_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h")

CH, IN_CH, SE_H, FC1_OUT = 32, 23, 4, 32
P32 = "v4se23_ch32_s42_"

# (INT8 权重名, FP32 权重名, FP32 形状, 是否需要 [out][k][in] -> [out][in][k] 转置)
CONV = [("w_tcn_0_conv1", (CH, IN_CH, 3)), ("w_tcn_0_conv2", (CH, CH, 3)),
        ("w_tcn_0_residual", (CH, IN_CH, 1))]
for bi in (1, 2):
    CONV.append(("w_tcn_%d_conv1" % bi, (CH, CH, 3)))
    CONV.append(("w_tcn_%d_conv2" % bi, (CH, CH, 3)))
SE = [("w_tcn_%d_se_fc1" % bi, (SE_H, CH)) for bi in (0, 1, 2)]
SE += [("w_tcn_%d_se_fc2" % bi, (CH, SE_H)) for bi in (0, 1, 2)]
FC = [("w_fc1", (FC1_OUT, CH)), ("w_fc2", (1, FC1_OUT))]

LSB_TOL = 0.5 / 127.0     # 0.39%


def main():
    raw = parse_fp32_header(FP32_HDR)
    i8 = parse_int8_header(INT8_HDR)

    print("=" * 96)
    print("L0 weight check: INT8 dequant vs FP32 weights")
    print("=" * 96)
    print("%-22s %11s %10s %11s %10s %6s" %
          ("tensor", "max|dW|", "rel(max)", "mean dW", "cos", "verdict"))
    ok_all = True
    worst_rel = 0.0
    rows = []
    for group, transpose in ((CONV, True), (SE, False), (FC, False)):
        for iname, shape in group:
            # FP32 header 的命名: w_tcn_* 保留 "w_", 但 fc1/fc2 没有 (fc1_weight)
            base = iname if iname.startswith("w_tcn") else iname[2:]
            w32 = np.asarray(raw[P32 + base + "_weight"],
                             dtype=np.float64).reshape(shape)
            q = np.asarray(i8[iname], dtype=np.float64)
            sw = np.asarray(i8["sw_" + iname[len("w_"):]], dtype=np.float64)
            if transpose:
                # INT8 [out][k][in] -> [out][in][k]
                qm = q.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1).ravel()
                swm = sw.reshape(-1, 1)
                w32m = w32.reshape(shape[0], -1)
            else:
                qm = q
                swm = sw.reshape(-1, 1) if sw.size > 1 else sw.reshape(1, 1)
                w32m = w32.reshape(shape[0], -1)
            deq = qm.reshape(w32m.shape) * swm
            d = deq - w32m
            absd = np.abs(d)
            rel = absd.max() / max(np.abs(w32m).max(), 1e-12)
            mean_d = d.mean()
            cos = float((deq.ravel() @ w32m.ravel()) /
                        (np.linalg.norm(deq.ravel()) * np.linalg.norm(w32m.ravel()) + 1e-30))
            ok = rel <= LSB_TOL and cos > 0.9999 and abs(mean_d) <= 0.02 * np.abs(w32m).max()
            ok_all = ok_all and ok
            worst_rel = max(worst_rel, rel)
            rows.append((iname, absd.max(), rel, mean_d, cos, ok))
            print("%-22s %11.3e %9.4f%% %11.3e %10.6f %6s" %
                  (iname, absd.max(), 100 * rel, mean_d, cos, "OK" if ok else "!!"))
    print("-" * 96)
    print("worst relative error %.4f%% (criterion <= %.4f%%)  ->  %s"
          % (100 * worst_rel, 100 * LSB_TOL, "ALL PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
