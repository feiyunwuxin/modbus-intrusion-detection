"""
diag_tflm_in_pc.py  -- 路线 D PC 配对: 加载 v1 INT8 .tflite, 跑 X_test 前 5 windows,
                       打 FP32 input + int8 input + logit, 跟 H743 [TFLM-IN]/[TFLM-IQ] 配对

输出格式跟 H743 端 tflite_micro_runtime.cpp [TFLM-IN]/[TFLM-IQ] 一致:
  [TFLM-IN] w#N s=X.XXXXXX zp=Z fp32_ch12_t0..15=V0,V1,...,V15
  [TFLM-IQ] w#N int8_ch12_t0..15=Q0,Q1,...,Q15
  [TFLM-OUT] w#N logit=X.XXX

配对判断:
  - FP32 input bit-perfect → 上游 sliding_window 算式链正确 (已 [SW-DBG] 验证)
  - int8 input bit-perfect → quant 边界 round 一致 (lroundf vs np.round)
  - logit bit-perfect → TFLM runtime 算子正确

不匹配定位:
  - FP32 不一致 → 上游问题 (但 [SW-DBG] 已验证, 这里应该是 bit-perfect)
  - int8 不一致 → lroundf vs np.round 边界差异 (改 lrintf 或 floor+0.5)
  - logit 不一致 → TFLM runtime 算子 (MicroInterpreter 版本)
"""
import sys
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
SEED = 42
N_WINDOWS = 5


def load_data():
    X = np.load('X_test_binary_v2_scada_window16.npy').astype(np.float32)[:, :, KEEP_23]
    X = np.clip(X, -10, 10)
    y = np.load('y_test_binary_v2_scada_window16.npy').astype(np.int64)
    return X, y


def quantize_pc(x_fp32, scale, zp):
    """跟 H743 lroundf(x/s) + zp 一致 (numpy round 默认 banker's rounding 跟 lround 一致)."""
    return np.clip(np.round(x_fp32 / scale + zp), -128, 127).astype(np.int8)


def main():
    tflite_path = f'model_v4_se_23dim_ch32_qat_s{SEED}_int8.tflite'
    print(f"加载 v1 INT8 .tflite: {tflite_path}")
    X, y = load_data()
    print(f"X_test shape: {X.shape}  (windows={len(X)}, time={X.shape[1]}, channels={X.shape[2]})")
    print(f"y_test attack%={y.mean()*100:.1f}")
    print(f"配对 H743: 前 {N_WINDOWS} windows (跟 X_test[0..4])\n")

    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']
    print(f"input:  dtype={inp_d['dtype']}  scale={in_scale:.6f}  zp={in_zp}")
    print(f"output: dtype={out_d['dtype']}  scale={out_scale:.6f}  zp={out_zp}\n")

    print("=" * 70)
    print(f"[PC sim] 跑前 {N_WINDOWS} windows, 打 [TFLM-IN]/[TFLM-IQ] 配 H743")
    print("=" * 70)
    for w in range(N_WINDOWS):
        # X[w] shape (16, 23), transpose 成 (23, 16), flatten row-major → 跟 H743 x[c*16+t] 一致
        x_fp32 = X[w].T.flatten().astype(np.float32)   # (23*16,) row-major

        # 打 [TFLM-IN] — ch12 = crc_rate, t=0..15
        fp32_ch12_t = [x_fp32[12*16+t] for t in range(16)]
        print(f"[TFLM-IN] w#{w} s={in_scale:.6f} zp={in_zp} "
              f"fp32_ch12_t0..15={','.join(f'{v:.3f}' for v in fp32_ch12_t)}")

        # quantize PC 端 (跟 H743 lroundf(x/s) + zp 等价)
        x_q = quantize_pc(x_fp32, in_scale, in_zp)
        iq_ch12_t = [int(x_q[12*16+t]) for t in range(16)]
        print(f"[TFLM-IQ] w#{w} int8_ch12_t0..15={','.join(str(v) for v in iq_ch12_t)}")

        # 跑推理
        # 注意: TFLite input 是 (1, C, T) = (1, 23, 16) — H743 给 (23, 16) row-major flatten 实际就是 (1, 23, 16)
        x_q_input = x_q.reshape(1, 23, 16)
        interp.set_tensor(inp_d['index'], x_q_input)
        interp.invoke()
        y_q = interp.get_tensor(out_d['index'])
        logit = (y_q.astype(np.float32) - out_zp) * out_scale
        logit_v = float(logit[0]) if logit.ndim > 0 else float(logit)
        prob = 1.0 / (1.0 + np.exp(-logit_v))
        print(f"[TFLM-OUT] w#{w} y_q={int(y_q[0]) if y_q.ndim>0 else int(y_q)} "
              f"logit={logit_v:.3f} prob={prob:.4f} y={y[w]}")
        print()

    # PC sim 给的对比预期
    print("=" * 70)
    print("[对比预期] H743 收到 [TFLM-IN]/[TFLM-IQ] 后:")
    print("  1. fp32_ch12_t0..15 完全一致 → 上游 sw_buf → ch32_forward 输入 bit-perfect")
    print("  2. int8_ch12_t0..15 完全一致 → quant 边界 round 一致 (lroundf vs np.round)")
    print("  3. logit 一致 → TFLM runtime 算子正确")
    print()
    print("  不匹配定位:")
    print("    FP32 不一致 → 上游问题 (但 [SW-DBG] 已验证, 应 bit-perfect)")
    print("    int8 不一致 → lroundf vs np.round 边界差异 (改 floorf+0.5)")
    print("    logit 不一致 → TFLM runtime 算子 (MicroInterpreter 版本 / op 实现)")
    print()
    print("  当前 H743 实测 logit=11.098 (T=2.0 scaled). 如果 PC sim logit=-0.876 (normal) /")
    print("  +14.784 (attack), H743 logit 远高于 PC → TFLM input 一定不一致 (FP32 或 int8)")


if __name__ == '__main__':
    main()
