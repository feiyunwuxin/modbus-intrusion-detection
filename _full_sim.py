import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["H743_CSV"] = "IanArffDatasetF32.csv"
from parity_fp32_c_vs_onnx import c_forward, parse_h
from h743_windows import build_windows, window_labels

raw = parse_h(os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h"))
W = {"w_tcn_0_conv1": raw["v4se23_ch32_s42_w_tcn_0_conv1_weight"],
     "b_tcn_0_conv1": raw["v4se23_ch32_s42_b_tcn_0_conv1_bias"],
     "w_tcn_0_conv2": raw["v4se23_ch32_s42_w_tcn_0_conv2_weight"],
     "b_tcn_0_conv2": raw["v4se23_ch32_s42_b_tcn_0_conv2_bias"],
     "w_tcn_0_residual": raw["v4se23_ch32_s42_w_tcn_0_residual_weight"],
     "b_tcn_0_residual": raw["v4se23_ch32_s42_b_tcn_0_residual_bias"],
     "fc1": raw["v4se23_ch32_s42_fc1_weight"], "b_fc1": raw["v4se23_ch32_s42_b_fc1_bias"],
     "fc2": raw["v4se23_ch32_s42_fc2_weight"], "b_fc2": raw["v4se23_ch32_s42_b_fc2_bias"]}
for bi in (0, 1, 2):
    for nm in ("conv1", "conv2", "se_fc1", "se_fc2"):
        W["w_tcn_%d_%s" % (bi, nm)] = raw["v4se23_ch32_s42_w_tcn_%d_%s_weight" % (bi, nm)]
        W["b_tcn_%d_%s" % (bi, nm)] = raw["v4se23_ch32_s42_b_tcn_%d_%s_bias" % (bi, nm)]

print("[build] 全量 274628 帧 ...", flush=True)
wins, labels = build_windows(start=0, n=274628, phase=0)
atk = np.array(window_labels(wins, labels), dtype=bool)
print("[fwd] %d window ..." % len(wins), flush=True)
probs = np.array([1.0 / (1.0 + math.exp(-c_forward(W, w[2]))) for w in wins])
np.savez(os.path.join(HERE, "fullsim_f32.npz"),
         probs=probs, atk=atk, frames=np.array([w[0] for w in wins]))
print("done, n=%d" % len(wins))
