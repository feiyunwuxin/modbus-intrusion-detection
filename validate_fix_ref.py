"""
validate_fix_ref.py -- 验证 "修好后的 C 语义" 到底应该等于什么.

三方对比 (同一真实 23-dim window 输入):
  1) PyTorch .pt 模型 (带 BN, 训练时真实 forward)
  2) BN-folded 权重 + PyTorch 对称 same padding   (修好 C kernel 后应等价于此)
  3) ONNX runtime (model_v4_se_23dim_ch32_s42.onnx)
  4) C 语义 (因果) -- 当前 H743 实际跑的
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from h743_view_sim import load_header_weights
from h743_windows import build_windows
from conv_sem_test import forward_c_sem, forward_torch

PT_PATH = "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"
ONNX = "model_v4_se_23dim_ch32_s42.onnx"

A = load_header_weights()

ck = torch.load(PT_PATH, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
sys.path.insert(0, "modbus_replay_simulator")
from modbus_replay.ids_worker_torch import build_tcn_se_model
model = build_tcn_se_model(
    in_ch=ck["n_features"], n_blocks=ck["n_blocks"], channels=ck["channels"],
    kernel_size=ck["kernel_size"], dilations=tuple(ck["dilations"]),
    dropout=ck["dropout"], se_reduction=ck["se_reduction"])
model.load_state_dict(sd)
model.eval()


def pt_forward(x23):
    with torch.no_grad():
        logit = float(model(torch.from_numpy(x23).float().unsqueeze(0)).item())
    return logit


try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    have_onnx = True
except Exception as e:
    print("onnxruntime 不可用:", e)
    have_onnx = False

wins, labels = build_windows(n=74628)
pick = [0, 5, 40, 100, 2000, 4663]
print("%-6s %-14s %-14s %-14s %-14s" % ("case", "PyTorch+BN", "BNfold+对称", "ONNX", "C因果(现在)"))
rows = []
for i in pick:
    fidx, xpc, xh7 = wins[i]
    x = xpc
    l_pt = pt_forward(x)
    l_bf = math.log(max(forward_torch(A, x), 1e-300) / max(1 - forward_torch(A, x), 1e-300))
    l_cs = math.log(max(forward_c_sem(A, x), 1e-300) / max(1 - forward_c_sem(A, x), 1e-300))
    l_on = float("nan")
    if have_onnx:
        out = sess.run(None, {sess.get_inputs()[0].name: x[None, ...]})[0]
        l_on = float(np.array(out).ravel()[0])
    print("%-6d %-14.6f %-14.6f %-14.6f %-14.6f" % (fidx, l_pt, l_bf, l_on, l_cs))
    rows.append((l_pt, l_bf, l_on, l_cs))

r = np.array(rows)
print()
print("  |PyTorch+BN - BNfold+对称| max = %.3e   <- 修好后应 ~0" % np.nanmax(np.abs(r[:, 0] - r[:, 1])))
if have_onnx:
    print("  |PyTorch+BN - ONNX|        max = %.3e" % np.nanmax(np.abs(r[:, 0] - r[:, 2])))
print("  |PyTorch+BN - C因果|       max = %.3e   <- 当前 bug 的偏差" % np.nanmax(np.abs(r[:, 0] - r[:, 3])))
