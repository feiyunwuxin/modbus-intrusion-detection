#!/usr/bin/env python3
"""正确版: 用训练好的 27-dim npy + 23-dim 模型做 baseline, 然后控制变量改 time_diff."""
from __future__ import annotations
import sys, numpy as np, torch

MODEL = 'model_v4_se_23dim_b64_ch32_do01_window16_s42.pt'  # try this first

# 找模型文件
import os
for cand in ['model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
             'model_v4_se_23dim_b64ch32_do01_window16_s42.pt']:
    if os.path.exists(cand):
        MODEL = cand; break
print(f'Model: {MODEL}')

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]
CLIP = 10.0

# 加载数据
X27 = np.load('X_test_binary_v2_scada_window16.npy')   # (3432, 16, 27)
y   = np.load('y_test_binary_v2_scada_window16.npy')    # (3432,)
print(f'X27: {X27.shape}, y: {y.shape}, attack ratio: {y.mean():.4f}')

# 加载模型 (用 verify_h743_vs_pc 的 TCNClassifierSE)
import importlib.util
spec = importlib.util.spec_from_file_location('vh', 'verify_h743_vs_pc.py')
vh = importlib.util.module_from_spec(spec); spec.loader.exec_module(vh)
vh.PC_MODEL_PATH = MODEL
model = vh.load_pc_model()
model.eval()


def run(X27_in, label):
    X23 = X27_in[:, :, KEEP_23].copy()
    X23 = np.clip(X23, -CLIP, CLIP)
    x = torch.from_numpy(X23.transpose(0, 2, 1).astype(np.float32))
    with torch.no_grad():
        p = torch.sigmoid(model(x)).squeeze(-1).numpy()
    pred = (p >= 0.49).astype(np.int32)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
    print(f'[{label}] TP={tp} FP={fp} FN={fn} TN={tn}  P={pr:.4f} R={rc:.4f} F1={f1:.4f}')
    return f1


# [A] baseline: 用原 27-dim
print('\n[A] 原始 27-dim (训练分布)')
f1_a = run(X27, 'baseline')

# [B] 把 idx=15 (time_diff) 强制成 -0.91σ 对应的 raw: (z*sigma + mu)
#    mu=1.4268, sigma=1.5211 → z=-0.91 → raw = 1.4268 + (-0.91)*1.5211 = 0.043s
print('\n[B] time_diff 强制为 0.04s (-0.91σ, 模拟 40ms 注入)')
X_b = X27.copy()
X_b[:, :, 15] = 0.04  # raw value, scaling happens in 27-dim (已经是 scaled)
# 实际: npy 已经是 scaled, 所以改 scaled value 而不是 raw
#   scaled(z=-0.91) = (0.04 - 1.4268) / 1.5211 = -0.91
X_b[:, :, 15] = -0.91
f1_b = run(X_b, '40ms simulated')

# [C] 同时改 time_since_same (idx 16) 也强制为 -0.95σ
print('\n[C] time_diff + time_since_same 都强制 (40ms)')
X_c = X27.copy()
X_c[:, :, 15] = -0.91
X_c[:, :, 16] = -0.95
f1_c = run(X_c, '40ms both')

print('\n' + '=' * 70)
print(f'对比 H743 实测: F1=0.5346, Recall=0.3852, Precision=0.8731')
print(f'[A] baseline F1 = {f1_a:.4f}')
print(f'[B] time_diff only F1 = {f1_b:.4f}')
print(f'[C] time_diff + time_since_same F1 = {f1_c:.4f}')
if abs(f1_c - 0.53) < 0.05:
    print('✅✅✅ time_diff+time_since_same 同时错位是根因')
elif abs(f1_b - 0.53) < 0.05:
    print('✅✅ time_diff 错位就够')
else:
    print('❌ 仍不是主因')
