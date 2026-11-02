"""
diag_xtrain_truth.py  --  用 X_train 真实训练数据 + PyTorch s42 跑推理

训练数据已 fit 归一化过, KEEP_23 取前 23 维, 直接推理看 prob
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, r'C:\work\Claude\Issue')

X_train = np.load(r'C:\work\Claude\Issue\X_train_binary_v2_scada_window16.npy')
y_train = np.load(r'C:\work\Claude\Issue\y_train_binary_v2_scada_window16.npy')

KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
X_23 = X_train[:, :, KEEP_23]
X_23 = np.clip(X_23, -10.0, 10.0)
print(f'X_23 shape: {X_23.shape}')
print(f'y_train distribution: 0={np.sum(y_train==0)}/{len(y_train)}, 1={np.sum(y_train==1)}/{len(y_train)}')

# 加载 s42 模型
from retrain_tcn_23dim_b64_ch32_do01_savept import TCNClassifierSE
ckpt = torch.load(r'C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                  map_location='cpu', weights_only=False)
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# 跑前 1000 个 window 的推理, 看 F1 跟训练时是否一致
n_test = 1000
X_t = torch.from_numpy(X_23[:n_test].transpose(0, 2, 1).astype(np.float32))
with torch.no_grad():
    logits = model(X_t).numpy().squeeze()
probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
y_test = y_train[:n_test]

print(f'\n=== 前 {n_test} window 推理结果 ===')
# 头几个 window
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    print(f'  win {i}: prob={probs[i]:.4f}  y={y_test[i]}')

# 整体统计
from sklearn.metrics import f1_score, roc_auc_score
pred_05 = (probs >= 0.5).astype(int)
f1_05 = f1_score(y_test, pred_05, average='macro')
print(f'\n=== F1m @ thr=0.5: {f1_05:.4f} ===')

# 最关键的: 训练时这个模型的 F1m 是 0.8737 (s42 单一模型)
# 如果这里 F1m ≈ 0.87 → 模型/数据 OK, H743 端有别的 bug
# 如果这里 F1m ≈ 0 → 模型根本没学到东西
print(f'AUC: {roc_auc_score(y_test, probs):.4f}')

# 按真实 binary=0 (normal) 切片, 看误报率
normal_mask = (y_test == 0)
attack_mask = (y_test == 1)
print(f'\n=== 误报率 (normal 里 prob>=0.5) ===')
print(f'normal={normal_mask.sum()}, attack={attack_mask.sum()}')
print(f'FP (误报): {(probs[normal_mask] >= 0.5).sum()} / {normal_mask.sum()} = {(probs[normal_mask] >= 0.5).mean()*100:.2f}%')
print(f'TP (正确检出): {(probs[attack_mask] >= 0.5).sum()} / {attack_mask.sum()} = {(probs[attack_mask] >= 0.5).mean()*100:.2f}%')
