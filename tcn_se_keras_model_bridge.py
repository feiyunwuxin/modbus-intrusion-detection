"""
tcn_se_keras_model_bridge.py  --  PyTorch 镜像 TCN+SE 23-dim 架构
                                    跟 retrain_tcn_23dim_qat.py / savept 同源
                                    用途: 加载 .pt QAT 权重 → 导出 ONNX

不是 QAT-aware 模型 (FakeQuant 已在训练时 baked 进 weight rounding),
是干净 PT 推理模型: 量化参数 (scale/zp) 由后续 onnx2tf → tflite converter
自动 calibration 提取.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


SE_REDUCTION = 8
CHANNELS     = 32
N_BLOCKS     = 3
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4]
DROPOUT      = 0.1


class SEBlock(nn.Module):
    """TCN+SE: Channel-wise Squeeze-and-Excitation."""
    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class TCNBlockSE(nn.Module):
    """1 TCN residual block with SE channel attention."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, se_reduction=SE_REDUCTION):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=se_reduction)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return F.relu(x + residual)


class TCNClassifierSE_Torch(nn.Module):
    """跟 retrain_tcn_23dim_qat.py 同源 — dropout 默认 0 (export 时关)"""
    def __init__(self, in_features=23, window=16, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT, se_reduction=SE_REDUCTION):
        super().__init__()
        layers = [TCNBlockSE(in_features, channels, kernel_size, dilations[0], dropout, se_reduction)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout, se_reduction))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, T, C)  → transpose to (B, C, T) for Conv1d
        if x.dim() == 3 and x.shape[1] != self.tcn[0].conv1.in_channels:
            # 假定 (B, T, C) 输入 (Keras-style)
            x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)
