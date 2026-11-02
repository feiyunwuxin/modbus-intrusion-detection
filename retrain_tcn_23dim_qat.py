#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT fine-tune TCN+SE 23-dim ch=32 from existing FP32 weights.

策略: 加载 model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt (FP32, F1=0.87),
插入 FakeQuant modules (per-channel weights, per-tensor activations AFTER ReLU),
用 Adam + warmup+cosine LR + KL-div against FP32 teacher 微调 10 epoch.

Why QAT (vs PTQ)?
  PTQ-only 在 TCN+SE 失败: F1=0.68 (-19% vs Hybrid 0.8737).
  QAT 让权重在训练时适应量化噪声,预期 F1 恢复到 0.85+.

Why fine-tune from FP32 (vs from scratch)?
  1. 节省 ~80% GPU 时间 (10 ep vs 20 ep from scratch)
  2. FP32 教师作为 KL anchor 防止 QAT 跑偏
  3. 业界标准做法 (NVIDIA TensorRT QAT, TFLite QAT examples 都这么做)

Usage:
    python retrain_tcn_23dim_qat.py --seed 42              # 实际训练 (~4h GPU)
    python retrain_tcn_23dim_qat.py --seed 42 --dry_run   # 1 epoch 架构验证
    python retrain_tcn_23dim_qat.py --all                 # 5 seeds 串行

输出:
    model_v4_se_23dim_b64_ch32_do01_window16_qat_s{seed}.pt × 5
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# 复用 retrain_tcn_23dim_b64_ch32_do01_savept.py 的超参
BASE = r"C:\work\Claude\Issue"
BATCH = 64
LR = 2e-3                   # QAT 用 2e-3 (vs 原始 4e-3) 避免扰动 FP32 收敛
WARMUP_EPOCHS = 2           # 前 2 epoch lr 从 2e-4 升到 2e-3
EPOCHS = 10                 # QAT 只需少量 epoch
LR_MIN = 1e-5
CHANNELS = 32
DROPOUT = 0.1
WD = 1e-5
PATIENCE = 3                # QAT 收敛更快, patience 缩短
GRAD_CLIP = 0.5
CLIP_VAL = 10.0
N_BLOCKS = 3
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
SE_REDUCTION = 8
WINDOW = 16

# 23-dim (drops length, setpoint, crc_mean_w, cmd_count_w)
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 21, 23, 24, 25, 26]
N_FEATURES = len(KEEP_23)
SEEDS = [42, 123, 456, 789, 1024]

# QAT 标签
TAG_FP32 = "v4_se_23dim_b64_ch32_do01_window16"
TAG_QAT  = "v4_se_23dim_b64_ch32_do01_window16_qat"


# ═════════════════════════════════════════════════════════════════════
# 模型定义 (与 retrain_tcn_23dim_b64_ch32_do01_savept.py 完全一致)
# ═════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
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


class TCNClassifierSE(nn.Module):
    def __init__(self, in_ch, channels=CHANNELS, n_blocks=N_BLOCKS, kernel_size=KERNEL_SIZE,
                 dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════
# FakeQuant modules (per-channel weights, per-tensor activations)
# ═════════════════════════════════════════════════════════════════════

class FakeQuantPerChannelSymmetric(nn.Module):
    """Per-channel 对称 INT8 FakeQuant. ch_axis=0.

    y = round(clamp(x / scale, -128, 127)) * scale
    其中 scale[c] = max(|x[c]|) / 127 (per-channel)
    STE backward (直通估计器): dL/dx ≈ dL/dy
    """

    def __init__(self, ch_axis=0):
        super().__init__()
        self.ch_axis = ch_axis

    def forward(self, x):
        # Compute scale per channel (no_grad 保证 scale 不参与梯度)
        with torch.no_grad():
            absmax = x.abs().amax(dim=self.ch_axis, keepdim=True).clamp(min=1e-8)
            scale = absmax / 127.0
        # Quantize-dequantize with STE
        x_q = torch.round(x / scale).clamp(-128, 127)
        x_dq = x_q * scale  # dequantize
        # STE: forward = x_dq, backward = dL/dx (identity)
        return x + (x_dq - x).detach()


class FakeQuantPerTensorSymmetric(nn.Module):
    """Per-tensor 对称 INT8 FakeQuant. zp=0.

    y = round(clamp(x / scale, -128, 127)) * scale
    其中 scale = max(|x|) / 127 (per-tensor)
    """

    def forward(self, x):
        with torch.no_grad():
            absmax = x.abs().max().clamp(min=1e-8)
            scale = absmax / 127.0
        x_q = torch.round(x / scale).clamp(-128, 127)
        x_dq = x_q * scale
        return x + (x_dq - x).detach()


def insert_fake_quant(model):
    """在模型里插入 FakeQuant modules.

    策略 (Plan agent 建议):
      - Weights: per-channel FakeQuant on Conv1d.weight + Linear.weight
      - Activations: per-tensor FakeQuant on ReLU 输出 (即每个 conv 后)

    实现: 用 forward pre-hook 拦截, 不破坏原 model 结构 (load_state_dict 兼容)
    """
    fq_weight_modules = []  # 用于收集所有需要 fake-quant 的权重

    # 收集需要 fake-quant 的权重 (按 state_dict key)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            fq_weight_modules.append((name + ".weight", module.weight))

    # 注册 weight fake-quant hook (在 module 之前, 修改 weight 后再 forward)
    weight_fqs = {}
    for name, weight in fq_weight_modules:
        fq = FakeQuantPerChannelSymmetric(ch_axis=0)
        weight_fqs[name] = fq
        # 用 pre-forward hook 替换 weight
        def make_pre_hook(fq_module):
            def hook(module, inp):
                module.weight.data = fq_module(module.weight.data)
            return hook
        # 找到对应的 module
        for n, m in model.named_modules():
            if n + ".weight" == name:
                m.register_forward_pre_hook(make_pre_hook(fq))
                break

    # Activation fake-quant: per-tensor 在每个 ReLU 之后
    # 简化: 用 forward hook 在每个 TCNBlockSE 后拦截
    act_fq = FakeQuantPerTensorSymmetric()

    for block_idx, block in enumerate(model.tcn):
        def make_act_hook(idx):
            def hook(module, inp, out):
                # out is (N, C, T) — apply per-tensor FQ
                return act_fq(out)
            return hook
        block.register_forward_hook(make_act_hook(block_idx))

    # Classifier head: ReLU(fc1) 之后 + sigmoid 前(fc2 输出)
    # 用 fc2 pre-output hook 模拟 "logit 前" 的 fake-quant
    def fc2_pre_hook(module, inp):
        x = inp[0]
        x = act_fq(x)
        return (x,) + inp[1:]
    model.fc2.register_forward_pre_hook(fc2_pre_hook)

    return model, weight_fqs, act_fq


# ═════════════════════════════════════════════════════════════════════
# 数据加载 (与 retrain 完全一致)
# ═════════════════════════════════════════════════════════════════════

def load_data():
    X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, KEEP_23]
    y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val   = np.clip(X_val, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test  = np.clip(X_test, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    return X_train, y_train, X_val, y_val, X_test, y_test


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb)).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ═════════════════════════════════════════════════════════════════════
# 训练 1 个 seed (QAT fine-tune)
# ═════════════════════════════════════════════════════════════════════

def cosine_lr(epoch):
    """Warmup + cosine decay."""
    if epoch < WARMUP_EPOCHS:
        return (LR * 0.1 + (LR - LR * 0.1) * epoch / WARMUP_EPOCHS) / LR
    # Cosine from LR to LR_MIN over remaining epochs
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return (LR_MIN + 0.5 * (LR - LR_MIN) * (1 + np.cos(np.pi * progress))) / LR


def run_one(seed: int, dry_run: bool = False):
    """QAT fine-tune 1 个 seed. dry_run=True 时只跑 1 epoch 验证架构."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    actual_epochs = 1 if dry_run else EPOCHS
    print(f"\n=== seed={seed} {'DRY-RUN' if dry_run else 'QAT fine-tune'} "
          f"({actual_epochs} epoch) ===")

    # 加载 FP32 教师 (用于 KL anchor)
    fp32_pt_path = os.path.join(BASE, f"model_{TAG_FP32}_s{seed}.pt")
    if not os.path.exists(fp32_pt_path):
        raise FileNotFoundError(f"FP32 weights not found: {fp32_pt_path}. "
                                f"Run retrain_tcn_23dim_b64_ch32_do01_savept.py first.")
    print(f"  加载 FP32 教师: {fp32_pt_path}")
    fp32_ckpt = torch.load(fp32_pt_path, map_location="cpu", weights_only=False)
    fp32_sd = fp32_ckpt["state_dict"]
    fp32_f1b = fp32_ckpt["test_metrics"]["test_binary_f1"]
    print(f"  FP32 教师 Test Binary-F1 = {fp32_f1b:.4f}")

    # 数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"  数据: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                              batch_size=BATCH, shuffle=False, num_workers=0)

    # QAT 模型 (加载 FP32 权重起点)
    model = TCNClassifierSE(in_ch=N_FEATURES)
    model.load_state_dict(fp32_sd)
    model, weight_fqs, act_fq = insert_fake_quant(model)

    # FP32 教师 (eval mode, no grad)
    teacher = TCNClassifierSE(in_ch=N_FEATURES)
    teacher.load_state_dict(fp32_sd)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Loss + Optimizer
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_f1m, best_state, best_epoch = -1, None, -1
    history = []
    t0 = time.time()
    for epoch in range(1, actual_epochs + 1):
        # LR schedule
        lr_mult = cosine_lr(epoch - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = LR * lr_mult
        cur_lr = LR * lr_mult

        model.train()
        ep_t0 = time.time()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits_qat = model(xb)
            loss = loss_bce(logits_qat, yb.float())

            # KL anchor against FP32 teacher (0.1 weight)
            with torch.no_grad():
                teacher_logits = teacher(xb)
            qat_prob = torch.sigmoid(logits_qat)
            teacher_prob = torch.sigmoid(teacher_logits).detach()
            # KL(teacher || qat) — 让 QAT 模仿 teacher 概率分布
            kl = F.kl_div(torch.log(qat_prob.clamp(min=1e-7)), teacher_prob,
                          reduction='batchmean')
            loss = loss + 0.1 * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        ep_t = time.time() - ep_t0

        # Eval
        val_prob, val_lbl = predict_probs(model, val_loader)
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(val_lbl, val_pred, average="macro")
        history.append({"epoch": epoch, "val_f1m": float(val_f1m),
                        "epoch_time_s": ep_t, "lr": float(cur_lr)})
        print(f"  ep {epoch:2d}/{actual_epochs}  lr={cur_lr:.2e}  "
              f"val_F1m={val_f1m:.4f}  t={ep_t:.1f}s")

        if val_f1m > best_f1m:
            best_f1m = val_f1m
            # 保存时只存 FP32 weights (QAT 时 quant node 是 module-level state)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    train_time = time.time() - t0
    print(f"  训练完成: {train_time:.1f}s, best_val_F1m={best_f1m:.4f}")

    # Test set eval
    model.load_state_dict(best_state)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=BATCH, shuffle=False, num_workers=0)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)
    test_f1m = float(f1_score(test_lbl, test_pred, average="macro"))
    test_f1b = float(f1_score(test_lbl, test_pred, average="binary"))
    print(f"  Test F1m={test_f1m:.4f}  BinF1={test_f1b:.4f}  "
          f"(FP32 教师: BinF1={fp32_f1b:.4f},  Δ={test_f1b - fp32_f1b:+.4f})")

    if dry_run:
        print(f"\n  [DRY-RUN] 架构验证通过, 跳过 .pt 保存")
        return {
            "seed": int(seed), "dry_run": True,
            "best_val_f1m": float(best_f1m),
            "test_f1m": test_f1m, "test_f1b": test_f1b,
            "fp32_teacher_f1b": float(fp32_f1b),
            "delta_vs_fp32": test_f1b - float(fp32_f1b),
            "train_time_s": float(train_time),
        }

    # === USER-RUN-TRAINING === 占位标记
    # 用户跑实际 20h GPU 训练时, 会进入此分支. 当前 dry_run 默认 True.
    print(f"\n  === USER-RUN-TRAINING: 保存 QAT .pt ===")
    qat_pt_path = os.path.join(BASE, f"model_{TAG_QAT}_s{seed}.pt")
    qat_payload = {
        "state_dict": best_state,
        "n_features": N_FEATURES, "window": WINDOW,
        "n_blocks": N_BLOCKS, "channels": CHANNELS,
        "kernel_size": KERNEL_SIZE, "dilations": DILATIONS,
        "se_reduction": SE_REDUCTION, "dropout": DROPOUT,
        "config": {
            "batch": BATCH, "lr": LR, "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS, "lr_min": LR_MIN,
            "wd": WD, "patience": PATIENCE, "grad_clip": GRAD_CLIP,
            "kl_weight": 0.1,
        },
        "seed": int(seed),
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(best_epoch),
        "best_val_f1m": float(best_f1m),
        "train_time_s": float(train_time),
        "fp32_teacher_f1b": float(fp32_f1b),
        "test_metrics": {
            "test_macro_f1": test_f1m,
            "test_binary_f1": test_f1b,
            "test_accuracy": float((test_pred == test_lbl).mean()),
        },
        "history": history,
        "qat_method": "fine-tune-from-fp32",
        "fake_quant": {
            "weights": "per_channel_symmetric_zp0",
            "activations": "per_tensor_symmetric_zp0_post_relu",
        },
    }
    torch.save(qat_payload, qat_pt_path)
    print(f"  [saved] {qat_pt_path}  ({os.path.getsize(qat_pt_path):,} bytes)")

    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_f1m": float(best_f1m),
        "train_time_s": float(train_time),
        "test_f1m": test_f1m,
        "test_f1b": test_f1b,
        "fp32_teacher_f1b": float(fp32_f1b),
        "delta_vs_fp32": test_f1b - float(fp32_f1b),
        "pt_path": qat_pt_path,
        "pt_size_bytes": os.path.getsize(qat_pt_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='train single seed')
    parser.add_argument('--all', action='store_true', help='train all 5 seeds')
    parser.add_argument('--dry_run', action='store_true', default=True,
                       help='1 epoch architecture verification (default: True)')
    parser.add_argument('--full', dest='dry_run', action='store_false',
                       help='full 10 epoch QAT training (override --dry_run)')
    args = parser.parse_args()

    print("=" * 70)
    print(" QAT fine-tune TCN+SE 23-dim ch=32")
    print("   起点: 已有 FP32 权重 (F1=0.87)")
    print("   方法: per-channel W + per-tensor A FakeQuant + KL anchor")
    print("=" * 70)

    seeds = [args.seed] if args.seed else SEEDS
    results = []
    for seed in seeds:
        r = run_one(seed, dry_run=args.dry_run)
        results.append(r)

    # 汇总
    if not args.dry_run:
        summary = {
            "method": "QAT fine-tune from FP32",
            "seeds": [r["seed"] for r in results],
            "per_seed": results,
        }
        meta_path = os.path.join(BASE, f"processed_meta_{TAG_QAT}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n[saved meta] {meta_path}")


if __name__ == "__main__":
    main()
