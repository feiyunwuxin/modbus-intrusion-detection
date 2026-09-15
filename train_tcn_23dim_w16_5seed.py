#!/usr/bin/env python3
"""TCN baseline (no SE) on 23-dim x window=16, 5 seeds.

Hyperparams match retrain_tcn_23dim_b64_ch32_do01_savept.py except:
  - No SEBlock in TCNBlock (this is the baseline)
  - Otherwise identical TCN architecture
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from _common_train import train_one_seed, save_meta_json, BASE_PATH, SEEDS

TAG = "tcn_23dim_w16"
N_BLOCKS = 3
CHANNELS = 32
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
DROPOUT = 0.1


class TCNBlock(nn.Module):
    """TCN block WITHOUT SE — same conv structure as TCN+SE, but no channel attention."""
    def __init__(self, in_ch, out_ch, k, d, dropout):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return F.relu(x + residual)


class TCNClassifier(nn.Module):
    def __init__(self, in_ch=23, channels=CHANNELS, n_blocks=N_BLOCKS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        layers = [TCNBlock(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlock(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc_drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)


def main():
    print(f"=== TCN baseline 23-dim x window=16 x 5 seeds ===")
    results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        r = train_one_seed(TCNClassifier, seed, tag=TAG, epochs=20)
        print(f"  F1m={r['test_macro_f1']:.4f}  BestEp={r['best_epoch']}/{r['actual_epochs_run']}  "
              f"Params={r['n_params']:,}  TrainT={r['train_time_s']:.1f}s")
        results.append(r)

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="TCN (no SE)", tag=TAG,
                   config={"batch": 64, "lr": 4e-3, "epochs": 20, "channels": CHANNELS,
                           "dropout": DROPOUT, "n_blocks": N_BLOCKS, "kernel_size": KERNEL_SIZE,
                           "dilations": DILATIONS, "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()
