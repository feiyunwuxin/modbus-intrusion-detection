#!/usr/bin/env python3
"""Pure 1D CNN (no recurrent, no attention) on 23-dim × window=16, 5 seeds.

Used as v2 baseline for cross-arch leaderboard expansion. Compared against
TCN baseline (no SE, 3 residual blocks with dilations) and BiLSTM/BiGRU.

Architecture:
  Conv1d(23→32, k=3, pad=1) → BN → ReLU → Dropout
  Conv1d(32→32, k=3, pad=1) → BN → ReLU → Dropout
  Conv1d(32→32, k=3, pad=1) → BN → ReLU → Dropout
  AdaptiveAvgPool1d(1) → FC(32→32) → ReLU → Dropout → FC(32→1)

Params ≈ 9.5K (much smaller than CNN-LSTM hybrid 162K).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from _common_train import train_one_seed, save_meta_json, BASE_PATH, SEEDS

TAG = "cnn_23dim_w16"
CHANNELS = 32
N_LAYERS = 3
KERNEL_SIZE = 3
DROPOUT = 0.1


class PureCNNClassifier(nn.Module):
    """Pure 1D CNN — stacked Conv1d blocks then GAP → FC."""
    def __init__(self, in_ch=23, channels=CHANNELS, n_layers=N_LAYERS,
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        layers = []
        in_c = in_ch
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_c, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_c = channels
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc_drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, 23, 16)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)


def main():
    print(f"=== Pure CNN 23-dim × window=16 × 5 seeds ===")
    results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        r = train_one_seed(PureCNNClassifier, seed, tag=TAG, epochs=20)
        print(f"  F1m={r['test_macro_f1']:.4f}  BestEp={r['best_epoch']}/{r['actual_epochs_run']}  "
              f"Params={r['n_params']:,}  TrainT={r['train_time_s']:.1f}s")
        results.append(r)

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="Pure CNN", tag=TAG,
                   config={"batch": 64, "lr": 4e-3, "epochs": 20, "channels": CHANNELS,
                           "n_layers": N_LAYERS, "kernel_size": KERNEL_SIZE,
                           "dropout": DROPOUT, "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()