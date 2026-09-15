#!/usr/bin/env python3
"""BiGRU classifier on 23-dim × window=16, 5 seeds."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from _common_train import train_one_seed, save_meta_json, BASE_PATH, SEEDS

TAG = "gru_23dim_w16"
HIDDEN = 64
NUM_LAYERS = 1
DROPOUT = 0.1


class BiGRUClassifier(nn.Module):
    def __init__(self, in_ch=23, hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=0.0)
        self.fc1 = nn.Linear(hidden * 2, 32)
        self.fc_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, 23, 16) — transpose to (B, 16, 23) for sequence-first GRU
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        # Concat final forward + backward hidden states (last time step)
        x = torch.cat([out[:, -1, :HIDDEN], out[:, 0, HIDDEN:]], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)


def main():
    print(f"=== BiGRU 23-dim × window=16 × 5 seeds ===")
    results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        r = train_one_seed(BiGRUClassifier, seed, tag=TAG, epochs=20)
        print(f"  F1m={r['test_macro_f1']:.4f}  BestEp={r['best_epoch']}/{r['actual_epochs_run']}  "
              f"Params={r['n_params']:,}  TrainT={r['train_time_s']:.1f}s")
        results.append(r)

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="BiGRU", tag=TAG,
                   config={"batch": 64, "lr": 4e-3, "epochs": 20, "hidden": HIDDEN,
                           "num_layers": NUM_LAYERS, "dropout": DROPOUT,
                           "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()