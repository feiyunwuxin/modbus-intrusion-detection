#!/usr/bin/env python3
"""Re-evaluate saved BiLSTM v2 window=8 with the 4 Binary metrics
(Accuracy / Precision / Recall / F1-Score) per
feedback-ids-binary-precision-convention.md.

Inputs:
  - X_test_binary.npy / y_test_binary.npy
  - model_lstm_binary_v2_window8.pt (state_dict + meta, bidirectional=True)

Output:
  - predictions_test_bilstm_v2_window8.csv  (y_true, prob, pred_at_0.5)
  - bistlm_v2_window8_4metrics.json         (Acc / P / R / F1 / Macro-F1)
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

BASE = r"C:\work\Claude\Issue"
CKPT = os.path.join(BASE, "model_lstm_binary_v2_window8.pt")
PRED_OUT = os.path.join(BASE, "predictions_test_bilstm_v2_window8.csv")
JSON_OUT = os.path.join(BASE, "bilstm_v2_window8_4metrics.json")

WINDOW = 8
THRESHOLD = 0.5
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def onehot(values, cats):
    out = np.zeros((len(values), len(cats)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(cats)}
    for r, v in enumerate(values):
        i = idx.get(int(v))
        if i is not None:
            out[r, i] = 1.0
    return out


class BiLSTM(nn.Module):
    def __init__(self, in_features, hidden=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.0)
        # Match saved checkpoint key names (fc1 / fc2) — fc1 output 32
        self.fc1 = nn.Linear(hidden * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Match original lstm_binary_v2_window8.py: concat last-layer forward + backward hidden states
        out, (h, c) = self.lstm(x)
        if self.lstm.bidirectional:
            last = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            last = h[-1]
        h2 = torch.relu(self.fc1(last))
        h2 = self.dropout(h2)
        return self.fc2(h2).squeeze(-1)


def build_windows(X, y, function_idx, window=WINDOW):
    n = (len(X) // window) * window
    Xw = X[:n].reshape(n // window, window, X.shape[1])
    yw = y[:n].reshape(n // window, window).max(axis=1)  # window positive if any row pos
    fw = function_idx[:n].reshape(n // window, window)
    return Xw, yw, fw


def main():
    print("[load] data ...")
    X_te = np.load(os.path.join(BASE, "X_test_binary.npy")).astype(np.float32)
    y_te = np.load(os.path.join(BASE, "y_test_binary.npy")).astype(int)
    fn_te = X_te[:, 0].astype(int)  # 'function' column is index 0
    print(f"   X_test={X_te.shape} y_test={y_te.shape}")

    print("[load] ckpt ...")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    function_values = ckpt["function_values"]
    n_features = ckpt["n_features"]
    print(f"   n_features={n_features} (17 raw + {len(function_values)} one-hot = 17 + 28 = 45 expected, actual {n_features})")
    print(f"   bidirectional={ckpt['bidirectional']} hidden={ckpt['hidden']} window={ckpt['window']}")

    print("[build] sliding windows of 8 ...")
    Xw, yw, fw = build_windows(X_te, y_te, fn_te, window=WINDOW)
    print(f"   Xw={Xw.shape} yw={yw.shape}")

    # Match lstm_binary_v2_window8.py: keep col 0 (address) + cols 2..16 (15 features) = 16 numeric
    # Then concatenate 28-dim one-hot of function at each timestep → 44 dim total
    keep_idx = [0] + list(range(2, 17))
    num_w = Xw[:, :, keep_idx]                          # (N, 8, 16)
    onehot_all = onehot(fw.reshape(-1), function_values) # (N*8, 28)
    onehot_all = onehot_all.reshape(Xw.shape[0], Xw.shape[1], -1)
    Xw_full = np.concatenate([num_w, onehot_all], axis=2).astype(np.float32)
    Xw_full = np.clip(Xw_full, -10.0, 10.0).astype(np.float32)  # match original CLIP_VAL=10
    print(f"   Xw_full={Xw_full.shape} (16 numeric + 28 one-hot = 44, clipped to [-10, 10])")

    print("[model] build + load state_dict ...")
    model = BiLSTM(in_features=n_features, hidden=ckpt["hidden"], num_layers=ckpt["num_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("[predict] sliding windows ...")
    t0 = time.time()
    with torch.no_grad():
        Xt = torch.from_numpy(Xw_full)
        # batched to avoid memory blow-up
        probs = []
        BS = 1024
        for i in range(0, len(Xt), BS):
            logits = model(Xt[i:i+BS])
            probs.append(torch.sigmoid(logits).numpy())
    probs = np.concatenate(probs)
    print(f"   {len(probs)} windows in {time.time()-t0:.2f}s, "
          f"pos rate @0.5={(probs>=0.5).mean():.3f}, true pos rate={yw.mean():.3f}")

    y_pred = (probs >= THRESHOLD).astype(int)

    # 4 Binary metrics (Attack = positive class)
    acc  = accuracy_score(yw, y_pred)
    prec = precision_score(yw, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(yw, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(yw, y_pred, pos_label=1, zero_division=0)
    f1m  = f1_score(yw, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(yw, y_pred).tolist()

    print()
    print(f"   BiLSTM v2 window=8 (Binary convention, thr={THRESHOLD}):")
    print(f"   Accuracy  = {acc:.4f}")
    print(f"   Precision = {prec:.4f}")
    print(f"   Recall    = {rec:.4f}")
    print(f"   F1-Score  = {f1:.4f}")
    print(f"   Macro-F1  = {f1m:.4f}  (reference)")
    print(f"   CM (rows=true [0,1], cols=pred [0,1]): {cm}")

    # save csv
    pd.DataFrame({"y_true": yw, "prob_attack": probs, "pred_0.5": y_pred}).to_csv(PRED_OUT, index=False)
    print(f"   saved: {PRED_OUT}")

    # save json
    out = {
        "model": "BiLSTM v2 window=8",
        "test_set": "binary_v2_window8",
        "n_windows": int(len(yw)),
        "n_normal": int((yw == 0).sum()),
        "n_attack": int((yw == 1).sum()),
        "convention": "Binary / Positive-class (Attack)",
        "threshold": THRESHOLD,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "Macro_F1": float(f1m),
        "confusion_matrix": cm,
        "n_params": int(ckpt["n_params"]),
        "best_epoch": int(ckpt["best_epoch"]),
        "best_val_f1m": float(ckpt["best_val_f1m"]),
        "source_ckpt": os.path.basename(CKPT),
        "history_note": "Originally trained 2026-06-09, F1m=0.8144. Now re-evaluated with Binary convention per feedback-ids-binary-precision-convention.md (2026-09-13)."
    }
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   saved: {JSON_OUT}")


if __name__ == "__main__":
    main()
