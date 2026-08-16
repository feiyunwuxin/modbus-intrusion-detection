#!/usr/bin/env python3
"""
Unified training script for classic DL models on 19-dim SCADA features (window=16).

Trains sequentially and saves results:
  - CNN1D
  - Vanilla RNN
  - LSTM
  - BiLSTM
  - CNN-LSTM hybrid

Input: X_*_binary_v2_scada.npy (19 dim row-level) + window=16 reshape
No one-hot (function stays as int)
"""

import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

BASE = r"C:\work\Claude\Issue"

# ─── Data ───
X_TR = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada.npy")

# ─── Hyperparameters ───
SEED = 42
WINDOW = 16
EPOCHS = 30
BATCH = 512
LR = 5e-4
WD = 1e-5
PATIENCE = 7
GRAD_CLIP = 0.5
DROPOUT = 0.3
CLIP_VAL = 10.0
HIDDEN = 64

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, in_ch, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


class VanillaRNN(nn.Module):
    def __init__(self, in_ch, hidden=HIDDEN, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.rnn = nn.RNN(in_ch, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        # x: (B, F, T) → permute to (B, T, F)
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        x = out[:, -1, :]  # last timestep
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, in_ch, hidden=HIDDEN, num_layers=2, dropout=DROPOUT, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_dim, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        if self.bidirectional:
            x = torch.cat([out[:, -1, :self.lstm.hidden_size], out[:, 0, self.lstm.hidden_size:]], dim=1)
        else:
            x = out[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(self, in_ch, cnn_channels=HIDDEN, lstm_hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, num_layers=1, batch_first=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = x.transpose(1, 2)  # (B, T, C)
        out, _ = self.lstm(x)
        x = self.gap(out.transpose(1, 2)).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


# ─────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────
def train_model(name, model, train_loader, val_loader, test_loader, y_val, y_test,
                y_train_w, epochs=EPOCHS, lr=LR, tag="dl_19dim"):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*72}\n[{name}] params = {n_params:,}\n{'='*72}")

    n_pos = int((y_train_w == 1).sum())
    n_neg = int((y_train_w == 0).sum())
    n_pos = max(n_pos, 1); n_neg = max(n_neg, 1)
    pos_weight = torch.tensor([n_neg / n_pos])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        probs = []
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
        return np.concatenate(probs)

    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss /= train_n

        val_prob = predict(val_loader)
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(y_val, val_pred, average="macro")
        scheduler.step(val_f1m)
        print(f"   ep{epoch:2d}  loss={train_loss:.4f}  val f1m={val_f1m:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"   early stop at epoch {epoch}")
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    val_prob  = predict(val_loader)
    test_prob = predict(test_loader)

    # threshold tuning
    thresholds = np.arange(0.10, 0.91, 0.01)
    sweep = [(float(t), float(f1_score(y_val, (val_prob >= t).astype(int), average="macro"))) for t in thresholds]
    best_idx = int(np.argmax([s[1] for s in sweep]))
    best_thr = sweep[best_idx][0]
    print(f"   best epoch={best_epoch}  best thr={best_thr:.2f}  train_time={train_time:.1f}s")

    def evaluate(prob, y, thr, label):
        pred = (prob >= thr).astype(int)
        return {
            "label": label, "threshold": float(thr),
            "macro_f1":  float(f1_score(y, pred, average="macro")),
            "binary_f1": float(f1_score(y, pred, average="binary")),
            "accuracy":  float((pred == y).mean()),
            "roc_auc":   float(roc_auc_score(y, prob)),
            "pr_auc":    float(average_precision_score(y, prob)),
            "cm":        confusion_matrix(y, pred).tolist(),
        }

    test_thr = evaluate(test_prob, y_test, best_thr, f"test@{best_thr:.2f}")
    print(f"   test@{best_thr:.2f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Params={n_params:,}  Train={train_time:.1f}s")

    # save
    torch.save({"state_dict": best_state, "n_params": n_params, "best_epoch": best_epoch,
                "best_threshold": best_thr, "model_name": name, "preprocessing": "v3_19dim"},
               os.path.join(BASE, f"model_{name}_v3_19dim.pt"))
    meta = {
        "model": name, "n_features": 19, "n_params": int(n_params),
        "window": WINDOW, "best_epoch": best_epoch, "best_threshold": best_thr,
        "train_time_seconds": float(train_time),
        "metrics": {"test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]}},
    }
    with open(os.path.join(BASE, f"processed_meta_{name}_v3_19dim.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {"name": name, "n_params": n_params, "train_time": train_time,
            "test_f1m": test_thr['macro_f1'], "test_prauc": test_thr['pr_auc'],
            "test_bin_f1": test_thr['binary_f1'], "test_acc": test_thr['accuracy']}


# ─────────────────────────────────────────────
# Load data + window
# ─────────────────────────────────────────────
print("STEP 1: load 19-dim row-level ...")
X_train = np.load(X_TR).astype(np.float32)
X_val   = np.load(X_VA).astype(np.float32)
X_test  = np.load(X_TE).astype(np.float32)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)
print(f"   shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

print(f"STEP 2: window={WINDOW} reshape ...")
def make_windows(X, y, win):
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    return Xw, yw
X_tr_w, y_tr_w = make_windows(X_train, y_train, WINDOW)
X_va_w, y_va_w = make_windows(X_val,   y_val,   WINDOW)
X_te_w, y_te_w = make_windows(X_test,  y_test,  WINDOW)
print(f"   windowed: train={X_tr_w.shape} val={X_va_w.shape} test={X_te_w.shape}")

X_tr_w = np.clip(X_tr_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_va_w = np.clip(X_va_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
X_te_w = np.clip(X_te_w, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
N_FEATURES = X_tr_w.shape[1]
print(f"   TCN/CNN input: (B, F={N_FEATURES}, T={WINDOW})")

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_w), torch.from_numpy(y_tr_w)),
                          batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_va_w), torch.from_numpy(y_va_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_te_w), torch.from_numpy(y_te_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Train all 5 DL models
# ─────────────────────────────────────────────
results = []
models_to_train = [
    ("cnn1d_19dim", lambda: CNN1D(N_FEATURES)),
    ("rnn_19dim",    lambda: VanillaRNN(N_FEATURES)),
    ("lstm_19dim",   lambda: LSTMModel(N_FEATURES, bidirectional=False)),
    ("bilstm_19dim", lambda: LSTMModel(N_FEATURES, bidirectional=True)),
    ("cnn_lstm_19dim", lambda: CNNLSTM(N_FEATURES)),
]
for name, factory in models_to_train:
    torch.manual_seed(SEED)
    model = factory().to(device)
    r = train_model(name, model, train_loader, val_loader, test_loader,
                    y_va_w, y_te_w, y_tr_w, epochs=EPOCHS, tag="dl_19dim")
    results.append(r)

# ─────────────────────────────────────────────
# Final comparison
# ─────────────────────────────────────────────
print("\n" + "="*72)
print("CLASSIC DL 19-DIM COMPARISON (5 models)")
print("="*72)
print(f"{'Model':<20} {'Params':>10} {'Train(s)':>10} {'F1m':>8} {'PR-AUC':>8} {'Bin-F1':>8} {'Acc':>8}")
print("-" * 72)
for r in results:
    print(f"{r['name']:<20} {r['n_params']:>10,} {r['train_time']:>10.1f} "
          f"{r['test_f1m']:>8.4f} {r['test_prauc']:>8.4f} {r['test_bin_f1']:>8.4f} {r['test_acc']:>8.4f}")
print("="*72)

with open(os.path.join(BASE, "classic_dl_19dim_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)