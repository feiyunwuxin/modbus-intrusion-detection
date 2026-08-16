#!/usr/bin/env python3
"""
Unified training for lightweight CNN family on 19-dim SCADA (window=16).

Models:
  - SqueezeNet 1D (Fire modules)
  - MobileNet 1D (depthwise separable)
  - ShuffleNet 1D (group conv + channel shuffle)
  - GhostNet 1D (ghost features)
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

X_TR = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada.npy")

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
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cpu")


# ─────────────────────────────────────────────
# SqueezeNet 1D (Fire modules)
# ─────────────────────────────────────────────
class Fire1D(nn.Module):
    def __init__(self, in_ch, squeeze, expand):
        super().__init__()
        self.sq = nn.Conv1d(in_ch, squeeze, 1)
        self.bn_sq = nn.BatchNorm1d(squeeze)
        self.ex1 = nn.Conv1d(squeeze, expand, 1)
        self.ex3 = nn.Conv1d(squeeze, expand, 3, padding=1)
        self.bn_ex1 = nn.BatchNorm1d(expand)
        self.bn_ex3 = nn.BatchNorm1d(expand)
    def forward(self, x):
        s = F.relu(self.bn_sq(self.sq(x)))
        e1 = F.relu(self.bn_ex1(self.ex1(s)))
        e3 = F.relu(self.bn_ex3(self.ex3(s)))
        return torch.cat([e1, e3], dim=1)


class SqueezeNet1D(nn.Module):
    def __init__(self, in_ch=19, base=32, dropout=DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
        )
        self.f1 = Fire1D(base,    base//2, base*2)   # 32 → 16, 64 → 64
        self.f2 = Fire1D(base*4,  base,   base*2)    # 128 → 32, 64 → 128
        self.f3 = Fire1D(base*4,  base*2, base*4)    # 128 → 64, 128 → 128
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base*8, 1)
    def forward(self, x):
        x = self.stem(x)
        x = self.f1(x); x = self.f2(x); x = self.f3(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


# ─────────────────────────────────────────────
# MobileNet 1D (Depthwise Separable)
# ─────────────────────────────────────────────
class DSConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class MobileNet1D(nn.Module):
    def __init__(self, in_ch=19, dropout=DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.b1 = DSConv1D(32,  64)
        self.b2 = DSConv1D(64, 128)
        self.b3 = DSConv1D(128, 128)
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


# ─────────────────────────────────────────────
# ShuffleNet 1D (Group conv + channel shuffle)
# ─────────────────────────────────────────────
def channel_shuffle(x, groups):
    B, C, T = x.shape
    return x.view(B, groups, C // groups, T).transpose(1, 2).contiguous().view(B, C, T)


class ShuffleUnit1D(nn.Module):
    """Simplified ShuffleNet unit: stride=1 only, channel split + branch."""
    def __init__(self, in_ch, out_ch, groups=2):
        super().__init__()
        assert in_ch == out_ch
        assert in_ch % 4 == 0
        mid = in_ch // 4
        self.groups = groups
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch // 2, mid, 1, groups=groups, bias=False),
            nn.BatchNorm1d(mid), nn.ReLU(inplace=True),
            nn.Conv1d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm1d(mid),
            nn.Conv1d(mid, in_ch // 2, 1, groups=groups, bias=False),
            nn.BatchNorm1d(in_ch // 2), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat([self.branch1(x1), x2], dim=1)
        return channel_shuffle(out, self.groups)


class ShuffleNet1D(nn.Module):
    def __init__(self, in_ch=19, base=64, groups=2, n_units=3, dropout=DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
        )
        self.units = nn.Sequential(*[ShuffleUnit1D(base, base, groups=groups) for _ in range(n_units)])
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base, 1)
    def forward(self, x):
        x = self.stem(x); x = self.units(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


# ─────────────────────────────────────────────
# GhostNet 1D (Ghost features)
# ─────────────────────────────────────────────
class GhostConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_ch = out_ch
        init_ch = out_ch // ratio  # intrinsic features
        new_ch = init_ch * (ratio - 1)  # ghost features
        # intrinsic
        self.primary = nn.Sequential(
            nn.Conv1d(in_ch, init_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(init_ch), nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        # ghost via cheap depthwise
        self.cheap = nn.Sequential(
            nn.Conv1d(init_ch, new_ch, dw_size, stride=1, padding=dw_size // 2, groups=init_ch, bias=False),
            nn.BatchNorm1d(new_ch), nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, ratio=2):
        super().__init__()
        self.conv1 = GhostConv1D(in_ch, out_ch, kernel_size=1, ratio=ratio, relu=True)
        self.dw = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, groups=out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.conv2 = GhostConv1D(out_ch, out_ch, kernel_size=1, ratio=ratio, relu=False)
        self.shortcut = nn.Sequential() if (stride == 1 and in_ch == out_ch) else nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )
    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.dw(x)
        x = self.conv2(x)
        return x + res


class GhostNet1D(nn.Module):
    def __init__(self, in_ch=19, base=32, dropout=DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
        )
        self.b1 = GhostBottleneck(base,     base * 2, kernel_size=3, stride=1)
        self.b2 = GhostBottleneck(base * 2, base * 4, kernel_size=3, stride=1)
        self.b3 = GhostBottleneck(base * 4, base * 4, kernel_size=3, stride=1)
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base * 4, 1)
    def forward(self, x):
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_model(name, model, train_loader, val_loader, test_loader, y_val, y_test, y_train_w):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*72}\n[{name}] params = {n_params:,}\n{'='*72}")

    n_pos = int((y_train_w == 1).sum()); n_neg = int((y_train_w == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
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
    for epoch in range(1, EPOCHS + 1):
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
        val_f1m = f1_score(y_val, (val_prob >= 0.5).astype(int), average="macro")
        print(f"   ep{epoch:2d}  loss={train_loss:.4f}  val f1m={val_f1m:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(val_f1m)
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
print("STEP 1: load + window 19-dim ...")
X_train = np.load(X_TR).astype(np.float32)
X_val   = np.load(X_VA).astype(np.float32)
X_test  = np.load(X_TE).astype(np.float32)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)
print(f"   shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

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
print(f"   input: (B, F={N_FEATURES}, T={WINDOW})")

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_w), torch.from_numpy(y_tr_w)),
                          batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_va_w), torch.from_numpy(y_va_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_te_w), torch.from_numpy(y_te_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

results = []
for name, factory in [
    ("squeezenet1d_19dim", lambda: SqueezeNet1D(N_FEATURES)),
    ("mobilenet1d_19dim",  lambda: MobileNet1D(N_FEATURES)),
    ("shufflenet1d_19dim", lambda: ShuffleNet1D(N_FEATURES, base=64, groups=2, n_units=3)),
    ("ghostnet1d_19dim",   lambda: GhostNet1D(N_FEATURES, base=32)),
]:
    torch.manual_seed(SEED)
    model = factory().to(device)
    r = train_model(name, model, train_loader, val_loader, test_loader,
                    y_va_w, y_te_w, y_tr_w)
    results.append(r)

print("\n" + "="*72)
print("LIGHTWEIGHT CNN 19-DIM COMPARISON (4 models)")
print("="*72)
print(f"{'Model':<25} {'Params':>10} {'Train(s)':>10} {'F1m':>8} {'PR-AUC':>8} {'Bin-F1':>8} {'Acc':>8}")
print("-" * 72)
for r in results:
    print(f"{r['name']:<25} {r['n_params']:>10,} {r['train_time']:>10.1f} "
          f"{r['test_f1m']:>8.4f} {r['test_prauc']:>8.4f} {r['test_bin_f1']:>8.4f} {r['test_acc']:>8.4f}")
print("="*72)

with open(os.path.join(BASE, "lightweight_cnn_19dim_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)