#!/usr/bin/env python3
"""Ablation v2: even smaller TCN v4+SE variants. Push from 8K → 543 params."""
import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"
X_TR = os.path.join(BASE, "X_train_binary.npy")
X_VA = os.path.join(BASE, "X_val_binary.npy")
X_TE = os.path.join(BASE, "X_test_binary.npy")
Y_TR = os.path.join(BASE, "y_train_binary.npy")
Y_VA = os.path.join(BASE, "y_val_binary.npy")
Y_TE = os.path.join(BASE, "y_test_binary.npy")

SEED, WINDOW, EPOCHS, BATCH = 42, 16, 35, 512
LR, WD, PATIENCE, GRAD_CLIP, DROPOUT, CLIP_VAL = 5e-4, 1e-5, 7, 0.5, 0.3, 10.0
DILATIONS, KERNEL_SIZE, SE_REDUCTION = [1, 2, 4], 3, 8
FUNCTION_COL = 1
torch.manual_seed(SEED); np.random.seed(SEED)

print("Loading data ...", flush=True)
X_train_raw = np.load(X_TR).astype(np.float32)
X_val_raw   = np.load(X_VA).astype(np.float32)
X_test_raw  = np.load(X_TE).astype(np.float32)
y_train     = np.load(Y_TR).astype(np.int64)
y_val       = np.load(Y_VA).astype(np.int64)
y_test      = np.load(Y_TE).astype(np.int64)

def make_windows(X, y, w=WINDOW):
    n = (len(X) // w) * w
    Xw = X[:n].reshape(-1, w, X.shape[-1])
    yw = y[:n].reshape(-1, w)
    return Xw, (yw.sum(axis=1) >= 1).astype(np.int64)

X_tr_w, y_tr_w = make_windows(X_train_raw, y_train)
X_va_w, y_va_w = make_windows(X_val_raw,   y_val)
X_te_w, y_te_w = make_windows(X_test_raw,  y_test)

func_all = np.unique(np.concatenate([
    X_tr_w[..., FUNCTION_COL].astype(np.int64).ravel(),
    X_va_w[..., FUNCTION_COL].astype(np.int64).ravel(),
    X_te_w[..., FUNCTION_COL].astype(np.int64).ravel(),
]))
n_cats = len(func_all)
code2idx = {int(c): i for i, c in enumerate(func_all)}

def encode(Xw):
    codes = Xw[..., FUNCTION_COL].astype(np.int64).ravel()
    idx   = np.array([code2idx[int(c)] for c in codes], dtype=np.int64)
    oh    = np.zeros((len(idx), n_cats), dtype=np.float32)
    oh[np.arange(len(idx)), idx] = 1.0
    oh    = oh.reshape(Xw.shape[0], WINDOW, n_cats)
    other = np.delete(Xw, FUNCTION_COL, axis=-1)
    out   = np.concatenate([other, oh], axis=-1)
    return np.clip(out, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1).astype(np.float32)

X_tr = encode(X_tr_w); X_va = encode(X_va_w); X_te = encode(X_te_w)
print(f"   shapes: train={X_tr.shape}", flush=True)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_w)),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te_w)),
                          batch_size=BATCH, shuffle=False, num_workers=0)

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

class TCNBlock(nn.Module):
    def __init__(self, ic, oc, k, d, dr=DROPOUT, use_se=True, r=SE_REDUCTION):
        super().__init__()
        p = (k-1)*d//2
        self.conv1 = nn.Conv1d(ic, oc, k, padding=p, dilation=d)
        self.bn1 = nn.BatchNorm1d(oc)
        self.conv2 = nn.Conv1d(oc, oc, k, padding=p, dilation=d)
        self.bn2 = nn.BatchNorm1d(oc)
        self.drop = nn.Dropout(dr)
        self.se = SEBlock(oc, r) if use_se else nn.Identity()
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        r = self.res(x)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        return F.relu(self.se(x) + r)

class TCN(nn.Module):
    def __init__(self, in_ch, n_blocks, channels, k=KERNEL_SIZE, dilations=DILATIONS, use_se=True, fc_h=32):
        super().__init__()
        layers = [TCNBlock(in_ch, channels, k, dilations[0], use_se=use_se)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlock(channels, channels, k, d, use_se=use_se))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, fc_h)
        self.fc2 = nn.Linear(fc_h, 1)
    def forward(self, x):
        x = self.tcn(x); x = self.gap(x).squeeze(-1)
        return self.fc2(F.dropout(F.relu(self.fc1(x)), 0.3, training=self.training)).squeeze(-1)

@torch.no_grad()
def predict(loader, model):
    model.eval()
    p, l = [], []
    for xb, yb in loader:
        p.append(torch.sigmoid(model(xb)).numpy())
        l.append(yb.numpy())
    return np.concatenate(p), np.concatenate(l)

def train_one(channels, n_blocks, label, use_se=True, fc_h=32):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = TCN(44, n_blocks, channels, use_se=use_se, fc_h=fc_h)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_tr_w == 1).sum()); n_neg = int((y_tr_w == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_f1m, best_state, best_epoch, no_imp = -1, None, -1, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        vp, vl = predict(val_loader, model)
        vfm = f1_score(vl, (vp >= 0.5).astype(int), average="macro")
        sch.step(vfm)
        if vfm > best_f1m:
            best_f1m, best_epoch, no_imp = vfm, ep, 0
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE: break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    tp, tl = predict(test_loader, model)
    macro = f1_score(tl, (tp >= 0.5).astype(int), average="macro")
    binary = f1_score(tl, (tp >= 0.5).astype(int), average="binary")
    auc = roc_auc_score(tl, tp)
    pr = average_precision_score(tl, tp)
    return {"label": label, "channels": channels, "n_blocks": n_blocks,
            "use_se": use_se, "fc_h": fc_h,
            "params": n_params, "best_epoch": best_epoch, "train_time_s": round(train_time, 1),
            "val_macro_f1": round(best_f1m, 4),
            "test_macro_f1": round(macro, 4), "test_binary_f1": round(binary, 4),
            "test_roc_auc": round(auc, 4), "test_pr_auc": round(pr, 4)}

VARIANTS = [
    # (label,                                   channels, n_blocks, use_se, fc_h)
    ("V6  ch=8,  b=3, SE=8",                      8, 3, True,  32),
    ("V10 ch=8,  b=3, NO SE",                     8, 3, False, 32),
    ("V7  ch=8,  b=2, SE=8",                      8, 2, True,  32),
    ("V12 ch=4,  b=2, SE=8",                      4, 2, True,  32),
    ("V15 ch=2,  b=2, NO SE",                     2, 2, False, 32),
]

results = []
print("\n" + "="*100)
print(f"{'Variant':<40} {'Params':>8} {'Best Ep':>8} {'Time(s)':>8} {'Val F1m':>8} {'Tst F1m':>8} {'Tst PR':>8} {'Tst BinF1':>9}")
print("="*100)
for label, ch, nb, se, fch in VARIANTS:
    r = train_one(ch, nb, label, use_se=se, fc_h=fch)
    results.append(r)
    print(f"{r['label']:<40} {r['params']:>8,} {r['best_epoch']:>8} {r['train_time_s']:>8} "
          f"{r['val_macro_f1']:>8.4f} {r['test_macro_f1']:>8.4f} {r['test_pr_auc']:>8.4f} {r['test_binary_f1']:>9.4f}", flush=True)
print("="*100)

# combined view with V0 and V2 from previous round
print("\nCombined view (V0/V2 from previous ablation):")
all_results = [
    {"label": "V0  ch=64, b=3, SE=8 (baseline)",   "params": 79321, "test_macro_f1": 0.8334, "test_pr_auc": 0.9025, "test_binary_f1": 0.8199, "test_roc_auc": 0.8588, "val_macro_f1": 0.8523, "best_epoch": 32, "train_time_s": 60.2},
    {"label": "V2  ch=16, b=3, SE=8 (current best)","params":  7981, "test_macro_f1": 0.8298, "test_pr_auc": 0.9092, "test_binary_f1": 0.8152, "test_roc_auc": 0.8605, "val_macro_f1": 0.8470, "best_epoch": 35, "train_time_s": 31.9},
] + results

print(f"\n{'Variant':<40} {'Params':>8} {'vs V0':>8} {'Tst F1m':>8} {'Δ F1m':>8} {'Tst PR':>8} {'Tst BinF1':>9}")
print("-"*100)
for r in all_results:
    delta = r['test_macro_f1'] - 0.8334
    vs = (r['params'] / 79321 - 1) * 100
    print(f"{r['label']:<40} {r['params']:>8,} {vs:>+7.1f}% {r['test_macro_f1']:>8.4f} {delta:>+8.4f} {r['test_pr_auc']:>8.4f} {r['test_binary_f1']:>9.4f}")

print()
print("Pareto frontier:")
for r in sorted(all_results, key=lambda x: x['params']):
    print(f"  {r['params']:>6,} params  →  F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}")

with open(os.path.join(BASE, "ablation_extreme_compress.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: ablation_extreme_compress.json")
