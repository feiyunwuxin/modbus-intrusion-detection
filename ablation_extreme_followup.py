#!/usr/bin/env python3
"""Follow-up: explain why V6 (ch=8, b=3) fails while V7 (ch=8, b=2) works.
Hypothesis: d=4 dilation collapses with only 8 channels.
"""
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
FUNCTION_COL = 1
torch.manual_seed(SEED); np.random.seed(SEED)

print("Loading data ...", flush=True)
Xtr = np.load(X_TR).astype(np.float32)
Xva = np.load(X_VA).astype(np.float32)
Xte = np.load(X_TE).astype(np.float32)
ytr = np.load(Y_TR).astype(np.int64)
yva = np.load(Y_VA).astype(np.int64)
yte = np.load(Y_TE).astype(np.int64)

def make_windows(X, y, w=WINDOW):
    n = (len(X) // w) * w
    return X[:n].reshape(-1, w, X.shape[-1]), (y[:n].reshape(-1, w).sum(axis=1) >= 1).astype(np.int64)

Xtrw, ytrw = make_windows(Xtr, ytr)
Xvaw, yvaw = make_windows(Xva, yva)
Xtew, ytew = make_windows(Xte, yte)

fA = np.unique(np.concatenate([Xtrw[..., 1].astype(np.int64).ravel(),
                                Xvaw[..., 1].astype(np.int64).ravel(),
                                Xtew[..., 1].astype(np.int64).ravel()]))
n_cats = len(fA); c2i = {int(c): i for i, c in enumerate(fA)}

def encode(Xw):
    codes = Xw[..., 1].astype(np.int64).ravel()
    idx = np.array([c2i[int(c)] for c in codes], dtype=np.int64)
    oh = np.zeros((len(idx), n_cats), dtype=np.float32); oh[np.arange(len(idx)), idx] = 1.0
    oh = oh.reshape(Xw.shape[0], WINDOW, n_cats)
    return np.clip(np.concatenate([np.delete(Xw, 1, axis=-1), oh], axis=-1), -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1).astype(np.float32)

Xtr = encode(Xtrw); Xva = encode(Xvaw); Xte = encode(Xtew)
TL = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytrw)), batch_size=BATCH, shuffle=True)
VL = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yvaw)), batch_size=BATCH, shuffle=False)
TE = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(ytew)), batch_size=BATCH, shuffle=False)

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c, h); self.fc2 = nn.Linear(h, c)
    def forward(self, x):
        s = F.relu(self.fc1(self.gap(x).squeeze(-1)))
        return x * torch.sigmoid(self.fc2(s)).unsqueeze(-1)

class B(nn.Module):
    def __init__(self, ic, oc, k, d, use_se=True, r=8):
        super().__init__()
        p = (k-1)*d//2
        self.c1 = nn.Conv1d(ic, oc, k, padding=p, dilation=d); self.b1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc, oc, k, padding=p, dilation=d); self.b2 = nn.BatchNorm1d(oc)
        self.dr = nn.Dropout(0.3)
        self.se = SE(oc, r) if use_se else nn.Identity()
        self.rs = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        r = self.rs(x)
        x = self.dr(F.relu(self.b1(self.c1(x))))
        x = self.dr(F.relu(self.b2(self.c2(x))))
        return F.relu(self.se(x) + r)

class M(nn.Module):
    def __init__(self, in_ch, nb, ch, dils, use_se=True, fc_h=32):
        super().__init__()
        ls = [B(in_ch, ch, 3, dils[0], use_se)]
        for d in dils[1:nb]:
            ls.append(B(ch, ch, 3, d, use_se))
        self.tcn = nn.Sequential(*ls)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(ch, fc_h); self.fc2 = nn.Linear(fc_h, 1)
    def forward(self, x):
        x = self.tcn(x); x = self.gap(x).squeeze(-1)
        return self.fc2(F.dropout(F.relu(self.fc1(x)), 0.3, training=self.training)).squeeze(-1)

@torch.no_grad()
def P(L, m):
    m.eval(); ps, ls = [], []
    for x, y in L: ps.append(torch.sigmoid(m(x)).numpy()); ls.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ls)

def run(label, ch, nb, dils, use_se=True, fc_h=32):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = M(44, nb, ch, dils, use_se=use_se, fc_h=fc_h)
    np_ = sum(p.numel() for p in m.parameters())
    np_pos = int((ytrw == 1).sum()); np_neg = int((ytrw == 0).sum())
    pw = torch.tensor([np_neg / max(np_pos, 1)])
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    op = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    sc = torch.optim.lr_scheduler.ReduceLROnPlateau(op, mode="max", factor=0.5, patience=2)
    bf, bs, be, ni = -1, None, -1, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        m.train()
        for x, y in TL:
            op.zero_grad(); lf(m(x), y.float()).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP); op.step()
        vp, vl = P(VL, m)
        vfm = f1_score(vl, (vp >= 0.5).astype(int), average="macro")
        sc.step(vfm)
        if vfm > bf:
            bf, be, ni = vfm, ep, 0
            bs = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            ni += 1
            if ni >= PATIENCE: break
    tt = time.time() - t0
    m.load_state_dict(bs)
    tp, tl_ = P(TE, m)
    return {"label": label, "ch": ch, "nb": nb, "dils": dils, "use_se": use_se, "fc_h": fc_h,
            "params": np_, "best_epoch": be, "time": round(tt, 1),
            "val_f1m": round(bf, 4), "test_f1m": round(f1_score(tl_, (tp >= 0.5).astype(int), average="macro"), 4),
            "test_pr": round(average_precision_score(tl_, tp), 4),
            "test_bin": round(f1_score(tl_, (tp >= 0.5).astype(int), average="binary"), 4)}

# Tests
VARIANTS = [
    ("V7  ch=8,  b=2, SE=8 (sweet spot)",          8, 2, [1, 2, 4], True,  32),
    ("V17 ch=8,  b=2, NO SE",                      8, 2, [1, 2, 4], False, 32),
    ("V18 ch=8,  b=2, dil=[1,2] only (no d=4)",    8, 2, [1, 2],    True,  32),
    ("V19 ch=12, b=2, SE=8 (ch=12 middle)",        12, 2, [1, 2, 4], True,  32),
    ("V20 ch=8,  b=2, fc=8 (smaller head)",        8, 2, [1, 2, 4], True,   8),
    ("V21 ch=8,  b=2, dil=[1,2,4], fc=8",          8, 2, [1, 2, 4], True,   8),
]

print(f"\n{'Variant':<50} {'Params':>7} {'Ep':>4} {'Time':>5} {'Val F1m':>8} {'Tst F1m':>8} {'Tst PR':>8}")
print("="*100)
results = []
for label, ch, nb, dils, se, fch in VARIANTS:
    r = run(label, ch, nb, dils, use_se=se, fc_h=fch)
    results.append(r)
    print(f"{r['label']:<50} {r['params']:>7,} {r['best_epoch']:>4} {r['time']:>5} {r['val_f1m']:>8.4f} {r['test_f1m']:>8.4f} {r['test_pr']:>8.4f}", flush=True)

with open(os.path.join(BASE, "ablation_extreme_followup.json"), "w") as f:
    json.dump(results, f, indent=2)
