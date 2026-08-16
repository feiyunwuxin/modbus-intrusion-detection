#!/usr/bin/env python3
"""Train V19 (ch=12, b=2, SE=8) to get real weights, then quantize to INT8 and compare."""
import os, time, json, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import tracemalloc

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

# ── architecture (V19: ch=12, b=2, SE=8) ───────────────────────────────
class SE(nn.Module):
    def __init__(self, c, r=SE_REDUCTION):
        super().__init__()
        h = max(c // r, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c, h)
        self.fc2 = nn.Linear(h, c)
    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s)).unsqueeze(-1)

class B(nn.Module):
    def __init__(self, ic, oc, k, d):
        super().__init__()
        p = (k-1)*d//2
        self.c1 = nn.Conv1d(ic, oc, k, padding=p, dilation=d); self.b1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc, oc, k, padding=p, dilation=d); self.b2 = nn.BatchNorm1d(oc)
        self.dr = nn.Dropout(DROPOUT)
        self.se = SE(oc)
        self.rs = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        r = self.rs(x)
        x = self.dr(F.relu(self.b1(self.c1(x))))
        x = self.dr(F.relu(self.b2(self.c2(x))))
        return F.relu(self.se(x) + r)

class V19(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.Sequential(B(44, 12, 3, 1), B(12, 12, 3, 2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(12, 32); self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.tcn(x); x = self.gap(x).squeeze(-1)
        return self.fc2(F.dropout(F.relu(self.fc1(x)), 0.3, training=self.training)).squeeze(-1)

# ── load + window + one-hot ─────────────────────────────────────────────
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

fA = np.unique(np.concatenate([
    Xtrw[..., 1].astype(np.int64).ravel(),
    Xvaw[..., 1].astype(np.int64).ravel(),
    Xtew[..., 1].astype(np.int64).ravel(),
]))
n_cats = len(fA); c2i = {int(c): i for i, c in enumerate(fA)}

def encode(Xw):
    codes = Xw[..., 1].astype(np.int64).ravel()
    idx = np.array([c2i[int(c)] for c in codes], dtype=np.int64)
    oh = np.zeros((len(idx), n_cats), dtype=np.float32); oh[np.arange(len(idx)), idx] = 1.0
    oh = oh.reshape(Xw.shape[0], WINDOW, n_cats)
    return np.clip(np.concatenate([np.delete(Xw, 1, axis=-1), oh], axis=-1), -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1).astype(np.float32)

Xtr = encode(Xtrw); Xva = encode(Xvaw); Xte = encode(Xtew)
print(f"   shapes: train={Xtr.shape} val={Xva.shape} test={Xte.shape}", flush=True)

TL = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytrw)), batch_size=BATCH, shuffle=True)
VL = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yvaw)), batch_size=BATCH, shuffle=False)
TE = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(ytew)), batch_size=BATCH, shuffle=False)

# ── train V19 ───────────────────────────────────────────────────────────
print("\nTraining V19 (ch=12, b=2, SE=8) for 35 epochs ...", flush=True)
torch.manual_seed(SEED); np.random_seed = np.random.seed(SEED)
model = V19()
n_params = sum(p.numel() for p in model.parameters())
print(f"   V19 params: {n_params:,}", flush=True)

n_pos = int((ytrw == 1).sum()); n_neg = int((ytrw == 0).sum())
pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
lf = nn.BCEWithLogitsLoss(pos_weight=pw)
op = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
sc = torch.optim.lr_scheduler.ReduceLROnPlateau(op, mode="max", factor=0.5, patience=2)

@torch.no_grad()
def P(L, m):
    m.eval(); ps, ls = [], []
    for x, y in L: ps.append(torch.sigmoid(m(x)).numpy()); ls.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ls)

bf, bs, be, ni = -1, None, -1, 0
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for x, y in TL:
        op.zero_grad(); lf(model(x), y.float()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); op.step()
    vp, vl = P(VL, model)
    vfm = f1_score(vl, (vp >= 0.5).astype(int), average="macro")
    sc.step(vfm)
    if vfm > bf:
        bf, be, ni = vfm, ep, 0
        bs = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        ni += 1
        if ni >= PATIENCE: break

train_time = time.time() - t0
model.load_state_dict(bs)
print(f"   best ep={be}, best val F1m={bf:.4f}, train time={train_time:.1f}s", flush=True)

# save FP32 weights
torch.save({"state_dict": bs, "n_params": n_params, "best_epoch": be, "best_val_f1m": bf},
           os.path.join(BASE, "model_v19_fp32.pt"))

# ── test FP32 ───────────────────────────────────────────────────────────
tp, tl_ = P(TE, model)
f1m_fp32 = f1_score(tl_, (tp >= 0.5).astype(int), average="macro")
pr_fp32  = average_precision_score(tl_, tp)
bin_fp32 = f1_score(tl_, (tp >= 0.5).astype(int), average="binary")
print(f"\n[FP32]   Test F1m={f1m_fp32:.4f}  PR-AUC={pr_fp32:.4f}  Bin-F1={bin_fp32:.4f}", flush=True)

# ── INT8 dynamic quantization ───────────────────────────────────────────
print("\nApplying dynamic INT8 quantization ...", flush=True)
# 1) fresh model with trained weights
fresh = V19()
fresh.load_state_dict(bs)
fresh.eval()
# 2) quantize the trained model
qmodel = torch.ao.quantization.quantize_dynamic(
    fresh, {nn.Linear, nn.Conv1d}, dtype=torch.qint8)
qmodel.eval()

# save INT8
torch.save(qmodel.state_dict(), os.path.join(BASE, "model_v19_int8.pt"))

# test INT8
tp, tl_ = P(TE, qmodel)
f1m_int8 = f1_score(tl_, (tp >= 0.5).astype(int), average="macro")
pr_int8  = average_precision_score(tl_, tp)
bin_int8 = f1_score(tl_, (tp >= 0.5).astype(int), average="binary")
print(f"[INT8]   Test F1m={f1m_int8:.4f}  PR-AUC={pr_int8:.4f}  Bin-F1={bin_int8:.4f}", flush=True)

# ── size comparison ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("Disk size comparison")
print("="*80)
sz_fp32_pt  = os.path.getsize(os.path.join(BASE, "model_v19_fp32.pt"))
sz_int8_pt  = os.path.getsize(os.path.join(BASE, "model_v19_int8.pt"))
print(f"  V19 FP32  PyTorch .pt:  {sz_fp32_pt:>7,} bytes  ({sz_fp32_pt/1024:.2f} KB)")
print(f"  V19 INT8  PyTorch .pt:  {sz_int8_pt:>7,} bytes  ({sz_int8_pt/1024:.2f} KB)")
print(f"  ratio INT8/FP32: {sz_int8_pt/sz_fp32_pt*100:.1f}%")

# raw bytes
def raw_save(state_dict, path):
    """Dump weights as raw float32 bytes (no metadata) — for ultra-compact storage."""
    with open(path, "wb") as f:
        for k, v in state_dict.items():
            f.write(v.detach().cpu().contiguous().numpy().astype(np.float32).tobytes())

raw_save(bs, os.path.join(BASE, "model_v19_fp32_raw.bin"))
sz_fp32_raw = os.path.getsize(os.path.join(BASE, "model_v19_fp32_raw.bin"))
print(f"  V19 FP32  raw bytes:    {sz_fp32_raw:>7,} bytes  ({sz_fp32_raw/1024:.2f} KB)  ← 真正的权重大小")

# INT8 raw bytes (per-tensor absmax + scale)
def raw_save_int8(state_dict, path):
    with open(path, "wb") as f:
        for k, v in state_dict.items():
            t = v.detach().cpu().contiguous().float()
            amax = t.abs().max()
            if amax > 0:
                q = torch.round(t / amax * 127).clamp(-127, 127).to(torch.int8)
            else:
                q = torch.zeros_like(t, dtype=torch.int8)
            f.write(q.numpy().tobytes())
            f.write(np.float32(amax.item()).tobytes())  # per-tensor scale

raw_save_int8(bs, os.path.join(BASE, "model_v19_int8_raw.bin"))
sz_int8_raw = os.path.getsize(os.path.join(BASE, "model_v19_int8_raw.bin"))
print(f"  V19 INT8  raw bytes:    {sz_int8_raw:>7,} bytes  ({sz_int8_raw/1024:.2f} KB)  ← INT8 权重 + scale")

# ── inference latency ───────────────────────────────────────────────────
print("\n" + "="*80)
print("Inference latency (CPU, batch=1, 2000 runs)")
print("="*80)
for name, m in [("FP32", model), ("INT8", qmodel)]:
    m.eval()
    x = torch.randn(1, 44, 16)
    with torch.no_grad():
        for _ in range(20): _ = m(x)  # warmup
    t0 = time.time()
    with torch.no_grad():
        for _ in range(2000): _ = m(x)
    lat = (time.time() - t0) / 2000 * 1000
    print(f"  {name:<6}: {lat:.3f} ms/inference  ({1000/lat:.0f} inf/s)")

# ── RAM at inference ────────────────────────────────────────────────────
print("\n" + "="*80)
print("RAM at inference (CPU, batch=1)")
print("="*80)
for name, m in [("FP32", model), ("INT8", qmodel)]:
    m.eval()
    tracemalloc.start()
    with torch.no_grad():
        _ = m(torch.randn(1, 44, 16))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    weight_mem = sum(p.numel() * p.element_size() for p in m.parameters())
    print(f"  {name:<6}: weights={weight_mem:>5,}B ({weight_mem/1024:.2f}KB)  "
          f"act peak={peak:>5,}B ({peak/1024:.2f}KB)  total={(peak+weight_mem)/1024:.2f}KB")

# ── final summary ────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL SUMMARY: V19 (4,237 params) FP32 vs INT8")
print("="*80)
print(f"{'Metric':<28} {'FP32':>15} {'INT8':>15} {'Δ':>12}")
print("-"*80)
print(f"{'Test F1m':<28} {f1m_fp32:>15.4f} {f1m_int8:>15.4f} {f1m_int8-f1m_fp32:>+12.4f}")
print(f"{'Test PR-AUC':<28} {pr_fp32:>15.4f} {pr_int8:>15.4f} {pr_int8-pr_fp32:>+12.4f}")
print(f"{'Test Bin-F1':<28} {bin_fp32:>15.4f} {bin_int8:>15.4f} {bin_int8-bin_fp32:>+12.4f}")
print(f"{'Disk (.pt)':<28} {sz_fp32_pt:>13,} B {sz_int8_pt:>13,} B {sz_int8_pt-sz_fp32_pt:>+10,} B")
print(f"{'Disk (raw bytes)':<28} {sz_fp32_raw:>13,} B {sz_int8_raw:>13,} B {sz_int8_raw-sz_fp32_raw:>+10,} B")

# cleanup tmp
for f in ['model_v19_fp32_raw.bin', 'model_v19_int8_raw.bin']:
    fp = os.path.join(BASE, f)
    if os.path.exists(fp): os.remove(fp)
