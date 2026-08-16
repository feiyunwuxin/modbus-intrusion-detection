#!/usr/bin/env python3
"""
1D-CNN binary intrusion detector for IanArffDataset.
- Same preprocessing as pipeline.py (fillna, clip, temporal 70/30, RobustScaler)
- Conv1D stack + GlobalAvgPool + FC + Sigmoid
- BCEWithLogitsLoss + pos_weight for class imbalance
- Adam + ReduceLROnPlateau + early stopping
- Compare to LightGBM baseline
"""

import os, time, json, sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)

BASE = r"C:\work\Claude\issue"
CSV_IN     = os.path.join(BASE, "IanArffDataset.csv")
SCALER_IN  = os.path.join(BASE, "scaler.joblib")
MODEL_OUT  = os.path.join(BASE, "model_cnn_binary.pt")
EVAL_OUT   = os.path.join(BASE, "evaluation_cnn_binary.txt")
CM_OUT     = os.path.join(BASE, "confusion_matrix_cnn.png")
HIST_OUT   = os.path.join(BASE, "training_history_cnn.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"[device] {DEVICE}", flush=True)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ──────────────────────────────────────────────
# 1. 加载 + 预处理（与 pipeline.py 一致）
# ──────────────────────────────────────────────
log("loading + preprocessing ...")
df = pd.read_csv(CSV_IN).replace("?", np.nan)
float_cols = ["address","function","length","setpoint","gain","reset rate",
              "deadband","cycle time","rate","system mode","control scheme",
              "pump","solenoid","pressure measurement","crc rate","time"]
for c in float_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["binary result","categorized result","specific result","command response"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

df = df.sort_values("time").reset_index(drop=True)

# 特征工程
df["dt"]  = df["time"].diff().fillna(0)
ts = pd.to_datetime(df["time"], unit="s")
df["hour"] = ts.dt.hour; df["dow"] = ts.dt.dayofweek; df["minute"] = ts.dt.minute
for col in ["pressure measurement", "crc rate"]:
    df[f"{col}_diff1"]   = df[col].diff().fillna(0)
    df[f"{col}_rmean4"]  = df[col].rolling(4, min_periods=1).mean()
    df[f"{col}_rstd4"]   = df[col].rolling(4, min_periods=1).std().fillna(0)
    df[f"{col}_rmin4"]   = df[col].rolling(4, min_periods=1).min()
    df[f"{col}_rmax4"]   = df[col].rolling(4, min_periods=1).max()
df["is_read"]  = (df["function"] == 3).astype(int)
df["is_write"] = (df["function"] == 16).astype(int)

# 缺失值
miss_cols = ["setpoint","gain","reset rate","deadband","cycle time",
             "rate","system mode","control scheme","pump","solenoid",
             "pressure measurement"]
df[miss_cols] = df.groupby(["address","function"])[miss_cols].transform(
    lambda s: s.fillna(s.median())
)
for c in miss_cols:
    df[c] = df[c].fillna(df[c].median())
df.loc[df["pressure measurement"].abs() < 1e-30, "pressure measurement"] = np.nan
df.loc[df["crc rate"].abs() < 1e-30, "crc rate"] = np.nan
for c in ["pressure measurement","crc rate"]:
    df[c] = df.groupby(["address","function"])[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df[c].fillna(df[c].median())
df["pressure measurement"] = df["pressure measurement"].clip(0, 100)

# 时序切分
DROP = ["binary result","categorized result","specific result","command response","time"]
X = df.drop(columns=DROP).values.astype(np.float32)
y = df["binary result"].astype(int).values
n = len(df)
split = int(n * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 内部再划 90/10 做验证
val_split = int(len(X_train) * 0.9)
X_tr, X_va = X_train[:val_split], X_train[val_split:]
y_tr, y_va = y_train[:val_split], y_train[val_split:]

log(f"   train={len(X_tr):,}  val={len(X_va):,}  test={len(X_test):,}")

# 归一化（fit 仅用 train）
scaler = RobustScaler()
X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
X_va  = scaler.transform(X_va).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)
joblib.dump(scaler, os.path.join(BASE, "scaler_cnn_binary.joblib"))

# 安全清洗：clip 极端值（避免 conv 数值爆炸），并替换任何残留 NaN
def sanitize(a, lo=-10.0, hi=10.0):
    a = np.nan_to_num(a, nan=0.0, posinf=hi, neginf=lo)
    return np.clip(a, lo, hi).astype(np.float32)
X_tr = sanitize(X_tr); X_va = sanitize(X_va); X_test_s = sanitize(X_test_s)
log(f"   sanitized; X_tr range [{X_tr.min():.2f}, {X_tr.max():.2f}]")

# Conv1D 期望 (B, C, L) — 把特征维度视为序列长度，通道=1
X_tr_t  = torch.from_numpy(X_tr).unsqueeze(1)   # (N, 1, F)
X_va_t  = torch.from_numpy(X_va).unsqueeze(1)
X_te_t  = torch.from_numpy(X_test_s).unsqueeze(1)
y_tr_t  = torch.from_numpy(y_tr).float()
y_va_t  = torch.from_numpy(y_va).float()
y_te_t  = torch.from_numpy(y_test).long()

# DataLoader
BATCH = 1024
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=BATCH*2, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=BATCH*2, shuffle=False, num_workers=0)

# 类别权重
n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
log(f"   pos={n_pos:,}  neg={n_neg:,}  pos_weight={pos_weight.item():.3f}")

# ──────────────────────────────────────────────
# 2. 1D-CNN 模型
# ──────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, in_features, p_drop=0.3):
        super().__init__()
        # 输入 (B, 1, F)
        self.b1 = nn.Sequential(
            nn.Conv1d(1,  64, kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p_drop),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p_drop),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 256, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 1),    # logits
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(x).squeeze(-1)

model = CNN1D(in_features=X_tr.shape[1]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
log(f"   model params: {n_params:,}")

# ──────────────────────────────────────────────
# 3. 训练
# ──────────────────────────────────────────────
EPOCHS = 25
LR = 5e-4
PATIENCE = 4

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

best_f1   = -1.0
best_state= None
patience_left = PATIENCE
history = {"train_loss":[], "val_f1":[], "val_auc":[]}

log("training ...")
for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    epoch_loss = 0.0
    n_batches  = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        # 防 log(0) / NaN loss
        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更紧的裁剪
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
        n_batches  += 1
    epoch_loss = epoch_loss / max(1, n_batches * BATCH) if n_batches else float("nan")
    history["train_loss"].append(epoch_loss)

    # val
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            probs.append(p); labels.append(yb.numpy())
    probs  = np.concatenate(probs)
    labels = np.concatenate(labels)
    probs  = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    pred   = (probs >= 0.5).astype(int)
    f1m    = f1_score(labels, pred, average="macro")
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    history["val_f1"].append(f1m)
    history["val_auc"].append(auc)
    scheduler.step(f1m)
    cur_lr = optimizer.param_groups[0]["lr"]

    log(f"   ep{epoch:02d}  loss={epoch_loss:.4f}  val_f1={f1m:.4f}  val_auc={auc:.4f}  lr={cur_lr:.2e}")

    if f1m > best_f1 + 1e-4:
        best_f1 = f1m
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            log(f"   early stop at epoch {epoch}")
            break

# 恢复最优
if best_state is not None:
    model.load_state_dict(best_state)
torch.save({"state_dict": best_state, "n_features": X_tr.shape[1]}, MODEL_OUT)
log(f"   best val F1 = {best_f1:.4f}, model saved -> {MODEL_OUT}")

# ──────────────────────────────────────────────
# 4. 测试集评估
# ──────────────────────────────────────────────
log("evaluating on test set ...")
model.eval()
probs, labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        p = torch.sigmoid(model(xb)).cpu().numpy()
        probs.append(p); labels.append(yb.numpy())
probs  = np.concatenate(probs)
labels = np.concatenate(labels)
# 安全清洗
probs  = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
pred   = (probs >= 0.5).astype(int)

f1m  = f1_score(labels, pred, average="macro")
f1b  = f1_score(labels, pred, average="binary")
auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
ap   = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
acc  = (pred == labels).mean()

# 阈值扫描：找最佳 F1 阈值
best_thr, best_f1m_thr = 0.5, f1m
for thr in np.linspace(0.1, 0.9, 81):
    p = (probs >= thr).astype(int)
    f = f1_score(labels, p, average="macro")
    if f > best_f1m_thr:
        best_f1m_thr = f; best_thr = thr

lines = []
lines.append("="*70)
lines.append("1D-CNN BINARY CLASSIFICATION REPORT")
lines.append("="*70)
lines.append(f"Model       : Conv1D(64)→Conv1D(128)→Conv1D(256)→GAP→FC(128)→FC(1)")
lines.append(f"Params      : {n_params:,}")
lines.append(f"Device      : {DEVICE}")
lines.append(f"Train/Val/Test : {len(X_tr):,} / {len(X_va):,} / {len(X_test):,}")
lines.append(f"Features    : {X_tr.shape[1]}")
lines.append("")
lines.append("─"*70)
lines.append(f"Test @ threshold=0.50")
lines.append("─"*70)
lines.append(classification_report(labels, pred, target_names=["Normal(0)","Attack(1)"], digits=4))
lines.append(f"Accuracy    : {acc:.4f}")
lines.append(f"Macro-F1    : {f1m:.4f}")
lines.append(f"Binary-F1   : {f1b:.4f}")
lines.append(f"ROC-AUC     : {auc:.4f}")
lines.append(f"PR-AUC (AP) : {ap:.4f}")
lines.append("")
lines.append(f"Best threshold (by Macro-F1) = {best_thr:.3f}  ->  Macro-F1 = {best_f1m_thr:.4f}")
lines.append("")
lines.append("─"*70)
lines.append("COMPARISON vs LightGBM baseline (from pipeline.py)")
lines.append("─"*70)
lines.append(f"{'Metric':<18}  {'1D-CNN':>10}  {'LightGBM':>10}  {'Delta':>10}")
lines.append(f"{'Macro-F1':<18}  {f1m:>10.4f}  {0.7630:>10.4f}  {f1m-0.7630:>+10.4f}")
lines.append(f"{'Accuracy':<18}  {acc:>10.4f}  {0.8201:>10.4f}  {acc-0.8201:>+10.4f}")
lines.append(f"{'ROC-AUC':<18}  {auc:>10.4f}  {'—':>10}  {'—':>10}")
lines.append(f"{'PR-AUC':<18}  {ap:>10.4f}  {'—':>10}  {'—':>10}")
lines.append("")
lines.append("─"*70)
lines.append("TRAINING HISTORY")
lines.append("─"*70)
for i, (l, f, a) in enumerate(zip(history["train_loss"], history["val_f1"], history["val_auc"]), 1):
    lines.append(f"  ep{i:02d}  loss={l:.4f}  val_f1={f:.4f}  val_auc={a:.4f}")

with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# 打印到控制台
for ln in lines: print(ln)
log(f"   report saved -> {EVAL_OUT}")

# ──────────────────────────────────────────────
# 5. 绘图
# ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history["train_loss"], label="train loss", color="C0")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss"); axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(history["val_f1"],  label="val Macro-F1", color="C1")
    axes[1].plot(history["val_auc"], label="val ROC-AUC", color="C2")
    axes[1].axhline(0.7630, ls="--", color="grey", label="LGB Macro-F1 = 0.7630")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].set_title("Validation metrics")
    axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout(); plt.savefig(HIST_OUT, dpi=120); plt.close()
    log(f"   training history plot -> {HIST_OUT}")

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Attack"]); ax.set_yticklabels(["Normal","Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"1D-CNN Confusion Matrix (F1={f1m:.4f})")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(CM_OUT, dpi=120); plt.close()
    log(f"   confusion matrix -> {CM_OUT}")

    pr_prec, pr_rec, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(pr_rec, pr_prec, color="C1")
    ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
    ax[0].set_title(f"PR Curve (AP={ap:.4f})"); ax[0].grid(alpha=0.3)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    ax[1].plot(fpr, tpr, color="C0")
    ax[1].plot([0,1],[0,1],"--", color="grey")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].set_title(f"ROC Curve (AUC={auc:.4f})"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(BASE,"pr_roc_cnn.png"), dpi=120); plt.close()
    log(f"   PR/ROC curves -> pr_roc_cnn.png")
except Exception as e:
    log(f"   plotting skipped: {e}")

log("DONE.")
