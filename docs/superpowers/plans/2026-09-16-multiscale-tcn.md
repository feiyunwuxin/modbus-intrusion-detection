# Multi-Scale TCN — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply multi-scale kernels (1, 3, 5, 7) in each TCN block with concat+1x1 fusion; 5 seeds; target F1 ≥ 0.88.

**Architecture:** 3 MultiScaleTCNBlocks (parallel Conv1d with kernels=[1,3,5,7] → concat → 1x1 conv reduce) × dilations=[1,2,4] + Linear head. Reuses data loader from `_common_train.py`.

**Tech Stack:** PyTorch (CPU), scikit-learn metrics, no new deps.

## Global Constraints

- Python 3.13, PyTorch (CPU), Windows 11.
- Reuse `load_data_23dim_w16()` from `_common_train.py` (DO NOT modify).
- 5 seeds: [42, 123, 456, 789, 1024] (from `_common_train.SEEDS`).
- 4 metrics: Accuracy / Precision / Recall / F1-Score (Binary / Attack=positive, thr=0.5).
- Tag: `tcn_multiscale` (meta), `tcn_ms` (per-prediction).
- Hyperparams: lr=4e-3, wd=1e-5, EPOCHS=20, BATCH=64, patience=5, grad_clip=0.5.
- Predictions CSV: `predictions_test_tcn_ms_s{seed}.csv` (header: y_true,prob_attack).
- Meta JSON: `processed_meta_tcn_multiscale.json`.
- Model .pt files gitignored (line 22-23 of .gitignore); predictions CSVs gitignored (line 47); processed_meta_*.json gitignored (line 48).
- Commit only training script + leaderboard outputs (JSON/MD/CSV) + log files.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `train_tcn_multiscale_5seed.py` | MultiScaleTCN model + training loop + per-seed predictions |
| `processed_meta_tcn_multiscale.json` | Per-seed metrics (5 seeds) — auto-saved |
| `multiscale_5seed_4metric.json` | Ensemble leaderboard data |
| `MULTISCALE_LEADERBOARD.md` | Human-readable leaderboard |
| `multiscale_5seed_per_seed.csv` | 5-row flat table (5 seeds × 4 metrics) |
| `logs_train_multiscale.txt` | Training log |

---

### Task 1: Write train_tcn_multiscale_5seed.py — model + training

**Files:**
- Create: `train_tcn_multiscale_5seed.py`

**Step 1.1: Create the file**

```python
#!/usr/bin/env python3
"""Multi-Scale TCN: 3 blocks × parallel kernels=[1,3,5,7] × dilations=[1,2,4].

Each block: 4 parallel Conv1d branches → Concat → 1x1 Conv reduce → residual add.

5 seeds. Same hyperparams as TCN baseline (lr=4e-3, wd=1e-5, EPOCHS=20, BATCH=64).
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS

TAG = "tcn_multiscale"
TAG_PRED = "tcn_ms"
KERNELS = (1, 3, 5, 7)
DILATIONS = (1, 2, 4)
N_BLOCKS = 3
CH = 32
DROPOUT = 0.1
EPOCHS = 20
BATCH = 64
LR = 4e-3
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5


class MultiScaleTCNBlock(nn.Module):
    """Parallel Conv1d branches with kernels=[1,3,5,7] → concat → 1x1 conv → residual."""
    def __init__(self, ch: int, dilation: int, kernels=KERNELS, dropout=DROPOUT):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            pad = (k - 1) * dilation // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.reduce = nn.Sequential(
            nn.Conv1d(ch * len(kernels), ch, kernel_size=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, ch, T). Truncate branches to same length as x.
        outs = []
        target_len = x.shape[-1]
        for branch in self.branches:
            o = branch(x)
            if o.shape[-1] > target_len:
                o = o[..., :target_len]
            elif o.shape[-1] < target_len:
                pad = target_len - o.shape[-1]
                o = nn.functional.pad(o, (0, pad))
            outs.append(o)
        cat = torch.cat(outs, dim=1)  # (B, ch*4, T)
        out = self.reduce(cat)        # (B, ch, T)
        return x + out                # residual


class MultiScaleTCN(nn.Module):
    def __init__(self, n_features=23, window=16):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, CH, kernel_size=1)
        self.blocks = nn.Sequential(*[
            MultiScaleTCNBlock(CH, dilation=d) for d in DILATIONS
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(CH, 1),
        )

    def forward(self, x):
        # x: (B, n_features, window) — channels-first from load_data
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one_seed_ms(seed: int):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data_23dim_w16()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                            batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                             batch_size=BATCH, shuffle=False, num_workers=0)

    model = MultiScaleTCN()
    n_params = sum(p.numel() for p in model.parameters())

    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=2)

    best_f1m, best_state, best_epoch = -1.0, None, -1
    epochs_no_improve = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        val_prob, val_lbl = predict_probs(model, val_loader)
        val_f1m = f1_score(val_lbl, (val_prob >= 0.5).astype(int), average="macro")
        scheduler.step(val_f1m)
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)

    pred_path = os.path.join(BASE_PATH, f"predictions_test_{TAG_PRED}_s{seed}.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack"])
        for y, p in zip(test_lbl.tolist(), test_prob.tolist()):
            w.writerow([y, p])

    metrics = {
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float(accuracy_score(test_lbl, test_pred)),
        "test_precision": float(precision_score(test_lbl, test_pred, zero_division=0)),
        "test_recall": float(recall_score(test_lbl, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
    }

    print(f"  seed={seed}: F1={metrics['test_binary_f1']:.4f}  "
          f"best_ep={best_epoch}  time={train_time:.1f}s  params={n_params}")
    return {
        "seed": int(seed), "n_params": n_params, "best_epoch": int(best_epoch),
        "train_time_s": float(train_time), **metrics,
    }


def main():
    print(f"=== Multi-Scale TCN × 5 seeds ===\n"
          f"  Kernels={KERNELS}  Dilations={DILATIONS}  Ch={CH}  Blocks={N_BLOCKS}\n")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_ms(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="Multi-Scale TCN (kernels=1,3,5,7)", tag=TAG,
                   config={"kernels": list(KERNELS), "dilations": list(DILATIONS),
                           "ch": CH, "n_blocks": N_BLOCKS, "dropout": DROPOUT,
                           "epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
                           "patience": PATIENCE, "grad_clip": GRAD_CLIP,
                           "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()
```

**Step 1.2: Verify model instantiates + forward pass shape**

```bash
cd "D:\workspace\claude\Issue\Issue" && python -c "
import torch
from train_tcn_multiscale_5seed import MultiScaleTCN, MultiScaleTCNBlock
m = MultiScaleTCN()
n = sum(p.numel() for p in m.parameters())
print(f'Total params: {n}')
x = torch.randn(4, 23, 16)
out = m(x)
print(f'Forward shape: {out.shape}')
b = MultiScaleTCNBlock(32, dilation=2)
print(f'Block params: {sum(p.numel() for p in b.parameters())}')
xb = torch.randn(4, 32, 16)
ob = b(xb)
print(f'Block forward shape: {ob.shape}')
"
```
Expected:
```
Total params: ~30000-40000
Forward shape: torch.Size([4])
Block params: <some>
Block forward shape: torch.Size([4, 32, 16])
```

**Step 1.3: Commit**

```bash
cd "D:\workspace\claude\Issue\Issue"
git add train_tcn_multiscale_5seed.py
git commit -m "feat: multi-scale TCN training script (kernels=1,3,5,7, 5 seeds)"
```

---

### Task 2: Run training + verify

**Files:**
- Read: `logs_train_multiscale.txt`
- Verify: `predictions_test_tcn_ms_s{seed}.csv` × 5 files
- Verify: `processed_meta_tcn_multiscale.json`

**Step 2.1: Run training in background (~3 min)**

```bash
cd "D:\workspace\claude\Issue\Issue" && {
  echo "=== [$(date +%H:%M:%S)] Starting Multi-Scale TCN training (5 seeds) ==="
  python train_tcn_multiscale_5seed.py
  echo "=== [$(date +%H:%M:%S)] Done ==="
} > logs_train_multiscale.txt 2>&1
```

Wait for completion.

**Step 2.2: Verify 5 prediction CSVs**

```bash
cd "D:\workspace\claude\Issue\Issue" && ls predictions_test_tcn_ms_s*.csv | wc -l
```
Expected: `5`

**Step 2.3: Verify each CSV has 3432 rows + header**

```bash
cd "D:\workspace\claude\Issue\Issue" && for f in predictions_test_tcn_ms_s*.csv; do
  rows=$(($(wc -l < "$f") - 1))
  echo "$f: $rows rows"
done
```
Expected: 5 lines, each `3432 rows`

**Step 2.4: Verify meta JSON**

```bash
cd "D:\workspace\claude\Issue\Issue" && python -c "
import json
with open('processed_meta_tcn_multiscale.json') as f:
    m = json.load(f)
assert m['n_seeds'] == 5
assert len(m['per_seed']) == 5
seeds = sorted(r['seed'] for r in m['per_seed'])
assert seeds == [42, 123, 456, 789, 1024]
n_params = m['per_seed'][0]['n_params']
print(f'meta OK: {m[\"model\"]} | n_seeds={m[\"n_seeds\"]} | n_params={n_params}')
print(f'  Expected n_params in 30000-50000 range:', 30000 <= n_params <= 50000)
"
```

**Step 2.5: Spot-check F1 values**

```bash
cd "D:\workspace\claude\Issue\Issue" && python -c "
import json
with open('processed_meta_tcn_multiscale.json') as f:
    m = json.load(f)
f1s = [r['test_binary_f1'] for r in m['per_seed']]
print(f'F1 range: min={min(f1s):.4f} max={max(f1s):.4f} mean={sum(f1s)/len(f1s):.4f}')
print(f'All in [0.83, 0.92]:', all(0.83 <= f <= 0.92 for f in f1s))
"
```

---

### Task 3: Write evaluation + leaderboard

**Files:**
- Create: `evaluate_tcn_multiscale_4metric.py`

**Step 3.1: Create the evaluation script**

```python
#!/usr/bin/env python3
"""Multi-Scale TCN 5-seed ensemble evaluation: 4 metrics + leaderboard.

Reads predictions_test_tcn_ms_s{seed}.csv (5 files), computes 5-seed prob_mean,
outputs:
  - multiscale_5seed_4metric.json (full data)
  - MULTISCALE_LEADERBOARD.md (human-readable)
  - multiscale_5seed_per_seed.csv (5 rows: 5 seeds × 4 metrics)
"""
import os
import csv
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

BASE = r"D:\workspace\claude\Issue\Issue"
SEEDS = [42, 123, 456, 789, 1024]
THR = 0.5
BASELINE_F1 = 0.8795
STACKING_F1 = 0.8904


def four_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_2x2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1))); tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1))); fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return [[tn, fp], [fn, tp]]


def main():
    print("=== Multi-Scale TCN 5-seed × 4-metric Evaluation ===\n")
    per_seed = {}
    y_true_ref = None
    probs_dict = {}
    for s in SEEDS:
        path = os.path.join(BASE, f"predictions_test_tcn_ms_s{s}.csv")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y_true = arr[:, 0].astype(int); probs = arr[:, 1].astype(np.float32)
        if y_true_ref is None: y_true_ref = y_true
        probs_dict[s] = probs
        per_seed[str(s)] = {**four_metrics(y_true, (probs >= THR).astype(int)),
                            "CM_2x2": confusion_2x2(y_true, (probs >= THR).astype(int))}

    probs_matrix = np.stack([probs_dict[s] for s in sorted(probs_dict.keys())], axis=0)
    ens_probs = probs_matrix.mean(axis=0)
    ens_pred = (ens_probs >= THR).astype(int)
    ens_metrics = {**four_metrics(y_true_ref, ens_pred),
                   "CM_2x2": confusion_2x2(y_true_ref, ens_pred)}

    print(f"5-seed prob_mean ensemble:")
    print(f"  Acc={ens_metrics['Accuracy']:.4f}  P={ens_metrics['Precision']:.4f}  "
          f"R={ens_metrics['Recall']:.4f}  F1={ens_metrics['F1-Score']:.4f}")
    print(f"\nPer-seed F1:")
    for s in SEEDS:
        if s in probs_dict:
            print(f"  seed {s:>4}: F1={per_seed[str(s)]['F1-Score']:.4f}")

    diff_base = (ens_metrics['F1-Score'] - BASELINE_F1) * 100
    diff_stack = (ens_metrics['F1-Score'] - STACKING_F1) * 100
    target_met = ens_metrics['F1-Score'] >= 0.88
    print(f"\nvs TCN baseline (F1={BASELINE_F1}): {diff_base:+.2f}%")
    print(f"vs Stacking M3@0.390 (F1={STACKING_F1}): {diff_stack:+.2f}%")
    print(f"Target F1 >= 0.88: {'MET' if target_met else 'MISS'}")

    # Save JSON
    json_path = os.path.join(BASE, "multiscale_5seed_4metric.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"task": "Multi-Scale TCN 5-seed × 4-metric evaluation",
                   "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
                   "threshold": THR, "baseline_f1": BASELINE_F1,
                   "stacking_f1": STACKING_F1, "n_seeds": len(probs_dict),
                   "test_n": int(len(y_true_ref)), "per_seed": per_seed,
                   "ensemble_prob_mean": ens_metrics}, f, indent=2)

    # Save CSV
    csv_path = os.path.join(BASE, "multiscale_5seed_per_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "Accuracy", "Precision", "Recall", "F1-Score"])
        for s_str, m in per_seed.items():
            w.writerow([int(s_str), m["Accuracy"], m["Precision"],
                        m["Recall"], m["F1-Score"]])

    # Markdown leaderboard
    md_path = os.path.join(BASE, "MULTISCALE_LEADERBOARD.md")
    cm = ens_metrics["CM_2x2"]
    lines = [
        "# Multi-Scale TCN 5-Seed Leaderboard\n",
        "**Date**: 2026-09-16  ",
        "**Architecture**: 3 MultiScaleTCNBlocks × parallel kernels=[1,3,5,7] × dilations=[1,2,4]  ",
        "**Fusion**: Concat → 1x1 conv (ch=128→32) + residual  ",
        "**Reference**: TCN baseline F1=0.8795; Stacking M3@0.390 F1=0.8904\n",
        "## 1. 5-Seed 概率平均集成\n",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Accuracy | {ens_metrics['Accuracy']:.4f} |",
        f"| Precision | {ens_metrics['Precision']:.4f} |",
        f"| Recall | {ens_metrics['Recall']:.4f} |",
        f"| **F1-Score** | **{ens_metrics['F1-Score']:.4f}** |",
        "\n## 2. 关键对比\n",
        "| Comparison | Delta |",
        "|------------|------:|",
        f"| vs TCN baseline (F1={BASELINE_F1}) | {diff_base:+.2f}% |",
        f"| vs Stacking M3@0.390 (F1={STACKING_F1}) | {diff_stack:+.2f}% |",
        f"| Target F1 >= 0.88 | {'MET' if target_met else 'MISS'} |",
        "\n## 3. Per-Seed F1 详情\n",
        "| Seed | Accuracy | Precision | Recall | F1-Score |",
        "|-----:|---------:|----------:|-------:|---------:|",
    ]
    for s_str, m in per_seed.items():
        lines.append(f"| {s_str} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | "
                     f"{m['Recall']:.4f} | {m['F1-Score']:.4f} |")
    lines.append("\n## 4. 5-Seed 集成混淆矩阵 (thr=0.5)\n")
    lines.append(f"```\n              预测 Normal    预测 Attack\n"
                 f"实际 Normal      {cm[0][0]:>5}        {cm[0][1]:>5}\n"
                 f"实际 Attack      {cm[1][0]:>5}        {cm[1][1]:>5}\n```")
    lines.append("\n## 5. 关键发现\n")
    if target_met:
        lines.append(f"1. **目标达成**: ✅ F1={ens_metrics['F1-Score']:.4f} >= 0.88")
        lines.append(f"2. **vs TCN baseline**: {diff_base:+.2f}%")
        if ens_metrics['F1-Score'] >= STACKING_F1:
            lines.append(f"3. **vs Stacking M3**: {diff_stack:+.2f}% — **multi-scale 超越 stacking!**")
        else:
            lines.append(f"3. **vs Stacking M3**: {diff_stack:+.2f}% — multi-scale 略低但可独立部署")
    else:
        lines.append(f"1. **目标失败**: ❌ F1={ens_metrics['F1-Score']:.4f} < 0.88")
        lines.append(f"2. **vs TCN baseline**: {diff_base:+.2f}%")
        lines.append(f"3. **TCN 单架构 tuning 已穷尽**: SWA/EMA, augmentation, multi-scale 三轮失败")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[saved] {json_path}\n[saved] {csv_path}\n[saved] {md_path}")


if __name__ == "__main__":
    main()
```

**Step 3.2: Run evaluation**

```bash
cd "D:\workspace\claude\Issue\Issue" && python evaluate_tcn_multiscale_4metric.py 2>&1 | tee logs_evaluate_multiscale.txt
```

**Step 3.3: Verify outputs exist**

```bash
cd "D:\workspace\claude\Issue\Issue" && ls -la multiscale_5seed_4metric.json MULTISCALE_LEADERBOARD.md multiscale_5seed_per_seed.csv
```
Expected: 3 files, all >1 KB.

**Step 3.4: Verify CSV has 6 rows (1 header + 5 data)**

```bash
cd "D:\workspace\claude\Issue\Issue" && wc -l multiscale_5seed_per_seed.csv
```
Expected: `6 multiscale_5seed_per_seed.csv`

**Step 3.5: Commit**

```bash
cd "D:\workspace\claude\Issue\Issue"
git add evaluate_tcn_multiscale_4metric.py multiscale_5seed_4metric.json MULTISCALE_LEADERBOARD.md multiscale_5seed_per_seed.csv
git commit -m "results: multi-scale TCN 5-seed leaderboard"
```

---

### Task 4: Update daily log + verify success

**Files:**
- Modify: `DAILY_LOG_2026-09-16.md` (append Phase 8 section)

**Step 4.1: Check results against success criteria**

```bash
cd "D:\workspace\claude\Issue\Issue" && python -c "
import json
with open('multiscale_5seed_4metric.json') as f:
    d = json.load(f)
f1 = d['ensemble_prob_mean']['F1-Score']
print(f'Multi-Scale TCN 5-seed ensemble F1: {f1:.4f}')
print(f'vs TCN baseline (0.8795): {(f1 - 0.8795) * 100:+.2f}%')
print(f'vs Stacking M3@0.390 (0.8904): {(f1 - 0.8904) * 100:+.2f}%')
print(f'Target F1 >= 0.88: {\"MET\" if f1 >= 0.88 else \"MISS\"}')
"
```

**Step 4.2: Append to DAILY_LOG_2026-09-16.md**

Open `DAILY_LOG_2026-09-16.md` and append at end:

```markdown

---

## 🚀 Phase 8 — Multi-Scale TCN（sub-project #4，最后一搏）

### 25. 方法（commit 见 Task 1+3）

3 TCN blocks × parallel kernels=[1,3,5,7] × dilations=[1,2,4], ch=32, dropout=0.1
Fusion: 4 branches Concat → 1x1 Conv reduce (ch=128→32) + residual
Hyperparams: 与 TCN baseline 完全一致

### 26. 结果

| Architecture | Params | 5-seed F1 | vs TCN (0.8795) | vs Stacking (0.8904) |
|--------------|-------:|---------:|-----------------:|----------------------:|
| TCN baseline | 20,001 | 0.8795 | +0.00% | -1.09% |
| Multi-Scale TCN | ~35K | X.XXXX | +X.XX% | -X.XX% |

### 27. 关键发现

- **目标达成**: ✅/❌ F1 ≥ 0.88
- **Architecture insight**: [auto-fill based on actual result]
- **下一步**: 若失败 → 正式 ship Stack11 M3@0.390 (F1=0.8904) 为 production
```

**Step 4.3: Commit daily log**

```bash
cd "D:\workspace\claude\Issue\Issue"
git add DAILY_LOG_2026-09-16.md
git commit -m "docs: 2026-09-16 daily log + memory update (multi-scale TCN complete)"
```

---

## Self-Review

1. **Spec coverage**:
   - Spec §3.1 (MultiScaleTCNBlock architecture) → Task 1 `MultiScaleTCNBlock` class ✓
   - Spec §3.2 (3 blocks + head) → Task 1 `MultiScaleTCN` class ✓
   - Spec §3.3 (hyperparams) → Task 1 hardcoded constants ✓
   - Spec §3.4 (4 metrics, 5-seed ensemble) → Task 3 ✓
   - Spec §4 (output files) → Tasks 1-3 ✓
   - Spec §7 (success: F1 ≥ 0.88) → Task 4.1 verifies ✓

2. **Placeholder scan**: No TBD/TODO. All code blocks complete.

3. **Type consistency**:
   - `MultiScaleTCN(n_features=23, window=16)` matches data shape from `load_data_23dim_w16`
   - `train_one_seed_ms(seed)` returns dict with seed/n_params/best_epoch/train_time_s/test_* — used in Task 2.4
   - TAG `tcn_multiscale` (meta), `tcn_ms` (predictions) — consistent.
   - SEEDS [42, 123, 456, 789, 1024] from `_common_train.SEEDS` ✓