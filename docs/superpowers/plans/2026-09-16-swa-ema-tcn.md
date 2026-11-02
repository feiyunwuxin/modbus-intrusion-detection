# SWA + EMA Regularization for TCN Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA) to TCN baseline, evaluate 3 views × 5 seeds, target F1 ≥ 0.88 vs baseline 0.8795.

**Architecture:** Self-contained script that reuses TCNClassifier from existing train_tcn_23dim_w16_5seed.py. Implements SWA (epoch ≥ 10 snapshot averaging + BN re-update) and EMA (α=0.999, full trajectory). 3 views (raw / SWA / EMA) × 5 seeds, ensemble via prob_mean, 4 metrics.

**Tech Stack:** PyTorch (existing), scikit-learn metrics (existing), no new deps.

## Global Constraints

- Python 3.13, PyTorch (CPU), Windows 11.
- Reuse `_common_train.load_data_23dim_w16()` for data loading (DO NOT modify `_common_train.py`).
- Reuse `TCNClassifier` from `train_tcn_23dim_w16_5seed.py` (import, don't redefine).
- SEEDS = [42, 123, 456, 789, 1024] from `_common_train.SEEDS`.
- 4 metrics: Accuracy / Precision / Recall / F1-Score (Binary / Attack=positive, threshold 0.5).
- Tag prefix: `tcn_swa_ema` for meta JSON; per-view tags: `tcn_raw`, `tcn_swa`, `tcn_ema`.
- Predictions CSV format: `y_true,prob_attack` (header row + data rows, N=3432 test samples).
- Model .pt files are gitignored (line 22-23 of .gitignore); predictions CSVs are gitignored (line 47); processed_meta_*.json are gitignored (line 48).
- Commit only leaderboard outputs (JSON/MD/CSV) + training script + log files.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `train_tcn_swa_ema_5seed.py` | Training loop with SWA + EMA hooks + 3-view evaluation |
| `processed_meta_tcn_swa_ema.json` | Per-seed metrics for 3 views (auto-saved by training script) |
| `swa_ema_5seed_3view.json` | Ensemble leaderboard data (3 views × 4 metrics + CMs) |
| `SWA_EMA_LEADERBOARD.md` | Human-readable leaderboard with comparison to baseline |
| `swa_ema_5seed_per_view.csv` | Flat table (15 rows: 3 views × 5 seeds × 4 metrics) |
| `logs_train_swa_ema.txt` | Training log (stdout/stderr capture) |

---

### Task 1: Write train_tcn_swa_ema_5seed.py — training loop with hooks

**Files:**
- Create: `train_tcn_swa_ema_5seed.py`

**Step 1.1: Create file header and imports**

```python
#!/usr/bin/env python3
"""TCN baseline + SWA + EMA on 23-dim x window=16, 5 seeds.

Self-contained: reuses TCNClassifier from train_tcn_23dim_w16_5seed.py
and data loader from _common_train.py. Trains 20 epochs with:
  - SWA snapshots from epoch >= 10, averaged + BN re-update at end
  - EMA (alpha=0.999) updated every epoch

Outputs 3 prediction CSVs per seed (raw / SWA / EMA) + meta JSON.
"""
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS
from train_tcn_23dim_w16_5seed import TCNClassifier

TAG_PREFIX = "tcn_swa_ema"
SWA_START_EPOCH = 10
EPOCHS = 20
EMA_DECAY = 0.999
BATCH = 64
LR = 4e-3
WD = 1e-5
PATIENCE = 5
GRAD_CLIP = 0.5


def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb)).numpy())
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def average_state_dicts(state_dicts):
    """Element-wise mean of N state_dicts. Returns new state_dict."""
    avg = copy.deepcopy(state_dicts[0])
    for k in avg.keys():
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        avg[k] = stacked.mean(dim=0).to(avg[k].dtype)
    return avg


def bn_re_update(model, train_loader):
    """Recompute BatchNorm running stats with one forward pass over train data."""
    model.train()
    with torch.no_grad():
        for xb, _ in train_loader:
            model(xb)
    model.eval()


def train_one_seed_swa_ema(seed):
    """Train one seed, evaluate 3 views, save CSVs. Returns metrics dict."""
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

    model = TCNClassifier()
    n_params = sum(p.numel() for p in model.parameters())

    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=2)

    swa_snapshots = []
    ema_state = None
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

        # SWA snapshot
        if epoch >= SWA_START_EPOCH:
            swa_snapshots.append({k: v.clone() for k, v in model.state_dict().items()})

        # EMA update
        cur_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ema_state is None:
            ema_state = cur_state
        else:
            for k in ema_state.keys():
                ema_state[k] = EMA_DECAY * cur_state[k] + (1 - EMA_DECAY) * ema_state[k]

        if epochs_no_improve >= PATIENCE:
            break

    train_time = time.time() - t0
    n_swa_snapshots = len(swa_snapshots)
    swa_warning = ""
    if n_swa_snapshots == 0:
        swa_snapshots = [best_state]
        swa_warning = "FALLBACK: no SWA snapshots (early stop), used best_state"

    # View 1: raw best
    model.load_state_dict(best_state)
    raw_prob, test_lbl = predict_probs(model, test_loader)

    # View 2: SWA averaged + BN re-update
    swa_state = average_state_dicts(swa_snapshots)
    model.load_state_dict(swa_state)
    bn_re_update(model, train_loader)
    swa_prob, _ = predict_probs(model, test_loader)

    # View 3: EMA
    model.load_state_dict(ema_state)
    ema_prob, _ = predict_probs(model, test_loader)

    def four_metrics(probs):
        pred = (probs >= 0.5).astype(int)
        return {
            "test_macro_f1": float(f1_score(test_lbl, pred, average="macro")),
            "test_binary_f1": float(f1_score(test_lbl, pred, average="binary")),
            "test_accuracy": float(accuracy_score(test_lbl, pred)),
            "test_precision": float(precision_score(test_lbl, pred, zero_division=0)),
            "test_recall": float(recall_score(test_lbl, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(test_lbl, probs)),
        }

    raw_metrics = four_metrics(raw_prob)
    swa_metrics = four_metrics(swa_prob)
    ema_metrics = four_metrics(ema_prob)

    # Save 3 prediction CSVs
    for view, probs in [("raw", raw_prob), ("swa", swa_prob), ("ema", ema_prob)]:
        with open(os.path.join(BASE_PATH, f"predictions_test_tcn_{view}_s{seed}.csv"), "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["y_true", "prob_attack"])
            for y, p in zip(test_lbl.tolist(), probs.tolist()):
                w.writerow([y, p])

    print(f"  seed={seed}: "
          f"raw F1={raw_metrics['test_binary_f1']:.4f}  "
          f"SWA F1={swa_metrics['test_binary_f1']:.4f}  "
          f"EMA F1={ema_metrics['test_binary_f1']:.4f}  "
          f"swa_snaps={n_swa_snapshots}  time={train_time:.1f}s"
          + (f"  [{swa_warning}]" if swa_warning else ""))

    return {
        "seed": int(seed),
        "n_params": n_params,
        "best_epoch": int(best_epoch),
        "n_swa_snapshots": n_swa_snapshots,
        "swa_warning": swa_warning,
        "train_time_s": float(train_time),
        "raw": raw_metrics,
        "swa": swa_metrics,
        "ema": ema_metrics,
    }


def main():
    print(f"=== TCN + SWA + EMA: 23-dim x window=16 x 5 seeds ===\n"
          f"  SWA start epoch: {SWA_START_EPOCH}, EMA decay: {EMA_DECAY}\n")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_swa_ema(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG_PREFIX}.json"),
                   model_name="TCN + SWA + EMA", tag=TAG_PREFIX,
                   config={"epochs": EPOCHS, "swa_start_epoch": SWA_START_EPOCH,
                           "ema_decay": EMA_DECAY, "batch": BATCH, "lr": LR, "wd": WD,
                           "patience": PATIENCE, "n_archs": 3,
                           "n_params_input": 23, "window": 16})
    print(f"\n[saved meta] processed_meta_{TAG_PREFIX}.json")


if __name__ == "__main__":
    main()
```

**Step 1.2: Verify imports work**

Run: `cd "D:\workspace\claude\Issue\Issue" && python -c "from train_tcn_23dim_w16_5seed import TCNClassifier; m = TCNClassifier(); print('TCNClassifier OK, params:', sum(p.numel() for p in m.parameters()))"`
Expected: `TCNClassifier OK, params: 20001`

**Step 1.3: Commit**

```bash
cd "D:\workspace\claude\Issue\Issue"
git add train_tcn_swa_ema_5seed.py
git commit -m "feat: TCN + SWA + EMA training script (3 views, 5 seeds)"
```

---

### Task 2: Run training + verify outputs

**Files:**
- Read: `logs_train_swa_ema.txt` (will be created)
- Verify: `predictions_test_tcn_{raw,swa,ema}_s{seed}.csv` × 5 seeds × 3 views = 15 files
- Verify: `processed_meta_tcn_swa_ema.json`

**Step 2.1: Run training in background (~5-7 min for 5 seeds)**

```bash
cd "D:\workspace\claude\Issue\Issue" && {
  echo "=== [$(date +%H:%M:%S)] Starting TCN+SWA+EMA training (5 seeds) ==="
  python train_tcn_swa_ema_5seed.py
  echo "=== [$(date +%H:%M:%S)] Done ==="
} > logs_train_swa_ema.txt 2>&1
```

Run in background. Wait for completion notification.

Expected: 5 lines like `seed=42: raw F1=X.XXXXSwa S=X.XXXX ema F1=Y.YYYY swa_snaps=N time=T.Ts`

**Step 2.2: Verify 15 prediction CSVs exist**

```bash
cd "D:\workspace\claude\Issue\Issue" && ls predictions_test_tcn_{raw,swa,ema}_s*.csv | wc -l
```
Expected: `15`

**Step 2.3: Verify each CSV has 3432 rows + header**

```bash
cd "D:\workspace\claude\Issue\Issue" && for f in predictions_test_tcn_swa_s*.csv; do
  rows=$(($(wc -l < "$f") - 1))
  echo "$f: $rows rows"
done
```
Expected: 5 lines, each saying `predictions_test_tcn_swa_s{seed}.csv: 3432 rows`

**Step 2.4: Verify meta JSON exists and parses**

```bash
cd "D:\workspace\claude\Issue\Issue" && python -c "
import json
with open('processed_meta_tcn_swa_ema.json') as f:
    m = json.load(f)
assert m['n_seeds'] == 5
assert all('raw' in r and 'swa' in r and 'ema' in r for r in m['per_seed'])
print('meta OK:', m['model'])
"
```
Expected: `meta OK: TCN + SWA + EMA`

**Step 2.5: Spot-check 5-seed F1 values are in reasonable range**

```bash
cd "D:\workspace\claude\Issue\Issue" && python -c "
import json
with open('processed_meta_tcn_swa_ema.json') as f:
    m = json.load(f)
for view in ['raw', 'swa', 'ema']:
    f1s = [r[view]['test_binary_f1'] for r in m['per_seed']]
    print(f'{view}: min={min(f1s):.4f} max={max(f1s):.4f} mean={sum(f1s)/len(f1s):.4f}')
"
```
Expected: All 3 views should have mean F1 in [0.85, 0.92] range (TCN baseline is 0.88; SWA/EMA should be similar or better).

If any view mean < 0.83, abort and investigate.

---

### Task 3: Write ensemble evaluation + leaderboard generation

**Files:**
- Create: `evaluate_swa_ema_3view.py`

**Step 3.1: Create the evaluation script**

```python
#!/usr/bin/env python3
"""Evaluate 3 views (raw / SWA / EMA) × 5 seeds on TCN + SWA/EMA.

Reads predictions_test_tcn_{view}_s{seed}.csv for each combination,
computes 5-seed prob_mean ensemble metrics per view, outputs:
  - swa_ema_5seed_3view.json (full data)
  - SWA_EMA_LEADERBOARD.md (human-readable)
  - swa_ema_5seed_per_view.csv (15-row flat table)
"""
import os
import csv
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

BASE = r"D:\workspace\claude\Issue\Issue"
VIEWS = ["raw", "swa", "ema"]
VIEW_DISPLAY = {"raw": "TCN baseline (raw best)", "swa": "SWA (epoch ≥10 avg + BN re-update)", "ema": "EMA (α=0.999)"}
SEEDS = [42, 123, 456, 789, 1024]
THR = 0.5
BASELINE_F1 = 0.8795  # TCN 5-seed prob_mean F1 (cross-arch v1)


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


def evaluate_view(view):
    per_seed = {}
    y_true_ref = None
    probs_dict = {}
    for s in SEEDS:
        path = os.path.join(BASE, f"predictions_test_tcn_{view}_s{s}.csv")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        y_true = arr[:, 0].astype(int); probs = arr[:, 1].astype(np.float32)
        if y_true_ref is None: y_true_ref = y_true
        probs_dict[s] = probs
        per_seed[str(s)] = {**four_metrics(y_true, (probs >= THR).astype(int)),
                            "CM_2x2": confusion_2x2(y_true, (probs >= THR).astype(int))}

    seeds_avail = sorted(probs_dict.keys())
    probs_matrix = np.stack([probs_dict[s] for s in seeds_avail], axis=0)
    ens_probs = probs_matrix.mean(axis=0)
    ens_pred = (ens_probs >= THR).astype(int)
    ens_metrics = {**four_metrics(y_true_ref, ens_pred),
                   "CM_2x2": confusion_2x2(y_true_ref, ens_pred)}

    print(f"\n=== {VIEW_DISPLAY[view]} ===")
    print(f"  5-seed prob_mean: Acc={ens_metrics['Accuracy']:.4f}  P={ens_metrics['Precision']:.4f}  "
          f"R={ens_metrics['Recall']:.4f}  F1={ens_metrics['F1-Score']:.4f}")
    for s in SEEDS:
        if s in probs_dict:
            m = per_seed[str(s)]
            print(f"  seed {s:>4}: F1={m['F1-Score']:.4f}")

    return {"view": view, "display_name": VIEW_DISPLAY[view], "n_seeds": len(seeds_avail),
            "test_n": int(len(y_true_ref)), "threshold": THR, "per_seed": per_seed,
            "ensemble_prob_mean": ens_metrics}


def write_outputs(results):
    json_path = os.path.join(BASE, "swa_ema_5seed_3view.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"task": "TCN + SWA + EMA 3-view × 5-seed × 4-metric",
                   "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
                   "threshold": THR, "baseline_f1": BASELINE_F1, "views": results}, f, indent=2)

    csv_path = os.path.join(BASE, "swa_ema_5seed_per_view.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["view", "seed", "Accuracy", "Precision", "Recall", "F1-Score"])
        for r in results:
            for s_str, m in r["per_seed"].items():
                w.writerow([r["view"], int(s_str), m["Accuracy"], m["Precision"],
                            m["Recall"], m["F1-Score"]])

    md_path = os.path.join(BASE, "SWA_EMA_LEADERBOARD.md")
    lines = ["# TCN + SWA + EMA 3-View Leaderboard\n",
             "**生成时间**: 2026-09-16  ",
             "**基线**: TCN (no SE) 5-seed prob_mean F1=0.8795  ",
             "**目标**: 任一 view F1 ≥ 0.88\n",
             "## 1. 5-Seed 概率平均集成（3 views × 4 metrics）\n",
             "| Rank | View | Accuracy | Precision | Recall | F1-Score | vs baseline |",
             "|------|------|---------:|----------:|-------:|---------:|-----------:|"]
    sorted_results = sorted(results, key=lambda r: -r["ensemble_prob_mean"]["F1-Score"])
    for i, r in enumerate(sorted_results, 1):
        m = r["ensemble_prob_mean"]
        diff = (m["F1-Score"] - BASELINE_F1) * 100
        marker = "🎯" if m["F1-Score"] >= 0.88 else ""
        lines.append(f"| {i} | {r['display_name']} | {m['Accuracy']:.4f} | "
                     f"{m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1-Score']:.4f} | "
                     f"{diff:+.2f}% {marker} |")

    lines.append("\n## 2. Per-Seed F1 详情\n")
    for r in sorted_results:
        lines.append(f"\n### {r['display_name']}\n")
        lines.append("| Seed | Accuracy | Precision | Recall | F1-Score |")
        lines.append("|-----:|---------:|----------:|-------:|---------:|")
        for s_str, m in r["per_seed"].items():
            lines.append(f"| {s_str} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | "
                         f"{m['Recall']:.4f} | {m['F1-Score']:.4f} |")
        cm = r["ensemble_prob_mean"]["CM_2x2"]
        lines.append(f"\n**5-seed 集成混淆矩阵** (thr={THR}):  ")
        lines.append(f"```\n              预测 Normal    预测 Attack\n"
                     f"实际 Normal      {cm[0][0]:>5}        {cm[0][1]:>5}\n"
                     f"实际 Attack      {cm[1][0]:>5}        {cm[1][1]:>5}\n```")

    lines.append("\n## 3. 关键发现\n")
    best = sorted_results[0]
    improvement = (best["ensemble_prob_mean"]["F1-Score"] - BASELINE_F1) * 100
    lines.append(f"1. **最佳 view**: {best['display_name']} — 5-seed 集成 F1 = {best['ensemble_prob_mean']['F1-Score']:.4f}")
    lines.append(f"2. **vs TCN 基线** (F1={BASELINE_F1}): {improvement:+.2f}%")
    target_met = best["ensemble_prob_mean"]["F1-Score"] >= 0.88
    lines.append(f"3. **目标达成**: {'✅ 是' if target_met else '❌ 否'} (F1 ≥ 0.88)")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[saved] {json_path}\n[saved] {csv_path}\n[saved] {md_path}")


def main():
    print("=== TCN + SWA + EMA 3-View × 5-Seed × 4-Metric Evaluation ===\n")
    results = []
    for view in VIEWS:
        results.append(evaluate_view(view))
    write_outputs(results)


if __name__ == "__main__":
    main()
```

**Step 3.2: Run evaluation**

```bash
cd "D:\workspace\claude\Issue\Issue" && python evaluate_swa_ema_3view.py 2>&1 | tee logs_evaluate_swa_ema.txt
```

Expected output: 3 view sections, each showing 5-seed prob_mean metrics + per-seed breakdown.

**Step 3.3: Verify outputs exist**

```bash
cd "D:\workspace\claude\Issue\Issue" && ls -la swa_ema_5seed_3view.json SWA_EMA_LEADERBOARD.md swa_ema_5seed_per_view.csv
```
Expected: 3 files, all >1 KB.

**Step 3.4: Verify CSV has 15 rows + header**

```bash
cd "D:\workspace\claude\Issue\Issue" && wc -l swa_ema_5seed_per_view.csv
```
Expected: `16 swa_ema_5seed_per_view.csv` (1 header + 15 data rows = 3 views × 5 seeds)

**Step 3.5: Commit**

```bash
cd "D:\workspace\claude\Issue\Issue"
git add evaluate_swa_ema_3view.py swa_ema_5seed_3view.json SWA_EMA_LEADERBOARD.md swa_ema_5seed_per_view.csv
git commit -m "results: TCN + SWA + EMA 3-view leaderboard"
```

---

### Task 4: Update daily log + verify success

**Files:**
- Modify: `DAILY_LOG_2026-09-16.md` (append SWA/EMA section)

**Step 4.1: Check results against success criteria**

```bash
cd "D:\workspace\claude\Issue\Issue" && python -c "
import json
with open('swa_ema_5seed_3view.json') as f:
    d = json.load(f)
print('View results (5-seed prob_mean F1):')
for v in d['views']:
    f1 = v['ensemble_prob_mean']['F1-Score']
    print(f'  {v[\"display_name\"]:50s}  F1={f1:.4f}  target={\"✓\" if f1 >= 0.88 else \"✗\"}')
best_f1 = max(v['ensemble_prob_mean']['F1-Score'] for v in d['views'])
print(f'\nBest F1: {best_f1:.4f}  vs baseline 0.8795: {(best_f1-0.8795)*100:+.2f}%')
print('SUCCESS' if best_f1 >= 0.88 else 'FAIL — proceed to sub-project #3 anyway')
"
```

**Step 4.2: Append SWA/EMA section to daily log**

Open `DAILY_LOG_2026-09-16.md` and append at end:

```markdown

---

## 🚀 Phase 3 — TCN + SWA + EMA（sub-project #2）

### 8. 方法（commit 见 Task 1+3）

3 个 view × 5 seeds × 4 指标：

| View | 实现 | 5-seed F1 vs baseline (0.8795) |
|------|------|--------------------------------:|
| raw (best state) | early-stopped by val F1m | X.XXXX (+X%) |
| SWA | epoch 10-20 state_dict 平均 + BN re-update | X.XXXX (+X%) |
| EMA | α=0.999 整个训练轨迹 | X.XXXX (+X%) |

### 9. 关键发现

- **最佳 view**: [auto-fill from script output]
- **目标达成**: ✅/❌ F1 ≥ 0.88
- **SWA 价值**: [interpret]
- **EMA 价值**: [interpret]
- **下一步: sub-project #3 (data augmentation)**
```

**Step 4.3: Update memory file**

Append to `C:\Users\17977\.claude\projects\D--workspace-claude-Issue\memory\MEMORY.md`:

```
- [SCADA TCN SWA/EMA results](scada-tcn-swa-ema-results.md) — [auto-fill: best view F1=..., SWA effect +/-X%, EMA effect +/-X%]
```

If the best view F1 < 0.88, document the failure mode in the memory file instead.

**Step 4.4: Commit daily log**

```bash
cd "D:\workspace\claude\Issue\Issue"
git add DAILY_LOG_2026-09-16.md
git commit -m "docs: 2026-09-16 daily log + memory update (SWA/EMA complete)"
```

---

## Self-Review

1. **Spec coverage**:
   - Spec §3.1 (TCN config) → Task 1 imports TCNClassifier, uses identical hyperparams ✓
   - Spec §3.2 (SWA: start epoch 10, every epoch, BN re-update) → Task 1 train loop + `bn_re_update()` ✓
   - Spec §3.3 (EMA: α=0.999, every epoch) → Task 1 EMA update block ✓
   - Spec §3.4 (3-view evaluation, 5-seed ensemble, 4 metrics) → Task 3 evaluation script ✓
   - Spec §4 (output files: script, 3×5 CSVs, meta JSON, leaderboard JSON/MD/CSV, log) → Tasks 1-3 ✓
   - Spec §6 (error handling: fallback SWA, BN re-update) → Task 1 ✓
   - Spec §7 (success: any view F1 ≥ 0.88) → Task 4.1 verifies ✓

2. **Placeholder scan**: No TBD/TODO. All code blocks complete.

3. **Type consistency**:
   - `train_one_seed_swa_ema(seed)` returns dict with `seed, n_params, best_epoch, n_swa_snapshots, swa_warning, train_time_s, raw, swa, ema` (each containing `test_*` metrics) — used consistently in Task 1 + Task 2.4 + Task 4.
   - `evaluate_view(view)` returns dict with `view, display_name, n_seeds, test_n, threshold, per_seed, ensemble_prob_mean` — used in Task 3 write_outputs.
   - Tag prefixes: `tcn_swa_ema` (meta JSON), `tcn_{raw,swa,ema}` (per-view CSVs).