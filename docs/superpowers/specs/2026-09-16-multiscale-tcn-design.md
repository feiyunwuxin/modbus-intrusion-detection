# TCN + Multi-Scale Kernels — Design

> **设计批准日期**: 2026-09-16
> **作者**: Claude Code (brainstorming session)
> **关联**: SCADA cross-arch 扩展实验 sub-project #4
> **基线**: TCN (no SE) 5-seed F1=0.8795
> **Stacking 基线**: M3@0.390 F1=0.8904
> **目标**: Multi-scale TCN 5-seed F1 ≥ 0.88

---

## 1. 背景

Sub-project #2 (SWA/EMA) 与 #3 (Augmentation) 都已验证**单一架构 TCN 修改**（权重空间 + 输入空间）无法在已收敛的小模型上带来收益。本次尝试**架构空间的最后一搏**——multi-scale 卷积核，用 kernel 多样性捕捉不同时序粒度的模式。

**参考**: Inception/GoogLeNet 在图像分类中的成功；TCN 文献中 Multi-Scale TCN 也常用于动作识别、时序预测。

**期望**: 如果 multi-scale TCN 单架构能达 0.88，则可独立部署（无 ensemble 推理开销）；否则正式关闭 TCN 单架构 tuning 方向。

---

## 2. 数据契约

### 2.1 输入（已存在）

| 文件 | 内容 |
|---|---|
| `X_train_binary_v2_scada_window16.npy` 等 | 23-dim × window=16 数据 |
| `train_tcn_23dim_w16_5seed.py` | TCN baseline 源码 |
| `TCNClassifier` | baseline 架构，可参照但本 sub-project 用新模型类 |

数据统计（不变）：
- train N=12014 (48.7% Normal / 51.3% Attack)
- val N=1716, test N=3432

### 2.2 输出（新增）

| 文件 | 内容 |
|---|---|
| `train_tcn_multiscale_5seed.py` | 自包含训练 + 评估脚本 |
| `predictions_test_tcn_ms_s{seed}.csv` × 5 | per-seed predictions |
| `processed_meta_tcn_multiscale.json` | per-seed metrics (5 seeds) |
| `multiscale_5seed_4metric.json` | ensemble leaderboard data |
| `MULTISCALE_LEADERBOARD.md` | human-readable leaderboard |
| `multiscale_5seed_per_seed.csv` | 5 行 flat table (5 seeds × 4 metrics) |
| `logs_train_multiscale.txt` | training log |

---

## 3. 方法

### 3.1 Multi-Scale TCNBlock 架构

每个 block 内部:
```
input (B, 32, 16)
  ├─ Conv1d(32→32, k=1, padding=0) → BN → ReLU → Dropout
  ├─ Conv1d(32→32, k=3, padding=1) → BN → ReLU → Dropout
  ├─ Conv1d(32→32, k=5, padding=2) → BN → ReLU → Dropout
  └─ Conv1d(32→32, k=7, padding=3) → BN → ReLU → Dropout
Concat → Conv1d(128→32, k=1)  ← 1x1 conv to reduce channels
Add input (residual) → output (B, 32, 16)
```

### 3.2 完整模型 (MultiScaleTCN)

- 3 MultiScaleTCNBlocks × dilations=[1, 2, 4]
- 末层: AdaptiveAvgPool1d(1) → Flatten → Linear(32 → 1)
- Dropout = 0.1
- 参数量预估: ~30000-40000 (原 TCN baseline 20001)

### 3.3 训练超参数（与 baseline 一致）

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam lr=4e-3, wd=1e-5 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 20 |
| Batch | 64 |
| Pos weight | n_neg / n_pos |
| Early stopping patience | 5 (on val F1m) |
| Grad clip | 0.5 |
| Loss | BCEWithLogitsLoss |
| Seeds | [42, 123, 456, 789, 1024] |

### 3.4 评估

- 4 指标: Accuracy / Precision / Recall / F1-Score (thr=0.5, Binary/Attack=positive)
- 5-seed prob_mean ensemble
- 对比 reference: TCN baseline 0.8795; stacking M3@0.390 0.8904

---

## 4. 实施计划

### Step 1: 写 `train_tcn_multiscale_5seed.py`

- 定义 `MultiScaleTCNBlock` (nn.Module)
- 定义 `MultiScaleTCN` (3 blocks + head)
- 实现 `train_one_seed_ms(seed)`:
  - Standard training loop + early stopping + grad clip + scheduler
  - 预测保存到 `predictions_test_tcn_ms_s{seed}.csv`
- Main: 5 seeds × 1 model = 5 trainings (~3 min total)

### Step 2: 评估 + leaderboard

- Load 5 prediction CSVs
- 5-seed prob_mean ensemble per setting (just 1 setting: multi-scale)
- 4 metrics per ensemble
- Save JSON + MD + CSV outputs

### Step 3: Commit + update daily log

- 1 training script
- 3 output files (JSON/MD/CSV)
- 1 log file
- Daily log addendum

---

## 5. 文件变更清单

**新增**:
- `train_tcn_multiscale_5seed.py`
- `processed_meta_tcn_multiscale.json`
- `multiscale_5seed_4metric.json`
- `MULTISCALE_LEADERBOARD.md`
- `multiscale_5seed_per_seed.csv`
- `logs_train_multiscale.txt`

**修改**:
- `DAILY_LOG_2026-09-16.md` (新增 Phase 8 章节)

**未入库**（.gitignore 排除）:
- `model_*.pt` × 5
- `predictions_test_*.csv` × 5

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Multi-scale 参数膨胀过拟合 | 参数量从 20K → ~35K，仍远小于 12014 train samples |
| 4 kernels 训练慢 | 每个 block 多 4 倍 conv；总时长 ~3 min（vs TCN baseline ~3.3 min），可接受 |
| 1x1 conv 退化为 identity | residual + 1x1 给模型"走原路径"的选项；不强制学 multi-scale |
| Padding + kernel=7 在 window=16 可能饱和 | window=16, padding=3, output_len=16+6=22 → Conv1d 自动 trim，匹配原维度 |
| 与 baseline hyperparam 不匹配 | 完全复用 baseline lr/wd/scheduler/early stop，便于公平对比 |

---

## 7. 成功标准

- [ ] 5 seeds 全部跑通
- [ ] 输出 json/md/csv 三件套
- [ ] 5-seed prob_mean F1 ≥ 0.88（vs baseline 0.8795）
- [ ] 参数量 report 在 30K-50K 之间（vs baseline 20K）

---

## 8. 后续

- **如果成功 (F1 ≥ 0.88)**: Multi-scale TCN 可独立部署，绕过 stacking 推理开销
- **如果失败 (F1 < 0.88)**: 正式宣布"TCN 单架构 tuning 已穷尽"，Ship Stack11 M3@0.390 (F1=0.8904) 为唯一 production baseline