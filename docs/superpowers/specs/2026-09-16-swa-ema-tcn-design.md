# SWA + EMA Regularization for TCN Baseline — Design

> **设计批准日期**: 2026-09-16
> **作者**: Claude Code (brainstorming session)
> **关联**: SCADA cross-arch 扩展实验 sub-project #2
> **基线**: TCN (no SE) 5-seed F1=0.8795
> **目标**: F1 ≥ 0.88 (≥ +0.05% vs baseline)

---

## 1. 背景与目标

TCN baseline 在 5-seed 集成上达到 F1=0.8795 (prob_mean)。Stacking sub-project (#1) 通过 11-arch × 5-seed × LR meta-learner 达到 F1=0.8861。本 sub-project 探索**单架构 + weight averaging 正则化**能否逼近 stacking 效果：
- **SWA (Stochastic Weight Averaging)**: Averaging 后半段训练权重 + BN re-update
- **EMA (Exponential Moving Average)**: 每 epoch 平滑更新 EMA 状态

**成功标准**: 任一方法 5-seed prob_mean F1 ≥ 0.88 (≥ +0.05% vs 0.8795)

---

## 2. 数据契约

### 2.1 输入（已存在）

| 文件 | 内容 |
|---|---|
| `X_train_binary_v2_scada_window16.npy` 等 | 23-dim × window=16 数据（已有） |
| `train_tcn_23dim_w16_5seed.py` | TCNClassifier 源码可复用 |

### 2.2 输出（新增）

| 文件 | 内容 |
|---|---|
| `train_tcn_swa_ema_5seed.py` | 自包含训练 + 评估脚本 |
| `predictions_test_tcn_raw_s{seed}.csv` × 5 | raw best (epoch early-stopped) per seed |
| `predictions_test_tcn_swa_s{seed}.csv` × 5 | SWA averaged weights per seed |
| `predictions_test_tcn_ema_s{seed}.csv` × 5 | EMA smoothed weights per seed |
| `processed_meta_tcn_swa_ema.json` | full per-seed metrics (3 views each) |
| `swa_ema_5seed_3view.json` | ensemble leaderboard data |
| `SWA_EMA_LEADERBOARD.md` | human-readable leaderboard |
| `swa_ema_5seed_per_view.csv` | 15-row flat table (3 views × 5 seeds) |
| `logs_train_swa_ema.txt` | training log |

---

## 3. 方法

### 3.1 TCN Baseline 配置（与 cross-arch 一致）

| Hyperparameter | Value |
|---|---|
| Architecture | 3 TCN blocks × ch=32, kernel=3, dilations=[1,2,4], dropout=0.1 |
| Optimizer | Adam lr=4e-3, wd=1e-5 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 20 |
| Batch | 64 |
| Pos weight | n_neg / n_pos |
| Early stopping patience | 5 (on val F1m) |
| Grad clip | 0.5 |
| Loss | BCEWithLogitsLoss |
| Seeds | [42, 123, 456, 789, 1024] |

### 3.2 SWA 配置

| 项 | 值 | 理由 |
|---|---|---|
| Start epoch | 10 (训练总 20 epoch 的 50%) | 短训练场景提早开始 averaging |
| Snapshot frequency | 每 epoch | 简化实现，与 SWA 原文一致 |
| BN re-update | train data 1 forward pass (no grad) | SWA 原文要求，BN 对均值敏感 |

### 3.3 EMA 配置

| 项 | 值 | 理由 |
|---|---|---|
| Decay α | 0.999 | 长记忆；20 epochs × ~187 steps/epoch ≈ 3740 步，α=0.999 半衰期 ~693 步 |
| Initial state | epoch 0 = model.state_dict() | 文档化 |
| Update | 每 epoch 训练后 | 与 SWA snapshot 时机一致 |

### 3.4 评估

| View | 模型状态 | 5-seed ensemble |
|---|---|---|
| V1 raw | best_state (early-stopped by val F1m) | prob_mean across 5 seeds |
| V2 SWA | averaged snapshots + BN re-updated | prob_mean across 5 seeds |
| V3 EMA | ema_state (no BN re-update needed) | prob_mean across 5 seeds |

4 指标: Accuracy / Precision / Recall / F1-Score (thr=0.5, Binary / Attack=positive)

---

## 4. 实施计划

### Step 1: 写 `train_tcn_swa_ema_5seed.py` (~180 行)

- Import TCNClassifier from `train_tcn_23dim_w16_5seed`
- Implement `train_one_seed_swa_ema(seed)`:
  1. Set seeds
  2. Load data (from `_common_train.load_data_23dim_w16`)
  3. Init model, optimizer, scheduler, loss (same as baseline)
  4. Init `swa_snapshots = []`, `ema_state = None`
  5. Loop 20 epochs:
     - Train step
     - Val eval → update best_state
     - If epoch ≥ 10: `swa_snapshots.append({k: v.clone() for k, v in model.state_dict().items()})`
     - If ema_state is None: `ema_state = {k: v.clone() for k, v in model.state_dict().items()}`
     - Else: `ema_state[k] = α·current + (1-α)·ema_state[k] for each k`
     - Scheduler step + early stop check
  6. End of training — 3 evaluations:
     - **raw**: load best_state → predict_proba on test
     - **SWA**: avg snapshots → load_state_dict → BN re-update (model.train() + 1 forward pass over train data) → model.eval() → predict_proba on test
     - **EMA**: load ema_state → predict_proba on test
  7. Save 3 prediction CSVs + meta dict

### Step 2: 5-seed orchestration + ensemble evaluation

- Loop 5 seeds, call `train_one_seed_swa_ema(seed)`
- After loop: 3-view ensemble = avg probs across 5 seeds per view
- Compute 4 metrics for each view
- Save:
  - `processed_meta_tcn_swa_ema.json`
  - `swa_ema_5seed_3view.json`
  - `SWA_EMA_LEADERBOARD.md`
  - `swa_ema_5seed_per_view.csv` (15 rows: view × seed)

### Step 3: Commit + update daily log (~5 min)

- 1 new training script
- 3 × 5 = 15 prediction CSVs (gitignored except via log)
- 3 output files (json/md/csv)
- 1 log file
- Daily log addendum

---

## 5. 文件变更清单

**新增**:
- `train_tcn_swa_ema_5seed.py`
- `processed_meta_tcn_swa_ema.json`
- `swa_ema_5seed_3view.json`
- `SWA_EMA_LEADERBOARD.md`
- `swa_ema_5seed_per_view.csv`
- `logs_train_swa_ema.txt`

**修改**:
- `DAILY_LOG_2026-09-16.md` (新增 SWA/EMA 章节)

**未入库**（.gitignore 排除）:
- `model_*.pt` × 5
- `predictions_test_*.csv` × 15

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| SWA snapshots 为空（early stop < epoch 10） | Fall back to best_state + log warning |
| BN re-update 改变 BN running stats 但损坏预测 | 先 model.train() 跑 1 pass，再 model.eval()；与 SWA 原文协议一致 |
| EMA α=0.999 对短训练几乎不动 | 验证：20 epoch 后 ema_state 应与 best_state 不同但接近 |
| 单 seed F1 波动 ±1% | 5-seed prob_mean 集成抑制噪声；如失败，报告 std |
| BN re-update 增加训练时间 | 仅 1 pass train data，~2s 开销 |

---

## 7. 成功标准

- [ ] 3 个 view (raw / SWA / EMA) 全部跑通 5 seeds
- [ ] 输出 json/md/csv 三件套
- [ ] SWA 或 EMA 5-seed prob_mean F1 ≥ 0.88
- [ ] SWA 比 raw baseline 提升 OR EMA 比 raw baseline 提升（任一）

---

## 8. 后续

- 如果 SWA/EMA 成功：继续 sub-project #3 (data augmentation)
- 如果失败：回退到 raw baseline，分析为何 averaging 未带来收益