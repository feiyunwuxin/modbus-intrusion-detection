# TCN + Data Augmentation — Design

> **设计批准日期**: 2026-09-16
> **作者**: Claude Code (brainstorming session)
> **关联**: SCADA cross-arch 扩展实验 sub-project #3
> **基线**: TCN (no SE) 5-seed F1=0.8795
> **Stacking M3 基线**: F1=0.8861
> **目标**: 任一 augmentation 设置 5-seed F1 ≥ 0.88

---

## 1. 背景

TCN baseline 已通过 ReduceLROnPlateau + early stopping 充分收敛。Sub-project #2 (SWA/EMA) 验证 weight averaging 无法在已收敛 + 短训练 + 小模型场景提供增益（sWAF1=0.8733, EMA F1=0.8768）。Stacking sub-project (#1) 通过 11-arch 集成达到 F1=0.8861。

本 sub-project 探索 **input-space augmentation** 作为单架构性能提升的第三条路：
- **Gaussian noise**: per-feature normalized σ=0.05×feature_std
- **Feature mask**: 随机零 10% 特征列
- **Time mask**: 随机零 10% 时间步
- **Combined**: 三者组合

**目标**: 任一设置 5-seed F1 ≥ 0.88

---

## 2. 数据契约

### 2.1 输入（已存在）

| 文件 | 内容 |
|---|---|
| `X_train_binary_v2_scada_window16.npy` 等 | 23-dim × window=16 数据 |
| `train_tcn_23dim_w16_5seed.py` | TCNClassifier 源码可复用 |

数据统计：
- train N=12014 (48.7% Normal / 51.3% Attack)
- val N=1716, test N=3432
- 特征 std 范围: 0.02 – 4.23 (median 0.43)
- 类平衡良好，无需过采样

### 2.2 输出（新增）

| 文件 | 内容 |
|---|---|
| `train_tcn_augmentation_5seed.py` | 自包含训练 + 评估脚本 |
| `predictions_test_tcn_aug_{none,noise,feat_mask,time_mask,combined}_s{seed}.csv` × 5 × 4 | per-seed predictions (20 files total) |
| `processed_meta_tcn_augmentation.json` | per-seed metrics (4 settings × 5 seeds) |
| `augmentation_5seed_4setting.json` | ensemble leaderboard data |
| `AUGMENTATION_LEADERBOARD.md` | human-readable leaderboard |
| `augmentation_5seed_per_setting.csv` | 4 × 5 = 20 row flat table |
| `logs_train_augmentation.txt` | training log |

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

### 3.2 Augmentation 设置（5 种）

| Method | 实现 |
|---|---|
| `none` | 不做 augmentation (baseline, 用于对照) |
| `noise` | `x + 0.05 * feature_std * N(0,1)` |
| `feat_mask` | 随机零 10% 特征列 (per sample) |
| `time_mask` | 随机零 10% 时间步 (per sample) |
| `combined` | noise → feat_mask → time_mask 顺序应用 |

**关键约束**:
- 只在 train 阶段（`if model.training:`）应用
- val / test 永远保持原始
- Per-feature std 在 epoch 0 一次性从 train data 计算（不每 batch 重算）

### 3.3 评估

- 4 指标: Accuracy / Precision / Recall / F1-Score (thr=0.5, Binary / Attack=positive)
- 5-seed prob_mean ensemble per setting
- 跨 setting 比较: report delta vs baseline (`none`)

---

## 4. 实施计划

### Step 1: 写 `train_tcn_augmentation_5seed.py` (~200 行)

- Import TCNClassifier from `train_tcn_23dim_w16_5seed.py`
- Implement `add_gaussian_noise(x, feature_std, sigma=0.05)` — pure function
- Implement `feature_mask(x, mask_rate=0.1)` — pure function
- Implement `time_mask(x, mask_rate=0.1)` — pure function
- Implement `apply_augmentation(x, method, feature_std)` — dispatcher
- Compute per-feature std from train data once (precomputed array of shape (23,))
- Implement `train_one_seed_aug(seed, method, feature_std)`:
  - Standard training loop, but augment `xb` BEFORE forward pass IF `model.training`
  - Save predictions to `predictions_test_tcn_aug_{method}_s{seed}.csv`
- Main loop: for each of 5 methods × 5 seeds = 25 trainings

### Step 2: Evaluation + leaderboard

- Load 20 prediction CSVs (4 methods × 5 seeds)
- 5-seed prob_mean ensemble per method
- 4 metrics per method
- Save JSON + MD + CSV outputs

### Step 3: Commit + update daily log

- 1 training script
- 3 output files (JSON/MD/CSV)
- 1 log file
- Daily log addendum

---

## 5. 文件变更清单

**新增**:
- `train_tcn_augmentation_5seed.py`
- `processed_meta_tcn_augmentation.json`
- `augmentation_5seed_4setting.json`
- `AUGMENTATION_LEADERBOARD.md`
- `augmentation_5seed_per_setting.csv`
- `logs_train_augmentation.txt`

**修改**:
- `DAILY_LOG_2026-09-16.md` (新增 augmentation 章节)

**未入库**（.gitignore 排除）:
- `model_*.pt` × 20
- `predictions_test_*.csv` × 20

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Augmentation 过激破坏信号 | mask_rate=0.1 + σ=0.05 保守起步；先 single-seed smoke test |
| 4 settings × 5 seeds = 20 trainings 耗时 | ~30s × 20 = 10 min 总耗时，可接受 |
| `combined` 同时三个 augmentation 可能负效应 | ablation 设计允许隔离单一贡献 |
| Feature std 计算错误（用 train 时 leak val/test） | 仅从 train 计算 |
| Augmentation applied to test by mistake | `if model.training:` guard + 单测 |
| 训练脚本结构改变影响其他 baseline | 独立新脚本，不修改 `_common_train.py` |

---

## 7. 成功标准

- [ ] 5 settings × 5 seeds 全部跑通
- [ ] 输出 json/md/csv 三件套
- [ ] 任一 setting 5-seed prob_mean F1 ≥ 0.88
- [ ] Ablation 能识别各 augmentation 单独贡献

---

## 8. 后续

- 如果 augmentation 成功：继续 sub-project #4 (multi-scale TCN)
- 如果失败：分析是否 augmentation 与 early stopping 冲突；回退到 `none` + 部署 stacking M3 作为 production baseline