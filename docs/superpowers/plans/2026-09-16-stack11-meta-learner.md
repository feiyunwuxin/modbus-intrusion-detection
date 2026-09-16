# Stack11: 跨架构 11-arch × 5-seed × 3-meta-learner 集成（设计 + 计划）

> **设计批准日期**: 2026-09-16
> **作者**: Claude Code (brainstorming session)
> **关联决策**: stacking sub-project #1 (原 4 个扩展方向之一)

---

## 1. 目标

在已有 9 架构（TCN/TCN+SE/BiLSTM/BiGRU/CNN-LSTM + Pure CNN/BiLSTM-h128/BiGRU-h128/CNN-LSTM-ch256）+ LGB + RF 上做 11 架构 5-seed stacking，对比 3 种 meta-learner 方法。

**基线**: TCN (no SE) 5-seed F1=0.8795
**目标**: F1 ≥ 0.88（突破基线 ≥ 0.5%）

---

## 2. 数据契约

### 2.1 输入（部分已存在）

| 文件 | 用途 | 状态 |
|---|---|---|
| `predictions_test_*_s{seed}.csv` × 45 | 9 深度架构 5-seed 预测 | ✅ 已存在（今日 cross-arch v1+v2）|
| `predictions_test_lgb_23dim_w16_s{seed}.csv` × 5 | LGB 5-seed 预测 | ❌ 待训练 |
| `predictions_test_rf_23dim_w16_s{seed}.csv` × 5 | RF 5-seed 预测 | ❌ 待训练 |

格式: `y_true,prob_attack` CSV，每行一个测试样本 (N=3432)

### 2.2 输出

| 文件 | 内容 |
|---|---|
| `stack_11arch_3method.json` | 3 方法 × 4 指标 + CM |
| `STACK_11ARCH_LEADERBOARD.md` | 人类可读对比报告 |
| `stack_11arch_per_seed.csv` | 11×5=55 行扁平表 |

---

## 3. 训练方法

| 项 | 值 |
|---|---|
| Level-1 输入 | 55 个 per-seed 概率 (列) × 3432 样本 |
| Level-2 训练数据 | **val 集 (1716 样本)** — 用于 meta-learner 训练（参照 V1 协议）|
| Level-2 测试 | test 集 (3432 样本) — 最终评估 |

---

## 4. Meta-Learner 3 方法对比

| 方法 | 实现 | 优缺点 |
|---|---|---|
| **M1 Simple average** | 55 列均值 → thr 0.5 | baseline，无学习成本 |
| **M2 Weighted average** | SLSQP on val, weights≥0, Σw=1 | 保留可解释性，可看出哪个 arch 重要 |
| **M3 LR stacking** | L2 LR, C via 5-fold CV on val | 经典 stacking，能学 arch 间交互 |

**输出 4 指标**: Accuracy / Precision / Recall / F1-Score (thr=0.5)

---

## 5. 实施计划

### Phase 1 — LGB + RF 训练（~15 min）

**Step 1.1**: 写 `train_lgb_23dim_w16_5seed.py`
- 输入: `X_train_binary_v2_scada_window16.npy` (N, 16, 28) → 选 23 dim → flatten (N, 23*16=368)
- 模型: `lightgbm.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6)`
- 5 seeds: [42, 123, 456, 789, 1024]
- 输出: `model_lgb_23dim_w16_s{seed}.pkl`, `predictions_test_lgb_23dim_w16_s{seed}.csv`

**Step 1.2**: 写 `train_rf_23dim_w16_5seed.py`
- 同上但 `RandomForestClassifier(n_estimators=200, max_depth=12)`
- 输出: `model_rf_23dim_w16_s{seed}.pkl`, `predictions_test_rf_23dim_w16_s{seed}.csv`

**Step 1.3**: 训练 5 seeds of LGB + 5 seeds of RF (background)
- 预估: ~15 min 总

### Phase 2 — 更新 evaluate 脚本（~5 min）

**Step 2.1**: `evaluate_5arch_4metrics.py` 的 ARCHS 列表加 LGB + RF（11 archs）

**Step 2.2**: 跑一次 evaluate 验证 11 archs 都跑通

### Phase 3 — Stacking 脚本（~20 min）

**Step 3.1**: 写 `stack_meta_learner.py`
- 加载 55 个 per-seed 预测 CSV
- 对齐 y_true（assert 一致）
- 实现 3 种 meta-learner
- 输出 leaderboard JSON + MD + CSV

**Step 3.2**: 验证输出格式

### Phase 4 — Commit（~5 min）

- 4 个新训练脚本 + 1 个 stacking 脚本
- LGB/RF outputs (CSVs, 不含 .pkl — gitignored)
- 3 个 leaderboard 输出
- Daily log 更新

---

## 6. 文件清单

**新增**:
- `train_lgb_23dim_w16_5seed.py`
- `train_rf_23dim_w16_5seed.py`
- `stack_meta_learner.py`
- `stack_11arch_3method.json`
- `STACK_11ARCH_LEADERBOARD.md`
- `stack_11arch_per_seed.csv`
- `logs_train_lgb.txt` / `logs_train_rf.txt` / `logs_stack.txt`

**修改**:
- `evaluate_5arch_4metrics.py` (ARCHS 加 LGB+RF)
- `DAILY_LOG_2026-09-16.md` (新增 stacking 章节)

---

## 7. 成功标准

- [ ] 3 种 meta-learner 全部跑通
- [ ] 输出 JSON/MD/CSV 三件套
- [ ] M3 (LR stacking) F1 ≥ 0.88 (vs 基线 0.8795)
- [ ] M2 weighted avg 显示 arch 权重分布合理

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| LGB/RF 训练慢 | sklearn 单线程够用，10-15 min 内必完 |
| Val 集小（1716）做 LR 训练 | 55 特征 > 1716 样本 — 有过拟合风险。用 L2 + CV 缓解；如失败则改用 ElasticNet 或降到 per-arch 5 特征 |
| 55 特征 vs 3432 test 样本 | 测试集远大于特征数，meta-learner 应用阶段无问题 |

## 9. 后续

如果 stacking 成功（≥ 0.88 F1），继续：
- Sub-project #2: SWA/EMA for TCN
- Sub-project #3: Data augmentation
- Sub-project #4: Multi-scale TCN

如果 stacking 失败，回退方案：用 top-5 架构（按 F1）做 stacking，减小特征维度。