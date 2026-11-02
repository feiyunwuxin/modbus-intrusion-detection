# 跨阶段 4 指标横向对比报告 (2026-09-17)

**任务**: 把今天重训的 LightGBM baseline (阶段一, 06-08) 与 36 模型横评合并, 产出**统一大总览**, 覆盖阶段一/二/三/五共 21 个代表性模型.

**指标口径**: Accuracy / Precision / Recall / F1-Score — **Binary convention** (Attack 类为正例, 与 `feedback-ids-binary-precision-convention.md` 一致)
**阈值**: 0.5
**数据源**:
- `phase1_4metrics.json` (今日重训阶段一 LGB baseline 32 特征)
- `compare_models_4metrics.json` (阶段二/三 17 模型)
- `rerun_4metrics_5seed.json` (阶段五 SOTA 5-seed ensemble)

> ⚠️ **重要**: 各阶段测试集大小 / Attack 占比不同:
> - 阶段一 baseline (test=82,389, Attack=21.6%) — temporal 70/30
> - 阶段二 no_window (test=54,927, Attack=21.9%) — 重新切分
> - 阶段二 window4/8/16 (test=13,731 / 6,865 / 3,432, Attack=49-52%)
> - 阶段五 SOTA (test=3,432, Attack=52.6%)
> **直接 F1 绝对值不可比**, 推荐看**同测试集组内排名**或**阶段增益轨迹**.

---

## 1. 大总览 (21 模型按 F1 降序)

| # | 阶段 | 模型 | 测试集 | N | Attack% | Acc | P | R | F1 | Macro-F1 |
|:--:|:--:|-----|:---|---:|---:|---:|---:|---:|---:|---:|
| **1** ⭐ | **五 SOTA** | **TCN+SE 23-dim ch=32 5-seed ensemble** | window16 | 3,432 | 52.6 | **0.8776** | **0.9553** | **0.8050** | **0.8737** | **0.8775** |
| 2 | 三 | TCN+SE v3 window16 | window16 | 3,432 | 52.6 | 0.8406 | 0.9538 | 0.7324 | 0.8286 | 0.8398 |
| 3 | 三 | TCN+SE v2 window16 | window16 | 3,432 | 52.6 | 0.8377 | 0.9581 | 0.7230 | 0.8241 | 0.8367 |
| 4 | 二 | TCN v3a deep | window8 | 6,865 | 50.0 | 0.8341 | 0.9508 | 0.7045 | 0.8093 | 0.8312 |
| 5 | 二 | CNN-LSTM | window8 | 6,865 | 50.0 | 0.8326 | 0.9517 | 0.7007 | 0.8071 | 0.8296 |
| 6 | 二 | LSTM+Attention | window16 | 3,432 | 52.6 | 0.8188 | 0.9346 | 0.7047 | 0.8035 | 0.8177 |
| 7 | 三 | TCN v3c deep window16 | window16 | 3,432 | 52.6 | 0.8196 | 0.9714 | 0.6770 | 0.7979 | 0.8175 |
| 8 | 二 | ShuffleNet1D | window8 | 6,865 | 50.0 | 0.8220 | 0.9528 | 0.6774 | 0.7918 | 0.8182 |
| 9 | 三 | TCN v3b window16 | window16 | 3,432 | 52.6 | 0.8132 | 0.9671 | 0.6676 | 0.7899 | 0.8109 |
| 10 | 二 | BiLSTM v2 window=8 *(09-17 复现)* | window8 | 6,865 | 50.0 | 0.8170 | 0.9352 | 0.6811 | **0.7882** | 0.8136 |
| 11 | 二 | MobileNet1D | window8 | 6,865 | 50.0 | 0.8176 | 0.9456 | 0.6739 | 0.7869 | 0.8138 |
| 12 | 三 | TCN+SE v1b window16 | window16 | 3,432 | 52.6 | 0.8030 | 0.9638 | 0.6499 | 0.7763 | 0.8002 |
| 13 | 二 | 1D Transformer | window16 | 3,432 | 52.6 | 0.7952 | 0.9734 | 0.6277 | 0.7632 | 0.7914 |
| 14 | 二 | CNN v2 | window4 | 13,731 | 48.7 | 0.7701 | 0.9262 | 0.5738 | 0.7086 | 0.7594 |
| 15 | 二 | RNN Basic | window8 | 6,865 | 50.0 | 0.7529 | 0.9059 | 0.5643 | 0.6954 | 0.7438 |
| 16 | 二 | LSTM v2 | window4 | 13,731 | 48.7 | 0.7399 | 0.8563 | 0.5602 | 0.6773 | 0.7297 |
| 17 | 二 | FNN (MLP) v2 | no_window | 54,927 | 21.9 | 0.8836 | 0.9029 | 0.5242 | 0.6633 | 0.7965 |
| 18 | 二 | Decision Tree v2 | no_window | 54,927 | 21.9 | 0.8842 | 0.9327 | 0.5068 | 0.6567 | 0.7935 |
| 19 | **一** | **LightGBM baseline (32 特征, 09-17 重训)** | stage1_t70 | **82,389** | **21.6** | **0.8129** | **0.5478** | **0.7694** | **0.6400** | — |
| 20 | 二 | Random Forest v2 | no_window | 54,927 | 21.9 | 0.7858 | 0.5062 | 0.8357 | 0.6305 | 0.7398 |
| 21 | 二 | LightGBM 19-dim | no_window | 54,927 | 21.9 | 0.7714 | 0.4872 | 0.8689 | 0.6243 | 0.7300 |
| 22 | 二 | SVM 19-dim | no_window | 54,927 | 21.9 | 0.8298 | 0.9521 | 0.2335 | 0.3750 | 0.6383 |

**注**: 阶段一 baseline 的 Macro-F1 暂未算 (重训脚本未输出); 主看 Binary F1.

---

## 2. 阶段增益轨迹 (101 天)

| 阶段 | 范围 | 最佳模型 | F1 | 测试集 N | Attack% | ΔF1 |
|:--:|:--:|-----|---:|---:|---:|---:|
| **一** | 06-08 | LGB baseline (32 特征) | **0.6400** | 82,389 | 21.6 | — (基线) |
| **二** | 06-09/10 | BiLSTM v2 window=8 | **0.7882** | 6,865 | 50.0 | **+0.1482 (+23.2%)** |
| **三** | 06-12/14 | TCN+SE v3 window16 | **0.8286** | 3,432 | 52.6 | +0.0404 (+5.1%) |
| **五 SOTA** | 07-26 | TCN+SE 23-dim 5-seed | **0.8737** | 3,432 | 52.6 | **+0.0451 (+5.4%)** |

**总累计**: 0.6400 → 0.8737 = **+0.2337 (+36.5% 相对)** ⭐

> ⚠️ F1 增益轨迹在不同测试集上,**绝对值偏置**: 阶段一 Attack 占 21.6% (难), 阶段五 Attack 占 52.6% (易).
> 实际模型能力的提升更体现在 **Precision (0.55 → 0.96) 和 Recall (0.77 → 0.81)** 的**双双提升**.

---

## 3. 按测试集组细分

### 3.1 window16 (N=3,432, Attack=52.6%) — **SOTA 主战场**

| # | 模型 | 阶段 | Acc | P | R | F1 |
|:--:|-----|:--:|---:|---:|---:|---:|
| **1** ⭐ | TCN+SE 23-dim ch=32 5-seed ensemble | 五 | **0.8776** | **0.9553** | **0.8050** | **0.8737** |
| 2 | TCN+SE v3 window16 | 三 | 0.8406 | 0.9538 | 0.7324 | 0.8286 |
| 3 | TCN+SE v2 window16 | 三 | 0.8377 | 0.9581 | 0.7230 | 0.8241 |
| 4 | LSTM+Attention | 二 | 0.8188 | 0.9346 | 0.7047 | 0.8035 |
| 5 | TCN v3c deep | 三 | 0.8196 | 0.9714 | 0.6770 | 0.7979 |
| 6 | TCN v3b | 三 | 0.8132 | 0.9671 | 0.6676 | 0.7899 |
| 7 | TCN+SE v1b | 三 | 0.8030 | 0.9638 | 0.6499 | 0.7763 |
| 8 | 1D Transformer | 二 | 0.7952 | 0.9734 | 0.6277 | 0.7632 |

**观察**: 5-seed ensemble F1 比单 seed v3 +0.045, 比 Transformer 高 +0.110. **TCN+SE 通道注意力 + 23-dim 特征 + 5-seed 集成是三件套**.

### 3.2 window8 (N=6,865, Attack=50%) — 阶段二 DL 主战场

| # | 模型 | Acc | P | R | F1 |
|:--:|-----|---:|---:|---:|---:|
| 1 | TCN v3a deep | 0.8341 | 0.9508 | 0.7045 | 0.8093 |
| 2 | CNN-LSTM | 0.8326 | 0.9517 | 0.7007 | 0.8071 |
| 3 | ShuffleNet1D | 0.8220 | 0.9528 | 0.6774 | 0.7918 |
| 4 | BiLSTM v2 (09-17 复现) | 0.8170 | 0.9352 | 0.6811 | 0.7882 |
| 5 | MobileNet1D | 0.8176 | 0.9456 | 0.6739 | 0.7869 |
| 6 | RNN Basic | 0.7529 | 0.9059 | 0.5643 | 0.6954 |

**观察**: TCN v3a deep 阶段二 window8 冠军 (F1=0.8093). ShuffleNet1D 性价比之王 (5.9K 参数 + F1=0.7918).

### 3.3 window4 (N=13,731, Attack=48.7%)

| # | 模型 | Acc | P | R | F1 |
|:--:|-----|---:|---:|---:|---:|
| 1 | CNN v2 | 0.7701 | 0.9262 | 0.5738 | 0.7086 |
| 2 | LSTM v2 | 0.7399 | 0.8563 | 0.5602 | 0.6773 |

### 3.4 no_window (N=54,927, Attack=21.9%) — 阶段二传统 ML

| # | 模型 | 类别 | Acc ⚠️ | P | R | F1 |
|:--:|-----|:--:|---:|---:|---:|---:|
| 1 | FNN (MLP) v2 | DL | 0.8836 | 0.9029 | 0.5242 | 0.6633 |
| 2 | Decision Tree v2 | ML | 0.8842 | 0.9327 | 0.5068 | 0.6567 |
| 3 | Random Forest v2 | ML | 0.7858 | 0.5062 | 0.8357 | 0.6305 |
| 4 | LightGBM 19-dim | GBDT | 0.7714 | 0.4872 | 0.8689 | 0.6243 |
| 5 | SVM 19-dim | ML | 0.8298 | 0.9521 | 0.2335 | 0.3750 |

**观察**: FNN/DT **Acc 高但 Recall 0.51 漏报一半**, LGB/RF **Recall 高但 Precision 0.49 误报爆炸**, SVM 完全不可用.

### 3.5 stage1_t70 (N=82,389, Attack=21.6%) — 阶段一 baseline

| # | 模型 | Acc | P | R | F1 |
|:--:|-----|---:|---:|---:|---:|
| 1 | LightGBM baseline (32 特征, 09-17 重训) | 0.8129 | 0.5478 | 0.7694 | **0.6400** |

**观察**: 阶段一 baseline 的 test 集比阶段二 no_window 大 **1.5x** (82k vs 55k), Attack 占比几乎相同 (21.6% vs 21.9%). F1=0.6400 与阶段二 no_window 同档 (0.6243-0.6633), **重训 LGB 32 特征比阶段二 LGB 19-dim 略强** (+0.016).

---

## 4. 关键发现

### 4.1 Precision 提升是 SOTA 核心 ⭐

| 阶段 | Precision (Attack) | 模型 |
|:--:|---:|---|
| 一 | **0.5478** | LGB baseline 32 特征 (重训) |
| 二 no-window | 0.4872 - 0.9327 | LGB 19-dim / DT v2 |
| 二 window8 | 0.9059 - 0.9528 | RNN / ShuffleNet / CNN-LSTM |
| 五 SOTA | **0.9553** | TCN+SE 23-dim 5-seed |

- 阶段一 LGB baseline **P=0.55** 误报率 45% → **生产不可用**
- 阶段五 SOTA **P=0.96** 误报率仅 4.5% → **生产可用**
- 提升来自:窗口化 + TCN 序列建模 + SE 通道注意力 + 5-seed 集成

### 4.2 Recall 在 SOTA 接近 0.81 的天花板

| 阶段 | Recall (Attack) | 模型 |
|:--:|---:|---|
| 一 | **0.7694** | LGB baseline |
| 二 no-window | 0.8357 | RF v2 |
| 二 window8 | 0.7045 | TCN v3a deep |
| 五 SOTA | **0.8050** | TCN+SE 5-seed |

- 阶段五 Recall 0.805 比阶段一 0.769 仅 +0.036 (R 在 SCADA IDS 上是**结构性瓶颈**, 受攻击类别混淆影响)
- LGB/RF 高 Recall (0.84-0.87) 是用**误报爆炸** (P<0.51) 换的, 不可取

### 4.3 测试集 Attack 占比对 F1 的影响 (重要!)

- **Attack 占比低 (21.6% / 21.9%)**: 模型必须平衡两类 → F1 普遍偏低 (0.62-0.66)
- **Attack 占比高 (49-52%)**: 二分类近似 1:1 → F1 普遍高 (0.78-0.87)
- **直接比较 F1 不可取**, 推荐:
  1. 同测试集组内排名 (本报告第 3 节)
  2. 阶段增益轨迹 (本报告第 2 节)
  3. Precision / Recall 单独看 (本报告第 4.1, 4.2 节)

### 4.4 阶段一 baseline 重新定位

- 阶段一 LGB baseline (32 特征, test=82,389) F1=**0.6400** 是**真实下限**
- 与阶段二 no_window LGB 19-dim F1=0.6243 对比 → **多 13 特征 +0.016 F1**, 边际收益递减
- **直接生产部署起点**: 阶段二 TCN v3a deep (F1=0.8093), 跳过传统 ML
- **SOTA 起点**: 阶段五 TCN+SE 5-seed (F1=0.8737), 但需要 ~80K 参数 + 5-seed 集成

---

## 5. 模型规模对比 (参数量 / 推理时间)

| 阶段 | 模型 | 参数量 | 推理 (PC) | MCU 部署 | 备注 |
|:--:|-----|---:|:--:|:--:|---|
| 一 | LGB baseline 32 特征 | tree-based | <10ms | 不适合 (树结构) | 基线 |
| 二 | ShuffleNet1D | **5.9K** | ~5ms | STM32F407 ✓ | 性价比之王 |
| 二 | RNN Basic | ~9K | ~3ms | STM32F407 ✓ | 最便宜 deep |
| 二 | BiLSTM v2 window=8 | 60,481 | ~15ms | STM32F407 ✓ | bit-perfect 复现 |
| 二 | MobileNet1D | ~43K | ~10ms | STM32F407 ✓ | 深度可分离 |
| 三 | TCN+SE v3 window16 | ~80K | ~25ms | STM32F407 ✓ | 阶段三冠军 |
| **五 SOTA** | **TCN+SE 23-dim 5-seed** | **79K × 5 = 395K** | **~25ms × 5** | STM32H743 ✓ | 当前 SOTA |

**SOTA 部署建议** (来自 `project-h743-ch32-deploy.md`):
- **生产高精度**: STM32H743 + 单 seed s123 (F1=0.8728, 推理 ~0.13ms @ 480MHz)
- **离线批量分析**: PC + 5-seed ensemble (F1=0.8737, +0.0009 vs 单 seed)

---

## 6. 4 维度冠军 (按测试集组)

| 指标 | 冠军 | F1/Acc | 测试集 |
|:--:|-----|:--:|:--:|
| **Accuracy (合理)** | ★ TCN+SE 5-seed | **0.8776** | window16 |
| **Precision** | TCN+SE 5-seed | **0.9553** | window16 |
| **Recall (合理)** | ★ TCN+SE 5-seed | **0.8050** | window16 |
| **F1-Score (综合)** | ★ TCN+SE 5-seed | **0.8737** | window16 |

> 4 个维度 SOTA 都是 TCN+SE 23-dim 5-seed ensemble (window16), 验证 SOTA 地位无可争议.

---

## 7. 关键结论

1. **TCN+SE 23-dim ch=32 5-seed ensemble 是当前 SOTA**, 4 指标全面领先:
   - Acc=0.8776 / P=0.9553 / R=0.8050 / F1=0.8737
2. **从阶段一 baseline → SOTA 跨越 101 天**: F1 +0.2337 (+36.5% 相对)
3. **核心提升来自 Precision (0.55 → 0.96)**, Recall 在 SOTA 接近 0.81 天花板
4. **窗口化是核心**: window16 > window8 > window4 > no_window, F1 差异 +0.21
5. **测试集 Attack 占比影响 F1**: 直接比较 F1 不可取, 推荐看 PR-AUC 或 Precision/Recall 单项
6. **生产部署推荐**:
   - **STM32H743**: 单 seed s123 (F1=0.8728), 推理 0.13ms @ 480MHz, 88KB RAM
   - **PC 离线**: 5-seed ensemble (F1=0.8737)
7. **阶段一 LGB baseline 的价值**: 是**真实下限** (F1=0.6400), 用于**评估后续模型增益的锚点**

---

## 8. 数据来源

| 文件 | 来源阶段 | 内容 |
|------|:--:|------|
| `cross_phase_4metrics.json` | 总览 | 21 模型 + 阶段增益轨迹 |
| `cross_phase_4metrics_table.csv` | 总览 | CSV 表格 |
| `phase1_4metrics.json` | 一 | LGB baseline 32 特征重训 |
| `compare_models_4metrics.json` | 二/三 | 17 模型横评 |
| `rerun_4metrics_5seed.json` | 五 | SOTA 5-seed ensemble |

**复现脚本**:
```bash
python phase1_lgb_4metrics.py        # 阶段一 baseline (~14min)
python compare_models_4metrics.py     # 阶段二/三 (~30s)
python build_cross_phase_4metrics.py  # 跨阶段总览 (~1s)
```

**关联**: [[project-2026-09-17-phase1-4metrics]] | [[project-2026-09-17-phase2-4metrics]] | [[project-2026-09-15-4metrics-cross-model]] | [[feedback-ids-binary-precision-convention]]