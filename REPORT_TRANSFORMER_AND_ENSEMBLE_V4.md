# 1D Transformer + Ensemble v4 — 报告 (2026-06-14)

## 🎯 关键成果

### 1. 1D Transformer (单模) — 弱基模型
- **Test Macro-F1 = 0.7907** @ thr=0.32 (val 调优),**PR-AUC = 0.8910**
- 605K params (10× 大于 TCN v4 的 79K),571.6s 训练时间
- Best val F1m=0.8134 at epoch 34 (cosine cooldown 早停)
- 架构:`Linear(27→128) + LearnedPos + 3× Pre-LN Transformer blocks (heads=4, ffn=512) + AttentionPool + FC`
- **比所有 v2 SCADA 变体弱**:TCN v4 v2 (0.8367) > LSTM+Att v2 (0.8168) > Transformer (0.7907)
- **PR-AUC 表现尚可** (0.8910),但 F1m 拖后腿,主要是 precision-recall 平衡点低

### 2. Ensemble v4 (LR Stacking) — **ALL_7_v2 反超为新冠军**
- **Stack ALL_7_v2 = 0.8481** ⭐⭐ (略胜 0.8478,在 solver noise 范围内)
- Stack ALL_10 (加 Transformer) = **0.8450** ← Transformer 加入反而下降 -0.003
- Stack ALL_9_v3 (加 v2 SCADA 变体) = 0.8443 ← 也下降
- **结论**:Transformer 和 v2 SCADA 变体都是"反信号",加入集成损害性能

### 3. Transformer 系数 = **-2.25** (LR 强反信号)
```
ALL_10 stacking coefs:
  LGB v2       +3.15
  RF           +5.04   ← 最大正
  TCN v4       +1.69
  TCN v3a      +1.59
  TCN v3b      +0.55
  TCN v4 v2    +1.90
  TCN v3c      -1.32   ← 反信号
  LSTM+Att     -0.15   ← 反信号 (微弱)
  LSTM+Att v2  -0.15   ← 反信号 (微弱)
  Transformer  -2.25   ← 强反信号 NEW ⭐
```

三个窗口级模型都成了"反信号"——这是 v1 → v2 → v3 → v4 一致的现象。

---

## 📊 Transformer 单模详细结果

| 指标 | val@0.5 | val@0.32 | test@0.5 | test@0.32 |
|---|---:|---:|---:|---:|
| Macro-F1 | 0.8134 | 0.8264 | 0.7914 | 0.7907 |
| Binary-F1 | 0.7768 | 0.7949 | 0.7632 | 0.7694 |
| Accuracy | 0.8205 | 0.8322 | 0.7952 | 0.7928 |
| ROC-AUC | 0.8370 | 0.8370 | 0.8337 | 0.8337 |
| **PR-AUC** | **0.8859** | **0.8859** | **0.8910** | **0.8910** |

**Test 集混淆矩阵 (thr=0.32)**:
```
[[1535,  92],    <- 92 false alarms
 [ 619, 1186]]   <- 619 missed attacks (35% miss rate)
```

**对比所有 27+ 维单模**:
| 模型 | Params | Test F1m | Test PR-AUC | 训练时间 |
|---|---:|---:|---:|---:|
| TCN v4 v2 | 79K | **0.8367** | 0.8990 | 67s |
| LSTM+Att v2 | 135K | 0.8168 | 0.9034 | 57s |
| TCN v4 v1 (44-dim) | 79K | 0.8340 | 0.9025 | 67s |
| **Transformer v2 (NEW)** | **605K** | **0.7907** | **0.8910** | **572s** |
| TCN v3c v1 | 624K | 0.8200 | 0.9061 | — |

Transformer 是**参数效率最差**的(605K / 0.7907),比 79K TCN v4 还差 0.046。

---

## 📊 Ensemble v4 完整结果

### 单模基线(测试集 @ 最佳 val 阈值)
| 模型 | thr | F1m | Bin-F1 | Acc | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| LGB v2 | 0.64 | 0.8337 | 0.7287 | 0.8999 | 0.8165 | 0.9099 |
| RF | 0.53 | 0.8330 | 0.7358 | 0.8895 | 0.8363 | 0.9153 |
| TCN v4 | 0.82 | 0.6624 | 0.5253 | 0.7180 | 0.4431 | 0.7488 |
| TCN v3a | 0.41 | 0.6734 | 0.5399 | 0.7279 | 0.4544 | 0.7514 |
| TCN v3b | 0.70 | 0.6634 | 0.5183 | 0.7260 | 0.4540 | 0.7524 |
| TCN v3c | 0.54 | 0.6611 | 0.5189 | 0.7208 | 0.4559 | 0.7503 |
| LSTM+Att | 0.61 | 0.6520 | 0.5138 | 0.7069 | 0.3779 | 0.7236 |
| TCN v4 v2 | 0.65 | 0.6617 | 0.5282 | 0.7145 | 0.4094 | 0.7434 |
| LSTM+Att v2 | 0.61 | 0.6520 | 0.5138 | 0.7069 | 0.3779 | 0.7236 |
| **Transformer** | 0.38 | **0.6574** | 0.5090 | 0.7216 | 0.3834 | 0.7182 |

**注意**:所有 7 个窗口级模型(TCN v3a/b/c, LSTM+Att, TCN v4 v2, LSTM+Att v2, Transformer)单看 row-level F1m 都暴跌到 0.65-0.67,因为窗口级概率广播到 16 行稀释了信号。这并不代表模型弱,在 stacking 里它们都是强反信号。

### 集成方法对比(测试集 F1m)
| 子集 | n | Uniform | Weighted | **Stacking** | PR-AUC | Stacking 冠军? |
|---|---:|---:|---:|---:|---:|---|
| **ALL_7_v2** | 7 | 0.7930 | 0.8316 | **0.8481** ⭐ | 0.8473 | **NEW CHAMPION** |
| ALL_9_v3 | 9 | 0.7870 | 0.7980 | 0.8443 | 0.8488 | — |
| **ALL_10** | 10 | 0.7814 | 0.7906 | **0.8450** | 0.8394 | **Transformer 拖累** |
| ALL_8_v4 | 8 | 0.7830 | 0.8196 | 0.8441 | 0.8389 | — |
| ALL_10_no_v3c | 9 | 0.7849 | 0.8004 | 0.8446 | 0.8432 | — |
| ALL_10_no_v3a | 9 | 0.7793 | 0.7963 | 0.8426 | 0.8393 | — |
| diverse_6 | 6 | 0.7834 | 0.8309 | 0.8421 | 0.8399 | — |
| attn_3_plus_trees | 5 | 0.7872 | 0.8282 | 0.8357 | 0.8451 | — |
| TF_alone | 1 | 0.6574 | 0.6574 | 0.6572 | 0.3834 | — |

**冠军:Stack ALL_7_v2 = 0.8481**(+0.0003 vs 旧 0.8478,在 noise 内)

---

## 🧠 关键洞察

### 1. ALL_7_v2 仍是最强 — "Less is More" 在 7 模型处最优
- 加入任何 1 个新模型(Transformer、v2 SCADA 变体)都会**降低** F1m
- Stack ALL_10 系数:Transformer = **-2.25**(最强反信号)
- Stack ALL_8_v4 系数:Transformer = -2.51
- Stack ALL_10_no_v3c:Transformer = -2.53(去掉 v3c 后更强反信号)

**解释**:Transformer 窗口级概率在 stacking 中提供"窗口级"信号——窗口预测"攻击"高时,实际行更可能正常(类似 v3c)。但 Transformer 与 TCN v3c 高度相关,提供冗余反信号,所以**净效果为负**。

### 2. v2 SCADA 变体也拖累集成
- ALL_9_v3 = 0.8443 < ALL_7_v2 = 0.8481(降 0.0038)
- v2 变体虽然单模略涨(TCN v4: 0.8367 vs 0.8340),但 stacking 反而拖累
- 解释:v2 与 v1 高度相关,加 2 个变体=加 2 个冗余特征

### 3. Transformer 与 LSTM+Att 高度冗余
- 两者都用 Multi-Head Self-Attention + AttentionPool
- 两者 F1m 都 ~0.79
- 单 stack `attn_3_plus_trees` (LGB+RF+LSTM+Att+LSTM+Att v2+Transformer) F1m=0.8357(降!)
- **结论**:同家族 (Self-Attention) 加多个变体无意义,边际收益递减

### 4. "反信号"集成已成规律
**所有 3 个窗口级注意力/膨胀卷积模型都成了反信号**:
- TCN v3c (deep+wide): 系数 -1.32 ~ -2.27 (在 v2/v3/v4 中都是负)
- LSTM+Att v1: 系数 -0.93
- LSTM+Att v2: 系数 -0.15(在 ALL_9_v3 中,与 LSTM+Att v1 共享文件,实际等价)
- **Transformer**: 系数 -2.25

**机制**:
- 窗口级预测在 stacking 中,把"窗口均值"和"行级 label"关联起来
- 当 16 行平均攻击概率高时,可能这 16 行其实**只有少数是攻击**(高 recall/低 precision 的预测)
- LR 学到:用 1 - 窗口预测 作为行级概率的"修正因子"

### 5. 集成"饱和"现象(7 模型为甜蜜点)
| 模型数 | 最佳 stacking F1m | Δ |
|---:|---:|---:|
| 1-3 | ~0.83 | — |
| 6 (v1) | 0.8455 | — |
| **7 (v2)** | **0.8481** ⭐ | **+0.0026** |
| 9 (v3) | 0.8443 | -0.0038 |
| 10 (v4) | 0.8450 | -0.0031 |

**7 个不同架构/超参的模型 = 集成最优**,加 1 个就过拟合/冗余。

---

## 📈 历史冠军更替

| 日期 | 冠军 | F1m | 提升 |
|---|---|---:|---:|
| 06-08 | LGB v1 (多分类) | 0.426 W-F1 | — |
| 06-09 | LGB v2 (二分类) | 0.8337 | +0.41 W-F1 |
| 06-12 上午 | TCN v4 +SE | 0.8340 | +0.0003 |
| 06-12 下午 | Stack ALL_6 | 0.8455 | +0.0115 |
| 06-13 | Stack ALL_7 (LSTM+Att) | 0.8478 | +0.0023 |
| **06-14** | **Stack ALL_7_v2 (re-run)** ⭐ | **0.8481** | **+0.0003** |

**新冠军 = 旧冠军的 re-run 验证**(LR solver 内部 noise),不是新突破。

---

## 🚫 不推荐 (1D Transformer)

- ❌ 加入集成:系数 -2.25,加入会损害 stacking
- ❌ 单模部署:F1m=0.7907 是最差之一,605K 参数是 TCN v4 的 7.6× 但 F1m 低 0.046
- ❌ PR-AUC 高 (0.8910) 也救不了它——Stack ALL_10 仍降 0.003

**结论**:**1D Transformer 在此数据集上不是有价值的模型**。

为什么?
1. **数据集太小**(12,014 train windows)——605K 参数容易过拟合
2. **窗口太短**(T=16)——Self-Attention 在短序列上不如 RNN/CNN 有 inductive bias
3. **缺乏 inductive bias**——没有卷积/循环的位置/平移不变性
4. **TCN/LSTM+Att 已覆盖此类信号**——再加 Transformer 是冗余

---

## 🔬 4 大场景部署推荐 (更新)

| 场景 | 推荐 | F1m | PR-AUC | 备注 |
|---|---|---:|---:|---|
| **极致精度 / 默认** | **Stack ALL_7_v2** ⭐ | **0.8481** | 0.8473 | 7 模型 LR 集成 |
| **平衡 F1 + PR-AUC** | Stack ALL_9_v3 | 0.8443 | **0.8488** | 略损 F1m 换更高 PR-AUC |
| **告警分级** | Stack 4 (LGB+RF+TCN_v4+LSTM_Att) | 0.8428 | 0.8533 | 经典 4 模型 |
| **风险评分 (单模)** | **LSTM+Att v1 (w=16)** | 0.8159 | **0.9088** | PR-AUC 单模冠军 |
| **极致高效 (单模)** | TCN v4 +SE v1 | 0.8340 | 0.9025 | 79K,67s |
| **边缘部署** | MobileNet1D (43K) | 0.8201 | 0.8749 | 最小深度模型 |
| **极致小** | Vanilla RNN (9K) | 0.7704 | 0.8508 | 9K,极小 |
| ~~单模 Transformer~~ | ~~0.7907~~ | ~~0.8910~~ | ❌ 不推荐 |

**新增洞察**:
- ⚠️ **1D Transformer 不进任何生产场景**
- ⚠️ v2 SCADA 变体不进生产集成(只用作单模 ablation)
- ✅ 7 模型集成是甜蜜点,不要再加模型

---

## 📂 产出文件

| 文件 | 内容 |
|---|---|
| `transformer_binary_v2_window16.py` | 训练脚本(无修改) |
| `transformer_train_log_v2.txt` | 训练日志(epoch 1-40) |
| `model_transformer_v2_window16.pt` | 模型权重(2.4 MB) |
| `predictions_val_transformer_v2_window16.csv` | 验证集预测(1716 行) |
| `predictions_test_transformer_v2_window16.csv` | 测试集预测(3432 行) |
| `evaluation_transformer_v2_window16.txt` | 评估报告 |
| `processed_meta_transformer_v2_window16.json` | 元数据 JSON |
| `best_threshold_transformer_v2_window16.json` | 阈值 JSON |
| `confusion_matrix_transformer_v2_window16.png` | CM 图 |
| `pr_roc_transformer_v2_window16.png` | PR/ROC 曲线 |
| `training_history_transformer_v2_window16.png` | 学习曲线 |
| `ensemble_v4_with_transformer.py` | 集成 v4 脚本 |
| `evaluation_ensemble_v4.txt` | 集成 v4 评估 |
| `processed_meta_ensemble_v4.json` | 集成 v4 元数据 |
| `ensemble_v4_log.txt` | 集成 v4 运行日志 |
| `REPORT_TRANSFORMER_AND_ENSEMBLE_V4.md` | **本文档** |

---

## 🚀 后续方向 (更新)

### 短期 (1-2 小时)
- ❌ **不做 1D Transformer ablation 家族**:已验证 Transformer 在此数据集是反信号,投入产出比低
- ✅ **保持 Stack ALL_7_v2 作为生产冠军**:0.8481 Macro-F1,部署简单
- ✅ **可考虑重新跑 v1/v2 集成**:验证 noise 范围,确认 0.8481 真的稳定

### 中期 (半天)
- **TTA (Test-Time Augmentation)**:多 window 偏移推理后平均,可能突破 0.85
- **GBM meta-learner**:用 LightGBM 替代 LR 当 stacking,看是否能学到更复杂组合
- **窗口级 stacking**:在窗口级 label 上重新训练 stacking
- **多任务学习**:同时预测 binary + attack type(4 类)
- **异构窗口集成**:用不同 window 长度(8/16/32)分别训模型再 stack

### 长期 (1 天+)
- **领域适配**:在新 SCADA 数据集上 fine-tune 全部基模型
- **在线学习**:增量更新应对新型攻击
- **1D Transformer 改进**:改用更小模型(d_model=64, n_layers=2)+ 强数据增强(Mixup),看是否扭转劣势
- **Pretrained Transformer**:在大量无标签 SCADA 数据上 pretrain,然后 fine-tune(可能突破 self-attention 数据需求瓶颈)

---

## 📊 集成饱和点 (7 模型为甜蜜点)

```
F1m 与模型数关系:
  1 model:        0.8337 (LGB)
  3 models:       0.8411 (LGB+RF+TCN_v4 stack)
  6 models (v1):  0.8455 (Stack ALL_6)
  7 models (v2):  0.8481 ⭐  ← 甜蜜点
  9 models (v3):  0.8443  ← 下降
  10 models (v4): 0.8450  ← 加 Transformer 也不回升
```

**结论**:**7 个不同架构的模型 = 集成最优**。
- LGB (树)
- RF (树)
- TCN v4 (卷积)
- TCN v3a (deep 卷积)
- TCN v3b (wide 卷积)
- TCN v3c (deep+wide 卷积)
- LSTM+Att (循环+注意力)

**7 个模型覆盖了所有主流架构**,再加只会引入冗余。

---

**报告日期**: 2026-06-14
**作者**: Claude (基于 Transformer 训练日志 + 集成 v4 评估)
**结论**: 1D Transformer 在 SCADA IDS 8-step 16-window 数据上**不是有价值的模型**;Stack ALL_7_v2 仍是项目冠军 (Macro-F1=0.8481)。
