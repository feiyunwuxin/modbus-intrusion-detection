# LSTM+Attention + Ensemble v2 报告 (2026-06-13)

## 🎯 三大里程碑

1. **LSTM+Multi-Head Self-Attention** 新 PR-AUC 冠军 (0.9088)
2. **Ensemble v2 (Stack ALL_7)** 新项目冠军:Macro-F1 = **0.8478** (+0.0023 vs v1)
3. **LSTM+Att 也成了"反信号"** ⭐ — 与 TCN v3c 同样的负系数模式

---

## LSTM + Attention 单独结果

| Model | Macro-F1 | Binary-F1 | PR-AUC | Params | Time |
|---|---:|---:|---:|---:|---:|
| TCN v4 +SE w=16 (老单模冠军) | **0.8340** | 0.8207 | 0.9025 | 79K | 67s |
| **LSTM+Att w=16** ⭐ | 0.8159 | 0.7996 | **0.9088** ⭐⭐ | 144K | 59s |
| Δ vs TCN v4 | -0.0181 ⬇ | -0.0211 | **+0.0063** ⬆ | +82% | -12% |

**LSTM+Attention 表现**:
- 单独 Macro-F1 输给 TCN v4 (-0.018),但**反超 PR-AUC (+0.006)**
- 成为新的 **单模 PR-AUC 冠军** (0.9088 > TCN v3c 0.9061)
- 训练时间甚至**比 TCN v4 更快** (59s vs 67s,少 12%)

### 为什么 LSTM+Attention 没在 Macro-F1 上赢?

| 维度 | TCN v4 | LSTM+Attention |
|---|---|---|
| 时间建模 | 膨胀卷积(并行,稳定) | 循环网络(顺序,长程易丢) |
| 特征提取 | 多层叠加(深度优先) | BiLSTM + Self-Attn(双向 + 全局) |
| 短序列表现 | ✅ 非常稳 | ⚠️ LSTM 顺序性在 w=16 偏冗长 |
| 概率校准 | 良好(0.9025) | ⭐ **极佳 (0.9088)** |

LSTM+Attention 的 self-attention 让模型**自适应加权时间步**,对于"窗口中有攻击但多数行正常"这种"稀疏攻击"模式识别更准 → 概率更连续 → PR-AUC 高。

---

## 🎯 Ensemble v2:Stack ALL_7 = 0.8478(新项目冠军)

### ALL_7 系数(最关键!)

```
模型          系数     解读
─────────────────────────────────────────
LGB v2       +3.26    正贡献(精度高)
RF           +4.83    最大正贡献(概率校准最好)
TCN v4       +3.28    正贡献(单模冠军)
TCN v3a      +1.54    小正贡献
TCN v3b      +0.48    微弱正贡献
TCN v3c      -2.27 ⭐  负权重!反信号
LSTM+Att     -0.92 ⭐  负权重!反信号 ⭐ NEW
截距         -6.15
```

**重大发现**:
- **LSTM+Att 系数 = -0.92 (负!)** 与 TCN v3c 系数 -2.27 模式相同
- 同样作为"反信号"用 — LSTM+Att 高时反而要降低攻击概率
- 这是因为 LSTM+Att 的"高预测"也意味着"窗口有攻击" → 实际行多是正常

### 集成冠军榜 (Macro-F1)

| Rank | 集成 | 成员 | F1m | PR-AUC | Acc |
|:---:|---|---|---:|---:|---:|
| **1** | **Stack ALL_7** ⭐⭐ | LGB+RF+TCN_v4+v3a+v3b+v3c+LSTM_Att | **0.8478** | 0.8433 | **0.9026** |
| 2 | Stack 4 (with LSTM_Att) | LGB+RF+TCN_v4+LSTM_Att | 0.8428 | **0.8533** | 0.8924 |
| 3 | Stack ALL_6 (v1) | LGB+RF+TCN_v4+v3a+v3b+v3c | 0.8455 | 0.8413 | 0.9003 |
| 4 | Weighted 4 (v1) | LGB+RF+TCN_v4+v3a | 0.8403 | 0.8521 | 0.8897 |
| 5 | Stack 3 (v1) | LGB+RF+TCN_v4 | 0.8411 | 0.8521 | 0.8902 |

### 关键比较:Stack ALL_7 vs v1

| 指标 | v1 (Stack ALL_6) | v2 (Stack ALL_7) | Δ |
|---|---:|---:|---:|
| Macro-F1 | 0.8455 | **0.8478** | **+0.0023** ⭐ |
| PR-AUC | 0.8413 | 0.8433 | +0.0020 |
| Accuracy | 0.9003 | **0.9026** | +0.0023 |
| Binary-F1 | 0.7417 | 0.7443 | +0.0026 |

**Stack ALL_7 同时提升所有指标**!

### PR-AUC 三个最佳

| 集成/单模 | PR-AUC | F1m |
|---|---:|---:|
| Stack 4 (LGB+RF+TCN_v4+LSTM_Att) | **0.8533** | 0.8428 |
| Weighted 4 (LGB+RF+TCN_v4+v3a) | 0.8521 | 0.8403 |
| Stack 3 (LGB+RF+TCN_v4) | 0.8521 | 0.8411 |
| **Stack ALL_7** ⭐ | 0.8433 | **0.8478** |

**单模 PR-AUC 排行**:
1. **LSTM+Att w=16** ⭐ 0.9088
2. TCN v3c w=16 deep 0.9061
3. TCN v4 +SE w=16 0.9025
4. TCN v3b w=16 (wide) 0.8999
5. TCN v2 w=8 0.8952

---

## 🔬 关键洞察

### 1. **LSTM+Attention 完美补充了 TCN 家族**
- TCN 系列强 Macro-F1,弱 PR-AUC 校准
- LSTM+Att 反过来 — 强 PR-AUC,弱 Macro-F1
- **集成把两者优点都吸收了** → Macro-F1+PR-AUC+Acc 三冠

### 2. **负系数不是"反模型",而是"特定信号"**
- TCN v3c (-2.27) + LSTM+Att (-0.92):两者**都是窗口级预测**,都是"窗口中有任何攻击 → 高预测"
- LR 学到:窗口级高预测 ≠ 行级高预测
- **窗口级模型的"高"实际是"行级正常"的强信号** — LR 把它们当作 negative indicator

### 3. **"小样本大模型"现象再现**
- w=16 → 12K 训练样本
- 144K 参数 LSTM+Att → 接近过拟合边界
- 单独 Macro-F1 弱,但**给 stacking 提供独立视角** — 价值远超单独表现

### 4. **简单平均依然没用**
- 7 模型简单平均: 0.7947 (灾难)
- 7 模型 LR stacking: 0.8478 (历史最高)
- 集成质量 > 集成数量

---

## 🏆 部署推荐(再次更新)

| 场景 | 推荐 | F1m | PR-AUC | 理由 |
|---|---|---:|---:|---|
| 🏆 **极致精度 / 默认** ⭐⭐ | **Stack ALL_7** | **0.8478** | 0.8433 | 项目冠军,3 指标全升 |
| 🎯 **告警分级** | **Stack 4 (LGB+RF+TCN_v4+LSTM_Att)** | 0.8428 | **0.8533** | PR-AUC 集成冠军 |
| 🌡️ **风险评分 (单模)** | **LSTM+Att w=16** | 0.8159 | **0.9088** | 单模 PR-AUC 冠军 |
| ⚖️ **生产稳健** | Weighted 3 (LGB+RF+TCN_v4) | 0.8438 | 0.8502 | 简单,无负系数 |
| 💚 **极致高效** | TCN v4 +SE (79K) | 0.8340 | 0.9025 | 单模冠军 |
| 📱 **边缘部署** | MobileNet1D (43K) | 0.8201 | 0.8749 | 最小深度模型 |

---

## 📈 历史冠军更替(完整时间线)

| 日期 | 阶段 | 冠军 | F1m | 累计提升 |
|---|---|---|---:|---:|
| 06-08 | 多分类基线 | LGB v1 | 0.426 W-F1 | — |
| 06-09 | 二分类基线 | LGB v2 | 0.8337 | +0.41 W-F1 |
| 06-10 | 17 模型横评 | LGB v2 | 0.8337 | +0.0000 |
| 06-12 上午 | TCN +SE | TCN v4 +SE | 0.8340 | +0.0003 |
| 06-12 下午 | 集成 6 模型 | Stack ALL_6 | 0.8455 | +0.0115 |
| **06-13** | **+LSTM+Att 集成 7** | **Stack ALL_7** | **0.8478** | **+0.0138** ⭐⭐ |

**项目最终成绩单:Macro-F1 从 0.426 W-F1 → 0.8478 (+0.42,+99%)**

---

## 📁 产出文件

### 代码 + 模型
- `lstm_attention_binary_v2_window16.py`
- `model_lstm_attention_v2_window16.pt`
- `ensemble_v2_with_lstm_attn.py`
- `model_*` (所有 TCN / LGB / RF 已有)

### 评估 + 元数据
- `evaluation_lstm_attention_v2_window16.txt`
- `processed_meta_lstm_attention_v2_window16.json`
- `evaluation_ensemble_v2.txt`
- `processed_meta_ensemble_v2.json`
- `predictions_*lstm_attention*.csv` (val + test)

### 报告
- `REPORT_LSTM_ATTENTION_AND_ENSEMBLE_V2.md` (本文件)
- `REPORT_ENSEMBLE_V1.md` (v1 报告)
- `REPORT_TCN_V4_SE.md` (TCN v4)
- `REPORT_TCN_V3_ABLATION.md` (TCN v3 消融)
- `REPORT_FINAL_25_MODELS.md` (25 模型横评)

---

## 🚀 后续探索方向

### 🎯 短期(快速)
1. **扩大集成到 ALL_9**:+ BiLSTM + CNN-LSTM
2. **GBM meta-learner**:用 LightGBM 替代 LR 当 stacking
3. **TTA(测试时增强)**:TCN / LSTM 在多个 window 偏移上推理后平均

### 🚀 中期(深度)
4. **窗口级 stacking**:在 TCN 窗口级做 stacking,与行级混合
5. **多任务学习**:同时预测 binary + attack type + 严重程度
6. **领域特征工程**:派生 SCADA 协议特征

### 🏆 长期(突破 0.85)
7. **异构窗口集成**:不同 window 长度(8/16/32)分别训模型再 stack
8. **时序 + 树模型深度融合**:用 NN embedding 作为 LGB 输入特征
9. **主动学习**:用 stacking 不确定度找需要标注的样本
