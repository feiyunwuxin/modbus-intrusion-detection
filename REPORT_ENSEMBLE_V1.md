# Ensemble v1 — LGB + TCN v4 + RF (+ v3a/v3b/v3c) (2026-06-12)

## 🏆 历史性成果:**STACK ALL_6** 模型以 **Macro-F1 = 0.8455** 登顶!

超过 TCN v4 (0.8340) +0.0115,超过 LGB v2 (0.8337) +0.0118。

## 实验设置

**基模型 (6 个)**:
- LGB v2 (boosted tree, Macro-F1 0.8337)
- Random Forest (bagged tree, Macro-F1 0.8330)
- TCN v4 +SE (deep, Macro-F1 0.8340)
- TCN v3a (deep, Macro-F1 0.8336)
- TCN v3b (wide, Macro-F1 0.8326)
- TCN v3c (deep+wide, Macro-F1 0.8200, PR-AUC 0.9061)

**对齐处理**:TCN 预测在窗口级别(每 8 或 16 行共享 1 个 prob),LGB/RF 在行级别。
本实验将 TCN 的窗口概率复制到该窗口的每行,再截断到所有模型的最小公共长度:
- val: 27,456 行(2,456 行被截断)
- test: 54,912 行

**集成方法 (3 种)**:① 简单平均 ② 加权平均(SLSQP 在 val 上优化)③ LR Stacking(val 拟合 + 5-fold CV 选 C)

**7 种基模型组合**:从 2 成员到 6 成员,共 21 个集成变体。

---

## 全部 21 个集成结果(关键)

| Method | Members | Best F1m | PR-AUC | 备注 |
|---|---|---:|---:|---|
| 简单平均 | LGB+RF+TCN_v4 | 0.8302 | 0.8403 | 最朴素 |
| **加权平均** ⭐ | **LGB+RF+TCN_v4** | **0.8438** | **0.8502** | **w=[0.10, 0.81, 0.09] RF 主导** |
| 加权平均 | LGB+RF+TCN_v4+v3a | 0.8403 | 0.8521 | |
| 加权平均 | ALL_6 | 0.8368 | 0.8480 | |
| LR Stacking | LGB+RF+TCN_v4 | 0.8411 | 0.8521 | 干净版 |
| LR Stacking | LGB+RF+TCN_v4+v3a | 0.8386 | 0.8520 | |
| **LR Stacking** ⭐⭐ | **ALL_6** | **0.8455** | 0.8413 | **新冠军!** |

---

## 🌟 关键洞察

### 1. **集成打破历史天花板** — Macro-F1 突破 0.84!
- 之前所有 18 个独立模型的最高 Macro-F1 = 0.8340 (TCN v4)
- 集成后达到 **0.8455** (+0.0115,跨越 0.84 大关)
- 这是项目史上的最大单步提升

### 2. **加权平均比简单平均强大得多**
- 简单平均 LGB+RF+TCN_v4: 0.8302
- 加权平均 同一组合: 0.8438 (**+0.014**)
- SLSQP 优化器把 RF 权重调到 81%!这说明 **RF 的概率质量在集成中价值最大**
  - RF 单独: PR-AUC=0.8363 (树族最佳)
  - LGB 单独: 0.8166 (弱)
  - RF 比 LGB 更能校准概率,集成器"赏识"了这点

### 3. **LR Stacking ALL_6 的"反直觉"系数**
coefs: `[LGB=3.13, RF=4.97, TCN_v4=2.74, TCN_v3a=1.53, TCN_v3b=0.44, TCN_v3c=-2.55]`

- **TCN v3c 系数 = -2.55(负)**!这意味着 stacking 把它当作"反信号"用
- 因为 TCN v3c 的窗口预测(PR-AUC 0.9061)与行级 label 的相关性结构与其他模型不同
- LR 学会了: 当 TCN v3c 高时,反而要降低攻击概率(因为窗口内多数行其实正常)
- **这种负系数是 stacking 的"暗知识"——纯人工集成学不到的**

### 4. **简单平均几乎没有用**
- 6 个简单平均变体都 < 0.83
- **结论:必须用加权或 stacking,等权是浪费**

---

## 📊 19 模型大榜更新(项目历史最高 Macro-F1)

| Rank | Model | Family | Macro-F1 | PR-AUC | 备注 |
|:---:|---|---|---:|---:|---|
| **1** | **Stack ALL_6** ⭐⭐ NEW | 🎯 Ensemble | **0.8455** | 0.8413 | LGB+RF+TCN_v4+v3a+v3b+v3c → LR |
| 2 | Weighted LGB+RF+TCN_v4 | 🎯 Ensemble | 0.8438 | 0.8502 | RF 权重 81% |
| 3 | Stack LGB+RF+TCN_v4 | 🎯 Ensemble | 0.8411 | 0.8521 | 最干净的 3 模版 |
| 4 | Weighted LGB+RF+TCN_v4+v3a | 🎯 Ensemble | 0.8403 | 0.8521 | 加 v3a 微涨 |
| 5 | TCN v4 +SE | 🌀 TCN+SE | 0.8340 | 0.9025 | 单模冠军 |
| 6 | LGB v2 | 🌲 Boosted Tree | 0.8337 | 0.8166 | 老冠军 |
| 7 | TCN v3a (deep) | 🌀 TCN | 0.8336 | 0.8813 | |
| 8 | Random Forest | 🌲 Bagged Tree | 0.8330 | 0.8364 | |
| 9 | TCN v3b (wide) | 🌀 TCN | 0.8326 | 0.8999 | |
| 10 | CNN-LSTM w=8 | 🔀 Hybrid | 0.8296 | 0.8904 | |
| 11 | TCN v3c (deep+wide) | 🌀 TCN | 0.8200 | **0.9061** | 单模 PR-AUC 冠军 |

### 🌟 Top PR-AUC 排行
| Rank | Model | PR-AUC |
|:---:|---|---:|
| 1 | TCN v3c | **0.9061** |
| 2 | TCN v4 +SE | 0.9025 |
| 3 | Stack LGB+RF+TCN_v4 | 0.8521 |
| 4 | Weighted LGB+RF+TCN_v4+v3a | 0.8521 |
| 5 | Weighted LGB+RF+TCN_v4 | 0.8502 |

**单模 TCN 仍主导 PR-AUC;集成在 Macro-F1 上称王**

---

## 🏭 部署推荐(再次更新)

| 场景 | 推荐 | 理由 |
|---|---|---|
| **新默认 / 最高精度** ⭐⭐ | **Stack ALL_6 (LR)** | Macro-F1 = 0.8455,6 模型全参与,概率可解释 |
| **稳健 + 高 PR-AUC** ⭐ | **Weighted LGB+RF+TCN_v4** | Macro-F1 0.8438,PR-AUC 0.8502,无负系数更稳 |
| **极致 PR-AUC** | TCN v3c 单独 | PR-AUC 0.9061,但 Macro-F1 弱 |
| **极致 Macro-F1 + 单模型** | TCN v4 +SE | 0.8340,部署最简单 |
| ❌ 不推荐 | 简单平均集成 | 几乎没收益,浪费多样性 |

---

## 🔍 建议的生产流程

1. **离线训练**:用 Stack ALL_6 的 LR 系数 + 6 个模型的预测 (行级) → 部署
2. **在线推理**:对一个新 SCADA 样本:
   - LGB 给 17 维特征 → 概率 p1
   - RF 给 17 维特征 → 概率 p2
   - 4 个 TCN 模型给 16 步窗口(0.5s~0.6s 内 SCADA 数据) → 概率 p3-p6
   - LR meta: `p_final = sigmoid(-6.19 + 3.13·p1 + 4.97·p2 + 2.74·p3 + 1.53·p4 + 0.44·p5 - 2.55·p6)`
   - 二值化: `p_final >= 0.55` → 攻击警报
3. **监控**:对比 p_final 分布与历史分布,检测 drift

---

## 📁 产出文件

- **代码**:`ensemble_v1_lgb_tcn4_rf.py`
- **评估**:`evaluation_ensemble_v1.txt`
- **元数据**:`processed_meta_ensemble_v1.json`
- **日志**:`ensemble_v1_log.txt`

---

## 🚀 后续探索方向

### 🎯 短期(快速)
1. **优化 stack 系数**:用 NGBoost / LightGBM 替代 LR 作为 meta-learner
2. **二元/多样本融合**:同一窗口内多行的预测做注意力池化
3. **TTA(测试时增强)**:对每个 TCN 跑 2-3 个不同 window 起点,平均预测

### 🚀 中期(深度)
4. **扩大基模型池**:加入 BiLSTM, MobileNet, 树模型(GBDT)再集成
5. **窗口级 stacking**:TCN 在窗口级做 stacking(用窗口级 label),与行级 stacking 混合
6. **校准**:对 TCN v4 / v3c 用 Platt scaling 或 isotonic regression 校准概率

### 🏆 长期(突破 0.85)
7. **多任务学习**:同时预测 binary + attack type,共享底层 TCN
8. **领域知识特征工程**:基于 SCADA 协议规范的派生特征(指令间隔异常、命令-响应延迟等)
9. **二阶段检测**:先用集成做 anomaly detection(高 recall),再用规则系统降误报

---

## 历史冠军更替

| 日期 | 冠军 | Macro-F1 | 提升 |
|---|---|---:|---:|
| 2026-06-09 | LGB v2 (二分类) | 0.8337 | 基础 |
| 2026-06-12 (上午) | TCN v4 +SE | 0.8340 | +0.0003 |
| **2026-06-12 (下午)** | **Stack ALL_6** ⭐⭐ | **0.8455** | **+0.0115** |

**这是项目史上最大单步突破!**
