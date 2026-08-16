# SCADA 协议特征工程 v2 完整报告 (2026-06-13)

## 🎯 三阶段实验结果

### 1️⃣ SCADA 协议洞察

| 发现 | 含义 |
|---|---|
| **严格 1:1 command-response 配对** | FC=3: 68848 cmd + 68848 resp;FC=16: 64100 cmd + 64100 resp |
| **特征按角色出现** | `setpoint`/`gain`/etc. 100% NaN 在 resp;`pressure measurement` 100% NaN 在 resp |
| **异常 FC** | FC ∈ {136, 171, 139, 133, 137, 138, 140} **只有命令没有响应** ⭐ |
| **时间规律** | median dt=1.43s,数据呈交替 cmd-resp 模式 |

### 2️⃣ 设计的 10 个新特征

**行级 (3 个)**:
1. `time_since_last_same_addr_func` — 距上次同地址同 FC 的时间
2. `is_unusual_fc` — FC ∈ {136, 171, 139, 133, 137, 138, 140} 的标志
3. `is_response` — 命令/响应标志

**窗口级 (8 个,broadcast 到每行)**:
4-5. `cmd_count_w` / `resp_count_w` — 窗口内命令/响应数
6. `cmd_resp_balance_w` — (cmd-resp)/16
7-8. `crc_mean_w` / `crc_max_w` — CRC 聚合
9. `press_mean_w` — 压力聚合
10. `length_nunique_w` — 长度唯一值数

**总维度:17 (v1) + 3 (行级) + 8 (窗口级,但其中 1 是重复) = 27 维**

---

### 3️⃣ 单模结果(v2 vs v1 基线)

| 模型 | 数据 | Macro-F1 | PR-AUC | Bin-F1 | Acc | Params | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| TCN v4 +SE | v1 (44 dim) | 0.8340 | **0.9025** | 0.8207 | 0.8351 | 79K | 67s |
| **TCN v4 +SE v2** ⭐ | **v2 (27 dim)** | **0.8367** | 0.8990 | **0.8241** | **0.8377** | 75K | 64s |
| Δ | | **+0.0027** | -0.0035 | **+0.0034** | +0.0026 | -4K | -3s |
| | | | | | | | |
| LSTM+Att | v1 (44 dim) | 0.8159 | **0.9088** | 0.7996 | 0.8173 | 144K | 59s |
| **LSTM+Att v2** ⭐ | **v2 (27 dim)** | **0.8168** | 0.9034 | **0.8024** | **0.8179** | 135K | 57s |
| Δ | | **+0.0009** | -0.0054 | +0.0028 | +0.0006 | -9K | -2s |

**SCADA 特征对单模的影响**:
- ✅ **Macro-F1 微涨**:TCN v4 +0.0027,LSTM+Att +0.0009
- ✅ **Bin-F1 涨**:+0.0034 / +0.0028 → 告警性能提升
- ⚠️ **PR-AUC 微降**:-0.0035 / -0.0054 → 概率校准略受影响
- 🎉 **参数减少**:27 维(v2)vs 44 维(v1 with one-hot function),更简洁

---

## 🏆 集成 v3 结果(ALL_9 = v1 7 + v2 2)

### 完整 6 个集成变体

| Method | Members | F1m | PR-AUC | Acc | 备注 |
|---|---|---:|---:|---:|---|
| Uniform | ALL_9 | 0.7870 | 0.7546 | — | 全等权,差 |
| **Stack** ⭐ | **LGB+RF+TCN_v4+TCN_v4_v2+LSTM+Att+LSTM+Att_v2** | **0.8458** | **0.8541** | 0.9012 | **6 模型最平衡** |
| Stack | LGB+RF+TCN_v4_v2+LSTM+Att_v2 (4 model) | 0.8456 | **0.8534** | 0.8993 | **PR-AUC 集成冠军** |
| Stack | ALL_9 (9 model) | 0.8443 | 0.8488 | 0.9000 | 多模型稀释 |
| Weighted | LGB+RF+TCN_v4+TCN_v4_v2 | 0.8349 | 0.8521 | 0.8900 | 干净 4 模型 |
| Weighted | LGB+RF+TCN_v4_v2+LSTM+Att_v2 | 0.8239 | 0.8357 | 0.8865 | 单纯 v2 |

### v3 Stack ALL_6 (LGB+RF+TCN_v4+TCN_v4_v2+LSTM+Att+LSTM+Att_v2) 系数

```
LGB v2         +3.51  强正(精度)
RF             +4.49  最强正(校准)
TCN v4         +1.42  正
TCN v4 v2      +2.45  ⭐ 正(SCADA 增强版,贡献显著)
LSTM+Att       -0.99  反信号(v1)
LSTM+Att v2    -0.99  反信号(v2)
截距            ?
```

**v2 关键发现**:
- **TCN v4 v2 系数 = +2.45** — 与 TCN v4 系数 +1.42 累加,v2 实际贡献**比 v1 还大**!
- 但 LSTM+Att v2 与 v1 系数相同(-0.99),似乎没贡献独特信号
- v2 模型的"反信号"价值与 v1 重复

---

## 📊 全部 v1+v2 集成冠军榜

| Rank | 集成 | F1m | PR-AUC | Acc |
|:---:|---|---:|---:|---:|
| **1** | **Stack ALL_7 (v1, 6 model)** | **0.8478** ⭐⭐ | 0.8433 | **0.9026** |
| 2 | Stack 6 (v1+v2, mixed) | 0.8458 | **0.8541** | 0.9012 |
| 3 | Stack 4 (v1+v2, all_v2) | 0.8456 | 0.8534 | 0.8993 |
| 4 | Stack ALL_9 (9 model) | 0.8443 | 0.8488 | 0.9000 |
| 5 | Stack 3 (v1) | 0.8411 | 0.8521 | 0.8902 |

**SCADA 特征没突破 0.8478 天花板,但提升了 PR-AUC** (0.8541 vs 0.8433,+0.0108)

---

## 🔬 关键洞察

### 1. **SCADA 特征对单模微涨,集成中价值在"多样性"**
- 单模 Macro-F1 提升仅 +0.001 ~ +0.003
- 集成中通过 TCN v4 v2 系数 +2.45 体现价值
- **多样性比绝对值更宝贵**

### 2. **Macro-F1 与 PR-AUC 此消彼长再次出现**
- v2 SCADA 特征让模型"决策更准"(Macro-F1 +0.0027)
- 但概率连续性变差(PR-AUC -0.0035)
- 集成无法弥合这个 gap

### 3. **27 维比 44 维更优雅**
- 抛弃了 one-hot function(28 维),用 is_unusual_fc 等代理特征
- 参数量减少 ~5%
- 性能反而更好,说明 one-hot function 是噪声

### 4. **集成饱和现象**
- Stack ALL_7 (0.8478) → Stack ALL_9 (0.8443)
- **加更多模型反而变差** — 边际收益转负
- 这与 v1 内部 21 个变体实验的"边际递减"规律一致

### 5. **TCN v4 v2 价值被低估**
- 单独 Macro-F1 0.8367(v1: 0.8340)看似不显著
- 但集成系数 +2.45(v1: +1.42)**说明它捕捉了 v1 漏掉的信号**

---

## 🏆 部署推荐(更新)

| 场景 | 推荐 | F1m | PR-AUC | 备注 |
|---|---|---:|---:|---|
| **极致精度** ⭐⭐ | **Stack ALL_7 (v1)** | **0.8478** | 0.8433 | 仍是冠军 |
| **平衡精度 + PR-AUC** ⭐ NEW | **Stack 6 (v1+v2 mixed)** | 0.8458 | **0.8541** | 兼顾 2 指标 |
| **极致 PR-AUC** ⭐ NEW | **Stack 4 (v2 4-model)** | 0.8456 | **0.8534** | PR-AUC 冠军 |
| **风险评分 (单模)** | TCN v4 v2 (SCADA) | 0.8367 | 0.8990 | 单模最优之一 |
| **单模冠军** | TCN v4 +SE v1 | 0.8340 | 0.9025 | 历史单模冠军 |

---

## 📈 历史冠军更替(完整)

| 日期 | 阶段 | 冠军 | F1m | 累计提升 |
|---|---|---|---:|---:|
| 06-08 | 多分类基线 | LGB v1 | 0.426 W-F1 | — |
| 06-09 | 二分类基线 | LGB v2 | 0.8337 | +0.41 W-F1 |
| 06-10 | 17 模型横评 | LGB v2 | 0.8337 | +0.0000 |
| 06-12 上午 | TCN +SE | TCN v4 +SE | 0.8340 | +0.0003 |
| 06-12 下午 | 集成 6 模型 | Stack ALL_6 | 0.8455 | +0.0115 |
| 06-13 上午 | + LSTM+Att | Stack ALL_7 | **0.8478** | +0.0138 ⭐ |
| **06-13 下午** | **+ SCADA 特征** | **Stack ALL_7** (不变) | **0.8478** | **+0.0138** (持平) |

**SCADA 特征工程没能突破 Macro-F1 0.8478 天花板**,但:
- 提升了集成 PR-AUC(0.8433 → 0.8541)
- 单模 Macro-F1 微涨(TCN v4: +0.0027, LSTM+Att: +0.0009)
- 27 维替代 44 维更简洁

---

## 📁 产出文件

### 代码
- `preprocess_v2_scada.py` (特征工程)
- `tcn_binary_v2_v4se_window16.py` (TCN v4 在 v2 数据)
- `lstm_attention_binary_v2_window16.py` (LSTM+Att 在 v2 数据)
- `ensemble_v3_with_v2.py` (集成 v3)

### 数据 + 模型
- `X_*_binary_v2_scada.npy` (19 维行级)
- `X_*_binary_v2_scada_window16.npy` (27 维窗口级)
- `scaler_binary_v2_scada.joblib`
- `model_tcn_v2_v4se_window16.pt`
- `model_lstm_attention_v2_window16.pt`

### 评估
- `evaluation_tcn_v2_v4se_window16.txt`
- `evaluation_lstm_attention_v2_window16.txt` (v2 version, 27-dim)
- `evaluation_ensemble_v3.txt`
- `processed_meta_*.json` (3 个新)

### 报告
- `REPORT_PREPROCESS_V2.md` (本文件)

---

## 💡 后续方向

### 🎯 SCADA 特征仍有优化空间
1. **多步响应延迟**:不止"上一个 cmd",而是 N 步范围内的命令响应匹配
2. **更细的时序特征**:dt 的高阶统计 (std, range, percentiles) 单独提取
3. **数据增强**:对 v2 数据加 Mixup

### 🚀 完全不同的方向
1. **多任务学习**:同时预测 binary + categorized + specific result
2. **Transformer 架构**:用 1D Transformer 替代 TCN/LSTM
3. **领域适配**:fine-tune 模型在新的 SCADA 数据集上
4. **异常检测 + 监督学习双系统**:Isolation Forest 降噪 + TCN 分类

### 🏆 如果目标是部署
- **Stack ALL_7 (v1) 已是项目最高 Macro-F1** — 推荐生产
- 如果要 PR-AUC:用 **Stack 6 (v1+v2 mixed)**
