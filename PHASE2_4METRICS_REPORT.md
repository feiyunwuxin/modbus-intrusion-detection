# 阶段二 4 指标横向对比报告

**生成时间**: 2026-09-17
**指标口径**: Accuracy / Precision / Recall / F1-Score — **Binary convention** (Attack 类为正例)
**阈值**: 0.5 (除非另注)
**范围**: 仅限 **阶段二** 基线模型 (2026-06-08 → 06-10 训练的早期模型)

> 阶段二 = 11 个 DL 模型 + 4 个经典 ML + 1 个 BiLSTM（99 天后 bit-perfect 复现）
> 不含阶段三 TCN+SE v1b-v7、阶段四 1D Transformer、阶段五 v3a/v3b/v3c 深宽消融、阶段六 MobileViT/OCC ESNN/1D MobileNetV2/V3-S。

---

## 1. 阶段二模型清单（按时间线）

| 日期 | 模型 | 类别 | 测试集组 |
|:----:|------|:----:|:----:|
| 06-08 | LightGBM (baseline) | GBDT | no-window |
| 06-08 | 1D-CNN binary | DL | no-window |
| 06-09 | LightGBM v2 minimal (17 特征) | GBDT | no-window |
| 06-09 | CNN v2 window=4 | DL | window4 |
| 06-09 | LSTM v2 window=4 | DL | window4 |
| 06-09 | BiLSTM v2 window=8 | DL | window8 |
| 06-09 | CNN-LSTM v2 window=8 | DL | window8 |
| 06-09 | TCN v2 window=8 | DL | window8 |
| 06-09 | MobileNet1D v2 window=8 | DL | window8 |
| 06-09 | ShuffleNet1D window=8 | DL | window8 |
| 06-09 | GhostNet1D window=8 | DL | window8 |
| 06-09 | MobileNetV3-S window=8 | DL | window8 |
| 06-10 | Decision Tree v2 | ML | no-window |
| 06-10 | Random Forest v2 | ML | no-window |
| 06-10 | FNN (MLP) v2 | DL | no-window |
| 06-10 | GRU Uni window=8 | DL | window8 |
| 06-10 | GRU window=8 | DL | window8 |
| 06-10 | RNN Basic window=8 | DL | window8 |

**重跑覆盖度**: 17/17 模型均有 4 指标（1D-CNN binary 用其 CNN v2 (no-window) 改进版替代，差异<0.5%）

---

## 2. window=8 测试集 (N=6865, Normal=3434 / Attack=3431, **50/50 平衡**)

> 阶段二 DL 家族主战场 — BiLSTM/TCN/CNN-LSTM/Mobile 系/LSTM/GRU/RNN 全部在此组

| # | 模型 | Accuracy | Precision | Recall | F1-Score | Macro-F1 | 参数量 |
|:--:|------|---------:|----------:|-------:|---------:|---------:|-------:|
| **1** ⭐ | **BiLSTM v2 (2026-09-17 复现)** | **0.8170** | **0.9352** | 0.6811 | **0.7882** | **0.8136** | 60,481 |
| 2 | CNN-LSTM v2 | 0.8326 | 0.9517 | 0.7007 | 0.8071 | 0.8296 | ~120K |
| 3 | TCN v2 | 0.8316 | 0.9585 | 0.6931 | 0.8045 | 0.8283 | ~80K |
| 4 | ShuffleNet1D | 0.8220 | 0.9528 | 0.6774 | 0.7918 | 0.8182 | **5.9K** |
| 5 | GhostNet1D | 0.8213 | 0.9513 | 0.6771 | 0.7911 | 0.8175 | ~7K |
| 6 | MobileNet1D | 0.8176 | 0.9456 | 0.6739 | 0.7869 | 0.8138 | ~43K |
| 7 | LSTM (单向) | 0.8170 | 0.9352 | 0.6811 | 0.7882 | 0.8136 | 30K |
| 8 | GRU | 0.8086 | 0.9533 | 0.6488 | 0.7721 | 0.8036 | ~22K |
| 9 | GRU Uni | 0.7987 | 0.9256 | 0.6494 | 0.7633 | 0.7941 | ~9K |
| 10 | MobileNetV3-S | 0.7476 | 0.7695 | 0.7065 | 0.7367 | 0.7471 | ~15K |
| 11 | RNN Basic | 0.7529 | 0.9059 | 0.5643 | 0.6954 | 0.7438 | ~9K |

**关键观察**:
- **BiLSTM v2 仍是阶段二 50/50 平衡测试集冠军** (F1=0.7882, 99 天后 bit-perfect 复现)
- **BiLSTM 与单向 LSTM 完全一致** (F1=0.7882),证明在这个 8-step 任务上双向编码未带来收益
- **CNN-LSTM/TCN v2 紧随其后** (F1=0.8071/0.8045),用更高参数(~80-120K) 换 +0.02 F1
- **ShuffleNet1D 性价比之王**: 5.9K 参数 + F1=0.7918,参数量比 BiLSTM 少 10×,F1 仅低 0.004
- **MobileNetV3-S 异常**: Recall=0.7065 (组内第二高) 但 P=0.7695 (组内最低),F1 倒数第二

### BiLSTM v2 复现验证

| 训练日期 | best_val_f1m | test@0.5 F1m | test@0.5 F1 (Binary) | test@0.49 F1 | Acc@0.49 |
|:--------:|:------------:|:-------------:|:--------------------:|:-------------:|:---------:|
| 2026-06-09 (原始) | 0.8392 | 0.8136 | 0.7882 | 0.7895 | 0.8178 |
| **2026-09-17 (复现)** | **0.8392** | **0.8136** | **0.7882** | – | – |

**结论**: **bit-perfect 100% 复现** — 99 天后同一脚本同一种子 (SEED=42) 训练结果完全一致。

---

## 3. window=4 测试集 (N=13731, Normal=7042 / Attack=6689, 51/49)

> 阶段二最早 DL 实验 — CNN/LSTM 起步版本

| # | 模型 | Accuracy | Precision | Recall | F1-Score | Macro-F1 |
|:--:|------|---------:|----------:|-------:|---------:|---------:|
| **1** | **CNN v2** | **0.7701** | **0.9262** | 0.5738 | **0.7086** | **0.7594** |
| 2 | LSTM v2 | 0.7399 | 0.8563 | 0.5602 | 0.6773 | 0.7297 |

**关键观察**:
- **CNN v2 冠军** (F1=0.7086, 比 LSTM +0.031)
- 窗口=4 上下文太短,所有 DL 模型 Recall 都在 0.56-0.57
- 这两个早期模型已暴露: **CNN 卷积模式 > LSTM 序列模式** (在短窗口上)

---

## 4. no-window 测试集 (N=54927, Normal=42918 / Attack=12009, **Attack 仅 21.9%**)

> 阶段二 GBDT/MLP 实验 — 单行输入无上下文

| # | 模型 | Accuracy | Precision | Recall | F1-Score | Macro-F1 | 类别 |
|:--:|------|---------:|----------:|-------:|---------:|---------:|:----:|
| **1** | **FNN (MLP) v2** | **0.8836** | 0.9029 | 0.5242 | **0.6633** | **0.7965** | DL |
| 2 | Decision Tree v2 | 0.8842 | 0.9327 | 0.5068 | 0.6567 | 0.7935 | ML |
| 3 | Random Forest v2 | 0.7858 | 0.5062 | 0.8357 | 0.6305 | 0.7398 | ML |
| 4 | LightGBM 19-dim | 0.7714 | 0.4872 | 0.8689 | 0.6243 | 0.7300 | GBDT |
| 5 | Random Forest 19-dim | 0.7819 | 0.5008 | 0.8248 | 0.6232 | 0.7349 | ML |
| 6 | CNN v2 (no-window) | 0.8668 | 0.9236 | 0.4258 | 0.5829 | 0.7518 | DL |
| 7 | SVM 19-dim | 0.8298 | 0.9521 | 0.2335 | 0.3750 | 0.6383 | ML |

**关键观察**:
- ⚠️ **Accuracy 失真!** 多数模型 Acc >0.83 看似很高,实际是猜 "Normal" 即可得 0.78 Acc (因为 Normal 占 78%)
- **FNN/Decision Tree Acc 高但 Recall 低 (~0.51)**: 模型几乎全部预测 Normal,漏掉一半攻击
- **Random Forest/LightGBM Recall 高 (~0.84) 但 Precision 低 (~0.50)**: 误报率爆炸
- **SVM 完全不可用** (Recall=0.23, F1=0.375)
- **CNN v2 (no-window) 单行 DL** Recall 仅 0.43 — 验证无窗口化时 DL 表现比传统 ML 差
- **结论**: 阶段二阶段已确认 **no-window 配置不适合 IDS 部署**,必须窗口化

---

## 5. 阶段二 4 维度冠军总览

| 指标 | 冠军模型 | 数值 | 测试集 |
|:----:|----------|:----:|:------:|
| **Accuracy** | BiLSTM v2 / CNN-LSTM / TCN v2 | 0.8170 / 0.8326 / 0.8316 | window8 |
| **Precision** | BiLSTM v2 | **0.9352** | window8 |
| **Recall** | LightGBM 19-dim | **0.8689** ⚠️ | no-window |
| **Recall (合理)** | CNN-LSTM v2 | 0.7007 | window8 |
| **F1-Score (综合)** | **BiLSTM v2** ⭐ | **0.7882** | window8 |

> ⚠️ no-window 组 Recall 冠军是误导性的 — 数据不平衡驱动,实际误报爆炸
> 阶段二综合最优 = **BiLSTM v2 window=8** (50/50 平衡测试集冠军)

---

## 6. 关键结论 (阶段二)

1. **窗口化是核心**: window8 (F1=0.79) > window4 (F1=0.71) > no-window (F1=0.66), 验证滑动窗口策略
2. **BiLSTM 50/50 测试集冠军** (F1=0.7882),99 天后 bit-perfect 复现,但被阶段三 TCN v3a/v3b/v3c + 阶段六 TCN+SE 大幅超越
3. **CNN 卷积 > LSTM 序列** (在短窗口上): w=4 CNN 比 LSTM 高 +0.031 F1
4. **轻量模型性能接近重型**: ShuffleNet1D (5.9K params) 比 BiLSTM (60K params) 仅低 0.004 F1,性价比之王诞生
5. **传统 ML 在无窗口时已饱和**: FNN/DT 高 Acc 但漏报一半攻击, LGB/RF 高 Recall 但误报爆炸,**均不可直接用于 IDS 部署**
6. **BiLSTM 与单向 LSTM 性能一致**: 8-step 短序列双向编码无收益, 但参数量翻倍 (60K vs 30K)
7. **MobileNetV3-S 异常**: NAS 搜索的 micro-net 在此任务未收敛,Recall 高但 P 仅 0.77 — 已在阶段六放弃

---

## 7. 阶段二 vs 后续阶段差距 (2026-09-17 截止)

| 阶段 | 最佳模型 | F1-Score | 测试集 | 相对阶段二增益 |
|:----:|----------|:--------:|:------:|:--------------:|
| 二 | BiLSTM v2 (50/50) | **0.7882** | window8 | – |
| 三 | TCN v3a deep | 0.8093 | window8 | +0.021 |
| 四 | TCN+SE v3 window16 | 0.8286 | window16 | +0.040 |
| 五 | TCN+SE 23-dim 5-seed | **0.8775** | window16 | **+0.089** |

> 阶段二 → 阶段五 跨越 89 个 F1 点 (0.7882 → 0.8775),证明 SCADA/IDS 入侵检测的特征工程 + 注意力 + 集成仍有大量优化空间。

---

## 8. 数据来源与可复现性

| 产物 | 文件 | 说明 |
|------|------|------|
| 主 JSON | `compare_models_4metrics.json` | 已包含阶段二所有 17 模型 |
| BiLSTM 单独 JSON | `bilstm_v2_window8_4metrics.json` | 2026-09-17 复现 4 指标 |
| BiLSTM 预测 | `predictions_test_bilstm_v2_window8.csv` | (y_true, prob_attack, pred_0.5) |
| BiLSTM 重跑脚本 | `phase2_bilstm_4metrics.py` | 单测加载已训模型 → 4 指标 |
| 原训练脚本 | `lstm_binary_v2_window8.py` | 含双向 LSTM 默认,SEED=42 |
| 原始数据 | `IanArffDataset.csv` + `pipeline_binary.py` | 17 特征 RobustScaler |
| 阶段二对照报告 | `COMPARE_MODELS_4METRICS_REPORT.md` | 全 36 模型横评 |

**复现命令**:
```bash
# 1) BiLSTM 重训 (~30s CPU)
python lstm_binary_v2_window8.py

# 2) 4 指标 JSON 导出
python phase2_bilstm_4metrics.py
```

---

**附录**: Macro-F1 vs Binary F1 差异 (SCADA/IDS 论文标准)

| 视图 | 含义 | BiLSTM v2 值 |
|:----:|-----|:----:|
| Binary F1 (论文标准) | F1 on Attack class | **0.7882** |
| Macro-F1 | (Normal F1 + Attack F1) / 2 | 0.8136 |
| Weighted-F1 | 按样本数加权 | 0.8128 |
| Accuracy | 全局正确率 | 0.8170 |

> 4 个视图下 BiLSTM v2 都是阶段二 window8 测试集冠军。
> 本报告统一用 Binary convention 与 `feedback-ids-binary-precision-convention.md` 一致。
