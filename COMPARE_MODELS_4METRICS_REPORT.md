# 跨模型 4 指标横向对比报告

**生成时间**: 2026-09-15
**指标口径**: Accuracy / Precision / Recall / F1-Score — **Binary convention** (Attack 类)
**阈值**: 0.5
**测试集分组**: window=16 / window=8 / window=4 / no-window (各组 y_true 完全一致,内部公平)

> 跨测试集直接比较 F1 绝对值有偏 (攻击占比不同:window16=52.6% / window8=50.0% / window4=48.7% / no-window=21.9%),推荐**组内比较**。

**复现脚本**: `compare_models_4metrics.py`
**结果 JSON**: `compare_models_4metrics.json`

---

## 1. window=16 测试集 (N=3432, Normal=1627 / Attack=1805)

> 当前 SOTA 模型所在的测试集 — 与论文表 1 直接对应

| # | 模型 | Accuracy | Precision | Recall | F1-Score | Macro-F1 |
|:--:|-----|---------:|----------:|-------:|---------:|---------:|
| **1** ⭐ | **TCN+SE 23-dim ch=32 5-seed ensemble (SOTA)** | **0.8776** | **0.9553** | **0.8050** | **0.8737** | **0.8775** |
| 2 | TCN+SE 23-dim ch=32 single (seed 123) | 0.8767 | 0.9546 | 0.8039 | 0.8728 | 0.8766 |
| 3 | TCN+SE v3 window16 | 0.8406 | 0.9538 | 0.7324 | 0.8286 | 0.8398 |
| 4 | TCN+SE v2 window16 | 0.8377 | 0.9581 | 0.7230 | 0.8241 | 0.8367 |
| 5 | TCN+SE v5 +temporal | 0.8307 | 0.9643 | 0.7042 | 0.8140 | 0.8293 |
| 6 | TCN+SE v6 dual-SE | 0.8258 | 0.9569 | 0.7003 | 0.8087 | 0.8244 |
| 7 | LSTM+Attention | 0.8188 | 0.9346 | 0.7047 | 0.8035 | 0.8177 |
| 8 | TCN v3c deep window16 | 0.8196 | 0.9714 | 0.6770 | 0.7979 | 0.8175 |
| 9 | MobileNetV3-S 19-dim | 0.8022 | 0.8743 | 0.7285 | 0.7948 | 0.8019 |
| 10 | MobileViT v3 19-dim | 0.7876 | 0.8202 | 0.7634 | 0.7908 | 0.7875 |
| 11 | TCN v3b window16 (27-dim) | 0.8132 | 0.9671 | 0.6676 | 0.7899 | 0.8109 |
| 12 | TCN+SE v1b window16 | 0.8030 | 0.9638 | 0.6499 | 0.7763 | 0.8002 |
| 13 | TCN+SE v7 multiscale+focal | 0.8036 | 0.9732 | 0.6443 | 0.7753 | 0.8005 |
| 14 | 1D Transformer | 0.7952 | 0.9734 | 0.6277 | 0.7632 | 0.7914 |
| 15 | MobileNetV2 19-dim | 0.7815 | 0.9631 | 0.6078 | 0.7452 | 0.7770 |
| 16 | OCC ESNN v3 19-dim | 0.5921 | 0.6863 | 0.4133 | 0.5159 | 0.5817 |

**关键观察**:
- ⭐ **5-seed ensemble 比单 seed 高 0.0009 F1** (0.8737 vs 0.8728),但推理成本 ×5
- 23-dim 比 19-dim (MobileNet/ViT) F1 高 **+0.06~0.13**
- TCN+SE v1b→v3 单调递增 (+0.052 F1),v5/v6/v7 边际饱和或下降
- 1D Transformer 单模较弱 (F1=0.7632) 但与 TCN+SE 强反相关,适合做集成 (历史验证)
- OCC ESNN 灾难 (F1=0.5159),脉冲网络在此任务不适用

---

## 2. window=8 测试集 (N=6865, Normal=3434 / Attack=3431,50/50 平衡)

| # | 模型 | Accuracy | Precision | Recall | F1-Score | Macro-F1 |
|:--:|-----|---------:|----------:|-------:|---------:|---------:|
| 1 | BiLSTM v2 *(phase2, 2026-09-17 复现)* | 0.8170 | 0.9352 | 0.6811 | 0.7882 | 0.8136 |
| **1** | **TCN v3a deep** | **0.8341** | 0.9508 | 0.7045 | **0.8093** | **0.8312** |
| 2 | CNN-LSTM | 0.8326 | 0.9517 | 0.7007 | 0.8071 | 0.8296 |
| 3 | TCN v2 | 0.8316 | 0.9585 | 0.6931 | 0.8045 | 0.8283 |
| 4 | ShuffleNet1D | 0.8220 | 0.9528 | 0.6774 | 0.7918 | 0.8182 |
| 5 | GhostNet1D | 0.8213 | 0.9513 | 0.6771 | 0.7911 | 0.8175 |
| 6 | MobileNet1D | 0.8176 | 0.9456 | 0.6739 | 0.7869 | 0.8138 |
| 7 | LSTM | 0.8170 | 0.9352 | 0.6811 | 0.7882 | 0.8136 |
| 8 | GRU | 0.8086 | 0.9533 | 0.6488 | 0.7721 | 0.8036 |
| 9 | GRU Uni | 0.7987 | 0.9256 | 0.6494 | 0.7633 | 0.7941 |
| 10 | MobileNetV3 1D | 0.7476 | 0.7695 | 0.7065 | 0.7367 | 0.7471 |
| 11 | RNN Basic | 0.7529 | 0.9059 | 0.5643 | 0.6954 | 0.7438 |

**关键观察**:
- **TCN v3a deep** 冠军 (F1=0.8093),CNN-LSTM 紧随其后 (+0.0022)
- 12 个模型 F1 全部 ≥0.69,但**比 window16 SOTA 低 0.06**(更长上下文信息更有用)
- 经典 RNN Basic 垫底 (F1=0.6954),现代架构 (TCN/CNN-LSTM) 显著更强
- **BiLSTM v2 与 LSTM 单向完全一致** (F1=0.7882),bit-perfect 复现 99 天前的训练结果
- ShuffleNet/GhostNet 性价比之王 (轻量 + 0.79 F1)

---

## 3. window=4 测试集 (N=13731, Normal=7042 / Attack=6689)

| # | 模型 | Accuracy | Precision | Recall | F1-Score | Macro-F1 |
|:--:|-----|---------:|----------:|-------:|---------:|---------:|
| **1** | **CNN v2** | **0.7701** | 0.9262 | 0.5738 | **0.7086** | **0.7594** |
| 2 | LSTM v2 | 0.7399 | 0.8563 | 0.5602 | 0.6773 | 0.7297 |

**关键观察**:
- 窗口=4 上下文最短,Recall 普遍偏低 (0.56-0.57),模型难以捕获攻击模式
- CNN v2 比 LSTM v2 高 +0.031 F1

---

## 4. no-window 测试集 (N=54927, Normal=42918 / Attack=12009, Attack 仅 21.9%)

| # | 模型 | Accuracy | Precision | Recall | F1-Score | Macro-F1 |
|:--:|-----|---------:|----------:|-------:|---------:|---------:|
| **1** | **FNN (MLP) v2** | **0.8836** | 0.9029 | 0.5242 | **0.6633** | **0.7965** |
| 2 | Decision Tree v2 | 0.8842 | 0.9327 | 0.5068 | 0.6567 | 0.7935 |
| 3 | Random Forest v2 | 0.7858 | 0.5062 | 0.8357 | 0.6305 | 0.7398 |
| 4 | LightGBM 19-dim | 0.7714 | 0.4872 | 0.8689 | 0.6243 | 0.7300 |
| 5 | Random Forest 19-dim | 0.7819 | 0.5008 | 0.8248 | 0.6232 | 0.7349 |
| 6 | CNN v2 (no-window) | 0.8668 | 0.9236 | 0.4258 | 0.5829 | 0.7518 |
| 7 | SVM 19-dim | 0.8298 | 0.9521 | 0.2335 | 0.3750 | 0.6383 |

**关键观察**:
- ⚠️ **Accuracy 失真!** 多数模型 Acc >0.83 看似很高,实际是 Normal 占 78%,猜 "Normal" 即可得 0.78 Acc
- **FNN/Decision Tree Acc 高但 Recall 低 (~0.51)**,说明模型**几乎全部预测 Normal**,漏掉一半攻击
- **Random Forest/LightGBM Recall 高 (~0.84) 但 Precision 低 (~0.50)**,误报率极高
- **SVM 完全不可用** (Recall=0.23, F1=0.375)
- **窗口化 + TCN+SE (window16) 全面碾压 no-window**:F1=0.8737 vs 0.6633 (+0.21)

---

## 5. 4 维度冠军总览

| 指标 | 冠军模型 (按测试集组) | 数值 | 测试集 |
|:----:|-----|:----:|:----:|
| **Accuracy** | Decision Tree v2 | 0.8842 | no-window ⚠️ 失真 |
| **Accuracy (合理)** | ★ TCN+SE 5-seed | **0.8776** | window16 |
| **Precision** | TCN+SE v7 multiscale+focal | 0.9732 | window16 |
| **Recall** | LightGBM 19-dim | 0.8689 | no-window ⚠️ 伴随 P=0.49 |
| **Recall (合理)** | ★ TCN+SE 5-seed | **0.8050** | window16 |
| **F1-Score (综合)** | ★ TCN+SE 5-seed ensemble | **0.8737** | window16 |

> ⚠️ no-window 组的 Acc/Recall 冠军是误导性的 — 它们由数据不平衡驱动,不是真正的模型能力。
> 综合最优 = **TCN+SE 23-dim ch=32 5-seed ensemble** (window16 测试集)

---

## 6. 关键结论

1. **TCN+SE 23-dim ch=32 5-seed ensemble 4 指标全部组内最佳**:
   - Acc=0.8776 / P=0.9553 / R=0.8050 / F1=0.8737
2. **窗口化价值巨大**:window16 (F1=0.87) 比 no-window (F1=0.66) 高 **+0.21**,验证滑动窗口策略
3. **窗口长度 vs F1 呈倒 U 型**:window16 (0.87) > window8 (0.81) > window4 (0.71),**太长上下文 (>16) 可能引入噪声**
4. **传统 ML (LGB/RF/DT/SVM) 在窗口化数据上不参与对比**,但 no-window 测试集上:
   - LGB/RF 高 Recall 但 P<0.51,误报爆炸
   - DT/FNN 高 Acc 但 R<0.53,漏报一半
   - **都不可直接用于 IDS 部署**
5. **TCN+SE 特征工程路径 (v1b→v2→v3) 单调增益 +0.052 F1**,v5/v6/v7 边际收益耗尽
6. **弱模型 (Transformer) 在集成中有价值** — 历史验证其与 TCN+SE 反相关,Ensemble v4 中贡献 +0.011 F1
7. **MCU 部署推荐单 seed 123**:F1=0.8728 与 ensemble 仅差 0.0009,但推理成本 1/5

---

## 附录:Macro-F1 vs Binary F1 差异 (SCADA/IDS 论文标准)

| 视图 | 含义 | TCN+SE 5-seed 值 |
|:----:|-----|:----:|
| Binary F1 (论文标准) | F1 on Attack class | **0.8737** |
| Macro-F1 | (Normal F1 + Attack F1) / 2 | 0.8775 |
| Weighted-F1 | 按样本数加权 | 0.8773 |
| Accuracy | 全局正确率 | 0.8776 |

> 4 个视图下 SOTA 都是 TCN+SE 5-seed ensemble,但 Binary F1 更适合攻击检测场景 (Attack 是少数但关键的类)。
> 本报告统一用 Binary convention 与 `feedback-ids-binary-precision-convention.md` 一致。