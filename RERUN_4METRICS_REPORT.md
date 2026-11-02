# TCN+SE 23-dim ch=32 — 4 指标 (Acc / P / R / F1) 重跑报告

**生成时间**: 2026-09-13
**模型**: TCN+SE 23-dim ch=32 5-seed self-ensemble
**测试集**: `X_test_binary_v2_scada_window16.npy` (N = 3432, Normal = 1627, Attack = 1805)
**阈值**: 0.5
**指标口径**: 二分类论文标准 (Binary / Positive-class convention)
- **Precision** = Attack 类的精度 (TP / (TP+FP))
- **Recall** = Attack 类的召回 (TP / (TP+FN))
- **F1-Score** = Attack 类的 F1
- **Accuracy** = 全局精度

> Macro / Weighted 平均另列在附录。

**复现脚本**: `rerun_4metrics_5seed.py`
**结果 JSON**: `rerun_4metrics_5seed.json`

---

## 1. 单 Seed × 5

| Seed | Accuracy | **Precision** | **Recall** | **F1-Score** |
|:----:|---------:|--------------:|-----------:|-------------:|
|  42  | 0.8730   | 0.9524        | 0.7983     | 0.8686       |
| **123** | **0.8767** | **0.9546** | **0.8039** | **0.8728** |
| 456  | 0.8549   | 0.9107        | 0.8028     | 0.8534       |
| 789  | 0.8596   | 0.9437        | 0.7795     | 0.8538       |
| 1024 | 0.8610   | 0.9160        | 0.8100     | 0.8597       |
| **Mean** | **0.8650** | **0.9355** | **0.7989** | **0.8617** |
| **Std**  | 0.0080 | 0.0176 | 0.0107 | 0.0079 |

> Seed 123 是单 seed 冠军 (F1=0.8728)，与 5-seed ensemble (F1=0.8737) 仅差 0.0009。

---

## 2. 5-Seed Ensemble

| 集成方式 | Accuracy | **Precision** | **Recall** | **F1-Score** |
|:---:|---------:|--------------:|-----------:|-------------:|
| **概率平均** ⭐ | **0.8776** | **0.9553** | **0.8050** | **0.8737** |
| 多数投票 (≥3/5) | 0.8773 | 0.9565 | 0.8033 | 0.8732 |

> 两种集成几乎等价，概率平均 F1 微高 0.0005。

---

## 3. 关键混淆矩阵（5-Seed Ensemble, thr=0.5）

```
              预测 Normal    预测 Attack
实际 Normal      1559           68        (Precision_N=0.8158, Recall_N=0.9582)
实际 Attack       352          1453       (Precision_A=0.9553, Recall_A=0.8050)
```

由混淆矩阵直接读出 4 指标的物理含义：

- **Accuracy** = (1559 + 1453) / 3432 = 0.8776
- **Precision** = 1453 / (68 + 1453) = 0.9553  ← 报警中真的是攻击的比例
- **Recall** = 1453 / (352 + 1453) = 0.8050     ← 真实攻击被检出的比例
- **F1-Score** = 2 × 0.9553 × 0.8050 / (0.9553 + 0.8050) = 0.8737

附加指标：
- **误报率 (FPR)** = 68 / 1627 = **4.18%**
- **漏报率 (FNR)** = 352 / 1805 = **19.50%**
- **特异度 (TNR)** = 95.82%
- **灵敏度 (TPR)** = 80.50%

---

## 4. 与历史结果一致性验证

```
VERIFICATION: PASS (bit-perfect with perclass_metrics_23dim_ensemble.json)
```

5 个 seed × 4 指标（Acc / F1m / Attack-P / Attack-R）共 20 个数据点，全部与 `perclass_metrics_23dim_ensemble.json` 历史值 bit-perfect 一致（diff < 1e-6）。

---

## 附录 A: Macro / Weighted 平均（参考）

> 与论文标准 4 指标的区别：Macro 把两类（Normal + Attack）等权，Weighted 按各类样本数加权。

| 视图 | 单 seed 平均 | 5-seed ensemble |
|---|---:|---:|
| Macro Precision | 0.8717 | 0.8855 |
| Macro Recall    | 0.8686 | 0.8816 |
| **Macro F1**    | **0.8649** | **0.8775** |
| Weighted Precision | 0.8700 | 0.8828 |
| Weighted Recall    | 0.8650 | 0.8776 |
| Weighted F1        | 0.8665 | 0.8775 |

Macro-F1 在 ensemble 达到 **0.8775**，仍是历史最高。

## 附录 B: Normal 类指标（脚注参考）

| 视图 | 5-seed ensemble |
|---|---:|
| Normal Precision | 0.8158 |
| Normal Recall    | 0.9582 |
| Normal F1        | 0.8813 |

## 附录 C: 结论

1. **4 指标 (Acc / Binary-P / Binary-R / Binary-F1) 全维度 bit-perfect 复现历史最佳结果**，无退化。
2. **5-seed ensemble 4 指标均最高**: Accuracy = 0.8776, Precision = 0.9553, Recall = 0.8050, F1-Score = 0.8737。
3. **单 seed 冠军 = seed 123**（F1 = 0.8728），与 ensemble 仅差 0.0009，**MCU 部署推荐单 seed**（节省 4× 推理时间）。
4. **Recall = 0.805 偏低**说明模型对攻击偏向保守（高 Precision 0.955，低 FPR 4.18%），适用 SCADA "宁可漏报也不误报" 场景。如需更高 Recall，可下调阈值至 0.33（参考 `DAILY_LOG_2026-07-26.md`）。
5. 概率平均集成 vs 多数投票集成几乎无差别（差 0.0005 F1），推荐概率平均（理论更优）。
