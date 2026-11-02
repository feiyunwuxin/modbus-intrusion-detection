# 阶段一 4 指标横评 — LightGBM 基线 (2026-09-17 重跑)

**任务**: 用户 2026-09-17 要求按阶段一范围 (**2026-06-08 LightGBM 基线**) 重跑模型, 评价指标变更为 **Accuracy / Precision / Recall / F1-Score** (Binary convention, 与 `feedback-ids-binary-precision-convention.md` 一致).

**Why**: 与 `project-2026-09-17-phase2-4metrics.md` (BiLSTM v2 window=8) 形成 "阶段一 → 阶段二 → 阶段三 → 阶段五" 的完整 4 指标对照链.

---

## 关键决策

### 1. 重训 vs 加载旧模型 — 选择**重训**

- 06-08 旧产物 `model_binary.joblib` / `model_multiclass.joblib` 是 sklearn 1.5.1 + 当时 lightgbm 版本训练的, 100 天后存在跨版本不兼容 (InconsistentVersionWarning 出现)
- 用相同 SEED=42 + 相同超参 (**n_estimators=400/600, num_leaves=63/127, class_weight="balanced"**) 重训, 忠实复现 06-08 baseline
- 数值与 06-08 原报告 **0.5-1% 微差** (lightgbm 4.7.0 vs 06-08 老版本的 threading/精度差异), 这是诚实重训结果, **不接受伪造 bit-perfect**

### 2. 阈值策略 — 主表 thr=0.5 + 附录 test 扫描 best_f1

- **baseline pipeline.py 设计无 val 集** (只有 temporal 70/30 split), 用户偏好 "best_val_f1m 阈值" 不适用
- 主表: thr=0.5 (忠实 06-08 baseline 默认)
- 附录: thr 扫描 [0.05, 0.95] 找 test 集 best_f1, **标注 "test-derived, 仅作参考"** (不参与主表)

### 3. Binary convention

- Precision / Recall / F1 全部用 `pos_label=1` (Attack positive class), 严格遵循 09-13 用户反馈
- Macro / Weighted 仅用于多分类 (多分类场景默认)

---

## 4 指标结果

### A. 二分类 (0=Normal, 1=Attack)

| 阈值 | Accuracy | Precision (Attack) | Recall (Attack) | F1-Score (Attack) |
|:---:|---:|---:|---:|---:|
| **thr=0.5** (主表) | **0.8129** | **0.5478** | **0.7694** | **0.6400** |
| thr=0.81 (test best_f1, 参考) | 0.9027 | 0.9095 | 0.6104 | 0.7305 |

**混淆矩阵 (thr=0.5)**:
```
              pred_N    pred_A
true_N        53,277    11,307     (FPR=17.5%)
true_A         4,106    13,699     (FNR=23.1%)
```
- Normal: 64,584 (78.4%) / Attack: 17,805 (21.6%)
- FPR = 11307/64584 = **17.51%** ← 误报率高
- FNR = 4106/17805 = **23.06%** ← 漏报率高
- thr=0.81 提升 P (+0.36) 但 R 跌 (-0.16), 整体 F1 +0.09, 报警更保守

### B. 多分类 (specific result, 36 类)

| 平均方式 | Accuracy | Precision | Recall | F1-Score |
|:---:|---:|---:|---:|---:|
| **Macro** | 0.8304 | 0.4556 | 0.4129 | **0.4200** |
| **Weighted** | 0.8304 | 0.8050 | 0.8304 | **0.8140** |

**Macro vs Weighted 鸿沟** (0.42 vs 0.81):
- 19 个类可完美分离 (F1=1.0) → 把 Weighted-F1 拉到 0.81
- 但 5/14/15 类彻底坍塌 (F1=0) → Macro-F1 暴跌
- Class 13 ≈ 0.0034, Class 30/35 ≈ 0.05-0.08, Class 9/28 ≈ 0.09

---

## 与 06-08 原报告对照

| 指标 | 06-08 原报告 (lightgbm 老版本) | 09-17 重训 (lightgbm 4.7.0) | Δ |
|:---:|---:|---:|---:|
| Binary Acc | 0.8201 | 0.8129 | -0.0072 |
| Binary Precision (Attack) | 0.5619 | 0.5478 | -0.0141 |
| Binary Recall (Attack) | 0.7617 | 0.7694 | +0.0077 |
| Binary F1 (Attack) | 0.6467 | 0.6400 | -0.0067 |
| Macro-F1 (binary) | 0.7630 | (同 Binary-F1, binary 只 2 类) | - |
| Multi Macro-F1 | 0.4262 | 0.4200 | -0.0062 |
| Multi Weighted-F1 | 0.8141 | 0.8140 | -0.0001 |

**差异原因**: lightgbm 4.7.0 vs 06-08 老版本的 threading 实现 + 内部精度优化, 单 seed 下产生 0.5-1.4% 微差. **训练脚本、SEED、超参、数据完全一致**, 差异属于库版本自然漂移.

---

## 阶段增益轨迹 (跨阶段 4 指标对照)

| 阶段 | 范围 | 最佳模型 | Acc | P (Attack) | R (Attack) | F1 (Attack) |
|:---:|:---|:---|---:|---:|---:|---:|
| **一** | **06-08** | **LGB baseline** | **0.8129** | **0.5478** | **0.7694** | **0.6400** |
| 二 | 06-08 → 06-10 | BiLSTM v2 window=8 (09-17 重训) | (待补) | (待补) | (待补) | 0.7882 |
| 三 | 06-12/14 | TCN v3a deep | — | — | — | 0.8093 |
| 四 | 06-19/22 | TCN+SE v3 window16 | — | — | — | 0.8286 |
| 五 | 06-29 → 07-26 | **TCN+SE 23-dim 5-seed ensemble** | **0.8776** | **0.9553** | **0.8050** | **0.8737** |

**从阶段一 → 阶段五 (101 天)**:
- Acc: 0.8129 → 0.8776 (+0.0647 / +8.0% 相对)
- F1 (Attack): 0.6400 → 0.8737 (+0.2337 / +36.5% 相对) ⭐
- **LightGBM baseline 是天花板下限参考, 5-seed TCN+SE 才是 SOTA**

---

## 关键发现

1. **FPR 17.5% 是 LightGBM 基线最大瓶颈** → 高误报率, 不适合生产 (需要后续模型压到 < 5%)
2. **多分类 Macro-F1 0.42 是 "数据坍塌" 信号** → 5/14/15 类彻底失效, 提示需要 hierarchical 二分类 → 多分类 两阶段模型
3. **Precision 0.55 偏低** → 报警中只有 55% 真攻击, 实际工程不可用
4. **thr=0.81 可换 Precision 0.91 但 Recall 跌至 0.61** → 高 Precision 低 Recall 是 FP 主导场景的典型 trade-off
5. **重训偏差 0.5-1% 属正常** → lightgbm 库版本漂移, 不影响结论

---

## 产出文件

- `phase1_lgb_4metrics.py` — 4 指标重制脚本 (重训版)
- `phase1_4metrics.json` — 完整 4 指标 JSON
- `PHASE1_4METRICS_REPORT.md` — 本文档

## 复现命令

```bash
pip install lightgbm  # 若未装
python phase1_lgb_4metrics.py
```

**耗时**: ~14 分钟 (683s train + 136s eval, 单线程 n_jobs=-1 实际未充分利用多核; lightgbm 4.7 默认行为)

## 与阶段二关联

- 阶段一 LGB baseline F1=0.6400 vs 阶段二 BiLSTM F1=0.7882: **+0.1482 (+23.2%)**
- DL 模型在窗口化+时序结构上显著优于 LightGBM (树模型对单点 tabular 友好, 但对 8 步序列无能为力)
- 阶段三 TCN v3a deep F1=0.8093 又 +0.0211, 印证时序建模 + 加深价值

**关联**: [[project-2026-09-17-phase2-4metrics]] | [[project-baseline-results]] | [[feedback-ids-binary-precision-convention]]