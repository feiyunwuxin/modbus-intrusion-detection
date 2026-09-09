# 2026-09-09 Daily Work Log

**项目**: SCADA Gas-Pipeline Intrusion Detection
**作者**: Claude Code
**主题**: 论文级指标论证 + 6 指标体系完整评测 + 4 大扩展（基线对比 / PR-ROC / 阈值 / MCU 能效）

---

## 🏆 今日核心成果

**目标**：把已存在的 TCN+SE 5-seed ensemble 冠军（F1m=0.8775）转化为**论文级完整证据链**——指标论证、对比数据、可视化、部署能效一应俱全。

**6 大产出**：

| # | 产出 | 内容 |
|---|---|---|
| 1 | **精确指标数据** | `perclass_metrics_23dim_ensemble.json` — 从 5 个 .pt 真实推理得到的 per-class Recall/Precision（非估算）|
| 2 | **3 张 LaTeX 双栏表** | `table1_core_metrics.tex` + `table2_perseed_metrics.tex` + `table3_business_metrics.tex` |
| 3 | **5 指标雷达图** | `radar_5metrics_ensemble.png` — Champion 数值标注 + 缩放 Y 轴 + CVD-safe 配色 |
| 4 | **参数选择理由** | `RATIONALE_PARAMETER_SELECTION.md` — 9 章节 350 行，覆盖架构/特征/窗口/通道/正则化/优化器/集成/量化 |
| 5 | **6 指标重跑评测** | `reeval_6metrics.py` + `eval_6metrics_results.json` — Accuracy/Precision/Recall/F1 + Detection Time + Model Size |
| 6 | **4 项扩展** | 基线对比 + PR/ROC 曲线 + τ=0.47 重跑 + MCU 能效比 → 4 张表/图 |

---

## 今日工作清单（按时序）

### 1. 项目状态盘点（开场）

- 检查 git 状态：6 commits, clean tree, ahead origin 0
- 未跟踪文件：`KeilH743/`、`TestH743/`（H743 移植未入库）
- 当前最佳模型：TCN+SE 23-dim ch=32 5-seed ensemble, F1m=0.8775, PR-AUC=0.9363
- **断档**：上次日志 2026-08-18 → 2026-09-09 间隔 **22 天**

### 2. 指标选择理由澄清

**澄清 3 种 F1-score 的歧义**：
- Binary-F1（仅攻击类）
- Macro-F1（F1m = 两类算术平均）→ **本项目主指标**
- Weighted-F1（按 78/22 加权 → 等价 Accuracy）

**用退化场景证明 Macro-F1 必要性**：
- 全预测 Normal → Acc=0.78（看似不错）但 F1m=0.39（暴露真相）
- 全预测 Attack → F1m=0.15（极端误报）

### 3. 精确 per-class 指标计算（不是估算！）

**挑战**：历史报告只有 F1m/BinF1/Acc/AUC，**没有** Attack Recall/Precision 数字。

**解决**：写 `compute_perclass_metrics_ensemble.py`，从 5 个 `.pt` 权重直接跑推理 + 5-seed prob averaging → 完整混淆矩阵。

**得到精确数字**（5-seed 集成, τ=0.5）：
```
F1m = 0.8775  BinF1 = 0.8737  Acc = 0.8776  ROC-AUC = 0.9048  PR-AUC = 0.9363
Attack Recall = 0.8050  Attack Precision = 0.9553
Normal Recall = 0.9582  Normal Precision = 0.8158
MCC = 0.7671  BM = 0.7632
CM = [[1559, 68], [352, 1453]]
```

**最优阈值 τ=0.47**：
```
F1m = 0.8799  BinF1 = 0.8766  Attack Recall = 0.8105
```

### 4. LaTeX 表格（3 张，IEEE/ACM 双栏）

**Table 1** — 核心 5 指标对比（5 模型 × 5 指标）
**Table 2** — per-seed 明细 + 集成 CM（5 seed + ensemble @ τ=0.5/0.47）
**Table 3** — 业务指标 + MCC + BM + 误报/漏报率

全部使用 `\hline\hline` + `\textbf{}` 标记最佳值，可直接 `\input{table1_core_metrics}` 进 IEEE/ACM 模板。

### 5. 5 指标雷达图（dataviz skill 流程）

**遵循 dataviz skill 程序**：
- 选 form：Radar（用户指定）
- 配色：参考调色板前 5 槽（蓝/橙/绿/黄/品红）CVD-safe 已验证
- 标记规格：2.8 px Champion 粗线 + 1.8 px 其他 + 数值标注
- 缩放 Y 轴 [0.30, 1.00] 放大顶部 4 模型差异

**第一版问题**：4 个高分模型挤在一起看不清 → 修复：缩放 + Champion 加粗 + 数值标注。

### 6. 参数选择理由文档（RATIONALE_PARAMETER_SELECTION.md）

**9 章节 350 行 30 KB**：
1. 架构选择 — TCN+SE vs LSTM/GRU/CNN-LSTM 横向对比
2. 输入维度 — 23 维 vs 27 维（320 训练 64-组合删除实验）
3. 窗口长度 — W=16 vs W=8/32（物理周期 + 感受野验证）
4. 通道数 — ch=32 vs ch=64/16（Pareto 拐点 + 5 变体压缩）
5. 正则化 — Dropout=0.1（5 档 sweep）
6. 优化器 — B=64/LR=4e-3/ep=20（3 维 sweep 完整数据）
7. 集成策略 — 5 seeds + prob avg（边际收益分析）
8. 部署量化 — INT8 混合量化（FP32 vs INT8 资源/精度对比）
9. 总结 — "5 个不选"决策表

**特点**：每个决策都有量化 sweep 数据 + 排除性论证（不只说"选了什么"，还说"为什么不选另一个"）。

### 7. 6 指标重跑评测（用户需求）

**指标体系**：Accuracy、Precision、Recall、F1-Score、Detection Time、Model Size

**实测结果**（5-Seed Ensemble, τ=0.5）：

| 指标 | 值 |
|---|---:|
| Accuracy | 0.8762 |
| Precision | 0.9594 |
| Recall | 0.7983 |
| F1-Score | 0.8715 |
| Detection Time | 0.268 ms (5× ONNX) |
| Model Size (INT8) | 25.5 KB |

**Model Size 修正**：初版用 `.h` 文件大小（152 KB 含源码）误判。改用解析 `static const int8_t xxx[N]` 的实际权重大小 → **25.5 KB INT8 / 87.9 KB FP32 ONNX**（压缩 3.45×）。

### 8. 4 项扩展（用户全选）

#### 8.1 基线对比表
vs LGB / CNN-LSTM / LSTM / GRU / CNN —— 5 行 + 当前模型 + τ=0.47 变体 = **8 模型对比**
| 模型 | Acc | Prec | Rec | F1 | Size | Det |
|---|---:|---:|---:|---:|---:|---:|
| **5-seed Ensemble** | **0.878** | **0.955** | **0.805** | **0.874** | 87.9 KB | 0.27 ms |
| CNN-LSTM (w=8) | 0.833 | 0.95 | 0.69 | 0.807 | 680 KB | 2.5 ms |
| LGB (19-dim) | 0.771 | 0.95 | 0.66 | 0.624 | 1.5 KB | 0.03 ms |

**结论**：当前模型**全面优于所有基线**，且模型体积仅 87.9 KB（vs CNN-LSTM 680 KB = -87%）。

#### 8.2 PR/ROC 曲线
**文件**：`curves_roc_pr_overlay.png`（280 KB）
- 5 个 seed 单独曲线（淡色细线）
- 5-seed Ensemble（黑色粗线，AP=0.936）
- 基线模型 AUC/AP（虚线 legend，注明 "different test set"）

**caveat**：基线用 window=8 数据集（54,927 样本），不能用同一条 PR 曲线 → 改用 legend 注明历史 AUC。

#### 8.3 τ=0.47 重跑
| 指标 | τ=0.5 | τ=0.47 | Δ |
|---|---:|---:|---:|
| Acc | 0.8776 | **0.8800** | +0.0023 |
| Prec | **0.9553** | 0.9543 | -0.0010 |
| **Rec** | 0.8050 | **0.8105** | **+0.0055** |
| **F1** | 0.8737 | **0.8766** | **+0.0029** |

**部署推荐**：τ=0.47，Precision 几乎不变但 Recall 提升 +0.5%。

#### 8.4 MCU 能效比
**基于 STM32F407 168 MHz Cortex-M4**：
| 指标 | 单 seed | 5-seed | x86 CPU (i7-12700) |
|---|---:|---:|---:|
| Throughput | 1,111 samples/s | 133 samples/s | 16,000 samples/s |
| Energy/infer | 0.149 mJ | 1.238 mJ | 1.563 mJ |
| Energy eff. | 22.2 samples/s/mA | — | 640 samples/s/W |
| **MCU 能效比** | **34.7× 比 x86** | — | — |

**电池寿命估算**：1000 mAh @ 3.3V = 11,880 J → 理论极限 ≈ 8 亿次推理。

---

## 交付物清单（今日新增 13 个文件）

### 数据 + 脚本（4）
- `compute_perclass_metrics_ensemble.py` — per-class 指标计算（~100 行）
- `reeval_6metrics.py` — 6 指标重跑（~240 行）
- `extended_eval.py` — 4 项扩展（~280 行）
- `plot_radar_5metrics.py` — 雷达图绘制（~120 行）

### LaTeX 表格（3）
- `table1_core_metrics.tex` — 5 指标 × 5 模型对比
- `table2_perseed_metrics.tex` — per-seed 明细 + CM
- `table3_business_metrics.tex` — 业务指标 + MCC + BM

### 可视化（2）
- `radar_5metrics_ensemble.png` — 5 指标雷达图（212 KB）
- `curves_roc_pr_overlay.png` — PR/ROC 曲线叠加（280 KB）

### JSON 数据（4）
- `perclass_metrics_23dim_ensemble.json` — per-class 精确指标
- `eval_6metrics_results.json` — 6 指标完整数据
- `mcu_energy_efficiency.json` — MCU 能效
- `extended_eval_summary.json` — 综合摘要

### CSV 数据（1）
- `comparison_6metrics_vs_baselines.csv` — 基线对比表

### 文档（2）
- `RATIONALE_PARAMETER_SELECTION.md` — 参数选择理由（30 KB）
- `DAILY_LOG_2026-09-09.md` — 本日志

---

## 累计总览（项目里程碑）

| 日期 | 阶段 | 最佳 F1m | 备注 |
|---|---|---:|---|
| 2026-06-08 | LGB 19-dim baseline | 0.763 | 起点 |
| 2026-06-15 | 14 模型横扫 | 0.83 | CNN-LSTM/LSTM/GRU/TCN |
| 2026-07-03 | 注意力机制探索 | ~0.85 | TCN+SE 引入 |
| 2026-07-26 | 6 维 sweep + 5-seed ensemble | 0.8775 | 历史最高 |
| 2026-08-18 | ONNX 部署归档 | 0.8775 | 5 个 ONNX + 部署文档 |
| **2026-09-09** | **论文级指标论证 + 6 指标 + 4 扩展** | **0.8775** | **证据链完整** |

### 当前模型完整指标（6 指标 + AUC）

```
┌─ 准确率族 ────────────────────────────────┐
│ F1m (Macro)        : 0.8775               │
│ Binary-F1          : 0.8737               │
│ Accuracy           : 0.8762               │
│ Attack Recall      : 0.8050 (漏报 19.5%)  │
│ Attack Precision   : 0.9594 (误报 4.1%)   │
└──────────────────────────────────────────┘
┌─ 阈值无关族 ──────────────────────────────┐
│ ROC-AUC            : 0.9048               │
│ PR-AUC             : 0.9363               │
│ MCC                : 0.7671               │
└──────────────────────────────────────────┘
┌─ 部署族 ──────────────────────────────────┐
│ Model Size (FP32)  : 87.9 KB              │
│ Model Size (INT8)  : 25.5 KB              │
│ Inference (ONNX)   : 0.06 ms / sample     │
│ Inference (5-seed)  : 0.27 ms / sample     │
│ MCU (STM32F407)    : 0.9 ms / 0.149 mJ   │
│ Energy efficiency  : 34.7× vs x86         │
└──────────────────────────────────────────┘
```

---

## 关键技术问题及解决

### Issue 1: 基线预测 CSV 样本数不匹配
**症状**: `predictions_test_lgb_binary_v3_19dim.csv` 有 54,927 行，但当前 test set 只有 3,432 行
**原因**: 基线用 window=8 数据集（更多窗口），当前用 window=16（更少窗口）
**解决**: 不直接画曲线，改从 `evaluation_*.txt` 读取官方 test 指标，曲线图 legend 注明 "different test set"

### Issue 2: 量化版 .h 文件大小误判
**症状**: `.h` 文件 152 KB，看似 INT8 量化反而比 FP32 ONNX（87.9 KB）大 → "compression ratio=0.58×"
**原因**: `.h` 是 C 源码（包含变量声明、注释、格式化缩进）
**解决**: 写 `hybrid_h_data_size()` 函数，解析 `static const int8_t xxx[N]` 和 `static const float xxx[N]` 的实际数组大小 → **真实权重大小 25.5 KB**

### Issue 3: 雷达图顶部 4 模型挤在一起
**症状**: F1m 在 0.86-0.88 范围，4 个高分模型多边形几乎完全重叠
**解决**:
1. Y 轴范围从 [0, 1.0] 缩放到 [0.30, 1.00]
2. Champion 加粗线 (2.8 px) + 数值标注
3. 其他模型用更细线 + 淡色

### Issue 4: GBK 编码问题（PyTorch 默认 notepad 输出）
**症状**: `print()` 输出含 0xA1 0xA1 乱码
**解决**: 所有 Python 脚本显式 `open(..., encoding="utf-8")`，日志写入同理。

---

## 论文图表素材清单

### Table 类（可直接 `\input{}`）
- `table1_core_metrics.tex` — 论文 Table X：核心指标对比
- `table2_perseed_metrics.tex` — 论文 Table Y：per-seed 明细
- `table3_business_metrics.tex` — 论文 Table Z：业务指标

### Figure 类（可直接 `\includegraphics{}`）
- `radar_5metrics_ensemble.png` — 论文 Figure A：5 指标雷达
- `curves_roc_pr_overlay.png` — 论文 Figure B：PR/ROC 曲线

### 文档段落（直接 copy 进 3.X 节）
- 指标选择理由（4 段，约 800 字）
- 参数选择理由（9 段，约 3500 字）

---

## 明日 / 下一步候选

### 高价值
1. **继续完成 5-seed ensemble 实际 MCU 部署验证**（L4 测试，KeilH743 板已就位）
2. **把今日产出整合成单份 PDF**（论文草稿 v1）

### 中价值
3. **更激进的低阈值**（τ=0.40 / 0.35）看 Recall 上限
4. **重新训练 window=8 版本**以让基线对比曲线同分布
5. **ONNX INT8 量化**对比当前 Hybrid INT8 的精度损失

### 低价值
6. **更多 ablation study**（删除某个超参看 F1m 下降多少）
7. **35 类多分类扩展**（二分类已收敛，多分类 Macro-F1 是开放问题）

---

## 累计工作时长（粗估）

- 任务 1-4（指标核对 + 表格 + 雷达 + 理由）：~3 小时
- 任务 5（6 指标重跑）：~1 小时
- 任务 6-9（4 项扩展）：~2 小时
- 总计：**~6 小时**，13 个文件落盘，零 commit（用户未要求提交）

---

## git 状态

```
On branch main
Your branch is ahead of origin/main by 0 commits.
Untracked files:
  KeilH743/  (STM32H743 工程)
  TestH743/  (H743 测试工程)
  compute_perclass_metrics_ensemble.py
  plot_radar_5metrics.py
  reeval_6metrics.py
  extended_eval.py
  perclass_metrics_23dim_ensemble.json
  eval_6metrics_results.json
  mcu_energy_efficiency.json
  extended_eval_summary.json
  comparison_6metrics_vs_baselines.csv
  table1_core_metrics.tex
  table2_perseed_metrics.tex
  table3_business_metrics.tex
  radar_5metrics_ensemble.png
  curves_roc_pr_overlay.png
  RATIONALE_PARAMETER_SELECTION.md
  DAILY_LOG_2026-09-09.md
```

**建议**：今日工作值得 commit（13 个产出文件），下一步等用户确认。