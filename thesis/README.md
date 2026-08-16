# 硕士论文 LaTeX 模板 — 教学诊断版

> **面向 SCADA 工控系统的轻量化入侵检测方法研究**
> 基于 TCN-SE 架构与极限压缩量化

## 文件清单

```
thesis/
├── main.tex                       # 主文件（含 \pitfall/\todo/\grade 宏）
├── README.md                      # 本文件
├── student-pitfalls.md            # ★ 23 类学生常犯错误清单（老师评阅对照表）
├── teacher-guide.md               # ★ 教师使用说明
│
├── chapters/                      # 章节源文件
│   ├── cover.tex                  # 封面
│   ├── declaration.tex            # 诚信声明
│   ├── abstract_zh.tex            # 中文摘要
│   ├── abstract_en.tex            # 英文摘要
│   ├── chapter1_diagnostic.tex    # ★ 第 1 章（教学版完整示范，含 6 处诊断）
│   ├── chapter2.tex               # 第 2 章（占位 + 2 处诊断）
│   ├── chapter3.tex               # 第 3 章（占位 + 4 处诊断）
│   ├── chapter4.tex               # 第 4 章（占位 + 3 处诊断）
│   ├── chapter5.tex               # 第 5 章（占位 + 3 处诊断）
│   ├── chapter6.tex               # 第 6 章（占位 + 1 处诊断）
│   ├── acknowledgment.tex         # 致谢
│   ├── publications.tex           # 科研成果
│   └── appendix_a.tex             # 附录 A：C99 推理代码
│
├── figures/                       # 论文图片
├── tables/                        # 论文表格
├── code/                          # 论文代码
└── references/
    └── refs.bib                   # BibLaTeX 参考文献数据库
```

## 编译方法

### 推荐：XeLaTeX + biber

```bash
cd thesis
xelatex main
biber main
xelatex main
xelatex main
```

### 依赖宏包

- 编译引擎：XeLaTeX（必须，CTeX 中文支持）
- 文献管理：biber（GB/T 7714-2015）
- 颜色框：tcolorbox（用于 \pitfall / \todo / \grade）
- 图表：graphicx, booktabs, subcaption
- 代码：listings

### 编译错误排查

| 错误 | 原因 | 解决 |
|---|---|---|
| `! Undefined control sequence` | 缺宏包 | 检查 preamble 加载 |
| `! LaTeX Error: File 'xxx.sty' not found` | 缺宏包 | `tlmgr install xxx` |
| 中文乱码 | 没用 XeLaTeX | 切换编译引擎 |
| 参考文献为 `[?]` | 没跑 biber | `biber main` |

## 核心宏命令

### 1. `\pitfall{编号}{描述}` — 学生常犯错误

```latex
\pitfall{P-1.4}{学生常写成"本文提出了一种新方法"——必须每个创新点都有
可量化的指标提升。}
```

红色警示框。**草稿版显示，最终版自动隐藏**（通过 `\finaltrue/finalfalse` 切换）。

### 2. `\todo[负责人]{描述}` — 待补充内容

```latex
\todo[学生]{补充 5 折交叉验证结果}
```

黄色待办框。

### 3. `\grade[等级]{评语}` — 老师评分批注

```latex
\grade[B]{方法合理但实验规模偏小}
```

蓝色评分框，**默认不显示**，老师评阅时手动启用。

### 4. `\good{描述}` — 优秀示例

```latex
\good{本段使用"震网"、"乌克兰电网"、"Colonial Pipeline"三个递进案例}
```

绿色文字，标出"应当这样写"的范例。

## 三种使用场景

### 场景 A：老师评阅学生初稿
1. 学生提交时使用 `\finalfalse`（带红色警示）
2. 老师打开 `student-pitfalls.md`
3. 按 23 类逐项对照评阅，给出修改意见
4. 学生修改后改为 `\finaltrue`，输出最终版

### 场景 B：学生自查
1. 编译时使用 `\finalfalse`
2. 对照红色 `\pitfall{}` 检查自己是否犯该错误
3. 解决后改为 `\finaltrue`

### 场景 C：答辩前准备
1. `\finaltrue` 输出干净版
2. 按 `student-pitfalls.md` 第九节"答辩准备"准备 5 个高频问题的预案

## 23 类错误速查

详见 [`student-pitfalls.md`](./student-pitfalls.md)：

| 类别 | 数量 | 难度 |
|---|---|---|
| 整体框架 | 5 | ⭐⭐⭐ |
| 摘要目录 | 4 | ⭐⭐ |
| 第 1 章绪论 | 6 | ⭐⭐ |
| 第 2 章相关工作 | 4 | ⭐⭐ |
| 第 3-5 章方法 | 4 | ⭐⭐⭐ |
| 图表公式 | 3 | ⭐ |
| 写作排版 | 3 | ⭐⭐ |
| 参考文献 | 2 | ⭐ |
| 答辩准备 | 2 | ⭐⭐⭐ |

**最易扣分 TOP 5**：
1. P-1.4 创新点无量化数据（-2 分）
2. P-3.2 消融实验不完整（-1.5 分）
3. P-3.1 无基线对比（-1.5 分）
4. P-1.8 文献综述堆砌（-1 分）
5. P-3.3 无统计显著性检验（-1 分）

## 论文基本信息

- **题目**：面向 SCADA 工控系统的轻量化入侵检测方法研究
- **副标题**：基于时序卷积网络与极限压缩量化
- **作者**：XXX（待填）
- **导师**：XXX 教授（待填）
- **学科专业**：XXX（待填）
- **字数**：约 65,000 字（不含附录）
- **页数**：约 70 页正文 + 10 页附录
- **完成时间**：2026 年 6 月

## 三大创新点回顾

1. **基于 SCADA 协议语义的 27 维特征工程** — 领域知识驱动的特征设计
2. **TCN-SE 轻量入侵检测模型** — DL 首次击败 LGB（Macro-F1 0.8340）
3. **极限压缩与混合精度量化** — 4K 参数 V19 + 5.5KB MCU 部署

## 答辩高频问题

| 评委可能问 | 建议回答 |
|---|---|
| 为什么不用更长的窗口？ | w=8 是 Pareto 拐点；附消融数据 |
| Transformer 真的不如 TCN？ | 在 8 步序列下是的；附对比表 |
| INT8 量化误差多大？ | 11 层 SNR ≥ 38dB；附逐层分析 |
| 实际工控场景能用吗？ | 第 5 章 MCU 实测 0.47ms @ Cortex-M4 |

## License

本模板仅供学术使用。
