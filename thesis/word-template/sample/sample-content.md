# Word 模板范文示例（第 1 章）

> **本文件展示在 Word 中实际使用诊断框的效果**。
> 每个标注对应 LaTeX 版本 `chapters/chapter1.tex`。

---

# 第 1 章 绪论

## 1.1 研究背景与意义

> ⚠️ **【学生常见错误 P-1.1】** 数据无出处。学生常引"据统计"、"研究表明"等模糊表述，无可追溯的数据来源。

随着工业互联网（Industrial Internet of Things, IIoT）与人工智能技术的快速发展，SCADA（Supervisory Control and Data Acquisition，监控与数据采集）系统作为国家关键基础设施的神经中枢，正面临日益严峻的网络安全挑战 [1]。根据国家工业信息安全发展研究中心发布的报告，2024 年全球工业领域网络攻击事件同比增长 27%，其中针对能源、水务等 SCADA 系统的定向攻击占比超过 35%。

SCADA 系统广泛应用于燃气管道、电力调度、轨道交通、自来水处理等关键领域，其安全性直接关系到国计民生与公共安全。2010 年伊朗"震网"病毒、2015 年乌克兰电网攻击、2021 年美国 Colonial Pipeline 输油管道勒索事件，无一不揭示了 SCADA 系统一旦失守可能造成的灾难性后果。

> ✓ **【优秀示例】** 本段使用"震网"、"乌克兰电网"、"Colonial Pipeline"三个递进案例，从国家→地区→企业层面论证，逻辑递进清晰。

> 📝 **【待补充（学生）】** 增加 1-2 个中文案例，如"2020 年某省电网工控系统入侵事件"。

在此背景下，入侵检测（Intrusion Detection System, IDS）作为 SCADA 安全的最后一道防线，已成为学术界与工业界共同关注的研究热点。

> ⚠️ **【学生常见错误 P-0.5】** 摘要写成"引言缩写"。本章第一节是背景，不是摘要副本；摘要应单独有四段式结构。

## 1.2 国内外研究现状

> ⚠️ **【学生常见错误 P-1.2】** 文献堆砌。学生常罗列 20+ 篇文献却不评价。建议：每段文献综述末尾必须有 1 句评价。

现有 SCADA 入侵检测研究大致可分为三类：基于规则与签名的方法、基于传统机器学习的方法、基于深度学习的方法。

### 1.2.1 基于规则与签名的方法

早期研究多采用专家系统与特征库匹配 [4]。该方法对已知攻击检测准确率高、可解释性强，但完全依赖人工维护规则库，对未知攻击（zero-day）几乎无能为力。

> ✓ **【优秀示例】** 末句评价指出局限性，自然引出下一小节。

### 1.2.2 基于传统机器学习的方法

随着 KDD Cup 99、NSL-KDD、UNSW-NB15 等入侵检测基准数据集的发布，基于传统机器学习的方法成为研究主流。支持向量机 [5]、随机森林 [6]、XGBoost [7] 等模型在多个数据集上取得了 F1 ≥ 0.85 的检测性能。

### 1.2.3 基于深度学习的方法

> ⚠️ **【学生常见错误 P-1.3】** 缺乏协议层切入。这是本领域最大空白。SCADA 与传统 IT 网络的根本区别在协议层；学生若不强调"协议语义"，会和通用 IDS 论文混为一谈。

近年来，深度学习在时序建模领域的突破为入侵检测带来了新思路。文献 [8] 首次将 1D-CNN 应用于网络流量分类，取得了比 SVM 高 3% 的准确率；文献 [9] 提出 LSTM-IDS，利用长短期记忆网络捕捉时序依赖；文献 [10] 进一步引入注意力机制，在多个公开数据集上达到 SOTA。然而，上述工作大多基于通用网络入侵数据集（KDD/NSL-KDD/UNSW-NB15），鲜有针对 SCADA 协议语义（如 Modbus、DNP3）的专门优化。

## 1.3 现有研究不足

综合分析国内外研究现状，当前 SCADA 入侵检测领域存在以下不足：

> ✓ **【优秀示例】** 三个"不足"恰好对应后文三个"创新点"——这是 1:1 映射的最佳实践。

1. **特征体系缺乏协议语义**：现有研究多沿用通用网络入侵特征，未充分利用 Modbus 主-从交互结构等协议层先验；
2. **深度学习模型未稳定超越树模型**：在多个公开数据集上，LightGBM / XGBoost 仍占据 SOTA，DL 的优势尚未充分释放；
3. **模型部署可行性研究不足**：现有工作多报告 GPU 推理延迟，对 MCU/PLC 等受限设备的部署研究较少。

## 1.4 研究内容与创新点

> ⚠️ **【学生常见错误 P-1.4】** 创新点无量化数据。"创新点"是论文最核心的部分，但学生常写成"提出了一种新方法"——必须每个创新点都有可量化的指标提升。

> ⚠️ **【学生常见错误 P-1.5】** 创新点对应关系模糊。"对应第 X 章"必须明确写出，且在第 X 章开头必须有"本章对应创新点 N"的回指。

针对上述不足，本文以 IanArffDataset 数据集为研究对象，开展从数据特征、模型架构到边缘部署的全栈研究，主要创新点如下：

1. **基于 SCADA 协议语义的领域特征工程**（对应第 3 章）。分析 Modbus 4 行主-从交互结构，提出 27 维协议语义特征，使所有模型 Macro-F1 一致提升 0.2--0.7 个百分点；
2. **TCN-SE 轻量入侵检测模型**（对应第 4 章）。在 27 种时序模型的横向对比基础上，融合 SE 通道注意力，TCN-SE 在测试集取得 Macro-F1 = 0.8340，首次在该任务上超越 LightGBM；
3. **面向 MCU 的极限压缩与混合精度量化**（对应第 5 章）。提出 4,237 参数的 TCN V19 变体与权重 INT8 / 激活 FP32 混合方案，在 Cortex-M4 平台上实现 5.5 KB 模型、0.47 ms 推理。

## 1.5 论文组织结构

> ✓ **【优秀示例】** 每章一句话点明"做什么"，避免冗长。

本文章节安排如下：

- 第 1 章 绪论：介绍研究背景、国内外现状与本文创新点；
- 第 2 章 相关理论与技术基础：SCADA 入侵检测任务定义、TCN 与注意力机制原理、量化基础；
- 第 3 章 SCADA 协议特征工程：详细阐述 27 维特征的设计动机与对比实验；
- 第 4 章 TCN-SE 入侵检测模型：模型架构、27 模型横评、消融实验；
- 第 5 章 极限压缩与混合精度量化：V19 变体设计、INT8 量化方案、MCU 部署实测；
- 第 6 章 总结与展望。

---

> ✏️ **【老师评语 [等级：B]】** 第 1 章基本结构完整，创新点量化指标清晰。但 1.1 节案例可增加 1 个中文事件；1.2 节末缺少对各方法的统一对比表（建议增加表 1-1）。

---

**参考文献**（本节示例，对应 Word 中的参考文献章节）：

[1] ZHOU T, LI Y, WU L. A comprehensive survey of SCADA systems: Architecture, vulnerabilities and defense strategies[J]. Computers & Security, 2022, 118: 102734.

[2] AL-GARADI M A, MOHAMED A, AL-ALI A K. Deep learning approaches for IoT security: A systematic survey[J]. IEEE Internet of Things Journal, 2023, 10(11): 9501-9520.

[4] ROESCH M. Snort: Lightweight intrusion detection for networks[C]//LISA. 1999, 99(1): 229-238.

[5] MUKKAMALA S, JANOSKI G, SUNG A. Intrusion detection using neural networks and support vector machines[C]//IJCNN. IEEE, 2002: 1702-1707.

[6] ZHANG J, ZULKERINE M, HAQUE A. Random-forests-based network intrusion detection systems[J]. IEEE Transactions on Systems, Man, and Cybernetics, Part C, 2008, 38(5): 649-659.

[7] DHALIWAL S S, NAHID A, ABBAS R. Effective intrusion detection system using XGBoost[J]. Information, 2018, 9(7): 149.

[8] WANG W, ZHU M, ZENG X, et al. Malware traffic classification using convolutional neural network for representation learning[C]//ICOIN. IEEE, 2017: 712-717.

[9] KIM J, KIM H. An effective intrusion detection classifier using long short-term memory with gradient descent optimization[C]//ICTC. IEEE, 2017: 1154-1159.

[10] WU Z, ZHANG H, WANG P, et al. Network intrusion detection based on transformer[C]//AEMCSE. IEEE, 2022: 147-152.
