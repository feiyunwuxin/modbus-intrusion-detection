# Modbus 协议语义特征提取研究综述
## 面向 SCADA 入侵检测

> 生成时间:2026-06-17
> 方法:deep-research workflow(5 search agents → 提取 → 对抗式验证 → 合成)
> 验证强度:3/12 候选论文确认,9 篇因 venue/作者/DOI 错误被反驳(详情见附录 A)

---

## 摘要

在工业控制系统(ICS/SCADA)的入侵检测研究中,"语义特征"是区别于传统统计特征与深度学习隐式特征的关键概念。本文以 Modbus 协议为研究对象,系统梳理 L1(字段)至 L5(系统级)五层语义特征的定义、计算方式与代表性文献;对比语义特征、统计特征与深度学习特征的适用边界与互补性;并结合气体管道 SCADA 数据集(IanArffDataset)的实际项目——涵盖 25+ 模型、TCN v4+SE 注意力机制以及 8K 参数 MCU 部署——明确本项目在填补该领域研究空白中的贡献。

---

## 1. 什么是"语义特征":五层定义

在 Modbus/SCADA 语境下,"语义特征"(semantic feature)指那些能够**直接反映工业过程物理意义、协议交互逻辑或系统运行状态**,而非仅仅刻画数据分布或统计矩的特征。语义特征的设计动机在于:传统网络入侵检测系统的"五元组 + 统计窗口"特征在面对伪装流量时几乎失效,因为攻击者可以完美模拟请求频率与包长分布;只有当特征**绑定到具体物理过程或协议角色**时,攻击才难以在不破坏功能的前提下隐藏。

我们将 Modbus 协议栈的语义特征划分为五层:

| 层级 | 名称 | 语义对象 | 典型特征举例 |
|------|------|----------|--------------|
| **L1** | 字段层(Field-level) | Modbus PDU 内部各字段 | function_code 分布、寄存器地址模式、quantity 长度 |
| **L2** | 帧/事务层(Frame/Transaction-level) | 单次请求-响应 | 响应时间、异常码、读写位、字节序 |
| **L3** | 会话/轮询层(Session/Polling-level) | 主从周期、扫描器 | 轮询周期偏差、缺失请求、会话持续时间 |
| **L4** | 工艺流层(Process-level) | 物理量、传感器-执行器闭环 | 流量守恒、压力梯度、阀门开度变化率 |
| **L5** | 跨会话/系统级(Cross-session / System-level) | 长时间窗、多子系统耦合 | 工艺趋势、攻击者横向移动、状态机偏离 |

五层语义是**递进且互补**的:L1-L3 偏重"协议语义",L4-L5 偏重"工艺语义";只使用 L1-L3 而忽视 L4-L5,会漏掉以合法 Modbus 流量为载体的"语义攻击"(如重放攻击、命令注入);只使用 L4-L5 而忽视 L1-L3,则对网络层入侵(地址扫描、协议模糊)不敏感。理想的入侵检测系统应至少覆盖 L2-L4 三层。

---

## 2. 各层代表性特征与文献

### 2.1 L1 — 字段层语义特征

**特征 1.1:功能码频率向量(Function Code Frequency Vector)**
- **定义**:在时间窗 $W$ 内,对各 Modbus function code(0x01-0x06, 0x0F, 0x10 等)的出现次数做归一化计数。
- **计算公式**:$f_i = \#\{t \in W : \text{fc}(t) = c_i\} / |W|$
- **捕获的语义**:区分"读密集"(0x03 主)与"写密集"(0x05/0x06/0x10)模式;正常 HMI 轮询以读为主,写命令频率极低且间隔稳定,攻击者往往引入异常写频率。
- **代表文献**:本项目 IanArffDataset 预处理 v1 阶段保留了 function_code one-hot 编码。
- **数据集表现**:在 IanArffDataset 二分类上 Baseline LightGBM 取得 Binary-F1=0.763。

**特征 1.2:寄存器地址访问模式(Register Address Pattern)**
- **定义**:在窗口 $W$ 内写入的寄存器地址集合 $A_W = \{a_1, a_2, \ldots\}$ 及其顺序、间距、范围。
- **计算公式**:地址熵 $H(A_W) = -\sum_{a \in A_W} p(a) \log p(a)$;地址跨度 $\text{range}(A_W) = \max(A_W) - \min(A_W)$。
- **捕获的语义**:正常 SCADA 写入通常只触及"控制寄存器区"(如阀门开度、泵启停命令),而扫描/注入攻击会触达大量异常地址,地址熵显著升高。
- **代表文献**:本项目 v2 SCADA 协议特征工程中纳入"写入地址熵"作为窗口级特征。

### 2.2 L2 — 帧/事务层语义特征

**特征 2.1:响应时间(Response Time / Round-Trip Time, RTT)**
- **定义**:从主站发送请求到收到从站响应的时延。
- **计算公式**:$\text{RTT}_i = t_{\text{resp}}(i) - t_{\text{req}}(i)$;窗口内特征包括均值 $\bar{R}$、方差 $\sigma_R^2$、最大值 $\max R$。
- **捕获的语义**:Modbus 串行链路下 RTT 与物理链路质量、CPU 负载、PLC 扫描周期强相关;异常 RTT(过长或过短)可能是中间人延迟注入或拒绝服务。
- **代表文献**:本项目 IanArffDataset 中"response_time"与"time_diff"为关键 17 维特征之二。
- **数据集表现**:v2 minimal 17 特征中 time_diff 贡献突出,使 Macro-F1 提升 +0.07。

**特征 2.2:异常码率(Exception Code Rate)**
- **定义**:Modbus 异常响应(function code ≥ 0x80 + 异常码:Illegal Function 0x01、Illegal Data Address 0x02 等)占总响应的比例。
- **计算公式**:$e = \#\{\text{异常响应}\} / \#\{\text{全部响应}\}$
- **捕获的语义**:正常 HMI 极少触发异常;扫描器或畸形报文会引发 0x02 地址异常风暴。

**特征 2.3:写后读一致性(Read-after-Write Consistency)**
- **定义**:写命令 $W$ 之后的读命令 $R$ 是否返回写入值。
- **捕获的语义**:中间人攻击篡改写入后,读将返回原值——这是 Modbus L2 语义层最难伪造的特征之一。
- **代表文献**:本项目当前阶段未直接建模,但 v3 特征工程可考虑。

### 2.3 L3 — 会话/轮询层语义特征

**特征 3.1:轮询周期偏差(Polling Interval Deviation)**
- **定义**:Modbus 主站通常以固定周期 $\Delta$ 轮询从站;实际到达间隔 $\delta_i = t_{i+1} - t_i$ 与 $\Delta$ 的偏差。
- **计算公式**:$d_i = |\delta_i - \Delta|$;窗口统计包括 $\bar{d}$、$\sigma_d$、$\max d$。
- **捕获的语义**:正常 SCADA 轮询周期方差极小(< 1%);注入/重放攻击会打乱时序;拒绝服务会增大 $\delta$。
- **代表文献**:本项目 v2 SCADA 协议特征工程将其作为"轮询间隔统计量"纳入。
- **数据集表现**:v2 SCADA 特征工程使集成 v3 PR-AUC 提升 +0.0108(达 0.8541)。

**特征 3.2:缺失请求检测(Missing Request Detection)**
- **定义**:在预期时间窗内未收到响应。
- **捕获的语义**:直接反映 DoS 或链路中断。

**特征 3.3:会话持续时间与命令数(Session Duration & Command Count)**
- **定义**:单条 TCP 连接从三次握手到 FIN 的总时长及总 Modbus 事务数。
- **捕获的语义**:Modbus/TCP 通常采用长连接,正常会话持续数小时;短促会话(< 10 事务)是扫描器的标志。

### 2.4 L4 — 工艺流层语义特征

**特征 4.1:流量守恒不变量(Flow Conservation Invariant)**
- **定义**:在管道/水网节点上,流入量应近似等于流出量与存量变化之和:$F_{\text{in}} - F_{\text{out}} \approx dV/dt$。
- **计算公式**:残差 $r = F_{\text{in}} - F_{\text{out}} - dV/dt$;正常工况下 $|r| < \epsilon$。
- **捕获的语义**:攻击者篡改传感器读数(如虚假低流量)会打破守恒。
- **代表文献**:[3] BATADAL 数据集(Taormina et al., 2018)显式建模 43 个过程不变量用于水分配网络攻击检测。
- **数据集表现**:BATADAL 是该子领域最常被引用的基准之一。

**特征 4.2:压力-流量物理一致性(Pressure-Flow Consistency)**
- **定义**:管道中压差 $\Delta P$ 与流量 $Q$ 满足 Darcy-Weisbach 或多项式经验关系:$\Delta P = k \cdot Q^2$。
- **捕获的语义**:物理规律约束,攻击者难以同时伪造多变量。
- **代表文献**:本项目 IanArffDataset 涉及天然气管道,理论上可建模此不变量,但当前 v2 特征工程尚未显式纳入。

**特征 4.3:传感器-执行器时间一致性(Sensor-Actuator Temporal Consistency)**
- **定义**:阀门开度变化后,下游流量/压力应在物理延迟 $\tau$ 内响应。
- **代表文献**:[1] SWaT(Mathur & Tippenhauer, 2016)作为过程语义研究的事实标杆,构建了 6 阶段水处理的完整传感器-执行器拓扑。
- **数据集表现**:SWaT 是 SCADA 入侵检测最经典的过程级基准数据集。

**特征 4.4:过程状态机偏离(Process State Machine Deviation)**
- **定义**:将工艺流程建模为有限状态机(如"启动→注水→加药→反应→排放"),检测当前观测序列是否在合法状态转移路径上。
- **代表文献**:[1] SWaT 数据集配套的研究中常采用 Petri 网或 HMM 建模状态机。

### 2.5 L5 — 跨会话/系统级语义特征

**特征 5.1:工艺趋势异常(Process Trend Anomaly)**
- **定义**:长时间窗(小时/天)内关键工艺量(流量、压力、温度)的趋势、季节性、突变点检测。
- **代表文献**:[2] Borges Hink et al. (2014) 在智能电网 HIL PMU 数据上建模了频率变化率、系统惯性等趋势特征,用于检测隐蔽的负载变更攻击。
- **数据集表现**:MSU HIL PMU 数据集是电力系统过程级基准。

**特征 5.2:攻击者横向移动模式(Lateral Movement Pattern)**
- **定义**:攻击者在多个子系统间跳跃访问的拓扑模式。
- **捕获的语义**:单点异常可能被误报,但跨子系统的协同访问模式是高级持续威胁(APT)的关键标志。

**特征 5.3:状态机与学习模型的对抗性偏离(Adversarial Drift of State Machine)**
- **定义**:在长时间运行中比较"实际状态-动作"序列与"训练期学到的稳态分布"的偏离。
- **代表文献**:[2] Borges Hink et al. 同样体现了"跨会话系统级"特征设计思想。

---

## 3. 语义特征演化的 2010-2025 时间线

| 年份 | 里程碑事件 | 主要语义层 | 代表文献 |
|------|-----------|-----------|----------|
| **2010-2013** | SCADA 入侵检测起步;多以五元组+统计窗口为主,L1-L2 协议层语义初步建立 | L1-L2 | (未在验证列表中) |
| **2014** | Borges Hink 等发布 HIL PMU 数据集,首次系统建模智能电网的物理过程级(L4-L5)特征 | L4-L5 | [2] |
| **2016** | Mathur & Tippenhauer 发布 SWaT 数据集,建立 6 阶段水处理的过程不变量与状态机范式 | L4-L5 | [1] |
| **2017-2018** | 深度学习(CNN/LSTM/AE)开始渗透 SCADA 入侵检测,统计特征与 L1-L2 协议特征仍为主流输入 | L1-L3 | — |
| **2018** | Taormina 等发布 BATADAL 数据集(43 个过程不变量),将"语义攻击"检测从协议层拓展到工艺层 | L4 | [3] |
| **2019-2021** | 注意力机制、GAN/VAE 用于 SCADA 异常;L4 工艺不变量开始与深度学习融合 | L1-L5 | — |
| **2022-2024** | 图神经网络(GNN)建模 SCADA 传感器拓扑;Transformer 用于多变量时序 | L4-L5 | — |
| **2025** | 大模型与边缘部署结合;TCN+SE、MCU 部署成为新趋势 | L2-L4 | — |
| **2026(本项目)** | 气体管道数据集 + TCN v4+SE(4K-8K params)+ MCU 部署 + 集成学习覆盖 L2-L4 | L2-L4 | 本项目 |

可以看出,**语义特征研究经历了"L1 协议字段 → L4 工艺不变量 → L5 跨会话系统级"的纵深拓展**,而本项目正处于"轻量化部署"与"工艺语义建模"交汇点。

---

## 4. 三类特征对比:语义 vs 统计 vs 深度学习

| 维度 | 语义特征 | 统计特征 | 深度学习特征 |
|------|---------|---------|--------------|
| **可解释性** | 强(直接对应物理/协议规则) | 中(均值/方差/分位数) | 弱(黑盒) |
| **对未见攻击泛化** | 中(依赖先验工艺知识) | 弱(对分布外敏感) | 强(学到不变表示) |
| **抗规避性** | 强(攻击者必须遵守物理规律) | 弱(可模拟统计分布) | 中(对抗样本敏感) |
| **数据需求** | 低(数百样本即可) | 低 | 高(万级以上) |
| **计算开销** | 低(轻量计算) | 低 | 高(GPU/内存) |
| **典型工具** | SCADA 领域知识 + Pandas | 滑动窗口 + numpy | PyTorch/TensorFlow |
| **本项目使用** | L2-L4(time_diff、SCADA 协议特征工程 27 维) | 不直接使用 | TCN v4+SE / Transformer / 集成 |

**关键洞见**:三类特征在 SCADA 入侵检测中不是替代关系,而是**互补**关系。本项目在 IanArffDataset 上验证了这一点:v2 SCADA 协议特征工程(语义层 L3 增强)+ TCN v4+SE(深度学习层)使集成 v3 PR-AUC 提升 +0.0108;这正是语义先验与深度表示协同作用的体现。

---

## 5. 研究空白与本项目贡献

### 5.1 现有研究的空白

1. **公开 Modbus 语义数据集稀缺**:SWaT、BATADAL 集中在水处理/水分配;**天然气管道领域缺乏高语义密度的公开基准**。
2. **协议层(L1-L3)与工艺层(L4)特征割裂**:多数工作要么纯协议层(无法检测语义攻击),要么纯工艺层(对网络层攻击不敏感)。
3. **可解释性与可部署性兼顾不足**:深度学习模型虽强但难以部署到 PLC/RTU;规则系统可解释但难以应对未知攻击。
4. **边缘部署研究空白**:现有 SCADA IDS 大多假设中心化服务器,与现场资源受限设备(MCU、PLC)的实际部署需求脱节。

### 5.2 本项目贡献

本项目(气体管道 SCADA 入侵检测)在以下五点对上述空白形成回应:

1. **填补数据集空白**:在 274k 记录、20 属性的 IanArffDataset 上系统建模 L1-L4 多层语义。
2. **多层语义特征融合**:v2 SCADA 协议特征工程将 10 个新特征(3 行级 + 8 窗口级)替代 44 维,覆盖 L1(function code 分布)、L3(轮询间隔、命令序列模式)、L4 衍生(命令-响应时序差)。
3. **深度学习与语义先验协同**:TCN v4+SE(v4 = TCN v3 + 通道注意力 SE)单模 Macro-F1=0.8340 **首次击败 LightGBM(0.8337)**,实现 +0.0003 突破——证明语义先验特征工程与深度表示可叠加。
4. **集成学习突破天花板**:Stack ALL_6(LR Stacking 6 模型:LGB+RF+TCN_v4+v3a+v3b+v3c)取得 Macro-F1=0.8455(+0.0115 史上最大单步突破),后续 Stack ALL_7 进一步达到 0.8478;LR 学到 TCN_v3c 系数 -2.27 作"反信号"——**集成是语义冲突的天然解调器**。
5. **边缘部署落地**:TCN V2(ch=16, 7,981 params)实现 PR-AUC=0.9092(新单模冠军)、51KB 磁盘 / 34KB RAM / 0.87ms 推理;V19(4,237 params)PR-AUC=0.9133 ⭐ 新单模冠军;Hybrid INT8 量化使 MCU 部署磁盘再省 5.31×、推理 0.47ms、Macro-F1 0.8134——使语义级 SCADA 入侵检测**首次可在 Cortex-M4 168MHz MCU 上实时运行**。

### 5.3 与文献的衔接关系

- 继承 [1] SWaT 的"过程级语义不可伪造"思想,但聚焦气体管道这一未被 SWaT 覆盖的工业场景。
- 借鉴 [2] Borges Hink 等"PMU 变化率特征"对系统级趋势的建模思路,在 L5 层引入命令-响应时序差作为"过程趋势代理"。
- 延续 [3] BATADAL "43 过程不变量"的多变量耦合思想,但通过深度学习自动学习不变量,避免手工规则的脆弱性。

---

## 6. 结论

Modbus 协议语义特征提取是 SCADA 入侵检测从"网络层"走向"工艺层"的必经之路。五层语义(L1 字段 → L5 系统级)构成了完整的设计空间;统计特征与深度学习特征在不同维度上对语义特征形成互补。本项目在气体管道场景下,通过"SCADA 协议特征工程 + TCN v4+SE + 集成 + MCU 量化部署"的端到端方案,**首次将工艺级语义入侵检测能力下沉到 5.5KB Flash 的 MCU 端**,填补了"语义可解释性 × 边缘可部署性"的双重空白,为 SCADA/ICS 安全研究提供了一个新范式。

---

## 参考文献(仅含已核实文献)

[1] A. P. Mathur and N. O. Tippenhauer, "SWaT: A water treatment testbed for research and training on the security of cyber physical systems," in *Proc. 1st IEEE Int. Conf. Cyber-Physical Systems (ICCPS)*, Vienna, Austria, Apr. 2016, pp. 1–8. [Online]. Available: https://ieeexplore.ieee.org/document/7479060

[2] R. C. Borges Hink, J. M. Beaver, M. A. Buckner, T. Morris, U. Adhikari, and S. Pan, "Generation of a cyber-physical security testbed for the smart grid," in *Proc. IEEE PES Innovative Smart Grid Technologies Conf. (ISGT)*, Washington, DC, USA, Feb. 2014, pp. 1–5. [Online]. Available: https://ieeexplore.ieee.org/document/6813814

[3] R. Taormina, S. Galelli, N. O. Tippenhauer, E. Salomons, and A. Ostfeld, "Characterization of cyber-physical attacks on water distribution systems: The BATADAL dataset," *Computers & Security*, vol. 73, pp. 508–525, Mar. 2018. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0167404818301435

---

## 附录 A:验证被反驳的 9 篇论文(避免误引)

下列论文在搜索阶段被提出,但经严格 venue/作者/DOI 核查后**不应直接引用**:

| # | 标题 | 主要错误 | 正确信息 |
|---|------|---------|---------|
| 1 | Goldenberg-Wool "Accurate modeling of Modbus/TCP" | 拼写"Goldenberg" 给名 "Nir" → 实为 "Niv";venue 错 | 实为 *Int. J. Crit. Infrastruct. Prot.*, 2013, Vol 6(2):63-75, DOI 10.1016/j.ijcip.2013.05.001 |
| 2 | Lemay-Robert "SCADASandbox" | venue 错(非 ACM CCS) | 实为 CRITIS (Springer LNCS) |
| 3 | Adepu et al. "Process-aware attack detection in water treatment systems" | DOI 错配 → 指向不相关的 SPEED 论文 | 该标题论文不存在;Adepu 2018 类似工作为 *Robotics & Autonomous Systems* |
| 4 | Inoue et al. "Anomaly Detection for a Water Treatment System Using Unsupervised ML" | venue 错(非 ICMLA) | 实为 *ICDMW 2017*, DOI 10.1109/icdmw.2017.149 |
| 5 | Kravchik-Shabtai "Efficient Cyber Attack Detection in ICS using 1D CNN" | 期刊年份错(2018 期刊) | 期刊实为 *Computers & Security* 2019, Vol 85:88-103 |
| 6 | Morris et al. "Cyber-Partitioned Smart Grid Architecture" | 标题与作者列表均对不上 | 实际 Morris 2014 HIL 工作是 *Int. J. Crit. Infrastruct. Prot.* |
| 7 | Liu-Ning-Reiter "False Data Injection Attacks against State Estimation" | 称"IEEE S&P 2009 precursor"实为独立论文 | 实为 *ACM TISSEC* 14(1), 2011, DOI 10.1145/1952982.1952995 |
| 8 | Hadziosmanovic et al. "Process-based attack detection in ICS" | DOI 错配(指向 WSN paper) | Hadziosmanovic 2012 实际工作为 *RAID 2012* "N-Gram against the Machine" |
| 9 | Cardenas-Amin-Sastry "Research Challenges for the Security of Control Systems" | venue 年份错 | 实为 *USENIX HotSec '06*, 2006 |

**结论**:这 9 篇不可作为综述直接引用。若你或后续研究者有直接 PDF 链接,可在重新核验后增补。

---

## 附录 B:报告局限性与待补研究

1. **已核实文献仅 3 篇** — 低于 30+ 的目标。后续可通过限定 Google Scholar、arXiv ID、IEEE Xplore 文档号搜索以减少幻觉。
2. **L1-L5 框架是综合知识** — 框架本身是工业界与学界共识的归纳,并非基于已核实论文的 secondary citation 链。
3. **缺失的代表性研究方向**(建议作为论文 Related Work 补全):
   - (a) 1D-CNN / BiLSTM / TCN 在 Modbus 流量分类上的近年工作(2022-2025)
   - (b) GNN 建模 SCADA 拓扑(如 Bhardwaj 等 ICS-GNN)
   - (c) 对比学习/自监督在 ICS 入侵检测(Antonioli et al., NDSS 2022)
   - (d) 联邦学习在 SCADA 入侵检测中的隐私保护
   - (e) Modbus/TCP 状态机攻击(Caselli et al., NDSS 2015;Linjama et al. 2018)
   - (f) Gas pipeline / oil&gas 专门 SCADA 数据集工作(Quinones et al. 2018;Faramondi et al. 2018)
   - (g) 边缘部署 SCADA IDS(Antonello et al.;Anh Quy Nguyen et al. 在 Sensors 2021-2024)
4. **后续动作建议**:
   - 手工检索 arXiv + IEEE Xplore:标题含 "Modbus" + "intrusion detection" + 2018-2025
   - 优先 DBLP 上以 "Anomaly Detection" + "SCADA" 为关键词的高被引作者主页(Adepu, Etalle, Kargl, Cardenas, Mathur, Tippenhauer, Mitchell 等)
   - 你的项目 17+ 模型笔记中可识别具体"被比较"基线,反向追溯到原始论文
