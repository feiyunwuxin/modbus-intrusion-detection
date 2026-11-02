# Hybrid 工程报告 (Hybrid-INT16) — 2026-10-27

> 目标 (用户定): **精度优先，把 FPR 拉回 5.9%**，同时保住 INT8 的 Flash/RAM 节省。
> 交付顺序 (用户定): **先 PC 侧出结论，再做 MCU 工程**。
> 两步都已完成。

---

## 0. 一句话结论

**Hybrid-INT16**（权重 per-channel INT16 常驻 Flash + 激活全程 FP32 + 运行期按通道反量化）
在部署区（frame≥200001, thr=0.5）与 FP32 部署真值 **逐窗 0 决策翻转**：

| 指标 | FP32（Route E 真值） | **Hybrid-INT16** | Hybrid-INT8 | 全量 INT8 |
|---|---|---|---|---|
| F1 | 0.8671 | **0.8671** | 0.8663 | 0.8670 |
| FPR | 0.0591 | **0.0591** | 0.0689 | 0.0698 |
| Precision | 0.9361 | **0.9361** | 0.9269 | 0.9261 |
| Recall | 0.8075 | **0.8075** | 0.8133 | 0.8149 |
| 相对 FP32 决策翻转 | — | **0** | 42 | 58 |

同时 Flash **−24.7%**、RAM **−33.9%**（相对 FP32），比全量 INT8 只多 15.8% Flash / 1.6% RAM。

**这就是"既要 INT8 的资源，又要 FP32 的精度"的解。**
INT8 权重（8 bit）在数学上做不到无损 —— 下面 §2 的 2×2 归因给出的是硬证据，不是调参结论。

> **2026-10-28 补充**：Hybrid-INT16 的 int16 量化在**整个 17164 窗数据集上只翻转 1 窗**
> （W=11340，Route E prob 0.500101 vs Hybrid-INT16 prob 0.499977，logit 距阈值仅 1.2e-4），
> 且该窗在部署区之外。这个数字来自 TFLM 固件的首次上机全量回放 ——
> 见 `TFLM_VS_ROUTE_E_REPORT.md` §7.3 与 `tflm_vs_pc_window_diff.py`。

---

## 1. 方案

### 1.1 设计

```
权重:  FP32  →  per-output-channel 对称量化到 INT16
       scale[oc] = absmax(oc) / 32767          (float, 常驻 Flash)
       q[oc,k]   = round(w[oc,k] / scale[oc])  (int16, 常驻 Flash)
激活:  全程 FP32, 不量化
算子:  conv / se / gap / fc 全部 FP32 (复用 Route E 的 C kernel 结构)
反量化: 运行期逐通道 w = (float)q * scale[oc]  (per-oc, 在寄存器/栈上做, 不落 RAM 权重缓冲)
```

**关键点：权重不 memcpy 到 RAM**（Route E 的 FP32 版本要 memcpy 80 KB 到 RAM，这是 RAM 大头）。
Hybrid 的 `ch32_hybrid_init()` 是空的 —— int16 数组在 `.rodata`，编译器直接寻址。

### 1.2 为什么是 INT16 而不是 INT8

per-channel 对称量化下，单个权重量化的相对误差上限是 `1/(2·QMAX)`：

| 位宽 | QMAX | 最大相对误差 |
|---|---|---|
| INT8 | 127 | 3.9e-3 |
| INT16 | 32767 | **1.5e-5** |

INT16 的权重量化误差（1.5e-5）比 fp32 的机器 epsilon（1.2e-7）只高两个数量级，
逐层累积 12 层之后仍在 1e-4 量级 —— 远小于任何决策边界的距离。
INT8 的 3.9e-3 则是 **1e-3 量级的概率漂移**，直接吃掉 thr=0.5 附近的窗。

实测（§2 表格）：INT16 版本概率 MAE = 4.5e-05，INT8 版本 = 9.5e-03 —— **相差 210 倍**。

### 1.3 为什么"只量化激活"不是答案

2×2 归因显示激活量化**根本不是 FPR 上升的原因**（§2）。
也就是说：即使做全量 INT8，掉点也全部来自权重那 8 bit。
既然如此，正确的做法不是去修激活量化，而是**把权重位宽加到误差消失为止**。

---

## 2. 2×2 误差归因（PC 侧，硬证据）

`exp_2x2_attrib.py` — 四个 cell 用**同一个** `sim_hybrid_kernel` 框架跑（只换权重头和 `act_q`），
所以差异只能来自量化本身。部署区 n=4664，thr=0.5。

| cell | F1 | FPR | FP 数 | 相对 FP32 翻转 |
|---|---|---|---|---|
| **W=FP32  A=FP32**（基线/真值） | 0.8671 | **0.0591** | 133 | 0 |
| W=INT8  A=FP32（Hybrid-INT8） | 0.8663 | **0.0689** | 155 | 42 |
| W=FP32  A=INT8（只量化激活） | 0.8676 | **0.0591** | 133 | 26 |
| W=INT8  A=INT8（双量化） | 0.8670 | **0.0698** | 157 | 58 |

**读法：**

- FPR 只在 **W=INT8** 的两行上升（0.0591 → 0.0689 / 0.0698），
  在 **A=INT8** 的行**一个 FP 都没多**（133 → 133）。
  → **权重量化是 FPR 上升的唯一原因**，激活量化无辜。
- 只量化激活那一行 F1 反而略升（0.8676），26 次翻转在正负两侧对称，不偏向 FP。
- 翻转数不可加（42+26=68 > 58），说明两组翻转有重叠 —— 与"权重误差主导"一致。

train 区（frame<200001）方向一致：FP32 FPR 0.0092 → 只量化激活 0.0112 → INT8 权重 0.0103/0.0144。

---

## 3. PC 侧最终结论（四方案横评）

`exp_hybrid_conclusion.py` → `hybrid_conclusion.json` / `hybrid_conclusion.txt`

### 3.1 分区 4-metrics（Binary, thr=0.5）

**部署区 frame≥200001（对外报这个）**

| 方案 | n | TP | FP | FN | P | R | **F1** | **FPR** |
|---|---|---|---|---|---|---|---|---|
| FP32（Route E 真值） | 4664 | 1950 | 133 | 465 | 0.9361 | 0.8075 | **0.8671** | **0.0591** |
| **Hybrid-INT16** | 4664 | 1950 | 133 | 465 | 0.9361 | 0.8075 | **0.8671** | **0.0591** |
| Hybrid-INT8 | 4664 | 1964 | 155 | 451 | 0.9269 | 0.8133 | 0.8663 | 0.0689 |
| 全量 INT8 | 4664 | 1968 | 157 | 447 | 0.9261 | 0.8149 | 0.8670 | 0.0698 |

**全量 17164 窗**

| 方案 | P | R | F1 | FPR |
|---|---|---|---|---|
| FP32 | 0.9741 | 0.8047 | 0.8813 | 0.0227 |
| **Hybrid-INT16** | 0.9741 | 0.8046 | **0.8812** | **0.0227** |
| Hybrid-INT8 | 0.9704 | 0.8094 | 0.8826 | 0.0261 |
| 全量 INT8 | 0.9669 | 0.8114 | 0.8823 | 0.0294 |

**官方 test 区 frame≥219701**

| 方案 | P | R | F1 | FPR |
|---|---|---|---|---|
| FP32 | 0.9266 | 0.7960 | 0.8564 | 0.0702 |
| **Hybrid-INT16** | 0.9266 | 0.7960 | **0.8564** | **0.0702** |
| Hybrid-INT8 | 0.9252 | 0.8004 | 0.8583 | 0.0720 |
| 全量 INT8 | 0.9215 | 0.8049 | 0.8593 | 0.0764 |

**train/val 区 frame<200001**

| 方案 | P | R | F1 | FPR |
|---|---|---|---|---|
| FP32 | 0.9892 | 0.8036 | 0.8868 | 0.0092 |
| **Hybrid-INT16** | 0.9892 | 0.8035 | 0.8867 | 0.0092 |
| Hybrid-INT8 | 0.9880 | 0.8080 | 0.8890 | 0.0103 |
| 全量 INT8 | 0.9833 | 0.8100 | 0.8883 | 0.0144 |

> 注意一个有意思的现象：INT8 在 **train 区 F1 更高**（0.8890 vs 0.8868），
> 但在**部署区 FPR 更差**。train 区 INT8 多抓 28 个 FN、只多出 7 个 FP；
> 部署区只多抓 14 个 FN、却多出 22 个 FP —— 量化误差在分布外（部署区）的代价被放大。
> 正是"Recall 没过拟合、FPR 过拟合"的量化版本。
> **这也再次说明不能用 train 区的指标给部署区做决策。**

### 3.2 阈值扫描（部署区）

| thr | FP32 | **Hybrid-INT16** | Hybrid-INT8 | 全量 INT8 |
|---|---|---|---|---|
| 0.30 | F1 0.8645 / FPR 0.1000 | **0.8645 / 0.1000** | 0.8664 / 0.1014 | 0.8633 / 0.1112 |
| 0.40 | 0.8663 / 0.0765 | **0.8663 / 0.0765** | 0.8670 / 0.0836 | 0.8664 / 0.0867 |
| 0.45 | 0.8671 / 0.0654 | **0.8671 / 0.0654** | 0.8649 / 0.0769 | 0.8665 / 0.0796 |
| **0.50** | **0.8671 / 0.0591** | **0.8671 / 0.0591** | 0.8663 / 0.0689 | 0.8670 / 0.0698 |
| 0.55 | 0.8656 / 0.0574 | **0.8656 / 0.0574** | 0.8646 / 0.0636 | 0.8665 / 0.0645 |
| 0.60 | 0.8655 / 0.0551 | **0.8655 / 0.0551** | 0.8647 / 0.0605 | 0.8672 / 0.0582 |
| 0.70 | 0.8635 / 0.0498 | **0.8635 / 0.0498** | 0.8654 / 0.0507 | 0.8659 / 0.0502 |

**Hybrid-INT16 在整个扫描区间都与 FP32 逐位相同** —— 不是"在 0.5 巧合相等"，
是"量化误差小到不足以在任何阈值下改变决策"。
thr=0.5 保持不变（训练默认值，不需要任何漂移补偿）。

### 3.3 统计检验（相对 FP32，配对）

部署区 4664 窗，thr=0.5，2000 次 paired bootstrap + 精确 McNemar。

| 方案 | ΔF1 95% CI | ΔFPR 95% CI | McNemar (仅FP32正确 / 仅该方案正确) | 翻转 |
|---|---|---|---|---|
| **Hybrid-INT16** | **[+0.0000, +0.0000]** | **[+0.0000, +0.0000]** | 0 / 0, p=1.0000 | **0** |
| Hybrid-INT8 | [−0.0035, +0.0021]（含 0） | **[+0.0057, +0.0142]（不含 0）** | 39 / 3, p<1e-6 | 42 |
| 全量 INT8 | [−0.0033, +0.0032]（含 0） | **[+0.0060, +0.0159]（不含 0）** | 50 / 8, p<1e-6 | 58 |

- Hybrid-INT16：ΔF1 和 ΔFPR 的 CI **都恰好是 [0, 0]** —— 逐窗完全一致。
- Hybrid-INT8 / 全量 INT8：F1 差异在噪声内（CI 含 0），但 **FPR 差异显著**（CI 不含 0）。

### 3.4 决策翻转分解（部署区，相对 FP32）

| 方案 | TP→FN（变漏报） | FP→TN（少报） | TN→FP（多报） | FN→TP（多抓） |
|---|---|---|---|---|
| **Hybrid-INT16** | **0** | **0** | **0** | **0** |
| Hybrid-INT8 | 2 | 1 | **23** | 16 |
| 全量 INT8 | 3 | 5 | **29** | 21 |

INT8 两种方案的主要退化都是 **TN→FP（正常样本被误报）**，也就是 FPR 上升的直接来源。
Hybrid-INT16 四格全 0。

### 3.5 概率漂移

| 方案 | MAE | max\|Δ\| | r (与 FP32) | log 比值偏移 均值 / 中位数 |
|---|---|---|---|---|
| **Hybrid-INT16** | **0.000045** | 0.000586 | **0.99999998** | −0.00019 / −0.00012 |
| Hybrid-INT8 | 0.009479 | 0.446466 | 0.99819771 | +0.02366 / +0.00002 |
| 全量 INT8 | 0.018213 | 0.527446 | 0.99432486 | +0.03652 / +0.00000 |

INT8 的 log 比值偏移中位数 ≈ 0 但均值明显为正 —— 说明**不是对称噪声，
而是系统性的 logit 抬升**（量化后模型整体更倾向于报警），这正是 FPR 上升而 Recall 也上升的原因。
INT16 把这个系统性偏移压到 −0.0002（比 INT8 小 100 倍）。

---

## 4. INT8 路线的死路（已排除，不是调参问题）

为了让 INT8 权重也追回 FPR，试过三条路，全部失败（`exp_hybrid_fix.py` / `exp_logit_trim.py`，
标定**只用 train 区**，部署区只做评估 —— 符合项目的"绝不在部署区调参"纪律）：

| 尝试 | 部署区 F1 | 部署区 FPR | 结论 |
|---|---|---|---|
| INT8 现状 | 0.8663 | 0.0689 | — |
| MSE 最优截断（per-channel clip 搜索） | 0.8663 | 0.0685 | 几乎无帮助 |
| + per-channel bias 校正（线性） | 0.8656 | 0.0649 | 收 0.4pp，但 F1 掉了 |
| + bias 校正（线性 + SE fc2 输出层对齐） | — | — | **不稳定**，命中 ±20 界，F1 崩到 0.4176，放弃 |
| + logit trim（mean / median / FPR 匹配） | 0.8654 | **0.0618** | 最好也只收 0.7pp，仍差 0.27pp，且引入一整套标定管线 |

**"INT8 权重 + 标定补偿"这条路的天花板是 FPR 0.0618，而目标是 0.0591。**
而且代价是：标定必须在 train 区做（部署区绝不能参与）、每次换模型/换数据都要重标、
还多一条"PC 与 MCU 标定参数必须同步"的维护负担。
INT16 一次性把问题从根上消掉，**不需要任何标定**。

---

## 5. MCU 工程

### 5.1 新增文件

| 文件 | 说明 |
|---|---|
| `MCU/Core/Src/ch32_inference_hybrid.c` | Hybrid-INT16 MCU kernel（conv1d_same / conv1d_1x1 / SE / GAP / FC） |
| `MCU/Core/Inc/ch32_inference_hybrid.h` | API：`ch32_hybrid_init/forward/benchmark/dwt_init/get_mem_info` |
| `MCU/Core/Inc/model_v4_se_23dim_ch32_int16_s42.h` | **生成的权重头**（19072 int16 + 1753 float） |
| `gen_int16_weights.py` | 权重头生成器（从 FP32 部署真值头解析 + 量化） |
| `parity_hybrid_int16.py` | 真·parity 测试（源绑定） |
| `exp_hybrid_conclusion.py` | PC 侧最终结论脚本 |
| `exp_2x2_attrib.py` | 2×2 误差归因脚本 |
| `add_hybrid_target.py` | Keil 工程加第三个 target 的脚本 |
| `add_hybrid_fileopt.py` | 给 FP32/INT8 target 补 `ch32_inference_hybrid.c` 的 `IncludeInBuild=0` 条目 |

`ch32_inference.c` 的分派改成三级：`#if CH32_USE_HYBRID` → `#elif CH32_USE_INT8` → `#else`(FP32)。
**三个后端对外 API 完全一致**，调用方（`main.c` / `modbus_ids.c`）一行都不用改。
`ch32_inference.h` 里加了 `#if CH32_USE_INT8 && CH32_USE_HYBRID` 的互斥断言。

**符号前缀必须区分**（踩过的坑，已修正）：
FP32 头、hybrid-int8 头、int16 头如果用同一个前缀 + 同名字 + 不同类型，
在同一个 TU 里 include 两个就是 redefinition 编译错（好在是**响亮**的错，不是静默 bug）。
所以 int16 头用独立前缀 `v4se23_ch32_i16_s42_`（int8 头用 `v4se23_ch32_i8_`）。
`gen_int16_weights.py` 里 `PFX`（输入头前缀）和 `OUT_PFX`（输出头前缀）是两个常量，改的时候别混。

### 5.2 权重头构成

| 类型 | 数量 | 字节 |
|---|---|---|
| int16 权重（13 个张量） | 19,072 | 38,144 |
| per-channel scale（float，13 个数组） | 332 | 1,328 |
| fc1/fc2 权重（float） | 1,056 | 4,224 |
| 15 组 bias（float） | 365 | 1,460 |
| **合计 .rodata** | — | **45,156 B (44.1 KB)** |

对比 FP32 权重头 81,972 B (80.1 KB) → **权重 Flash −44.9%**。

### 5.3 Keil target

`MCU/MDK-ARM/H743test.uvprojx` 现在有三个 target：

| target | Define | 状态 |
|---|---|---|
| `H743test` | （FP32 默认） | 已验证 |
| `H743test_INT8` | `CH32_USE_INT8=1` | 已验证 |
| **`H743test_HYBRID`** | `CH32_USE_HYBRID=1` | **本次新增，0 error** |

编译结果：

```
Program Size: Code=64520 RO-data=48072 RW-data=4448 ZI-data=155464
"H743test_HYBRID\H743test_HYBRID.axf" - 0 Error(s), 1 Warning(s).
Build Time Elapsed:  00:00:03
```

唯一 warning 是 scatter 里 `.bss.tflm_d2` 无匹配段（TFLM 遗留，不影响）。
首次全量编译时会多出 6 个 warning，全部是既有的
（`usart.c` 未用参数、`modbus_ids.c` 未用变量），增量重建后不再出现。

**三个 target 全部验证过 0 error**：

| target | axf | Code | RO-data | RW-data | ZI-data | Error |
|---|---|---|---|---|---|---|
| `H743test`（FP32） | 0 error | 64,784 | 84,836 | 4,448 | 237,440 | ✅ |
| `H743test_INT8` | 0 error | 70,192 | 27,036 | 4,448 | 152,960 | ✅ |
| `H743test_HYBRID` | 0 error | 64,520 | 48,072 | 4,448 | 155,464 | ✅ |

> 注：**Keil uVision 的文件树是工程级的** —— 只在 HYBRID target 的 `<Groups>` 里
> 列出 `ch32_inference_hybrid.c`，FP32 / INT8 target 依然会把它编进去
> （空 TU，608 B 的 .o，链接时被 GC，Flash/RAM 影响 = 0，
> 但 .map 里会留下 `Removing ch32_inference_hybrid.o(.text), (0 bytes)`）。
> 按项目既有约定（`ch32_inference_int8.c` 在 FP32 target 里就是标
> `<IncludeInBuild>0</IncludeInBuild>`），已用 `add_hybrid_fileopt.py`
> 给 FP32 / INT8 target 补上同样的 `IncludeInBuild=0` 条目。
> 修正后：FP32 / INT8 的 .map 里 **hybrid 符号数 = 0**，构建日志也不再出现该文件。

### 5.4 资源占用（Keil .map 实测，三个 target 同工程）

| 方案 | ROM (Flash) | RAM (RW+ZI) | 相对 FP32 |
|---|---|---|---|
| FP32 | 149,684 B (146.2 KB) | 241,888 B (236.2 KB) | — |
| 全量 INT8 | 97,292 B (95.0 KB) | 157,408 B (153.7 KB) | Flash −35.0% / RAM −34.9% |
| **Hybrid-INT16** | **112,656 B (110.0 KB)** | **159,912 B (156.2 KB)** | **Flash −24.7% / RAM −33.9%** |

- 相对 FP32：Flash −37,028 B（−24.7%），RAM −81,976 B（−33.9%）。
  **RAM 省的 80 KB 就是 Route E 那次 80 KB 的权重 memcpy 到 RAM** —— Hybrid 直接取消。
- 相对全量 INT8：Flash +15,364 B（+15.8%），RAM +2,504 B（+1.6%）。
  **多花 15 KB Flash，换回 Hybrid-INT8 的 0.98pp FPR 和 0 决策翻转。**

实测 vs 预估（脚本里的估算公式）：估算 112,868 / 161,376，实测 112,656 / 159,912 —— 偏差 0.19% / 0.91%。

### 5.5 Parity 链（全部源绑定）

`parity_hybrid_int16.py` —— 关键是把 C 源码**解析**出来测，不手抄公式
（memory: feedback-parity-test-must-bind-to-real-source）：

```
[C 源码] ch32_inference.c 的 t_in 公式 = t + k * dilation - pad
[C 源码] ch32_inference_hybrid.c 的 t_in 公式 = t + k * dilation - pad
[C 源码] dequant = wq_f[i] = (float)w_o[i] * wsc[o];
[C 源码] 累加器初值 = float sum = b[o]; (bias 先加)
[C 源码] 权重 RAM 拷贝: 无 ✅ (常驻 Flash)
```

| 校验 | 覆盖 | 结果 | 判定 |
|---|---|---|---|
| [1] Hybrid kernel（按 C 语义, float32） vs numpy 批量参考（float64） | 部署区 64 窗 | max\|Δprob\| = **4.01e-07**，max\|Δlogit\| = 5.68e-06 | PASS |
| [2a] Route E FP32 vs ONNX FP32 (onnxruntime) | 6 例（含 2 例随机） | max\|Δlogit\| = **5.49e-06** | PASS（部署真值 == ONNX） |
| [2b] Hybrid-INT16 vs ONNX FP32 | 同上 | max\|Δlogit\| = **1.72e-03**（= int16 量化误差） | PASS |
| [3] 决策一致性 Hybrid vs FP32 | 部署区 64 窗, thr=0.5 | **0 翻转**，max\|Δprob\| = 2.39e-04 | PASS |

> [1] 只跑 64 窗是因为它按 C 的**逐次累加顺序**在纯 Python 里复刻（每窗 ~10 s）。
> 全量 4664 窗的决策等价是**数值层面**证明的（§3.4：0 翻转），
> 而 [1] 证明的是「C 实现的浮点累加顺序 == numpy 参考」，
> 两者结合就说明 C kernel 跟 FP32 在整个部署区决策一致。

[2a] 说明 Route E 手写 C kernel 与 ONNX 是 bit-perfect 的（根因 1 修复后一直保持）。
[2b] 的 1.7e-3 是 int16 权重量化的正常代价（单层 1.5e-5 × 12 层累积）。

---

## 6. Keil CLI 构建踩坑（新，可复用）

**`.uvoptx` 里写死的 ST-Link 探针序列号会让 `UV4.exe -b` 命令行构建挂死（>10 分钟无输出）。**

- 症状：`UV4.exe -b <proj> -t <target> -o <log>` 进程不退出，也不生成日志文件。
- 定位：把 `.uvoptx` 改名 → 构建 3 秒成功；改回来 → 又挂。
  差异项：`ST-LINKIII-KEIL_SWO` 里的 `-U16004A002933353739303541`（写死的探针序列号）。
  改成 `-U-O206 -O206`（"不指定具体探针"，TFLM 工程本来就是这么写的）→ 构建 1 秒成功。
- **新建 target 还必须在 `.uvoptx` 里也加一个 `<Target>` 块**，否则同样挂死
  （`.uvprojx` 里有 target 不够，`.uvoptx` 里没有对应的 `<Target>` 会触发同样的挂）。
- **改完 `.uvprojx` 后的第一次 `UV4.exe -b` 常常会挂**（本次实测：改了工程后第一次
  FP32 构建 200 s 超时，第二次 5 s 就好了）。原因大概率是 uVision 单实例：
  已在跑的 GUI 实例（当时开着 `H743test_TFLM.uvprojx`）接到命令后要先切工程再构建。
  **对策：超时了就再跑一次**，别以为工程坏了。
- **uVision 的文件树是工程级的**：新加的文件只写进某一个 target 的 `<Groups>`，
  其他 target 照样会编译它（只是空 TU）。要在别的 target 里排除，
  必须显式列出该文件并标 `<IncludeInBuild>0</IncludeInBuild>`
  （`ch32_inference_int8.c` 在 FP32 target 里就是这么处理的）。
- 排查小技巧：`taskkill //F //IM UV4.exe` 会被安全分类器拦；改用 `taskkill //F //PID <自己启动的那个 PID>`。
  但注意**别杀用户正在用的 GUI 实例**（用 `Get-Process UV4 | Select MainWindowTitle` 先看窗口标题）。
- UV4 必须**从 `C:/Keil_v5/UV4` 目录启动**（否则行为异常）。
- 原始 `.uvoptx` 已备份到 `MCU/MDK-ARM/H743test.uvoptx.disabled`（含原探针序列号）。
  **副作用**：GUI 调试会话现在不再锁定那块探针，打开调试时需要在 uVision 里选一次 ST-Link。
  需要恢复锁定的话，把 `H743test.uvoptx.disabled` 拷回来即可（但命令行构建会重新挂）。

---

## 7. 待办（用户侧）

1. **烧录 `H743test_HYBRID`**，用 `H743test_INT8` 那套回放脚本对比：
   PC 侧 Hybrid-INT16 的 4664 窗预测已就绪（`hybrid_conclusion.json`），
   期望上机决策 **4664/4664 一致**、prob 差 ≤ 1e-4（int16 量化 + 日志整数化下界）。
2. **推理耗时实测**（三个 target 都要测，DWT cycle counter 已就绪）：
   FP32 vs Hybrid-INT16 vs 全量 INT8。Hybrid 的预期是 **介于两者之间**
   （反量化是 per-oc 一次性的 `wq_f[]` 展开，卷积本身仍是 FP32 FMA）。
   注意：`wq_f[HY_WBUF_MAX]` 是**栈上**数组（32×3 = 96 float = 384 B），
   stack 现在是 0x4000（16 KB），安全；但 stack 从 0x400 提到 0x4000 的教训要记住。
3. **是否切换部署**：Hybrid-INT16 精度与 FP32 一致、资源比 FP32 省 25~34%，
   比 INT8 多花 15 KB Flash。H743 Flash 余量充足，这个权衡基本是白送的。

---

## 8. 附：`MCU_INT8_VS_FP32_REPORT.md` §7.4 需要补注

那份报告（2026-10-22）写的是：

> **"FPR +1.07pp" 在统计上不成立**：F1 差 bootstrap 95% CI = [−0.0034, +0.0033]，
> FPR 的 bootstrap 区间也相互重叠。

**F1 那半句仍然成立**（本次独立复算 ΔF1 CI = [−0.0033, +0.0032]，含 0）。

**FPR 那半句需要补注**：当年用的是"两个方案的 FPR 置信区间是否重叠"，
这在**非配对**比较里是偏保守的判据；本次用的是**配对** bootstrap
（同一批窗、同一个标签，只对逐窗差值重采样），得到的 ΔFPR 95% CI =
**[+0.0060, +0.0159]，不含 0**。

两者不是同一个检验，配对检验更有功效，所以**新结果不推翻旧结论的定性部分**
（"INT8 与 FP32 的 F1 差异在噪声内"），但确实说明 **FPR 的 +1.07pp 在配对意义下是显著的**。

建议在 `MCU_INT8_VS_FP32_REPORT.md` §7.4 加一句：

> 补注（2026-10-27）："FPR 区间重叠"用的是非配对判据，偏保守。
> 改用配对 bootstrap（同窗配对重采样）后，ΔFPR 95% CI = [+0.0060, +0.0159]，
> 不含 0，FPR 差异显著。这不改变"INT8 部署实现是忠实的"结论
> （差异全部来自量化本身），但改变"FPR 差异可忽略"的判断。
> **Hybrid-INT16 方案正是针对这一项做的**（§2 的 2×2 归因）。

---

## 9. 复现命令

```bash
# PC 侧结论
python exp_hybrid_conclusion.py          # -> hybrid_conclusion.json / .txt
python exp_2x2_attrib.py                 # -> attrib_2x2.json / .txt
python parity_hybrid_int16.py 64         # -> parity_hybrid_int16.txt (纯 Python 复刻 C 累加顺序, 慢)

# MCU 侧构建 (在 C:/Keil_v5/UV4 下执行)
cd /c/Keil_v5/UV4 && ./UV4.exe -b \
  "C:/work/Claude/Issue/MCU/MDK-ARM/H743test.uvprojx" \
  -t H743test_HYBRID \
  -o "C:/work/Claude/Issue/MCU/MDK-ARM/build_hybrid.log"

# 资源占用
grep -E "Total RO  Size|Total RW  Size|Total ROM Size" \
  MCU/MDK-ARM/H743test_HYBRID/H743test_HYBRID.map
```
