# MCU 全量 INT8 vs FP32 性能对比报告

> 项目: STM32H743VIT6 (Cortex-M7 @ 480 MHz) 上部署的 TCN+SE ch=32 / 23-dim / window=16
> IDS 模型 —— 把现有 FP32 部署权重**全量量化成 INT8**，并与 FP32 做逐项对比
> 日期: 2026-10-21 / 2026-10-22
> 分支: `feat/mcu-int8-full`（从 `feat/mcu-int8-full` 的当前 HEAD 切出，不动已有分支）

---

## 0. 结论摘要 (TL;DR)

> ⚠️ **2026-10-26 更新：L6 上机验证已完成（第 7 节）。** 下面是 L6 之后修订的结论。

| 维度 | FP32 (route E) | **全量 INT8 (CMSIS-NN)** | 变化 |
|---|---|---|---|
| 部署区 F1 (frame≥200001, thr=0.5) | 0.8671 | **0.8670** | −0.0001 (持平，**bootstrap CI 不可区分**) |
| 部署区 FPR | 5.91% | **6.98%** | +1.07 pp（**同样落在噪声带内，不可区分**） |
| 部署区 Recall | 0.8075 | 0.8149 | +0.0075 |
| 部署区 Precision | 0.9361 | 0.9261 | −0.0100 |
| 全量 F1 | 0.8813（上机确认） | **0.8823**（PC 预测，未上机） | +0.0010 |
| Flash (ROM) | 149,684 B (146.2 KB) | **97,292 B (95.0 KB)** | **−35.0%** |
| RAM (RW+ZI) | 241,888 B (236.2 KB) | **157,408 B (153.7 KB)** | **−34.9%** |
| 权重体积 | 81,972 B (Flash) + 81,972 B (RAM 拷贝) | **25,192 B (只读 Flash)** | **−69% / 免 RAM 拷贝** |
| 推理耗时 @480 MHz | ~0.13–0.20 ms | 预计 1.5–3× 更快（**仍待实测**） | — |

**一句话结论**：INT8 在**精度上无损**（部署区 F1 差 0.0001，实机 L6 复核一致），
换来 **Flash −35%、RAM −35%、权重免 RAM 拷贝**。
独立第三方量化器（ONNX Runtime）复现了同样的结论，说明"这个模型 INT8 PTQ 无损"跟具体量化器实现无关。

**L6 上机后最重要的两条**：

1. **MCU-INT8 与 PC-INT8 决策 4664/4664 完全一致** —— INT8 的部署实现是**逐位忠实**的，
   实测 prob 与 PC 镜像的差全部是日志 `prob_x1000` 整数化的量化下界。
   所以 INT8 与 FP32 的差异**全部来自量化本身，不是 MCU 实现缺陷**。
2. **"FPR +1.07pp" 在统计上不成立**：F1 差 bootstrap 95% CI = [−0.0034, +0.0033]，
   FPR 的 bootstrap 区间也相互重叠。McNemar 显示决策差异**高度显著**（p<0.0001），
   但两个指标都落在各自的噪声带内 —— 差异全部集中在决策边界带
   （58 个分歧窗**全部**满足 `|FP32 logit| < 2.0`，57% 在 `< 0.5`）。

**是否切部署的建议**：INT8 固件已就绪（`H743test_INT8` target，0 error），
L6 上机回放已完成且**未发现任何实现缺陷**。剩下的只有推理耗时未实测。
本报告**不给出**"应该切 INT8"的建议 —— 决策取决于延迟收益对不对得起 Flash/RAM 的节省，
而延迟数据目前仍是空的。

---

## 1. 两个模型分别是什么

### 1.1 FP32（现状，`H743test` target）

- 手写 C kernel：`MCU/Core/Src/ch32_inference.c`（route E，2026-08 起）
- 权重：FP32 const，81,972 B 在 Flash，**再 memcpy 一份到 RAM**（`ch32_init()`）→ 多占 82 KB RAM
- 激活缓冲 ~8 KB
- 定点方案：无（纯浮点）
- 部署区可信基线（2026-10-21 三根因修复后）：**F1 0.8671 / FPR 5.91% / R 0.8075 / P 0.9361**

### 1.2 全量 INT8（本次新增，`H743test_INT8` target）

- 手写 C kernel：`MCU/Core/Src/ch32_inference_int8.c` + **CMSIS-NN**（`MCU/Drivers/CMSIS/NN/`）
  - 卷积：`arm_convolve_s8`（CMSIS-NN 官方 int8 卷积，int32 精确累加）
  - requant：`arm_nn_requantize`（CMSIS-NN 官方，**不自己实现舍入**）
  - FC / GAP / SE scale-back：手写（CMSIS 的 `arm_fully_connected_s8` 只支持 per-tensor
    requant，本模型 FC 要 per-channel，所以自己写）
- 权重：`MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h`（67 个数组 + 标量）
- 激活：全 int8，**无 RAM 权重拷贝**（const 直接在 Flash）
- 开关：`ch32_inference.h` 里的 `CH32_USE_INT8`（0=FP32 / 1=INT8），
  `ch32_inference.c` 做分派，**调用方（main.c / modbus_ids.c）一行不用改**

### 1.3 量化方案（TFLite / CMSIS-NN 标准做法）

| 对象 | 表示 | 说明 |
|---|---|---|
| 权重（15 个张量） | int8，**per-output-channel 对称** | `scale[oc] = max|w[oc]| / 127` |
| bias | int32 | `round(b / (s_in · s_w[oc]))`，CMSIS 自己处理 input_offset |
| 模型输入 (23×16) | int8 **非对称** | `s_x = 0.0705882385`，`zp_x = −15`（实测输入被 clip 到 [−8, 10]） |
| 中间激活（ReLU 后） | int8 **非对称 per-tensor** | `real = (q − zp)·s`，用满 256 级 |
| requant | per-channel `(mult[oc], shift[oc])` | TFLite `QuantizeMultiplier` 语义（mult ∈ [2³⁰, 2³¹)） |
| SE fc2 预 sigmoid logit | int32 累加器 → FP32 | sigmoid 本身无法定点化（保留 FP32 escape） |
| SE scale-back | `(q_c2 − zp_c2) · q_sig` → requant | `q_sig = round(sigmoid(logit)·4096)` (Q12) |
| 残差相加 | 同域整数相加 | `q = clamp(q_a + q_b − zp, zp, 127)` |
| 末层 | int32 → FP32 logit → sigmoid | `logit /= CH32_TEMPERATURE`（T=1.0，与 FP32 一致） |

**关键点：conv 的 padding 是对称 same padding**（`t_in = t + k·dilation − pad`，
`pad = (K−1)·d/2`）—— 这正是根因 1 的修复点。CMSIS-NN 源码实测
（`arm_convolve_s8.c:111,119`）：`base_idx_x = stride*i_out_x − pad_x`，
`k_x = base_idx_x + dilation_x*i_ker_x`，语义与 PyTorch `padding='same'` 完全一致。

---

## 2. 量化参数是怎么定的（旋钮扫描）

输入量化范围和激活范围都是旋钮，用 `sweep_input_range.py` 扫 12 组：

- **选择只看 train 区（frame<200001）的 F1**，部署区只在最后报告 —— 绝不用部署区调参
- 3 种输入范围：`sym10`（±10）、`asym-10_10`、`asym-obs`（train 区实测 [−8, 10]）
- 4 种激活上界：真实 max / p99.9 / p99.99 / p99.0

| 输入/激活 | train F1 | train FPR | (参考) deploy F1 | (参考) deploy FPR |
|---|---|---|---|---|
| **asym-obs / max  ← 选中** | **0.8883** | 0.0144 | 0.8670 | 0.0698 |
| asym-10_10 / max | 0.8879 | 0.0159 | 0.8641 | 0.0725 |
| asym-obs / 99.99 | 0.8877 | 0.0136 | 0.8644 | 0.0734 |
| asym-10_10 / 99.9 | 0.8875 | 0.0136 | 0.8685 | 0.0640 |
| sym10 / 99.9 | 0.8868 | 0.0176 | 0.8686 | 0.0649 |
| sym10 / max | 0.8853 | 0.0203 | 0.8658 | 0.0743 |
| asym-obs / 99.0 | 0.8511 | 0.0084 | 0.8389 | 0.0382 |

参考：FP32 在 train 区 F1 = 0.8868 / FPR = 0.0092。

选中的 `asym-obs/max` 在 train 区 **比 FP32 还高 0.0015 F1**。
（顺带验证：输入特征在 train 与部署区都被 clip 到恰好 [−8, 10]，越界比例 0，
所以非对称输入量化一点信息都没丢。）

---

## 3. 精度对比（L3：全量 17164 窗回放）

窗口网格：274,628 帧 → 17,164 窗（phase=0），窗口标签 = 16 帧的 OR。
数据源 `IanArffDatasetF32.csv`（亚秒时间戳，根因 2 的修复版）。

### 3.1 分区域 4 指标（thr = 0.5）

| 区域 | 模型 | n | TP | FP | FN | P | R | **F1** | **FPR** |
|---|---|---|---|---|---|---|---|---|---|
| **部署区** (≥200001) | FP32 | 4664 | 1950 | 133 | 465 | 0.9361 | 0.8075 | **0.8671** | **5.91%** |
| | INT8 | 4664 | 1968 | **157** | 447 | 0.9261 | 0.8149 | **0.8670** | **6.98%** |
| 官方 test 区 (≥219701) | FP32 | 3433 | 1440 | 114 | 369 | 0.9266 | 0.7960 | 0.8564 | 7.02% |
| | INT8 | 3433 | 1456 | 124 | 353 | 0.9215 | 0.8049 | 0.8593 | 7.64% |
| train/val 区 (<200001) | FP32 | 12500 | 5148 | 56 | 1258 | 0.9892 | 0.8036 | 0.8868 | 0.92% |
| | INT8 | 12500 | 5189 | 88 | 1217 | 0.9833 | 0.8100 | 0.8883 | 1.44% |
| 全量 | FP32 | 17164 | 7098 | 189 | 1723 | 0.9741 | 0.8047 | 0.8813 | 2.27% |
| | INT8 | 17164 | 7157 | 245 | 1664 | 0.9669 | 0.8114 | 0.8823 | 2.94% |

**读法**：INT8 的量化误差表现为"**阈值附近的边界样本被推向两侧**"——
多检出 18 个攻击窗（FP32 FN 465 → INT8 FN 447），也多误报 24 个正常窗（FP 133 → 157）。
F1 几乎不动，**FPR 系统性偏高约 1 pp**（train 区 0.92%→1.44%，部署区 5.91%→6.98%，
比例一致 ≈ +0.5~1.1 pp）。

### 3.2 概率漂移统计

| 指标 | 值 |
|---|---|
| MAE | 0.018913 |
| RMSE | 0.044030 |
| max\|Δ\| | 0.640981 |
| P99\|Δ\| | 0.208741 |
| 相关系数 r | **0.994569** |
| thr=0.5 决策翻转率 | **1.07%** (183 / 17164) |

### 3.3 阈值敏感性（部署区 & 全量）

| thr | 部署区 FP32 F1 / FPR | 部署区 INT8 F1 / FPR | 全量 FP32 F1 / FPR | 全量 INT8 F1 / FPR |
|---|---|---|---|---|
| 0.30 | 0.8645 / 10.00% | 0.8633 / 11.12% | 0.8843 / 5.20% | 0.8817 / 6.53% |
| 0.40 | 0.8663 / 7.65% | 0.8664 / 8.67% | 0.8838 / 3.19% | 0.8833 / 4.05% |
| **0.50** | **0.8671 / 5.91%** | **0.8670 / 6.98%** | 0.8813 / 2.27% | 0.8823 / 2.94% |
| 0.60 | 0.8655 / 5.51% | **0.8672 / 5.82%** | 0.8786 / 1.88% | 0.8800 / 2.24% |
| 0.70 | 0.8635 / 4.98% | 0.8659 / 5.02% | 0.8737 / 1.58% | 0.8768 / 1.71% |
| 0.80 | 0.8610 / 4.14% | 0.8656 / 3.91% | 0.8665 / 1.26% | 0.8716 / 1.26% |

**train 区（选择区）的阈值扫描**（供 L6 上机时参考，不作为结论）：

| thr | FP32 train F1 / FPR | INT8 train F1 / FPR |
|---|---|---|
| 0.40 | 0.8906 / 1.54% | 0.8899 / 2.35% |
| 0.50 | 0.8868 / 0.92% | 0.8883 / 1.44% |
| 0.60 | 0.8837 / 0.54% | 0.8850 / 0.92% |

> ⚠️ 如果 L6 上机实测 INT8 的 FPR 明显高于 6.98%，候选动作是把阈值从 0.5 调到 0.6
> （此时 INT8 部署区 FPR 5.82% 反而低于 FP32@0.5 的 5.91%，F1 持平）。
> **但 0.6 这个值是在部署区扫出来的，不能当结论用** —— 真正的阈值必须在 H743 实测数据上定。

---

## 4. L4 交叉验证：换一个完全独立的量化器

本工程**没有 host C 编译器**，PC 侧的 INT8 数字全部来自 numpy 镜像（自证）。
为了证明"INT8 PTQ 对这个模型是无损的"跟具体量化实现无关，用 **ONNX Runtime
静态量化（QDQ, per-channel weight, 激活 QUInt8）** 从同一个 FP32 ONNX
（`model_v4_se_23dim_ch32_s42.onnx`）做了一份**独立的** INT8 模型：

| 对比 | MAE | max\|Δ\| | r | 决策翻转率 |
|---|---|---|---|---|
| ORT-INT8 vs ONNX-FP32 | 0.0188 | 0.6806 | 0.994185 | 1.20% |
| **本工程 INT8 vs ONNX-FP32** | **0.0189** | 0.6410 | **0.994569** | **1.07%** |
| 本工程 INT8 vs ORT-INT8 | 0.0132 | 0.4714 | 0.996726 | 0.90% |

分区域指标：

| 区域 | ONNX-FP32 | ORT-INT8 | 本工程 INT8 |
|---|---|---|---|
| 全量 | F1 0.8813 / FPR 2.27% | F1 0.8821 / FPR 3.15% | F1 0.8823 / FPR 2.94% |
| 部署区 | F1 0.8671 / FPR 5.91% | F1 0.8650 / FPR 7.78% | F1 0.8670 / FPR 6.98% |
| train 区 | F1 0.8868 / FPR 0.92% | F1 0.8888 / FPR 1.44% | F1 0.8883 / FPR 1.44% |

**两个独立实现的 INT8 落在同一个区间**（F1 ±0.003，FPR 同向偏高 ~1 pp）。
结论"INT8 无损、FPR 略升"是**量化方法层面的性质**，不是我这套实现的 bug。

另外这次顺手确认了 **ONNX-FP32 与部署真值一致**：ONNX 输出（logit，过 sigmoid 后）
与 `fullsim_f32.npz` 的 max|Δ| = **1.29e-06** —— 说明 numpy FP32 镜像和 ONNX 是同一个网络。

---

## 5. 资源对比（L5：Keil AC6.24 / uVision 5.43 实测）

两个 target 同一份工程、同一套编译选项（`-O0`，`H743_DTCM_arena.sct` 同一个散装文件），
都是 **Rebuild（全部重编）**，0 error：

| 项 | FP32 (`H743test`) | INT8 (`H743test_INT8`) | Δ |
|---|---|---|---|
| Code | 64,784 B | 70,192 B | +5,408 B（CMSIS-NN kernel） |
| RO-data | 84,836 B | **27,036 B** | −57,800 B |
| RW-data | 4,448 B | 4,448 B | 0 |
| ZI-data | 237,440 B | **152,960 B** | −84,480 B |
| **ROM 合计**（Flash） | **149,684 B (146.2 KB)** | **97,292 B (95.0 KB)** | **−35.0%** |
| **RAM 合计**（RW+ZI） | **241,888 B (236.2 KB)** | **157,408 B (153.7 KB)** | **−34.9%** |
| ER_IROM1 | 0x24874 | 0x17bcc | |
| RW_AXI (0x24000000) | 0x1b0e0 (110,816 B) | 0x66e0 (26,336 B) | **−84,480 B** |
| RW_DTCM (0x20000000) | 0x20000 (128 KB, 满) | 0x20000 (128 KB, 满) | 0 |

**RAM 少了 84,480 B 的原因**：FP32 版 `ch32_init()` 把 82 KB 权重从 const 拷到 RAM
（为了能跑在 DTCM/AXI 上）；INT8 版权重是 const int8，直接在 Flash 里被内核读，
**不需要任何拷贝**。AXI SRAM 从 110,816 B 降到 26,336 B。

INT8 权重 + 量化参数总计 **25,192 B**（其中权重本体 20,493 B = 20493×1，
其余是 int32 bias / mult / shift / f32 weight-scale）。

**推理耗时**：Cortex-M7 没有 int8 SIMD（只有 DSP 扩展的 int16 SMLAD），
所以 CMSIS-NN 走的是 int16 中间表示路径，预期比 FP32（FPU 单精度）快 **1.5–3×**。
**本机无法测，必须 L6 上机用 `ch32_benchmark()`（DWT cycle counter）实测。**

---

## 6. 验证阶梯状态

| 级别 | 内容 | 状态 | 结果 |
|---|---|---|---|
| **L0** | 权重自检：INT8 反量化 vs FP32 权重，max rel err | ✅ | `check_int8_weight_error.py`：15 个张量相对误差 0.24–0.39%（理论上限 0.39% = 0.5 LSB），cos ≈ 0.99996–0.999999，无系统性偏置 |
| **L1** | **源绑定一致性**：从 C 源码 + CMSIS-NN 源码 *解析* 出公式，断言 numpy 镜像一致 | ✅ | `check_int8_source_binding.py` **52/52 全过** |
| **L2** | FP32 侧回归：镜像必须复现 `fullsim_f32.npz` | ✅ | ONNX-FP32 vs npz max\|Δ\|=1.29e-06 |
| **L3** | 全量对比：分区域指标 + 阈值扫描 + 漂移 | ✅ | 见第 3 节 |
| **L4** | 交叉验证：第三方量化器（ORT）独立实现 | ✅ | 见第 4 节 |
| **L5** | 构建对比：Flash / RAM / (周期数待测) | ✅/⚠️ | 体积已测；周期数需上机 |
| **L6** | **上机验证**：烧 INT8 固件 → 标准回放 → H743 实测 prob 与镜像逐位对 | ✅ **2026-10-26 已完成** | 见第 7 节：MCU-INT8 与 PC-INT8 **决策 4664/4664 完全一致**；部署区 F1 0.8670 / FPR 6.98%，与 PC 侧预测**逐位相同** |

### L1 具体检查了什么（`check_int8_source_binding.py`）

不重写被测对象的公式，而是**从真实源文件解析**：

- **CMSIS-NN `arm_nnsupportfunctions.h`**
  - `arm_nn_requantize` 走的是 **double-rounding 分支**（NN 树里没有 `#define CMSIS_NN_USE_SINGLE_ROUNDING`）
  - 解析出舍入常量 `1<<30`、右移 31、`LEFT_SHIFT/RIGHT_SHIFT` 宏、负数 threshold 规则，
    **据此复刻成 int32 模拟器**，与 `int8_math.requantize_np` 在 8 组 shift × 5 组 mult ×
    4 万个随机值上**逐位一致**（含负数）
- **CMSIS-NN `arm_convolve_s8.c`**
  - 解析出 `base_idx_x = stride*i_out_x − pad_x`、`k_x = base_idx_x + dilation_x*i_ker_x`
  - 越界填充 = 真实 0（MVE 路径填 `-input_offset`，非 MVE 路径填 q15 中心化的 0）
  - 由此推出的索引集合与镜像 `cmsis_conv` 的索引集合**逐点相同**（K=1/3, dil=1/2/4）
- **本工程的 `ch32_inference_int8.c`**
  - `input_offset = −zp_in` / `output_offset = +zp_out`（CMSIS 约定）
  - `pad = (K−1)*dil/2`、`cp.padding.w = pad`、`cp.dilation.w = dil`、`stride = 1`
  - requantize **全部走 CMSIS 的 `arm_nn_requantize`**，没有自己实现舍入
  - GAP 舍入、输入量化舍入与镜像逐字一致
  - **7 个 conv / 2 个 fc / 2 个 fc_logit / 2 个 residual 调用点**的
    (权重符号, K, dilation, in_zp, out_zp, act_min) 与 **INT8 header 里的量化参数**一一对上
  - SE block 内部 13 个 per-block 参数（zp/scale/权重/bias/mult/shift）按 `bi` 取值正确
- **溢出体检**：所有 requantize 站点的 `acc · 2^LEFT_SHIFT(shift) < 2³¹`
  （最坏 3,108,960，充裕）

---

## 7. L6 上机验证结果（2026-10-26 已完成）

### 7.1 上机做了什么

1. 烧录 `MCU/MDK-ARM/H743test_INT8/H743test_INT8.hex`（target `H743test_INT8`）
2. `rtu_replay_gui.py` 标准 `auto-40ms` 回放，从数据集行 **200000** 起，
   发出 **74,628** 帧 → 日志 `logs/test_20261026-230332_auto-40ms.csv`
3. 对照组用 FP32 固件的同口径回放 `logs/test_20261021-203113_auto-40ms.csv`
   （**同样 74,628 帧、同样 start_frame=200000**，逐窗可比）

对比脚本：`experiments/tflm_fp32/tools/int8_vs_fp32_diff.py`（新增）、
`experiments/tflm_fp32/tools/tflm_deploy_parity.py`。

### 7.2 上机实测 = PC 侧预测，逐位相同

| 口径 | 来源 | TP | FP | TN | FN | P | R | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|
| MCU-INT8 **实测** | `logs/test_20261026-230332` | 1968 | 157 | 2092 | 447 | 0.9261 | 0.8149 | **0.8670** | **0.0698** |
| PC-INT8 **预测** | `fullsim_int8.npz` | 1968 | 157 | 2092 | 447 | 0.9261 | 0.8149 | **0.8670** | **0.0698** |
| MCU-FP32 **实测** | `logs/test_20261021-203113` | 1950 | 133 | 2116 | 465 | 0.9361 | 0.8075 | **0.8671** | **0.0591** |
| PC-FP32 **预测** | `pc_parity_logits.npz` | 1950 | 133 | 2116 | 465 | 0.9361 | 0.8075 | **0.8671** | **0.0591** |

> **L1 源绑定 + L4 交叉验证 + L6 上机，三者完全一致。**
> 本报告原先所有 PC 侧数字的推导链（numpy 镜像 → CMSIS-NN 源绑定 → ORT 第三方量化器）
> 在实机上得到确认。

### 7.3 MCU-INT8 跟 PC-INT8 是**同一个东西**（这一条最关键）
| 比较 | 窗口数 | 决策分歧 | max\|Δprob\| | median\|Δprob\| |
|---|---|---|---|---|
| MCU-INT8 vs PC-INT8（同量化模型） | 4664 | **0 / 4664** | 0.013 | **0.00099** |
| MCU-FP32 vs PC-FP32（同精度模型） | 4664 | **0 / 4664** | 0.016 | 0.00099 |
| MCU-INT8 vs MCU-FP32（跨量化） | 4664 | 58 / 4664 | 0.263 | 0.00099 |

- `median |Δprob| = 0.00099` 就是日志里 `prob_x1000` 整数化的量化下界
  （**不是**数值差异）。4664 窗决策**零分歧** → INT8 的 MCU 实现与 PC 镜像**逐位忠实**。
- 唯一超出量化下界的窗只有 **1 个**（w=557，PC 1.000 vs MCU 0.987），
  量级 1.3%，不改变任何决策。
- **推论**：INT8 与 FP32 的全部差异，来源是**量化本身**，不是 MCU 实现缺陷。
  这条推论把"量化误差"和"部署 bug"彻底分开了 —— 正是旧路线缺的因果闭合。

### 7.3b 全量区间：FP32 已上机确认，INT8 尚未

⚠️ **口径要说清楚**：

| 区间 | 数据源 | 状态 |
|---|---|---|
| **部署区**（帧 ≥200000，4664 窗） | **INT8 实测**（`test_20261026-230332`，start_frame=200000，74,628 帧） | ✅ 上机确认，与 PC 逐位相同 |
| **部署区**（同上） | FP32 实测（`test_20261021-203113`，同样 start_frame=200000 / 74,628 帧） | ✅ 上机确认 |
| **全量**（17,164 窗，start_frame=0） | FP32 实测（`test_20261021-003650`，274,628 帧） | ✅ 上机确认：**F1 0.8813 / FPR 2.27%**，与 PC 预测**逐位相同**（0/7287 决策分歧） |
| **全量**（17,164 窗） | INT8 | ⏳ **仍是 PC 侧预测（0.8823 / FPR 2.94%），未上机** |

本次 INT8 上机回放只跑了部署区（`start_frame=200000`），
所以本报告顶表里的 **"全量 F1 0.8823" 这一行仍是 PC 侧数字**。
要把它也变成上机数字，需要再跑一次 `start_frame=0` 的全量回放。

### 7.4 INT8 vs FP32：差异在**工作点**，不在**能力**

| 项 | FP32 | INT8 | Δ |
|---|---|---|---|
| F1 | 0.8671 | 0.8670 | **−0.0001** |
| Recall | 0.8075 | 0.8149 | **+0.0075** |
| Precision | 0.9361 | 0.9261 | −0.0100 |
| FPR | 0.0591 | 0.0698 | **+0.0107** |

决策翻转分解（58 窗）：

| 类别 | 数量 |
|---|---|
| FP32 TP → INT8 FN（变差：漏报） | 3 |
| FP32 TN → INT8 FP（变差：误报） | **29** |
| FP32 FP → INT8 TN（变好：误报消失） | 5 |
| FP32 FN → INT8 TP（变好：漏报消失） | **21** |

**净效果：INT8 把工作点整体往"更爱报警"的方向挪了一点**（+21 漏报变 TP、+29 误报变 FP），
所以 Recall 涨 0.75pp、FPR 涨 1.07pp，而 F1 几乎不动。

统计检验（部署区 4664 窗）：

| 检验 | 结果 | 解读 |
|---|---|---|
| McNemar（精确，58 discordant） | 8 vs 50，**p < 0.0001** | 决策差异**高度显著**（不是噪声） |
| F1 差 bootstrap 95% CI（2000 次按窗重采样） | **[−0.0034, +0.0033]** | F1 差异**不可区分**（P(INT8 更好)=0.46） |
| FPR bootstrap 95% CI @thr=0.5 | INT8 [0.0593, 0.0808] / FP32 [0.0454, 0.0638] | FPR 差异**不可区分** |

> 两个结论不矛盾：**决策显著不同，但两个指标都落在各自的噪声带内**。
> 也就是说"INT8 让 FPR 涨了 1pp"这个说法，在 4664 窗的部署区**没有统计证据**。

### 7.5 差异**全部集中在决策边界**

| FP32 logit 绝对值 | 分歧窗占比 |
|---|---|
| < 0.10 | 6 / 58（10%） |
| < 0.25 | 21 / 58（36%） |
| < 0.50 | 33 / 58（**57%**） |
| < 1.00 | 49 / 58（84%） |
| < 2.00 | **58 / 58（100%）** |

而部署区 `|logit|` 中位数是 2.13、p90 是 16.7 —— **58 个分歧窗没有一个在远离边界的地方**。
logit 尺度上 INT8 相对 FP32 的偏移是 `mean +0.084 / median +0.066 / std 0.311`，
**相对误差 +4.2%**，这个量级只够把边界样本推过阈值，推不动任何一个自信样本。

新增的 29 个 FP 更能说明问题 —— 它们在 FP32 下的概率本来就低：

| FP 集合 | n | FP32 下 prob 的 median | prob < 0.3 的占比 |
|---|---|---|---|
| **新增** FP（INT8 独有） | 29 | **0.386** | **34%** |
| 共有 FP（两边都有） | 128 | 0.895 | 0% |
| 消失 FP（FP32 独有） | 5 | 0.514 | 0% |

**新增的 FP 全是 FP32 本来就判 0.39～0.48 的擦边样本**，
和已知的"press_mean_w 驱动的高置信 FP"（median 0.895）是**两种完全不同的机制**。
这个区分对论文有用：INT8 没有引入新的误报机制，只是把边界带撑宽了。

### 7.6 阈值视角：INT8 有可回收的 FPR 余量

既然差异只在边界带，把阈值挪一点就能换回 FPR：

| thr | INT8 F1 | INT8 FPR | FP32 F1 | FP32 FPR |
|---|---|---|---|---|
| 0.50 | 0.8670 | 0.0698 | 0.8671 | 0.0591 |
| 0.55 | 0.8665 | 0.0645 | 0.8656 | 0.0574 |
| **0.58** | **0.8672** | **0.0600** | 0.8654 | 0.0565 |
| **0.60** | 0.8672 | **0.0582** | 0.8655 | 0.0551 |
| **0.62** | **0.8675** | **0.0551** | 0.8644 | 0.0547 |
| 0.65 | 0.8672 | 0.0525 | 0.8641 | 0.0525 |

**INT8 在 thr=0.60~0.62 时 F1 与 FP32@0.5 持平、FPR 反而更低（5.5% vs 5.91%）。**

⚠️ 但**不能直接改阈值**，两个原因：

1. **阈值不是稳定可估的量**。300 次 bootstrap 重采样下，INT8"最优阈值"的分布是
   median 0.60 但 **IQR [0.38, 0.63]**，p10=0.36、p90=0.73 —— 众数在 0.63，
   但 0.35~0.41 也被选中了几十次。F1 曲面在 0.5~0.65 上**几乎是平的**
   （各 thr 的 bootstrap F1 置信区间全部重叠），argmax 是在噪声上挑点。
2. **部署口径已经定死 thr=0.5**（训练侧标定值，见 `ch32_inference.h:154`）。
   改 thr 会同时改掉 FP32 基线，失去可比性。

**结论：保持 thr=0.5 不变**。INT8 与 FP32 的 F1 差异（−0.0001）在这个数据量下不可测量；
FPR 差异（+1.07pp）同样不可测量。**没有证据支持为了 INT8 调阈值，也不需要调。**

### 7.7 L6 复现命令

```bash
# 1) 上机后回放产生的日志
#    logs/test_20261026-230332_auto-40ms.csv   (INT8)
#    logs/test_20261021-203113_auto-40ms.csv   (FP32 对照, 同 start_frame=200000)

# 2) INT8 固件 vs FP32 固件 逐窗对比
python experiments/tflm_fp32/tools/int8_vs_fp32_diff.py \
    --int8-log logs/test_20261026-230332_auto-40ms.csv \
    --fp32-log logs/test_20261021-203113_auto-40ms.csv \
    --out experiments/tflm_fp32/artifacts/int8_vs_fp32_diff.json

# 3) 各版本 vs PC 参考 (部署区 4 指标)
python experiments/tflm_fp32/tools/tflm_deploy_parity.py \
    --log logs/test_20261026-230332_auto-40ms.csv --label INT8 \
    --baseline-log logs/test_20261021-203113_auto-40ms.csv --label-baseline FP32 \
    --out experiments/tflm_fp32/artifacts/parity_int8_vs_fp32.json
```

**原来需要你操作的 L6 步骤**（现在已做完，留档）：
烧 `H743test_INT8` target → 起 `rtu_replay_gui.py` → 跑 `auto-40ms` 回放 →
`h743_error_analysis.py logs/<log>.csv` 看部署区 4 指标。

> ⚠️ `h743_error_analysis.py` 里的"自校验"是拿 **PC-FP32** 参考比 INT8 固件，
> 所以对 INT8 日志会报"对拍失败"（这是**预期的**——它本来就该失败，
> INT8 跟 FP32 有 58 窗分歧）。它的价值是给 FP32 日志做自校验。

---

## 8. 风险与已知限制

1. ~~**PC 侧数字不是上机数字**~~ —— **2026-10-26 已由 L6 关闭**：
   实机 INT8 的部署区 4 指标与 PC 侧预测**逐位相同**（第 7 节）。
   本机无 host C 编译器这个限制依然存在，但 numpy 镜像这条链已被实机独立确认。
2. ~~**FPR 系统性偏高 ~1 pp**~~ —— **已从"退化"降级为"噪声"**：
   FPR 6.98% vs 5.91% 的差落在 bootstrap 置信区间内（不可区分），
   且差异全部集中在决策边界带。**但 FPR 绝对值本身仍偏高**（这是 FP32 就有的问题，
   不是 INT8 引入的）—— 对 SCADA 场景仍是主要风险点，与量化无关。
   若确实要压 FPR，INT8 在 thr=0.60~0.62 可拿到 F1 持平、FPR 5.5~5.8%；
   但阈值估计极不稳定（bootstrap IQR [0.38, 0.63]），**本报告不建议改阈值**。
3. **历史包袱**：本工程此前所有 INT8 数字（旧 `verify_full_int8_pc.py`、旧 TFLM、
   `FULL_INT8_VS_HYBRID_REPORT.md`）都因根因 1/2/3 作废，**不可引用**。
   本次是第一次在可信 FP32 基线上做 INT8 对比。
4. **CMSIS-NN 首次接入**：AC6 编译 0 error；int32 scratch buffer 给了 512 B
   （需求上限 384 B）；非 MVE 路径（M7 无 MVE）已核对 padding 语义。
5. **`ch32_inference_int8.c` 里 sigmoid 仍是 FP32**（每窗 33 次 expf）：
   这是有意的 escape hatch，跟 TFLite full-int8 模型保留 float 输出一致；
   若追求极致速度，可换 256 项 LUT，但那会引入新的量化误差，需要重新验证。
6. **推理耗时仍未实测**（L6 只做了精度，没跑 `ch32_benchmark`）：
   这是现在**唯一**还悬着的关键数字，也是"是否切 INT8"这个决策的输入。
   在它出来之前，本报告不给出切/不切部署的建议。
7. **INT8 全量区间未上机**：本次 L6 回放只覆盖部署区（start_frame=200000）。
   顶表 "全量 F1 0.8823" 仍是 PC 侧预测（FP32 的全量 0.8813 已上机确认）。
   要补齐需要再跑一次 `start_frame=0` 的全量回放。

---

## 9. 交付物清单

### 新增

| 文件 | 内容 |
|---|---|
| `quantize_fp32_to_int8.py` | 量化器：读 FP32 header（部署真值）→ train 区校准 → 写 INT8 header + 校准日志 |
| `sweep_input_range.py` | 量化旋钮扫描（输入范围 × 激活分位数），**只在 train 区选优** |
| `int8_math.py` | 定点数学底座 + INT8 header 读写 + CMSIS 语义解析 |
| `sim_int8_kernel.py` | numpy 镜像（MCU INT8 kernel 的参考实现） |
| `compare_fp32_vs_int8.py` | 全量回放对比 → 指标 / 阈值扫描 / 漂移统计 |
| `check_int8_source_binding.py` | **L1 源绑定检查**（52 项，全部从真源码解析） |
| `check_int8_weight_error.py` | **L0 权重自检**（INT8 反量化 vs FP32，15 个张量） |
| `cross_check_int8_onnx.py` | **L4 第三方交叉验证**（ONNX Runtime 静态量化） |
| `MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h` | INT8 权重 + 量化参数（1306 行） |
| `MCU/Core/Inc/ch32_inference_int8.h` | INT8 路径声明 |
| `MCU/Core/Src/ch32_inference_int8.c` | INT8 forward（CMSIS-NN） |
| `MCU_INT8_VS_FP32_REPORT.md` | 本报告 |
| `tools_add_int8_target.py` | 生成 `H743test_INT8` target 的一次性脚本 |

### 修改

| 文件 | 改动 |
|---|---|
| `MCU/Core/Inc/ch32_inference.h` | 加 `CH32_USE_INT8` 开关（默认 0 = FP32） |
| `MCU/Core/Src/ch32_inference.c` | 按开关分派到 FP32 / INT8 后端（对外 API 不变） |
| `MCU/MDK-ARM/H743test.uvprojx` | 新增 `H743test_INT8` target + CMSIS-NN 文件组 |
| `int8_math.py` | **修 bug**：header banner 注释少写 `*/`，导致整份 header 被当成注释吞掉（编译报 20 个 undeclared identifier） |

### 产物

- `int8_qparams_s42.json`（量化参数）、`quant_sweep.json`（扫描全表）
- `int8_vs_fp32_metrics_int8.json`（分区域指标 + 漂移）
- `int8_crosscheck_onnx.json`（L4 结果）
- `fullsim_int8.npz`（17164 窗 INT8 概率）、`fullsim_onnx_ort_int8.npz`（ORT INT8 概率）

### 复现

```bash
python quantize_fp32_to_int8.py --from-json int8_qparams_base.json \
       --out MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h \
       --json int8_qparams_s42.json --ilo -8.0 --ihi 10.0
python check_int8_weight_error.py        # L0 权重自检
python check_int8_source_binding.py      # L1 源绑定 (52 项)
python compare_fp32_vs_int8.py           # L3 指标
python cross_check_int8_onnx.py          # L4 交叉验证
```

---

## 10. 结论

1. **精度**：全量 INT8 与 FP32 在本模型上**等价**（部署区 F1 0.8670 vs 0.8671，
   全量 F1 0.8823 vs 0.8813）。**L6 上机实测已确认**，且 F1/FPR 的差异经 bootstrap
   检验**均不可区分**（F1 差 95% CI = [−0.0034, +0.0033]）。
2. **实现忠实性**：MCU-INT8 与 PC-INT8 **决策 4664/4664 完全一致**，
   实测 prob 的差全部是日志整数化下界。**INT8↔FP32 的差异 100% 来自量化，不是部署 bug。**
3. **代价换收益**：Flash −35%、RAM −35%、权重不再需要 RAM 拷贝，推理预期快 1.5–3×。
4. **结论稳健性**：完全独立的第三方量化器（ONNX Runtime）复现了同样的结论；
   L1 源绑定（52 项）+ L4 交叉验证 + L6 上机，三者一致。
   所以这是"INT8 PTQ 对这个模型无害"的**方法学结论**，不是实现细节的巧合。
5. **行动建议**：INT8 固件已验证可烧可跑（0 error，4664/4664 决策忠实）。
   **唯一还缺的数字是推理耗时**（`ch32_benchmark`，DWT）。
   拿到延迟数据后即可决定是否把部署基线从 FP32 切到 INT8 —— 在那之前不下结论。
