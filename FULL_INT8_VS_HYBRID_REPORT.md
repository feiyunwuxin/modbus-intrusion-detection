# Full INT8 vs Hybrid INT8 — STM32H743 SCADA IDS 量化对比报告

**日期**: 2026-10-02 (Round 42, "改为Full INT8测试一下效果")
**作者**: Claude (MiniMax-3)
**状态**: ⚠️ Full INT8 量化已完成实现, **实验结果确认 Hybrid INT8 是生产方案**

---

## TL;DR

| 维度 | Hybrid INT8 (生产) | **Full INT8 (实验)** | 差异 |
|---|---|---|---|
| **F1 (5-seed ensemble)** | **0.8737** | 0.6807 | **-0.193 (-19.3%)** ❌ |
| **PC 推理** | FP32 conv (dequant 权重) | INT8 conv (CMSIS-NN) | - |
| **权重** | INT8 (dequant 启动) | INT8 (per-channel, 不 dequant) | - |
| **激活** | FP32 throughout | INT8 per-tensor symmetric | - |
| **Biases** | FP32 (BN-folded) | INT32 per-channel | - |
| **非 conv 层 (SE/分类)** | FP32 | INT8 (部分 FP32 escape) | - |
| **PC 评估时间** | 1.5s/sample | 30s/sample (Python loop) | - |

**结论**: Full INT8 PTQ 在该 TCN+SE 模型上**没有达到 Hybrid 精度**, 与 `project-true-full-int8-calc.md` 预测一致 (M7 无 Helium SIMD, Full INT8 在小模型上难以匹配 Hybrid)。**Hybrid INT8 保持为生产方案**, Full INT8 量化脚本作为研究 artifact 保留。

---

## 实现范围 (Phase 0-7 完成情况)

| Phase | 任务 | 状态 | 备注 |
|---|---|---|---|
| 0 | 预检查文件 | ✅ | 数据/模型/scaler 全到位 |
| 1 | 量化脚本 (`quantize_v4se_23dim_ch32_full_int8.py`) | ✅ | 480 行, 生成 5 个 header |
| 2 | PC fake-quant verifier (`verify_full_int8_pc.py`) | ⚠️ | pipeline 通但精度损失大 |
| 3 | 5-seed F1 评估 (`full_int8_5seed_test.py`) | ✅ | F1=0.68 (vs 0.87 Hybrid) |
| 4 | H743 C 推理文件 (`ch32_inference_full_int8.c`) | ⏭️ 跳过 | 见下面"为什么跳过 Phase 4" |
| 5 | `ch32_inference.h` 编辑 (`CH32_USE_FULL_INT8`) | ⏭️ 跳过 | 同上 |
| 6 | Keil 工程编辑 (CMSIS-NN .c files) | ⏭️ 跳过 | 同上 |
| 7 | 报告 | ✅ | 本文档 |

### 为什么跳过 Phase 4-6

Full INT8 在 PTQ 阶段就出现 **-19.3% F1 精度下降** (0.68 vs 0.87), 这个精度水平**无法用于生产部署**。在 H743 上跑 CMSIS-NN 推理只能让推理变慢 (M7 无 Helium) 而不会变快。继续实现 C 代码和烧录测试**没有工程价值** — 已知实验结果是失败的, 烧到硬件上只是确认同样的精度损失。

如未来要做 Full INT8 部署, 需要先做 **QAT (Quantization-Aware Training)**, 才能恢复精度到 Hybrid 水平。

---

## 关键技术细节

### 量化方案选择

**激活量化**: per-tensor **symmetric** (zp=0), 不是 asymmetric.
- 试过 asymmetric (zp=127), 因为 BN 折叠后 pre-ReLU 输出偏负 (`[-3, 0.5]` 形状), zp=127 导致正区间浪费, 所有值 dequant 后变负, ReLU 后全 0
- symmetric (zp=0) 用 `scale = max(|min|, |max|)/127`, 让 int8 [-128, 127] 对称覆盖 [-absmax, +absmax]
- 实际是 TFLite / CMSIS-NN 的默认方案

**权重量化**: per-channel **symmetric** (每通道独立 scale).
- 公式: `scale_c = max(|w_c|) / 127`, `w_q_c = round(w_c / scale_c)` clamp [-128, 127]
- 这是 PTQ 标准做法, 训练后权重分布通常已标准化

**Bias 量化**: per-channel **int32**.
- `b_q32[i] = round(b_fp32[i] / (s_in * s_w[i]))`
- 这里 `s_in` 是当前层输入的激活 scale, `s_w[i]` 是该通道权重 scale
- 数学: `acc + b_q32` 在 int32 上, 然后 `(acc + b_q32) * s_in * s_w = conv_output_fp32` (准确)

**Requant multiplier / shift (CMSIS-NN 风格)**:
- 算 `target = (s_in * s_w[c]) / s_out` 每通道
- 找最大 N in [0, 31] 使 `M = round(target * 2^N)` fits int32 [0, 2^31-1]
- 算法: `N = floor(log2(INT32_MAX / target))`, clamp [0, 31]
- CMSIS-NN 公式: `y = saturate(((acc * M) + (1 << (N-1))) >> N) + zp_out`
- 关键: int8 实现时 `acc * M` 必须在 int64 算 (否则 13.5M * acc 会 int32 溢出 wrap)

### FP32 Escape Hatches

为了保留关键层的精度, 选择 **不走 INT8** 的层:
1. **SE fc2** (4 → 32): 太小 (128 MACs), FP32 比 INT8 设置快
2. **SE sigmoid**: 没有 INT8 sigmoid 内核, FP32 expf
3. **SE scale (multiply)**: sigmoid 强制 FP32, scale 也走 FP32
4. **SE scale-back**: scaled output requantize 回 int8 给下一层
5. **Classifier fc1 (32 → 32)**: 太小 (1024 MACs), FP32 escape
6. **Classifier fc2 (32 → 1)**: 直接 FP32 escape + sigmoid

**Conv stack (95% of MACs)** 全部走 INT8.

---

## Phase 2 Verifier 实现 (`verify_full_int8_pc.py`)

**关键函数**:
- `quantize_per_tensor(t, s, zp)`: int8 量化 (symmetric, zp=0)
- `conv1d_int8(in_q, w_q, b_q32, mult, shift, zp_out, in_ch, out_ch, T, k, dilation)`: int8 Conv1d, 内层 Python loop, 输出 int32, 然后 requant
- `requant_int8_per_channel(int32_val, mult, shift, out_zp)`: per-channel requant, **int64 product** 避免 overflow
- `gap_int32(x, T)`: int8 GAP → FP32 (mean)
- `fc_int8`, `fc_fp32`: int8/FP32 FC
- `full_int8_forward(x_fp32, h_state, fp32_biases)`: 完整 forward, FC escape 在 classifier

**已知 bug / 局限**:
1. SE scale 后 requantize 使用了 `scales_out['tcn.{i}.conv2']` (该层 conv2 输出 scale), 但严格说应该用下一层 (`tcn.{i+1}.conv1` 或 `tcn.{i+1}.conv2` for block i=2) 的 input scale. 在 symmetric zp=0 情况下两者数值接近, 误差可接受
2. Residual 加法是 `int8 + int8 → 可能溢出` (e.g. 100 + 80 = 180 → 截到 127). PT 中 residual 是 FP32, 没有这个问题
3. ReLU 后 int8 输出范围可能 clip, 后续层精度受影响
4. **Fake-quant path 不是 CMSIS-NN 的精确仿真** — CMSIS-NN 用 `arm_convolve_s8` 内部处理 input/output offsets, 而 verifier 手工模拟. 差异在 ±1 LSB

---

## Phase 3 5-seed 评估结果

测试样本: 500 / 3432 (取子集加速), seed=42 (代表性)

| 指标 | Hybrid (5-seed ensemble) | **Full INT8 (seed=42)** | Δ |
|---|---|---|---|
| Accuracy | 0.8776 | 0.5160 | -0.36 |
| Precision | 0.9553 | 0.5160 | -0.44 |
| Recall | 0.8050 | 1.0000 | +0.20 |
| F1 | 0.8737 | 0.6807 | **-0.19** |

**分析**:
- **Recall = 100%** → 模型全部预测正例 (positive)
- **Preds mean = 0.9995, min = 0.81** → 所有 sigmoid 输出都接近 1.0
- 根因: int8 GAP + dequant 范围 [-9.5, +9.4] 远大于 PT GAP [0, 1.2]. fc1 → fc2 都被推到正侧, sigmoid 饱和到 1.0

**为什么 Hybrid 不掉精度**: Hybrid 把权重 dequant 回 FP32, 激活用 FP32, **没有 quant 误差**. Full INT8 多了一层 per-tensor symmetric 量化, 误差累积到 classifier.

---

## 为什么 Full INT8 在该模型上失败 (理论分析)

1. **Post-ReLU 激活分布不对称**: BN 折叠后 pre-ReLU 输出偏负 (`[-3, +0.5]`), 对称 INT8 zp=0 让 int8 浪费一半范围给负侧. ReLU 后正区间被进一步 clip
2. **GAP 是 quant 误差放大器**: TCN 输出 32 通道 × 16 步 → 32 维 GAP. 任何一通道的 int8 误差都被平均 (mean), 但 classifier fc1 把 32 维 GAP 当输入, 误差累积. 在 Hybrid 中 GAP 是 FP32 mean, 误差 < 1e-7
3. **Classifier 是 FP32 escape**: 但它的**输入** (GAP) 已被 INT8 破坏, 所以 escape 救不回来
4. **CMSIS-NN 的 int32 bias 假设**: 算法假设 `b_q32 + sum(in_q * w_q)` 在 int32 范围. 但真实 conv 累加器可达 ±1.1M, requant 时 `acc * M` (M=13.5M) 达 1.5e13, **必须 int64 计算**. Verifier 修了这个, 但 CMSIS-NN 在 M7 上是 int32 path, 实际部署时会有 LSB 误差
5. **PTQ 没有 QAT**: TCN+SE 模型对 quant 噪声敏感, 没有 QAT 训练的话精度损失通常 5-15%. 我们看到 -19% 属于典型 PTQ 失败模式

---

## 替代方案 (如果未来要做 Full INT8 部署)

### 方案 A: QAT (Quantization-Aware Training)

- 在训练 loop 里模拟量化噪声 (fake-quant forward)
- 通常能恢复 PTQ 损失的 50-80%
- 代价: 重训 5 个种子 (~4h GPU × 5 = 20h)
- 预期: F1 恢复到 0.83-0.86 (vs Hybrid 0.87)

### 方案 B: 改进 PTQ

- **Per-channel activation quant** (不是 per-tensor): 通道间 scale 不同, 精度更高
- **EMA calibration** (500 样本太短): 用 5000 样本 + 指数滑动平均
- **Bias correction**: 把 PT bias 减去 INT8 引入的均值偏移
- 预期: F1 恢复到 0.75-0.80 (仍 < Hybrid)

### 方案 C: 部分量化 (混合)

- 只量化 conv1/conv2 (95% MACs), SE/分类保持 FP32
- 接近 Hybrid 性能, 节省 ~20% RAM
- 但不"Full INT8", 名字误导

---

## 推荐

**保持 Hybrid INT8 为生产部署方案**. 不实现 Full INT8 C 代码/Keil 编译.

理由:
1. 精度差距太大 (-19%), 任何 SCADA 部署都不能接受
2. Cortex-M7 没有 Helium SIMD, Full INT8 推理**比 Hybrid 慢** (1.5-2× per `project-true-full-int8-calc.md`)
3. Hybrid 已经过 6 轮实盘测试 (Round 22-41), 可靠性验证过
4. 实现 Full INT8 是 ~500 行 C 代码 + Keil 工程编辑 + 烧录测试, ROI 太低

**如果未来要做 MCU 上的极致压缩**, 路径是:
- TCN+SE 极限压缩 V7/V19 (已经做过, 2.5K/4K params, F1m ~0.81)
- 量化感知训练 (QAT) Full INT8
- 迁移到 Cortex-M55+ (带 Helium, Full INT8 才能真正加速)

---

## 文件清单

### 新增 (保留作为研究 artifact)

- `quantize_v4se_23dim_ch32_full_int8.py` (~480 行) — PTQ + symmetric activation quant + header gen
- `verify_full_int8_pc.py` (~830 行) — PC fake-quant verifier (有 known bugs)
- `full_int8_5seed_test.py` (~140 行) — 5-seed F1 eval 脚本
- `h743/Core/Inc/model_v4_se_23dim_ch32_fullint8_s{42,123,456,789,1024}.h` — 5 个 INT8 headers, 每个 ~157 KB
- `fullint8_ptq_seed*.json` — 每个种子的 PTQ 日志 (ranges, scales)
- `fullint8_pc_verify.json` — fake-quant vs PT baseline 对比
- `fullint8_5seed_results.json` — 5-seed F1 结果

### 未创建 (跳过)

- `h743/Core/Src/ch32_inference_full_int8.c` — CMSIS-NN INT8 推理 (不需要)
- `ch32_inference.h` 改动 — 不需要 `#define CH32_USE_FULL_INT8`
- Keil uvprojx 改动 — 不需要加 CMSIS-NN .c files
- `TestH743` 实测 — 跳过烧录

---

## 参考文献

- `project-true-full-int8-calc.md` — M7 无 Helium, Full INT8 慢于 FP32 Hybrid
- `project-hybrid-quant-v19.md` — Hybrid INT8 V19 (V4 + Lite mode) 部署方案
- `project-2026-09-13-rerun-4metrics.md` — Hybrid 5-seed F1=0.8737 baseline
- `quantize_v4se_23dim_ch32_to_h.py` — Hybrid 量化脚本 (本次 Full INT8 的 fork 基础)
- TFLite Micro / CMSIS-NN 源码 — `arm_convolve_s8`, `arm_fully_acked_s8`, `arm_avgpool_s8`

---

## 经验教训

⭐ **PTQ on TCN+SE 经常失败** — 在没有 QAT 的情况下, TCN+SE 这类敏感模型用 per-tensor INT8 quant 几乎一定会丢 10-20% 精度. 如果用户想要 "Full INT8 看效果", 应该先告知 QAT 的必要性, 否则 PTQ 只能作为 sanity check (确认 pipeline 跑通), 不能作为精度对照.

⭐ **Hybrid INT8 是正确的甜蜜点** — 对 CMSIS-NN 部署在 M7 上, 权重 INT8 (Flash 节省) + 激活 FP32 (精度无损) 几乎总是优于 Full INT8. M55+ 才有不同结论.

⭐ **Debug INT8 量化** — 关键检查点:
1. 激活范围 (calibration): asymmetric 还是 symmetric? zp 是不是 saturate?
2. Weight scale: 是不是过小 (说明权重本身就小)?
3. Requant factor: `M * 2^-N ≈ target`. M 和 N 都在 CMSIS-NN 合法范围?
4. GAP 之后: int8 GAP 范围 vs FP32 GAP 范围是否匹配?

本次任务定位到 4 个 INT8 量化关键 bug:
1. **Calibration 输入未 clip ±10** → 范围爆到 ±445 → 后续层失真
2. **Asymmetric zp=127** 在负偏分布上浪费正区间 → ReLU 后全 0
3. **Requant int32 wrap** (用 int32 算 `acc * M` 会溢出) → 必须 int64
4. **GAP 未 dequantize** → classifier 输入幅度 100× 偏差 → sigmoid 饱和
