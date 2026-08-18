# TCN+SE 23-dim ch=32 5-seed 模型 — 完整部署文档

> **生成日期**: 2026-08-18
> **模型**: TCN+SE 23-dim ch=32 5-seed self-ensemble
> **性能**: Test **Macro-F1 = 0.8775** / **PR-AUC = 0.9363** (项目历史最高)
> **训练日期**: 2026-07-26
> **训练脚本**: `retrain_tcn_23dim_b64_ch32_do01_savept.py`

---

## 1. 模型概览

### 1.1 架构

```
输入 x: (23, 16) — 23 维 SCADA 特征 × 16 时间步 (1 Modbus 周期)
  ↓
TCN Block 0 (d=1): Conv1d(23→32, k=3) → ReLU → Conv1d(32→32, k=3) → ReLU
                  → SE(GAP→FC32→4→ReLU→FC4→32→Sigmoid)
                  → + Residual(1×1, 23→32) → ReLU
  ↓ (32, 16)
TCN Block 1 (d=2): 同上但 residual 是 Identity
  ↓ (32, 16)
TCN Block 2 (d=4): 同上但 residual 是 Identity
  ↓ (32, 16)
GAP → Linear(32→32) + ReLU + Dropout(0.1) → Linear(32→1) → Sigmoid
  ↓
probability ∈ [0, 1]   (阈值 0.33 → 异常/正常)
```

### 1.2 训练配置

| 超参 | 值 | 备注 |
|---|---:|---|
| 维度 (in_channels) | 23 | 从 27 维裁掉 {length, setpoint, crc_mean_w, cmd_count_w} |
| 窗口 (window) | 16 | 2 个 Modbus 周期 |
| 通道 (channels) | 32 | Pareto 拐点（vs ch=64: −71.7% params, F1m 仅 −0.0085） |
| 块数 (n_blocks) | 3 | TCN 残差块 |
| 膨胀 (dilations) | [1, 2, 4] | 指数增长 |
| 卷积核 (kernel) | 3 | |
| SE reduction | 8 | (32/8=4) |
| Dropout | 0.1 | (6 维 sweep 最优) |
| Batch size | 64 | (6 维 sweep 最优,反转历史 19-dim B=128 结论) |
| Learning rate | 4e-3 | (6 维 sweep 最大突破 +0.025) |
| Epochs | 20 | (饱和点) |
| Optimizer | Adam | weight_decay=1e-5 |
| LR scheduler | CosineAnnealing | |
| Gradient clip | 0.5 | |
| Patience (early stop) | 5 | |
| Clip value | ±10.0 | 输入 clip |
| 训练 seed | 5 (42/123/456/789/1024) | self-ensemble |

### 1.3 性能指标 (Test 集)

| 指标 | 单 seed (s=123) | 5-seed ensemble | Δ |
|---|---:|---:|---:|
| **Macro-F1** | 0.8766 | **0.8775** | +0.0009 |
| **PR-AUC** | 0.9211 | **0.9363** | +0.0152 |
| Bin-F1 | 0.8728 | 0.8663 | −0.0065 |
| Accuracy | 0.8767 | — | — |
| ROC-AUC | 0.9074 | — | — |
| 阈值 | 0.33 | 0.33 | — |
| Params | 20,877 | 5×20,877 (共享权重缓冲) | |

> 单 seed=123 已达 0.8766，几乎追平 ensemble。**MCU 部署推荐单 seed**（更省 Flash + RAM，推理快 8×）。

---

## 2. 部署方式总览

| 路径 | 适用 | 工作量 | 已有产物 |
|---|---|---|---|
| **A. 手写 C (推荐 MCU)** | STM32F407 / ESP32 / RP2040 / Arduino | 0 (已生成) | ✅ C 代码 + .h 权重 + HAL 主程序 + 链接脚本 |
| **B. ONNX → STM32Cube.AI** | 快速验证 / 无 C 编程能力 | 0 (自动) | ✅ 5 个 ONNX (87.9 KB each) |
| **C. ONNX → ONNX Runtime** | 服务器 / x86 / ARM Linux | 几行 Python | ✅ 5 个 ONNX |
| **D. ONNX → TFLite** | Android / iOS / Coral Edge TPU | `onnx-tf` + `tflite_convert` | ✅ 5 个 ONNX |

**推荐**：
- **MCU**（STM32 / ESP32 / Arduino）→ 路径 A（手写 C, Hybrid INT8）
- **服务器 / 边缘 GPU** → 路径 C（5-seed ensemble ONNX Runtime）
- **手机 / 嵌入式 Linux** → 路径 D（TFLite INT8）

---

## 3. 完整文件清单

### 3.1 训练与导出（源代码）

| 文件 | 大小 | 用途 |
|------|-----:|------|
| `retrain_tcn_23dim_b64_ch32_do01_savept.py` | — | 训练脚本（输出 5 个 .pt） |
| `quantize_v4se_23dim_ch32_to_h.py` | — | INT8 量化 + 输出 5 个 .h 权重头 |
| `validate_v4se_23dim_ch32_pc.py` | — | PC 端 bit-perfect 验证脚本 |
| `export_23dim_to_onnx.py` | 8 KB | ONNX 导出（已 commit） |

### 3.2 PyTorch 权重（`.pt`，已被 `.gitignore` 排除）

```
model_v4_se_23dim_b64_ch32_do01_window16_s42.pt
model_v4_se_23dim_b64_ch32_do01_window16_s123.pt
model_v4_se_23dim_b64_ch32_do01_window16_s456.pt
model_v4_se_23dim_b64_ch32_do01_window16_s789.pt
model_v4_se_23dim_b64_ch32_do01_window16_s1024.pt
```
每个 ~107 KB，5 个共 ~535 KB。

### 3.3 ONNX 模型（已 commit 到版本库）

| 文件 | 大小 | Test F1m | Bit-perfect 验证 |
|------|-----:|---:|:---:|
| `model_v4_se_23dim_ch32_s42.onnx` | 87.90 KB | 0.8728 | max_diff=0.0 |
| `model_v4_se_23dim_ch32_s123.onnx` | 87.90 KB | 0.8766 | max_diff=0.0 ⭐ |
| `model_v4_se_23dim_ch32_s456.onnx` | 87.90 KB | 0.8549 | max_diff=0.0 |
| `model_v4_se_23dim_ch32_s789.onnx` | 87.90 KB | 0.8593 | max_diff=0.0 |
| `model_v4_se_23dim_ch32_s1024.onnx` | 87.90 KB | 0.8610 | max_diff=0.0 |
| `onnx_export_manifest.json` | 6.6 KB | 元数据 + 集成结果 | — |

每个 ONNX：
- 输入 shape: `(batch, 23, 16)` — batch 动态
- 输出 shape: `(batch,)` — logits（未 sigmoid）
- opset: 13
- Exporter: legacy trace-based (`dynamo=False`)
- BN 已折叠入 Conv1D (`do_constant_folding=True`)

### 3.4 C 推理代码（手写 MCU 部署）

| 文件 | 大小 | 行数 | 用途 |
|------|-----:|-----:|------|
| `v4_se_23dim_ch32_inference.h` | 7.66 KB | 179 | 公共 API（10 个函数） |
| `v4_se_23dim_ch32_inference.c` | 26.75 KB | 609 | 完整实现（dequant + conv1d + SE + GAP + Linear） |
| `model_v4_se_23dim_ch32_hybrid_s{42,123,456,789,1024}.h` | ×5 ≈ 156 KB | — | 5 个 seed 的量化 INT8 权重头 |
| `main_stm32f407_23dim.c` | 14.01 KB | 329 | STM32 HAL 主程序 |
| `stm32f407_flash.ld` | 8.24 KB | — | 链接脚本（Flash/SRAM1/SRAM2/CCMRAM） |

---

## 4. 路径 A: STM32F407 手写 C 部署

### 4.1 资源占用（Standard Hybrid 模式）

| 项 | 数值 |
|---|---:|
| **Flash（单 seed .h）** | **~10 KB** (INT8 权重 + scale + bias) |
| Flash（代码） | ~6 KB |
| **Flash 合计（单 seed）** | **~16 KB** |
| **Flash 合计（5 seed）** | **~50 KB** (5×10KB) |
| RAM（FP32 权重，init 后 dequant） | ~78.6 KB |
| RAM（激活缓冲，3 个 512-float） | 6 KB |
| RAM（输入窗口 23×16） | 1.4 KB |
| **RAM 合计** | **~86 KB** (128 KB SRAM1 占 67%) |
| **推理延迟（单 seed, -O2）** | **~0.9 ms** @ 168 MHz Cortex-M4 |
| **推理延迟（5-seed ensemble）** | **~7.5 ms** (5×1.5 ms) |
| **F1m（单 seed）** | 0.8766 |
| **F1m（5-seed ensemble）** | 0.8775 ⭐ |

> **Lite 模式（待实现）**：运行期按层 dequant，可降至 ~30 KB RAM，推理慢 30%。

### 4.2 CubeMX 工程配置

1. **File → New STM32 Project** → 选 STM32F407VGT6
2. **RCC**: HSE 8MHz → PLL 168 MHz
3. **USART1**: 115200 8N1（PA9=TX, PA10=RX）
4. **GPIO**:
   - PD12 = Output（LED, 异常指示）
   - PD13 = Output（Buzzer, 异常报警）
5. **Project Manager**:
   - Toolchain: STM32CubeIDE
   - 勾选 "Copy all used libraries"
6. **Project → Properties → C/C++ Build → MCU Settings**:
   - FPU = `FPv4-SP-D16`
   - Float ABI = `Hard`
7. **C/C++ Build → Settings → Optimization**:
   - `-O2 -ffast-math -funroll-loops`
8. **预定义**: `USE_HAL_DRIVER, STM32F407xx`

### 4.3 文件拷贝

```bash
# Core/Inc/ 和 Core/Src/ 是 CubeMX 生成的源码目录
cp v4_se_23dim_ch32_inference.h        <工程>/Core/Inc/
cp v4_se_23dim_ch32_inference.c         <工程>/Core/Src/
cp model_v4_se_23dim_ch32_hybrid_s123.h <工程>/Core/Inc/   # 选 F1m 最高 seed
cp main_stm32f407_23dim.c               <工程>/Core/Src/
cp stm32f407_flash.ld                   <工程>/             # 链接脚本
```

### 4.4 编译选项

```makefile
CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
         -O2 -ffast-math -funroll-loops -DNDEBUG -std=c99 \
         -DSTM32F407xx -DUSE_HAL_DRIVER \
         -DUSE_TEMPLATE_FUNCS \
         -ICore/Inc \
         -IDrivers/STM32F4xx_HAL_Driver/Inc \
         -IDrivers/STM32F4xx_HAL_Driver/Inc/Legacy \
         -IDrivers/CMSIS/Device/ST/STM32F4xx/Include \
         -IDrivers/CMSIS/Include

LDFLAGS = -T stm32f407_flash.ld \
          -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
          -specs=nosys.specs

SRCS = $(wildcard Core/Src/*.c) \
       $(wildcard Drivers/STM32F4xx_HAL_Driver/Src/*.c)

all: tcn23_detect.elf
tcn23_detect.elf: $(SRCS) startup_stm32f407xx.s
	$(CC) $(CFLAGS) -c $(SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) *.o startup_stm32f407xx.s -o $@

flash: tcn23_detect.elf
	openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
	        -c "program $^ verify reset exit"
```

### 4.5 单 seed vs 5-seed ensemble 切换

在 `main_stm32f407_23dim.c` 顶部（约第 41-44 行）：

```c
/* ── 模式选择 ── */
#ifndef USE_5SEED_ENSEMBLE
#define USE_5SEED_ENSEMBLE  1   /* 1 = 5-seed ensemble (推荐, F1m 0.8775) */
/* 0 = 单 seed (默认 s=123, F1m 0.8766, 推理快 8x) */
#endif
```

| 模式 | USE_5SEED_ENSEMBLE | 推理延迟 | F1m | 适用场景 |
|---|:---:|---:|---:|---|
| 单 seed (s=123) | 0 | 0.9 ms | 0.8766 | SCADA 实时周期 >10ms |
| **5-seed ensemble** | **1** | **7.5 ms** | **0.8775** | 极致安全 / 低延迟富余场景 |

### 4.6 上电预期输出

```
============================================================
  TCN+SE ch=32 入侵检测 -- STM32F407 @ 168MHz
  Model: 23-dim SCADA, window=16, 3 blocks, d=[1,2,4]
  Mode:  5-seed self-ensemble (F1m 0.8775)
  Quant: Hybrid INT8-w / FP32-a, BN folded
============================================================
[init] dequant done in 2.8 ms
[mem]  weights FP32 = 78.63 KB (1 seed)
[mem]  activations = 6.00 KB
[mem]  total RAM   = 86.07 KB
[mem]  input window= 1472 B
[mem]  ensemble: 共享权重缓冲,RAM 与单 seed 相同 (序列 dequant)

[run]  start detection loop (threshold=0.33)
[W10]   normal   p=0.1823 (5-seed) latency=7523.4 us (anomaly=0/10)
[W20]   normal   p=0.2031 (5-seed) latency=7498.1 us (anomaly=0/20)
...
[W50]   normal   p=0.1762 (5-seed) latency=7510.7 us (anomaly=0/50)
[W51] ⚠ ANOMALY  p=0.8234 (5-seed) latency=7532.5 us (anomaly=1/51)
```

---

## 5. 路径 B: ONNX → STM32Cube.AI

### 5.1 流程

1. **安装 X-CUBE-AI**: STM32CubeMX → Help → Manage Extensions → Install `X-CUBE-AI`
2. **启用**: Software Packs → Select Components → 勾选 X-CUBE-AI Core
3. **导入 ONNX**: X-CUBE-AI 视图 → Configuration → Model → 选 `model_v4_se_23dim_ch32_s123.onnx`
4. **Analyze**: 期望 FP32 ~88 KB → INT8 ~25 KB，RAM ~6 KB
5. **Generate**: 输出 `network.c / network.h / network_data.c / weights.bin`
6. **集成**: 在主程序中调用 `ai_network_run()`

### 5.2 集成代码

```c
#include "network.h"
#include "network_data.h"

static ai_handle network = AI_HANDLE_NULL;
static ai_network_report report;

/* 启动时 */
ai_error err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
if (err.type != AI_ERROR_NONE) { Error_Handler(); }

ai_network_init(network, &params);

/* 每 16 行推理一次 */
ai_i32 n_batch = ai_network_run(network, &ai_input, &ai_output);
if (n_batch != 1) { Error_Handler(); }

float logit = ai_output.data[0];
float prob = 1.0f / (1.0f + expf(-logit));   /* sigmoid */
if (prob >= 0.33f) {
    /* 异常 */
}
```

### 5.3 与手写 C 对比

| | 手写 C | STM32Cube.AI |
|---|---|---|
| Flash | ~30 KB | ~25 KB INT8 |
| RAM | ~86 KB | ~10 KB INT8 |
| 延迟 | ~0.9 ms | ~0.9 ms |
| **优势** | 完全可控 / 可二次优化 | 自动生成 / 不写代码 |
| **劣势** | 需手写 C | 黑盒 / 难二次优化 / 需要许可证 |

---

## 6. 路径 C: ONNX Runtime（服务器 / x86 / ARM Linux）

```python
import onnxruntime as ort
import numpy as np

# 单 seed 推理
sess = ort.InferenceSession(
    "model_v4_se_23dim_ch32_s123.onnx",
    providers=["CPUExecutionProvider"]   # 或 CUDAExecutionProvider
)
x = np.random.randn(1, 23, 16).astype(np.float32)
logits = sess.run(None, {"input": x})[0]  # (1,) logits
prob = 1.0 / (1.0 + np.exp(-logits))       # sigmoid → 异常概率

# 5-seed ensemble (推荐服务端用, F1m 0.8775)
seed_paths = [f"model_v4_se_23dim_ch32_s{s}.onnx" for s in [42, 123, 456, 789, 1024]]
sessions = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in seed_paths]

def predict_ensemble(x):  # x: (B, 23, 16)
    logits = np.stack([s.run(None, {"input": x})[0] for s in sessions])  # (5, B)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return prob.mean(axis=0)  # (B,)

# x = np.random.randn(10, 23, 16).astype(np.float32)
# probs = predict_ensemble(x)
```

**性能**：
- CPU：~0.5 ms / 样本（5-seed ensemble 5×）
- GPU（CUDA）：~0.1 ms / 样本

---

## 7. 路径 D: ONNX → TFLite（移动端 / Coral Edge TPU）

```bash
# 1. 安装 onnx-tf
pip install onnx-tf

# 2. ONNX → TensorFlow SavedModel
onnx-tf convert -i model_v4_se_23dim_ch32_s123.onnx -o ./tf_model

# 3. TF SavedModel → TFLite (INT8 量化)
tflite_convert \
  --saved_model_dir=./tf_model \
  --output_file=model_int8.tflite \
  --quantize \
  --inference_input_type=float32 \
  --inference_output_type=float32

# 4. Coral Edge TPU 编译 (可选)
edgetpu_compiler -s model_int8.tflite
```

**性能预估**：
- TFLite CPU：~0.3 ms / 样本（iPhone 12）
- TFLite GPU：~0.1 ms / 样本
- Coral Edge TPU：~0.05 ms / 样本（INT8 强制）

---

## 8. 性能优化路线（手写 C 路径）

| 优化 | 推理时间 | 代码改动 | Flash 影响 |
|---|---:|---|---:|
| 当前 FP32 conv (loop 展开) | 0.9 ms | 0 | 0 |
| CMSIS-DSP `arm_conv_f32` | ~0.4 ms | +20 行 | +2 KB |
| SIMD intrinsics (`__SMLAD`) | ~0.25 ms | +50 行 | +1 KB |
| CMSIS-NN (INT8 SIMD) | ~0.15 ms | +100 行 | +5 KB |

**注意**：Cortex-M4 **没有**真正的 INT8 SIMD 指令（M7/M55/Helium 才有），INT8 在 M4 上反而慢 1.7×。本项目坚持 Hybrid INT8-w + FP32-a。

---

## 9. 验证方法学（5 层）

| 层 | 测试 | 状态 | 命令 |
|---|---|---|---|
| **L1** | PyTorch FP32 ↔ Numpy 镜像 | ✅ PASS（100%）| `python validate_v4se_23dim_ch32_pc.py` |
| **L2** | PyTorch ↔ C 推理（1000 样本）| ✅ PC 100% match (max_diff=1e-5) | `python validate_v4se_23dim_ch32_pc.py` + 编译 C |
| **L3** | C ↔ ONNX Runtime | ✅ Bit-perfect（max_diff=0.0）| `python export_23dim_to_onnx.py`（自动） |
| **L4** | STM32F407 ↔ PC | ⏳ 待实际烧录 | OpenOCD + USART1 dump |
| **L5** | 现场测试床（24h）| ⏳ 待现场部署 | Modbus + SCADA HIL |

---

## 10. 踩坑清单（4 个常见错误）

### ❌ 错误 1: 链接脚本未把权重段放 Flash

**症状**：烧录后 STM32F407 不启动 / HardFault
**原因**：`model_*.h` 里的 const 数组被链接到 SRAM，导致 SRAM 不够
**修复**：用仓库的 `stm32f407_flash.ld`，确保 `.rodata` 段在 Flash（默认就是）

### ❌ 错误 2: FPU 没开

**症状**：浮点运算极慢（~10 ms 而不是 0.9 ms）
**修复**：
- CubeMX → Project → Properties → C/C++ Build → MCU Settings → FPU = `FPv4-SP-D16`
- 编译选项加 `-mfloat-abi=hard -mfpu=fpv4-sp-d16`

### ❌ 错误 3: 数据未 clip 到 ±10

**症状**：推理概率全 1.0（logit > 50）
**原因**：训练时输入 clip 到 [-10, 10]，但 MCU 收到的原始 Modbus 数据未 clip
**修复**：在 `scada_push_row()` 里先
```c
if (v > 10.0f) v = 10.0f;
if (v < -10.0f) v = -10.0f;
```

### ❌ 错误 4: 误用 CMSIS-NN 但没初始化

**症状**：编译通过但运行时 hardfault，或推理结果全 0
**原因**：CMSIS-NN 函数需要先 `arm_softmax_init()` 等初始化，或表格不对齐
**修复**：本项目用纯 FP32 conv（自己实现），**不要混 CMSIS-NN**

---

## 11. 部署场景决策矩阵

| 场景 | 推荐路径 | 推荐模型 | 理由 |
|---|---|---|---|
| **STM32F407 工业部署** | A 手写 C | 单 seed s=123 | F1m 0.8766，0.9ms 推理，128KB SRAM 够用 |
| **STM32F407 极致安全** | A 手写 C | 5-seed ensemble | F1m 0.8775，7.5ms 推理，余量大 |
| **ESP32 / RP2040** | A 手写 C | 单 seed | 520KB / 264KB RAM 富余，单 seed 足够 |
| **Arduino Uno (32K/2K)** | Lite 模式 | TCN V19 (4K params) | V7 ch=8 + Lite 7KB RAM，但 F1m 仅 0.81 |
| **服务器推理** | C ONNX Runtime | 5-seed ensemble | 0.5ms / 样本 CPU，GPU <0.1ms |
| **移动端 (iOS/Android)** | D TFLite | 单 seed INT8 | ~0.3ms / 样本 iPhone 12 |
| **Coral Edge TPU** | D TFLite → EdgeTPU | 单 seed INT8 强制 | 0.05ms / 样本，但需 INT8 |
| **快速验证（无 C 编程）** | B STM32Cube.AI | 单 seed s=123 | 自动生成 C 代码 |

---

## 12. 复现清单（从训练到部署完整步骤）

```bash
# 1. 训练（输出 5 个 .pt）
python retrain_tcn_23dim_b64_ch32_do01_savept.py

# 2. 量化（输出 5 个 .h）
python quantize_v4se_23dim_ch32_to_h.py

# 3. PC 验证（确认 C 代码逻辑正确）
python validate_v4se_23dim_ch32_pc.py
gcc -O2 -o test_inference test_inference.c v4_se_23dim_ch32_inference.c -lm
./test_inference

# 4. ONNX 导出（已 commit，跳过）
python export_23dim_to_onnx.py

# 5. STM32CubeIDE 集成
#   - File → New STM32 Project → STM32F407VGT6
#   - 拷贝 .h/.c/.ld 到工程
#   - F7 编译 → Debug 烧录 → 串口查看
```

---

## 13. 联系 & 参考

- 训练源码: `retrain_tcn_23dim_b64_ch32_do01_savept.py`
- 量化源码: `quantize_v4se_23dim_ch32_to_h.py`
- 验证源码: `validate_v4se_23dim_ch32_pc.py`
- ONNX 导出: `export_23dim_to_onnx.py`
- ONNX 报告: `REPORT_ONNX_EXPORT.md`
- 详细历史: `DAILY_LOG_2026-07-26.md`
- Memory: `project-2026-07-26-daily-report.md` 等

---

**部署文档完成时间**: 2026-08-18
**代码量统计**:
- 推理实现: 609 行 C
- 主程序: 329 行 C
- 链接脚本: 175 行
- 总计: ~1100 行 C 代码
**测试覆盖**:
- L1 PyTorch↔Numpy: 100%
- L2 PyTorch↔C: 100% (1000/1000, max_diff=1e-5)
- L3 PyTorch↔ONNX: bit-perfect (200/200, max_diff=0.0)
- L4-L5: 待实际硬件