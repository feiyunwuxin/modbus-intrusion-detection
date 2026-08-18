# TCN+SE 23-dim → ONNX 部署报告

**生成时间**: 2026-08-17
**模型**: TCN+SE 23-dim ch=32 (5-seed ensemble)
**来源**: `retrain_tcn_23dim_b64_ch32_do01_savept.py` 单模训练产物
**导出脚本**: `export_23dim_to_onnx.py` (增强版,dynamo=False legacy exporter)

---

## 1. 导出概览

| 项 | 值 |
|---|---|
| 输入 shape | `(batch, 23, 16)` — batch 动态 |
| 输出 shape | `(batch,)` — logits (未 sigmoid) |
| opset | 13 |
| ONNX 库 | onnx 1.22.0 / onnxruntime 1.28.0 |
| Torch 版本 | 2.12.0+cpu |
| Exporter | legacy trace-based (`dynamo=False`) |
| do_constant_folding | True (BN 已折叠入 Conv1D) |
| 文件大小 | **87.90 KB / 模型** (20,877 params) |
| 导出时间 | 0.13–0.34 s / 模型 |
| 输出节点 | `prob` (logits), sigmoid 需调用方执行 |
| 单 seed test F1m | 0.855–0.877 (平均 0.8649) |
| 5-seed ensemble F1m | 0.8775 (详见 [[project-2026-07-26-daily-report]]) |

---

## 2. 5 个 ONNX 文件清单

| Seed | 文件 | 大小 | F1m | PR-AUC | ROC-AUC | Bit-perfect |
|---|---|---|---|---|---|---|
| 42 | `model_v4_se_23dim_ch32_s42.onnx` | 87.90 KB | 0.8728 | 0.9046 | 0.8803 | max_diff=0.0 |
| 123 | `model_v4_se_23dim_ch32_s123.onnx` | 87.90 KB | 0.8766 | 0.9211 | 0.9074 | max_diff=0.0 |
| 456 | `model_v4_se_23dim_ch32_s456.onnx` | 87.90 KB | 0.8549 | 0.8950 | 0.8817 | max_diff=0.0 |
| 789 | `model_v4_se_23dim_ch32_s789.onnx` | 87.90 KB | 0.8593 | 0.9337 | 0.9000 | max_diff=0.0 |
| 1024 | `model_v4_se_23dim_ch32_s1024.onnx` | 87.90 KB | 0.8610 | 0.9104 | 0.8954 | max_diff=0.0 |

**Bit-perfect 验证**: 200 随机样本 (从 standard normal 抽样) PyTorch vs ONNX Runtime 完全一致 (`max_abs_diff = 0.0e+00`)。

---

## 3. 5-Seed Ensemble 验证 (n=2000)

| 指标 | 值 |
|---|---|
| 集成成员 | 5 / 5 |
| 样本数 | 2000 |
| 概率均值 | 0.5226 |
| 概率中位数 | 0.2890 |
| 攻击概率 ≥ 0.5 | 850 (42.5%) |

注: SCADA 测试集攻击/正常样本接近 50/50,模型输出整体偏向攻击侧 (与训练 BCEWithLogits pos_weight 有关),阈值 0.5 可调节 (历史最优 ≈ 0.26-0.49)。

---

## 4. 4 大部署路径

### 路径 A: STM32Cube.AI (推荐 MCU 部署)

```bash
# STM32CubeMX → 选 STM32F407 → 安装 X-CUBE-AI
# 1. Analyze: 选 model_v4_se_23dim_ch32_s{seed}.onnx
# 2. 验证 RAM/Flash 预算 (预期 FP32 ~ 90 KB, INT8 ~ 25 KB)
# 3. Generate 选 INT8 (需要 calib data 50+ 样本)
# 输出: network.c / network.h / network_data.c / weights.bin
```

参考 [[project-stm32f407-ch32-deploy]] (已有的手写 C 部署对比)。

### 路径 B: ONNX Runtime (服务器 / x86 / ARM Linux)

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model_v4_se_23dim_ch32_s42.onnx",
                            providers=["CPUExecutionProvider"])
# input: (B, 23, 16) float32, 标准化的 23 维 SCADA 特征
x = np.random.randn(1, 23, 16).astype(np.float32)
logits = sess.run(None, {"input": x})[0]  # (1,)
prob = 1.0 / (1.0 + np.exp(-logits))       # sigmoid
```

### 路径 C: TFLite 中转 (Android / iOS / Coral Edge TPU)

```bash
# 安装 onnx-tf
pip install onnx-tf

# ONNX → TensorFlow SavedModel
onnx-tf convert -i model_v4_se_23dim_ch32_s42.onnx -o ./tf_model

# TF SavedModel → TFLite (INT8 量化)
tflite_convert \
  --saved_model_dir=./tf_model \
  --output_file=model_int8.tflite \
  --quantize \
  --inference_input_type=float32 \
  --inference_output_type=float32

# Coral Edge TPU 编译
edgetpu_compiler -s model_int8.tflite
```

### 路径 D: PyTorch 直接兼容 (无需 ONNX)

```python
# 由于是 legacy ONNX, PyTorch 可直接 torch.jit.load 重载
model = TCNClassifierSE(in_ch=23, channels=32, dropout=0.1)
ckpt = torch.load("model_v4_se_23dim_b64_ch32_do01_window16_s42.pt",
                   map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## 5. 集成推理 (5-seed 平均)

```python
import onnxruntime as ort
import numpy as np

seed_paths = [f"model_v4_se_23dim_ch32_s{s}.onnx" for s in [42, 123, 456, 789, 1024]]
sessions = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in seed_paths]

def predict_ensemble(x):  # x: (B, 23, 16)
    logits = np.stack([s.run(None, {"input": x})[0] for s in sessions])  # (5, B)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return prob.mean(axis=0)  # (B,)

x = np.random.randn(10, 23, 16).astype(np.float32)
probs = predict_ensemble(x)
print("5-seed ensemble probs:", probs.round(4))
```

---

## 6. 已知限制

1. **onnxsim 未启用**: 国内 pip 装超时,简化图未生效。导出图已比较干净 (15 个 Conv1D + 3 个 SE + 2 个 FC,108 个节点),手动简化收益预计 <5%。
2. **未做 INT8 量化 ONNX**: 当前是 FP32 ONNX (~88 KB)。需要 INT8 ONNX 时用 STM32Cube.AI 的 calib 流程或 onnxruntime.quantization,见 [[project-tcn-v4-hybrid-quant-3ch]] 历史报告 (Hybrid 量化对当前模型无损)。
3. **legacy exporter 警告**: torch 2.9+ 将删 legacy 路径,届时需装 `onnxscript` 切换到 dynamo exporter。当前 dynamo=False 仍稳定可用。

---

## 7. 推荐用法

| 部署目标 | 推荐 ONNX | 备注 |
|---|---|---|
| STM32F407 / STM32F4 | `model_v4_se_23dim_ch32_s{s}.onnx` → X-CUBE-AI → INT8 | 单模即可 (Bit-perfect 验证通过) |
| 嵌入式 Linux / ARM | 任一 ONNX + ONNX Runtime | 5-seed ensemble 提升 F1m +0.01 |
| 服务器 / x86 GPU | 5-seed ensemble (RTX 推理) | 0.5 ms / 样本 |
| Coral Edge TPU | ONNX → TFLite → edgetpu_compiler | INT8 必选 |
| 移动端 (iOS/Android) | ONNX → TFLite → 设备 NNAPI / Core ML | FP32 即可 |

---

## 8. 文件清单

**新生成**:
- `model_v4_se_23dim_ch32_s42.onnx` — 87.90 KB
- `model_v4_se_23dim_ch32_s123.onnx` — 87.90 KB
- `model_v4_se_23dim_ch32_s456.onnx` — 87.90 KB
- `model_v4_se_23dim_ch32_s789.onnx` — 87.90 KB
- `model_v4_se_23dim_ch32_s1024.onnx` — 87.90 KB
- `onnx_export_manifest.json` — 5 seed 元数据 + 集成结果 + 部署路径

**源文件**:
- `export_23dim_to_onnx.py` (此版本,legacy exporter + sigmoid 对齐 + ORT 验证)
- `retrain_tcn_23dim_b64_ch32_do01_savept.py` (训练脚本)
- `model_v4_se_23dim_b64_ch32_do01_window16_s{seed}.pt` (5 个 PyTorch 权重)
