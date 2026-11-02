# TFLite Micro H743 部署 — 实施进度

**日期**: 2026-10-14
**会话**: H743 SCADA IDS Full INT8 重构为 TensorFlow Lite Micro
**状态**: Stage 1 进行中，等待 QAT 训练 + TF 安装

---

## 阶段进度

| Stage | 任务 | 状态 | 备注 |
|------|------|------|------|
| **Stage 1** | PC Keras + TFLite pipeline | 🔄 进行中 | Keras 模型 + 训练/转换/验证/ensemble 脚本写完，等待 TF 装完跑 |
| **Stage 2** | PC baseline + verify | ⏳ 待跑 | 等 Stage 1 跑完 |
| **Stage 3** | TFLu MCU 集成 (C 代码) | ⏳ 待写 | keil uvprojx + cmake 配置 + tflite_micro runtime wrapper |
| **Stage 4** | MCU 主程序 + Modbus 链路重写 | ⏳ 待写 | main.c / modbus_ids.c / usart.c 重写 |
| **Stage 5** | Keil 工程配置 + 模型打包脚本 | 🔄 部分 | `tflite_to_c_array.py` 已写；uvprojx 编辑待 |
| **Stage 6** | 端到端验证 + 报告 | ⏳ 待跑 | 用户烧录 + F1 验证 |

### 后台任务状态

- ✅ **QAT 5-seed full 训练**：后台运行中（epoch 9/10 完成 seed=42）
  - 日志: `C:\work\Claude\Issue\qat_full_5seed.log`
  - 预期 2-3 min 后完成 5 seed
- 🔄 **conda install TF 2.15 in `tflite` env**：后台运行中（"Collecting package metadata"）
  - 日志: `C:\work\Claude\Issue\conda_tflite_install.log`
  - 预期 5-30 min（依赖网络）

---

## 已交付文件

### PC 端 Stage 1 (重写)

| 文件 | 行数 | 用途 |
|------|------|------|
| `tcn_se_keras_model.py` | ~250 | Keras TCN+SE 模型定义 (TCNBlockSE + TCNClassifierSE) |
| `train_tcn_se_keras.py` | ~300 | Keras FP32 baseline 训练 (5 seed × 20 epoch) |
| `convert_tcn_se_to_tflite.py` | ~150 | .h5 → .tflite FP32 / INT8 PTQ |
| `qat_tcn_se_keras.py` | ~180 | TFMOT QAT fine-tune → INT8 .tflite |
| `verify_tflite_pc.py` | ~180 | TFLite ↔ Keras FP32 max_diff/agreement verify |
| `ensemble_5seed_test.py` | ~170 | 5-seed 概率均值 ensemble F1 + threshold sweep |

### MCU 端 (待写)

| 文件 | 状态 | 用途 |
|------|------|------|
| `MCU/scripts/tflite_to_c_array.py` | ✅ | .tflite → uint8_t[] C array header |
| `MCU/Core/Inc/tflite_micro_config.h` | ⏳ | TFLu 配置 (arena size + input shape) |
| `MCU/Core/Src/tflite_micro_runtime.c` | ⏳ | TFLu MicroInterpreter + Invoke 封装 |
| `MCU/Core/Src/main.c` (重写) | ⏳ | SystemClock + TFLu init + 主循环 |
| `MCU/Core/Src/modbus_ids.c` (重写) | ⏳ | 帧解析 + 16 帧聚合 + 调 TFLu + JSON |
| `MCU/ThirdParty/tensorflow/` | ⏳ 待 clone | TFLu micro 子树 (git, ~100 MB) |

---

## 待删除的旧文件

> 用户决策："不采用原来的代码"
> = 重写整个 MCU 应用层 + 全部 PC 配套脚本

```
❌ MCU/Core/Inc/ch32_inference.h
❌ MCU/Core/Src/ch32_inference.c
❌ MCU/Core/Src/ch32_inference_full_int8.c
❌ MCU/Core/Src/ch32_inference_full_int8_qat.c
❌ MCU/Core/Inc/model_v4_se_23dim_ch32_*.h × 5
```

(将在 Stage 5 Keil 工程配置时一次性删除)

---

## 验收

| 指标 | 目标 | 验证 |
|------|------|------|
| PC FP32 baseline F1m | ≥ 0.87 | `ensemble_5seed_test.py --quant fp32` |
| PC INT8 QAT F1m | ≥ 0.85 | `ensemble_5seed_test.py --quant qat` |
| PC TFLite ↔ Keras 概率一致性 | max_diff < 0.05 | `verify_tflite_pc.py` |
| H743 推理时间 | < 0.5 ms @ 480 MHz | DWT cycle counter |
| H743 Flash 权重 | ≤ 30 KB | `.map` 报告 |
| H743 激活 RAM | ≤ 20 KB | DTCM arena |
| H743 端到端 F1m | ≥ 0.85 (full 274k) | modbus_replay_simulator + GUI |

---

## 快速重启用法

```bash
# Step 1: 激活已装好 TF 的 conda env
conda activate tflite

# Step 2: 跑 5-seed Keras 训练 (产 model_*_keras_s{seed}.h5)
python train_tcn_se_keras.py --all

# Step 3: 转 INT8 QAT .tflite (产 model_*_qat_s{seed}.tflite)
python qat_tcn_se_keras.py --all

# Step 4: 验证 (max_diff < 0.05)
python verify_tflite_pc.py --seed 42 --tflite model_v4_se_23dim_ch32_qat_s42.tflite

# Step 5: 5-seed ensemble F1 测试
python ensemble_5seed_test.py --quant qat

# Step 6: 打包 .tflite → MCU C header
python MCU/scripts/tflite_to_c_array.py \
    --tflite model_v4_se_23dim_ch32_qat_s42.tflite \
    --out   MCU/Core/Inc/model_data_s42.h
```
