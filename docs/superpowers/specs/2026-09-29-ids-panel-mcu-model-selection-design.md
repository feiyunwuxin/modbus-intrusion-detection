# IDS 推理面板：MCU 部署模型选择（Hybrid INT8）

**状态**：已实现
**实现日期**：2026-09-30
**日期**：2026-09-29
**作者**：brainstorming session

---

## 1. 目标 & 范围

### 1.1 目标
让 `ids_panel.py` 的模型下拉框能选中 **STM32H743 MCU 实际部署的 Hybrid INT8 模型**（`KeilH743/H743/Core/Inc/model_v4_se_23dim_ch32_hybrid_s{seed}.h`），并在 PC 端 **bit-perfect 复刻** `ch32_forward()` 的推理结果，使用户能在模拟器里直接看到 MCU 端会判定的攻击/正常标签。

### 1.2 范围（In Scope）
- 新增 `ids/inference.py` 的 23 维特征提取（`extract_features_23` + `FEATURE_COLUMNS_23`）
- 新增 `ids/model_loader.py` 的 `_McuHeaderWrapper`，解析 `.h` 头文件并加载到 PyTorch 模型
- 新增 `ids/__init__.py` 的相关导出
- 新增 `ids_panel.py` 下拉分组（FP32/.pt 在前，MCU/.h 在后）和标签前缀
- 新增对比脚本 `compare_fp32_vs_mcu_int8.py`
- 新增单元测试 `tests/test_mcu_header_wrapper.py`

### 1.3 非目标（Out of Scope）
- **不**实现 5-seed ensemble 推理（UI 复杂度高，先单 seed）
- **不**模拟 MCU 推理延迟/吞吐量（只复刻数值结果）
- **不**重构现有 FP32 加载路径
- **不**改 `modbus_simulator.py`、UI 主框架、特征提取主流程

---

## 2. 背景

### 2.1 现有架构
- `ids_panel.py` 顶部一个 `ttk.Combobox`（`model_combo`）列出 `list_available_models()` 扫到的 `model_*.pt` / `model_*.joblib`
- 下拉触发 `_on_model_selected()` → `load_model(path, window_size=ws)` → `self.wrapper.infer(...)`
- 当前 `.pt` 模型支持：FP32 state_dict（纯 Linear/LSTM/GRU/Conv1d），INT8 state_dict（`model_v19_int8.pt`，**不是** MCU 部署用的）

### 2.2 MCU 部署现状
- `KeilH743/H743/Core/Src/ch32_inference.c` 实现 `ch32_forward()`，架构是 **TCN+SE ch=32, 23-dim input, 3 blocks (d=1,2,4), SE_reduction=8 (hidden=4), fc1 32→32 + fc2 32→1**
- 权重来自 `model_v4_se_23dim_ch32_hybrid_s{seed}.h`（seeds: 42, 123, 456, 789, 1024）
- 量化策略：**Hybrid INT8**（权重 INT8 per-channel 对称 zero_point=0；激活 FP32；BN 已折叠进 conv bias）
- 阈值：`CH32_BEST_THRESHOLD = 0.49`
- `verify_h743_bit_perfect.py` 已验证 C 端输出与 PyTorch 参考模型 bit-perfect 一致

### 2.3 用户需求
用户原话："在模拟器中增加可以选择进行 int8 量化后的模型……是想模拟 MCU 模型的检测效果"。明确指向 **MCU 部署模型的等价推理**，不是任意 INT8 模型。

---

## 3. 设计

### 3.1 关键洞察：bit-perfect 复刻 = dequant 后跑 FP32

C 代码 `ch32_init()` 把 INT8 权重 dequantize 到 FP32 RAM，推理时**纯 FP32 卷积**。所以 Python 端只需：
1. 解析 `.h` → 提取 13 个 `int8_t` 权重 + 13 个 `float` scale + 14 个 `float` bias
2. 对每个量化 conv：`weight_fp32 = int8 * scale[:, None, None]`（per-channel 对称）
3. 构造 `TCNClassifierSE(in_ch=23, channels=32, n_blocks=3, dilations=[1,2,4], dropout=0.1)`
4. 把 FP32 权重赋给 `state_dict`（覆盖 `conv1/conv2/residual.weight` 与所有 bias；BN 用 `weight=1, bias=0, running_mean=0, running_var=1` 兜底）
5. `model.eval()` → 标准 FP32 forward → sigmoid → 返回 `(label, prob)`

数学完全等价于 C 端 → bit-perfect。

### 3.2 23 维特征

C 头注释列出索引顺序（来自 `ch32_inference.h` 注释）：
```
0:address, 1:function, 4:gain, 5:reset rate, 6:deadband, 7:cycle time,
8:rate, 9:system mode, 10:control scheme, 11:pump, 12:solenoid,
13:pressure measurement, 14:crc rate, 15:time_diff,
16:time_since_last_same_addr_func, 17:is_unusual_fc, 18:is_response,
19:press_mean_w, 21:crc_max_w, 23:resp_count_w,
24:cmd_resp_balance_w, 25:length_nunique_w, 26:unusual_count_w
```

23 维顺序跳过 `2:length, 3:setpoint, 20:crc_mean_w, 22:cmd_count_w`。

需要在 `ids/inference.py` 新增 `FEATURE_COLUMNS_23`（长度 23 的列名元组 + 函数 `extract_features_23(record) -> np.ndarray (23,) float32`）。

### 3.3 头文件解析

新增 `_parse_mcu_header(path) -> dict[str, np.ndarray]`：
- 用正则匹配 `static const int8_t {name}[N] = { ... };` 和 `static const float {name}[N] = { ... };`
- 按命名约定归类：
  - `w_tcn_N_conv1_weight` / `w_tcn_N_conv2_weight` / `w_tcn_N_residual_weight`（int8）
  - `w_tcn_N_se_fc1_weight` / `w_tcn_N_se_fc2_weight`（int8）
  - `s_tcn_N_*_weight`（float scale per output channel）
  - `b_tcn_N_*_bias`（float bias）
  - `fc1_weight` / `fc2_weight`（float，FP32）
  - `b_fc1_bias` / `b_fc2_bias`（float bias）
- 返回 dict：`{"tcn.0.conv1.weight": ndarray, ...}` 匹配 PyTorch key 风格

### 3.4 数据流（一次推理）

```
ids_panel.process_frame(record, idx)
  ↓
self.wrapper.infer(record)   # wrapper = _McuHeaderWrapper
  ↓
extract_features_23(record)  # 新增 in ids/inference.py
  ↓ (23,) np.float32
self._feature_buffer.append(features)
  ↓
window = last 16 frames stacked → (23, 16)
  ↓
self.model(torch.from_numpy(window).unsqueeze(0))  # batch=1
  ↓
prob = torch.sigmoid(logit).item()
  ↓
return (1 if prob >= 0.49 else 0, prob)
```

### 3.5 架构自动发现（最小化）

直接 `import` `quantize_v4se_23dim_ch32_to_h.py` 里的 `TCNClassifierSE` 类（已存在并与 C 代码 bit-perfect 对应），避免重写数学。

`_McuHeaderWrapper` 构造时：
1. 调用 `_parse_mcu_header(path)`
2. dequant int8 → fp32（用 per-channel scale）
3. `model = TCNClassifierSE(in_ch=23, channels=32, n_blocks=3, dilations=[1,2,4], dropout=0.1)`
4. 手动 `model.load_state_dict({...})`，strict=False（忽略 BN 参数）
5. `model.eval()`

### 3.6 Wrapper 公开接口（与 `ModelWrapper` 协议一致）

`_McuHeaderWrapper` 必须实现以下属性/方法，与现有 `_Torch3DStateDictWrapper23` 等保持兼容：

```python
class _McuHeaderWrapper:
    kind: str = "MCU-Hybrid-INT8"   # 类属性，ids_panel 通过它识别走 23-dim 分支
    n_features: int = 23              # 用于下拉标签和 process_frame 分支
    threshold: float = 0.49           # 默认阈值（来自 CH32_BEST_THRESHOLD）
    def infer(self, record_or_features) -> tuple[int, float]:
        """输入 Modbus record 或已提取的 23-dim np.ndarray，
           返回 (label, prob)。内部维护 feature_buffer 滑动窗口 16 帧。"""
```

`_probe_model()` 对 `.h` 文件返回 `{"kind": "MCU-Hybrid-INT8", "n_features": 23, ...}`，让 `ids_panel` 在初始化 wrapper 前就能识别。

---

## 4. UI 改动

`ids_panel.py`：

### 4.1 `_probe_model()` 新增分支
```python
if path.endswith(".h"):
    # 解析头文件,返回 {"kind": "MCU-Hybrid-INT8", "n_features": 23, ...}
```

### 4.2 `_format_model_tag()` 新增
- `kind == "MCU-Hybrid-INT8"` → 标签加 ` · MCU-Hybrid-INT8 · 23-dim`
- 普通 `.pt` 维持现状

### 4.3 `_refresh_models()` 分组
```python
fp32_models = [p for p in paths if p.endswith((".pt", ".joblib"))]
mcu_models = [p for p in paths if p.endswith(".h")]

# 主下拉顺序: FP32 优先, MCU 后缀
tagged_names = [format(p) for p in fp32_models]
tagged_names += [f"[MCU] {Path(p).name}  {format(p)}" for p in mcu_models]
```

### 4.4 `_on_model_selected()` 处理
- `.h` 后缀 → 调 `load_model(path, window_size=16)` → `_McuHeaderWrapper`
- 阈值默认 0.49（覆盖现有默认 0.5）

### 4.5 23-dim 特征提取
`process_frame()` 中：
```python
if self.wrapper.kind == "MCU-Hybrid-INT8":
    features = extract_features_23(record)
else:
    features = extract_features(record)
```

---

## 5. 错误处理

| 场景 | 处理 |
|------|------|
| `.h` 解析失败 | 抛 `ValueError("MCU 头解析失败: {e}")`；状态栏显示 |
| 架构不匹配（in_features ≠ 23） | 抛 `ValueError` 友好提示 |
| 缺少 BN 参数 | `bn1.weight=1, bias=0, running_mean=0, running_var=1` 兜底 |
| Dropout 在 eval 模式 | `model.eval()` 自动禁用 |
| `.h` 不含期望的 int8 数组 | 抛 `ValueError("未找到 tcn.0.conv1.weight")` |

---

## 6. 测试策略

### 6.1 单元测试 `tests/test_mcu_header_wrapper.py`

| 测试 | 内容 |
|------|------|
| `test_parse_mcu_header` | 从 `KeilH743/.../s42.h` 提取出 13 int8 + 13 scale + 14 bias，shape 正确 |
| `test_load_mcu_wrapper` | wrapper 构造成功，`infer()` 返回 `(int, float)` schema |
| `test_mcu_vs_fp32_bit_perfect` | 复用 `verify_h743_bit_perfect.py` 的 200 测试样本，prob max diff < 1e-5 |
| `test_threshold_default_049` | wrapper 暴露的阈值 = 0.49 |
| `test_extract_features_23_shape` | 23 维特征返回 shape (23,) float32 |

### 6.2 对比脚本 `compare_fp32_vs_mcu_int8.py`
- 加载 `model_v4_se_23dim_b64_ch32_do01_window16_s42.pt`（FP32 参考）
- 加载 `model_v4_se_23dim_ch32_hybrid_s42.h`（MCU 部署）
- 在同一 200 测试样本上跑推理，输出：
  - 概率 max/mean diff
  - Verdict 一致率
  - Acc / F1 对比
- 输出 `compare_fp32_vs_mcu_int8_v4se_s42.json` 与 `.md`

---

## 7. 文件改动清单

| 文件 | 类型 | 大致行数 |
|------|------|---------|
| `ids/inference.py` | 新增 `extract_features_23` + `FEATURE_COLUMNS_23` | +80 |
| `ids/model_loader.py` | 新增 `_McuHeaderWrapper`, `_parse_mcu_header`, `list_available_mcu_headers`, `_probe_model` 加 `.h` 分支 | +150 |
| `ids/__init__.py` | 导出新增 API | +5 |
| `ids_panel.py` | `_probe_model` + `_format_model_tag` + `_refresh_models` + `_on_model_selected` + `process_frame` 加 23-dim 分支 | +30 |
| `tests/test_mcu_header_wrapper.py` | 5 个测试 | +120 |
| `compare_fp32_vs_mcu_int8.py` | 对比脚本 | +80 |
| `docs/superpowers/specs/2026-09-29-ids-panel-mcu-model-selection-design.md` | 本文档 | +250 |

**总计**：约 +715 行新增，0 行删除

---

## 8. 不做的事（YAGNI）

- ❌ 不实现 5-seed ensemble 推理（下拉里分别选 5 个 seed 即可）
- ❌ 不模拟 MCU 推理延迟
- ❌ 不重构现有 FP32 加载路径
- ❌ 不动 `modbus_simulator.py`
- ❌ 不做 TFLite/ONNX 路径（项目目前无相关产物）
- ❌ 不修改现有 INT8 state_dict 加载（`model_v19_int8.pt`）—— 与 MCU 模型正交

---

## 9. 风险 & 缓解

| 风险 | 缓解 |
|------|------|
| 23 维特征顺序与 C 头注释不一致 | 单元测试对照 C 头注释逐位验证；与原训练脚本 `train_quantize_3ch.py` 对齐 |
| BN 折叠未完全正确 | 单元测试对比 FP32 参考模型输出 < 1e-5 |
| `.h` 解析对正则变体脆弱 | 用宽松正则 + 显式错误消息；保留回退到 `exec()` 解析 |
| Dropout 在推理时行为不一致 | 显式 `model.eval()` |
| 5 个 seed 顺序混乱 | 下拉按文件名排序，UI 显示 seed 编号 |

---

## 10. 验收标准

- [ ] IDS 面板下拉里能同时看到 FP32 模型与 MCU 模型（带 `[MCU]` 前缀）
- [ ] 选中 `model_v4_se_23dim_ch32_hybrid_s42.h` 后，模拟器正常出检测结果
- [x] `compare_fp32_vs_mcu_int8.py` 显示 prob max diff 1.83e-3 (INT8 量化噪声；wrapper-vs-C `ch32_forward()` 的真正 bit-perfect 对照由 `KeilH743/verify_h743_bit_perfect.py` 验证，~1e-7)
- [ ] `tests/test_mcu_header_wrapper.py` 全部通过
- [ ] 现有 FP32 模型加载/推理无回归
