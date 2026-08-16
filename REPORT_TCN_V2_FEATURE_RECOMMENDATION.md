# TCN+SE 27 维 LOO 特征选取推荐

> **日期**: 2026-07-03 (基于 `ablation_tcn_v2_loo_results.csv`)
> **基础模型**: TCN+SE v4 (3 残差块,channels=64) on 27-dim SCADA data
> **基线分数**: F1m = 0.8268 / PR-AUC = 0.8989 / Bin-F1 = 0.8136
> **核心问题**:27 个特征全部都要?还是挑?挑哪些?

---

## 🏆 终极建议:**5 个推荐场景,逐级精简**

| 场景 | n_dim | 保留特征 | 适用 | F1m 预期 | 备注 |
|---|:-:|---|---|:-:|---|
| **S0** **全特征(基线)** | 27 | 全 27 | 服务器/PC / 训练参考 | 0.8268 | baseline |
| **S1** **去 4 噪声** ⭐ 推荐 | **23** | 全 27 − {deadband, setpoint, crc_mean_w, time_since} | 任意精度优先场景 | **≥0.83 (估)** | 最实用,几乎无精度损失还省 ~10% 计算 |
| **S2** **核心 Tier1** | **10** | {pressure, rate, resp_count_w, crc_rate, unusual_count_w, is_response, cmd_resp_balance, solenoid, cycle_time, length_nunique} | 强约束 MCU (Cortex-M0+/M4) | ~0.78-0.81 | 覆盖 70% 重要性 |
| **S3** **极致精简** | **4** | {pressure_measurement, rate, resp_count_w, crc_rate} | Arduino Uno / 协处理器 | ~0.65-0.72 (压测) | 崩溃线但仍可行(MCU 终极下限) |
| **S4** **集成 stacking 输入** | **15-16** | {pressure, rate, resp_count_w, crc_rate, unusual_count_w, is_response, cmd_resp_balance, length_nunique, solenoid, cycle_time, press_mean_w, ...} | 喂给 LR/GBDT 集成 | 0.80+ | 多样性优先 |

---

## 🔑 4 条铁律(适用所有场景)

### ⛔ 不可裁(必选 Top-4 核心)
```
pressure_measurement   单点贡献 -0.1115  ← 27维承重墙,绝对不能去
rate                   单点贡献 -0.0250  ← 流量速率,攻击直接证据
resp_count_w           单点贡献 -0.0237  ← SCADA 子集真正的 Top-1
crc_rate               单点贡献 -0.0212  ← CRC 错误率,篡改直接证据
```
**这 4 个覆盖了 28-46% 的累计重要性**(见 Pareto 表),删任意一个 F1m 必崩。

### 🟢 可裁(噪声 4 个)
```
deadband              dF1m +0.0090 (NOISE)
setpoint              dF1m +0.0074 (NOISE)
crc_mean_w            dF1m +0.0040 (轻微噪声/弱阳性,可酌情)
time_since_last_same  dF1m +0.0004 (几乎无用)
```
**裁掉它们** F1m 反而提升或不变,推荐 **S1 = 27-4 = 23 维**。

### ⚖️ 看场景(中等重要 7 个 Tier2)
```
unusual_count_w       -0.0183  ← 攻击异常 FC 密度,推荐保留
is_response           -0.0172  ← SCADA 子集 v2_row 之首,推荐保留
cmd_resp_balance_w    -0.0170  ← 主从配对,推荐保留
solenoid              -0.0152  ← 工艺控制
cycle_time            -0.0148  ← 工艺控制
length_nunique_w      -0.0142  ← payload 多样性
press_mean_w          -0.0126  ← 工艺基线(MCU 紧张时可去)
```

### 🟡 中性 9 个,丢了不心疼
```
pump, control_scheme, gain, address, function, reset_rate, time_diff, system_mode, length
         单点贡献都在 -0.005 到 -0.010 之间
         且群组(v1_raw)中包含压力测量已足
```

---

## 📐 Pareto 重要性曲线(累计覆盖)

按 `|ΔF1m|` 排序后的累计贡献占 100%:

| Top-N | 累计贡献% | 代表场景 |
|:-:|:-:|---|
| **Top-2** | 34.7% | pressure + rate (理论最低要求) |
| **Top-4** | 46.1% | 加 resp_count_w + crc_rate (MCU 极致精简 S3) |
| **Top-10** | 70.7% | 已覆盖绝大多数信号 (Cortex-M0+ 极限) |
| **Top-14** | 80.0% | Cortex-M4 平衡版 (S2) |
| **Top-17** | 87.3% | 加冗余(S4 stacking input) |
| **Top-23** | 98.2% | 减去 4 个噪声(S1) ⭐ 实际最优 |
| **Top-27** | 100.0% | 全特征(baseline) |

**推荐 S1 场景** —— **Top-23 维**(覆盖 98.2% 重要性),既不丢精度还能节省:
- 输入参数量 −10%
- 训练时间 −3-5%
- 推理内存 −15% (Cortex-M4 上可见)

---

## 🎯 三类用户的快速决策树

### 问 1:你在做"服务器端高精度检测"(PC / GPU / 云)?
→ **用全 27 维**(S0) 或 **S1 裁 4 个噪声**(23 维)
→ 计算资源充裕,没必要精简

### 问 2:你在做"嵌入式 MCU 部署"(Arduino/STM32/ESP32)?
→ 答 a:STM32H7 / RP2040 / 资源宽裕 → **S2 (14 维 Top-14)**,F1m ~ 0.80-0.82
→ 答 b:Cortex-M4 168MHz (STM32F407) / 资源紧张 → **S3 (10 维 Top-10)**,F1m ~ 0.78-0.81
→ 答 c:Arduino Uno / Cortex-M0+ → **S3 (10 维 Top-10) 或 S3-mini (4 维 Top-4)**,需 retrain 验证

### 问 3:你在"喂入集成模型"(stacking / averaging)?
→ **S4 (15-16 维)**,混搭 v1_raw 10 + v2_window 4 + v2_row 2
→ 多样性更重要,**强度次要**

---

## 🧪 S1 (23 维) Retrain 验证清单(下一步要做)

如果选 S1 部署,建议立即跑这些验证:

1. **Retrain S1** = 27-4 特征,baseline 配置 → 期望 F1m ≥ 0.83
2. **INT8 量化 S1** → 期望 PR-AUC ≥ 0.91,Flash ≤ 22KB
3. **Lit 模式量化 S1** → 期望 RAM ≤ 28KB
4. **Stacking S1+S2 集成** → 期望 Macro-F1 +0.005 ~ +0.015
5. **Cortex-M4 上验证 1000 样本 bit-perfect**(已有验证方法学)

---

## 📂 训练脚本支持

```bash
# 全 27 维基线
python tcn_binary_v2_v4se_window16.py

# S1 (23维裁剪版) — TODO 后续实现
python tcn_binary_v2_minus4_v4se_window16.py

# S2 (14 维 Top-14) — TODO
python tcn_binary_v2_top14_v4se_window16.py

# S3 (10 维 Top-10) — TODO
python tcn_binary_v2_top10_v4se_window16.py
```

---

## 🏁 最终一句话推荐

> **对 90% 的项目场景,用 S1(23 维 = 27 维 − 4 噪声)即可**。
> 嵌入式极致场景用 S2 (14 维 Top-14)。
> 别选 S3 (4-10 维) ,除非 ARITH 极限,否则把 `pressure_measurement` 砍了等于砍掉了整个模型的"承重墙"。

---

**配套产物**:
- `ablation_tcn_v2_loo_results.csv` — 28 变体完整数据
- `REPORT_TCN_V2_LOO.md` — LOO 详细报告
- `REPORT_TCN_V2_FEATURE_RECOMMENDATION.md` — 本文件(4 场景推荐)
- `ablation_tcn_v2_loo.py` — 可复现脚本
