# IanArffDataset 分析与入侵检测预处理报告

> 日期：2026-06-08
> 数据集：Mississippi State SCADA Lab — Gas Pipeline Dataset (Ian Turnipseed, 2014)

---

## 一、数据集分析

### 1.1 元信息

| 项目 | 内容 |
|------|------|
| 关系名 | `@relation gas` |
| 作者 | Ian Turnipseed（导师：Dr. Morris） |
| 生成日期 | 2014-12-19 |
| 文件大小 | 17.5 MB（18,340,346 字节） |
| 总行数 | 274,659 |
| 数据记录数 | **274,628** |
| 属性数 | **20** |
| 时间跨度 | 2014-12-15 14:22 UTC → 2014-12-18 18:37 UTC（≈ 3.2 天） |

### 1.2 属性结构

#### ① 网络/通信特征 (1–15)
| # | 属性 | 类型 | 说明 |
|---|------|------|------|
| 1 | `address` | real | Modbus 寄存器地址 |
| 2 | `function` | real | Modbus 功能码（3=Read, 16=Write 等） |
| 3 | `length` | real | 数据长度（仅 6 个值：10/12/14/16/46/90） |
| 4 | `setpoint` | real | 设定值 |
| 5 | `gain` | real | 增益 |
| 6 | `reset rate` | real | 重置速率 |
| 7 | `deadband` | real | 死区 |
| 8 | `cycle time` | real | 周期时间 |
| 9 | `rate` | real | 速率 |
| 10 | `system mode` | real | 系统模式 |
| 11 | `control scheme` | real | 控制方案 |
| 12 | `pump` | real | 泵状态 |
| 13 | `solenoid` | real | 电磁阀 |
| 14 | `pressure measurement` | real | 压力测量值 |
| 15 | `crc rate` | real | CRC 校验速率 |

#### ② 命令/时间
| # | 属性 | 类型 | 说明 |
|---|------|------|------|
| 16 | `command response` | {0,1} | 命令响应 |
| 17 | `time` | real | Unix 时间戳（毫秒精度） |

#### ③ 标签列（3 个，层层细化，**信息冗余**）
| # | 属性 | 取值 | 含义 |
|---|------|------|------|
| 18 | `binary result` | '0','1' | 二分类：正常/攻击 |
| 19 | `categorized result` | '0'..'7' | 8 类分类 |
| 20 | `specific result` | 0..35 | 36 类分类 |

### 1.3 类别分布（specific result）

| 类别 | 数量 | 比例 |
|------|------|------|
| **0 (Normal)** | 214,580 | **78.13%** |
| 35 | 2,204 | 0.80% |
| 18 | 2,176 | 0.79% |
| 30 | 2,120 | 0.77% |
| 27 | 2,079 | 0.76% |
| ... | ... | ... |
| 24 | 1,160 | 0.42% |
| **20** | **666** | **0.24%**（最少） |

➡ **严重类别不平衡**。Normal 主导，35 个攻击类均匀分布。

### 1.4 关键数据模式

1. **4 行 Modbus 轮询周期**：
   ```
   4,3,16,...    ← Read Holding Registers
   4,3,46,...    ← 读响应（46 字节）
   4,16,90,...   ← Write Multiple Registers
   4,16,16,...   ← 写响应（16 字节）
   ```
2. **轮询间隔 ≈ 1.5–2 秒**
3. **大量 `?` 缺失**——读请求/写请求上下文不同时，另一侧的字段不存在（**结构缺失，非噪声**）
4. **存在 `4.99e-39`、`-0` 等异常值**——可能是传感器浮点错误或攻击痕迹

---

## 二、ARFF → CSV 转换

### 2.1 输出

| 指标 | 值 |
|------|------|
| 文件路径 | `C:\work\Claude\issue\IanArffDataset.csv` |
| 总行数 | 274,629（1 表头 + 274,628 数据） |
| 大小 | 17.8 MB（18,614,243 字节） |
| 编码 | UTF-8 |
| 分隔符 | 逗号 |
| 缺失值 | 保留为 `?`（与原 ARFF 一致） |

### 2.2 转换脚本

`C:\work\Claude\issue\arff_to_csv.py`

可重复使用；可调整分隔符、缺失值替换、列过滤等。

### 2.3 表头

```csv
address,function,length,setpoint,gain,reset rate,deadband,cycle time,rate,system mode,control scheme,pump,solenoid,pressure measurement,crc rate,command response,time,binary result,categorized result,specific result
```

---

## 三、入侵检测预处理方案

### 3.1 流水线总览

```
原始 CSV
    ↓
① 加载 + 类型转换
    ↓
② 缺失值处理（按 (address, function) 组填补）
    ↓
③ 特征工程（时间、滚动、协议级）
    ↓
④ 类别编码
    ↓
⑤ 异常值裁剪
    ↓
⑥ 时序切分（70/30，前段训练后段测试）
    ↓
⑦ 类别平衡（仅训练集）
    ↓
⑧ 归一化（RobustScaler，仅 fit on train）
    ↓
⑨ 特征选择（可选：mutual_info）
    ↓
模型可用的 X_train / y_train / X_test / y_test
```

### 3.2 完整可运行代码

```python
import pandas as pd, numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb

# ── ① 加载 ──
df = pd.read_csv("IanArffDataset.csv").replace("?", np.nan)
df = df.sort_values("time").reset_index(drop=True)

# ── ② 时序/滚动特征 ──
df["dt"]  = df["time"].diff().fillna(0)
ts = pd.to_datetime(df["time"], unit="s")
df["hour"] = ts.dt.hour
df["dow"]  = ts.dt.dayofweek
for col in ["pressure measurement", "crc rate"]:
    df[f"{col}_diff"]   = df[col].diff().fillna(0)
    df[f"{col}_rmean4"] = df[col].rolling(4).mean().fillna(df[col].median())
    df[f"{col}_rstd4"]  = df[col].rolling(4).std().fillna(0)

# ── ③ 协议级衍生 ──
df["is_read"]  = (df["function"] == 3).astype(int)
df["is_write"] = (df["function"] == 16).astype(int)

# ── ④ 缺失值：按 (address, function) 组内中位数填补 ──
miss_cols = ["setpoint","gain","reset rate","deadband","cycle time",
             "rate","system mode","control scheme","pump","solenoid",
             "pressure measurement"]
df[miss_cols] = df.groupby(["address","function"])[miss_cols].transform(
    lambda s: s.fillna(s.median())
)
df[miss_cols] = df[miss_cols].fillna(0)  # 兜底

# ── ⑤ 异常值裁剪 ──
df["pressure measurement"] = df["pressure measurement"].clip(0, 100)

# ── ⑥ 时序划分 ──
y = df["specific result"]
X = df.drop(columns=["binary result","categorized result","specific result"])
split = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── ⑦ 归一化 ──
scaler = RobustScaler()
num_cols = X_train.select_dtypes("number").columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# ── ⑧ 训练（LightGBM） ──
model = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    class_weight="balanced", n_jobs=-1, random_state=42
)
model.fit(X_train, y_train)

# ── ⑨ 评估 ──
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
```

### 3.3 各步骤详解

#### ① 数据加载与类型转换
- `replace("?", NaN)` 必须先做（ARFF 用 `?` 表示缺失，pandas 默认不识别）
- 15 个 `real` 列强制 `float`；标签列转 `int`

#### ② 缺失值处理 ⚠️ 重要
**不能 `dropna()`**——会丢失 90%+ 数据，攻击样本几乎被删光。

推荐：**按 `(address, function)` 分组后用组内中位数填补**
```python
df[miss_cols] = df.groupby(["address","function"])[miss_cols].transform(
    lambda s: s.fillna(s.median())
)
```
理由：数据是 4 行一组的 Modbus 轮询，读不到的值会出现在写响应里，反之亦然——同一 (address, function) 组内的中位数是最自然的填补基准。

可选增强：加 `_isna` 指示列——某些攻击只触发 Read 不触发 Write，缺失本身是信号。

#### ③ 特征工程
- **时间特征**：`dt = time.diff()` 捕获异常间隔；`hour`、`dayofweek` 捕获周期模式
- **滚动特征**（window=4 = 一个 SCADA 周期）：
  - `pressure measurement_diff / _rmean4 / _rstd4`
  - `crc rate_diff / _rmean4 / _rstd4`
- **协议特征**：`is_read`、`is_write` 显式区分命令方向

#### ④ 类别编码
- 标签列本就是 int，无需额外编码
- 若新增 `(function, length)` 组合特征，用 `LabelEncoder`

#### ⑤ 异常值清洗
- `pressure measurement` 物理约束为 0~100，`clip(0, 100)`
- `4.99e-39` / `-0` 等极小值 → 置 NaN → 重新填补

#### ⑥ 数据集划分 ⚠️ 关键
**绝对不能** `train_test_split(random_state=...)` —— 同一时刻的相邻记录会同时进 train/test，**时间泄露**，准确率虚高 20%+。

正确做法：
```python
df = df.sort_values("time")
split = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
```

#### ⑦ 类别平衡 ⚠️ 必要
Normal = 78%，35 个攻击类仅 22%。

两种方案：
- **方案 A**：`class_weight="balanced"`（推荐，最稳妥）
  ```python
  lgb.LGBMClassifier(class_weight="balanced", ...)
  ```
- **方案 B**：仅在训练集上做 SMOTE
  ```python
  from imblearn.over_sampling import SMOTENC
  sm = SMOTENC(categorical_features=cat_idx, random_state=42)
  X_res, y_res = sm.fit_resample(X_train, y_train)
  ```
  **严禁**在测试集上做 SMOTE。

#### ⑧ 归一化
- 树模型（RF / XGBoost / LightGBM）→ 可跳过
- 线性模型 / 神经网络 / SVM → 必须
- 推荐 `RobustScaler`（抗异常值）
- ⚠️ `fit_transform` 只在 train，`transform` 用于 test

#### ⑨ 特征选择
可选，用 `mutual_info_classif` 排序，保留 Top-K：
```python
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X_train, y_train, random_state=42)
```

### 3.4 避坑清单

| # | 坑 | 后果 | 正确做法 |
|---|----|------|---------|
| 1 | `train_test_split` 随机切分 | **时间泄露**，准确率虚高 20%+ | `TimeSeriesSplit` 或 70/30 前/后段 |
| 2 | 直接 `dropna` | 丢 90% 数据，攻击样本几乎删光 | 填补 + 缺失指示符 |
| 3 | 测试集也做 SMOTE | 评估指标失真 | 仅训练集 |
| 4 | 三个标签列同时作 y | 冗余目标冲突 | 二分类用 `binary result`；多分类用 `specific result` |
| 5 | `fit_transform` 同时用于 train+test | 测试集分布泄漏 | `fit` 只在 train |
| 6 | 把 `address` 当类别独热 | 未见 address 时模型崩溃 | 树模型保 int；线性模型用 target encoding |
| 7 | 树模型也强制归一化 | 浪费算力 | 树模型跳过归一化 |

---

## 四、产出文件清单

| 文件 | 大小 | 用途 |
|------|------|------|
| `C:\work\Claude\issue\IanArffDataset.arff` | 17.5 MB | 原始数据 |
| `C:\work\Claude\issue\IanArffDataset.csv` | 17.8 MB | 转换后 CSV（274,629 行） |
| `C:\work\Claude\issue\arff_to_csv.py` | ~2 KB | ARFF→CSV 转换脚本（可复用） |
| `C:\work\Claude\issue\REPORT.md` | 本文件 | 本报告 |

---

## 五、推荐后续步骤

1. **基线模型**：先跑 LightGBM 多分类（`specific result`，36 类），看 Macro-F1
2. **特征消融**：单独验证时序/滚动特征的贡献
3. **二分类 vs 多分类对比**：先看 `binary result` 任务的 F1，再扩展到 36 类
4. **不平衡策略对比**：`class_weight` vs `SMOTE` vs 代价敏感学习
5. **模型解释**：用 SHAP 分析哪些特征对攻击识别最重要（符合 SCADA 领域可解释性要求）
