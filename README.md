# Modbus SCADA 仿真 + 攻击注入 + MCU 部署

> PC 端完整 SCADA 仿真栈 — 用于气管道入侵检测模型的训练、测试、部署验证

---

## 🎯 项目目标

为 **TCN+SE 23-dim 入侵检测模型**(详见项目历史: [[project-2026-07-26-daily-report]], [[project-tcn-v4-se]])提供:
1. **真实 Modbus TCP 仿真服务器** — 完整模拟 IanArffDataset 气管道 SCADA 工艺
2. **7 类攻击注入器** — 复现 NMRI/CMRI/MSCI/MPCI/MFCI/DoS/Recon
3. **MCU 推理接口** — 输出 (8, 23) 窗口 JSONL,直接对接 STM32H743 部署
4. **PC 端 bit-perfect 验证** — 验证 C 推理代码与 PyTorch Python 一致

```
┌─────────────────┐  攻击注入  ┌──────────────────┐
│ modbus_scada_   │◀──────────│ attack_injector  │
│  server.py      │           │  .py             │
│  (PC 仿真)      │──────────▶│  (7 类攻击)      │
└─────────────────┘  工艺数据  └──────────────────┘
        │                                  │
        │ 真实 Modbus TCP                  │ 输出 (8,23) JSONL
        ▼                                  ▼
   ┌─────────┐                       ┌──────────┐
   │ HMI/    │                       │ TCN+SE   │
   │ SCADA   │                       │ ch=32    │
   │ Client  │                       │ 5-seed   │
   └─────────┘                       └──────────┘
                                            │
                                            ▼
                                       ┌──────────┐
                                       │ STM32H743│
                                       │ 部署验证  │
                                       └──────────┘
```

---

## 🚀 5 秒快速启动

### Windows 用户 (推荐)

```cmd
REM 1. 双击 modbus_menu.bat (出现菜单)
REM 2. 选 [2] 在新窗口启动服务器 (不阻塞)
REM 3. 选 [3] 客户端测试
REM 4. 完成!
```

### 命令行用户

```bash
# 终端 1: 启动服务器
python modbus_scada_server.py start --port 5020

# 终端 2: 客户端测试
python modbus_scada_server.py client --port 5020 --read 5
```

### 一键演示 (无需多终端)

```bash
python modbus_scada_server.py demo --port 5020
# 自动: 启动服务器 + 3 次正常读 + 3 类攻击 + 最终状态
```

---

## 📂 文件结构

```
C:\work\Claude\Issue\
├── 🐍 Python 核心
│   ├── modbus_scada_server.py     # Modbus TCP 服务器 + 客户端 (600 行)
│   ├── attack_injector.py         # 7 类攻击注入器 (937 行)
│   └── arff_to_csv.py             # ARFF→CSV 转换工具
│
├── 🪟 Windows 启动器 (.bat)
│   ├── modbus_menu.bat            # ★ 推荐入口: 菜单式
│   ├── start_server.bat           # 启动服务器
│   ├── start_client.bat           # 客户端测试
│   └── start_demo.bat             # 一键演示
│
├── 📊 数据集
│   ├── IanArffDataset.arff        # 原始 SCADA 数据 (274k 记录, 17 维)
│   └── IanArffDataset.csv         # CSV 格式
│
├── 🤖 训练代码 (TCN+SE)
│   ├── train_tcn_23dim_*.py       # 23-dim 冠军模型训练
│   ├── train_tcn_27dim_*.py       # 27-dim baseline
│   └── preprocess_v2_scada.py     # 特征工程 v2
│
├── 🛠️ MCU 部署 (STM32)
│   ├── KeilH743/                  # STM32H743 Keil 工程
│   └── TestH743/                  # 测试代码
│
├── 📁 生成数据 (运行后)
│   └── attack_test/
│       ├── raw_attack_dataset.csv
│       ├── mcu_attack_log.jsonl   # ★ (8, 23) 窗口格式
│       └── report.json
│
└── 📄 文档
    └── README.md                  # 本文件
```

---

## 🔧 详细使用

### 1. `modbus_scada_server.py` — Modbus TCP 服务器

#### 启动服务器
```bash
# 最简
python modbus_scada_server.py start --port 5020

# 自定义
python modbus_scada_server.py start --port 5020 --unit-id 4
python modbus_scada_server.py start --port 5020 --no-auto-step  # 静态寄存器
```

#### 客户端测试
```bash
# 读 5 次, 间隔 1s
python modbus_scada_server.py client --port 5020 --read 5

# 写 setpoint=25
python modbus_scada_server.py client --port 5020 --write-sp 25

# 远程连接
python modbus_scada_server.py client --host 192.168.1.100 --port 5020 --read 10
```

#### Python 代码调用
```python
from modbus_scada_server import ModbusSCADAClient, REG

client = ModbusSCADAClient("127.0.0.1", 5020)
client.connect()

# 读压力
p = client.read_pressure()  # 0.00-1.50

# 写单个寄存器
client.write_register(REG["setpoint"], 50)

# 写多个寄存器
client.write_registers(0, [50, 200, 2, 5, 1, 0, 0, 1, 1, 1])

# 读全部
regs = client.read_all()  # dict

client.close()
```

#### Modbus 寄存器映射

| 地址 | 名称 | 类型 | 范围 | 初始值 | 备注 |
|---|---|---|---|---|---|
| 0 | setpoint | RW | 0-65535 | 10 | 目标压力 |
| 1 | gain | RW | 0-65535 | 115 | 比例增益 |
| 2 | reset_rate | RW | 0-65535 | 2 | 积分速率 |
| 3 | deadband | RW | 0-65535 | 5 | 死区 |
| 8 | pump | RW | 0/1 | 0 | 0=off, 1=on |
| 9 | solenoid | RW | 0/1 | 0 | 0=close, 1=open |
| 20 | pressure_x100 | RO* | 0-150 | 70 | 0.70 bar |
| 22 | crc_rate | RO* | 10000-20000 | 17000 | CRC 计数 |
| 24 | flow_rate | RO* | 0-100 | 50 | 流量 × 100 |
| 25 | temperature | RO* | 200-400 | 250 | 25.0°C |

*RO = 物理上只读,但 Modbus 协议不限制 (模拟器可写用于 NMRI 攻击测试)

完整映射: `modbus_scada_server.py:REG`

#### 协议特性

- ✅ **FC=3** Read Holding Registers
- ✅ **FC=6** Write Single Register
- ✅ **FC=16** Write Multiple Registers
- ✅ 多客户端并发 (每客户端独立线程)
- ✅ 工艺仿真 (ProcessModel — 压力一阶滞后,流量泵阀控制)
- ✅ 100% 自实现, **无 pymodbus 依赖** (绕开 3.x 弃用陷阱)

---

### 2. `attack_injector.py` — 攻击注入器

#### 7 类攻击 (与 IanArffDataset categorized result 对齐)

| 编号 | 名称 | 中文 | 层级 | 注入方式 |
|---|---|---|---|---|
| 1 | NMRI | 简单响应注入 | L3 数据 | 篡改压力/CRC |
| 2 | CMRI | 复杂响应注入 | L3 数据 | 工艺一致漂移 |
| 3 | MSCI | 状态命令注入 | L3 数据 | 异常 system_mode |
| 4 | MPCI | 参数命令注入 | L3 数据 | 越限 setpoint/gain |
| 5 | MFCI | 功能码注入 | L2 协议 | 非法 FC (43/136/171) |
| 6 | DoS | 拒绝服务 | L4 时序 | 突发流量/重放 |
| 7 | Recon | 扫描嗅探 | L1+L4 协议 | 扫描 address/FC |

#### 生成攻击数据集
```bash
# 快速演示
python attack_injector.py demo --out ./demo.jsonl

# 完整数据集 (CSV + NPY + JSONL)
python attack_injector.py generate \
    --n-normal 2000 --n-attack 800 \
    --out-dir ./attack_test

# 输出文件:
#   raw_attack_dataset.csv   - 17 维原始 + 标签
#   X_attack_27dim.npy       - 27 维特征
#   X_attack_23dim.npy       - 23 维 (KEEP_23)
#   mcu_attack_log.jsonl     - (8, 23) 窗口 — ★ MCU 直接用
#   report.json              - 元数据
```

#### 启动 Modbus 注入服务器 (实时)
```bash
# 每 5% 概率注入 MFCI, 10% 概率注入 MPCI
python attack_injector.py server --port 5020 \
    --inject-cmd "5:0.05,4:0.10"
```

#### 跑推理评估
```bash
# 加载 .npy 跑 TCN+SE 推理
python attack_injector.py infer \
    --input X_attack_23dim.npy \
    --model tcns_ch32_seed42.pt \
    --threshold 0.5
```

---

### 3. Windows .bat 启动器

#### modbus_menu.bat (推荐入口)

```
============================================================
  Modbus SCADA 工具箱 v1.0
============================================================
  [1] 启动服务器 (前台,默认 5020)
  [2] 启动服务器 (后台新窗口,默认 5020)
  [3] 客户端测试 (读 5 次)
  [4] 客户端写 setpoint=25
  [5] 一键演示
  [6] 生成攻击数据集 (500+200)
  [7] 生成攻击数据 (2000+800)
  [8] 验证 Python 环境
  [9] 查看 5020 端口占用
  [0] 退出
```

#### start_server.bat 用法
```cmd
start_server.bat              REM 端口 5020
start_server.bat 5021         REM 端口 5021
start_server.bat 5021 4       REM 端口 5021, unit_id=4
```

#### start_client.bat 用法
```cmd
start_client.bat                              REM 默认 localhost:5020
start_client.bat 127.0.0.1 5020              REM 指定主机端口
start_client.bat 127.0.0.1 5020 10 0.5       REM 读 10 次, 间隔 0.5s
start_client.bat 127.0.0.1 5020 - 25         REM 写 setpoint=25
```

#### start_demo.bat 用法
```cmd
start_demo.bat           REM 端口 5020
start_demo.bat 5021      REM 端口 5021
```

---

## 🔄 完整工作流

### 场景 A: 训练数据生成 (离线)

```bash
# 1. 生成攻击数据集
python attack_injector.py generate \
    --n-normal 5000 --n-attack 1000 \
    --out-dir ./attack_data

# 2. 加载 NPY 训练 TCN+SE
python train_tcn_23dim.py --data ./attack_data/X_attack_23dim.npy
```

### 场景 B: PC 端 bit-perfect 验证

```bash
# 1. 启动服务器
python modbus_scada_server.py start --port 5020 &

# 2. 客户端发送测试样本
python -c "
from modbus_scada_server import ModbusSCADAClient
c = ModbusSCADAClient('127.0.0.1', 5020)
c.connect()
# 注入 1000 个攻击样本
for i in range(1000):
    c.write_register(0, 999 if i%2 else 10)  # 攻击 vs 正常
print('Done')
"

# 3. 在 STM32H743 上跑 C 推理
# 4. 对比 PyTorch 输出 vs C 输出 (max_diff ≤ 1e-5)
```

### 场景 C: MCU 部署 + UART 注入攻击

```
┌─────────────┐                ┌─────────────┐
│ PC 服务器   │──Modbus TCP──▶│ STM32H743   │
│ 5020        │                │ UART 注入   │
└─────────────┘                │ 攻击数据     │
                                └─────────────┘
                                     │
                                     ▼
                                ┌─────────────┐
                                │ TCN+SE      │
                                │ 推理 → SD卡 │
                                └─────────────┘
```

```bash
# 终端 1: PC 端
python modbus_scada_server.py start --port 5020

# 终端 2: STM32H743 烧录 ch32_inference.bin, 启动 UART 监听
# (按 MCU 代码流程)

# 终端 3: PC 端注入攻击
python -c "
from modbus_scada_server import ModbusSCADAClient
c = ModbusSCADAClient('127.0.0.1', 5020)
c.connect()
c.write_registers(0, [999, -500 & 0xFFFF, 2, 5, 1, 0, 0, 1, 1, 1])  # MPCI
"

# 终端 4: 验证 MCU 端警报
# 串口监视器 (PuTTY/MobaXterm) 应看到 F1m>0.5 警报
```

### 场景 D: 一键演示 (Demos)

```bash
# 1. 完整 SCADA 演示
python modbus_scada_server.py demo

# 2. 攻击注入演示
python attack_injector.py demo

# 3. Windows 菜单 (推荐)
modbus_menu.bat
# → 选 [5] 演示
# → 选 [7] 生成大攻击集
```

---

## 🛠️ 故障排除

### Q1: 端口被占用 `Address already in use`
```bash
# 解决 1: 换端口
python modbus_scada_server.py start --port 5021

# 解决 2: 找到占用进程
netstat -ano | findstr :5020
tasklist /fi "PID eq <PID>"
taskkill /PID <PID> /F

# 解决 3: Windows 菜单 [9] 查看端口
modbus_menu.bat
```

### Q2: Python 未找到
```bash
# 检查安装
python --version

# 如果用 py launcher
py -3 modbus_scada_server.py start

# 或修改 .bat 中的 python 为 py
```

### Q3: pymodbus 冲突
本项目**完全不用 pymodbus**!如果你的环境有 pymodbus 3.x,可能干扰:
```bash
pip uninstall pymodbus  # 可选
```

### Q4: numpy 未装 (NPY 输出失败)
```bash
pip install numpy

# 即使没装, CSV + JSONL 仍可用
python attack_injector.py generate --n-normal 500 --n-attack 200 --out-dir ./test
```

### Q5: 编码乱码 (.bat 中中文乱码)
```cmd
REM 已在 .bat 中加 chcp 65001 (UTF-8)
REM 如果仍乱码: 用 Windows Terminal 替代 cmd.exe
```

### Q6: 客户端连不上
```bash
# 检查服务器是否真的在跑
netstat -ano | findstr :5020

# 测试 TCP 连接
Test-NetConnection 127.0.0.1 -Port 5020   # PowerShell
# 或
telnet 127.0.0.1 5020                     # 如果有 telnet
```

### Q7: 攻击注入后无效果
检查注入器是否真的注入了:
```python
from attack_injector import SCADASimulator, AttackInjector
sim = SCADASimulator()
inj = AttackInjector(sim, 4)  # MPCI
cycle = sim.tick()
print("Before:", cycle[2].setpoint)  # 10.0
cycle = inj.inject_into_cycle(cycle)
print("After :", cycle[2].setpoint)  # 越限值
```

---

## 🧪 验证 Checklist

部署前必过:

```
[ ] 服务器启动 (port 5020, netstat 看到 LISTENING)
[ ] 客户端连接成功 (Connected to 127.0.0.1:5020)
[ ] FC=3 读正常 (压力在 0.5-0.8 波动)
[ ] FC=6 写正常 (setpoint 10→25, 读回确认)
[ ] FC=16 写正常 (10 寄存器批量写入)
[ ] 工艺仿真工作 (泵开 → 流量增加)
[ ] 7 类攻击生成 (MPCI 越限, MFCI 非法 FC 等)
[ ] JSONL 格式正确 (shape=(8, 23), 无 NaN)
[ ] NPY 格式正确 (numpy 可加载, dtype=float32)
[ ] 与 TCN+SE 推理对接 (mcu_attack_log.jsonl 输入模型)
[ ] STM32H743 烧录后 bit-perfect 验证 (max_diff ≤ 1e-5)
[ ] 24h 稳定性测试 (服务器无崩溃, 内存不泄漏)
[ ] 攻击警报测试 (MPCI 注入 → 模型输出 > 0.5 概率)
```

---

## 📊 项目历史参考

本项目与以下模型/部署报告关联:

- **TCN+SE 23-dim 冠军模型** — [[project-2026-07-26-daily-report]]
- **5-seed ONNX 部署** — [[project-2026-08-18-daily-report]]
- **STM32H743 完整移植** — [[project-h743-ch32-deploy]]
- **Modbus 协议特征工程** — [[project-preprocess-v2]]
- **SCADA 攻击综述 (L1-L5)** — [[modbus-semantic-feature-lit-review]]
- **14 模型 19 维横评** — [[project-14models-19dim-comparison]]

---

## 📝 依赖

| 依赖 | 必需 | 用途 |
|---|---|---|
| Python 3.8+ | ✅ | 核心运行时 |
| numpy | ❌ (可选) | NPY 输出, 无也能跑 CSV+JSONL |
| pymodbus | ❌ | 完全不用 (服务器自实现) |
| scapy | ❌ | 仅可选 fuzzing 功能 |
| Windows / Linux | ✅ | 跨平台 (Linux 同样可用) |

```bash
# 最小安装
python --version  # 检查 3.8+

# 可选
pip install numpy
```

---

## 🚦 快速命令速查

```bash
# 服务器
python modbus_scada_server.py start --port 5020
python modbus_scada_server.py start --port 5020 --no-auto-step

# 客户端
python modbus_scada_server.py client --port 5020 --read 5
python modbus_scada_server.py client --port 5020 --write-sp 25

# 演示
python modbus_scada_server.py demo
python attack_injector.py demo

# 攻击数据
python attack_injector.py generate --n-normal 2000 --n-attack 800 \
    --out-dir ./attack_data

# 推理
python attack_injector.py infer --input X_attack_23dim.npy \
    --model tcns_ch32_seed42.pt --threshold 0.5

# Windows 启动器
modbus_menu.bat          # 菜单
start_server.bat         # 服务器
start_client.bat         # 客户端
start_demo.bat           # 演示
```

---

## 📜 许可

仅供学术研究和安全测试使用。禁止用于未授权系统的攻击。

---

**最后更新**: 2026-09-02
**作者**: SCADA IDS Team
**版本**: v1.0
