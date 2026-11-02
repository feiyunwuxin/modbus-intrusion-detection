# TFLM vs Route E 对照报告

> **状态 (2026-10-28)**: PC 侧全部通过 ✅ / MCU 工程构建通过 ✅ / **上机全量回放已完成 ✅**
> （日志 `logs/test_20261027-232828_auto-40ms.csv`，17164 窗，**MCU TFLM 与 Route E 逐窗一致**，
> 仅回放尾部截断的末窗不同 —— 见 [§7.3](#73-上机结果-2026-10-28-全量回放)）
>
> 本文所有数字都可复现，命令见 [§8](#8-可复现清单)。**正结论或负结论都能写进论文**：
> 到目前为止没有出现"跑不起来"或"数字对不上"的不可解释现象。

---

## 1. 一句话结论

**PC 侧 TFLite (BUILTIN_REF) 与部署真值 Route E (手写 C kernel) 在部署区 4664 个窗口上
逐位等价（0 次决策翻转，指标完全相同：F1 0.8671 / FPR 5.91% / R 80.75% / P 93.61%）；
MCU 侧用一份全新的独立 Keil 工程把同一张图跑起来，构建 0 error 0 warning，
Flash 274.6 KB / 额外 RAM 96 KB arena；
2026-10-28 首次上机全量回放（17164 窗）实测：TFLM 固件与 Route E 固件
**逐窗决策完全一致**（唯一例外是回放尾部被截断的末窗 W=17163），
部署区 **F1 0.8668 / FPR 5.91% / TP 1949 / FP 133 / FN 466**，
与 10-27 21:27 Route E FP32 上机数字**完全相同** ⇒ TFLM 后端在 MCU 上没有引入任何精度损失。**

---

## 2. 为什么要"从头再建一个 TFLM 工程"

2026-10-15~18 的旧 TFLM 路线失败归档（[[project-h743-int8-tflm-line]]）留下两个致命伤：

| 旧路线的病 | 根因 | 本次怎么防 |
|---|---|---|
| `.tflite` 里有 PACK / SHAPE / STRIDED_SLICE / UNPACK | 用 Keras `Conv1D` 直接转 TF，没意识到 TFLite 的 lowering 产物 | **自己写 ONNX→TF→TFLite 导出器**，用 `tf.nn.conv1d(dilations=)` 让 TFLite 直接出带 `dilation_w_factor` 的 CONV_2D；再写一个**静态兼容性检查器**，逐条断言 op 集合 / dilation / shape |
| AllocateTensors 在 `Op[0]=EXPAND_DIMS` 直接失败 | OpResolver 漏注册 | 导出时用 manifest 记录**实际出现的 op 集合**，固件里按 manifest **逐个注册并检查返回值**；`tools/tflm_compat_check.py` 每条断言都引用 TFLM 源码行号 |
| 5 个 controlled input 输出**恒定** | 忽略 `AllocateTensors` / `Invoke` 返回码 + DCache stale | 运行时**必须检查返回码**，失败就拒绝推理；DCache 三步（clean 输入 / invalidate 输出 / clean-invalidate 输出后）；**上机自检从第一天就装上**（`'T'` 命令） |
| 结论"是 TFLM ref impl 跑这个模型是坏的" | 没有 PC↔MCU 对拍基线 | PC 侧先把 TFLite / ORT / 手写 kernel 三方对齐（0 翻转），再造 MCU 侧；**任何 MCU 差异都能定位到具体算子** |

设计前提直接取自手写 kernel 路线的踩坑记录：

- **conv 对齐语义**：统一为 `t_in = t + k*d - pad`（对称 same-padding），与修好的 Route E kernel 一致。
- **(C,T) vs (T,C) 陷阱**：本图输入设计成 `(1, 23, 16)` **channel-major**，跟 MCU 的 `sw_buf[c*16+t]` **逐字节一致** → MCU 端**零转置**。
- **时间精度 skew**：固件不用改，只把 PC 侧数据源换成 `IanArffDatasetF32.csv`。
- **alarm 帧归窗**：app 层归窗逻辑一行没改，保证与 Route E 同口径。
- **源绑定验证**：PC 参考值不是手抄公式算的，是从 `model_tflm_fp32_s42.tflite` 跑出来的。

---

## 3. 模型与导出（G0）

```
experiments/tflm_fp32/
├── tools/onnx_to_tflite.py        # ONNX → TF → TFLite 导出器（含 4 条硬断言）
├── tools/tflm_compat_check.py     # 静态 TFLM 兼容性检查（每条断言引用 TFLM 源码行号）
├── tools/tflite_pc_reference.py   # PC 参考：TFLite vs ORT vs 手写 kernel + 5 controlled input
├── tools/tflite_to_c_array.py     # .tflite → C 数组（含回读自检）
├── tools/gen_keil_project.py      # 生成 H743test_TFLM.uvprojx（**不改**既有工程）
├── tools/tflm_deploy_parity.py    # MCU↔PC 逐窗对拍 + 部署区 4 指标（G3）
└── artifacts/
    ├── model_tflm_fp32_s42.tflite   (93,596 B, sha256 eaee5340…)
    ├── model_manifest.json          (op 集合 / shape / dilation / sha256)
    ├── tflm_compat_report.json
    ├── tflm_selftest_ref.json       (5 个 controlled input 的 PC 参考值)
    ├── tflm_pc_report.json          (PC 侧三方 parity + 部署区指标)
    ├── pc_parity_logits.npz         (逐窗 logit: TFLite / ORT)
    └── windows_f32_mcu.npz          (逐窗输入 X[17164,23,16] + 标签)
```

**导出结果**

| 项 | 值 |
|---|---|
| op 集合（9 种） | ADD, CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, LOGISTIC, MEAN, MUL, RESHAPE, TRANSPOSE |
| op 总数 | 50 |
| 空洞卷积 `dilation_w_factor` | 1 / 2 / 4（三个 TCN block） |
| 图输入 | `[1, 23, 16]` float32，**channel-major** |
| 图输出 | `[1]` float32 = **logit**（sigmoid 在 MCU 端补，与 Route E 同语义） |
| 不含 | PACK / SHAPE / STRIDED_SLICE / SPACE_TO_BATCH_ND / BATCH_TO_SPACE_ND |

---

## 4. PC 侧结果（G0b）—— 已经跑完

`artifacts/tflm_pc_report.json`：

| 对比 | max \|Δ\| | 决策翻转 | 窗口数 |
|---|---|---|---|
| TFLite (BUILTIN_REF) vs **ONNX Runtime** | 1.144e-05（logit） | **0** | 17,164 |
| TFLite vs **手写 C kernel**（部署真值） | 1.221e-06（prob） | **0** | 17,164 |
| TFLite vs 手写 kernel 的 ref 实现 | 9.537e-06（logit） | **0** | 17,164 |

**部署区 4 指标（TFLite 与手写 kernel 完全相同）**

```
TP 1950  FP 133  TN 2116  FN 465
P 0.9361   R 0.8075   F1 0.8671   FPR 0.0591
```

这条数字跟官方基线（[[project-2026-10-21-h743-final-baseline]]）**逐位相同** ——
说明新的 TFLite 导出路径**独立地重新推导出了**部署真值，而不是抄来的。

**为什么用 BUILTIN_REF 而不是默认 interpreter**：TF 2.20 的 `tf.lite.Interpreter` 会偷偷挂 XNNPACK delegate
（日志里 "Created TensorFlow Lite XNNPACK delegate for CPU"），而 TFLM 的 float 路径走的是
`reference_ops::*`（conv.cc:61 / fully_connected.cc:141 / add.cc:52 / logistic.cc:49）。
用 XNNPACK 当 PC 参考等于拿另一个实现去对拍。**必须 `experimental_op_resolver_type=BUILTIN_REF`。**

---

## 5. MCU 工程（G1）—— 构建通过

### 5.1 文件布局

| 文件 | 作用 |
|---|---|
| `MCU/MDK-ARM/H743test_TFLM.uvprojx` | **新建**的独立工程，1 个 target `H743test_TFLM` |
| `MCU/MDK-ARM/H743_TFLM_arena.sct` | 链接脚本：arena 放 D2 SRAM `0x30000000` |
| `MCU/Core/Src/tflm_runtime.cpp` | 唯一 C++ TU：MicroInterpreter / OpResolver / arena / DCache / 自检 |
| `MCU/Core/Src/ch32_inference_tflm.c` | C shim，API 与 Route E **完全同名同义** |
| `MCU/Core/Inc/tflm_defs.h` | 24 个 libcxx 宏 fallback（`--include` 注入） |
| `MCU/Core/Inc/tflm_config.h` | 模型维度 / op 数 / arena 大小 / arena section 名 |
| `MCU/Core/Inc/model_tflm_fp32_s42_data.h` + `…_data.c` | flatbuffer → Flash 常量数组（8 字节对齐 + `_Static_assert`） |
| `MCU/Core/Src/main.c`、`stm32h7xx_it.c` | **只在 `#ifdef CH32_BACKEND_TFLM` 下**加 `'T'` 命令和 banner 分流 |

**既有工程零改动**：`H743test.uvprojx` 的两个 target（`H743test` / `H743test_INT8`）、
`ch32_inference.c`、`ch32_inference_int8.c` 一行没改。Route E 固件**重新构建 0 error**，
且二进制里搜不到任何 TFLM 字符串（`TFLite Micro Initial` / `[T-CMD]` / `[TFLM]` 全部 absent）。

### 5.2 链接结果（`H743test_TFLM.map`）

```
g_tflm_arena       0x30000000   98,304 B   (D2 SRAM, .bss.tflm_d2)
g_tflm_model_s42   0x08029698   93,596 B   (Flash, .rodata)
Execution Region ER_IROM1  0x08000000  Size 0x44a88 = 281,224 B (274.63 KB)
Execution Region RW_AXI    0x24000000  Size 0x6090  =  24,720 B
Execution Region RW_D2     0x30000000  Size 0x18000 =  98,304 B
Program Size: Code=166088 RO-data=115136 RW-data=4448 ZI-data=249648
```

> arena 为什么放 D2 而不是 DTCM：DTCM 128 KB 已经被 startup 的 `HEAP` 占满；
> D2 SRAM（`0x30000000`，256 KB）完全空着，正好放 96 KB arena。
> `H743_TFLM_arena.sct` 把 `RW_D2` 放在 `RW_DTCM` **之前**，不依赖 `.ANY` 选择器的优先级细节。

### 5.3 与 Route E 的体积对比

| | Route E | TFLM | 差 |
|---|---|---|---|
| Flash (ER_IROM1) | 146.11 KB | **274.63 KB** | **+128.52 KB** |
| 其中模型权重 | ~85 KB（copy 到 AXI RAM） | 91.4 KB（**留在 Flash**） | — |
| RAM 常驻 | 108.2 KB (AXI) | 24.2 KB (AXI) + **96 KB arena (D2)** | +12.2 KB |

**TFLM 的代价在 Flash，不在 RAM**：flatbuffer 本身比裸权重大，而且 TFLM 运行时多带 ~40 KB 代码。
但 TFLM **不用把权重拷进 RAM**（Route E 要），所以 RAM 反而略省。

---

## 6. MCU 侧的 uVision 坑（本次实测，值得单独记）

### 6.1 `FileType 5` 会被**静默跳过**

Keil uVision（v5.43）把 `<FileType>5</FileType>` 当作 "C++ Source file"，
但 **build 时直接不编译这个文件** —— 日志里连一行 `compiling xxx.cc` 都没有，
链接时才报一堆 `L6218E: Undefined symbol`。本次把 FileType 1~9 × UseCPPCompiler 0~3
共 36 种组合全试了一遍才定位。

**可行写法**（`tools/gen_keil_project.py` 里已固化，并带硬断言）：

```xml
<File>
  <FileName>micro_interpreter.cc</FileName>
  <FileType>1</FileType>            <!-- 一律 1, 不要 5 -->
  <FilePath>../thirdParty/tflite-micro/tensorflow/lite/micro/micro_interpreter.cc</FilePath>
  <FileOption><CommonProperty><UseCPPCompiler>1</UseCPPCompiler>...</CommonProperty>
    <FileArmAds><Cads><VariousControls>
      <MiscControls>--include ../Core/Inc/tflm_defs.h -x c++ -std=c++17 -fno-exceptions -fno-rtti</MiscControls>
      <IncludePath>...;C:\Keil_v5\ARM\ARMCLANG\include\libcxx;C:\Keil_v5\ARM\ARMCLANG\include</IncludePath>
    </VariousControls></Cads></FileArmAds>
  </FileOption>
</File>
```

`-x c++` 是**必须**的：uVision 传给 armclang 的语言是 C，
不给 `-x c++` 会报 `error: invalid argument '-std=c++17' not allowed with 'C'`。
uVision 自己的 `-x c` 排在 `MiscControls` **之前**，所以我们的 `-x c++` 生效。

### 6.2 漏掉工具链声明 → uVision 当成 C51 直接崩

新工程必须整块复制 `<Target>` 的所有子元素，缺一不可：

```xml
<ToolsetNumber>0x4</ToolsetNumber>
<ToolsetName>ARM-ADS</ToolsetName>
<pCCUsed>6240000::V6.24::ARMCLANG</pCCUsed>
<uAC6>1</uAC6>          <!-- 漏了这个 -> "requires 'C51' Toolchain which is not installed" + 段错误 -->
```

### 6.3 libcxx

- Keil 自带 libcxx 在 `C:\Keil_v5\ARM\ARMCLANG\include\libcxx`，**只加到 .cc 的 per-file IncludePath**。
  加到工程级的话，C 文件的 `<stdint.h>` 会被 libcxx 抢走。
- `__config` 在 2026-10-15 被旧路线**就地打过补丁**（pristine 备份在同目录 `__config.bak.2026-10-15`）。
  本次**保持补丁不动**，靠 `tflm_defs.h` 的 `__ARMCOMPILER_LIBCXX` 等 24 个宏把 ARM block 打开。
  ⚠ 这是一个**环境级依赖**，换机器复现时必须先检查 `__config` 有没有这个补丁。

---

## 7. 上机步骤（G2 / G3）—— ⏳ 待烧录

### 7.1 G2：5 个 controlled input 自检

烧录 `H743test_TFLM/H743test_TFLM.hex`，在串口（USART1, 115200）发一个字节 `T`，
会看到：

```
[T-CMD] rx 'T' Size=1, set cmd_rx_ready
[T-CMD] running TFLM controlled-input self-test
[TFLM-TEST] controlled input self-test (5 cases)
[TFLM-TEST] case         logit          prob
[TFLM-TEST] ZEROS    42.251404      1.00000000
[TFLM-TEST] ONES    224.277573      1.00000000
[TFLM-TEST] HALF    115.246002      1.00000000
[TFLM-TEST] NEG1     19.762009      0.99999999
[TFLM-TEST] TWOS    431.759369      1.00000000
[TFLM-TEST] OK: 5 outputs differ
```

**PC 参考值**（`artifacts/tflm_selftest_ref.json`，float32 精确十六进制）：

| case | fill | logit (f32 hex) | logit |
|---|---|---|---|
| ZEROS | 0.0 | `0x1.5202ep+5` | 42.25140380859375 |
| ONES | 1.0 | `0x1.c08e1ep+7` | 224.27757263183594 |
| HALF | 0.5 | `0x1.ccfbe8p+6` | 115.24600219726562 |
| NEG1 | -1.0 | `0x1.3c313p+4` | 19.762008666992188 |
| TWOS | 2.0 | `0x1.afc266p+8` | 431.7593688964844 |

**自动判定**（不用人眼看小数）：

```bash
python experiments/tflm_fp32/tools/tflm_selftest_check.py capture.txt
# 或直接管道: python .../tflm_selftest_check.py --stdin < capture.txt
```

退出码即结论：

| 退出码 | 现象 | 结论 |
|---|---|---|
| 0 | 5 个输出互不相同，且与 PC 参考一致（容差 1e-4） | 运行时 OK，问题在数据通路（特征 / 窗口） |
| 1 | 有 case 与 PC 不一致 | 算子数值实现差异，可逐层 dump 定位 |
| 2 | 5 个输出**恒定** | 图 / 运行时在 MCU 上坏了（AllocateTensors 失败或 DCache stale）——**旧 TFLM 路线的失败签名** |
| 3 | 日志里没有 `[TFLM-TEST]` 行 | 自检没跑起来（`'T'` 没被收到 / 串口没抓到） |

容差取 1e-4：MCU 打印是 `%14.6f`（自身截断 ≤5e-7），加上 float32 累加顺序差异，
1e-4 既能判真差异又不误报。

### 7.2 G3：部署区逐窗对拍

```bash
# 1) PC 回放到板子 (rtu_replay_gui.py), 产出 logs/test_<ts>.csv
# 2) 部署区 4 指标 + 逐窗 parity
python experiments/tflm_fp32/tools/tflm_deploy_parity.py \
       --log logs/test_<ts>.csv --label TFLM \
       --baseline-log logs/test_20261021-203113_auto-40ms.csv --label-baseline RouteE \
       --out experiments/tflm_fp32/artifacts/parity_tflm.json
```

**窗口对齐**（重要，且是实测出来的）：日志里的 `frame_idx` 是**回放内**的帧号（1-based，回放从数据集行 200000 起），
部署区窗号 `w_local = frame_idx // 16 - 1`，对应 PC 参考数组下标 `w_local`。
用 Route E 基线日志对 offset −1 / 0 / +1 / +2 四种对齐各算一遍 `max|Δprob|`：

| offset | max \|Δprob\| | 决策翻转 |
|---|---|---|
| −1 | 0.9252 | 296 |
| **0** | **0.0160** | **0** |
| +1 | 0.9380 | 295 |
| +2 | 0.9380 | 484 |

只有 offset=0 零翻转 → 这个对齐是对的。

**工具在 Route E 基线日志上的自检输出**（证明工具本身正确）：

```
===== Route E (基线) =====
  部署区窗口          : 4664
  MCU 报警窗 (prob>=thr): 2083
  MCU<->PC |Δprob|   : max 0.016000  median 0.000987  p99 0.001000
  决策翻转 (thr=0.5) : 0
    source      TP    FP    TN    FN        P        R       F1      FPR
    MCU       1950   133  2116   465   0.9361   0.8075   0.8671   0.0591
    PC        1950   133  2116   465   0.9361   0.8075   0.8671   0.0591
```

残余 `median |Δprob| ≈ 0.000987` 是**日志量化下界**（`prob_x1000` 是整数），
不是数值差异。

### 7.3 上机结果（2026-10-28，全量回放）

**这是 TFLM 后端的第一次实机全量回放。**

| 项 | 值 |
|---|---|
| 固件 | `MCU/MDK-ARM/H743test_TFLM/H743test_TFLM.hex`（构建 10-27 23:12，0 error） |
| 日志 | `logs/test_20261027-232828_auto-40ms.csv`（10.6 MB，回放 23:28–02:44） |
| 摘要 | `logs/test_20261027-232828_auto-40ms.csv.summary.txt` |
| 覆盖 | **全量**（不是部署区）：17164 窗，274,628 帧 |
| 对拍脚本 | `tflm_vs_pc_window_diff.py` → `tflm_vs_pc_window_diff.json` |

#### 7.3.1 MCU 侧指标（真源码 `WindowEvaluator` 回放，不 Python 复刻）

| 口径 | F1 | FPR | Precision | Recall | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| 全量 (n=17164) | 0.8812 | 0.0227 | 0.9741 | 0.8046 | 7097 | 189 | 8154 | 1724 |
| **部署区 (n=4664, frame≥200001)** | **0.8668** | **0.0591** | 0.9361 | 0.8070 | 1949 | 133 | 2116 | 466 |

> 部署区这行 **和 10-27 21:27 Route E FP32 上机数字逐项相同**（F1 0.8668 / TP 1949 / FP 133 / FN 466，
> 见 [[project-2026-10-27-fp32-reverify-window-diff]]）—— 两条完全不同的实现路径（手写 C kernel vs TFLM ref ops）
> 在同一块板子上给出同一个结果。

#### 7.3.2 逐窗对拍（17164 窗，thr=0.5）

窗口标签用**真源码回放器**（`replay_window_eval.py` 从 `rtu_replay_gui.py` exec 出 `WindowEvaluator` 类），
PC 侧用各方案自己的 prob 数组，窗口网格同为 phase=0 的 17164 窗（第 W 窗覆盖帧 `[16W, 16W+15]`）。

**先验对齐检查：attack 标签 0 / 17164 不一致** ⇒ 回放与 PC 的数据流完全对齐，下面的差异才可以归因到模型。

| PC 方案 | 全量 F1 | 全量 FPR | 部署区 F1 | 部署区 FPR | **与 MCU 的决策翻转** | 翻转窗号 |
|---|---|---|---|---|---|---|
| Route E FP32 | 0.8813 | 0.0227 | 0.8671 | 0.0591 | **1 / 17164** | 17163 |
| **Hybrid-INT16** | 0.8812 | 0.0227 | 0.8671 | 0.0591 | 2 / 17164 | 11340, 17163 |
| Hybrid-INT8 | 0.8826 | 0.0261 | 0.8663 | 0.0689 | 108 / 17164 | 310, 799… |
| 全量 INT8 | 0.8823 | 0.0294 | 0.8670 | 0.0698 | 184 / 17164 | 246, 310… |

⇒ **TFLM 固件 == Route E 固件**，逐窗一致，只差回放尾部截断的末窗。

#### 7.3.3 那两窗到底怎么回事

| 窗 | 标签 | MCU TFLM | Route E PC | Hybrid-INT16 PC | 判定 |
|---|---|---|---|---|---|
| **W=11340** | atk=True | **报警**，日志 `prob_x1000=500` | prob **0.500101** → 报警 | prob **0.499977** → 不报警 | **int16 量化翻转**：全数据集 int16 只翻这一窗 |
| **W=17163**（末窗） | atk=True | **不报警**（无 ALARM 行） | prob 1.000000 → 报警 | prob 1.000000 → 报警 | **回放尾部截断**（已知正常现象） |

* **W=11340**：MCU 跟 Route E 一致（都报警），是 **PC 侧 Hybrid-INT16** 跟它们不一致。
  该窗 logit 距阈值只有 `1.2e-4` —— 属于浮点舍入敏感区，跟 INT8 那 58 个边界带翻转是同一类，
  只是 int16 把偏移压到了 `1e-4` 量级。**不是 bug，是量化噪声恰好落在阈值上。**
* **W=17163**：最后一个窗口，回放到数据集末尾时输入被截断，MCU 侧拿到的特征不完整，
  所以没报警。**已知正常现象，按项目惯例不作 bug 处理。**

> 注意 W=17163 同时落在部署区里（frame 274,623 ≥ 200,001），
> 所以 MCU 部署区 TP=1949 比 PC 的 1950 少 1 —— 这**不是** TFLM 的问题，
> 10-27 的 Route E 固件上机也是同样的 1949。

#### 7.3.4 体积对比（TFLM 不是生产路径）

| 方案 | ROM (Flash) | RAM (RW+ZI) | 相对 Route E |
|---|---|---|---|
| Route E FP32 | 149,684 B | 241,888 B | — |
| **Hybrid-INT16** | **112,656 B** | **159,912 B** | **−24.7% / −33.9%** |
| 全量 INT8 | 97,292 B | 157,408 B | −35.0% / −34.9% |
| **TFLM FP32（对照）** | **281,288 B** | **254,096 B** | **+87.9% / +5.0%** |

TFLM 的大头是 **reference ops 运行时 + 96 KB arena**；`MicroInterpreter*` 符号已在
`H743test_TFLM.map` 中确认，模型本体 93,596 B。
⇒ **TFLM 作为"论文对照"成立，但作为生产路径在体积上不划算** —— 生产仍是 Route E 或 Hybrid-INT16。

#### 7.3.5 还没做的

- `ch32_benchmark` 推理延迟实测（TFLM ref impl 通常比手写 kernel 慢 3~10 倍，**必须实测**）。
- `'T'` controlled-input 自检命令未单独跑（本次全量回放已隐含证明运行时正常：
  17164 窗里只有 1 窗与 Route E 不同，且差异原因已定位到回放截断而非算子）。

---

## 8. 可复现清单

```bash
# G0  导出 + 静态兼容性检查
python experiments/tflm_fp32/tools/onnx_to_tflite.py
python experiments/tflm_fp32/tools/tflm_compat_check.py

# G0b PC 参考 (TFLite vs ORT vs 手写 kernel + 5 controlled input)
python experiments/tflm_fp32/tools/tflite_pc_reference.py

# G1  生成 MCU 工程 (不改既有工程) + 构建
python experiments/tflm_fp32/tools/gen_keil_project.py
cd "C:/Keil_v5/UV4" && ./UV4.exe -b \
   "C:/work/Claude/Issue/MCU/MDK-ARM/H743test_TFLM.uvprojx" \
   -t H743test_TFLM -o "C:/work/Claude/Issue/MCU/MDK-ARM/build_tflm.log"

# 回归: Route E 固件仍可构建 (0 error)
cd "C:/Keil_v5/UV4" && ./UV4.exe -b \
   "C:/work/Claude/Issue/MCU/MDK-ARM/H743test.uvprojx" \
   -t H743test -o "C:/work/Claude/Issue/MCU/MDK-ARM/build_routee.log"

# G2  上机自检判定 (需要先烧录 + 串口发 'T' 抓日志)
python experiments/tflm_fp32/tools/tflm_selftest_check.py capture.txt

# G3  部署区对拍 (需要先跑一次 PC 回放)
python experiments/tflm_fp32/tools/tflm_deploy_parity.py --log logs/test_<ts>.csv --label TFLM

# G3b 全量逐窗 vs PC 四方案 (2026-10-28 上机结果, §7.3)
python tflm_vs_pc_window_diff.py logs/test_20261027-232828_auto-40ms.csv
```

**验收断言（脚本内已硬编码）**

| # | 断言 | 位置 |
|---|---|---|
| 1 | op 集合 ⊆ 白名单，且不含 PACK/SHAPE/STRIDED_SLICE/SPACE_TO_BATCH_ND | `tflm_compat_check.py` |
| 2 | 三个空洞卷积 `dilation_w_factor` == 1/2/4（读 flatbuffer） | `tflm_compat_check.py` |
| 3 | 输入 `[1,23,16]` float32 / 输出 `[1]` float32 | `tflm_compat_check.py` |
| 4 | MEAN axes numel==1 + keep_dims / EXPAND_DIMS rank≤1 / TRANSPOSE perm rank==1 / RESHAPE int32 | `tflm_compat_check.py` |
| 5 | `.tflite` → C 数组后**回读逐字节比对** | `tflite_to_c_array.py` |
| 6 | 新工程里**不含** `ch32_inference.c` / `ch32_inference_int8.c` | `gen_keil_project.py` |
| 7 | 每个 `.cc` 是 FileType 1 且 per-file 带 `-x c++` / `-std=c++17` | `gen_keil_project.py` |
| 8 | 新工程引用的每个文件**真的存在** | `gen_keil_project.py` |
| 9 | 实机 5 个 controlled input 与 PC 参考一致（退出码 0/1/2/3 分别对应 4 种结论） | `tflm_selftest_check.py` |
| 10 | MCU↔PC 逐窗对拍：决策翻转 == 0 | `tflm_deploy_parity.py` |

---

## 9. 已知偏差与风险

| 项 | 说明 | 影响 |
|---|---|---|
| libcxx `__config` 就地补丁 | 2026-10-15 旧路线改过 `C:\Keil_v5\ARM\ARMCLANG\include\libcxx\__config`（pristine 备份在 `__config.bak.2026-10-15`） | **换机器复现前必须检查**；本报告的所有构建都在打过补丁的环境里 |
| PC 参考用 BUILTIN_REF | 与 TFLM 的 `reference_ops` 同源，但不是逐位相同实现 | 观测到 1.1e-05 logit 级差异（0 翻转），属正常 |
| 上机数字已采集（2026-10-28） | 全量回放 17164 窗，部署区 F1 0.8668 / FPR 5.91%，与 Route E 上机逐窗一致 | 无风险项（见 §7.3） |
| arena 大小 96 KB 是上限不是实测 | TFLM 的 greedy planner 实际占用要等 `AllocateTensors` 成功后才知道（`[TFLM] AllocateTensors OK, arena used N / 98304 B`） | 96 KB 偏保守，D2 有 256 KB 余量 |
| TFLM 推理延迟未实测 | Route E 是 ~0.13 ms @ 480 MHz；TFLM 参考实现通常慢 3~10 倍 | 需上机跑 `ch32_benchmark` 补数；对 40 ms 帧率的 Modbus 流大概率无影响，但**必须实测确认** |

---

## 10. 相关记忆

[[project-route-e-migration]] · [[project-2026-10-21-h743-final-baseline]] ·
[[project-h743-int8-tflm-line]] · [[project-2026-10-21-route-e-diag-tooling]] ·
[[feedback-parity-test-must-bind-to-real-source]] · [[feedback-eval-metrics-must-be-replayable]] ·
[[feedback-h743-m7-dcache-output-stale]] · [[feedback-ac6-libcxx-armcompiler-libcxx-macro]]
