# 参考文献管理指南

> Word 论文的参考文献管理有 3 种方案，本指南推荐 **Zotero（免费）**。

---

## 方案 A：Zotero（强烈推荐）

### 安装

1. 下载：https://www.zotero.org/
2. 安装 Zotero 主程序
3. 安装 Word 插件：https://www.zotero.org/support/plugins/word_processor_integration
4. 安装 GB/T 7714 样式：https://github.com/redleafnew/Chinese-STD-GB-T-7714-2015

### 配置 GB/T 7714 样式

1. 打开 Zotero → `编辑` → `设置` → `引用` → `样式` 旁的"管理样式"
2. 搜索 "GB/T 7714" 或 "Chinese Standard"
3. 双击安装

### 插入文献

1. Word 中把光标放在引用位置
2. 点击 Word 工具栏的"Zotero" → `Add/Edit Citation`
3. 在弹出框搜索文献标题
4. 选择并按 Enter 插入
5. 在参考文献章节按"Zotero" → `Insert Bibliography` 自动生成

### 修改样式

1. Word 中点击 Zotero → `Document Preferences`
2. 文献样式选 "Chinese Standard GB/T 7714-2015"
3. 点击"刷新"应用

---

## 方案 B：EndNote（学校常购）

### 安装

1. 学校一般提供 EndNote 授权
2. 安装 EndNote X9 或更新版本
3. 打开 Word 时会自动加载 EndNote 插件

### 插入 GB/T 7714 样式

1. EndNote 官网下载 Chinese Standard GB/T 7714 样式
2. 放入 EndNote 安装目录的 Styles 文件夹
3. 重启 Word
4. 工具栏选择 GB/T 7714 样式

### 插入文献

1. 光标在引用位置
2. EndNote 工具栏 → `Insert Citation`
3. 搜索文献
4. 工具栏 → `Insert Bibliography` 生成参考文献

---

## 方案 C：Mendeley（免费备选）

类似 Zotero，免费，支持 GB/T 7714 样式。功能略弱于 Zotero。

---

## 方案 D：手工插入

如不使用文献管理软件，按以下格式手工输入：

### 期刊 [J]

```
[1] 作者1, 作者2, 作者3. 文章题名[J]. 期刊名, 年, 卷(期): 起止页码.
```

示例：
```
[1] 周天逸, 李源, 吴磊. SCADA 系统综述: 架构、漏洞与防御[J]. 计算机安全, 2022, 118: 102734.
```

### 会议 [C]

```
[2] 作者1, 作者2. 论文题名[C]//会议名. 出版地: 出版者, 年: 起止页码.
```

示例：
```
[2] WANG W, ZHU M, ZENG X, et al. Malware traffic classification using CNN[C]//ICOIN. IEEE, 2017: 712-717.
```

### 学位论文 [D]

```
[3] 作者. 论文题名[D]. 保存地: 保存单位, 年.
```

### 专著 [M]

```
[4] 作者. 书名[M]. 版次. 出版地: 出版社, 年.
```

### 专利 [P]

```
[5] 申请人. 专利名称: 专利号[P]. 公开日期.
```

### 电子文献 [EB/OL]

```
[6] 作者. 题名[EB/OL]. (发布日期)[引用日期]. URL.
```

### arXiv 预印本 [Preprint]

```
[7] 作者. arXiv:编号[Preprint]. arXiv, 年. https://arxiv.org/abs/编号
```

---

## GB/T 7714-2015 关键规则

### 1. 作者署名

- 3 人以内：全部列出
- 超过 3 人：列前 3 人 + ", 等" (中文) / ", et al" (英文)
- 用逗号分隔

### 2. 引用标注

文内引用采用"顺序编码制"：
- 单篇：`[1]`
- 多篇：`[1, 3]` 或 `[1-3]`
- 多处引用：`[1, 3-5]`

### 3. 顺序

- 参考文献按正文中首次出现的顺序编号
- 多次引用同一文献：仍用首次编号

### 4. 大小写

- 英文期刊名：用缩写（Title Case 或全大写）
- 英文会议名：用全称（Title Case）
- 中文：用全称

### 5. 标点

- 中文用全角："，""。""："
- 英文用半角：",""."""
- 中英文混排时，标点用中文（与所在语言一致）

---

## 常见错误

| 错误 | 正确 | 错误类型 |
|---|---|---|
| `[1] Smith J. Title. 期刊, 2020, 1: 1-10.` | `[1] SMITH J. Title[J]. Journal, 2020, 1: 1-10.` | 缺 [J]、缺缩写 |
| `[1] Wang, Li, Zhang. ...` | `[1] WANG W, LI Y, ZHANG Z. ...` | 缺 et al 规则 |
| `[1] Wang Wei, et al. ...` | `[1] WANG W, et al. ...` | 名字格式不对 |
| `DOI: 10.xxxx/xxx [EB/OL]` | `DOI: 10.xxxx/xxx.` | 标点位置错 |

---

## 本模板推荐引用

以下文献已在 `references/refs.bib` 中维护，可直接复制：

| BibTeX key | 文献 |
|---|---|
| `ref:scada_survey_2022` | Zhou et al. SCADA 综述, 2022 |
| `ref:iiot_security_2023` | Al-Garadi et al. IoT 安全综述, 2023 |
| `ref:cnn_ids_2017` | Wang et al. CNN 入侵检测, 2017 |
| `ref:lstm_ids_2018` | Kim et al. LSTM-IDS, 2018 |
| `ref:transformer_ids_2021` | Wu et al. Transformer IDS, 2021 |
| `ref:bai2018tcn` | Bai et al. TCN 原始论文, 2018 |
| `ref:hu2018senet` | Hu et al. SE-Net, 2018 |
| `ref:jacob2018quant` | Jacob et al. 整数量化, 2018 |
| `ref:han2015prune` | Han et al. Deep Compression, 2015 |
| `ref:ian_arff_dataset` | Morris et al. IanArffDataset, 2011 |
| `ref:cmsis_nn` | Lai et al. CMSIS-NN, 2018 |
| `ref:tflite_micro` | David et al. TFLite Micro, 2020 |

---

## 引用次数建议

硕士论文参考文献数量：

| 类型 | 建议数量 |
|---|---|
| 期刊论文 [J] | 20-30 篇 |
| 会议论文 [C] | 5-10 篇 |
| 学位论文 [D] | 1-3 篇 |
| 专著 [M] | 2-3 本 |
| 其他 [R/P/EB/OL] | 1-3 篇 |
| **总计** | **30-50 篇** |

近 5 年文献应占 ≥ 60%。
