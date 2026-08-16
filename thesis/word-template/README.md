# Word 版硕士论文模板

> **配套 LaTeX 模板**：`../main.tex`
> **核心差异**：Word 没有编译开关，所有 `\pitfall{}` 警示默认显示，老师评阅后手动删除。
> **诊断机制**：通过 VBA 宏（`vba/ThesisTemplate.bas`）一键插入红/黄/绿框。

---

## 快速开始

### 方法 A：使用本目录的 `.docx` 模板

1. 打开 `thesis-template.docx`（已生成，45.3 KB，24 个诊断框示例）
2. 修改封面信息即可使用
3. 可选：按 `Word_Setup_Guide.md` 重新生成：`python generate_template.py`

### 方法 B：自己按指南配置

按 `Word_Setup_Guide.md` 一步步操作，5-10 分钟配置完毕。

### 方法 C：使用 VBA 宏辅助

1. 打开 Word → Alt+F11 打开 VBA 编辑器
2. 文件 → 导入文件 → 选择 `vba/ThesisTemplate.bas`
3. 在 Word 中按 Alt+F8 运行宏 `InsertPitfall` / `InsertTodo` / `InsertGrade` / `InsertGood`

---

## 文件清单

```
word-template/
├── README.md                       # 本文件
├── thesis-template.docx            # ★ Word 二进制模板（45.3 KB，2026-06-16 新增）
├── generate_template.py            # ★ 模板生成脚本（可重新生成 .docx）
├── Word_Setup_Guide.md             # ★ 详细 Word 配置指南（5-10 分钟）
├── Word_Style_Spec.md              # ★ 所有样式参数（页边距、字体、行距）
├── Citation_Guide.md               # 参考文献插入与格式化指南
│
├── vba/
│   └── ThesisTemplate.bas          # ★ VBA 宏（实现 pitfall/todo/grade 框）
│
├── sample/
│   ├── sample-content.md           # 范文示例（带诊断标注）
│   └── figures/                    # 示例图片占位
│
└── styles/
    └── (待生成) 样式导入说明
```

---

## 与 LaTeX 版本的核心差异

| 维度 | LaTeX | Word |
|---|---|---|
| 编译模式 | `\finaltrue` 切换 | 无开关，手动隐藏 |
| 诊断框 | `\pitfall{}` 宏 | VBA 宏一键插入红框 |
| 公式 | `equation` 环境 | MathType 或 Word 公式 |
| 文献 | `biblatex` 自动 | EndNote / Zotero / 手工 |
| 图表编号 | `\figref{}` 自动 | 交叉引用工具 |
| 编译 | 4 次 xelatex | 1 次 F9 刷新 |
| 协作 | 不友好 | 友好（可批注） |

**核心问题**：Word 没法像 LaTeX 那样用 `\finalfalse` 隐藏警示框。
**解决方案**：用 VBA 宏生成的"文本框"自带"显示/隐藏"快捷键 Ctrl+Shift+H。

---

## 4 种插入诊断框的方法

### 方法 1：VBA 宏（推荐）

按 Alt+F8 → 运行 `InsertPitfall` → 输入编号和描述 → 自动生成红框。

### 方法 2：手动插入

插入 → 文本框 → 设置红色边框 → 输入 `\pitfall{...}` 内容。

### 方法 3：复制粘贴

从本目录 `sample/sample-content.md` 复制现成的红/黄/绿框。

### 方法 4：自定义样式

参考 `Word_Setup_Guide.md` 第四节"自定义样式"，把"诊断框"定义为
Word 内置样式，未来一键套用。

---

## Word 模板要求清单

### 页面设置
- [ ] A4 纸
- [ ] 上下边距 2.5cm，左右边距 2.5cm
- [ ] 页眉 1.5cm，页脚 1.75cm
- [ ] 装订线 0.5cm（左侧）

### 字体设置
- [ ] 中文：宋体（正文）、黑体（一级标题）、楷体（强调）
- [ ] 英文：Times New Roman
- [ ] 数字：Times New Roman
- [ ] 代码：Consolas / Courier New

### 字号
- [ ] 一级标题（章）：二号黑体居中
- [ ] 二级标题（节）：三号黑体
- [ ] 三级标题（小节）：小三号黑体
- [ ] 正文：小四号宋体
- [ ] 图表标题：五号黑体
- [ 页码：五号 Times New Roman

### 行距与段落
- [ ] 正文 1.5 倍行距
- [ ] 段前 0 行，段后 0 行
- [ ] 首行缩进 2 字符
- [ ] 章标题段前 24pt，段后 18pt

### 页码
- [ ] 摘要、目录：罗马数字 I, II, III（居中）
- [ ] 正文：阿拉伯数字 1, 2, 3（右下角）
- [ ] 参考文献：续正文页码

### 页眉
- [ ] 奇数页：章标题
- [ ] 偶数页：论文标题
- [ ] 五号宋体

---

## 详细使用步骤

参见：
- 📐 [Word_Setup_Guide.md](./Word_Setup_Guide.md) — 详细配置步骤
- 🎨 [Word_Style_Spec.md](./Word_Style_Spec.md) — 所有样式参数
- 📚 [Citation_Guide.md](./Citation_Guide.md) — 参考文献管理
- 🔧 [vba/ThesisTemplate.bas](./vba/ThesisTemplate.bas) — VBA 宏源码
- 📄 [sample/sample-content.md](./sample/sample-content.md) — 范文示例
