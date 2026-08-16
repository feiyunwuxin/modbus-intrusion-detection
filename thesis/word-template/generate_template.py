"""
生成 thesis-template.docx — 老师视角的 SCADA 入侵检测硕士论文模板
================================================================
- 4 色诊断框样式（pitfall / todo / good / grade）
- 6 章结构 + 摘要/目录/致谢/参考文献
- 范文示例（带诊断标注）
- 4 套快捷键说明页

用法：python generate_template.py
输出：thesis-template.docx （与本脚本同目录）
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsmap
from docx.oxml import OxmlElement
import os

# ---------- 工具函数 ----------
def set_cell_border(cell, **kwargs):
    """设置单元格边框（颜色/宽度/样式）"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in kwargs:
            tag = qn(f'w:{edge}')
            elem = tcBorders.find(tag)
            if elem is None:
                elem = OxmlElement(f'w:{edge}')
                tcBorders.append(elem)
            for k, v in kwargs[edge].items():
                elem.set(qn(f'w:{k}'), v)

def set_cell_shading(cell, fill):
    """设置单元格底纹"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)

def add_diagnostic_box(doc, kind, code, text):
    """
    添加 4 色诊断框（1x1 表格实现）
    kind: 'pitfall' / 'todo' / 'good' / 'grade'
    """
    config = {
        'pitfall': {
            'border': 'C00000', 'shade': 'FFEBEB',
            'icon': '⚠️', 'tag': f'【学生常见错误 {code}】',
            'font': '楷体'
        },
        'todo': {
            'border': 'BF8F00', 'shade': 'FFFACD',
            'icon': '📝', 'tag': f'【待补充（学生） {code}】',
            'font': '楷体'
        },
        'good': {
            'border': '00B050', 'shade': 'E2EFDA',
            'icon': '✓', 'tag': f'【优秀示例 {code}】',
            'font': '楷体'
        },
        'grade': {
            'border': '2E75B6', 'shade': 'EBF5FF',
            'icon': '✏️', 'tag': f'【老师评语 {code}】',
            'font': '楷体'
        },
    }[kind]

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    set_cell_border(cell,
                    top={'val': 'single', 'sz': '12', 'color': config['border']},
                    left={'val': 'single', 'sz': '12', 'color': config['border']},
                    bottom={'val': 'single', 'sz': '12', 'color': config['border']},
                    right={'val': 'single', 'sz': '12', 'color': config['border']})
    set_cell_shading(cell, config['shade'])

    p = cell.paragraphs[0]
    p.paragraph_format.left_indent = Cm(0.2)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.3

    run = p.add_run(f"{config['icon']} {config['tag']}  {text}")
    run.font.size = Pt(10.5)
    run.font.name = config['font']
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), config['font'])
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')

    # 框后空行
    doc.add_paragraph()


def add_styled_paragraph(doc, text, style='Normal', align=None, indent=True):
    """添加样式化段落（宋体小四 1.5 倍行距 中文首行缩进 2 字符）"""
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if indent and style == 'Normal':
        p.paragraph_format.first_line_indent = Cm(0.74)  # 2 字符
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    if style == 'Normal':
        run.font.size = Pt(12)
        run.font.name = '宋体'
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.find(qn('w:rFonts'))
        if rFonts is None:
            rFonts = OxmlElement('w:rFonts')
            rPr.append(rFonts)
        rFonts.set(qn('w:eastAsia'), '宋体')
        rFonts.set(qn('w:ascii'), 'Times New Roman')
        rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    return p


def add_heading_1(doc, text):
    """一级标题（章）：黑体二号居中"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(18)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    run = p.add_run(text)
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.name = '黑体'
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), '黑体')
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    return p


def add_heading_2(doc, text):
    """二级标题（节）：黑体三号"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    run = p.add_run(text)
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.name = '黑体'
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), '黑体')
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    return p


def add_heading_3(doc, text):
    """三级标题（小节）：黑体小三"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    run = p.add_run(text)
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.name = '黑体'
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), '黑体')
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    return p


def setup_page(doc):
    """页面设置：A4 上下左右 2.5cm"""
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.header_distance = Cm(1.5)
    section.footer_distance = Cm(1.75)
    section.gutter = Cm(0.5)


# ---------- 主生成逻辑 ----------
doc = Document()
setup_page(doc)

# ========== 封面 ==========
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(120)
run = p.add_run('（学校名）')
run.font.size = Pt(16)
run.font.name = '黑体'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(12)
run = p.add_run('硕士专业学位论文')
run.font.size = Pt(36)
run.font.bold = True
run.font.name = '黑体'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(48)
run = p.add_run('题目：基于 TCN-SE 与混合精度量化的\nSCADA 入侵检测研究')
run.font.size = Pt(22)
run.font.name = '黑体'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(120)
run = p.add_run('作者：（姓名）\n指导教师：（教授）\n专业：（控制工程）\n学号：（学号）\n日期：2026 年 6 月')
run.font.size = Pt(14)
run.font.name = '宋体'

doc.add_page_break()

# ========== 模板使用说明 ==========
add_heading_1(doc, 'Word 模板使用说明')
add_styled_paragraph(doc, '本模板使用 Word VBA 宏实现诊断框一键插入，配合自定义样式（4 色）实现与 LaTeX 版本等价的教学诊断功能。')

add_heading_2(doc, '1. 4 色诊断框说明')
add_styled_paragraph(doc, '本模板定义了 4 种诊断样式，每种对应一个 VBA 宏和一组快捷键：')

table = doc.add_table(rows=5, cols=4)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = '样式名'
hdr[1].text = '颜色'
hdr[2].text = '快捷键'
hdr[3].text = '用途'
for cell in hdr:
    set_cell_shading(cell, 'D9E1F2')

rows_data = [
    ('学生常见错误', '红色 #C00000', 'Alt+Shift+P', '警示学生可能犯的错误'),
    ('待补充内容', '黄色 #BF8F00', 'Alt+Shift+T', '标记需要补充的地方'),
    ('优秀示例', '绿色 #00B050', 'Alt+Shift+G', '高亮写得好的地方'),
    ('老师评语', '蓝色 #2E75B6', 'Alt+Shift+C', '老师打分/总评'),
]
for i, (name, color, key, use) in enumerate(rows_data, 1):
    cells = table.rows[i].cells
    cells[0].text = name
    cells[1].text = color
    cells[2].text = key
    cells[3].text = use

add_heading_2(doc, '2. VBA 宏导入步骤')
add_styled_paragraph(doc, '（1）打开 Word → 按 Alt+F11 打开 VBA 编辑器；')
add_styled_paragraph(doc, '（2）菜单 文件 → 导入文件 → 选择 vba/ThesisTemplate.bas；')
add_styled_paragraph(doc, '（3）回到 Word 按 Alt+F8 运行 SetShortcuts 宏，自动绑定快捷键；')
add_styled_paragraph(doc, '（4）选中文字 → 按 Alt+Shift+P 插入红色 pitfall 框；')
add_styled_paragraph(doc, '（5）按 Ctrl+Shift+H 可一键隐藏/显示所有诊断框。')

add_heading_2(doc, '3. 自定义样式（可选）')
add_styled_paragraph(doc, '如需将诊断框定义为 Word 内置样式：开始 → 样式窗格 → 新建样式 → 命名（如"学生常见错误"）→ 设置楷体小四 + 红色 1.5pt 边框 + 浅红 #FFEBEB 底纹。详见 Word_Style_Spec.md。')

doc.add_page_break()

# ========== 摘要 ==========
add_heading_1(doc, '摘 要')
add_styled_paragraph(doc, '随着工业互联网与人工智能技术的快速发展，SCADA 系统作为国家关键基础设施的神经中枢，正面临日益严峻的网络安全挑战。现有的入侵检测研究多沿用通用网络入侵特征，缺乏对 Modbus 等 SCADA 协议语义的专门建模；同时，深度学习模型在该任务上尚未稳定超越 LightGBM 等树模型；面向 MCU 等受限设备的部署研究也相对不足。')
add_styled_paragraph(doc, '针对上述问题，本文以 IanArffDataset 数据集为研究对象，开展从数据特征、模型架构到边缘部署的全栈研究，主要工作与创新点如下：')
add_styled_paragraph(doc, '（1）基于 SCADA 协议语义的领域特征工程。分析 Modbus 4 行主-从交互结构，提出 27 维协议语义特征（17 维原始 + 3 维行级 + 8 维窗口级），使所有模型 Macro-F1 一致提升 0.2—0.7 个百分点。')
add_styled_paragraph(doc, '（2）TCN-SE 轻量入侵检测模型。在 27 种时序模型的横向对比基础上，融合 SE 通道注意力，TCN-SE 在测试集取得 Macro-F1 = 0.8340，首次在该任务上超越 LightGBM。')
add_styled_paragraph(doc, '（3）面向 MCU 的极限压缩与混合精度量化。提出 4,237 参数的 TCN V19 变体与权重 INT8 / 激活 FP32 混合方案，在 Cortex-M4 平台上实现 5.5 KB 模型、0.47 ms 推理。')
add_styled_paragraph(doc, '关键词：SCADA 入侵检测；时序卷积网络；通道注意力；模型压缩；混合精度量化；边缘部署')

doc.add_page_break()

# ========== 目录占位 ==========
add_heading_1(doc, '目 录')
add_styled_paragraph(doc, '（使用 Word 引用 → 目录 → 自动目录功能生成；显示级别 3）', indent=False)
doc.add_page_break()

# ========== 第 1 章 绪论（带诊断标注范文）==========
add_heading_1(doc, '第 1 章 绪 论')

add_heading_2(doc, '1.1 研究背景与意义')

add_diagnostic_box(doc, 'pitfall', 'P-1.1',
    '数据无出处。学生常引"据统计"、"研究表明"等模糊表述，无可追溯的数据来源。')
add_styled_paragraph(doc, '随着工业互联网（Industrial Internet of Things, IIoT）与人工智能技术的快速发展，SCADA（Supervisory Control and Data Acquisition，监控与数据采集）系统作为国家关键基础设施的神经中枢，正面临日益严峻的网络安全挑战 [1]。根据国家工业信息安全发展研究中心发布的报告，2024 年全球工业领域网络攻击事件同比增长 27%，其中针对能源、水务等 SCADA 系统的定向攻击占比超过 35%。')
add_styled_paragraph(doc, 'SCADA 系统广泛应用于燃气管道、电力调度、轨道交通、自来水处理等关键领域，其安全性直接关系到国计民生与公共安全。2010 年伊朗"震网"病毒、2015 年乌克兰电网攻击、2021 年美国 Colonial Pipeline 输油管道勒索事件，无一不揭示了 SCADA 系统一旦失守可能造成的灾难性后果。')

add_diagnostic_box(doc, 'good', 'G-1.1',
    '本段使用"震网"、"乌克兰电网"、"Colonial Pipeline"三个递进案例，从国家→地区→企业层面论证，逻辑递进清晰。')

add_diagnostic_box(doc, 'todo', 'T-1.1',
    '增加 1-2 个中文案例，如"2020 年某省电网工控系统入侵事件"。')

add_styled_paragraph(doc, '在此背景下，入侵检测（Intrusion Detection System, IDS）作为 SCADA 安全的最后一道防线，已成为学术界与工业界共同关注的研究热点。')

add_diagnostic_box(doc, 'pitfall', 'P-0.5',
    '摘要写成"引言缩写"。本章第一节是背景，不是摘要副本；摘要应单独有四段式结构。')

add_heading_2(doc, '1.2 国内外研究现状')

add_diagnostic_box(doc, 'pitfall', 'P-1.2',
    '文献堆砌。学生常罗列 20+ 篇文献却不评价。建议：每段文献综述末尾必须有 1 句评价。')

add_styled_paragraph(doc, '现有 SCADA 入侵检测研究大致可分为三类：基于规则与签名的方法、基于传统机器学习的方法、基于深度学习的方法。')

add_heading_3(doc, '1.2.1 基于规则与签名的方法')
add_styled_paragraph(doc, '早期研究多采用专家系统与特征库匹配 [4]。该方法对已知攻击检测准确率高、可解释性强，但完全依赖人工维护规则库，对未知攻击（zero-day）几乎无能为力。')

add_diagnostic_box(doc, 'good', 'G-1.2',
    '末句评价指出局限性，自然引出下一小节。')

add_heading_3(doc, '1.2.2 基于传统机器学习的方法')
add_styled_paragraph(doc, '随着 KDD Cup 99、NSL-KDD、UNSW-NB15 等入侵检测基准数据集的发布，基于传统机器学习的方法成为研究主流。支持向量机 [5]、随机森林 [6]、XGBoost [7]、LightGBM [11] 等模型在多个数据集上取得了 F1 ≥ 0.85 的检测性能。其中 LightGBM 凭借直方图加速与 leaf-wise 生长策略，在工业级数据上展现出显著优势。')

add_heading_3(doc, '1.2.3 基于深度学习的方法')

add_diagnostic_box(doc, 'pitfall', 'P-1.3',
    '缺乏协议层切入。这是本领域最大空白。SCADA 与传统 IT 网络的根本区别在协议层；学生若不强调"协议语义"，会和通用 IDS 论文混为一谈。')

add_styled_paragraph(doc, '近年来，深度学习在时序建模领域的突破为入侵检测带来了新思路。文献 [8] 首次将 1D-CNN 应用于网络流量分类，取得了比 SVM 高 3% 的准确率；文献 [9] 提出 LSTM-IDS，利用长短期记忆网络捕捉时序依赖；文献 [10] 进一步引入注意力机制，在多个公开数据集上达到 SOTA。然而，上述工作大多基于通用网络入侵数据集（KDD/NSL-KDD/UNSW-NB15），鲜有针对 SCADA 协议语义（如 Modbus、DNP3）的专门优化。')

add_heading_2(doc, '1.3 现有研究不足')
add_styled_paragraph(doc, '综合分析国内外研究现状，当前 SCADA 入侵检测领域存在以下不足：')

add_diagnostic_box(doc, 'good', 'G-1.3',
    '三个"不足"恰好对应后文三个"创新点"——这是 1:1 映射的最佳实践。')

add_styled_paragraph(doc, '1. 特征体系缺乏协议语义：现有研究多沿用通用网络入侵特征，未充分利用 Modbus 主-从交互结构等协议层先验；')
add_styled_paragraph(doc, '2. 深度学习模型未稳定超越树模型：在多个公开数据集上，LightGBM / XGBoost 仍占据 SOTA，DL 的优势尚未充分释放；')
add_styled_paragraph(doc, '3. 模型部署可行性研究不足：现有工作多报告 GPU 推理延迟，对 MCU/PLC 等受限设备的部署研究较少。')

add_heading_2(doc, '1.4 研究内容与创新点')

add_diagnostic_box(doc, 'pitfall', 'P-1.4',
    '创新点无量化数据。"创新点"是论文最核心的部分，但学生常写成"提出了一种新方法"——必须每个创新点都有可量化的指标提升。')

add_diagnostic_box(doc, 'pitfall', 'P-1.5',
    '创新点对应关系模糊。"对应第 X 章"必须明确写出，且在第 X 章开头必须有"本章对应创新点 N"的回指。')

add_styled_paragraph(doc, '针对上述不足，本文以 IanArffDataset 数据集为研究对象，开展从数据特征、模型架构到边缘部署的全栈研究，主要创新点如下：')
add_styled_paragraph(doc, '1. 基于 SCADA 协议语义的领域特征工程（对应第 3 章）。分析 Modbus 4 行主-从交互结构，提出 27 维协议语义特征，使所有模型 Macro-F1 一致提升 0.2—0.7 个百分点；')
add_styled_paragraph(doc, '2. TCN-SE 轻量入侵检测模型（对应第 4 章）。在 27 种时序模型的横向对比基础上，融合 SE 通道注意力，TCN-SE 在测试集取得 Macro-F1 = 0.8340，首次在该任务上超越 LightGBM；')
add_styled_paragraph(doc, '3. 面向 MCU 的极限压缩与混合精度量化（对应第 5 章）。提出 4,237 参数的 TCN V19 变体与权重 INT8 / 激活 FP32 混合方案，在 Cortex-M4 平台上实现 5.5 KB 模型、0.47 ms 推理。')

add_heading_2(doc, '1.5 论文组织结构')

add_diagnostic_box(doc, 'good', 'G-1.5',
    '每章一句话点明"做什么"，避免冗长。')

add_styled_paragraph(doc, '本文章节安排如下：')
add_styled_paragraph(doc, '第 1 章 绪论：介绍研究背景、国内外现状与本文创新点；')
add_styled_paragraph(doc, '第 2 章 相关理论与技术基础：SCADA 入侵检测任务定义、TCN 与注意力机制原理、量化基础；')
add_styled_paragraph(doc, '第 3 章 SCADA 协议特征工程：详细阐述 27 维特征的设计动机与对比实验；')
add_styled_paragraph(doc, '第 4 章 TCN-SE 入侵检测模型：模型架构、27 模型横评、消融实验；')
add_styled_paragraph(doc, '第 5 章 极限压缩与混合精度量化：V19 变体设计、INT8 量化方案、MCU 部署实测；')
add_styled_paragraph(doc, '第 6 章 总结与展望。')

add_diagnostic_box(doc, 'grade', 'A',
    '第 1 章基本结构完整，创新点量化指标清晰。但 1.1 节案例可增加 1 个中文事件；1.2 节末缺少对各方法的统一对比表（建议增加表 1-1）。')

doc.add_page_break()

# ========== 第 2 章 相关理论与技术基础 ==========
add_heading_1(doc, '第 2 章 相关理论与技术基础')

add_heading_2(doc, '2.1 SCADA 系统与入侵检测任务定义')
add_styled_paragraph(doc, 'SCADA 系统由主站（Master）、从站（Slave）、通信链路与人机界面（HMI）组成。本文中入侵检测任务被形式化为：给定连续 8 个时间步的 27 维特征序列，预测下一时刻是否为攻击行为。')

add_heading_2(doc, '2.2 时序卷积网络（TCN）')
add_heading_3(doc, '2.2.1 因果卷积')
add_styled_paragraph(doc, '因果卷积保证时刻 t 的输出仅依赖 t 及之前的输入，避免信息泄露。')
add_heading_3(doc, '2.2.2 膨胀卷积')
add_styled_paragraph(doc, '膨胀卷积以指数扩张的膨胀率扩大感受野，可在不增加参数的情况下捕获长程依赖 [12]。')

add_heading_2(doc, '2.3 通道注意力机制（SE-Net）')
add_styled_paragraph(doc, 'SE-Net 通过 Squeeze（全局平均池化）与 Excitation（两层全连接 + Sigmoid）两步，自适应地学习通道权重 [13]。本文将其嵌入 TCN 残差块，称为 TCN-SE。')

add_heading_2(doc, '2.4 模型量化基础')
add_styled_paragraph(doc, '模型量化将 32-bit 浮点参数映射为低 bit 整数。INT8 量化可将模型大小压缩 4 倍，但全 INT8 在 ARM Cortex-M 上反而因缺少专用指令导致推理变慢 [14]。本文采用权重 INT8 + 激活 FP32 的混合方案，兼顾压缩率与速度。')

doc.add_page_break()

# ========== 第 3 章 SCADA 协议特征工程 ==========
add_heading_1(doc, '第 3 章 SCADA 协议特征工程')

add_heading_2(doc, '3.1 Modbus 协议主-从交互分析')

add_diagnostic_box(doc, 'pitfall', 'P-3.1',
    '无基线对比。本章每张表必须同时报告"原始特征"、"新特征"、"新特征 - 原始"三列。')

add_styled_paragraph(doc, 'IanArffDataset 包含来自燃气管道的 Modbus 通信记录。数据集中同一主-从交互产生 4 行（请求 / 响应 / 写命令 / 写响应），本文以 4 行为 1 个语义单元，定义 8 个窗口级聚合特征。')

add_heading_2(doc, '3.2 27 维特征体系设计')
add_styled_paragraph(doc, '本文提出的 27 维特征体系包括：17 维原始特征（继承自数据集 [15]）+ 3 维行级特征（time_diff / is_attack / function_code_onehot）+ 8 维窗口级特征（response_delay / value_jump / request_freq 等）。')

add_heading_2(doc, '3.3 特征有效性消融实验')
add_styled_paragraph(doc, '使用 LightGBM 作为评估器，Macro-F1 变化如下：')
table = doc.add_table(rows=4, cols=3)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = '特征配置'
hdr[1].text = 'Macro-F1'
hdr[2].text = '对比基线'
for cell in hdr:
    set_cell_shading(cell, 'D9E1F2')

rows_data = [
    ('基线（17 维原始）', '0.8337', '—'),
    ('+ 3 维行级（time_diff 等）', '0.8364', '+0.0027'),
    ('+ 8 维窗口级', '0.8295', '-0.0042'),
]
for i, (cfg, f1, delta) in enumerate(rows_data, 1):
    cells = table.rows[i].cells
    cells[0].text = cfg
    cells[1].text = f1
    cells[2].text = delta

add_diagnostic_box(doc, 'good', 'G-3.1',
    '消融表清晰展示每个特征组的边际贡献，避免"feature engineering = 拍脑袋"的质疑。')

doc.add_page_break()

# ========== 第 4 章 TCN-SE 入侵检测模型 ==========
add_heading_1(doc, '第 4 章 TCN-SE 入侵检测模型')

add_heading_2(doc, '4.1 模型架构')
add_styled_paragraph(doc, 'TCN-SE 由 3 个残差块 + SE 通道注意力 + 全局平均池化 + 全连接分类器组成，参数量约 80K。')

add_heading_2(doc, '4.2 27 模型横向对比')

add_diagnostic_box(doc, 'pitfall', 'P-3.2',
    '消融实验不完整。"为什么是 TCN"必须从横评中给出客观依据，不能直接选最高 F1。')

add_styled_paragraph(doc, '本文对比了 27 种主流时序模型，结果显示 TCN-SE 在 Macro-F1（0.8340）上首次超越 LightGBM（0.8337），同时 PR-AUC 达到 0.9025。')
add_styled_paragraph(doc, '消融实验（去掉 SE 模块）后 Macro-F1 下降至 0.8310（-0.30pp），证明通道注意力对短序列时序建模有效。')

add_heading_2(doc, '4.3 与 Stacking 集成对比')
add_styled_paragraph(doc, '6 模型 LR Stacking（LightGBM + RF + 4 个 TCN 变体）可将 Macro-F1 进一步推至 0.8455，但代价是推理时需要同时运行 6 个模型，部署成本高。')

doc.add_page_break()

# ========== 第 5 章 极限压缩与混合精度量化 ==========
add_heading_1(doc, '第 5 章 极限压缩与混合精度量化')

add_heading_2(doc, '5.1 TCN V19 变体设计')
add_styled_paragraph(doc, '在保持 SE 注意力结构的前提下，将通道数从 64 压缩至 12，参数量从 79K 降至 4,237（-94.7%），PR-AUC 反升至 0.9133（+0.67%）。')

add_heading_2(doc, '5.2 混合精度量化方案')

add_diagnostic_box(doc, 'pitfall', 'P-3.3',
    '无统计显著性检验。TCN V19 提升 0.67pp，应使用 5 折交叉验证或 bootstrap 检验显著性。')

add_styled_paragraph(doc, '本文提出权重 INT8 + 激活 FP32 + BN 折叠的混合方案：')
add_styled_paragraph(doc, '（1）卷积权重与全连接权重 INT8 量化，节省 4× 磁盘；')
add_styled_paragraph(doc, '（2）激活与 BN 保持 FP32，避免 ARM Cortex-M 缺失 INT8 SIMD 指令的瓶颈；')
add_styled_paragraph(doc, '（3）测试集预测一致率 99.80%，SNR ≥ 38 dB 无损。')

add_heading_2(doc, '5.3 MCU 部署实测')
add_styled_paragraph(doc, '在 STM32F407（Cortex-M4 @ 168 MHz）平台上，单样本推理 0.47 ms；模型占 Flash 7 KB、RAM 19 KB。完整 C99 推理代码见附录 A。')

doc.add_page_break()

# ========== 第 6 章 总结与展望 ==========
add_heading_1(doc, '第 6 章 总结与展望')

add_heading_2(doc, '6.1 工作总结')
add_styled_paragraph(doc, '本文围绕 SCADA 入侵检测的"特征—模型—部署"三阶段开展了系统性研究。')
add_styled_paragraph(doc, '首先，分析了 Modbus 4 行主-从交互结构，提出 27 维协议语义特征，使所有模型 Macro-F1 一致提升 0.2—0.7 个百分点。')
add_styled_paragraph(doc, '其次，在 27 种时序模型的横向对比基础上，提出 TCN-SE 模型，在测试集取得 Macro-F1 = 0.8340，首次在该任务上超越 LightGBM。')
add_styled_paragraph(doc, '最后，面向 MCU 部署需求，提出 TCN V19 变体（4,237 参数）与混合精度量化方案，在 Cortex-M4 平台上实现 5.5 KB 模型、0.47 ms 推理。')

add_heading_2(doc, '6.2 研究展望')
add_styled_paragraph(doc, '未来工作可从以下方向展开：（1）联邦学习场景下多 SCADA 站点的协同入侵检测；（2）对抗样本鲁棒性研究；（3）模型可解释性（SHAP 值）的协议级归因。')

doc.add_page_break()

# ========== 参考文献 ==========
add_heading_1(doc, '参考文献')
add_styled_paragraph(doc, '本模板在原始 20 篇基础上扩充至 48 篇，覆盖 SCADA 协议、Modbus 安全、对抗样本、Tabular 深度学习、Edge AI 等方向。完整列表见 thesis/references/refs.bib。', indent=False)

# 列出 5 篇代表
add_styled_paragraph(doc, '[1] ZHOU T, LI Y, WU L. A comprehensive survey of SCADA systems: Architecture, vulnerabilities and defense strategies[J]. Computers & Security, 2022, 118: 102734.', indent=False)
add_styled_paragraph(doc, '[11] KE G, MENG Q, FINLEY T, et al. LightGBM: A highly efficient gradient boosting decision tree[C]//NeurIPS, 2017: 3146-3154.', indent=False)
add_styled_paragraph(doc, '[12] BAI S, KOLTER J Z, KOLTUN V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling[J]. arXiv:1803.01271, 2018.', indent=False)
add_styled_paragraph(doc, '[13] HU J, SHEN L, SUN G. Squeeze-and-excitation networks[C]//CVPR, 2018: 7132-7141.', indent=False)
add_styled_paragraph(doc, '[14] JACOB B, KLIGYS S, CHEN B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//CVPR, 2018: 2704-2713.', indent=False)

doc.add_page_break()

# ========== 致谢 ==========
add_heading_1(doc, '致 谢')
add_styled_paragraph(doc, '本文的研究工作是在导师（教授）的悉心指导下完成的。从选题、研究方案制定到论文撰写，导师都给予了细致的指导与帮助。同时感谢实验室各位同学在实验设计与论文写作中提出的宝贵建议。')
add_styled_paragraph(doc, '特别感谢（家人/爱人）的长期支持与鼓励，使我能够全身心投入研究工作。')
add_styled_paragraph(doc, '最后，向百忙之中评审本文的各位专家致以衷心的感谢。')

doc.add_page_break()

# ========== 附录 A 诊断框使用手册 ==========
add_heading_1(doc, '附录 A 诊断框使用手册')

add_heading_2(doc, 'A.1 4 色诊断样式对照表')
add_diagnostic_box(doc, 'pitfall', 'A.1', '红色 #C00000 边框 + 浅红 #FFEBEB 底纹 + 楷体小四。用于警示学生可能犯的错误。')
add_diagnostic_box(doc, 'todo', 'A.2', '黄色 #BF8F00 边框 + 浅黄 #FFFACD 底纹 + 楷体小四。用于标记待补充内容。')
add_diagnostic_box(doc, 'good', 'A.3', '绿色 #00B050 边框 + 浅绿 #E2EFDA 底纹 + 楷体小四。用于高亮优秀示例。')
add_diagnostic_box(doc, 'grade', 'A.4', '蓝色 #2E75B6 边框 + 浅蓝 #EBF5FF 底纹 + 楷体小四。用于老师评语/打分。')

add_heading_2(doc, 'A.2 快捷键速查')
table = doc.add_table(rows=5, cols=2)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = '快捷键'
hdr[1].text = '功能'
for cell in hdr:
    set_cell_shading(cell, 'D9E1F2')

rows_data = [
    ('Alt+Shift+P', '插入红色 pitfall 框'),
    ('Alt+Shift+T', '插入黄色 todo 框'),
    ('Alt+Shift+G', '插入绿色 good 框'),
    ('Alt+Shift+C', '插入蓝色 grade 框'),
]
for i, (key, func) in enumerate(rows_data, 1):
    cells = table.rows[i].cells
    cells[0].text = key
    cells[1].text = func

add_heading_2(doc, 'A.3 隐藏/显示所有诊断框')
add_styled_paragraph(doc, '按 Ctrl+Shift+H 即可一键切换所有诊断框的显示状态——这是 Word 版的"\\finalfalse"开关。')

add_heading_2(doc, 'A.4 自定义样式导入')
add_styled_paragraph(doc, '打开 Word → 开始 → 样式窗格 → 导入/导出 → 关闭 bas 文件中的样式定义（详见 vba/ThesisTemplate.bas 中的 SetStyles 子程序）。')

# ---------- 保存 ----------
out_path = os.path.join(os.path.dirname(__file__), 'thesis-template.docx')
doc.save(out_path)
print(f'✓ 生成成功: {out_path}')
print(f'  - 文件大小: {os.path.getsize(out_path) / 1024:.1f} KB')
print(f'  - 段落数: {len(doc.paragraphs)}')
print(f'  - 表格数: {len(doc.tables)}')
