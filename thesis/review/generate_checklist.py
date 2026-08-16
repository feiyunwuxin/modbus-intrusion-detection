"""
生成 review-checklist.xlsx — 老师评阅硕士论文的可勾选对照表
================================================================
- 33 类学生常见错误（与 student-pitfalls.md 一一对应）
- 严重度 + 默认扣分 + 评阅勾选 + 老师备注
- 评分汇总页（按 8 维度自动统计）
- 评分标准页（10 分制）

用法：python generate_checklist.py
输出：review-checklist.xlsx （与本脚本同目录）
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment, NamedStyle
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.worksheet.datavalidation import DataValidation
import os

# ---------- 33 类错误数据（与 student-pitfalls.md 同步）----------
# 字段：编号 / 类别 / 子项 / 描述 / 严重度(1-3) / 默认扣分 / 对应维度
PITFALLS = [
    # 一、整体框架（5 类）
    ('P-0.1', '整体框架', '章节逻辑', '第 3 章讲 A，第 4 章突然讲 B，缺乏 A→B 必然性', 2, 1.0, '方法合理性'),
    ('P-0.2', '整体框架', '创新点量化', '创新点无具体指标，仅"该方法有效"', 3, 1.5, '创新点'),
    ('P-0.3', '整体框架', '贡献度位置', '创新点放在第 1 章末但摘要无对应；或放在第 6 章', 2, 1.0, '创新点'),
    ('P-0.4', '整体框架', '论文长度', '正文 30 页（太少）或 100 页（太多）', 1, 0.5, '写作规范'),
    ('P-0.5', '整体框架', '摘要结构', '摘要 = 第 1 章浓缩，缺 4 段式结构', 2, 1.0, '创新点'),

    # 二、摘要与目录（4 类）
    ('P-1.1', '摘要目录', '数据出处', '"据统计"、"研究表明"无引用 [n]', 1, 0.5, '文献综述'),
    ('P-1.2', '摘要目录', '指标一致性', '摘要写"F1=0.8340"正文写"0.8337"', 3, 1.5, '写作规范'),
    ('P-1.3', '摘要目录', '关键词', '关键词 < 5 个或堆砌通用词', 1, 0.5, '写作规范'),
    ('P-1.4', '摘要目录', '章节编号', '章用阿拉伯、节用中文，混用', 1, 0.5, '写作规范'),

    # 三、第 1 章 绪论（6 类）
    ('P-1.5', '第 1 章', '创新对应', '创新点未标"（对应第 X 章）"，且章节无回指', 2, 1.0, '创新点'),
    ('P-1.6', '第 1 章', '虚大空', '"首次"、"首创"、"国际领先"无引文支撑', 3, 1.5, '写作规范'),
    ('P-1.7', '第 1 章', '背景篇幅', '背景 > 1.5 页，写成教科书', 1, 0.5, '写作规范'),
    ('P-1.8', '第 1 章', '文献堆砌', '20+ 篇文献无评价，无"该方法不足"引出', 2, 1.0, '文献综述'),
    ('P-1.9', '第 1 章', '不足凑数', '列 5 条不足，后文只解决 2 条', 1, 0.5, '创新点'),
    ('P-1.10', '第 1 章', '无导览', '1.5 节缺失或只写"第 2 章讲相关工作"', 1, 0.5, '写作规范'),

    # 四、第 2 章 相关工作（4 类）
    ('P-2.1', '第 2 章', '知识科普', '花 2 页讲 CNN/LSTM 数学公式', 1, 0.5, '文献综述'),
    ('P-2.2', '第 2 章', '缺对比表', '纯文字描述 20+ 篇文献优缺点', 2, 1.0, '图表质量'),
    ('P-2.3', '第 2 章', '引用格式', 'GB/T 7714、IEEE、APA 混用', 1, 0.5, '参考文献'),
    ('P-2.4', '第 2 章', '自引不当', '自引 0 次（缺）或 5+ 次（过）', 1, 0.5, '参考文献'),

    # 五、第 3-5 章 方法（4 类）
    ('P-3.1', '方法层', '无基线', '只报告本文方法，不与 SOTA 对比', 3, 1.5, '实验与结果'),
    ('P-3.2', '方法层', '消融不全', '只做 1 个 ablation；复杂方法应有 3-5 组', 3, 1.5, '实验与结果'),
    ('P-3.3', '方法层', '无显著性', 'A=0.8340 vs B=0.8337 直接结论"A 优"，无 t 检验', 2, 1.0, '实验与结果'),
    ('P-3.4', '方法层', '复现性差', '无随机种子、硬件、超参、数据划分', 2, 1.0, '方法合理性'),

    # 六、图表公式（3 类）
    ('P-4.1', '图表公式', '图编号', '图 3-2 写成图 3.2；图未引用', 1, 0.5, '图表质量'),
    ('P-4.2', '图表公式', '表头不规范', '表头无单位；表注缺失；非三线表', 1, 0.5, '图表质量'),
    ('P-4.3', '图表公式', '公式编号', '关键公式未编号；编号跨章不连续', 1, 0.5, '图表质量'),

    # 七、写作排版（3 类）
    ('P-5.1', '写作排版', '长句/口语', '句子 > 50 字；"我们认为"多次出现', 1, 0.5, '写作规范'),
    ('P-5.2', '写作排版', '标题层级', '三级跳五级；同级标题句式不平行', 1, 0.5, '写作规范'),
    ('P-5.3', '写作排版', '致谢失控', '致谢 1 页 +', 1, 0.5, '写作规范'),

    # 八、参考文献（2 类）
    ('P-6.1', '参考文献', '数量过少', '15 篇以下；或 2010 年前文献过多', 1, 0.5, '参考文献'),
    ('P-6.2', '参考文献', '类型混淆', 'arXiv 当正式发表；会议标 Journal', 1, 0.5, '参考文献'),

    # 九、答辩准备（2 类）
    ('P-7.1', '答辩准备', 'PPT 不一致', 'PPT 与论文关键数字不同', 2, 1.0, '写作规范'),
    ('P-7.2', '答辩准备', '不能回答', '"为什么不用 Transformer？""INT8 误差多大？"等核心问题无答案', 2, 1.0, '方法合理性'),
]

# ---------- 工具函数 ----------
def style_header(cell):
    """表头样式：黑体白字 + 深蓝底纹 + 居中 + 边框"""
    cell.font = Font(name='黑体', size=11, bold=True, color='FFFFFF')
    cell.fill = PatternFill('solid', fgColor='1F4E78')
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    cell.border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'))

def style_cell(cell, wrap=True):
    """普通单元格：边框 + 自动换行"""
    cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=wrap)
    cell.border = Border(
        left=Side(style='thin', color='B0B0B0'),
        right=Side(style='thin', color='B0B0B0'),
        top=Side(style='thin', color='B0B0B0'),
        bottom=Side(style='thin', color='B0B0B0'))

def style_severity(cell, level):
    """严重度染色：3=红 / 2=橙 / 1=黄"""
    color = {3: 'FFC7CE', 2: 'FFEB9C', 1: 'FFF2CC'}.get(level, 'FFFFFF')
    cell.fill = PatternFill('solid', fgColor=color)
    cell.font = Font(bold=True)
    cell.alignment = Alignment(horizontal='center', vertical='center')

# ---------- 主生成 ----------
wb = Workbook()

# ========== Sheet 1: 评阅对照表 ==========
ws1 = wb.active
ws1.title = '评阅对照表'

headers = ['编号', '类别', '子项', '典型症状 / 描述', '严重度', '默认扣分', '是否出现',
           '实际扣分', '出现位置', '老师备注']
for col, h in enumerate(headers, 1):
    cell = ws1.cell(row=1, column=col, value=h)
    style_header(cell)

# 列宽
widths = [10, 12, 14, 50, 10, 10, 10, 10, 18, 30]
for i, w in enumerate(widths, 1):
    ws1.column_dimensions[get_column_letter(i)].width = w

# 数据行
for row_idx, (code, cat, sub, desc, sev, default_deduct, dimension) in enumerate(PITFALLS, 2):
    ws1.cell(row=row_idx, column=1, value=code)
    ws1.cell(row=row_idx, column=2, value=cat)
    ws1.cell(row=row_idx, column=3, value=sub)
    ws1.cell(row=row_idx, column=4, value=desc)
    sev_cell = ws1.cell(row=row_idx, column=5, value=sev)
    style_severity(sev_cell, sev)
    ws1.cell(row=row_idx, column=6, value=default_deduct)

    # 是否出现：数据验证（下拉菜单）
    flag_cell = ws1.cell(row=row_idx, column=7, value='否')
    dv = DataValidation(type='list', formula1='"是,否,部分"', allow_blank=False)
    dv.add(flag_cell)
    ws1.add_data_validation(dv)

    # 实际扣分：公式（若"是"则取默认扣分；若"部分"则折半；否则 0）
    formula = f'=IF(G{row_idx}="是",F{row_idx},IF(G{row_idx}="部分",F{row_idx}*0.5,0))'
    ws1.cell(row=row_idx, column=8, value=formula)

    # 出现位置、老师备注：留空
    ws1.cell(row=row_idx, column=9, value='')
    ws1.cell(row=row_idx, column=10, value='')

    # 整行样式
    for col in range(1, 11):
        cell = ws1.cell(row=row_idx, column=col)
        style_cell(cell)
        if col in (1, 2, 3, 5):
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.font = Font(bold=True)

# 行高
ws1.row_dimensions[1].height = 30
for row in range(2, len(PITFALLS) + 2):
    ws1.row_dimensions[row].height = 35

# 冻结首行
ws1.freeze_panes = 'A2'

# 条件格式：实际扣分 > 0 时整行浅红
ws1.conditional_formatting.add(
    f'A2:J{len(PITFALLS) + 1}',
    FormulaRule(formula=[f'$H2>0'], fill=PatternFill('solid', fgColor='FFEBEE')))

# 汇总
sum_row = len(PITFALLS) + 3
ws1.cell(row=sum_row, column=1, value='汇总')
ws1.merge_cells(start_row=sum_row, end_row=sum_row, start_column=1, end_column=6)
ws1.cell(row=sum_row, column=1).font = Font(bold=True, size=12)
ws1.cell(row=sum_row, column=1).alignment = Alignment(horizontal='right', vertical='center')

ws1.cell(row=sum_row, column=7, value=f'=COUNTIF(G2:G{len(PITFALLS) + 1},"是")')
ws1.cell(row=sum_row, column=7).font = Font(bold=True, size=12, color='C00000')
ws1.cell(row=sum_row, column=7).alignment = Alignment(horizontal='center', vertical='center')

ws1.cell(row=sum_row, column=8, value=f'=SUM(H2:H{len(PITFALLS) + 1})')
ws1.cell(row=sum_row, column=8).font = Font(bold=True, size=12, color='C00000')
ws1.cell(row=sum_row, column=8).alignment = Alignment(horizontal='center', vertical='center')

ws1.cell(row=sum_row, column=9, value='（出现 / 扣分）')
ws1.merge_cells(start_row=sum_row, end_row=sum_row, start_column=9, end_column=10)
ws1.cell(row=sum_row, column=9).font = Font(italic=True, color='808080')
ws1.cell(row=sum_row, column=9).alignment = Alignment(horizontal='left', vertical='center')

# 给汇总标签加注释
for col in range(1, 11):
    cell = ws1.cell(row=sum_row, column=col)
    cell.fill = PatternFill('solid', fgColor='D9E1F2')
    cell.border = Border(
        top=Side(style='medium', color='1F4E78'),
        bottom=Side(style='medium', color='1F4E78'))

# ========== Sheet 2: 评分汇总 ==========
ws2 = wb.create_sheet('评分汇总')

ws2.cell(row=1, column=1, value='硕士论文评阅 — 8 维度评分汇总')
ws2.merge_cells('A1:E1')
ws2.cell(row=1, column=1).font = Font(name='黑体', size=14, bold=True, color='FFFFFF')
ws2.cell(row=1, column=1).fill = PatternFill('solid', fgColor='1F4E78')
ws2.cell(row=1, column=1).alignment = Alignment(horizontal='center', vertical='center')
ws2.row_dimensions[1].height = 30

dim_headers = ['维度', '满分', '扣分（来自对照表）', '实得分', '评级']
for col, h in enumerate(dim_headers, 1):
    cell = ws2.cell(row=3, column=col, value=h)
    style_header(cell)

# 8 维度
DIMENSIONS = [
    ('选题意义', 1.0),
    ('创新点', 2.0),
    ('文献综述', 1.0),
    ('方法合理性', 2.0),
    ('实验与结果', 2.0),
    ('写作规范', 1.0),
    ('图表质量', 0.5),
    ('参考文献', 0.5),
]

# 每个维度统计 PITFALLS 中对应维度的扣分
dim_to_codes = {}
for code, cat, sub, desc, sev, default_deduct, dim in PITFALLS:
    dim_to_codes.setdefault(dim, []).append((code, default_deduct))

ws2.row_dimensions[2].height = 6

for i, (dim, full) in enumerate(DIMENSIONS, 4):
    ws2.cell(row=i, column=1, value=dim)
    ws2.cell(row=i, column=2, value=full)

    # 扣分公式：用 SUMPRODUCT 跨表统计
    # 在对照表中，H列是实际扣分；我们要按"对应维度"列（J=10）的值筛选
    codes = dim_to_codes.get(dim, [])
    if codes:
        # 累加该维度下所有 P-X.Y 的实际扣分
        sum_parts = [f'SUMIF(评阅对照表!A:A,"{code}",评阅对照表!H:H)' for code, _ in codes]
        formula = '=' + '+'.join(sum_parts)
        ws2.cell(row=i, column=3, value=formula)
    else:
        ws2.cell(row=i, column=3, value=0)

    # 实得分 = 满分 - 扣分（不小于 0）
    ws2.cell(row=i, column=4, value=f'=MAX(0,B{i}-C{i})')

    # 评级：≥ 0.85 满分为"优"，≥ 0.7 为"良"，否则"中/差"
    ws2.cell(row=i, column=5, value=f'=IF(D{i}/B{i}>=0.85,"优",IF(D{i}/B{i}>=0.7,"良",IF(D{i}/B{i}>=0.55,"中","差")))')

    for col in range(1, 6):
        cell = ws2.cell(row=i, column=col)
        style_cell(cell)
        if col >= 2:
            cell.alignment = Alignment(horizontal='center', vertical='center')

# 总分行
total_row = 4 + len(DIMENSIONS)
ws2.cell(row=total_row, column=1, value='总分').font = Font(bold=True, size=12)
ws2.cell(row=total_row, column=2, value=f'=SUM(B4:B{total_row-1})')
ws2.cell(row=total_row, column=3, value=f'=SUM(C4:C{total_row-1})')
ws2.cell(row=total_row, column=4, value=f'=SUM(D4:D{total_row-1})')
ws2.cell(row=total_row, column=5, value=f'=IF(D{total_row}>=8,"优秀",IF(D{total_row}>=6,"合格","需大改"))')

for col in range(1, 6):
    cell = ws2.cell(row=total_row, column=col)
    cell.font = Font(bold=True, size=12, color='FFFFFF')
    cell.fill = PatternFill('solid', fgColor='1F4E78')
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.border = Border(
        top=Side(style='medium'), bottom=Side(style='medium'),
        left=Side(style='thin'), right=Side(style='thin'))

# 列宽
ws2.column_dimensions['A'].width = 18
ws2.column_dimensions['B'].width = 10
ws2.column_dimensions['C'].width = 20
ws2.column_dimensions['D'].width = 12
ws2.column_dimensions['E'].width = 12

# 注释：通过标准
note_row = total_row + 2
ws2.cell(row=note_row, column=1, value='通过标准')
ws2.cell(row=note_row, column=1).font = Font(bold=True, color='C00000')
ws2.cell(row=note_row + 1, column=1, value='≥ 8.0 优秀 | ≥ 6.0 合格 | < 6.0 需大改后重审')
ws2.cell(row=note_row + 1, column=1).font = Font(italic=True, color='606060')
ws2.merge_cells(start_row=note_row + 1, end_row=note_row + 1, start_column=1, end_column=5)

# ========== Sheet 3: 评分标准 ==========
ws3 = wb.create_sheet('评分标准')

ws3.cell(row=1, column=1, value='评分标准（10 分制，参考学校研究生论文评审标准）')
ws3.merge_cells('A1:D1')
ws3.cell(row=1, column=1).font = Font(name='黑体', size=14, bold=True, color='FFFFFF')
ws3.cell(row=1, column=1).fill = PatternFill('solid', fgColor='1F4E78')
ws3.cell(row=1, column=1).alignment = Alignment(horizontal='center', vertical='center')
ws3.row_dimensions[1].height = 30

std_headers = ['维度', '满分', '评分要点', '不合格典型']
for col, h in enumerate(std_headers, 1):
    cell = ws3.cell(row=3, column=col, value=h)
    style_header(cell)

std_data = [
    ('选题意义', 1.0, '明确的应用场景；具体要解决的工程/科学问题；问题规模合理',
     '无应用场景；问题宽泛；选题过大或过小'),
    ('创新点', 2.0, '每个创新点有量化指标；与现有工作对比清晰；3 大创新点对应 3 章',
     '"首次/首创"无引文；无对应章节；指标不可复现'),
    ('文献综述', 1.0, '本领域代表工作齐全；每段有评价；含统一对比表',
     '堆砌 20+ 篇无评价；缺对比表；近 5 年文献 < 60%'),
    ('方法合理性', 2.0, '方法有理论依据；复杂度可控；可复现（种子/硬件/超参齐全）',
     '无理论动机；复杂度不可控；无法复现'),
    ('实验与结果', 2.0, '基线 + SOTA + 本文方法 3 组对比；3-5 组消融；统计显著性检验',
     '无基线；无消融；单次实验下结论；无显著性'),
    ('写作规范', 1.0, '句子 ≤ 50 字；术语一致；致谢 200-400 字；无错别字',
     '长句口语化；术语前后不一；致谢 1 页+；错别字多'),
    ('图表质量', 0.5, '三线表；图编号引用一致；公式编号连续',
     '非三线表；图未引用；公式编号跨章不连续'),
    ('参考文献', 0.5, '40-60 篇；近 5 年 ≥ 60%；GB/T 7714 格式；自引 1-2 次',
     '< 15 篇；arXiv 当正式发表；格式混用；自引 5+'),
]

for i, (dim, full, key, bad) in enumerate(std_data, 4):
    ws3.cell(row=i, column=1, value=dim)
    ws3.cell(row=i, column=2, value=full)
    ws3.cell(row=i, column=3, value=key)
    ws3.cell(row=i, column=4, value=bad)
    for col in range(1, 5):
        cell = ws3.cell(row=i, column=col)
        style_cell(cell)
        if col <= 2:
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.font = Font(bold=True)

# 列宽
ws3.column_dimensions['A'].width = 14
ws3.column_dimensions['B'].width = 10
ws3.column_dimensions['C'].width = 45
ws3.column_dimensions['D'].width = 45
for r in range(4, 4 + len(std_data)):
    ws3.row_dimensions[r].height = 50

# ========== Sheet 4: 使用说明 ==========
ws4 = wb.create_sheet('使用说明')

ws4.cell(row=1, column=1, value='老师评阅流程 — 4 步 10 分钟')
ws4.merge_cells('A1:B1')
ws4.cell(row=1, column=1).font = Font(name='黑体', size=14, bold=True, color='FFFFFF')
ws4.cell(row=1, column=1).fill = PatternFill('solid', fgColor='1F4E78')
ws4.cell(row=1, column=1).alignment = Alignment(horizontal='center', vertical='center')
ws4.row_dimensions[1].height = 30

steps = [
    ('步骤 1', '打开"评阅对照表"，按 P-0.1 ~ P-7.2 顺序逐项检查（5 分钟）'),
    ('', '"是否出现"列：选"是 / 否 / 部分"，Excel 自动计算"实际扣分"'),
    ('', '"出现位置"列：填章节号或页码（例：第 3.2 节, p.42）'),
    ('', '"老师备注"列：记录具体问题（例：与 §3.1 数据矛盾）'),
    ('', ''),
    ('步骤 2', '对照表会自动加和：累计扣分 → 看右下方"汇总"行的扣分合计'),
    ('', ''),
    ('步骤 3', '切换到"评分汇总"Sheet — 8 维度按 PITFALL 自动汇总扣分；不需要手动算'),
    ('', '看右下角"总分"：≥ 8.0 优秀 / ≥ 6.0 合格 / < 6.0 需大改'),
    ('', ''),
    ('步骤 4', '对照"评分标准"Sheet，与学生面谈时引用具体评分要点'),
    ('', ''),
    ('快捷操作', '① 在"是否出现"列单击单元格 → 出现下拉箭头 → 选"是"'),
    ('', '② 改完后按 F9 强制刷新公式（或关闭重开）'),
    ('', '③ 打印：选中"评阅对照表" + "评分汇总"两页 → 打印'),
]

for i, (step, desc) in enumerate(steps, 3):
    ws4.cell(row=i, column=1, value=step).font = Font(bold=True, color='1F4E78' if step else '000000')
    ws4.cell(row=i, column=1).alignment = Alignment(horizontal='right', vertical='center')
    ws4.cell(row=i, column=2, value=desc)
    ws4.cell(row=i, column=2).alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)

# 列宽
ws4.column_dimensions['A'].width = 12
ws4.column_dimensions['B'].width = 80

# 文件信息
info_row = len(steps) + 5
ws4.cell(row=info_row, column=1, value='文件信息')
ws4.cell(row=info_row, column=1).font = Font(bold=True, color='FFFFFF')
ws4.cell(row=info_row, column=1).fill = PatternFill('solid', fgColor='1F4E78')
ws4.cell(row=info_row, column=2, value='')
ws4.cell(row=info_row, column=2).fill = PatternFill('solid', fgColor='1F4E78')

info_data = [
    ('文件名', 'review-checklist.xlsx'),
    ('生成日期', '2026-06-16'),
    ('配套文档', '../student-pitfalls.md（33 类错误详解）'),
    ('配套模板', '../main.tex（LaTeX 模板）/ thesis-template.docx（Word 模板）'),
    ('生成脚本', 'generate_checklist.py（可重新生成）'),
    ('PITFALL 同步', '33 项与 student-pitfalls.md P-0.1 ~ P-7.2 一一对应'),
]
for i, (k, v) in enumerate(info_data, info_row + 1):
    ws4.cell(row=i, column=1, value=k).font = Font(bold=True)
    ws4.cell(row=i, column=1).alignment = Alignment(horizontal='right', vertical='center')
    ws4.cell(row=i, column=2, value=v)
    ws4.cell(row=i, column=2).alignment = Alignment(horizontal='left', vertical='center')

# ---------- 保存 ----------
out_path = os.path.join(os.path.dirname(__file__), 'review-checklist.xlsx')
wb.save(out_path)
print(f'OK: {out_path}')
print(f'  - Size: {os.path.getsize(out_path) / 1024:.1f} KB')
print(f'  - Sheets: {wb.sheetnames}')
print(f'  - PITFALLS: {len(PITFALLS)}')
