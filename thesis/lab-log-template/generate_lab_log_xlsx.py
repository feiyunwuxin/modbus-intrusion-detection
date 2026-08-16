"""
generate_lab_log_xlsx.py — 实验日志 Excel 数据记录表
================================================================
- 每行 1 个实验，14 列结构化数据
- 含 3 个示例实验 + 表头
- 与 lab-log.md / pitfalls-lab-log.md 配合使用

用法：python generate_lab_log_xlsx.py
输出：lab-log-template.xlsx
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
import os

# ---------- 工具函数 ----------
def style_header(cell):
    cell.font = Font(name='黑体', size=11, bold=True, color='FFFFFF')
    cell.fill = PatternFill('solid', fgColor='1F4E78')
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    cell.border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'))


def style_cell(cell, wrap=True):
    cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=wrap)
    cell.border = Border(
        left=Side(style='thin', color='B0B0B0'),
        right=Side(style='thin', color='B0B0B0'),
        top=Side(style='thin', color='B0B0B0'),
        bottom=Side(style='thin', color='B0B0B0'))


# ---------- 主生成 ----------
wb = Workbook()

# ========== Sheet 1: 实验记录 ==========
ws = wb.active
ws.title = '实验记录'

headers = [
    '日期', '阶段', '实验编号', '数据集', '模型', '主要超参 (JSON)',
    '种子', 'Macro-F1', 'PR-AUC', 'Minority Recall',
    '训练时长(秒)', 'Git commit', '关键发现', '下一步',
]
for col, h in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=h)
    style_header(cell)

# 列宽
widths = [12, 10, 12, 22, 16, 35, 8, 10, 10, 14, 12, 12, 35, 30]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w

# 3 行示例数据
examples = [
    ('2026-09-10', '第 1 阶段', 'exp_001', 'IanArffDataset.csv', 'LightGBM',
     '{"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 500}',
     42, 0.8212, 0.8712, 0.6234, 8.2, 'a1b2c3d',
     '基线 LGB，原始 17 维；DoS 类 Recall 偏低',
     '加入 27 维特征，对比 LGB / RF / XGB'),

    ('2026-09-15', '第 2 阶段', 'exp_002', 'IanArffDataset_v2.csv (27 维)', 'LightGBM',
     '{"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 1000, "early_stopping": 50}',
     42, 0.8337, 0.8873, 0.7124, 12.4, 'b2c3d4e',
     '27 维特征使 Macro-F1 提升 +0.0125；Minority Recall 提升 +0.089',
     '对比 num_leaves 31/63/127；试 SMOTE 解决 DoS 不平衡'),

    ('2026-09-18', '第 2 阶段', 'exp_003', 'IanArffDataset_v2.csv (27 维)', 'LightGBM + SMOTE',
     '{"num_leaves": 63, "smote_k": 5, "smote_ratio": 0.5}',
     42, 0.8401, 0.8902, 0.7854, 14.1, 'c3d4e5f',
     'SMOTE 进一步 +0.0064；DoS Recall 0.62 -> 0.78',
     '进入第 3 阶段：尝试 1D-CNN / LSTM / TCN'),
]

for row_idx, ex in enumerate(examples, 2):
    for col, val in enumerate(ex, 1):
        cell = ws.cell(row=row_idx, column=col, value=val)
        style_cell(cell)
        if col in (1, 2, 3, 7, 8, 9, 10, 11):
            cell.alignment = Alignment(horizontal='center', vertical='center')

# 条件格式：Macro-F1 < 0.83 标黄
from openpyxl.formatting.rule import CellIsRule
ws.conditional_formatting.add(
    'H2:J100',
    CellIsRule(operator='lessThan', formula=['0.83'],
               fill=PatternFill('solid', fgColor='FFEB9C')))

# 数据验证：阶段列
dv = DataValidation(type='list',
                    formula1='"第 1 阶段,第 2 阶段,第 3 阶段,第 4 阶段,第 5 阶段"',
                    allow_blank=True)
dv.add('B2:B1000')
ws.add_data_validation(dv)

# 行高
ws.row_dimensions[1].height = 35
for r in range(2, 5):
    ws.row_dimensions[r].height = 60

# 冻结
ws.freeze_panes = 'A2'

# ========== Sheet 2: 8 字段详细日志（每日） ==========
ws2 = wb.create_sheet('每日详细日志')

detail_headers = ['日期', '阶段', '今日目标', '环境 (Python/包/硬件)',
                  '数据 (集/划分/分布)', '模型与超参', '结果 (含失败)',
                  '分析与下一步', '复现性自检']
for col, h in enumerate(detail_headers, 1):
    cell = ws2.cell(row=1, column=col, value=h)
    style_header(cell)

# 列宽
widths2 = [12, 10, 35, 30, 30, 35, 35, 40, 25]
for i, w in enumerate(widths2, 1):
    ws2.column_dimensions[get_column_letter(i)].width = w

# 1 行示例
example_detail = (
    '2026-09-15', '第 2 阶段',
    '对比 LGB num_leaves 31/63/127；测试 27 维 vs 17 维',
    'Python 3.9.13, lightgbm 3.3.5, RTX 3090, CUDA 11.7',
    'IanArffDataset.csv 274K 行；7:1.5:1.5 划分；DoS 14% / Probe 8% / ...',
    'LGB: num_leaves=63, lr=0.05, n_est=1000, early_stop=50, seed=42',
    'Macro-F1=0.8337, PR-AUC=0.8873; 失败: DoS Recall=0.58',
    'num_leaves=63 是 Pareto 拐点; 下一步: 试 SMOTE',
    '7/8 已完成 (split/json/script/git/save/分析); 缺 跨日对比',
)

for col, val in enumerate(example_detail, 1):
    cell = ws2.cell(row=2, column=col, value=val)
    style_cell(cell)

ws2.row_dimensions[1].height = 35
ws2.row_dimensions[2].height = 80

# 冻结
ws2.freeze_panes = 'A2'

# ========== Sheet 3: 跨周对比 ==========
ws3 = wb.create_sheet('跨周对比')

week_headers = ['周次', '起止日期', '本周实验数', '最佳 Macro-F1',
                '最佳 PR-AUC', '关键进展', '下周计划']
for col, h in enumerate(week_headers, 1):
    cell = ws3.cell(row=1, column=col, value=h)
    style_header(cell)

widths3 = [10, 22, 14, 14, 14, 40, 40]
for i, w in enumerate(widths3, 1):
    ws3.column_dimensions[get_column_letter(i)].width = w

# 3 周示例
week_examples = [
    ('W1', '2026-09-08 ~ 09-14', 8, 0.8212, 0.8712, '基线 LGB 跑通；27 维特征设计完成', '开始 27 维特征实验'),
    ('W2', '2026-09-15 ~ 09-21', 12, 0.8401, 0.8902, '27 维 + SMOTE 提升明显', '开始 DL 模型'),
    ('W3', '2026-09-22 ~ 09-28', 15, 0.8523, 0.8954, 'TCN-SE 超越 LGB 0.8340', '消融实验 + 加宽'),
]

for row_idx, wk in enumerate(week_examples, 2):
    for col, val in enumerate(wk, 1):
        cell = ws3.cell(row=row_idx, column=col, value=val)
        style_cell(cell)
        if col in (1, 2, 3, 4, 5):
            cell.alignment = Alignment(horizontal='center', vertical='center')

ws3.row_dimensions[1].height = 30
for r in range(2, 5):
    ws3.row_dimensions[r].height = 40

# ========== Sheet 4: 12 类日志错误对照表 ==========
ws4 = wb.create_sheet('12 类错误对照')

err_headers = ['编号', '类别', '错误', '正确做法', '自检']
for col, h in enumerate(err_headers, 1):
    cell = ws4.cell(row=1, column=col, value=h)
    style_header(cell)

errors = [
    ('P-L.1', '目标', '目标宽泛（"调模型"）', '1-2 个可验证子目标', '□'),
    ('P-L.2', '目标', '目标无验收标准', '明确"Macro-F1 ≥ X 即采纳"', '□'),
    ('P-L.3', '环境', '环境不记录', 'pip freeze + GPU 型号', '□'),
    ('P-L.4', '环境', 'Git commit 缺失', '每天 commit 1 次', '□'),
    ('P-L.5', '超参', '超参不全', '列出所有超参与值', '□'),
    ('P-L.6', '数据', '数据划分文件未保存', '保存 split_xxx.json', '□'),
    ('P-L.7', '结果', '选择性报告', '同时报告成功与失败', '□'),
    ('P-L.8', '结果', '关键指标缺失', '报告 ≥ 3 个指标', '□'),
    ('P-L.9', '分析', '无观察分析', '至少 2-3 句分析', '□'),
    ('P-L.10', '计划', '无下一步计划', '明确"明天要试什么"', '□'),
    ('P-L.11', '复现', '无复现性自检', '填 7 项自检清单', '□'),
    ('P-L.12', '复现', '模型权重未保存', '保存 model_xxx.pkl', '□'),
]

for row_idx, e in enumerate(errors, 2):
    for col, val in enumerate(e, 1):
        cell = ws4.cell(row=row_idx, column=col, value=val)
        style_cell(cell)
        if col in (1, 2, 5):
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.font = Font(bold=True)
        if col == 5:
            # 自检列加 checkbox-like 字体
            cell.font = Font(name='Wingdings 2', size=14, color='808080')

# 条件格式：勾选时变绿
from openpyxl.formatting.rule import CellIsRule
ws4.conditional_formatting.add(
    'E2:E13',
    CellIsRule(operator='equal', formula=['"☑"'],
               fill=PatternFill('solid', fgColor='C6EFCE'),
               font=Font(name='Wingdings 2', size=14, bold=True, color='006100')))

# 列宽
widths4 = [8, 10, 30, 35, 10]
for i, w in enumerate(widths4, 1):
    ws4.column_dimensions[get_column_letter(i)].width = w

for r in range(2, 14):
    ws4.row_dimensions[r].height = 30

# ---------- 保存 ----------
out_path = os.path.join(os.path.dirname(__file__), 'lab-log-template.xlsx')
wb.save(out_path)
print(f'OK: {out_path}')
print(f'  - Size: {os.path.getsize(out_path) / 1024:.1f} KB')
print(f'  - Sheets: {wb.sheetnames}')
