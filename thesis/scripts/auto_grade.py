"""
auto_grade.py — 自动扫描论文中的 pitfall 标注并评分
================================================================
- 扫描 .tex / .docx / .md 文件
- 提取 \\pitfall{P-X.Y}、VBA 框（[P-X.Y] 模式）、Markdown pitfall 标注
- 按章节、严重度、维度统计
- 输出 report.json + 终端汇总
- 配套 review-checklist.xlsx 使用

用法：
    python auto_grade.py                            # 扫描整个 thesis/ 目录
    python auto_grade.py thesis/main.tex            # 扫描单个文件
    python auto_grade.py --json report.json        # 输出 JSON 报告
    python auto_grade.py --chapter 3                # 只看第 3 章的 pitfall
    python auto_grade.py --missing                  # 检查哪些 P-X.Y 编号还没出现

PITFALL 严重度与维度映射来自 student-pitfalls.md (33 类)
"""
import re
import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# ---------- 33 类 pitfall 配置（与 student-pitfalls.md / review-checklist.xlsx 同步）----------
# 字段：编号 / 类别 / 严重度(1-3) / 默认扣分 / 对应维度
PITFALL_DB = {
    # 一、整体框架（5 类）
    'P-0.1': ('整体框架', '章节逻辑', 2, 1.0, '方法合理性'),
    'P-0.2': ('整体框架', '创新点量化', 3, 1.5, '创新点'),
    'P-0.3': ('整体框架', '贡献度位置', 2, 1.0, '创新点'),
    'P-0.4': ('整体框架', '论文长度', 1, 0.5, '写作规范'),
    'P-0.5': ('整体框架', '摘要结构', 2, 1.0, '创新点'),
    # 二、摘要与目录（4 类）
    'P-1.1': ('摘要目录', '数据出处', 1, 0.5, '文献综述'),
    'P-1.2': ('摘要目录', '指标一致性', 3, 1.5, '写作规范'),
    'P-1.3': ('摘要目录', '关键词', 1, 0.5, '写作规范'),
    'P-1.4': ('摘要目录', '章节编号', 1, 0.5, '写作规范'),
    # 三、第 1 章 绪论（6 类）
    'P-1.5': ('第 1 章', '创新对应', 2, 1.0, '创新点'),
    'P-1.6': ('第 1 章', '虚大空', 3, 1.5, '写作规范'),
    'P-1.7': ('第 1 章', '背景篇幅', 1, 0.5, '写作规范'),
    'P-1.8': ('第 1 章', '文献堆砌', 2, 1.0, '文献综述'),
    'P-1.9': ('第 1 章', '不足凑数', 1, 0.5, '创新点'),
    'P-1.10': ('第 1 章', '无导览', 1, 0.5, '写作规范'),
    # 四、第 2 章 相关工作（4 类）
    'P-2.1': ('第 2 章', '知识科普', 1, 0.5, '文献综述'),
    'P-2.2': ('第 2 章', '缺对比表', 2, 1.0, '图表质量'),
    'P-2.3': ('第 2 章', '引用格式', 1, 0.5, '参考文献'),
    'P-2.4': ('第 2 章', '自引不当', 1, 0.5, '参考文献'),
    # 五、方法层（4 类）
    'P-3.1': ('方法层', '无基线', 3, 1.5, '实验与结果'),
    'P-3.2': ('方法层', '消融不全', 3, 1.5, '实验与结果'),
    'P-3.3': ('方法层', '无显著性', 2, 1.0, '实验与结果'),
    'P-3.4': ('方法层', '复现性差', 2, 1.0, '方法合理性'),
    # 六、图表公式（3 类）
    'P-4.1': ('图表公式', '图编号', 1, 0.5, '图表质量'),
    'P-4.2': ('图表公式', '表头不规范', 1, 0.5, '图表质量'),
    'P-4.3': ('图表公式', '公式编号', 1, 0.5, '图表质量'),
    # 七、写作排版（3 类）
    'P-5.1': ('写作排版', '长句/口语', 1, 0.5, '写作规范'),
    'P-5.2': ('写作排版', '标题层级', 1, 0.5, '写作规范'),
    'P-5.3': ('写作排版', '致谢失控', 1, 0.5, '写作规范'),
    # 八、参考文献（2 类）
    'P-6.1': ('参考文献', '数量过少', 1, 0.5, '参考文献'),
    'P-6.2': ('参考文献', '类型混淆', 1, 0.5, '参考文献'),
    # 九、答辩准备（2 类）
    'P-7.1': ('答辩准备', 'PPT 不一致', 2, 1.0, '写作规范'),
    'P-7.2': ('答辩准备', '不能回答', 2, 1.0, '方法合理性'),
}

DIM_FULL = {  # 8 维度满分（与 review-checklist.xlsx 同步）
    '选题意义': 1.0,
    '创新点': 2.0,
    '文献综述': 1.0,
    '方法合理性': 2.0,
    '实验与结果': 2.0,
    '写作规范': 1.0,
    '图表质量': 0.5,
    '参考文献': 0.5,
}


# ---------- 文件类型识别 ----------
def detect_file_type(path):
    ext = Path(path).suffix.lower()
    return {'.tex': 'latex', '.md': 'markdown', '.docx': 'docx'}.get(ext, 'unknown')


# ---------- LaTeX pitfall 提取 ----------
def extract_latex(text, source_path):
    """
    匹配 \\pitfall{P-X.Y}{描述} 或 \\pitfall{P-X.Y}
    同时匹配 \\todo{T-X.Y}, \\good{G-X.Y}, \\grade{A/B/...}
    """
    pitfall_pattern = re.compile(
        r'\\pitfall\s*\{\s*(P-[\d.]+)\s*(?:\}\s*\{([^}]*)\})?',
        re.MULTILINE
    )
    todo_pattern = re.compile(r'\\todo\s*\{\s*(T-[\d.]+)', re.MULTILINE)
    good_pattern = re.compile(r'\\good\s*\{\s*(G-[\d.]+)', re.MULTILINE)
    grade_pattern = re.compile(r'\\grade\s*\{\s*([A-F][\+\-]?)', re.MULTILINE)

    findings = []
    for m in pitfall_pattern.finditer(text):
        code = m.group(1)
        desc = m.group(2) or ''
        line_no = text[:m.start()].count('\n') + 1
        findings.append({
            'code': code, 'type': 'pitfall', 'desc': desc.strip(),
            'line': line_no, 'source': source_path,
        })
    for m in todo_pattern.finditer(text):
        line_no = text[:m.start()].count('\n') + 1
        findings.append({
            'code': m.group(1), 'type': 'todo', 'desc': '',
            'line': line_no, 'source': source_path,
        })
    for m in good_pattern.finditer(text):
        line_no = text[:m.start()].count('\n') + 1
        findings.append({
            'code': m.group(1), 'type': 'good', 'desc': '',
            'line': line_no, 'source': source_path,
        })
    for m in grade_pattern.finditer(text):
        line_no = text[:m.start()].count('\n') + 1
        findings.append({
            'code': f'GRADE-{m.group(1)}', 'type': 'grade', 'desc': '',
            'line': line_no, 'source': source_path,
        })
    return findings


# ---------- Markdown pitfall 提取 ----------
def extract_markdown(text, source_path):
    """
    匹配中文模式的标注：
      ⚠️ 【学生常见错误 P-1.1】 描述...
      📝 【待补充（学生） T-1.1】
      ✓ 【优秀示例 G-1.1】
      ✏️ 【老师评语 [等级：B]】
    """
    patterns = [
        (r'【学生常见错误\s+(P-[\d.]+)】\s*([^\n]*)', 'pitfall'),
        (r'【待补充[（(]学生[)）]?\s+(T-[\d.]+)】\s*([^\n]*)', 'todo'),
        (r'【优秀示例\s+(G-[\d.]+)】\s*([^\n]*)', 'good'),
        (r'【老师评语[^\]]*?等级[：:]\s*([A-F][\+\-]?)[^】]*】', 'grade'),
    ]
    findings = []
    for pat, ptype in patterns:
        for m in re.finditer(pat, text):
            code = m.group(1)
            desc = m.group(2) if m.lastindex and m.lastindex >= 2 else ''
            line_no = text[:m.start()].count('\n') + 1
            findings.append({
                'code': code, 'type': ptype,
                'desc': desc.strip() if desc else '',
                'line': line_no, 'source': source_path,
            })
    return findings


# ---------- .docx pitfall 提取 ----------
def extract_docx(source_path):
    """从 docx 表格中提取诊断框（前缀 emoji + P-1.x 等）"""
    try:
        from docx import Document
    except ImportError:
        print('  ! 需要 python-docx: pip install python-docx', file=sys.stderr)
        return []

    doc = Document(source_path)
    findings = []
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()
                for ptype, pat, code_grp in [
                    ('pitfall', r'【学生常见错误\s+(P-[\d.]+)】\s*([^\n]*)', 1),
                    ('todo',    r'【待补充[（(]学生[)）]?\s+(T-[\d.]+)】\s*([^\n]*)', 1),
                    ('good',    r'【优秀示例\s+(G-[\d.]+)】\s*([^\n]*)', 1),
                    ('grade',   r'【老师评语[^\]]*?等级[：:]\s*([A-F][\+\-]?)[^】]*】', 1),
                ]:
                    for m in re.finditer(pat, text):
                        code = m.group(code_grp)
                        desc = m.group(2) if m.lastindex and m.lastindex >= 2 else ''
                        findings.append({
                            'code': code, 'type': ptype,
                            'desc': desc.strip() if desc else '',
                            'line': 0, 'source': source_path,
                        })
    return findings


# ---------- 章节识别 ----------
def detect_chapter(file_path, line_no=None):
    r"""
    从文件路径或行号推断章节
    文件路径约定：chapter1.tex, chapter2.tex, chapters/chapter3.tex
    行号约定: 扫描附近的 \chapter{} 或 # 第 N 章
    """
    p = Path(file_path).name.lower()

    # 优先从文件名
    m = re.search(r'chapter(\d+)', p)
    if m:
        return f'第 {m.group(1)} 章'

    # 次之从父目录
    parent = Path(file_path).parent.name.lower()
    if 'chapter' in parent:
        m = re.search(r'chapter(\d+)', parent)
        if m:
            return f'第 {m.group(1)} 章'

    # 从完整路径
    full = str(file_path).lower()
    m = re.search(r'chapter\s*(\d+)', full)
    if m:
        return f'第 {m.group(1)} 章'

    return '其他'


# ---------- 统计 ----------
def grade(findings):
    r"""
    输入 findings 列表
    输出评分结果（按 8 维度）
    """
    # 1) pitfall 严重度分布
    sev_dist = Counter()  # severity -> count
    dim_deduct = defaultdict(float)  # dim -> total deduct
    chapter_dist = Counter()  # chapter -> pitfall count
    by_code = defaultdict(lambda: {'count': 0, 'chapter': set(), 'lines': []})
    unknown_codes = set()

    for f in findings:
        if f['type'] != 'pitfall':
            continue
        code = f['code']
        if code not in PITFALL_DB:
            unknown_codes.add(code)
            continue
        cat, sub, sev, default_deduct, dim = PITFALL_DB[code]
        sev_dist[sev] += 1
        dim_deduct[dim] += default_deduct
        chapter = detect_chapter(f['source'], f.get('line'))
        chapter_dist[chapter] += 1
        by_code[code]['count'] += 1
        by_code[code]['chapter'].add(chapter)
        by_code[code]['lines'].append((f['source'], f.get('line', 0)))

    # 2) 评分
    dim_scores = {}
    for dim, full in DIM_FULL.items():
        deduct = dim_deduct.get(dim, 0.0)
        score = max(0.0, full - deduct)
        ratio = score / full if full > 0 else 0
        if ratio >= 0.85:
            grade_letter = 'A'
        elif ratio >= 0.7:
            grade_letter = 'B'
        elif ratio >= 0.55:
            grade_letter = 'C'
        else:
            grade_letter = 'D'
        dim_scores[dim] = {
            'full': full, 'deduct': round(deduct, 2),
            'score': round(score, 2), 'letter': grade_letter,
        }

    total_full = sum(DIM_FULL.values())
    total_score = sum(s['score'] for s in dim_scores.values())
    if total_score >= 8.0:
        overall = '优秀'
    elif total_score >= 6.0:
        overall = '合格'
    else:
        overall = '需大改'

    # 3) todo / good / grade 统计
    type_dist = Counter(f['type'] for f in findings)

    return {
        'summary': {
            'total_pitfalls': sum(sev_dist.values()),
            'severity_dist': dict(sev_dist),
            'type_dist': dict(type_dist),
            'total_deduct': round(sum(s['deduct'] for s in dim_scores.values()), 2),
            'total_score': round(total_score, 2),
            'total_full': total_full,
            'overall': overall,
            'unknown_codes': sorted(unknown_codes),
        },
        'by_chapter': dict(chapter_dist.most_common()),
        'by_code': {
            code: {
                'count': v['count'],
                'chapter': sorted(v['chapter']),
                'lines': v['lines'][:5],  # 最多 5 处
            }
            for code, v in sorted(by_code.items())
        },
        'by_dimension': dim_scores,
    }


# ---------- 报告输出 ----------
def print_terminal_report(findings, report, args):
    s = report['summary']
    print('=' * 70)
    print(f'  论文评阅报告 — {args.target or "thesis/"}')
    print('=' * 70)
    print()
    print(f'  扫描文件: {len(set(f["source"] for f in findings))} 个')
    print(f'  pitfall 总数: {s["total_pitfalls"]} (严重 3={s["severity_dist"].get(3, 0)}, '
          f'中等 2={s["severity_dist"].get(2, 0)}, 较轻 1={s["severity_dist"].get(1, 0)})')
    print(f'  其他标注: todo={s["type_dist"].get("todo", 0)}, '
          f'good={s["type_dist"].get("good", 0)}, '
          f'grade={s["type_dist"].get("grade", 0)}')
    print()
    print(f'  累计扣分: {s["total_deduct"]:.1f} / {s["total_full"]:.1f}')
    print(f'  最终得分: {s["total_score"]:.2f} / 10.00  -> {s["overall"]}')
    print()

    if s['unknown_codes']:
        print(f'  [WARN] 未知 P-X.Y 编号（不在 33 类清单中）: {s["unknown_codes"]}')
        print()

    if report['by_chapter']:
        print('  -- 按章节分布 --')
        for chap, cnt in sorted(report['by_chapter'].items(),
                                key=lambda x: -x[1]):
            bar = '#' * min(cnt, 10) + '.' * (10 - min(cnt, 10))
            print(f'  {chap:>8}  [{bar}]  {cnt}')
        print()

    print('  ── 8 维度评分 ──')
    print(f'  {"维度":<12} {"满分":>6} {"扣分":>6} {"实得":>6}  评级')
    print('  ' + '-' * 50)
    for dim, sc in report['by_dimension'].items():
        print(f'  {dim:<12} {sc["full"]:>6.1f} {sc["deduct"]:>6.2f} '
              f'{sc["score"]:>6.2f}   {sc["letter"]}')
    print('  ' + '-' * 50)
    print(f'  {"总分":<12} {s["total_full"]:>6.1f} {s["total_deduct"]:>6.2f} '
          f'{s["total_score"]:>6.2f}   → {s["overall"]}')
    print()

    if args.chapter is not None:
        target_chap = f'第 {args.chapter} 章'
        print(f'  -- {target_chap} pitfall 详情 --')
        for code, info in report['by_code'].items():
            if target_chap in info['chapter']:
                cat, sub, sev, dd, dim = PITFALL_DB[code]
                print(f'    {code} ({cat} / {sub}) 严重度={sev}  出现 {info["count"]} 次')
                for src, line in info['lines'][:3]:
                    print(f'        {Path(src).name}:{line}')
        print()

    if args.missing:
        print('  -- 未出现的 P-X.Y 编号（建议补充） --')
        appeared = set(report['by_code'].keys())
        missing = sorted(set(PITFALL_DB.keys()) - appeared)
        if missing:
            print(f'  共 {len(missing)} 类未出现:')
            for code in missing:
                cat, sub, sev, dd, dim = PITFALL_DB[code]
                print(f'    {code}  {cat:<8} {sub:<12} 严重度={sev}')
        else:
            print('  OK 33 类已全部覆盖')
        print()

    print('  与 review-checklist.xlsx 配合使用: 在 Excel 中"是否出现"列填"是/否"，')
    print('  对比本报告的 pitfall 列表，未出现的 = 不需要评；出现的 = 需要扣分。')
    print('=' * 70)


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(
        description='自动扫描论文 pitfall 标注并评分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('target', nargs='?', default=None,
                        help='目标文件/目录（默认扫描 thesis/）')
    parser.add_argument('--json', metavar='PATH',
                        help='输出 JSON 报告到指定文件')
    parser.add_argument('--chapter', type=int, metavar='N',
                        help='只显示第 N 章的 pitfall 详情')
    parser.add_argument('--missing', action='store_true',
                        help='列出未出现的 P-X.Y 编号')
    args = parser.parse_args()

    # 1) 收集文件
    if args.target is None:
        # 默认扫描 thesis/ 目录
        target_dir = Path('thesis')
        if not target_dir.exists():
            target_dir = Path('.')
        files = []
        for ext in ('*.tex', '*.md', '*.docx'):
            files.extend(target_dir.rglob(ext))
    else:
        target = Path(args.target)
        if target.is_file():
            files = [target]
        else:
            files = []
            for ext in ('*.tex', '*.md', '*.docx'):
                files.extend(target.rglob(ext))

    if not files:
        print(f'! 未找到文件: {args.target or "thesis/"}', file=sys.stderr)
        sys.exit(1)

    # 2) 提取 pitfall
    findings = []
    for f in files:
        ftype = detect_file_type(f)
        try:
            if ftype == 'latex':
                text = f.read_text(encoding='utf-8', errors='ignore')
                findings.extend(extract_latex(text, str(f)))
            elif ftype == 'markdown':
                text = f.read_text(encoding='utf-8', errors='ignore')
                findings.extend(extract_markdown(text, str(f)))
            elif ftype == 'docx':
                findings.extend(extract_docx(f))
        except Exception as e:
            print(f'  ! 跳过 {f}: {e}', file=sys.stderr)

    # 3) 评分
    report = grade(findings)

    # 4) 输出
    if args.json:
        # 转换 Path 为 str 以便 JSON 序列化
        def _to_jsonable(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, set):
                return sorted(obj)
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

        Path(args.json).write_text(
            json.dumps({'findings': findings, 'report': report},
                       ensure_ascii=False, indent=2,
                       default=_to_jsonable),
            encoding='utf-8'
        )
        print(f'  OK JSON 报告: {args.json}')
    else:
        print_terminal_report(findings, report, args)


if __name__ == '__main__':
    main()
