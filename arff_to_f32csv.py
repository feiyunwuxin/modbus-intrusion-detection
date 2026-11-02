"""
arff_to_f32csv.py -- ARFF → float32 精度 CSV (保留 time 字段小数位)

训练管线 IanArffDataset.csv (csv.DictReader) 是整数秒,丢失 0.1~0.2 sec 精度.
X_train 是从 ARFF (f64 1418682163.170388) 计算的 → feat_mu[13]=1.43 sec.

GUI 之前把整数 sec × 1000 → u32 ms → H743 /1000 = 整数 sec (凑巧对齐整数 CSV).
但训练 fit 的中心是 1.43 sec, integer CSV 上 diff=0 / diff=2 等巨大 → 全误报.

修复: GUI + 训练管线都改读 ARFF 高精度 f32 sec.
本脚本把 ARFF 转成保留小数的 CSV (time 字段保留 6 位小数, 跟 ARFF 一致).

用法: python arff_to_f32csv.py
输出: C:\work\Claude\Issue\IanArffDatasetF32.csv
"""
import csv
import os
import sys

ARFF_PATH = r'C:\work\Claude\Issue\IanArffDataset.arff'
CSV_OUT   = r'C:\work\Claude\Issue\IanArffDatasetF32.csv'


def arff_to_rows(arff_path):
    """解析 ARFF → list of dict."""
    rows = []
    with open(arff_path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.splitlines()

    # 找 @data 位置
    data_start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith('@data'):
            data_start = i + 1
            break
    if data_start is None:
        raise RuntimeError('No @data section found')

    # 找 @attribute 顺序 (ARFF 语法: @attribute 'name with spaces' real)
    # 本数据集所有 attribute 名都带单引号, 可能是 'name'
    attrs = []
    for ln in lines[:data_start]:
        s = ln.strip()
        if not s.lower().startswith('@attribute'):
            continue
        rest = s[len('@attribute'):].strip()
        # 名字可能带引号 'name', 提取引号内
        if rest.startswith("'"):
            end = rest.find("'", 1)
            if end > 0:
                name = rest[1:end]
            else:
                name = rest.split()[0].strip("'")
        else:
            name = rest.split()[0]
        attrs.append(name)
    print(f'Attributes ({len(attrs)}): {attrs}')

    for ln in lines[data_start:]:
        ln = ln.strip()
        if not ln or ln.startswith('%'):
            continue
        parts = ln.split(',')
        if len(parts) != len(attrs):
            print(f'WARN: row has {len(parts)} cols, expected {len(attrs)}: {ln[:80]}...')
            continue
        rows.append({attrs[i]: parts[i] for i in range(len(attrs))})
    return rows, attrs


def main():
    rows, attrs = arff_to_rows(ARFF_PATH)
    print(f'Total rows: {len(rows)}')

    # 时间字段保留 6 位小数 (跟 ARFF f64 一致)
    time_attr = 'time' if 'time' in attrs else None
    if time_attr is None:
        # 试找
        for a in attrs:
            if 'time' in a.lower():
                time_attr = a
                break

    with open(CSV_OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=attrs)
        w.writeheader()
        for row in rows:
            if time_attr is not None:
                t_raw = row.get(time_attr)
                if t_raw not in (None, '', '?'):
                    try:
                        t_f = float(t_raw)
                        row[time_attr] = f'{t_f:.6f}'
                    except ValueError:
                        pass
            w.writerow(row)

    print(f'Saved {len(rows)} rows to {CSV_OUT}')
    # 验证
    with open(CSV_OUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 3:
                print(f'  row {i+1}: time={row.get(time_attr)}, '
                      f'pressure={row.get("pressure measurement")}, '
                      f'function={row.get("function")}')
        # 检查小数
        rows_all = list(reader)
    if time_attr is not None:
        n_with_decimal = sum(1 for r in rows_all
                             if '.' in (r.get(time_attr) or ''))
        print(f'Rows with decimal time: {n_with_decimal}/{len(rows_all)}')


if __name__ == '__main__':
    main()