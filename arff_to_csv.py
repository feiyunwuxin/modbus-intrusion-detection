#!/usr/bin/env python3
"""Convert IanArffDataset.arff to CSV format."""

import csv
import sys
from pathlib import Path

SRC = Path(r"C:\work\Claude\issue\IanArffDataset.arff")
DST = Path(r"C:\work\Claude\issue\IanArffDataset.csv")

attributes = []          # 提取出的属性名
header = None             # 第一个数据行作为参考
in_data = False
data_rows = 0

with SRC.open("r", encoding="utf-8", errors="replace") as fin, \
     DST.open("w", encoding="utf-8", newline="") as fout:

    writer = None

    for raw in fin:
        line = raw.rstrip("\n").rstrip("\r")
        stripped = line.strip()

        # 跳过空行
        if not stripped:
            continue

        # 跳过注释
        if stripped.startswith("%"):
            continue

        # @relation
        if stripped.lower().startswith("@relation"):
            continue

        # @attribute
        if stripped.lower().startswith("@attribute"):
            # 形如: @attribute 'name' real    或   @attribute 'name' {a,b,c}
            # 提取单引号或空格分隔的属性名
            content = stripped[len("@attribute"):].strip()
            if content.startswith("'"):
                end = content.find("'", 1)
                name = content[1:end]
            else:
                name = content.split()[0]
            attributes.append(name)
            continue

        # @data
        if stripped.lower().startswith("@data"):
            in_data = True
            # 写入表头
            writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(attributes)
            continue

        # 数据行
        if in_data:
            # 保持与原文件一致：csv 模块处理引号内的逗号（虽然此文件没有引号字段）
            row = next(csv.reader([line]))
            # 仅保留与属性数等长的字段
            if len(row) == len(attributes):
                writer.writerow(row)
                data_rows += 1
            else:
                print(f"[warn] line {data_rows+1} has {len(row)} fields, expected {len(attributes)}; skipped",
                      file=sys.stderr)

print(f"[OK] converted -> {DST}")
print(f"     attributes: {len(attributes)}")
print(f"     data rows : {data_rows}")
print(f"     output    : {DST.stat().st_size:,} bytes")
