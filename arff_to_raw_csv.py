#!/usr/bin/env python3
"""
arff_to_raw_csv.py -- Clean ARFF -> CSV converter for IanArffDataset.

Differences vs the existing buggy IanArffDataset.csv:
  1. Missing values `?` are PRESERVED as-is (existing CSV converted them to 0).
  2. Float precision is preserved (existing CSV truncated time to integer).
  3. Output: IanArffDatasetRaw1.csv (next to this script).

Robust to:
  - quoted attribute names: @attribute 'crc rate' real
  - unquoted attribute names: @attribute foo {a,b,c}
  - sparse ARFF: {5 1.0, 7 2.0}  (not used in this dataset but supported)
  - rows with wrong field count (logged + skipped)

Usage:
  python arff_to_raw_csv.py
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC  = HERE / "IanArffDataset.arff"
DST  = HERE / "IanArffDatasetRaw1.csv"


def parse_attribute(line: str) -> str:
    """Extract attribute name from a @attribute line.
    Handles both quoted 'name' and unquoted name forms.
    """
    content = line[len("@attribute"):].strip()
    if content.startswith("'"):
        end = content.find("'", 1)
        if end < 0:
            raise ValueError("Unterminated quoted attribute name: %r" % line)
        return content[1:end]
    # unquoted: name is the first whitespace-delimited token
    return content.split()[0]


def is_sparse_row(line: str) -> bool:
    """Sparse ARFF rows start with '{'. This dataset has none, but guard anyway."""
    return line.lstrip().startswith("{")


def main():
    if not SRC.exists():
        print("[FAIL] ARFF source not found: %s" % SRC)
        sys.exit(1)

    attributes = []
    in_data = False
    data_rows = 0
    skipped = 0

    with SRC.open("r", encoding="utf-8", errors="replace") as fin, \
         DST.open("w", encoding="utf-8", newline="") as fout:

        writer = None

        for lineno, raw in enumerate(fin, 1):
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()

            if not stripped:
                continue
            if stripped.startswith("%"):
                continue

            lower = stripped.lower()

            if lower.startswith("@relation"):
                continue

            if lower.startswith("@attribute"):
                try:
                    name = parse_attribute(stripped)
                except ValueError as e:
                    print("[warn] L%d: %s" % (lineno, e))
                    continue
                attributes.append(name)
                continue

            if lower.startswith("@data"):
                in_data = True
                writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(attributes)
                print(f"[hdr] {len(attributes)} columns: {attributes}")
                continue

            if in_data:
                if is_sparse_row(line):
                    print("[warn] L%d: sparse row skipped (unsupported)" % lineno)
                    skipped += 1
                    continue
                try:
                    row = next(csv.reader([line]))
                except Exception as e:
                    print("[warn] L%d: csv parse error: %s" % (lineno, e))
                    skipped += 1
                    continue
                if len(row) != len(attributes):
                    if skipped < 5:
                        print("[warn] L%d: %d fields (expected %d); skipped"
                              % (lineno, len(row), len(attributes)))
                    skipped += 1
                    continue
                writer.writerow(row)
                data_rows += 1

    print(f"[OK] -> {DST.name}")
    print(f"     attributes : {len(attributes)}")
    print(f"     data rows  : {data_rows}")
    print(f"     skipped    : {skipped}")
    print(f"     output size: {DST.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()