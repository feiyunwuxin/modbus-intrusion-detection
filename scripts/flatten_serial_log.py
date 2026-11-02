#!/usr/bin/env python3
"""Flatten the serial capture log: strip [HH:MM:SS.mmm] RX (N B): prefix,
merge continuation lines, write a clean log to stdout for parity_compare_dump.py.

The H743 [SW-DBG2] dump is printed across multiple UART packets (~1152B each),
which the serial capture prefixes with a `] RX (NNN B): ` header. The dump itself
contains no newlines, so we treat the raw bytes between consecutive `..` markers
as a single line for the parity parser.
"""
import re
import sys
import pathlib

INFILE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None

if INFILE is None:
    print("Usage: python flatten_serial_log.py <h743_log_file>", file=sys.stderr)
    sys.exit(1)

text = INFILE.read_text(encoding='utf-8', errors='replace')

# Step 1: split on RX markers, keep just the payload that follows.
# Each "RX (NNN B):" line ends with ": " and the next RX marker starts a new chunk.
# Simpler: strip the leading "[HH:MM:SS.mmm] RX (NNN B): " prefix from each line.
prefix_re = re.compile(r'^\[[\d:.]+\] RX \(\d+ B\): ?')

out_lines = []
for line in text.splitlines():
    line = prefix_re.sub('', line)
    out_lines.append(line)

flat = '\n'.join(out_lines)
sys.stdout.write(flat)