#!/usr/bin/env python3
"""Capture H743 serial output to a file with no buffer limit.

Usage:
    python capture_serial.py COM_PORT=COM6 BAUD=115200 OUT=C:/tmp/h743/capture.log [SECONDS=20]

Reads COM port byte-by-byte, writes to OUT file with optional timestamp prefix,
flushes every chunk so a hard kill won't lose data.
"""
import sys
import time
import serial


def main():
    args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    port = args.get('COM_PORT', 'COM6')
    baud = int(args.get('BAUD', '115200'))
    out = args.get('OUT', 'C:/tmp/h743/capture.log')
    seconds = float(args.get('SECONDS', '0'))  # 0 = until Ctrl-C

    print(f"[cap] opening {port} @ {baud}  →  {out}")
    s = serial.Serial(port, baud, timeout=0.1, rtscts=False, xonxoff=False)
    s.reset_input_buffer()

    end_at = time.time() + seconds if seconds > 0 else None
    n_bytes = 0
    n_lines = 0
    cur_line = b''
    with open(out, 'wb') as f:
        try:
            while True:
                if end_at and time.time() > end_at:
                    print(f"[cap] {seconds}s reached, stopping")
                    break
                chunk = s.read(256)
                if not chunk:
                    continue
                f.write(chunk)
                f.flush()
                n_bytes += len(chunk)
                cur_line += chunk
                while b'\n' in cur_line:
                    line, cur_line = cur_line.split(b'\n', 1)
                    n_lines += 1
                if n_lines % 50 == 0 and n_lines > 0:
                    print(f"[cap] {n_lines} lines, {n_bytes} bytes")
        except KeyboardInterrupt:
            print(f"\n[cap] Ctrl-C, stopping")

    s.close()
    print(f"[cap] wrote {n_bytes} bytes / {n_lines} lines to {out}")


if __name__ == '__main__':
    main()