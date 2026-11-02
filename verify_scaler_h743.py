#!/usr/bin/env python3
"""P0 验证: H743 scaler_export.h 的 18-dim RobustScaler 是否与 PC 端 joblib 完全一致.

用法: python verify_scaler_h743.py
退出码: 0 = 全 bit-exact, 1 = 有偏差 (打印 MSE)
"""
from __future__ import annotations
import sys, joblib, re
import numpy as np

PC_PATH = 'scaler_binary_v2_scada.joblib'
H_PATH  = 'h743/Core/Inc/scaler_export.h'

# H743 KEEP_18: 18-dim idx -> 19-dim scaler idx (skip scaler[3]=setpoint)
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


def parse_array(path: str, name: str) -> np.ndarray:
    """Parse 'static const float NAME[N] = { a, b, c, ... };' from C header."""
    with open(path, encoding='utf-8', errors='replace') as f:
        src = f.read()
    m = re.search(rf'static const float {name}\[[A-Za-z0-9_]+\]\s*=\s*\{{([^}}]+)\}}',
                  src, re.S)
    if m is None:
        raise RuntimeError(f'{name} not found in {path}')
    nums = re.findall(r'(-?\d+\.\d+)f?', m.group(1))
    return np.array([float(x) for x in nums], dtype=np.float32)


def main() -> int:
    scaler = joblib.load(PC_PATH)
    pc_mu = scaler.center_.astype(np.float32)
    pc_sg = scaler.scale_.astype(np.float32)

    h743_mu = parse_array(H_PATH, 'feat_mu_18')
    h743_sg = parse_array(H_PATH, 'feat_sigma_18')

    print(f'PC scaler type:      {type(scaler).__name__}')
    print(f'PC center_  shape:   {pc_mu.shape}')
    print(f'PC scale_   shape:   {pc_sg.shape}')
    print(f'H743 feat_mu_18:     {h743_mu.shape}')
    print(f'H743 feat_sigma_18:  {h743_sg.shape}')
    print()

    # 期望值: PC 19-dim → KEEP_18 → 应 == H743 18-dim
    expect_mu = pc_mu[KEEP_18]
    expect_sg = pc_sg[KEEP_18]

    diff_mu = np.abs(h743_mu - expect_mu)
    diff_sg = np.abs(h743_sg - expect_sg)

    print(f'mu max abs diff:    {diff_mu.max():.6e}')
    print(f'mu mean abs diff:   {diff_mu.mean():.6e}')
    print(f'sigma max abs diff: {diff_sg.max():.6e}')
    print(f'sigma mean abs diff:{diff_sg.mean():.6e}')
    print()

    # Per-feature diff
    print('per-feature diff:')
    print(f'{"i":>2} {"PC_mu":>14} {"H743_mu":>14} {"diff":>14} {"PC_sg":>14} {"H743_sg":>14} {"diff":>14}')
    for i in range(18):
        print(f'{i:2d} {expect_mu[i]:>14.6f} {h743_mu[i]:>14.6f} {diff_mu[i]:>14.6e} '
              f'{expect_sg[i]:>14.6f} {h743_sg[i]:>14.6f} {diff_sg[i]:>14.6e}')

    bit_exact_mu = bool(np.array_equal(h743_mu, expect_mu))
    bit_exact_sg = bool(np.array_equal(h743_sg, expect_sg))
    print()
    print(f'bit-exact mu:    {bit_exact_mu}')
    print(f'bit-exact sigma: {bit_exact_sg}')

    return 0 if (bit_exact_mu and bit_exact_sg) else 1


if __name__ == '__main__':
    sys.exit(main())