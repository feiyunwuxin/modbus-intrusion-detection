#!/usr/bin/env python3
"""Parse [RAW18-DBG] SITE 4 dumps from a H743 serial capture.

Each SITE 4 dump emits two lines for the first frame of a suspect window:
  [RAW18-DBG w#N frame#F] raw18: r0 r1 ... r17
  [RAW18-DBG w#N]           out18: o0 o1 ... o17

We compare H743 raw18 with PC ARFF values at the same window's first frame,
and H743 out18 with the expected `(raw - mu) / sigma` (with clip +/- 10).

mu/sigma are hardcoded from MCU/Core/Inc/scaler_export.h (single source of
truth — keep in sync if you retrain).
"""
import re
import sys
import pathlib
import numpy as np


# From MCU/Core/Inc/scaler_export.h feat_mu_18[18] / feat_sigma_18[18].
MU_18 = np.array([
    4.0,        # [ 0] address
    3.0,        # [ 1] function
    16.0,       # [ 2] length
    0.0,        # [ 3] gain
    0.0,        # [ 4] reset rate
    0.0,        # [ 5] deadband
    0.0,        # [ 6] cycle time
    0.0,        # [ 7] rate
    0.0,        # [ 8] system mode
    0.0,        # [ 9] control scheme
    0.0,        # [10] pump
    0.0,        # [11] solenoid
    0.0,        # [12] pressure measurement (already clipped [0,100])
    14135.0,    # [13] crc rate
    1.426832,   # [14] time_diff
    3.116917,   # [15] time_since_last_same_addr_func
    0.0,        # [16] is_unusual_fc
    1.0,        # [17] is_response
], dtype=np.float64)

SIGMA_18 = np.array([
    1.0,        # [ 0]
    13.0,       # [ 1]
    30.0,       # [ 2]
    1.0,        # [ 3]
    1.0,        # [ 4]
    1.0,        # [ 5]
    1.0,        # [ 6]
    1.0,        # [ 7]
    1.0,        # [ 8]
    1.0,        # [ 9]
    1.0,        # [10]
    1.0,        # [11]
    1.0,        # [12]
    4849.0,     # [13]
    1.521081,   # [14]
    3.241358,   # [15]
    1.0,        # [16]
    1.0,        # [17]
], dtype=np.float64)


def flatten_capture(text: str) -> str:
    prefix_re = re.compile(r'\[[\d:.]+\] RX \(\d+ B\): ?')
    cleaned = [prefix_re.sub('', line) for line in text.splitlines()]
    # Strip the '..' block separators so the float regex doesn't capture them.
    joined = '\n'.join(cleaned)
    return joined.replace('..', ' ')


def parse_raw18(text: str):
    text = flatten_capture(text)
    out = {}
    raw_re = re.compile(
        r'\[RAW18-DBG w#(\d+) frame#(\d+)\] raw18:\s*'
        r'((?:-?[0-9.]+\s*)+?)\s*'
        r'\[RAW18-DBG w#\d+\]\s*out18:\s*'
        r'((?:-?[0-9.]+\s*)+?)\s*(?=\[|$)',
        re.MULTILINE,
    )
    for m in raw_re.finditer(text):
        win = int(m.group(1))
        frame = int(m.group(2))
        raw = [float(x) for x in m.group(3).split()]
        out18 = [float(x) for x in m.group(4).split()]
        out[win] = {'frame': frame, 'raw18': np.asarray(raw, dtype=np.float64),
                    'out18': np.asarray(out18, dtype=np.float64)}
    return out


def main():
    if len(sys.argv) > 1:
        text = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
    else:
        text = sys.stdin.read()

    dumps = parse_raw18(text)
    if not dumps:
        print("No [RAW18-DBG] dumps found")
        return

    # Load PC reference: Xtr_w is (N, 16, 27). First frame's 27-dim post-clip raw.
    Xtr_w = np.load('X_train_binary_v2_scada_window16.npy')
    ytr_w = np.load('y_train_binary_v2_scada_window16.npy')

    # scaler layout (27-dim): [ 0..2] address/func/length; [3] setpoint (skipped);
    # [4..18] = our sw_keep_18 indices (gain, reset, deadband, cycle, rate,
    #           system_mode, control, pump, solenoid, pressure, crc_rate,
    #           time_diff, time_since, is_unusual, is_response)
    sw_keep_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    print(f"Loaded {len(dumps)} SITE-4 dumps: {sorted(dumps.keys())}\n")

    for win in sorted(dumps):
        if win >= Xtr_w.shape[0]:
            print(f"W={win}: out-of-range (Xtr has {Xtr_w.shape[0]} windows)")
            continue
        d = dumps[win]
        h743_raw = d['raw18']
        h743_out = d['out18']

        # PC reference: first frame's 27-dim, take 18-dim via sw_keep_18.
        # IMPORTANT: PC Xtr_w is ALREADY normalized (scaler space), so the
        # reference for "out18" is Xtr_w itself (no mu/sigma re-apply).
        pc_norm_27 = Xtr_w[win, 0]
        pc_norm_18 = pc_norm_27[sw_keep_18].astype(np.float64)
        pc_norm_18 = np.clip(pc_norm_18, -10.0, 10.0)

        # Reconstruct what H743 SHOULD have in raw space (back from normalized).
        pc_raw_18 = pc_norm_18 * SIGMA_18 + MU_18

        # Bit-perfect (drift vs PC sim) check: H743_out (normalized) ↔ PC normalized.
        out_diff = h743_out - pc_norm_18
        max_out_d = float(np.abs(out_diff).max())

        # H743_raw vs back-reconstructed PC raw. Sanity check on scaler_export.
        raw_diff = h743_raw - pc_raw_18
        max_raw_d = float(np.abs(raw_diff).max())

        # Internal sanity: H743 raw → normalize should match H743 out.
        expected_out = (h743_raw - MU_18) / SIGMA_18
        expected_out = np.clip(expected_out, -10.0, 10.0)
        h743_internal = float(np.abs(expected_out - h743_out).max())

        print(f"=== W={win} frame={d['frame']} lbl={ytr_w[win]} ===")
        print(f"  max_Δout_norm (H743_out vs PC_norm) = {max_out_d:8.4f}  "
              f"[should be ~0 if data source same]")
        print(f"  max_Δraw (H743_raw vs PC_raw_back) = {max_raw_d:8.3f}  "
              f"[big ok if PC source semantically different]")
        print(f"  H743 internal (raw→normalize vs out) = {h743_internal:8.4f}  "
              f"[should be ~0 — sanity]")
        print(f"  {'ch':>3} {'H743_raw':>10} {'PC_raw*':>10} {'Δraw':>9}  "
              f"{'H743_out':>10} {'PC_norm':>10} {'Δout':>9}  note")
        for i in range(18):
            r_d = float(raw_diff[i])
            o_d = float(out_diff[i])
            note = ""
            if abs(o_d) > 0.05:
                note = "⚠ NORM drift (data source differs)"
            elif abs(r_d) > 0.5 * abs(SIGMA_18[i]) + 0.05:
                note = "  raw differ (expected)"
            print(f"  {i:>3} {h743_raw[i]:>10.3f} {pc_raw_18[i]:>10.3f} {r_d:>+9.3f}  "
                  f"{h743_out[i]:>10.3f} {pc_norm_18[i]:>10.3f} {o_d:>+9.3f}  {note}")
        print()


if __name__ == '__main__':
    main()