"""Compare H743 [SW-DBG2] dump with PC sim Xtr_w.

Reads H743 dump from logs/test_*.csv (or stdin) and compares each window's
23-dim ch32_input with PC sim Xtr_w[win, :, keep_idx] bit-perfect.

Usage:
    python parity_compare_dump.py <log_file>

Or:
    cat h743_dump.txt | python parity_compare_dump.py
"""
import re
import sys
import numpy as np


def parse_h743_dump(text):
    """Parse [SW-DBG2] dump from H743 log.

    Format:
        [SW-DBG2] w#N prob=X.XXXX | out23 (23x16, ch-major):
          ch00: v0 v1 ... v15
          ch01: v0 v1 ... v15
          ...
          ch22: v0 v1 ... v15
        [SW-DBG2] w#N end (label=..., expected PC sim prob~X.XX)

    Returns: dict {win_idx: {'prob': float, 'ch23': np.array (23, 16)}}
    """
    windows = {}
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'\[SW-DBG2\] w#(\d+) prob=([0-9.]+) \| out23', line)
        if m:
            win = int(m.group(1))
            prob = float(m.group(2))
            ch23 = np.zeros((23, 16), dtype=np.float32)
            i += 1
            for c in range(23):
                if i >= len(lines):
                    break
                # Format: "  ch00: v0 v1 ... v15"
                parts = lines[i].strip().split(':')
                if len(parts) == 2:
                    vals = parts[1].strip().split()
                    if len(vals) == 16:
                        ch23[c] = [float(v) for v in vals]
                i += 1
            windows[win] = {'prob': prob, 'ch23': ch23}
        else:
            i += 1
    return windows


def main():
    # Read H743 dump
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    h743_windows = parse_h743_dump(text)
    if not h743_windows:
        print("No [SW-DBG2] windows found in input")
        print("Usage: python parity_compare_dump.py <h743_log_file>")
        print("       cat h743_dump.txt | python parity_compare_dump.py")
        return

    # Load PC sim Xtr_w
    Xtr_w = np.load('X_train_binary_v2_scada_window16.npy')  # (N, 16, 27)
    ytr_w = np.load('y_train_binary_v2_scada_window16.npy')

    # Keep 23 (drop {2, 3, 20, 22})
    keep_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
    Xtr_23 = Xtr_w[..., keep_idx]  # (N, 16, 23)
    # H743 layout: (ch=23, t=16) row-major, i.e., out[c*16 + t]
    # PC sim Xtr_23 layout: (N, 16, 23) i.e., [t][c]
    # So to compare, we need Xtr_23[win].T to get (23, 16)

    print(f"Loaded {len(h743_windows)} H743 windows: {sorted(h743_windows.keys())}")
    print(f"PC sim Xtr_w: {Xtr_w.shape}, Xtr_23: {Xtr_23.shape}")
    print()

    for win, h743_data in sorted(h743_windows.items()):
        if win >= Xtr_23.shape[0]:
            print(f"W={win}: out of range (Xtr has {Xtr_23.shape[0]} windows)")
            continue

        h743_ch23 = h743_data['ch23']  # (23, 16)
        h743_prob = h743_data['prob']
        label = ytr_w[win]

        # PC sim reference: Xtr_23[win] is (16, 23), transpose to (23, 16)
        # Apply clip ±10 (H743 does this in sliding_window_get)
        pc_ch23 = np.clip(Xtr_23[win].T, -10.0, 10.0)  # (23, 16)

        diff = h743_ch23 - pc_ch23
        max_diff = np.abs(diff).max()
        n_diff_gt_01 = (np.abs(diff) > 0.1).sum()
        n_diff_gt_001 = (np.abs(diff) > 0.01).sum()

        # Run PC sim model on the dump
        import torch
        sys.path.insert(0, '.')
        import train_quantize_3ch as tq
        ckpt = torch.load('model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                          map_location='cpu', weights_only=False)
        model = tq.TCNClassifierSE(in_ch=23, channels=32, n_blocks=3, dropout=0.1)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        # H743 input → (B=1, C=23, T=16) for model
        with torch.no_grad():
            h743_input = torch.from_numpy(h743_ch23).float().unsqueeze(0)  # (1, 23, 16)
            h743_model_prob = torch.sigmoid(model(h743_input)).item()

            pc_input = torch.from_numpy(pc_ch23).float().unsqueeze(0)
            pc_model_prob = torch.sigmoid(model(pc_input)).item()

        # Identify channels with biggest diff
        big_diffs = []
        for c in range(23):
            ch_diff = np.abs(diff[c]).max()
            if ch_diff > 0.01:
                big_diffs.append((c, ch_diff, h743_ch23[c, 0], pc_ch23[c, 0]))

        print(f"W={win:3d} label={label} H743_prob={h743_prob:.4f} | "
              f"H743_input→model={h743_model_prob:.4f} | "
              f"PC_input→model={pc_model_prob:.4f} | "
              f"max_diff={max_diff:.4f} (n>{0.1}={n_diff_gt_01}, n>{0.01}={n_diff_gt_001})")

        if big_diffs:
            for c, ch_diff, h_v, p_v in big_diffs[:5]:
                # Find which t has the biggest diff
                t_diff = np.abs(diff[c])
                t_max = t_diff.argmax()
                print(f"    ch{c:2d} max_diff={ch_diff:.3f} at t={t_max}: "
                      f"H743={h743_ch23[c, t_max]:.3f} vs PC={pc_ch23[c, t_max]:.3f}")
        print()


if __name__ == '__main__':
    main()
