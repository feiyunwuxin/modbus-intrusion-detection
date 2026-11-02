#!/usr/bin/env python3
"""Parse [SW-DBG2] dumps from a H743 serial capture that may be wrapped across
multiple UART packets (the capture tool prefixes each chunk with
'[HH:MM:SS.mmm] RX (NNN B): ' and breaks the dump mid-line).

Strategy:
  1. Strip every '] RX (NNN B): ' prefix and join the remaining payload into one
     big string (no newlines).
  2. Re-split on the dump markers '[SW-DBG2] w#N prob=...' so each dump is a
     single block; the chXX: lines still carry the spaces the parser needs.
"""
import re
import sys
import pathlib
import numpy as np


def flatten_capture(text: str) -> str:
    """Strip RX prefix and join all lines into one (no newlines)."""
    prefix_re = re.compile(r'\[[\d:.]+\] RX \(\d+ B\): ?')
    cleaned = [prefix_re.sub('', line) for line in text.splitlines()]
    # Also strip literal '..' markers that bracket payloads — they confuse our split
    return '\n'.join(cleaned)


def parse_h743_dumps(text: str):
    """Return dict {win_idx: {'prob': float, 'ch23': np.array (23, 16)}}."""
    text = flatten_capture(text)
    windows = {}
    # Each block: [SW-DBG2] w#N prob=X.XXXX | out23 (23x16, ch-major):
    #   ch00: v0 ... v15
    #   ...
    #   ch22: v0 ... v15
    # [SW-DBG2] w#N end (label=...)
    block_re = re.compile(
        r'\[SW-DBG2\] w#(\d+) prob=([0-9.]+) \| out23 \(23x16, ch-major\):'
        r'(.*?)\[SW-DBG2\] w#\d+ end',
        re.DOTALL,
    )
    for m in block_re.finditer(text):
        win = int(m.group(1))
        prob = float(m.group(2))
        body = m.group(3)
        ch23 = np.zeros((23, 16), dtype=np.float32)
        # Strip newlines, parse 'chNN:' floats
        body_flat = body.replace('\n', ' ').replace('..', ' ')
        ch_re = re.compile(r'ch(\d{2}):\s*((?:-?[0-9.]+\s*)+)')
        for cm in ch_re.finditer(body_flat):
            c = int(cm.group(1))
            if c >= 23:
                continue
            vals = [float(x) for x in cm.group(2).split() if x]
            if len(vals) >= 16:
                ch23[c] = vals[:16]
        windows[win] = {'prob': prob, 'ch23': ch23}
    return windows


def main():
    if len(sys.argv) > 1:
        text = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
    else:
        text = sys.stdin.read()

    h743 = parse_h743_dumps(text)
    if not h743:
        print("No [SW-DBG2] windows found in input")
        return

    Xtr_w = np.load('X_train_binary_v2_scada_window16.npy')
    ytr_w = np.load('y_train_binary_v2_scada_window16.npy')
    keep_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 21, 23, 24, 25, 26]
    Xtr_23 = Xtr_w[..., keep_idx]   # (N, 16, 23)

    import torch
    sys.path.insert(0, '.')
    import train_quantize_3ch as tq
    ckpt = torch.load('model_v4_se_23dim_b64_ch32_do01_window16_s42.pt',
                      map_location='cpu', weights_only=False)
    model = tq.TCNClassifierSE(in_ch=23, channels=32, n_blocks=3, dropout=0.1)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print(f"Loaded {len(h743)} H743 windows: {sorted(h743.keys())}")
    print(f"PC sim Xtr_w: {Xtr_w.shape}, Xtr_23: {Xtr_23.shape}\n")

    print(f"{'Win':>4} {'lbl':>3} {'H743_prob':>9} "
          f"{'H743→model':>11} {'PC→model':>10} {'max_diff':>9} "
          f"{'n>0.05':>7} {'n>0.01':>7} verdict")
    print("-" * 90)
    for win in sorted(h743):
        if win >= Xtr_23.shape[0]:
            print(f"  {win:3d} out-of-range (Xtr has {Xtr_23.shape[0]})")
            continue
        d = h743[win]
        h743_ch23 = d['ch23']
        h743_prob = d['prob']
        label = ytr_w[win]
        pc_ch23 = np.clip(Xtr_23[win].T, -10.0, 10.0)   # (23, 16)
        diff = h743_ch23 - pc_ch23
        max_diff = float(np.abs(diff).max())
        n_gt_05 = int((np.abs(diff) > 0.05).sum())
        n_gt_01 = int((np.abs(diff) > 0.01).sum())

        with torch.no_grad():
            t_h743 = torch.from_numpy(h743_ch23).float().unsqueeze(0)
            t_pc = torch.from_numpy(pc_ch23).float().unsqueeze(0)
            h743_model_prob = torch.sigmoid(model(t_h743)).item()
            pc_model_prob = torch.sigmoid(model(t_pc)).item()

        if max_diff < 0.001:
            verdict = "✅ bit-perfect"
        elif h743_model_prob >= 0.5 and pc_model_prob >= 0.5:
            verdict = "OK both alarmed"
        elif h743_model_prob < 0.5 and pc_model_prob < 0.5:
            verdict = "OK both silent"
        elif max_diff < 0.5:
            verdict = "model-drift (close input)"
        else:
            verdict = "❗ data-flow drift"

        print(f"{win:>4} {label:>3} {h743_prob:>9.4f} "
              f"{h743_model_prob:>11.4f} {pc_model_prob:>10.4f} "
              f"{max_diff:>9.3f} {n_gt_05:>7d} {n_gt_01:>7d} {verdict}")

        # Show the worst channels for inspection
        if max_diff >= 0.05:
            for c in range(23):
                ch_diff = float(np.abs(diff[c]).max())
                if ch_diff > 0.05:
                    t_max = int(np.abs(diff[c]).argmax())
                    print(f"    ch{c:2d} max={ch_diff:.3f} at t={t_max}: "
                          f"H743={h743_ch23[c, t_max]:+.3f} vs PC={pc_ch23[c, t_max]:+.3f}")
        print()


if __name__ == '__main__':
    main()