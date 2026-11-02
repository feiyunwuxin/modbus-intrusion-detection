"""Generate mock [SW-DBG2] dump from Xtr_w (PC sim) to verify parity script.

Simulates what H743 would dump IF its ch32_input matched Xtr_w bit-perfect.
This proves the script parses correctly + computes the right diff (0 diff expected).
"""
import numpy as np

Xtr_w = np.load('X_train_binary_v2_scada_window16.npy')
keep_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
Xtr_23 = Xtr_w[..., keep_idx]  # (N, 16, 23)

# Mock probs from test_fp32_replay_quick2.py results
mock_probs = {2: 0.78, 3: 1.00, 7: 1.00, 15: 0.91, 51: 0.63, 0: 0.05, 1: 0.05, 8: 0.95}
labels = {0: 0, 1: 0, 2: 1, 3: 1, 7: 1, 8: 1, 15: 1, 51: 1}

with open('logs/h743_swdbg2_mock.txt', 'wb') as f:
    for win in [0, 1, 2, 3, 7, 8, 15, 51]:
        prob = mock_probs.get(win, 0.5)
        label = labels.get(win, 0)
        f.write(f"[SW-DBG2] w#{win} prob={prob:.4f} | out23 (23x16, ch-major):\n".encode())
        ch23 = np.clip(Xtr_23[win].T, -10.0, 10.0)  # (23, 16)
        for c in range(23):
            vals = ' '.join(f'{ch23[c, t]:.4f}' for t in range(16))
            f.write(f"  ch{c:02d}: {vals}\n".encode())
        f.write(f"[SW-DBG2] w#{win} end (FN={label}, expected PC sim prob~{prob:.2f})\n".encode())

print(f"Generated logs/h743_swdbg2_mock.txt with {len(mock_probs)} mock windows")
print("Run: python parity_compare_dump.py logs/h743_swdbg2_mock.txt")
