"""V10: Inspect official test windows to understand correct feature layout."""
import numpy as np

X_test = np.load('X_test_binary_v2_scada_window16.npy')
print(f'X_test shape: {X_test.shape}')
print(f'X_test dtype: {X_test.dtype}')
print(f'Min/Max/Mean per dim (27 dims):')
for i in range(27):
    col = X_test[:, :, i].flatten()
    print(f'  [{i:>2}] min={col.min():>8.4f} max={col.max():>8.4f} mean={col.mean():>8.4f} std={col.std():>8.4f}')

# Sample first window
print()
print(f'First window (rows 1-16), all 27 dims:')
w = X_test[0]
for t in range(16):
    vals = ' '.join(f'{w[t, c]:>8.3f}' for c in range(27))
    print(f'  t={t:>2}: {vals}')