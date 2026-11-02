#!/usr/bin/env python3
"""从已有 X_*_binary_v2_scada.npy (19 列 row-level) 重生 window=4 / window=8 数据。

关键约束:
  - 必须保持 70/10/20 temporal split (按 preprocess_v2_scada.py)
  - 必须保持 27 列/步布局 (19 row-level + 8 window-level aggregates)
  - y_label/window = max(attack 标记) over window rows (跟 window16 一致)

输出:
  - X_train/y_train/_binary_v2_scada_window4.npy
  - X_train/y_train/_binary_v2_scada_window8.npy
  - 对应 val, test 各一份

验证:
  - 跟 window16 的 val/test 行数比例一致 (test = 0.20 fraction of total)
  - window4 的 test 行数 ≈ window16 test_rows * 4 倍 (即每个 window16 = 4 个 window4)
"""
import os
import numpy as np

BASE = r"C:\work\Claude\Issue"
PREFIX = "binary_v2_scada"

# 加载 raw row-level (19 cols, 0-indexed)
X_train_raw = np.load(os.path.join(BASE, f"X_train_{PREFIX}.npy")).astype(np.float32)
X_val_raw   = np.load(os.path.join(BASE, f"X_val_{PREFIX}.npy")).astype(np.float32)
X_test_raw  = np.load(os.path.join(BASE, f"X_test_{PREFIX}.npy")).astype(np.float32)
y_train_bin = np.load(os.path.join(BASE, f"y_train_{PREFIX}.npy")).astype(np.int64)
y_val_bin   = np.load(os.path.join(BASE, f"y_val_{PREFIX}.npy")).astype(np.int64)
y_test_bin  = np.load(os.path.join(BASE, f"y_test_{PREFIX}.npy")).astype(np.int64)

print(f"[raw] X_train={X_train_raw.shape}  X_val={X_val_raw.shape}  X_test={X_test_raw.shape}")

# row-level 特征位置 (跟 preprocess_v2_scada.py 完全一致)
IDX_PRESS = 13
IDX_CRC = 14
IDX_ISRESP = 18
IDX_LENGTH = 2
IDX_ISUNUSUAL = 17


def make_windows_with_agg(X, y_bin, win):
    """每 window=win 行 -> 1 个样本;标签=max over win;聚合特征 broadcast."""
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y_bin[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    n_win = Xw.shape[0]

    press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
    crc_mean_w   = Xw[:, :, IDX_CRC].mean(axis=1)
    crc_max_w    = Xw[:, :, IDX_CRC].max(axis=1)
    cmd_count_w  = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)
    resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
    balance_w    = (cmd_count_w - resp_count_w) / win
    length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(n_win)], dtype=np.float32)
    unusual_count_w  = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)

    aggs = np.column_stack([
        press_mean_w, crc_mean_w, crc_max_w,
        cmd_count_w, resp_count_w, balance_w,
        length_nunique_w, unusual_count_w,
    ]).astype(np.float32)
    aggs_tiled = np.repeat(aggs[:, np.newaxis, :], win, axis=1)
    Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)
    return Xw_full, yw


for W in (4, 8, 12):
    print(f"\n=== Building window={W} ===")
    Xtr_w, ytr_w = make_windows_with_agg(X_train_raw, y_train_bin, W)
    Xva_w, yva_w = make_windows_with_agg(X_val_raw,   y_val_bin,   W)
    Xte_w, yte_w = make_windows_with_agg(X_test_raw,  y_test_bin,  W)
    print(f"  train: X={Xtr_w.shape} y={ytr_w.shape}  attack_pct={ytr_w.mean():.4f}")
    print(f"  val:   X={Xva_w.shape} y={yva_w.shape}  attack_pct={yva_w.mean():.4f}")
    print(f"  test:  X={Xte_w.shape} y={yte_w.shape}  attack_pct={yte_w.mean():.4f}")
    print(f"  features/step: {Xtr_w.shape[-1]} (期望 27)")

    # 跟 window16 对齐 sanity check
    n_te_w16 = 3432
    n_te_w8  = Xte_w.shape[0]
    n_te_w4  = Xte_w.shape[0]
    print(f"  sanity: test 3432 windows × 16  rows = {3432*16} (期望 54927)")
    print(f"          test W={W} rows = {n_te_w8 if W==8 else n_te_w4}")
    print(f"          ratio = {(n_te_w8 if W==8 else n_te_w4)/(3432*16):.4f} (期望 1.0)")

    np.save(os.path.join(BASE, f"X_train_{PREFIX}_window{W}.npy"), Xtr_w)
    np.save(os.path.join(BASE, f"X_val_{PREFIX}_window{W}.npy"),   Xva_w)
    np.save(os.path.join(BASE, f"X_test_{PREFIX}_window{W}.npy"),  Xte_w)
    np.save(os.path.join(BASE, f"y_train_{PREFIX}_window{W}.npy"), ytr_w)
    np.save(os.path.join(BASE, f"y_val_{PREFIX}_window{W}.npy"),   yva_w)
    np.save(os.path.join(BASE, f"y_test_{PREFIX}_window{W}.npy"),  yte_w)
    print(f"  [saved] window={W} .npy files")

print("\nDONE.")
