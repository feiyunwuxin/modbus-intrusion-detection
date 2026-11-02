#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC 端 23-dim TCN+SE 对齐 H743 实测 — 验证 OOD 根因.

3 个变体:
  A: 全 CSV 原 float (训练分布, in-dist)         → 预期 F1 ≥ 0.85 (复现 PC ONNX 仿真的 100/98)
  B: PC 当前 uint8 截断编码 (H743 实际收到的字节)  → 预期 F1 ≤ 0.33 (复现 H743 实测)
  C: A 的基础上把 f[2..5] 强制设成 uint8 截断值    → 单独验证这 4 个特征是否是嫌疑 #1

输入: IanArffDataset.csv (274k 行)
模型: model_v4_se_23dim_ch32_s{42,123,456,789,1024}.onnx (5-seed ensemble)
Scaler: scaler_binary_v2_scada.joblib (训练时 fit, fit on first 70%)
窗口: 16 (与训练一致)
KEEP_23 (部署用): [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
   去 {2=length, 3=setpoint, 20=crc_mean_w, 22=cmd_count_w}

输出: sim_pc_align_*.csv  (frame_idx | binary | prob_5seed | variant | ...)

用法:
  python sim_pc_align_h743_23dim.py [--limit N]
"""
import os, sys, time, csv, argparse
import numpy as np
import onnxruntime as ort
import joblib

BASE = r"C:\work\Claude\Issue"
CSV_PATH = os.path.join(BASE, "IanArffDataset.csv")
SCALER_PATH = os.path.join(BASE, "scaler_binary_v2_scada.joblib")
ONNX_TPL = "model_v4_se_23dim_ch32_s{seed}.onnx"
SEEDS = [42, 123, 456, 789, 1024]

WINDOW = 16
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
# 在 27 维原始列里的索引:
# 0=address, 1=function, 2=length, 3=setpoint, 4=gain, 5=reset rate,
# 6=deadband, 7=cycle time, 8=rate, 9=system mode, 10=control scheme,
# 11=pump, 12=solenoid, 13=pressure measurement, 14=crc rate,
# 15=time_diff, 16=time_since_last_same_addr_func, 17=is_unusual_fc,
# 18=is_response, 19=press_mean_w, 20=crc_mean_w, 21=crc_max_w,
# 22=cmd_count_w, 23=resp_count_w, 24=cmd_resp_balance_w,
# 25=length_nunique_w, 26=unusual_count_w

# rtu_replay_gui.py:213-214 截断规则:
#   for v in feat_vals[:10]:
#       body.append(max(0, min(255, int(round(v)))))
# feat_vals[:10] = setpoint, gain, reset rate, deadband, cycle time, rate,
#                  system mode, control scheme, pump, solenoid
# PC 端 setpoint (idx 3) 不被 H743 接收, 但 PC 还是发了 → 字节位错位吗?
# H743 端 RAW17_FEAT_COLS (rtu_replay_gui.py:165) 是 11 个不含 setpoint:
#   gain, reset rate, deadband, cycle time, rate, system mode,
#   control scheme, pump, solenoid, pressure measurement, crc rate
# 等等——是 11 个还是 12 个? 让我重数 RAW17_FEAT_COLS.
RAW17_FEAT_COLS_PC = [
    'setpoint', 'gain', 'reset rate', 'deadband',
    'cycle time', 'rate', 'system mode', 'control scheme',
    'pump', 'solenoid', 'pressure measurement'
]
# 这是 PC 端 build_raw17_frame 发的 11 个 float (length=46 时) 或 10+1 float (length=16 时)
# H743 接收: modbus_ids.c:68-77 从 g_last_raw17 提取的字段是 10 个 (无 setpoint):
H743_RECV_FIELDS = [
    'gain', 'reset_rate', 'deadband', 'cycle_time', 'rate',
    'system_mode', 'control_scheme', 'pump', 'solenoid',
    'pressure_measurement'
]
# → PC 发 11 个, H743 只读 10 个, setpoint 字节流被丢弃

# rtu_replay_gui.py:213-214 用 uint8 编码 (length=16 格式) 时的截断
def pc_uint8_truncation(v):
    """模拟 PC 端 length=16 格式的 uint8 截断."""
    return max(0, min(255, int(round(v))))


def log(msg, t0=None):
    if t0 is None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    else:
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def load_csv(limit=None):
    """Load IanArffDataset.csv → 按 time 排序后取前 limit 行.
       limit=None 时全量加载后再排序, 排序后取全量 (跟 preprocess_v2_scada.py:68 一致).
       返回 rows 列表 (元组), 元素顺序见下方."""
    log("=== STEP 1: load CSV ===")
    t0 = time.time()
    rows = []
    n_skip = 0
    n_total = 0
    with open(CSV_PATH, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            n_total += 1
            try:
                v0 = int(row['address']) if row['address'] != '?' else 0
                v1 = int(row['function']) if row['function'] != '?' else 0
                v2 = int(row['length']) if row['length'] != '?' else 0
                raw_csv_cols = ['setpoint','gain','reset rate','deadband',
                                'cycle time','rate','system mode','control scheme',
                                'pump','solenoid']
                vals_3_12 = []
                for c in raw_csv_cols:
                    try:
                        vals_3_12.append(float(row[c]) if row[c] != '?' else 0.0)
                    except ValueError:
                        vals_3_12.append(0.0)
                try: v13 = float(row['pressure measurement'])
                except: v13 = 0.0
                try: v14 = float(row['crc rate'])
                except: v14 = 0.0
                fc = int(row['function']) if row['function'] != '?' else 0
                UNUSUAL = {136, 171, 139, 133, 137, 138, 140}
                v17 = 1.0 if fc in UNUSUAL else 0.0
                try: v18 = float(row['command response'])
                except: v18 = 0.0
                v_agg = [0.0] * 8
                try: y_bin = int(row['binary result'])
                except: y_bin = -1
                try: t_ms = int(row['time'])
                except: t_ms = 0
                rows.append((v0, v1, v2,
                             *vals_3_12, v13, v14, 0.0, 0.0, v17, v18,
                             *v_agg, y_bin, t_ms))
            except Exception as e:
                n_skip += 1
                if n_skip <= 3:
                    log(f"  SKIP #{n_skip}: {e}", t0)
                continue
    log(f"  total CSV rows={n_total}, parsed={len(rows)}, skipped={n_skip}", t0)
    # 按 time 排序 (跟 preprocess_v2_scada.py:68 一致)
    rows.sort(key=lambda r: r[28])
    if limit is not None:
        rows = rows[:limit]
    n_att = sum(1 for r in rows if r[27] == 1)
    log(f"  after sort+limit: {len(rows)} rows, attack={n_att}", t0)
    return rows


def build_row_level_features(rows):
    """从 rows 计算 time_diff (idx 15), time_since_same (idx 16).
       返回 N×19 numpy 数组 (前 19 列, 即 window agg 之前的 row-level)."""
    n = len(rows)
    # 19 维: 0..18 (不含 agg)
    X = np.zeros((n, 19), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    times = np.zeros(n, dtype=np.int64)
    last_seen = {}  # addr_func -> time
    for i, r in enumerate(rows):
        X[i, 0] = r[0]   # address
        X[i, 1] = r[1]   # function
        X[i, 2] = r[2]   # length
        X[i, 3] = r[3]   # setpoint
        X[i, 4] = r[4]   # gain
        X[i, 5] = r[5]   # reset rate
        X[i, 6] = r[6]   # deadband
        X[i, 7] = r[7]   # cycle time
        X[i, 8] = r[8]   # rate
        X[i, 9] = r[9]   # system mode
        X[i, 10] = r[10] # control scheme
        X[i, 11] = r[11] # pump
        X[i, 12] = r[12] # solenoid
        X[i, 13] = r[13] # pressure measurement
        X[i, 14] = r[14] # crc rate
        # time_diff (15)
        if i == 0:
            X[i, 15] = 0.0
        else:
            X[i, 15] = max(0, r[28] - rows[i-1][28])  # r[28] = time
        # time_since_same (16)
        key = (r[0], r[1])  # (addr, func)
        if key in last_seen:
            X[i, 16] = max(0, r[28] - last_seen[key])
        else:
            X[i, 16] = 0.0
        last_seen[key] = r[28]
        X[i, 17] = r[19]  # is_unusual_fc
        X[i, 18] = r[20]  # is_response
        y[i] = r[27]      # binary
        times[i] = r[27]
    return X, y, times


def add_window_aggs(X, window=16):
    """按 preprocess_v2_scada.py:154 make_windows_with_agg 计算 8 个 agg 并 broadcast.
       返回 N×27 数组."""
    n = len(X)
    X27 = np.zeros((n, 27), dtype=np.float32)
    X27[:, :19] = X
    # 按 window=16 分块算 agg
    n_full = (n // window) * window
    if n_full < window:
        return X27
    Xw = X[:n_full].reshape(n_full // window, window, 19)
    IDX_PRESS = 13
    IDX_CRC = 14
    IDX_ISRESP = 18
    IDX_LENGTH = 2
    IDX_ISUNUSUAL = 17
    press_mean_w = Xw[:, :, IDX_PRESS].mean(axis=1)
    crc_mean_w = Xw[:, :, IDX_CRC].mean(axis=1)
    crc_max_w = Xw[:, :, IDX_CRC].max(axis=1)
    cmd_count_w = (1 - Xw[:, :, IDX_ISRESP]).sum(axis=1)
    resp_count_w = Xw[:, :, IDX_ISRESP].sum(axis=1)
    balance_w = (cmd_count_w - resp_count_w) / window
    length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH]))
                                   for i in range(n_full // window)],
                                  dtype=np.float32)
    unusual_count_w = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)
    aggs = np.column_stack([
        press_mean_w, crc_mean_w, crc_max_w, cmd_count_w,
        resp_count_w, balance_w, length_nunique_w, unusual_count_w
    ]).astype(np.float32)  # (n_win, 8)
    # tile + broadcast 到所有 16 步
    aggs_tiled = np.repeat(aggs[:, np.newaxis, :], window, axis=1)
    aggs_flat = aggs_tiled.reshape(n_full, 8)
    X27[:n_full, 19:27] = aggs_flat
    # 末尾不满 window 的行: 不算 agg (broadcast 不到), 保持 0
    return X27


def apply_variant(X27, variant):
    """对 X27 (N×27) 应用截断/clip 变体.
       variant: 'A' (原值), 'B' (uint8 截断), 'C' (A + f[2..5] 截断).
    """
    X = X27.copy()
    if variant == 'A':
        # A: 完全 float 原值 (训练分布), pressure clip [0,100] (跟训练一致)
        X[:, 13] = np.clip(X[:, 13], 0.0, 100.0)  # pressure
        return X
    if variant == 'B':
        # B: 模拟 PC build_raw17_frame 的 uint8 截断 (length=16 格式)
        # feat_vals[:10] = setpoint, gain, reset rate, deadband,
        #                  cycle time, rate, system mode, control scheme,
        #                  pump, solenoid
        for j in range(3, 13):  # 索引 3..12
            X[:, j] = np.clip(np.round(X[:, j]), 0, 255).astype(np.float32)
        # pressure (idx 13) float32 透传
        # crc_rate (idx 14) uint16 透传 (PC: max(0, min(65535, int(round(v)))))
        X[:, 14] = np.clip(np.round(X[:, 14]), 0, 65535).astype(np.float32)
        X[:, 13] = np.clip(X[:, 13], 0.0, 100.0)  # H743 clip
        return X
    if variant == 'C':
        # C: A 的基础上只对 f[2..5] (gain, reset_rate, deadband, cycle_time) 截断
        # f[2]=gain (27 维 idx 4), f[3]=reset_rate (idx 5), f[4]=deadband (idx 6),
        # f[5]=cycle_time (idx 7)
        for j in [4, 5, 6, 7]:
            X[:, j] = np.clip(np.round(X[:, j]), 0, 255).astype(np.float32)
        X[:, 13] = np.clip(X[:, 13], 0.0, 100.0)
        return X
    raise ValueError(f"unknown variant: {variant}")


def make_windows_for_inference(X27, window=16):
    """把 X27 (N×27) 切成 (n_win, window, 27) 滑窗.
       返回 (Xw, last_idx_per_win) — last_idx_per_win[i] 是该窗最后一行在原数组的 idx."""
    n = len(X27)
    n_full = (n // window) * window
    Xw = X27[:n_full].reshape(n_full // window, window, 27)
    last_idx = (np.arange(n_full // window) + 1) * window - 1
    return Xw.astype(np.float32), last_idx


def scale_and_select(Xw, scaler, keep_idx=KEEP_23):
    """(n_win, 16, 27) → 前 19 维 RobustScaler.transform + 后 8 维 identity → 取 KEEP_23 → (n_win, 16, 23).
       训练管线 fit scaler 在 19 维 row-level, 后 8 维 agg 不参与 fit, 见 preprocess_v2_scada.py:139 + scaler_export.h:6-7."""
    n_win, win, _ = Xw.shape
    X_flat = Xw.reshape(n_win * win, 27)
    X_first19 = X_flat[:, :19]
    X_agg8 = X_flat[:, 19:27]
    # 前 19 维 RobustScaler.transform
    X_scaled19 = scaler.transform(X_first19).astype(np.float32)
    X_scaled19 = np.clip(X_scaled19, -10.0, 10.0)
    # 后 8 维 identity (agg 不归一化, 见 scaler_export.h 第 6-7 行)
    X_scaled27 = np.concatenate([X_scaled19, X_agg8], axis=1)
    X_scaled27 = X_scaled27.reshape(n_win, win, 27)
    X_keep = X_scaled27[:, :, keep_idx]
    return X_keep  # (n_win, 16, 23)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensemble_predict(X_keep, onnx_paths):
    """5-seed ensemble 推理. 输入 (n_win, 16, 23) → (n_win,) 平均概率 (sigmoid 后).
       ONNX 输出是 logits, 需手动 sigmoid (见 export_23dim_to_onnx.py 确认是 BCEWithLogitsLoss 模型)."""
    n_win = X_keep.shape[0]
    probs = np.zeros(n_win, dtype=np.float32)
    for path in onnx_paths:
        sess = ort.InferenceSession(path)
        inp = X_keep.transpose(0, 2, 1).astype(np.float32)  # (N, 23, 16)
        out = sess.run(None, {'input': inp})[0].flatten()
        probs += _sigmoid(out)
    probs /= len(onnx_paths)
    return probs


def compute_metrics(y_true, y_pred):
    """Binary classification metrics: Acc/Precision/Recall/F1/FPR."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    n_pos = tp + fn
    n_neg = fp + tn
    acc = (tp + tn) / max(1, n_pos + n_neg)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, n_pos)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    fpr = fp / max(1, n_neg)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Acc': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'FPR': fpr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='只取前 N 行 (调试用)')
    parser.add_argument('--variants', type=str, default='A,B,C',
                        help='逗号分隔变体, 默认 A,B,C')
    parser.add_argument('--thr', type=float, default=0.5,
                        help='二分类阈值')
    args = parser.parse_args()

    t0 = time.time()
    rows = load_csv(limit=args.limit)
    log(f"=== STEP 2: row-level features (time_diff / time_since_same) ===")
    X19, y, times = build_row_level_features(rows)
    log(f"  X19 shape={X19.shape}, attack count={int((y==1).sum())}, "
        f"normal count={int((y==0).sum())}", t0)
    log(f"=== STEP 3: window-level aggs + 27 维 ===")
    X27 = add_window_aggs(X19, window=WINDOW)
    log(f"  X27 shape={X27.shape}", t0)
    log(f"=== STEP 4: load scaler + 5-seed ONNX ===")
    scaler = joblib.load(SCALER_PATH)
    onnx_paths = [os.path.join(BASE, ONNX_TPL.format(seed=s)) for s in SEEDS]
    for p in onnx_paths:
        assert os.path.exists(p), f"missing {p}"
    log(f"  scaler OK, {len(onnx_paths)} ONNX loaded", t0)

    for variant in args.variants.split(','):
        v = variant.strip().upper()
        if not v: continue
        log(f"=== STEP 5: variant {v} ===")
        X27_v = apply_variant(X27, v)
        Xw, last_idx = make_windows_for_inference(X27_v, window=WINDOW)
        log(f"  Xw shape={Xw.shape}", t0)
        X_keep = scale_and_select(Xw, scaler)
        log(f"  X_keep shape={X_keep.shape}", t0)
        probs = ensemble_predict(X_keep, onnx_paths)
        # 窗级 label = 该窗最后一行的 label (跟 H743 推理时机一致)
        y_win = y[last_idx]
        y_pred = (probs >= args.thr).astype(np.int64)
        m = compute_metrics(y_win, y_pred)
        log(f"  >>> variant {v} @ thr={args.thr}: "
            f"Acc={m['Acc']:.4f} P={m['Precision']:.4f} R={m['Recall']:.4f} "
            f"F1={m['F1']:.4f} FPR={m['FPR']:.4f} "
            f"(TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']})", t0)
        # dump CSV
        out_csv = os.path.join(BASE, f"sim_pc_align_{v}_thr{args.thr}.csv")
        with open(out_csv, 'w') as f:
            f.write("frame_idx_last,binary,prob_5seed,pred\n")
            for i in range(len(probs)):
                f.write(f"{int(last_idx[i])},{int(y_win[i])},"
                        f"{probs[i]:.6f},{int(y_pred[i])}\n")
        log(f"  dumped {out_csv}", t0)
        # dump feats (窗内最后一步的 23 维) — 跟 H743 [DBG_F] 对齐
        out_feats = os.path.join(BASE, f"sim_pc_align_{v}_thr{args.thr}_feats.csv")
        with open(out_feats, 'w') as f:
            f.write("frame_idx_last,binary," + ",".join(f"f{i}" for i in range(23)) + "\n")
            for i in range(len(probs)):
                feats_last = X_keep[i, -1, :]  # 窗内最后一步
                f.write(f"{int(last_idx[i])},{int(y_win[i])},"
                        + ",".join(f"{x:.4f}" for x in feats_last) + "\n")
        log(f"  dumped {out_feats}", t0)

    log("=== DONE ===", t0)


if __name__ == "__main__":
    main()
