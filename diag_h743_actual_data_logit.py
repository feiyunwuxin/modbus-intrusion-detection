"""
diag_h743_actual_data_logit.py  --  验证假设: H743 FPR=92.86% 是不是真 OOD
                                    PC sim 用 H743 实际跑的数据 (IanArffDataset.csv frame 1..457)
                                    + 跟训练 1:1 流水线 (scaler + windowize + KEEP_23)
                                    看 F1 是不是 ≈ 0.8646 (跟 X_test 一样)
                                    还是 ≈ 0.07 (跟 H743 一样 = 真 OOD)
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from tensorflow.lite.python.interpreter import Interpreter


BASE = r"C:\work\Claude\Issue"
CSV_PATH = os.path.join(BASE, "IanArffDataset.csv")
SCALER_PATH = os.path.join(BASE, "scaler_binary_v2_scada.joblib")
SEED = 42
WINDOW = 16
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}   # 跟 preprocess_v2_scada.py 一致

# Row-level feature order (跟 preprocess_v2_scada.py 一致, 19 维)
ROW_LEVEL = [
    "address", "function", "length",
    "setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
    "system mode", "control scheme", "pump", "solenoid",
    "pressure measurement", "crc rate",
    "time_diff", "time_since_last_same_addr_func", "is_unusual_fc", "is_response",
]

# Window agg feature indices (from preprocess_v2_scada.py)
# 19=press_mean_w, 20=crc_mean_w, 21=crc_max_w,
# 22=cmd_count_w, 23=resp_count_w, 24=balance_w, 25=length_nunique_w, 26=unusual_count_w
AGG_NAMES = ["press_mean_w", "crc_mean_w", "crc_max_w",
             "cmd_count_w", "resp_count_w", "balance_w",
             "length_nunique_w", "unusual_count_w"]


def main():
    print("=" * 70)
    print("[1] 加载 IanArffDataset.csv (274k 帧)")
    print("=" * 70)
    df = pd.read_csv(CSV_PATH)
    print(f"  raw shape: {df.shape}")
    print(f"  time range: {df['time'].min():.3f} ~ {df['time'].max():.3f} sec")
    print(f"  binary attack: {df['binary result'].sum()}/{len(df)} ({df['binary result'].mean()*100:.1f}%)")

    # 跟 preprocess_v2_scada.py 1:1 pipeline
    print()
    print("=" * 70)
    print("[2] preprocess_v2_scada.py 1:1 pipeline (row-level features)")
    print("=" * 70)
    NUMERIC_COLS = ["setpoint", "gain", "reset rate", "deadband", "cycle time",
                    "rate", "system mode", "control scheme", "pump", "solenoid",
                    "pressure measurement"]
    df = df.replace("?", np.nan)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    LABELS = ["binary result", "categorized result", "specific result"]
    y_full = df[LABELS].copy()
    # 去掉 CSV 末尾的 Unnamed 空列
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.drop(columns=LABELS)

    df = df.sort_values("time").reset_index(drop=True)
    y_full = y_full.loc[df.index]

    # time_diff
    df["time_diff"] = df["time"].diff().fillna(0)
    # time_since_last_same_addr_func
    df["_addr_func"] = df["address"].astype(int) * 1000 + df["function"].astype(int)
    last_seen = {}
    ts_l = np.zeros(len(df), dtype=np.float32)
    addr_func_arr = df["_addr_func"].values
    time_arr = df["time"].values
    for i in range(len(df)):
        key = addr_func_arr[i]
        if key in last_seen:
            ts_l[i] = time_arr[i] - last_seen[key]
        last_seen[key] = time_arr[i]
    df["time_since_last_same_addr_func"] = ts_l
    df = df.drop(columns=["_addr_func"])

    df["is_unusual_fc"] = df["function"].apply(lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)
    df["is_response"] = df["command response"].astype(np.float32)
    df = df.drop(columns=["time", "command response"])
    df = df.fillna(0)
    df["pressure measurement"] = df["pressure measurement"].clip(0, 100)
    for c in df.columns:
        df[c] = df[c].astype(np.float32)

    print(f"  after row-level: shape={df.shape}, columns={df.columns.tolist()}")
    # 应该 19 列
    assert len(df.columns) == 19, f"Expected 19 columns, got {len(df.columns)}"

    print()
    print("=" * 70)
    print("[3] 70/10/20 split (temporal) — H743 frame 1..457 在 TRAIN 区域")
    print("=" * 70)
    N = len(df)
    n_train = int(N * 0.70)
    n_val   = int(N * 0.10)
    print(f"  total: {N}  train: 0..{n_train-1}  val: {n_train}..{n_train+n_val-1}  test: {n_train+n_val}..{N-1}")
    print(f"  H743 frame 1..457 → 全在 TRAIN 区域 (frame 0..{n_train-1})")

    print()
    print("=" * 70)
    print("[4] RobustScaler fit on train + transform all (跟训练 1:1)")
    print("=" * 70)
    scaler = joblib.load(SCALER_PATH)
    X_all = scaler.transform(df.values)   # ⚠️ 跟训练 1:1: scaler.transform 不 fit
    print(f"  X_all shape: {X_all.shape}")
    y_all = y_full["binary result"].values.astype(np.int64)
    print(f"  y_all attack%: {y_all.mean()*100:.1f}")

    print()
    print("=" * 70)
    print("[5] window=16 + per-window aggregates (跟训练 1:1)")
    print("=" * 70)
    def make_windows_with_agg(X, y_bin, win):
        n = (len(X) // win) * win
        Xw = X[:n].reshape(n // win, win, -1)
        yw = y_bin[:n].reshape(n // win, win).max(axis=1).astype(np.int64)
        n_win = Xw.shape[0]
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
        balance_w = (cmd_count_w - resp_count_w) / win
        length_nunique_w = np.array([len(np.unique(Xw[i, :, IDX_LENGTH])) for i in range(n_win)], dtype=np.float32)
        unusual_count_w = Xw[:, :, IDX_ISUNUSUAL].sum(axis=1)
        aggs = np.column_stack([
            press_mean_w, crc_mean_w, crc_max_w,
            cmd_count_w, resp_count_w, balance_w,
            length_nunique_w, unusual_count_w
        ]).astype(np.float32)
        aggs_tiled = np.repeat(aggs[:, np.newaxis, :], win, axis=1)
        Xw_full = np.concatenate([Xw, aggs_tiled], axis=-1)
        return Xw_full, yw

    Xw_all, yw_all = make_windows_with_agg(X_all, y_all, WINDOW)
    print(f"  Xw_all shape: {Xw_all.shape}  (windows={Xw_all.shape[0]}, time={Xw_all.shape[1]}, dim={Xw_all.shape[2]})")

    print()
    print("=" * 70)
    print("[6] 截取 H743 实际跑的 28 windows (frame 0..457, 0-indexed)")
    print("=" * 70)
    Xw_h743 = Xw_all[:28]
    yw_h743 = yw_all[:28]
    print(f"  Xw_h743 shape: {Xw_h743.shape}")
    print(f"  yw_h743 attack windows: {yw_h743.sum()}/{len(yw_h743)} (max-agg)")

    # 验证 ch12 (crc_rate) frame 0..15 跟 H743 [TFLM-IN] 一致
    # 注: 27-dim col 12 = "is_response" (列 18), KEEP_23[12]=14=crc_rate
    print(f"  27-dim col 12 (is_response) frame 0..15 (window 0):")
    print(f"    {Xw_h743[0, :, 12]}")
    print(f"  27-dim col 14 (crc_rate) frame 0..15 (window 0) — 跟 H743 [TFLM-IN] 比对:")
    print(f"    {Xw_h743[0, :, 14]}")
    print(f"  期望 H743 [TFLM-IN]:")
    print(f"    [-0.261, -0.367, 0.636, 0.739, -0.261, 0.053, 0.636, 0.739,")
    print(f"     -0.261, 0.636, 0.636, 0.739, -0.261, -0.367, 0.636, 0.739]")

    print()
    print("=" * 70)
    print("[7] KEEP_23 + clip ±10 (跟 H743 1:1)")
    print("=" * 70)
    Xw_h743_23 = Xw_h743[:, :, KEEP_23]
    Xw_h743_23 = np.clip(Xw_h743_23, -10, 10)
    print(f"  Xw_h743_23 shape: {Xw_h743_23.shape}")
    # 对比 X_test (benchmark 数据)
    X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy"))[:, :, KEEP_23]
    X_test = np.clip(X_test, -10, 10)
    y_test = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy"))
    print(f"  X_test shape (对照组): {X_test.shape}")

    print()
    print("=" * 70)
    print("[8] 跑 v1 INT8 .tflite seed=42 (跟 H743 同一模型)")
    print("=" * 70)
    tflite_path = os.path.join(BASE, f"model_v4_se_23dim_ch32_qat_s{SEED}_int8.tflite")
    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']

    def run(X):
        probs = np.zeros(len(X), dtype=np.float32)
        logits = np.zeros(len(X), dtype=np.float32)
        for i in range(len(X)):
            x_fp32 = X[i].T.flatten().astype(np.float32)   # (C*T,) row-major
            x_q = np.clip(np.round(x_fp32 / in_scale + in_zp), -128, 127).astype(np.int8)
            x_q_input = x_q.reshape(1, 23, 16)
            interp.set_tensor(inp_d['index'], x_q_input)
            interp.invoke()
            y_q = interp.get_tensor(out_d['index'])
            logit = (y_q.astype(np.float32) - out_zp) * out_scale
            logit_v = float(logit[0]) if logit.ndim > 0 else float(logit)
            logits[i] = logit_v
            probs[i] = 1.0 / (1.0 + np.exp(-logit_v))
        return probs, logits

    # Eval helper
    def eval_(yw, probs, thr=0.5):
        yp = (probs >= thr).astype(int)
        tp = int(((yp == 1) & (yw == 1)).sum())
        fp = int(((yp == 1) & (yw == 0)).sum())
        tn = int(((yp == 0) & (yw == 0)).sum())
        fn = int(((yp == 0) & (yw == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) > 0 else 0
        recall = tp / (tp + fn) if (tp+fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        return tp, fp, tn, fn, f1, recall, fpr

    # [A] X_test baseline (对照组)
    print(f"  [A] X_test (对照组, 模型测试集):")
    probs_test, logits_test = run(X_test)
    tp, fp, tn, fn, f1, r, fpr = eval_(y_test, probs_test)
    print(f"      thr=0.5:  TP={tp} FP={fp} TN={tn} FN={fn}  F1={f1:.4f}  R={r:.4f}  FPR={fpr:.4f}")

    # [B] H743 实际数据 (IanArffDataset.csv frame 1..457)
    print(f"  [B] H743 实际数据 (IanArffDataset.csv frame 1..457, 28 windows):")
    probs_h743, logits_h743 = run(Xw_h743_23)
    tp, fp, tn, fn, f1, r, fpr = eval_(yw_h743, probs_h743)
    print(f"      thr=0.5:  TP={tp} FP={fp} TN={tn} FN={fn}  F1={f1:.4f}  R={r:.4f}  FPR={fpr:.4f}")

    # 同样 T scaling
    probs_h743_T2 = 1.0 / (1.0 + np.exp(-logits_h743 / 2.0))
    tp, fp, tn, fn, f1, r, fpr = eval_(yw_h743, probs_h743_T2)
    print(f"      thr=0.5 T=2.0:  TP={tp} FP={fp} TN={tn} FN={fn}  F1={f1:.4f}  R={r:.4f}  FPR={fpr:.4f}")

    print()
    print(f"  Logit 分布 [B]:")
    print(f"    min={logits_h743.min():.3f}  max={logits_h743.max():.3f}  mean={logits_h743.mean():.3f}  std={logits_h743.std():.3f}")
    print(f"    Normal windows logit:  min={logits_h743[yw_h743==0].min():.3f}  max={logits_h743[yw_h743==0].max():.3f}  mean={logits_h743[yw_h743==0].mean():.3f}")
    print(f"    Attack windows logit:  min={logits_h743[yw_h743==1].min():.3f}  max={logits_h743[yw_h743==1].max():.3f}  mean={logits_h743[yw_h743==1].mean():.3f}")

    print()
    print(f"  Per-window 详情 [B] (H743 实际数据):")
    for i in range(len(yw_h743)):
        marker = '(A)' if yw_h743[i] == 1 else '(N)'
        prob_t1 = probs_h743[i]
        prob_t2 = 1.0 / (1.0 + np.exp(-logits_h743[i] / 2.0))
        print(f"    w#{i:2d} {marker} y={yw_h743[i]}  logit={logits_h743[i]:7.3f}  prob_T1={prob_t1:.4f}  prob_T2={prob_t2:.4f}")


if __name__ == '__main__':
    main()
