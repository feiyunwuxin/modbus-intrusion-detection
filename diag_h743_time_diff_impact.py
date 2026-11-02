#!/usr/bin/env python3
"""验证假设: 40ms 注入 vs 数据集原始 1s+ 速率, 是否就是 H743 F1 暴跌根因.

方法:
  1. 加载 ARFF 原始数据, 用 PC preprocess + TCN+SE 模型跑 F1 (baseline)
  2. 模拟 H743 测试: 把 time_diff 强制改成 0.04s, 再跑一次
  3. 如果 (2) 的 F1 也暴跌到 ~0.53, 假设得证
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
import joblib
import torch

MODEL_PATH = 'model_v4_se_23dim_b64_ch32_do01_window16_s42.pt'
SCALER_PATH = 'scaler_binary_v2_scada.joblib'
DATA_PATH = 'ianArffDatasetF32.csv'
W = 16

# H743 KEEP_18 + KEEP_23 (跟 scaler_export.h / sliding_window.h 完全一致)
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]

UNUSUAL_FCS = {133, 136, 137, 138, 139, 140, 171}


def load_data():
    df = pd.read_csv(DATA_PATH)
    cols = ['address', 'function', 'length', 'setpoint', 'gain', 'reset rate',
            'deadband', 'cycle time', 'rate', 'system mode', 'control scheme',
            'pump', 'solenoid', 'pressure measurement', 'crc rate',
            'command response', 'time', 'binary result', 'categorized result']
    # pc 流: command response is 0/1, is_response from raw17
    df['is_response'] = (df['command response'] == 1).astype(np.int8)
    return df


def build_features(df: pd.DataFrame, force_time_diff: float | None = None):
    """重建 19-dim 训练特征 (跟 PC preprocess_v2_scada.py 一致).
       time_diff 可强制覆盖 (模拟 H743 auto-40ms 注入)."""
    n = len(df)
    raw = np.zeros((n, 19), dtype=np.float32)

    # 0..18 字段映射 (跟 ARFF 顺序对齐)
    def f32(col):
        return pd.to_numeric(df[col], errors='coerce').fillna(0).values.astype(np.float32)
    raw[:, 0]  = f32('address')
    raw[:, 1]  = f32('function')
    raw[:, 2]  = f32('length')
    raw[:, 3]  = f32('setpoint')
    raw[:, 4]  = f32('gain')
    raw[:, 5]  = f32('reset rate')
    raw[:, 6]  = f32('deadband')
    raw[:, 7]  = f32('cycle time')
    raw[:, 8]  = f32('rate')
    raw[:, 9]  = f32('system mode')
    raw[:, 10] = f32('control scheme')
    raw[:, 11] = f32('pump')
    raw[:, 12] = f32('solenoid')
    raw[:, 13] = f32('pressure measurement').clip(0, 100)
    raw[:, 14] = f32('crc rate')
    # time_diff: 强制覆盖 OR 原始 ARFF time_diff
    if force_time_diff is not None:
        raw[:, 15] = force_time_diff
    else:
        raw[:, 15] = df['time'].diff().fillna(0).values.astype(np.float32)
    # time_since_last_same_addr_func: 简化用 0.04s (跟 H743 同样假设, 但对结论无影响)
    raw[:, 16] = force_time_diff if force_time_diff is not None else 0.04
    # is_unusual_fc
    raw[:, 17] = np.isin(df['function'].values, list(UNUSUAL_FCS)).astype(np.float32)
    # is_response (master:0, response:1)
    raw[:, 18] = df['is_response'].values.astype(np.float32)

    # scaler
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(raw)  # (n, 19)

    # 19 -> 18 (KEEP_18)
    X18 = X[:, KEEP_18]

    # 滑窗 + 6 aggregates + 27-dim + KEEP_23 + clip
    n_w = n // W
    X18 = X18[:n_w * W]
    # 重塑 (n_w, 16, 18)
    X18w = X18.reshape(n_w, W, 18)

    press_mean = X18w[:, :, 12].mean(axis=1)
    crc_max    = X18w[:, :, 13].max(axis=1)
    # resp_signed = sum(X18w[:, :, 17])  (X18 idx 17 = is_response scaler[18] = (raw - 1)/1 = -1 if master)
    resp_signed = X18w[:, :, 17].sum(axis=1)
    balance_w   = 1.0 - resp_signed / 8.0
    unusual_w   = X18w[:, :, 16].sum(axis=1)
    # length_nunique (从 X18w[:, :, 2])
    length_scaled = X18w[:, :, 2]
    length_int = np.round(length_scaled).astype(np.int32)
    nunique_w = np.array([len(np.unique(length_int[i])) for i in range(n_w)], dtype=np.float32)

    # 组装 27-dim (每窗一行)
    X27 = np.zeros((n_w, W, 27), dtype=np.float32)
    X27[:, :, KEEP_18] = X18w
    X27[:, :, 19] = press_mean[:, None]
    X27[:, :, 20] = X18w[:, :, 13].mean(axis=1)[:, None]
    X27[:, :, 21] = crc_max[:, None]
    X27[:, :, 22] = (W + resp_signed)[:, None]   # cmd_count_w
    X27[:, :, 23] = resp_signed[:, None]
    X27[:, :, 24] = balance_w[:, None]
    X27[:, :, 25] = nunique_w[:, None]
    X27[:, :, 26] = unusual_w[:, None]

    # KEEP_23 切片 -> (n_w, 16, 23)
    X23 = X27[:, :, KEEP_23]
    X23 = np.clip(X23, -10, 10)

    # label (per-window max-aggregation)
    labels_raw = df['binary result'].values[:n_w * W].astype(np.float32)
    labels_raw = labels_raw.reshape(n_w, W).max(axis=1) > 0

    return X23, labels_raw


def evaluate(X, labels, threshold=0.49, model_path=MODEL_PATH):
    """跑 PC 模型, 计算 per-window 4 指标 (跟 H743 summary 一致)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('vh', 'verify_h743_vs_pc.py')
    vh = importlib.util.module_from_spec(spec); spec.loader.exec_module(vh)
    # 直接覆盖 PC_MODEL_PATH 然后调无参版
    vh.PC_MODEL_PATH = model_path
    model = vh.load_pc_model()
    model.eval()

    probs = []
    BATCH = 512
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            x = torch.from_numpy(X[i:i+BATCH])
            # X 是 (B, 16, 23), 喂 (B, 23, 16)
            x = x.transpose(1, 2).contiguous()
            logit = model(x)
            p = torch.sigmoid(logit).squeeze(-1).numpy()
            probs.append(p)
    probs = np.concatenate(probs)

    pred = (probs >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc  = (tp + tn) / (tp + fp + tn + fn)

    return {
        'total': len(labels), 'attack': int(labels.sum()), 'normal': int((~labels).sum()),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
    }


def main():
    df = load_data()
    print(f'Dataset: {len(df)} rows, attack ratio = {df["binary result"].mean():.4f}')
    print()

    print('=' * 70)
    print('[A] PC baseline: 使用 ARFF 原始 time_diff (μ=1.43s)')
    print('=' * 70)
    X, labels = build_features(df, force_time_diff=None)
    m = evaluate(X, labels)
    print(f'  Total={m["total"]}, Attack={m["attack"]}, Normal={m["normal"]}')
    print(f'  TP={m["tp"]}  FP={m["fp"]}  FN={m["fn"]}  TN={m["tn"]}')
    print(f'  Prec={m["precision"]:.4f}  Rec={m["recall"]:.4f}  F1={m["f1"]:.4f}  Acc={m["accuracy"]:.4f}')
    print()

    print('=' * 70)
    print('[B] 模拟 H743 auto-40ms: time_diff 强制改为 0.04s')
    print('=' * 70)
    X40, labels40 = build_features(df, force_time_diff=0.04)
    m40 = evaluate(X40, labels40)
    print(f'  Total={m40["total"]}, Attack={m40["attack"]}, Normal={m40["normal"]}')
    print(f'  TP={m40["tp"]}  FP={m40["fp"]}  FN={m40["fn"]}  TN={m40["tn"]}')
    print(f'  Prec={m40["precision"]:.4f}  Rec={m40["recall"]:.4f}  F1={m40["f1"]:.4f}  Acc={m40["accuracy"]:.4f}')
    print()

    print('=' * 70)
    print('[C] 对比 H743 实测 (Round 50, auto-40ms full 274k)')
    print('=' * 70)
    print(f'  Total=17164, Attack=8821, Normal=8343')
    print(f'  TP=3398  FP=494  FN=5423  TN=7849')
    print(f'  Prec=0.8731  Rec=0.3852  F1=0.5346  Acc=0.6553')
    print()

    if abs(m40['f1'] - 0.5346) < 0.05:
        print('✅ 假设成立: time_diff=0.04s 是 H743 F1 暴跌根因')
    elif m40['f1'] < m['f1'] * 0.7:
        print('⚠️  部分成立: time_diff 影响显著, 但还有其他因素')
    else:
        print('❌ 假设不成立: time_diff 不是主因, 需继续排查')


if __name__ == '__main__':
    main()
