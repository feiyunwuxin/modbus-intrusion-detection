#!/usr/bin/env python3
"""
pc_inference_765_vs_h743.py
===========================
PC 端用 model_tcn_v4_se_window16.pt (seed=42, 跟 H743 v4se23_ch32_s42 一致)
推理 IanArffDatasetF32.csv 前 765 行, 输出每帧 prob.

对比 H743 实际推理概率曲线, 看 H743 是否跟 PC 一致.
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CSV_PATH = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"
NPY_TEST = r"C:\work\Claude\Issue\X_test_binary_v2_scada_window16.npy"
NPY_Y_TEST = r"C:\work\Claude\Issue\y_test_binary_v2_scada_window16.npy"
SCALER_PATH = r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib"
MODEL_PATH = r"C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"

CLIP_VAL = 10.0
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}


# === TCN+SE 23-dim 模型架构 (跟 tcn_binary_v4_se_window16.py 一致) ===
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class TCNBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, se_reduction=8):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return torch.relu(x + residual)


class TCNClassifierSE(nn.Module):
    def __init__(self, in_ch=23, n_blocks=3, channels=32, kernel_size=3, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
        return self.fc2(x).squeeze(-1)


def build_18_raw_from_csv(df_765):
    """跟 h743 feature_extractor.c + preprocess_v2_scada.py 一致, 算 RAW 18-dim."""
    df = df_765.sort_values("time").reset_index(drop=True)
    df["time_diff"] = df["time"].diff().fillna(0)
    last = {}
    tsl = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        k = int(df["address"].iat[i]) * 1000 + int(df["function"].iat[i])
        if k in last:
            tsl[i] = df["time"].iat[i] - last[k]
        last[k] = df["time"].iat[i]
    df["time_since_last_same_addr_func"] = tsl
    df["is_unusual_fc"] = df["function"].apply(lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)
    df["is_response"] = df["command response"].astype(np.float32)
    df["pressure measurement"] = pd.to_numeric(df["pressure measurement"], errors="coerce").fillna(0).clip(0, 100)
    keep = ["address", "function", "length", "gain", "reset rate", "deadband", "cycle time",
            "rate", "system mode", "control scheme", "pump", "solenoid",
            "pressure measurement", "crc rate", "time_diff",
            "time_since_last_same_addr_func", "is_unusual_fc", "is_response"]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[keep].astype(np.float32).values  # (N, 18)


def sliding_window_18_to_23(norm18):
    """跟 h743 sliding_window.c 一致: 18-dim 滑动 → 27-dim 组装 → KEEP_23 切片 → clip ±10."""
    SW_WINDOW = 16
    KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    out = []
    for t0 in range(len(norm18) - SW_WINDOW + 1):
        win = norm18[t0:t0 + SW_WINDOW]
        press_sum = win[:, 12].sum()
        crc_max = win[:, 13].max()
        resp_signed = win[:, 17].sum()
        unusual_sign = win[:, 16].sum()
        seen = set()
        n = 0
        for v in win[:, 2]:
            k = int(round(float(v)))
            if k not in seen:
                seen.add(k)
                n += 1
        press_mean = press_sum / SW_WINDOW
        balance = 1.0 - resp_signed / 8.0
        crc_mean = win[:, 13].mean()
        cmd_count = float(SW_WINDOW - unusual_sign - resp_signed)
        frame27 = np.zeros((SW_WINDOW, 27), dtype=np.float32)
        for c18 in range(18):
            frame27[:, KEEP_18[c18]] = win[:, c18]
        frame27[:, 19] = press_mean
        frame27[:, 20] = crc_mean
        frame27[:, 21] = crc_max
        frame27[:, 22] = cmd_count
        frame27[:, 23] = resp_signed
        frame27[:, 24] = balance
        frame27[:, 25] = n
        frame27[:, 26] = unusual_sign
        sliced = frame27[:, KEEP_23]  # (16, 23)
        sliced = np.clip(sliced, -CLIP_VAL, CLIP_VAL)
        out.append(sliced.T.astype(np.float32))  # (23, 16)
    return np.stack(out, axis=0)


def main():
    # 1. 读 CSV 765 行 + 算 18-dim RAW + 归一化
    df_765 = pd.read_csv(CSV_PATH, nrows=765)
    raw18 = build_18_raw_from_csv(df_765)
    import joblib
    scaler = joblib.load(SCALER_PATH)
    KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    mu_18 = scaler.center_[KEEP_18].astype(np.float32)
    sg_18 = scaler.scale_[KEEP_18].astype(np.float32)
    norm18 = (raw18 - mu_18) / sg_18

    # 2. 算 (N_win, 23, 16) 输入
    X = sliding_window_18_to_23(norm18)
    print(f"X shape: {X.shape}  (N_win, 23, 16)")
    print(f"X max(|.|): {np.abs(X).max():.3f}")

    # 3. 加载模型 (ch=32, n_features=23, do=0.1, 跟 H743 v4se23_ch32_s42 一致)
    model = TCNClassifierSE(in_ch=23, n_blocks=3, channels=32, kernel_size=3,
                             dilations=(1, 2, 4), dropout=0.1)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"  ch=32, n_features=23, do=0.1, seed=42, best_val_f1m={ckpt.get('best_val_f1m', 'N/A')}")

    # 4. 推理
    with torch.no_grad():
        xb = torch.from_numpy(X).float()  # (N_win, 23, 16)
        probs = torch.sigmoid(model(xb)).squeeze(-1).numpy()  # (N_win,)
    print(f"probs shape: {probs.shape}, range=[{probs.min():.3f}, {probs.max():.3f}]")

    # 5. 跟 H743 实际推理比较 (从 ALARM log 提取)
    # H743 实际推理: 每帧 prob 在 ALARM 时刻触发, 但 H743 每帧都推理 (即使不报警)
    # 我们可以从 log 的 alarm prob 反推, 但非报警帧没记录. 先用 thr=0.49 报 PC 端的 FP/TP
    THR = 0.49
    # 每帧的概率: PC 推理是 N_win = N - 15, 每帧 prob 是从 frame_idx 16 开始 (前 15 帧没有 window)
    # 但 H743 是每帧都推理 (frame_idx 1-based, 第 16 帧开始有 window, 但前面 frame 0-15 应该不报警)
    # 实际上 H743 在前 16 帧就开始报警 (alarm#1 frame_idx=10), 说明 H743 的 window 不是简单的"过去 16 帧"
    # 或者 H743 等到 16 帧才推理, alarm#1 frame_idx=10 是 bug, 或 frame_idx 是 1-based 但已发 16 帧.
    # 简单做法: PC prob 映射到 frame_idx 16..765, 即 PC prob[i] 对应 frame_idx=i+16 (1-based)
    # 但 H743 alarm frame_idx=10 意味着第 10 帧就报警了 — 这需要看 H743 的实际行为.

    # 6. 按 frame_idx 输出 PC 端的报警
    fp_pc = []
    tp_pc = []
    csv_binary = df_765["binary result"].values  # 1-based idx i+1 → values[i]
    for i, p in enumerate(probs):
        frame_idx_1based = i + 16  # PC sliding window 是 N-15 个, frame 1..15 没有
        if frame_idx_1based > 765:
            break
        binary = csv_binary[frame_idx_1based - 1] if frame_idx_1based <= 765 else 0
        if p >= THR:
            if binary == 1:
                tp_pc.append((frame_idx_1based, p))
            else:
                fp_pc.append((frame_idx_1based, p))
    print()
    print(f"=== PC 端 (seed=42 模型) thr={THR} 推理结果 ===")
    print(f"  TP: {len(tp_pc)}")
    print(f"  FP: {len(fp_pc)}")
    print(f"  Total alarm: {len(tp_pc) + len(fp_pc)}")
    if tp_pc:
        recall = len(tp_pc) / 135
        precision = len(tp_pc) / (len(tp_pc) + len(fp_pc))
        f1 = 2 * precision * recall / (precision + recall)
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1: {f1:.4f}")
    print()
    print("=== PC TP (前 30) ===")
    for f, p in tp_pc[:30]:
        at = df_765.iloc[f-1]["specific result"]
        fc = df_765.iloc[f-1]["function"]
        print(f"  frame={f} prob={p:.3f} attack_type={at} fc={fc}")
    print()
    print("=== PC FP (前 20) ===")
    for f, p in fp_pc[:20]:
        at = df_765.iloc[f-1]["specific result"]
        fc = df_765.iloc[f-1]["function"]
        print(f"  frame={f} prob={p:.3f} binary=0 attack_type={at} fc={fc}")
    print()
    # 7. 对比 H743: H743 是按帧输出 prob, 但日志只在报警时记 prob
    # 我们用 H743 的 77 个 alarm 跟 PC 的报警对比
    # H743 TP frames (从前面分析): {63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119,
    #                                  123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 198, 238}
    # H743 FP frames: 49 个 | 167, 171, ..., 643 (battery=0)
    # 我们把这些 frame 的 PC prob 提取出来
    h743_tp_frames = [63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 198, 238]
    h743_fp_frames = [167, 171, 175, 179, 183, 187, 191, 195, 231, 235, 241, 245, 249, 253, 255,
                       523, 527, 531, 535, 539, 543, 547, 551, 555, 559, 563, 567, 571, 575, 579,
                       583, 587, 591, 595, 599, 603, 607, 611, 615, 619, 623, 627, 631, 635, 639, 643,
                       757, 761, 765]
    print("=== H743 TP frames vs PC prob ===")
    print(f"{'frame':<7} {'h743_prob':<10} {'pc_prob':<10} {'match':<7} {'attack_type':<6} {'fc':<4}")
    h743_tp_probs = {63: 0.621, 67: 0.731, 71: 0.843, 75: 0.870, 79: 0.868, 83: 0.871, 87: 0.872, 91: 0.844, 95: 0.844, 99: 0.855, 103: 0.852, 107: 0.852, 111: 0.849, 115: 0.862, 119: 0.860, 123: 0.861, 127: 0.860, 131: 0.861, 135: 0.855, 139: 0.908, 143: 0.934, 147: 0.949, 151: 0.963, 155: 0.959, 159: 0.965, 163: 0.971, 198: 0.681, 238: 0.505}
    for f in h743_tp_frames:
        pc_idx = f - 16
        if 0 <= pc_idx < len(probs):
            p_pc = float(probs[pc_idx])
        else:
            p_pc = -1
        p_h743 = h743_tp_probs.get(f, -1)
        at = df_765.iloc[f-1]["specific result"]
        fc = df_765.iloc[f-1]["function"]
        diff = p_pc - p_h743 if p_pc > 0 and p_h743 > 0 else 0
        print(f"{f:<7} {p_h743:<10.3f} {p_pc:<10.3f} {diff:+.3f}   {at:<6} {fc:<4}")
    print()
    print("=== H743 FP frames vs PC prob ===")
    print(f"{'frame':<7} {'h743_prob':<10} {'pc_prob':<10} {'delta':<8} {'fc':<4}")
    h743_fp_probs = {167: 0.983, 171: 0.984, 175: 0.984, 179: 0.950, 183: 0.937, 187: 0.949, 191: 0.947, 195: 0.956, 231: 0.965, 235: 0.963, 241: 0.858, 245: 0.848, 249: 0.830, 253: 0.798, 255: 0.511, 523: 0.601, 527: 0.847, 531: 0.951, 535: 0.972, 539: 0.966, 543: 0.969, 547: 0.968, 551: 0.977, 555: 0.968, 559: 0.978, 563: 0.974, 567: 0.980, 571: 0.980, 575: 0.981, 579: 0.981, 583: 0.975, 587: 0.969, 591: 0.969, 595: 0.971, 599: 0.976, 603: 0.976, 607: 0.969, 611: 0.976, 615: 0.966, 619: 0.973, 623: 0.965, 627: 0.974, 631: 0.972, 635: 0.941, 639: 0.781, 643: 0.543, 757: 0.585, 761: 0.798, 765: 0.819}
    for f in sorted(h743_fp_probs.keys()):
        pc_idx = f - 16
        p_pc = float(probs[pc_idx]) if 0 <= pc_idx < len(probs) else -1
        p_h743 = h743_fp_probs[f]
        fc = df_765.iloc[f-1]["function"]
        diff = p_pc - p_h743 if p_pc > 0 else 0
        print(f"{f:<7} {p_h743:<10.3f} {p_pc:<10.3f} {diff:+.3f}   {fc:<4}")
    print()
    print("=== H743 沉默期 frame 46-62 (PC 端 prob 是否本应 ALARM) ===")
    print(f"{'frame':<7} {'pc_prob':<10} {'binary':<8} {'attack_type':<6} {'fc':<4} {'h743_thr':<10}")
    for f in range(46, 63):
        pc_idx = f - 16
        p_pc = float(probs[pc_idx]) if 0 <= pc_idx < len(probs) else -1
        binary = int(df_765.iloc[f-1]["binary result"])
        at = int(df_765.iloc[f-1]["specific result"])
        fc = int(df_765.iloc[f-1]["function"])
        thr_alarm = "ALARM" if p_pc >= 0.49 else "silent"
        print(f"{f:<7} {p_pc:<10.3f} {binary:<8} {at:<6} {fc:<4} {thr_alarm}")
    print()
    print("=== Frame 17-46 (H743 完全沉默, PC 端什么 prob) ===")
    print(f"{'frame':<7} {'pc_prob':<10} {'binary':<8} {'attack_type':<6} {'fc':<4}")
    for f in [16, 17, 20, 25, 30, 35, 40, 45, 46, 50, 55, 60]:
        pc_idx = f - 16
        if 0 <= pc_idx < len(probs):
            p_pc = float(probs[pc_idx])
        else:
            p_pc = -1
        binary = int(df_765.iloc[f-1]["binary result"])
        at = int(df_765.iloc[f-1]["specific result"])
        fc = int(df_765.iloc[f-1]["function"])
        print(f"{f:<7} {p_pc:<10.3f} {binary:<8} {at:<6} {fc:<4}")


if __name__ == "__main__":
    main()