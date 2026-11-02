#!/usr/bin/env python3
"""
h743_thr_sweep_pc_baseline.py
==============================
基于 PC FP32 推理 765 帧, 在不同 thr 下算 TP/FP/Recall/Precision/F1.
然后用 H743 prob 跟 PC prob 的 delta (-0.05 ~ -0.5) 估计 H743 端的预期 KPI.

H743 当前 thr=0.49 太严, 因为 INT8 量化让 prob 整体偏低 0.05-0.5.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import sys

CSV_PATH = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"
SCALER_PATH = r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib"
MODEL_PATH = r"C:\work\Claude\Issue\model_v4_se_23dim_b64_ch32_do01_window16_s42.pt"

CLIP_VAL = 10.0
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}


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


def build_18(df_765):
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
    return df[keep].astype(np.float32).values


def main():
    df_765 = pd.read_csv(CSV_PATH, nrows=765)
    raw18 = build_18(df_765)
    scaler = joblib.load(SCALER_PATH)
    KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    mu_18 = scaler.center_[KEEP_18].astype(np.float32)
    sg_18 = scaler.scale_[KEEP_18].astype(np.float32)
    norm18 = (raw18 - mu_18) / sg_18

    SW_WINDOW = 16
    KEEP_18_IDX = KEEP_18
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
            frame27[:, KEEP_18_IDX[c18]] = win[:, c18]
        frame27[:, 19] = press_mean
        frame27[:, 20] = crc_mean
        frame27[:, 21] = crc_max
        frame27[:, 22] = cmd_count
        frame27[:, 23] = resp_signed
        frame27[:, 24] = balance
        frame27[:, 25] = n
        frame27[:, 26] = unusual_sign
        sliced = frame27[:, KEEP_23]
        sliced = np.clip(sliced, -CLIP_VAL, CLIP_VAL)
        out.append(sliced.T.astype(np.float32))
    X = np.stack(out, axis=0)

    model = TCNClassifierSE(in_ch=23, n_blocks=3, channels=32, kernel_size=3,
                             dilations=(1, 2, 4), dropout=0.1)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X).float()
        probs = torch.sigmoid(model(xb)).squeeze(-1).numpy()

    # === thr sweep ===
    csv_binary = df_765["binary result"].values
    n_attack = int(csv_binary.sum())
    n_normal = 765 - n_attack
    print(f"=== {n_attack} attack frames, {n_normal} normal frames ===")
    print()
    print(f"{'thr':<6} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5} {'Recall':<8} {'Precision':<10} {'F1':<8} {'FPR':<8}")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49, 0.50, 0.55, 0.60, 0.70]:
        tp = fp = fn = tn = 0
        for i, p in enumerate(probs):
            frame_idx_1based = i + 16
            if frame_idx_1based > 765:
                continue
            binary = csv_binary[frame_idx_1based - 1]
            alarm = p >= thr
            if alarm and binary == 1: tp += 1
            elif alarm and binary == 0: fp += 1
            elif not alarm and binary == 1: fn += 1
            else: tn += 1
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        fpr = fp / max(fp + tn, 1)
        print(f"{thr:<6.2f} {tp:<5} {fp:<5} {fn:<5} {tn:<5} {recall:<8.4f} {precision:<10.4f} {f1:<8.4f} {fpr:<8.4f}")

    # === H743 thr=0.49 实际值 ===
    h743_tp_probs = {63: 0.621, 67: 0.731, 71: 0.843, 75: 0.870, 79: 0.868, 83: 0.871, 87: 0.872, 91: 0.844, 95: 0.844, 99: 0.855, 103: 0.852, 107: 0.852, 111: 0.849, 115: 0.862, 119: 0.860, 123: 0.861, 127: 0.860, 131: 0.861, 135: 0.855, 139: 0.908, 143: 0.934, 147: 0.949, 151: 0.963, 155: 0.959, 159: 0.965, 163: 0.971, 198: 0.681, 238: 0.505}
    h743_fp_probs = {167: 0.983, 171: 0.984, 175: 0.984, 179: 0.950, 183: 0.937, 187: 0.949, 191: 0.947, 195: 0.956, 231: 0.965, 235: 0.963, 241: 0.858, 245: 0.848, 249: 0.830, 253: 0.798, 255: 0.511, 523: 0.601, 527: 0.847, 531: 0.951, 535: 0.972, 539: 0.966, 543: 0.969, 547: 0.968, 551: 0.977, 555: 0.968, 559: 0.978, 563: 0.974, 567: 0.980, 571: 0.980, 575: 0.981, 579: 0.981, 583: 0.975, 587: 0.969, 591: 0.969, 595: 0.971, 599: 0.976, 603: 0.976, 607: 0.969, 611: 0.976, 615: 0.966, 619: 0.973, 623: 0.965, 627: 0.974, 631: 0.972, 635: 0.941, 639: 0.781, 643: 0.543, 757: 0.585, 761: 0.798, 765: 0.819}
    print()
    print(f"=== H743 thr=0.49 实际值 (alarm log 抽 77 个 alarm) ===")
    print(f"  TP: {len(h743_tp_probs)}, FP: {len(h743_fp_probs)}, Recall: {len(h743_tp_probs)/135:.4f}, Precision: {len(h743_tp_probs)/(len(h743_tp_probs)+len(h743_fp_probs)):.4f}, F1: {2*len(h743_tp_probs)/(2*len(h743_tp_probs)+49+107):.4f}")
    # H743 frame 1-62 完全沉默, frame 47-62 PC 端 prob 都 > 0.49, 估算是 H743 INT8 drift ~0.4
    # 看 H743 在 thr=0.49 下没报的 frame (PC prob >= 0.49)
    missed_frames = []
    for i, p in enumerate(probs):
        frame_idx_1based = i + 16
        if frame_idx_1based > 765: continue
        binary = csv_binary[frame_idx_1based - 1]
        if p >= 0.49 and frame_idx_1based not in h743_tp_probs and frame_idx_1based not in h743_fp_probs:
            # 这个 frame PC 会报警, 但 H743 没报 (即使它属于 FP/FP, H743 也不报)
            missed_frames.append((frame_idx_1based, float(p), int(binary)))
    print(f"\n=== PC thr=0.49 会报警但 H743 没报的 frame ({len(missed_frames)} 个) ===")
    print(f"{'frame':<7} {'pc_prob':<10} {'binary':<8}")
    tp_missed = 0
    fp_missed = 0
    for f, p, b in missed_frames:
        kind = "TP_miss" if b == 1 else "FP_miss"
        if b == 1: tp_missed += 1
        else: fp_missed += 1
        if (tp_missed + fp_missed) <= 30:
            print(f"{f:<7} {p:<10.3f} {b:<8} {kind}")
    print(f"\n  TP missed (PC 报但 H743 没报): {tp_missed}")
    print(f"  FP missed (PC 报但 H743 没报): {fp_missed}")


if __name__ == "__main__":
    main()