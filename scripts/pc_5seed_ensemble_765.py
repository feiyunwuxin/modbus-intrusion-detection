#!/usr/bin/env python3
"""
pc_5seed_ensemble_765.py
========================
PC 端 5-seed ensemble 跑 765 帧, 跟单 seed 模型对比, 验证 Option C 价值.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

CSV_PATH = r"C:\work\Claude\Issue\IanArffDatasetF32.csv"
SCALER_PATH = r"C:\work\Claude\Issue\scaler_binary_v2_scada.joblib"
SEEDS = [42, 123, 456, 789, 1024]
MODEL_PATHS = [f"C:/work/Claude/Issue/model_v4_se_23dim_b64_ch32_do01_window16_s{s}.pt" for s in SEEDS]
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

    # 5 seeds
    all_probs = []
    for s, path in zip(SEEDS, MODEL_PATHS):
        model = TCNClassifierSE(in_ch=23, n_blocks=3, channels=32, kernel_size=3,
                                 dilations=(1, 2, 4), dropout=0.1)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).float()
            probs = torch.sigmoid(model(xb)).squeeze(-1).numpy()
        all_probs.append(probs)
        print(f"  seed={s}: probs range=[{probs.min():.3f}, {probs.max():.3f}]")
    # 5-seed ensemble (mean of probs)
    probs_ens = np.mean(all_probs, axis=0)
    print(f"\nEnsemble: probs range=[{probs_ens.min():.3f}, {probs_ens.max():.3f}]")

    csv_binary = df_765["binary result"].values
    n_attack = int(csv_binary.sum())

    # === thr sweep for ensemble ===
    print(f"\n=== 5-seed Ensemble thr sweep (n_attack={n_attack}) ===")
    print(f"{'thr':<6} {'TP':<5} {'FP':<5} {'FN':<5} {'Recall':<8} {'Precision':<10} {'F1':<8} {'FPR':<8}")
    for thr in [0.20, 0.30, 0.40, 0.49, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tp = fp = fn = 0
        for i, p in enumerate(probs_ens):
            frame_idx_1based = i + 16
            if frame_idx_1based > 765: continue
            binary = csv_binary[frame_idx_1based - 1]
            alarm = p >= thr
            if alarm and binary == 1: tp += 1
            elif alarm and binary == 0: fp += 1
            elif not alarm and binary == 1: fn += 1
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        fpr = fp / max(n_attack + 765 - n_attack - tp - fn, 1)
        print(f"{thr:<6.2f} {tp:<5} {fp:<5} {fn:<5} {recall:<8.4f} {precision:<10.4f} {f1:<8.4f} {fpr:<8.4f}")
    print()
    # === 对比: 单 seed s42 vs 5-seed ensemble thr=0.49 ===
    print("=== 对比总结 ===")
    print(f"  H743 实际 (thr=0.49, INT8 ch=32 s42):  F1=0.2642, Recall=0.2074, Precision=0.3636, FPR=0.0778")
    print(f"  PC FP32 ch=32 s42 (thr=0.49):           F1=0.5544, Recall=1.0000, Precision=0.3835, FPR=0.3528")
    print(f"  PC FP32 5-seed ens (thr=0.49):          F1=? (见上表)")
    print(f"  PC FP32 5-seed ens (thr=0.50):          F1=?")


if __name__ == "__main__":
    main()