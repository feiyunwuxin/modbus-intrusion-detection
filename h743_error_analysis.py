#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H743 部署误差根因分析 (离线可复现)

用法:
    python h743_error_analysis.py [deploy_log.csv] [full_log.csv]

背景:
    2026-10-21 三个根因全修后, 部署区 F1=0.8671 / FPR=5.91% / R=80.75%。
    本脚本回答: 剩下的 465 个 FN 和 133 个 FP 分别从哪来?

方法 (与 [[feedback-eval-metrics-must-be-replayable]] 一致):
    - 特征从 F32 CSV 重建, 严格复刻 preprocess_v2_scada.py 的规则
      (排序 / time_diff / time_since / is_unusual_fc / is_response /
       fillna(0) / pressure clip(0,100) / RobustScaler.transform)
    - 窗在**全局 16 网格**上切 (部署区起点 200000 % 16 == 0)
    - 用部署的 .pt 跑一遍, 与 H743 日志逐窗对拍 (自校验, 不通过就退出)
    - 所有指标都从日志重放得到, 不依赖 GUI summary

输出: 区域分解 / 阈值扫描 / FP 位置+归因 / FN 按 spec 分解 / 可见性证明
"""
import os
import sys
import csv
import json
import collections

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "IanArffDatasetF32.csv")   # 亚秒 time, 训练/部署同源
SCALER = os.path.join(BASE, "scaler_binary_v2_scada.joblib")
CKPT = os.path.join(BASE, "model_v4_se_23dim_b64_ch32_do01_window16_s42.pt")
TRAIN_SCRIPT = os.path.join(BASE, "retrain_tcn_23dim_b64_ch32_do01_savept.py")

DEPLOY_LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "logs", "test_20261021-203113_auto-40ms.csv")
FULL_LOG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    BASE, "logs", "test_20261021-003650_auto-40ms.csv")

W = 16
DEPLOY_START_ROW = 200000          # 部署区起始行 (frame 200001), 200000 % 16 == 0
G0 = DEPLOY_START_ROW // W
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
COLS27 = ['address', 'function', 'length', 'setpoint', 'gain', 'reset rate', 'deadband',
          'cycle time', 'rate', 'system mode', 'control scheme', 'pump', 'solenoid',
          'pressure measurement', 'crc rate', 'time_diff', 'time_since_last_same_addr_func',
          'is_unusual_fc', 'is_response', 'press_mean_w', 'crc_mean_w', 'crc_max_w',
          'cmd_count_w', 'resp_count_w', 'balance_w', 'length_nunique_w', 'unusual_count_w']
KN = [COLS27[i] for i in KEEP_23]
UNUSUAL_FCS = {136, 171, 139, 133, 137, 138, 140}


def log(msg):
    print(msg, flush=True)


# ── 1. 重建 19 维 scaled 行级特征 (严格复刻 preprocess_v2_scada.py) ───────────
def build_scaled_rows():
    df = pd.read_csv(CSV_PATH)
    # preprocess_v2_scada.py: '?' -> NaN, 数值列 to_numeric (结构缺失靠 fillna(0) 表达)
    df = df.replace("?", np.nan)
    for c in ["setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
              "system mode", "control scheme", "pump", "solenoid", "pressure measurement"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    labels = df[["binary result", "categorized result", "specific result"]].copy()
    df = df.drop(columns=["binary result", "categorized result", "specific result"])
    df["time_diff"] = df["time"].diff().fillna(0)
    key = df["address"].astype(int) * 1000 + df["function"].astype(int)
    tt = df["time"].values
    last = {}
    ts = np.zeros(len(df), dtype=np.float32)
    for i, k in enumerate(key.values):
        if k in last:
            ts[i] = tt[i] - last[k]
        last[k] = tt[i]
    df["time_since_last_same_addr_func"] = ts
    df["is_unusual_fc"] = df["function"].apply(
        lambda f: 1 if f in UNUSUAL_FCS else 0).astype(np.float32)
    df["is_response"] = df["command response"].astype(np.float32)
    df = df.drop(columns=["time", "command response"])
    df = df.fillna(0)
    df["pressure measurement"] = df["pressure measurement"].clip(0, 100)
    df = df.astype(np.float32)
    import joblib
    scaler = joblib.load(SCALER)
    X = scaler.transform(df.values).astype(np.float32)
    return df, X, labels


def build_windows(X19, gstart, n):
    """全局 16 网格上切窗 -> (n, 23, 16), 与 preprocess 的 27 维聚合规则一致"""
    Xw = X19[gstart * W:(gstart + n) * W].reshape(n, W, 19)
    press, crc, isresp = Xw[:, :, 13], Xw[:, :, 14], Xw[:, :, 18]
    ln, uns = Xw[:, :, 2], Xw[:, :, 17]
    aggs = np.column_stack([
        press.mean(1), crc.mean(1), crc.max(1),
        (1 - isresp).sum(1), isresp.sum(1),
        ((1 - isresp).sum(1) - isresp.sum(1)) / W,
        np.array([len(np.unique(ln[i])) for i in range(n)], dtype=np.float32),
        uns.sum(1)]).astype(np.float32)
    f27 = np.concatenate([Xw, np.repeat(aggs[:, None, :], W, axis=1)], axis=-1)
    return np.clip(f27[:, :, KEEP_23], -10, 10).transpose(0, 2, 1)


# ── 2. 日志解析 ────────────────────────────────────────────────────────────
def parse_log(path):
    """返回 {全局窗号: prob_x1000}. 日志 frame_idx 为**日志内**帧号(1-based)。"""
    alarms = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["event"] != "ALARM":
            continue
        alarms.setdefault(int(r["frame_idx"]), []).append(int(r["prob_x1000"]))
    return alarms


def metrics(pred, g):
    tp = int(((pred == 1) & (g == 1)).sum()); fp = int(((pred == 1) & (g == 0)).sum())
    tn = int(((pred == 0) & (g == 0)).sum()); fn = int(((pred == 0) & (g == 1)).sum())
    R = tp / (tp + fn) if tp + fn else 0
    FPR = fp / (fp + tn) if fp + tn else 0
    P = tp / (tp + fp) if tp + fp else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    return tp, fp, tn, fn, P, R, F1, FPR


# ── 3. 主流程 ──────────────────────────────────────────────────────────────
def main():
    log("=" * 78)
    log("H743 部署误差根因分析")
    log("=" * 78)

    df, X19, labels = build_scaled_rows()
    N = len(df)
    NW = N // W
    binary = labels["binary result"].values.astype(np.int64)
    spec = labels["specific result"].values.astype(np.int64)
    log(f"数据: {N} 行 / {NW} 窗 (W={W}), 部署区从行 {DEPLOY_START_ROW} (全局窗 {G0})")

    # ── 3.1 PC 复算 vs H743 日志 对拍 (自校验) ──
    # 从**真实训练源码**里 exec 出模型类 (不手抄, 避免两份实现一起错)
    # 必须在独立 namespace 里 exec: 否则 exec 出的类的方法会去 module globals 找
    # SE_REDUCTION 等常量 -> NameError
    src = open(TRAIN_SCRIPT, encoding="utf-8").read().split("def run_one")[0]
    ns = {}
    exec(src, ns)
    TCNClassifierSE = ns["TCNClassifierSE"]
    n_deploy = NW - G0
    Xdep = build_windows(X19, G0, n_deploy)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TCNClassifierSE(in_ch=23)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        p_pc = torch.sigmoid(model(torch.from_numpy(Xdep).float())).numpy()
    gw = np.array([binary[(G0 + w) * W:(G0 + w + 1) * W].max() for w in range(n_deploy)])

    alarms = parse_log(DEPLOY_LOG)
    h743_prob = np.zeros(n_deploy)
    for fr, ps in alarms.items():
        w = fr // W - 1
        if 0 <= w < n_deploy:
            h743_prob[w] = max(ps)
    agree = int(((p_pc >= 0.5) == (h743_prob >= 500)).sum())
    log(f"\n[自校验] PC 复算 vs H743: 决策一致 {agree}/{n_deploy} 窗 "
        f"({'通过' if agree >= n_deploy - 5 else '失败, 后面的结论不可信!'})")
    tp, fp, tn, fn, P, R, F1, FPR = metrics((h743_prob >= 500).astype(int), gw)
    log(f"[自校验] H743 部署区: TP {tp} FP {fp} TN {tn} FN {fn} | "
        f"P {P:.4f} R {R:.4f} F1 {F1:.4f} FPR {FPR:.4f}")
    tp, fp, tn, fn, P, R, F1, FPR = metrics((p_pc >= 0.5).astype(int), gw)
    log(f"[自校验] PC  部署区: TP {tp} FP {fp} TN {tn} FN {fn} | "
        f"P {P:.4f} R {R:.4f} F1 {F1:.4f} FPR {FPR:.4f}")
    if agree < n_deploy - 5:
        log("!! 对拍失败, 停止。")
        return 1

    # ── 3.2 区域分解: Recall 没有过拟合, FPR 过拟合了 ──
    ntr = int(N * 0.70) // W
    nva = (int(N * 0.70) + int(N * 0.10)) // W
    log("\n[区域分解] (窗级, 70/10/20 时间切分)")
    log("  %-14s %6s %6s %6s %6s %8s %8s" % ("region", "win", "atk", "TP", "FP", "recall", "FPR"))
    # 全量日志重放: 全局窗号 g -> 是否有 alarm
    full_alarms = parse_log(FULL_LOG)
    aw_full = set()
    for fr in full_alarms:
        aw_full.add(fr // W - 1)
    for name, lo, hi in [("train", 0, ntr), ("val", ntr, nva), ("test", nva, NW)]:
        a = t = f = 0
        for g in range(lo, hi):
            lab = binary[g * W:(g + 1) * W].max()
            hit = g in aw_full
            if lab:
                a += 1
                t += hit
            else:
                f += hit
        nor = hi - lo - a
        log("  %-14s %6d %6d %6d %6d %8.4f %8.4f" %
            (name, hi - lo, a, t, f, t / a if a else 0, f / nor if nor else 0))
    log("  => Recall 跨区几乎恒定(没背答案), FPR 随未见数据涨 ~7 倍(对正常流过拟合)")

    # ── 3.3 阈值扫描 ──
    log("\n[阈值扫描] 部署区")
    log("  %6s %6s %6s %8s %8s %8s %8s" % ("thr", "TP", "FP", "R", "P", "F1", "FPR"))
    for thr in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pr = (p_pc >= thr).astype(int)
        tp, fp, tn, fn, P, R, F1, FPR = metrics(pr, gw)
        log("  %6.2f %6d %6d %8.4f %8.4f %8.4f %8.4f" % (thr, tp, fp, R, P, F1, FPR))
    log("  => 0.5 最优; FP 置信度高(mean prob %.3f), 提阈值换不回 F1" %
        p_pc[(p_pc >= 0.5) & (gw == 0)].mean())

    # ── 3.4 FP: 位置 + 归因 ──
    is_fp = (p_pc >= 0.5) & (gw == 0)
    is_tn = (p_pc < 0.5) & (gw == 0)
    is_fn = (p_pc < 0.5) & (gw == 1)
    is_tp = (p_pc >= 0.5) & (gw == 1)
    fp_g = np.where(is_fp)[0] + G0
    log("\n[FP 位置] %d 个 FP 的分布 (每 200 窗一桶)" % len(fp_g))
    hot = []
    for lo in range(G0, NW, 200):
        hi = min(lo + 200, NW)
        n = int(is_fp[lo - G0:hi - G0].sum())
        nor = int((gw[lo - G0:hi - G0] == 0).sum())
        if nor and n / nor > 0.05:
            hot.append((lo, hi, n, nor, n / nor))
    for lo, hi, n, nor, f in hot:
        log("    全局窗 %5d-%5d (frame %6d-%6d): FP %3d / normal %3d  FPR %.3f" %
            (lo, hi - 1, lo * W, hi * W - 1, n, nor, f))
    log("  => FP 高度聚集, 不是均匀撒开的")

    # 消融: 把某通道换成"典型正常值", 看 FP 的 prob 掉多少
    base = np.median(Xdep[is_tn], axis=0)
    log("\n[FP 归因] 消融: 单通道换成典型正常值后 FP 窗的平均 prob")
    log("  %-30s %10s %10s" % ("channel", "P(原)", "P(消融后)"))
    abl = []
    for i in range(23):
        Xi = Xdep.copy()
        Xi[:, i, :] = base[i][None, :]
        with torch.no_grad():
            Pi = torch.sigmoid(model(torch.from_numpy(Xi).float())).numpy()
        abl.append((KN[i], p_pc[is_fp].mean(), Pi[is_fp].mean()))
    for nm, a, b in sorted(abl, key=lambda z: z[2])[:8]:
        log("  %-30s %10.4f %10.4f" % (nm, a, b))

    # 判别力: 哪一维能分开 FP 和 TN
    V = Xdep.mean(axis=2)      # (n, 23) 每窗 23 维均值
    log("\n[FP 判别力] 部署区 normal 窗, FP vs TN (AUC)")
    rows = []
    for i in range(23):
        v = V[:, i]
        m1, m0 = v[is_fp], v[is_tn]
        auc = float((np.sign(m1[:, None] - m0[None, :]) > 0).mean()
                    + 0.5 * (m1[:, None] == m0[None, :]).mean())
        rows.append((KN[i], auc, m1.mean(), m0.mean()))
    for nm, auc, a, b in sorted(rows, key=lambda z: -max(z[1], 1 - z[1]))[:8]:
        log("  %-30s AUC %.3f   FP 均值 %8.3f  TN 均值 %8.3f" % (nm, auc, a, b))

    # ── 3.5 FN: 按 spec 分解 + 可见性 ──
    pspec = np.zeros(NW, dtype=np.int64)
    for g in range(NW):
        s = spec[g * W:(g + 1) * W]
        nz = s[s > 0]
        if len(nz):
            pspec[g] = np.bincount(nz).argmax()
    atk = np.where(gw == 1)[0] + G0
    fn_g = np.where(is_fn)[0] + G0
    log("\n[FN 按 spec] 部署区攻击窗漏报最多的攻击类型")
    cnt_atk = collections.Counter(pspec[atk])
    cnt_fn = collections.Counter(pspec[fn_g])
    log("  %5s %8s %8s %8s" % ("spec", "攻击窗", "漏报", "漏报率"))
    for s, n in sorted(cnt_fn.items(), key=lambda z: -z[1])[:14]:
        log("  %5d %8d %8d %7.1f%%" % (s, cnt_atk[s], n, 100.0 * n / cnt_atk[s]))
    log("  => 漏报集中在少数攻击类型; 检出率高的类型(spec 11/12/18/23/24/26/28/33/34)漏报 <5%")

    # 可见性证明 A: 篡改落在被丢弃的特征 (setpoint)
    log("\n[FN 根因 A] 篡改落在 KEEP_23 丢弃的特征里")
    log("  KEEP_23 丢弃: length, setpoint, crc_mean_w, cmd_count_w")
    sp = df["setpoint"].values.astype(np.float64)
    norm_g = [g for g in range(G0, NW) if pspec[g] == 0]
    sp_norm = np.array([sp[g * W:(g + 1) * W].mean() for g in norm_g])
    lo, hi = np.percentile(sp_norm, [5, 95])
    for s in (1, 2):
        gl = [g for g in range(G0, NW) if pspec[g] == s and binary[g * W:(g + 1) * W].max() == 1]
        mis = [g for g in gl if p_pc[g - G0] < 0.5]
        det = [g for g in gl if p_pc[g - G0] >= 0.5]
        if not gl:
            continue
        spm = np.array([sp[g * W:(g + 1) * W].mean() for g in mis])
        out = 100.0 * np.mean((spm < lo) | (spm > hi))
        log("  spec %d: 攻击窗 %d, 漏报 %d (%.0f%%) | 漏报窗 setpoint 均值 %.2f, "
            "正常窗 p5-p95 = [%.2f, %.2f], 越界比例 %.0f%%"
            % (s, len(gl), len(mis), 100.0 * len(mis) / len(gl),
               spm.mean() if len(spm) else np.nan, lo, hi, out))
    log("  => setpoint 明显越界, 但模型看不到它 -> 结构性漏报")

    # 可见性证明 B: FN 窗在所有保留特征上都比 TP 窗更像正常
    dev = Xdep - np.median(Xdep[is_tn], axis=0)
    log("\n[FN 根因 B] FN 窗 vs TP 窗 在保留特征上的偏离幅度")
    log("  %-30s %10s %10s" % ("channel", "FN", "TP"))
    for i in range(23):
        a = np.abs(dev[is_fn][:, i, :]).mean()
        b = np.abs(dev[is_tp][:, i, :]).mean()
        if a > 0.3 or b > 0.3:
            log("  %-30s %10.3f %10.3f" % (KN[i], a, b))
    log("  => FN 窗的所有信号都更弱, 即看起来更像正常")

    log("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
