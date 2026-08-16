#!/usr/bin/env python3
"""TCN+SE 64 配置组合删除实验 (2026-07-18)

用户要求: 对以下 6 个目标特征的所有子集删除组合, 验证模型性能:
  function (idx 1), setpoint (idx 3), cmd_count_w (idx 22),
  crc_mean_w (idx 20), press_mean_w (idx 19), length (idx 2)

完整组合:
  K=0: 1   baseline
  K=1: 6   删除 1 个
  K=2: 15  删除 2 个
  K=3: 20  删除 3 个
  K=4: 15  删除 4 个
  K=5: 6   删除 5 个
  K=6: 1   删除 6 个
  Total: 64 configs

每档 × 5 seeds = 320 trainings

依据 [[project-tcn-v2-loo-5seed-correction]] 5-seed LOO 单点:
  idx 1 function:       -0.000401
  idx 3 setpoint:       +0.000266
  idx 22 cmd_count_w:   +0.000563
  idx 20 crc_mean_w:    +0.001035
  idx 19 press_mean_w:  +0.003517
  idx 2 length:         +0.005134

设置 (历史 5-seed LOO baseline 一致):
  B=512, LR=5e-4, WD=1e-5, Dropout=0.3, Epoch=20, Patience=5
  ch=64, k=3, dilations=[1,2,4], SE r=8, window=16
  seeds = [42, 123, 456, 789, 1024]
"""

import os, time, json, csv, sys, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BASE = r"C:\work\Claude\Issue"
EPOCHS = 20; BATCH = 512; LR = 5e-4; WD = 1e-5; PATIENCE = 5
GRAD_CLIP = 0.5; DROPOUT = 0.3; CLIP_VAL = 10.0
N_BLOCKS = 3; CHANNELS = 64; KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]; SE_REDUCTION = 8; WINDOW = 16
SEEDS = [42, 123, 456, 789, 1024]

FEATURE_NAMES_27 = [
    "address","function","length","setpoint","gain","reset rate","deadband",
    "cycle time","rate","system mode","control scheme","pump","solenoid",
    "pressure measurement","crc rate","time_diff","time_since_last_same_addr_func",
    "is_unusual_fc","is_response","press_mean_w","crc_mean_w","crc_max_w",
    "cmd_count_w","resp_count_w","cmd_resp_balance_w","length_nunique_w","unusual_count_w"
]

# 6 目标特征 (idx, name)
TARGET_FEATURES = {
    1: "function",
    3: "setpoint",
    22: "cmd_count_w",
    20: "crc_mean_w",
    19: "press_mean_w",
    2: "length",
}

# 5-seed LOO 单点 ΔF1m (依据 ablation_tcn_v2_loo_5seed)
LOO_5SEED_DELTA = {
    1: -0.000401,    # function
    3: +0.000266,    # setpoint
    22: +0.000563,   # cmd_count_w
    20: +0.001035,   # crc_mean_w
    19: +0.003517,   # press_mean_w
    2: +0.005134,    # length
}


# ====== 模型定义 (复用 train_tcn_22dim.py 完全一致) ======
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
    def forward(self, x):
        s = self.gap(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class TCNBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, se_reduction=SE_REDUCTION):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=se_reduction)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return F.relu(x + residual)


class TCNClassifierSE(nn.Module):
    def __init__(self, in_ch, n_blocks=N_BLOCKS, channels=CHANNELS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
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
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc2(x).squeeze(-1)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb)).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_one(keep_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X_train = np.load(os.path.join(BASE, "X_train_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    X_val   = np.load(os.path.join(BASE, "X_val_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    X_test  = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)[:, :, keep_idx]
    y_train = np.load(os.path.join(BASE, "y_train_binary_v2_scada_window16.npy")).astype(np.int64)
    y_val   = np.load(os.path.join(BASE, "y_val_binary_v2_scada_window16.npy")).astype(np.int64)
    y_test  = np.load(os.path.join(BASE, "y_test_binary_v2_scada_window16.npy")).astype(np.int64)
    X_train = np.clip(X_train, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_val   = np.clip(X_val, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    X_test  = np.clip(X_test, -CLIP_VAL, CLIP_VAL).transpose(0, 2, 1)
    N_FEATURES = X_train.shape[1]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                              batch_size=BATCH, shuffle=False, num_workers=0)
    model = TCNClassifierSE(in_ch=N_FEATURES)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_f1m, best_state, best_epoch = -1, None, -1
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        val_prob, val_lbl = predict_probs(model, val_loader)
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1m = f1_score(val_lbl, val_pred, average="macro")
        scheduler.step(val_f1m)
        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break
    model.load_state_dict(best_state)
    test_prob, test_lbl = predict_probs(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)
    return {
        "n_features": N_FEATURES, "n_params": n_params, "best_epoch": best_epoch,
        "test_macro_f1": float(f1_score(test_lbl, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(test_lbl, test_pred, average="binary")),
        "test_accuracy": float((test_pred == test_lbl).mean()),
        "test_roc_auc": float(roc_auc_score(test_lbl, test_prob)),
        "test_pr_auc": float(average_precision_score(test_lbl, test_prob)),
        "val_macro_f1": float(val_f1m),
    }


# ====== 枚举所有组合 ======
def make_combo_name(k, drop_set):
    drop_features = sorted([TARGET_FEATURES[i] for i in drop_set])
    if not drop_features:
        return "27dim_baseline_NONE"
    return f"{27-k}dim_drop_{'_'.join(drop_features)}"


def enumerate_all_combos():
    """K=0..6, C(6,K) combos in K-order, K=0 first"""
    combos = []
    target_ids = sorted(TARGET_FEATURES.keys())
    for k in range(0, 7):
        for combo in itertools.combinations(target_ids, k):
            drop_set = frozenset(combo)
            name = make_combo_name(k, drop_set)
            keep_idx = [i for i in range(27) if i not in drop_set]
            combos.append({
                "k": k, "n_features": 27-k, "drop_set": drop_set,
                "drop_features": [TARGET_FEATURES[i] for i in sorted(drop_set)],
                "keep_idx": keep_idx, "name": name,
            })
    return combos


# ====== 主循环 ======
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:8.1f}s] {msg}", flush=True)


def main():
    # 命令行参数: 可传 subset (例如 "5" 跑 K=5, "1,2,3" 跑 K=1,2,3)
    target_ks = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            target_ks = None
        else:
            target_ks = set(int(x) for x in sys.argv[1].split(","))

    all_combos = enumerate_all_combos()
    if target_ks is not None:
        all_combos = [c for c in all_combos if c["k"] in target_ks]
    log(f"TCN+SE 64-config combination deletion: {len(all_combos)} configs × {len(SEEDS)} seeds = {len(all_combos)*len(SEEDS)} runs")
    log(f"K target: {sorted(target_ks) if target_ks else 'all (0..6)'}")
    log(f"Target features (idx, name): {TARGET_FEATURES}")
    log(f"LOO_5SEED_DELTA: {LOO_5SEED_DELTA}")

    # 输出文件
    partial_json = os.path.join(BASE, "train_tcn_combos_64_partial.json")
    final_json = os.path.join(BASE, "train_tcn_combos_64_results.json")
    final_csv = os.path.join(BASE, "train_tcn_combos_64_results.csv")

    results_by_name = {}  # 累积部分结果, 中断可恢复

    # 检查已有部分结果
    if os.path.exists(partial_json):
        with open(partial_json) as f:
            saved = json.load(f)
        if "results_by_name" in saved:
            results_by_name = saved["results_by_name"]
            log(f"Loaded partial: {sum(len(v) for v in results_by_name.values())} configs done")

    total_runs = len(all_combos) * len(SEEDS)
    run_idx = 0
    for c in all_combos:
        k = c["k"]
        name = c["name"]
        keep_idx = c["keep_idx"]

        if name in results_by_name and len(results_by_name[name]) >= len(SEEDS):
            log(f"  [SKIP] {name} already done")
            run_idx += len(SEEDS)
            continue

        seeds_results = results_by_name.get(name, [])
        for seed in SEEDS:
            # 检查这个 (name, seed) 是否已完成
            already = any(r["seed"] == seed for r in seeds_results)
            if already:
                run_idx += 1
                continue

            run_idx += 1
            log(f"  [{run_idx}/{total_runs}] {name} seed={seed}")
            t_one = time.time()
            try:
                r = run_one(keep_idx, seed)
                r["seed"] = seed
                seeds_results.append(r)
                elapsed = time.time() - t_one
                log(f"    F1m={r['test_macro_f1']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  "
                    f"Epoch={r['best_epoch']}  Time={elapsed:.1f}s  CumulTime={(time.time()-t0)/60:.1f}min")
            except Exception as e:
                log(f"    ERROR: {e}")
                continue

            # 每 1 个 run 保存一次 (防止长程中断)
            results_by_name[name] = seeds_results
            with open(partial_json, "w") as f:
                json.dump({"results_by_name": results_by_name,
                           "combo_specs": all_combos}, f, indent=2, default=str)

    log(f"\n全部 training 完成, 聚合统计中...")

    # ====== 聚合 ======
    baseline_f1m = None
    aggregated = []
    for c in all_combos:
        name = c["name"]
        k = c["k"]
        n_feat = c["n_features"]
        drop_set = c["drop_set"]
        if name not in results_by_name:
            continue
        rs = results_by_name[name]
        if not rs:
            continue
        f1ms = np.array([r["test_macro_f1"] for r in rs])
        prs = np.array([r["test_pr_auc"] for r in rs])
        bfs = np.array([r["test_binary_f1"] for r in rs])
        accs = np.array([r["test_accuracy"] for r in rs])
        rocs = np.array([r["test_roc_auc"] for r in rs])
        eps = np.array([r["best_epoch"] for r in rs])
        n_collapsed = int((eps < 5).sum())

        # LOO 加性预测 = sum(LOO_5SEED_DELTA[i] for i in drop_set)
        loo_pred = sum(LOO_5SEED_DELTA[i] for i in drop_set)

        agg = {
            "name": name, "k": k, "n_features": n_feat,
            "drop_set": sorted(drop_set),
            "drop_features": c["drop_features"],
            "drop_idx_set": sorted(drop_set),
            "n_seeds": len(rs),
            "n_params": rs[0]["n_params"],
            "seeds": rs,
            "mean_f1m": float(f1ms.mean()),
            "std_f1m": float(f1ms.std(ddof=0)),
            "median_f1m": float(np.median(f1ms)),
            "min_f1m": float(f1ms.min()),
            "max_f1m": float(f1ms.max()),
            "mean_pr_auc": float(prs.mean()),
            "std_pr_auc": float(prs.std(ddof=0)),
            "mean_binary_f1": float(bfs.mean()),
            "mean_accuracy": float(accs.mean()),
            "mean_roc_auc": float(rocs.mean()),
            "n_collapsed_seeds": n_collapsed,
            "median_best_epoch": int(np.median(eps)),
            "loo_cumulative_predicted_delta_f1m": loo_pred,
        }
        aggregated.append(agg)
        if k == 0:
            baseline_f1m = agg["mean_f1m"]

    # baseline 必须为 K=0
    if baseline_f1m is None:
        log("ERROR: baseline not found")
        return
    log(f"Baseline (K=0, 27-dim) mean F1m = {baseline_f1m:.4f}")

    # 加 delta_f1m_vs_27
    for a in aggregated:
        a["delta_f1m_vs_27baseline"] = a["mean_f1m"] - baseline_f1m
        a["actual_minus_loo_overestimate"] = a["delta_f1m_vs_27baseline"] - a["loo_cumulative_predicted_delta_f1m"]

    # ====== 保存 final JSON + CSV ======
    with open(final_json, "w") as f:
        json.dump({
            "baseline_27_mean_f1m": baseline_f1m,
            "loo_5seed_deltas": LOO_5SEED_DELTA,
            "target_features": TARGET_FEATURES,
            "configs": aggregated,
        }, f, indent=2, default=str)
    log(f"Saved: {final_json}")

    # CSV
    with open(final_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "n_features", "name", "drop_set", "drop_idx_set", "drop_features",
                     "n_seeds", "n_params", "n_collapsed_seeds", "median_best_epoch",
                     "mean_f1m", "std_f1m", "median_f1m", "min_f1m", "max_f1m",
                     "mean_pr_auc", "std_pr_auc", "mean_binary_f1", "mean_accuracy", "mean_roc_auc",
                     "loo_cumulative_predicted_delta_f1m",
                     "delta_f1m_vs_27baseline", "actual_minus_loo_overestimate"])
        for a in sorted(aggregated, key=lambda x: (x["k"], -x["mean_f1m"])):
            wr.writerow([
                a["k"], a["n_features"], a["name"],
                ";".join(str(i) for i in a["drop_set"]),
                ";".join(str(i) for i in a["drop_set"]),
                ";".join(a["drop_features"]),
                a["n_seeds"], a["n_params"], a["n_collapsed_seeds"], a["median_best_epoch"],
                a["mean_f1m"], a["std_f1m"], a["median_f1m"], a["min_f1m"], a["max_f1m"],
                a["mean_pr_auc"], a["std_pr_auc"], a["mean_binary_f1"], a["mean_accuracy"], a["mean_roc_auc"],
                a["loo_cumulative_predicted_delta_f1m"],
                a["delta_f1m_vs_27baseline"], a["actual_minus_loo_overestimate"],
            ])
    log(f"Saved: {final_csv}")

    # ====== 输出表格到 stdout ======
    print("\n" + "=" * 160)
    print(f"=== 64 配置组合删除对比表 (baseline 27-dim mean F1m = {baseline_f1m:.4f}) ===")
    print("=" * 160)
    header = (f"{'K':>2} {'n':>3} {'drop_features':<55} {'F1m_mean':>9} {'±std':>7} "
              f"{'median':>7} {'PR_mean':>8} {'ΔF1m':>9} {'LOO_pred':>9} {'overest':>9} "
              f"{'col':>4} {'name':<35}")
    print(header)
    print("-" * 160)
    sorted_aggs = sorted(aggregated, key=lambda x: (x["k"], -x["mean_f1m"]))
    for a in sorted_aggs:
        drops_str = '+'.join(f.split('_')[0] for f in a["drop_features"]) if a["drop_features"] else "NONE"
        # signed format (manual: + sign for positive, - already)
        df = a["delta_f1m_vs_27baseline"]
        lp = a["loo_cumulative_predicted_delta_f1m"]
        ov = a["actual_minus_loo_overestimate"]
        df_s = f"{df:+.4f}"
        lp_s = f"{lp:+.4f}"
        ov_s = f"{ov:+.4f}"
        print(f"{a['k']:>2} {a['n_features']:>3} "
              f"{drops_str:<55} "
              f"{a['mean_f1m']:>9.4f} {a['std_f1m']:>7.4f} {a['median_f1m']:>7.4f} "
              f"{a['mean_pr_auc']:>8.4f} "
              f"{df_s:>9} "
              f"{lp_s:>9} "
              f"{ov_s:>9} "
              f"{a['n_collapsed_seeds']:>4} "
              f"{a['name']:<35}")
    print("=" * 160)

    log(f"\nTotal time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
