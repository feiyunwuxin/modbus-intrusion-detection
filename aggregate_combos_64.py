#!/usr/bin/env python3
"""聚合 worker partials → final CSV/JSON + plots (单独调用)

读 5 个 worker 的 partial JSON, 合并 K=0 baseline (任何来路),
按 K 档分组, 输出 CSV, JSON, 3 matplotlib 图.
"""

import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\work\Claude\Issue"
sys.path.insert(0, BASE)
from train_tcn_combos_64 import TARGET_FEATURES, LOO_5SEED_DELTA, enumerate_all_combos


def load_all_results():
    all_results = {}
    # 5 个 worker partials
    for wid in range(5):
        p = os.path.join(BASE, f"train_tcn_combos_64_partial_w{wid}.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    r = json.load(f)
                for name, rs in r.items():
                    all_results.setdefault(name, []).extend(rs)
            except Exception as e:
                print(f"WARN w{wid}: {e}")
    # 老的 partial (e.g., K=0 baseline)
    op = os.path.join(BASE, "train_tcn_combos_64_partial.json")
    if os.path.exists(op):
        try:
            with open(op) as f:
                old = json.load(f)
            old_results = old.get("results_by_name", old)
            if isinstance(old_results, dict):
                for name, rs in old_results.items():
                    if isinstance(rs, list) and rs and isinstance(rs[0], dict) and "seed" in rs[0]:
                        already = {r["seed"] for r in all_results.get(name, [])}
                        for r in rs:
                            if r["seed"] not in already:
                                all_results.setdefault(name, []).append(r)
        except Exception as e:
            print(f"WARN old partial: {e}")
    return all_results


def aggregate(all_results, combos):
    aggregated = []
    baseline_f1m = None
    name_to_combo = {c["name"]: c for c in combos}
    for name, combo in name_to_combo.items():
        rs = all_results.get(name, [])
        if not rs:
            continue
        f1ms = np.array([r["test_macro_f1"] for r in rs])
        prs = np.array([r["test_pr_auc"] for r in rs])
        bfs = np.array([r["test_binary_f1"] for r in rs])
        accs = np.array([r["test_accuracy"] for r in rs])
        rocs = np.array([r["test_roc_auc"] for r in rs])
        eps = np.array([r["best_epoch"] for r in rs])
        n_collapsed = int((eps < 5).sum())
        loo_pred = sum(LOO_5SEED_DELTA[i] for i in combo["drop_set"])
        agg = {
            "name": name, "k": combo["k"], "n_features": combo["n_features"],
            "drop_set": sorted(combo["drop_set"]),
            "drop_features": combo["drop_features"],
            "drop_idx_set": sorted(combo["drop_set"]),
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
        if combo["k"] == 0:
            baseline_f1m = agg["mean_f1m"]
    return aggregated, baseline_f1m


def save_csv(aggregated, baseline_f1m, csv_path):
    with open(csv_path, "w", newline="") as f:
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


def make_scatter(aggregated, baseline_f1m, png_path):
    """7 维散点图: x=n_features, y=mean_f1m ± std"""
    fig, ax = plt.subplots(figsize=(12, 7))
    sorted_aggs = sorted(aggregated, key=lambda x: (x["k"], -x["mean_f1m"]))

    ks = np.array([a["k"] for a in sorted_aggs])
    f1ms = np.array([a["mean_f1m"] for a in sorted_aggs])
    stds = np.array([a["std_f1m"] for a in sorted_aggs])
    nfs = np.array([a["n_features"] for a in sorted_aggs])

    # color by K
    cmap = plt.cm.viridis(np.linspace(0, 1, 7))
    for k_val in range(0, 7):
        mask = (ks == k_val)
        if not mask.any():
            continue
        ax.errorbar(nfs[mask], f1ms[mask], yerr=stds[mask],
                    fmt='o', markersize=8, capsize=5,
                    color=cmap[k_val], label=f"K={k_val}")

    ax.axhline(y=baseline_f1m, color='red', linestyle='--', alpha=0.7,
               label=f"27 baseline (F1m={baseline_f1m:.4f})")
    ax.set_xlabel("n_features")
    ax.set_ylabel("mean F1m (5-seed)")
    ax.set_title("64-config Combination Deletion: TCN+SE F1m vs n_features")
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close()


def make_loo_scatter(aggregated, baseline_f1m, png_path):
    """LOO 加性预测 vs 实测 散点图"""
    fig, ax = plt.subplots(figsize=(10, 10))
    loo_pred = np.array([a["loo_cumulative_predicted_delta_f1m"] for a in aggregated])
    actual_delta = np.array([a["mean_f1m"] - baseline_f1m for a in aggregated])
    ks = np.array([a["k"] for a in aggregated])

    cmap = plt.cm.viridis(np.linspace(0, 1, 7))
    for k_val in range(0, 7):
        mask = (ks == k_val)
        if not mask.any():
            continue
        ax.scatter(loo_pred[mask], actual_delta[mask],
                   s=60, alpha=0.7, color=cmap[k_val],
                   edgecolor='black', linewidth=0.5,
                   label=f"K={k_val} (n={mask.sum()})")

    # y=x line
    lo = min(loo_pred.min(), actual_delta.min()) - 0.01
    hi = max(loo_pred.max(), actual_delta.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label="y=x (perfect prediction)")
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("LOO additive prediction: ΔF1m (5-seed)")
    ax.set_ylabel("Actual: ΔF1m (mean 5-seed)")
    ax.set_title("LOO additive prediction vs Actual")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close()


def make_heatmap(aggregated, png_path):
    """6 特征 pairwise 相关 heatmap

    行=6 个目标特征 (删除), 列=6 个目标特征 (删除)
    cell = mean F1m over all configs that delete BOTH i and j
    """
    target_ids = sorted(TARGET_FEATURES.keys())
    n = len(target_ids)
    matrix = np.full((n, n), np.nan)

    for i_idx, idx_i in enumerate(target_ids):
        for j_idx, idx_j in enumerate(target_ids):
            if i_idx == j_idx:
                # diagonal: configs that delete only one
                # K=1 configs that delete idx_i
                rs = [a for a in aggregated if a["k"] == 1 and idx_i in a["drop_set"]]
                if rs:
                    matrix[i_idx, j_idx] = np.mean([a["mean_f1m"] for a in rs])
                continue
            # off-diagonal: K=2 configs that delete BOTH idx_i and idx_j
            rs = [a for a in aggregated if a["k"] == 2 and idx_i in a["drop_set"] and idx_j in a["drop_set"]]
            if rs:
                matrix[i_idx, j_idx] = np.mean([a["mean_f1m"] for a in rs])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.78, vmax=0.84, aspect='auto')
    plt.colorbar(im, ax=ax, label='mean F1m')

    target_names = [TARGET_FEATURES[idx] for idx in target_ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.set_yticklabels(target_names)

    # cell annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                txt = f"{matrix[i, j]:.4f}"
                color = 'black' if matrix[i, j] > 0.81 else 'white'
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=8)

    ax.set_xlabel("Also drop feature (column)")
    ax.set_ylabel("Drop feature (row)")
    ax.set_title("Pairwise Deletion Heatmap (cell = mean F1m)\n"
                 "Diagonal: delete only row feature (K=1) | Off-diagonal: delete BOTH (K=2)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close()


def main():
    combos = enumerate_all_combos()
    all_results = load_all_results()
    n_runs = sum(len(v) for v in all_results.values())
    n_configs_done = len(all_results)
    print(f"Loaded {n_runs} runs across {n_configs_done} configs")

    aggregated, baseline_f1m = aggregate(all_results, combos)
    print(f"Aggregated {len(aggregated)} configs")
    if baseline_f1m is None:
        print("WARNING: baseline (K=0) not found")
        baseline_f1m = 0.8205
    print(f"Baseline (K=0): F1m={baseline_f1m:.4f}")

    # 加 deltas
    for a in aggregated:
        a["delta_f1m_vs_27baseline"] = a["mean_f1m"] - baseline_f1m
        a["actual_minus_loo_overestimate"] = a["delta_f1m_vs_27baseline"] - a["loo_cumulative_predicted_delta_f1m"]

    final_json = os.path.join(BASE, "train_tcn_combos_64_results.json")
    final_csv = os.path.join(BASE, "train_tcn_combos_64_results.csv")
    with open(final_json, "w") as f:
        json.dump({
            "baseline_27_mean_f1m": baseline_f1m,
            "loo_5seed_deltas": LOO_5SEED_DELTA,
            "target_features": TARGET_FEATURES,
            "configs": aggregated,
        }, f, indent=2, default=str)
    save_csv(aggregated, baseline_f1m, final_csv)
    print(f"Saved: {final_json}")
    print(f"Saved: {final_csv}")

    # plots
    make_scatter(aggregated, baseline_f1m, os.path.join(BASE, "combos_64_scatter.png"))
    make_loo_scatter(aggregated, baseline_f1m, os.path.join(BASE, "combos_64_loo_scatter.png"))
    make_heatmap(aggregated, os.path.join(BASE, "combos_64_heatmap.png"))

    # 输出表格
    print("\n=== 64 配置聚合 (baseline F1m={:.4f}) ===".format(baseline_f1m))
    sorted_aggs = sorted(aggregated, key=lambda x: (x["k"], -x["mean_f1m"]))
    header = "{:>2} {:>3} {:<48} {:>9} {:>7} {:>7} {:>8} {:>9} {:>9} {:>9} {:>3} {:>7}".format(
        "K", "n", "drops", "F1m", "std", "med", "PR", "dF1m", "LOO_pr", "over", "col", "n_sds")
    print(header)
    for a in sorted_aggs:
        drops_str = '+'.join(f.split('_')[0] for f in a["drop_features"]) if a["drop_features"] else "NONE"
        ov = a["actual_minus_loo_overestimate"]
        df = a["delta_f1m_vs_27baseline"]
        loo = a["loo_cumulative_predicted_delta_f1m"]
        print("{:>2} {:>3} {:<48} {:>9.4f} {:>7.4f} {:>7.4f} {:>8.4f} {:>+9.4f} {:>+9.4f} {:>+9.4f} {:>3} {:>7}".format(
            a["k"], a["n_features"], drops_str,
            a["mean_f1m"], a["std_f1m"], a["median_f1m"], a["mean_pr_auc"],
            df, loo, ov,
            a["n_collapsed_seeds"], a["n_seeds"]))


if __name__ == "__main__":
    main()
