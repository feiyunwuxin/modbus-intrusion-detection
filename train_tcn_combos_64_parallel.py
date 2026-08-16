#!/usr/bin/env python3
"""TCN+SE 64 配置并行加速器 (Windows subprocess 模式)

策略: 每个 subprocess 跑 (1 seed × 64 configs = 64 runs) 然后 exit.
  - 5 个 subprocess 同时跑, 充分利用多核 CPU.
  - 用 subprocess 启动 (不用 multiprocessing) 因为 Windows 上 multiprocessing 经常有问题.
"""

import os, sys, time, json
import argparse
import subprocess

BASE = r"C:\work\Claude\Issue"


def main():
    SEEDS = [42, 123, 456, 789, 1024]
    n_w = len(SEEDS)
    procs = []

    for wid in range(n_w):
        seed = SEEDS[wid % len(SEEDS)]
        seeds_str = f"[{seed}]"

        wid_path = os.path.join(BASE, f"_worker_{wid}.py")
        with open(wid_path, "w") as f:
            f.write(f'''#!/usr/bin/env python3
import os, sys, json, time
sys.path.insert(0, r"{BASE}")
from train_tcn_combos_64 import run_one, enumerate_all_combos

MY_SEEDS = {seeds_str}
WORKER_ID = {wid}
PARTIAL = r"{BASE}\\train_tcn_combos_64_partial_w{wid}.json"

results = {{}}
if os.path.exists(PARTIAL):
    try:
        with open(PARTIAL) as f:
            results = json.load(f)
    except:
        results = {{}}

combos = enumerate_all_combos()
total = len(combos) * len(MY_SEEDS)
done = 0
t0 = time.time()
for c in combos:
    name = c["name"]
    keep = c["keep_idx"]
    for seed in MY_SEEDS:
        if any(r.get("seed") == seed for r in results.get(name, [])):
            done += 1
            continue
        done += 1
        t1 = time.time()
        try:
            r = run_one(keep, seed)
            r["seed"] = seed
            results.setdefault(name, []).append(r)
            print(f"[w{{WORKER_ID}}] [{{done}}/{{total}}] {{name}} seed={{seed}} "
                  f"F1m={{r['test_macro_f1']:.4f}} PR={{r['test_pr_auc']:.4f}} "
                  f"Epoch={{r['best_epoch']}} T={{time.time()-t1:.1f}}s "
                  f"CumT={{(time.time()-t0)/60:.1f}}m", flush=True)
        except Exception as e:
            print(f"[w{{WORKER_ID}}] ERR {{name}} s={{seed}}: {{e}}", flush=True)
        with open(PARTIAL, "w") as f:
            json.dump(results, f, indent=2, default=str)

print(f"[w{{WORKER_ID}}] DONE {{done}} runs in {{(time.time()-t0)/60:.1f}} min")
''')

        log_path = os.path.join(BASE, f"_worker_{wid}.log")
        log_f = open(log_path, "w")
        p = subprocess.Popen(
            [sys.executable, wid_path],
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=BASE,
        )
        procs.append((wid, p, log_f))
        print(f"Launched worker {wid} (seed {seed}) pid={p.pid} log={log_path}")

    print(f"\nAll {n_w} workers launched. Waiting...")

    # monitor
    while True:
        time.sleep(60)
        all_done = True
        for wid, p, _ in procs:
            if p.poll() is None:
                all_done = False
        # show progress every 60 sec
        n_done = 0
        n_configs = 0
        for wid, _, _ in procs:
            p = os.path.join(BASE, f"train_tcn_combos_64_partial_w{wid}.json")
            if os.path.exists(p):
                with open(p) as f:
                    r = json.load(f)
                n_done += sum(len(v) for v in r.values())
                n_configs += len(r)
        elapsed = time.time() - procs[0][1].pid  # placeholder
        print(f"  [MON {time.strftime('%H:%M:%S')}] Total runs done across all workers: {n_done} / 320 "
              f"({n_configs} configs seen)")
        if all_done:
            break

    print("\nAll workers completed.")
    for wid, p, log_f in procs:
        log_f.close()
        print(f"  worker {wid} exit code: {p.returncode}")

    # ====== 聚合 ======
    print("\nAggregating...")
    import csv
    import numpy as np
    from train_tcn_combos_64 import TARGET_FEATURES, LOO_5SEED_DELTA, enumerate_all_combos

    all_results = {}
    for wid, _, _ in procs:
        p = os.path.join(BASE, f"train_tcn_combos_64_partial_w{wid}.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            for name, rs in r.items():
                all_results.setdefault(name, []).extend(rs)

    # 也合并老的 partial (K=0 already done)
    old_partial = os.path.join(BASE, "train_tcn_combos_64_partial.json")
    if os.path.exists(old_partial):
        try:
            with open(old_partial) as f:
                old = json.load(f)
            old_results = old.get("results_by_name", old)
            for name, rs in old_results.items():
                if isinstance(rs, list) and rs and isinstance(rs[0], dict) and "seed" in rs[0]:
                    already_seeds = {r["seed"] for r in all_results.get(name, [])}
                    for r in rs:
                        if r["seed"] not in already_seeds:
                            all_results.setdefault(name, []).append(r)
        except Exception as e:
            print(f"WARN: could not merge old partial: {e}")

    combos = enumerate_all_combos()
    name_to_combo = {c["name"]: c for c in combos}

    aggregated = []
    baseline_f1m = None
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

    if baseline_f1m is None:
        print("ERROR: baseline not found")
        return

    for a in aggregated:
        a["delta_f1m_vs_27baseline"] = a["mean_f1m"] - baseline_f1m
        a["actual_minus_loo_overestimate"] = a["delta_f1m_vs_27baseline"] - a["loo_cumulative_predicted_delta_f1m"]

    final_path = os.path.join(BASE, "train_tcn_combos_64_results.json")
    final_csv = os.path.join(BASE, "train_tcn_combos_64_results.csv")
    with open(final_path, "w") as f:
        json.dump({
            "baseline_27_mean_f1m": baseline_f1m,
            "loo_5seed_deltas": LOO_5SEED_DELTA,
            "target_features": TARGET_FEATURES,
            "configs": aggregated,
        }, f, indent=2, default=str)

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
    print(f"Saved: {final_path}")
    print(f"Saved: {final_csv}")

    # 表格
    print(f"\n=== 64 配置聚合 (baseline F1m={baseline_f1m:.4f}) ===")
    sorted_aggs = sorted(aggregated, key=lambda x: (x["k"], -x["mean_f1m"]))
    for a in sorted_aggs:
        drops_str = '+'.join(f.split('_')[0] for f in a["drop_features"]) if a["drop_features"] else "NONE"
        print(f"K={a['k']} n={a['n_features']:>2} drops={drops_str:<40} "
              f"F1m={a['mean_f1m']:.4f} ±{a['std_f1m']:.4f} med={a['median_f1m']:.4f} "
              f"Δ={a['delta_f1m_vs_27baseline']:+.4f} PR={a['mean_pr_auc']:.4f} "
              f"col={a['n_collapsed_seeds']} name={a['name']}")


if __name__ == "__main__":
    main()
