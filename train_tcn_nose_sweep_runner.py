#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCN-NoSE 23-dim 超参扫描 5-way parallel runner

策略: 5 workers × 1 seed × 5 values = 5 parallel runs per dim, sequential dim launches.
镜像 train_tcn_23dim_v2_batch_sweep_runner.py 模式 (subprocess.Popen + log)。

每个 worker 跑一个 seed,扫描目标 dim 的所有 5 个 value。

Usage:
    # 单 dim
    python train_tcn_nose_sweep_runner.py --dim dropout

    # 全 5 dim
    python train_tcn_nose_sweep_runner.py --all

Partial 文件: train_tcn_nose_sweep_partial_{dim}_{val_str}_s{seed}.json
"""

import os, sys, time, subprocess, argparse

BASE = r"C:\work\Claude\Issue"
DIMS = {
    'batch':   [64, 96, 128, 192, 256],
    'ch':      [16, 32, 48, 64, 96],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'epochs':  [20, 30, 40, 50, 60],
    'lr':      [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3],
}
SEEDS = [42, 123, 456, 789, 1024]


def val_to_str(v):
    """Convert value to filename-safe string"""
    s = str(v).replace('.', '_').replace('e-0', 'e_').replace('-', '_')
    return s


def partial_path(dim, val, seed):
    return os.path.join(BASE, f"train_tcn_nose_sweep_partial_{dim}_{val_to_str(val)}_s{seed}.json")


def run_one_dim(dim):
    """Launch 5 workers (one per seed), each running 5 values for given dim"""
    values = DIMS[dim]
    print(f"\n{'='*80}\n=== Sweep dim={dim}, values={values} ===\n{'='*80}", flush=True)

    workers = []
    for wid, seed in enumerate(SEEDS):
        wid_path = os.path.join(BASE, f"_worker_nose_w{wid}.py")
        with open(wid_path, "w", encoding="utf-8") as f:
            f.write(f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker {wid} seed={seed}, dim={dim}, values={values}"""
import os, sys, time, json
sys.path.insert(0, r"{BASE}")
from train_tcn_nose_sweep import run_one

MY_SEED = {seed}
MY_DIM = "{dim}"
MY_VALUES = {values}
WORKER_ID = {wid}
BASE = r"{BASE}"


def val_to_str(v):
    s = str(v).replace('.', '_').replace('e-0', 'e_').replace('-', '_')
    return s


t0 = time.time()
for val in MY_VALUES:
    out = os.path.join(BASE, f"train_tcn_nose_sweep_partial_{{MY_DIM}}_{{val_to_str(val)}}_s{{MY_SEED}}.json")
    if os.path.exists(out):
        try:
            with open(out) as f: r = json.load(f)
            print(f"[w{{WORKER_ID}}] skip dim={{MY_DIM}} val={{val}} s={{MY_SEED}} (cached F1m={{r['test_macro_f1']:.4f}})", flush=True)
            continue
        except: pass
    t1 = time.time()
    try:
        kwargs = {{MY_DIM if MY_DIM != 'ch' else 'channels': val}}
        r = run_one(MY_SEED, **kwargs)
        r[MY_DIM if MY_DIM != 'ch' else 'channels'] = val
        r['seed'] = MY_SEED
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[w{{WORKER_ID}}] [done] dim={{MY_DIM}} val={{val}} s={{MY_SEED}} "
              f"F1m={{r['test_macro_f1']:.4f}} BinF1={{r['test_binary_f1']:.4f}} "
              f"PR={{r['test_pr_auc']:.4f}} Ep={{r['best_epoch']}} T={{time.time()-t1:.1f}}s "
              f"CumT={{(time.time()-t0)/60:.1f}}m", flush=True)
    except Exception as e:
        import traceback
        print(f"[w{{WORKER_ID}}] ERR dim={{MY_DIM}} val={{val}} s={{MY_SEED}}: {{e}}", flush=True)
        traceback.print_exc()

print(f"[w{{WORKER_ID}}] DONE total {{(time.time()-t0)/60:.1f}}m", flush=True)
''')
        log_path = os.path.join(BASE, f"_worker_nose_w{wid}.log")
        log_f = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, wid_path],
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=BASE,
        )
        workers.append((wid, p, log_f, seed))
        print(f"[launch] w{wid} seed={seed} pid={p.pid} log={log_path}", flush=True)

    print(f"\n[wait] {len(workers)} workers running for dim={dim} ...\n", flush=True)
    try:
        for wid, p, log_f, seed in workers:
            rc = p.wait()
            log_f.close()
            print(f"[finish] w{wid} seed={seed} rc={rc}", flush=True)
    except KeyboardInterrupt:
        print("\n[abort] killing all workers", flush=True)
        for wid, p, log_f, _ in workers:
            try: p.terminate()
            except: pass
            log_f.close()
        sys.exit(1)

    print(f"\n[dim={dim} done] at {time.strftime('%H:%M:%S')}", flush=True)


def main_runner():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', choices=list(DIMS.keys()))
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for dim in DIMS.keys():
            run_one_dim(dim)
    elif args.dim:
        run_one_dim(args.dim)
    else:
        print("Usage: --dim <batch|ch|dropout|epochs|lr> | --all")


if __name__ == "__main__":
    main_runner()