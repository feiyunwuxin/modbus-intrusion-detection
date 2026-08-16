#!/usr/bin/env python3
"""TCN+SE 23-dim (64-combos 冠军版) × batch size 扫描 — 5-way parallel runner

策略: 5 workers × 5 runs each = 25 runs 总数。每个 worker 跑 1 个 seed × 5 个 batch。
复用 [[project-train_tcn_combos_64_parallel]] 模式 (subprocess.Popen)。

调度:
  w0: seed=42  → B=64,96,128,192,256
  w1: seed=123 → B=64,96,128,192,256
  w2: seed=456 → B=64,96,128,192,256
  w3: seed=789 → B=64,96,128,192,256
  w4: seed=1024→ B=64,96,128,192,256

每个 worker 内按 batch asc 跑,小 batch (慢) 先跑,大 batch (快) 后跑,平均 ~9 min/wall.

输出 partial:
  train_tcn_23dim_v2_batch_sweep_partial_b{B}_s{seed}.json  (25 个)

最终汇总: train_tcn_23dim_v2_batch_sweep_results.json
"""

import os, sys, time, subprocess

BASE = r"C:\work\Claude\Issue"
BATCHES = [64, 96, 128, 192, 256]
SEEDS = [42, 123, 456, 789, 1024]


def main():
    workers = []
    for wid, seed in enumerate(SEEDS):
        # 写 worker 脚本 (UTF-8 显式编码,避免 Windows 默认 GBK 把破折号写成 0xA1)
        wid_path = os.path.join(BASE, f"_worker_b23v2_w{wid}.py")
        with open(wid_path, "w", encoding="utf-8") as f:
            f.write(f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker {wid} seed={seed}, 5 batches"""
import os, sys, time, json
sys.path.insert(0, r"{BASE}")
from train_tcn_23dim_v2_batch_sweep import run_one

MY_SEED = {seed}
MY_BATCHES = {BATCHES}
WORKER_ID = {wid}
BASE = r"{BASE}"

t0 = time.time()
for batch in MY_BATCHES:
    out = os.path.join(BASE, f"train_tcn_23dim_v2_batch_sweep_partial_b{{batch}}_s{{MY_SEED}}.json")
    if os.path.exists(out):
        try:
            with open(out) as f: r = json.load(f)
            print(f"[w{{WORKER_ID}}] skip b={{batch}} s={{MY_SEED}} (cached F1m={{r['test_macro_f1']:.4f}})", flush=True)
            continue
        except: pass
    t1 = time.time()
    try:
        r = run_one(batch, MY_SEED)
        r["batch"] = batch
        r["seed"] = MY_SEED
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[w{{WORKER_ID}}] [done] b={{batch}} s={{MY_SEED}} "
              f"F1m={{r['test_macro_f1']:.4f}} PR={{r['test_pr_auc']:.4f}} "
              f"Epoch={{r['best_epoch']}} T={{time.time()-t1:.1f}}s "
              f"CumT={{(time.time()-t0)/60:.1f}}m", flush=True)
    except Exception as e:
        import traceback
        print(f"[w{{WORKER_ID}}] ERR b={{batch}} s={{MY_SEED}}: {{e}}", flush=True)
        traceback.print_exc()

print(f"[w{{WORKER_ID}}] DONE total {{(time.time()-t0)/60:.1f}}m", flush=True)
''')
        log_path = os.path.join(BASE, f"_worker_b23v2_w{wid}.log")
        log_f = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, wid_path],
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=BASE,
        )
        workers.append((wid, p, log_f, seed))
        print(f"[launch] w{wid} seed={seed} pid={p.pid} log={log_path}", flush=True)

    print(f"\n[wait] {len(workers)} workers running, ctrl-c to abort...\n", flush=True)
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

    print(f"\n[all-done] all 5 workers finished at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
