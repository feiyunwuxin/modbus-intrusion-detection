#!/usr/bin/env python3
"""Retrain 5 seeds 5-way parallel runner (ch=32 do=0.1 冠军配置 + .pt 保存)"""

import os, sys, time, subprocess

BASE = r"C:\work\Claude\Issue"
SEEDS = [42, 123, 456, 789, 1024]


def main():
    workers = []
    for wid, seed in enumerate(SEEDS):
        wid_path = os.path.join(BASE, f"_worker_retrain23v2_w{wid}.py")
        with open(wid_path, "w", encoding="utf-8") as f:
            f.write(f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker {wid} seed={seed}"""
import os, sys, time, json
sys.path.insert(0, r"{BASE}")
from retrain_tcn_23dim_b64_ch32_do01_savept import run_one

MY_SEED = {seed}
WORKER_ID = {wid}
BASE = r"{BASE}"

t0 = time.time()
try:
    r = run_one(MY_SEED)
    r["seed"] = MY_SEED
    print(f"[w{{WORKER_ID}}] [done] s={{MY_SEED}} "
          f"F1m={{r['test_macro_f1']:.4f}} PR={{r['test_pr_auc']:.4f}} "
          f"BestEp={{r['best_epoch']}}/{{r['actual_epochs_run']}} "
          f"Params={{r['n_params']:,}} T={{time.time()-t0:.1f}}s", flush=True)
except Exception as e:
    import traceback
    print(f"[w{{WORKER_ID}}] ERR s={{MY_SEED}}: {{e}}", flush=True)
    traceback.print_exc()

print(f"[w{{WORKER_ID}}] DONE total {{(time.time()-t0)/60:.1f}}m", flush=True)
''')
        log_path = os.path.join(BASE, f"_worker_retrain23v2_w{wid}.log")
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
