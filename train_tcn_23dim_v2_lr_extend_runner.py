#!/usr/bin/env python3
"""TCN+SE 23-dim + B=64 × LR 扩展扫描 (LR=6e-3, 8e-3) — 5-way parallel

10 runs (2 LR × 5 seeds). 复用 train_tcn_23dim_v2_lr_sweep.py 的 run_one。
LR=4e-3 时 F1m=0.8705,看 6e-3/8e-3 是否继续上升。
"""

import os, sys, time, subprocess

BASE = r"C:\work\Claude\Issue"
NEW_LRS = [6e-3, 8e-3]
SEEDS = [42, 123, 456, 789, 1024]


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def main():
    workers = []
    for wid, seed in enumerate(SEEDS):
        wid_path = os.path.join(BASE, f"_worker_lr23v2ext_w{wid}.py")
        with open(wid_path, "w", encoding="utf-8") as f:
            f.write(f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker {wid} seed={seed}, 2 LR values (6e-3, 8e-3)"""
import os, sys, time, json
sys.path.insert(0, r"{BASE}")
from train_tcn_23dim_v2_lr_sweep import run_one, lr_tag

MY_SEED = {seed}
MY_LRS = {NEW_LRS}
WORKER_ID = {wid}
BASE = r"{BASE}"

t0 = time.time()
for lr in MY_LRS:
    tag = lr_tag(lr)
    out = os.path.join(BASE, f"train_tcn_23dim_v2_lr_sweep_partial_lr{{tag}}_s{{MY_SEED}}.json")
    if os.path.exists(out):
        try:
            with open(out) as f: r = json.load(f)
            print(f"[w{{WORKER_ID}}] skip lr={{lr}} s={{MY_SEED}} (cached F1m={{r['test_macro_f1']:.4f}})", flush=True)
            continue
        except: pass
    t1 = time.time()
    try:
        r = run_one(lr, MY_SEED)
        r["lr"] = lr
        r["seed"] = MY_SEED
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"[w{{WORKER_ID}}] [done] lr={{lr}} s={{MY_SEED}} "
              f"F1m={{r['test_macro_f1']:.4f}} PR={{r['test_pr_auc']:.4f}} "
              f"Epoch={{r['best_epoch']}} T={{time.time()-t1:.1f}}s "
              f"CumT={{(time.time()-t0)/60:.1f}}m", flush=True)
    except Exception as e:
        import traceback
        print(f"[w{{WORKER_ID}}] ERR lr={{lr}} s={{MY_SEED}}: {{e}}", flush=True)
        traceback.print_exc()

print(f"[w{{WORKER_ID}}] DONE total {{(time.time()-t0)/60:.1f}}m", flush=True)
''')
        log_path = os.path.join(BASE, f"_worker_lr23v2ext_w{wid}.log")
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
