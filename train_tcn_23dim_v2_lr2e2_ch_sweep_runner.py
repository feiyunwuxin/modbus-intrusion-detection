#!/usr/bin/env python3
"""TCN+SE 23-dim + B=64 + ep=20 × (LR=2e-2 × {ch=64, ch=128}) × 5-seed — 5-way parallel

10 runs total. 5 workers × 2 runs each (1 seed × 2 channels at LR=2e-2).
"""

import os, sys, time, subprocess

BASE = r"C:\work\Claude\Issue"
LR = 2e-2
CHS = [64, 128]
SEEDS = [42, 123, 456, 789, 1024]


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def main():
    workers = []
    for wid, seed in enumerate(SEEDS):
        wid_path = os.path.join(BASE, f"_worker_lr2e2ch_w{wid}.py")
        with open(wid_path, "w", encoding="utf-8") as f:
            f.write(f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker {wid} seed={seed}, 2 channels at LR=2e-2"""
import os, sys, time, json
sys.path.insert(0, r"{BASE}")
from train_tcn_23dim_v2_lr2e2_ch_sweep import run_one, lr_tag

MY_SEED = {seed}
MY_LR = {LR}
MY_CHS = {CHS}
WORKER_ID = {wid}
BASE = r"{BASE}"
MY_LR_TAG = lr_tag(MY_LR)

t0 = time.time()
for ch in MY_CHS:
    out = os.path.join(BASE, f"train_tcn_23dim_v2_lr2e2_ch_sweep_partial_lr{{MY_LR_TAG}}_ch{{ch}}_s{{MY_SEED}}.json")
    if os.path.exists(out):
        try:
            with open(out) as f: r = json.load(f)
            print(f"[w{{WORKER_ID}}] skip ch={{ch}} s={{MY_SEED}} (cached F1m={{r['test_macro_f1']:.4f}})", flush=True)
            continue
        except: pass
    t1 = time.time()
    try:
        r = run_one(MY_LR, ch, MY_SEED)
        r["lr"] = MY_LR
        r["ch_param"] = ch
        r["seed"] = MY_SEED
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"[w{{WORKER_ID}}] [done] lr={{MY_LR}} ch={{ch}} s={{MY_SEED}} "
              f"F1m={{r['test_macro_f1']:.4f}} PR={{r['test_pr_auc']:.4f}} "
              f"BestEp={{r['best_epoch']}}/{{r['actual_epochs_run']}} "
              f"Params={{r['n_params']:,}} T={{time.time()-t1:.1f}}s "
              f"CumT={{(time.time()-t0)/60:.1f}}m", flush=True)
    except Exception as e:
        import traceback
        print(f"[w{{WORKER_ID}}] ERR ch={{ch}} s={{MY_SEED}}: {{e}}", flush=True)
        traceback.print_exc()

print(f"[w{{WORKER_ID}}] DONE total {{(time.time()-t0)/60:.1f}}m", flush=True)
''')
        log_path = os.path.join(BASE, f"_worker_lr2e2ch_w{wid}.log")
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
