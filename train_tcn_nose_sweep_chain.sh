#!/usr/bin/env bash
# Sequentially run ch, batch, epochs, lr sweeps (dropout already running)
# Each sweep = 5-way parallel via train_tcn_nose_sweep_runner.py --dim <name>
set -e
PY="/c/Users/wangbin/Programs/anaconda3/envs/wangbin1/python.exe"
cd /c/work/Claude/Issue

for dim in ch batch epochs lr; do
    echo "=== [chain] starting dim=$dim at $(date +%H:%M:%S) ==="
    "$PY" train_tcn_nose_sweep_runner.py --dim $dim 2>&1 | tee "_chain_${dim}.log" | tail -50
    echo "=== [chain] done dim=$dim at $(date +%H:%M:%S) ==="
done

echo "=== [chain] ALL DONE at $(date +%H:%M:%S) ==="