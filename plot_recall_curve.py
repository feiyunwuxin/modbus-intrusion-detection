#!/usr/bin/env python3
"""Plot Recall evolution over time for the 234k-frame test (Round 28 fix verification)."""
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = 'logs/test_20261001-234034.csv'

ts_ms = []
recall = []
running_attack = []
running_tp = []

with open(LOG, encoding='utf-8') as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        if r['event'] == 'ALARM':
            a = int(r['running_attack']) if r['running_attack'] else 0
            t = int(r['running_tp']) if r['running_tp'] else 0
            if a > 0:
                ts_ms.append(int(r['t_event_ms']))
                running_attack.append(a)
                running_tp.append(t)
                recall.append(t / a * 100)

print(f"Total ALARM events: {len(ts_ms)}")
print(f"Final Attack: {running_attack[-1]}, Final TP: {running_tp[-1]}, Recall: {recall[-1]:.2f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

# Top: Recall over time
ax1.plot([t / 1000 for t in ts_ms], recall, color='#cc0000', linewidth=0.6, alpha=0.8)
ax1.axhline(y=100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.axhline(y=recall[-1], color='blue', linestyle='--', linewidth=0.8, alpha=0.6,
            label=f'Final Recall = {recall[-1]:.2f}%')
ax1.set_ylabel('Cumulative Recall (%)', fontsize=12)
ax1.set_ylim(0, 110)
ax1.set_title(
    f'Recall Evolution — test_20261001-234034.csv (74k frames, 53min)\n'
    f'Round 28 TP budget cap fix: 101.91% → 97.08% (TP=15731, Attack=16204)',
    fontsize=13
)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom: Running Attack & TP cumulative
ax2.plot([t / 1000 for t in ts_ms], running_attack, color='#0055aa', linewidth=0.8,
         label='Cumulative Attack (Sent)')
ax2.plot([t / 1000 for t in ts_ms], running_tp, color='#cc0000', linewidth=0.8,
         label='Cumulative TP')
ax2.fill_between([t / 1000 for t in ts_ms], running_tp, running_attack,
                 color='#ffcccc', alpha=0.4, label='FN gap (Attack − TP)')
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Cumulative count', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = 'recall_curve_234k_round28.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved: {out}")