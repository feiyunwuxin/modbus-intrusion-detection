import csv, numpy as np

d = np.load("parity_pysim_probs.npz")
frames = d["frames"]; probs = d["probs"]
pc = dict(zip(frames.tolist(), probs.tolist()))

h743 = {}
with open(r"logs\test_20261020-215840_auto-40ms.csv", encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        if r["event"]=="ALARM" and r["frame_idx"] and r["prob_x1000"]:
            h743[int(r["frame_idx"])] = int(r["prob_x1000"])/1000.0

hf = np.array(sorted(h743.keys()))
hp = np.array([h743[f] for f in hf])

print("H743 alarms:", len(hf))
print()
print("offset  MAE      median_diff   corr    H743>=thr(0.36) vs PC>=0.36 一致率")
best=None
for off in range(-20, 21):
    ps = np.array([pc.get(f+off, np.nan) for f in hf])
    m = ~np.isnan(ps)
    if m.sum() < 100: continue
    mae = np.mean(np.abs(hp[m]-ps[m]))
    md  = np.median(hp[m]-ps[m])
    corr = np.corrcoef(hp[m], ps[m])[0,1]
    agree = np.mean((hp[m]>=0.36) == (ps[m]>=0.36))
    print(f"{off:+4d}   {mae:.4f}   {md:+.4f}    {corr:.4f}   {agree:.4f}")
    if best is None or mae < best[1]:
        best = (off, mae)
print()
print("best offset:", best)
