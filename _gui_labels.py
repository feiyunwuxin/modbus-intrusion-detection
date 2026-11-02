import csv, numpy as np

# --- GUI 侧标签 ---
gui = {}   # frame_idx -> binary
with open(r"logs\test_20261020-215840_auto-40ms.csv", encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        if r["event"]=="SEND" and r["frame_idx"] and r["binary"] != "":
            gui[int(r["frame_idx"])] = int(r["binary"])
print("GUI SEND rows:", len(gui), " attack frames:", sum(gui.values()))

# --- CSV 原始标签 ---
csvl = {}
with open("IanArffDataset.csv", encoding="utf-8") as f:
    for i, r in enumerate(csv.DictReader(f), start=1):
        if i < 200000: continue
        if i > 200000+74628: break
        csvl[i-200000] = int(float(r["binary result"])) if r.get("binary result") not in (None,"","?") else 0
print("CSV rows:", len(csvl), " attack frames:", sum(csvl.values()))

common = sorted(set(gui) & set(csvl))
mis = [(f, gui[f], csvl[f]) for f in common if gui[f] != csvl[f]]
print(f"标签不一致帧数: {len(mis)} / {len(common)}")
if mis:
    print("  前 20:", mis[:20])

# 窗口标签
W=16
n_win = len(common)//W
gw = [any(gui[f] for f in range(w*16+1, w*16+17) if f in gui) for w in range(n_win)]
cw = [any(csvl[f] for f in range(w*16+1, w*16+17) if f in csvl) for w in range(n_win)]
print(f"windows={n_win}  GUI attack={sum(gw)}  CSV attack={sum(cw)}  不一致窗口={sum(1 for a,b in zip(gw,cw) if a!=b)}")
np.savez_compressed("gui_labels.npz",
    frames=np.array(common, dtype=np.int64),
    gui=np.array([gui[f] for f in common], dtype=np.int8),
    csv=np.array([csvl[f] for f in common], dtype=np.int8))
