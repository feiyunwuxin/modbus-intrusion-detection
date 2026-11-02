"""F 路线 Step 2 — H743 实际数据 vs ARFF train/test 分布对比

目的: 用 h743_swdbg2_dump.txt 解析 12 windows 的 23-dim 实际数据,
       跟 ARFF train/test 23-dim 分布对比, 找 H743 实际数据中漂移最大的 channel.

输出:
  - diag_f_route_h743_drift.md
  - diag_f_route_h743_drift.png  (H743 mean vs ARFF train mean bar chart)
"""
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
import json

DATA_DIR = "C:/work/Claude/Issue"
DUMP_FILE = f"{DATA_DIR}/logs/h743_swdbg2_dump.txt"
KEEP_IDX = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
OUT_MD = f"{DATA_DIR}/diag_f_route_h743_drift.md"
OUT_PNG = f"{DATA_DIR}/diag_f_route_h743_drift.png"

print("STEP 1: parse H743 dump file ...")
# Lines in dump have prefix: [HH:MM:SS.mmm] RX (size B): <content>
# Strip prefix, then find [SW-DBG2] blocks via regex.

with open(DUMP_FILE, "r", encoding="utf-8", errors="replace") as f:
    raw_lines = f.readlines()

# Strip "[ts] RX (...) B): " prefix from each line
content_lines = []
for line in raw_lines:
    line = line.rstrip("\n")
    # Match "[ts] RX (size B): <content>"
    m = re.match(r"^\[\d{2}:\d{2}:\d{2}\.\d+\]\s+RX\s+\(\d+\s+B\):\s*(.*)$", line)
    if m:
        content_lines.append(m.group(1))
    else:
        content_lines.append(line)

# Now content_lines contains the raw dump content without timestamps/RX markers.
# Find all [SW-DBG2] blocks.
full_text = "\n".join(content_lines)
print(f"  raw content lines: {len(content_lines)}")

# Regex: [SW-DBG2] w#N prob=X | out23 (23x16, ch-major):\n<ch lines>\n...[SW-DBG2] w#N end...
# Each ch line: "  chNN: <16 floats>" possibly continued on next line if split.
# Strategy: extract body, normalize all line breaks, parse chNN: ... (16 floats).
block_pattern = re.compile(
    r"\[SW-DBG2\]\s+w#(\d+)\s+prob=([\d.]+)\s+\|\s+out23\s+\(23x16,\s+ch-major\):(.*?)\[SW-DBG2\]\s+w#\d+\s+end",
    re.DOTALL,
)

windows = {}
for m in block_pattern.finditer(full_text):
    win_id = int(m.group(1))
    prob = float(m.group(2))
    body = m.group(3)

    # Strip ".." (continuation marker) and all whitespace, then split on "chNN:"
    body_clean = body.replace("..", " ")
    ch_splits = re.split(r"ch(\d+):", body_clean)

    ch_data = {}
    for i in range(1, len(ch_splits), 2):
        c = int(ch_splits[i])
        vals_str = ch_splits[i + 1].strip() if i + 1 < len(ch_splits) else ""
        # Strip non-float characters (keep digits, dot, minus, space)
        vals_str = re.sub(r"[^\d\.\-\s]", " ", vals_str)
        vals = vals_str.split()
        try:
            ch_data[c] = [float(x) for x in vals[:16]]
        except ValueError as e:
            ch_data[c] = []
    windows[win_id] = {"prob": prob, "ch": ch_data}

# Debug: show one window's structure
if windows:
    first_win = sorted(windows.keys())[0]
    print(f"  [debug] first parsed window w#{first_win}: ch00[:3]={windows[first_win]['ch'].get(0, [])[:3]}")
    print(f"    ch00 len={len(windows[first_win]['ch'].get(0, []))}, ch03 len={len(windows[first_win]['ch'].get(3, []))}")
    print(f"    ch03 vals={windows[first_win]['ch'].get(3, [])}")
    print(f"    ch08 vals={windows[first_win]['ch'].get(8, [])}")

# Filter valid windows (have all 23 channels × 16 vals)
valid_windows = {}
for w, d in windows.items():
    if len(d["ch"]) == 23 and all(len(v) == 16 for v in d["ch"].values()):
        valid_windows[w] = d

print(f"  parsed {len(windows)} windows, valid={len(valid_windows)}")
for w in sorted(valid_windows.keys()):
    print(f"    w#{w} prob={valid_windows[w]['prob']:.4f} channels={len(valid_windows[w]['ch'])}")

# Build H743 numpy array: (N_valid, 23, 16) — ch-major
N = len(valid_windows)
H743 = np.zeros((N, 23, 16))
probs = np.zeros(N)
for i, w in enumerate(sorted(valid_windows.keys())):
    for c in range(23):
        H743[i, c, :] = valid_windows[w]["ch"][c]
    probs[i] = valid_windows[w]["prob"]
print(f"  H743 shape: {H743.shape}  probs: {probs}")

print()
print("STEP 2: load ARFF train/test 27-dim npy -> 23-dim ...")
X_tr = np.load(f"{DATA_DIR}/X_train_binary_v2_scada_window16.npy")[..., KEEP_IDX]
X_te = np.load(f"{DATA_DIR}/X_test_binary_v2_scada_window16.npy")[..., KEEP_IDX]
print(f"  ARFF tr {X_tr.shape}  te {X_te.shape}")

# Convert all to channel-major layout: shape (23, N*T) where index [c] = channel c
# IMPORTANT: np.load(...)[..., KEEP_IDX] produces non-contiguous array. Must call ascontiguousarray
# BEFORE reshape, otherwise transpose().reshape() interprets memory wrong.
X_tr_c = np.ascontiguousarray(X_tr)  # (N, T, 23) - contiguous
X_te_c = np.ascontiguousarray(X_te)
H743_c = np.ascontiguousarray(H743)

# Now safe to transpose+reshape
X_tr_f = X_tr_c.transpose(0, 2, 1).reshape(23, -1)  # (23, N*T) ch-major
X_te_f = X_te_c.transpose(0, 2, 1).reshape(23, -1)
H743_f = H743_c.reshape(23, -1)  # (23, N*16) — already ch-major

# Now index as X_tr_f[c] for channel c
print(f"  X_tr_f: {X_tr_f.shape}  X_te_f: {X_te_f.shape}  H743_f: {H743_f.shape}")

print()
print("STEP 3: per-channel robust stats (median + IQR) ...")
print(f"  H743 only has {H743_f.shape[1]} samples (5 windows × 16 timesteps)")
print(f"  → use median (not mean) to avoid outlier distortion from single attack samples")
stats = {}
for c in range(23):
    a_tr = X_tr_f[c]
    a_te = X_te_f[c]
    a_h7 = H743_f[c]
    stats[c] = {
        "tr_mean": float(a_tr.mean()),
        "tr_std": float(a_tr.std()),
        "tr_med": float(np.median(a_tr)),
        "tr_p25": float(np.percentile(a_tr, 25)),
        "tr_p75": float(np.percentile(a_tr, 75)),
        "te_med": float(np.median(a_te)),
        "h7_med":  float(np.median(a_h7)),
        "h7_p25":  float(np.percentile(a_h7, 25)),
        "h7_p75":  float(np.percentile(a_h7, 75)),
        "h7_min":  float(a_h7.min()),
        "h7_max":  float(a_h7.max()),
    }

print()
print("STEP 4: drift H743 vs ARFF train (Wasserstein, KS) ...")
print(f"  ⚠  KS is biased when sample sizes are very different (192k vs 80). Use IQR-median distance as robust metric.")
drift = []
for c in range(23):
    a_tr = X_tr_f[c]
    a_h7 = H743_f[c]
    w = wasserstein_distance(a_tr, a_h7)
    ks, _ = ks_2samp(a_tr, a_h7)
    md = abs(stats[c]["h7_med"] - stats[c]["tr_med"])
    # IQR-median distance: how many IQRs apart are medians
    iqr_tr = stats[c]["tr_p75"] - stats[c]["tr_p25"]
    iqr_md = md / iqr_tr if iqr_tr > 0 else 0
    drift.append({
        "ch": c, "w_h7_tr": w, "ks_h7_tr": ks, "median_diff": md, "iqr_md": iqr_md
    })

drift_sorted = sorted(drift, key=lambda x: x["iqr_md"], reverse=True)
print(f"  Top 10 H743↔ARFF-train drift channels (by IQR-normalized median distance):")
for i, d in enumerate(drift_sorted[:10]):
    c = d["ch"]
    s = stats[c]
    print(f"    rank {i+1:2d}  ch{c:02d}  IQR-MD={d['iqr_md']:.2f}σ  W={d['w_h7_tr']:.4f}  KS={d['ks_h7_tr']:.4f}  "
          f"med_diff={d['median_diff']:.3f}  "
          f"ARFF_tr_med={s['tr_med']:+.3f} [p25={s['tr_p25']:+.3f},p75={s['tr_p75']:+.3f}]  "
          f"H743_med={s['h7_med']:+.3f} [p25={s['h7_p25']:+.3f},p75={s['h7_p75']:+.3f}]")

print()
print("STEP 5: plot H743 vs ARFF train MEDIAN (per channel bar chart) ...")
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

chs = list(range(23))
tr_meds = [stats[c]["tr_med"] for c in chs]
te_meds = [stats[c]["te_med"] for c in chs]
h7_meds = [stats[c]["h7_med"] for c in chs]

x = np.arange(23)
w_bar = 0.28
axes[0].bar(x - w_bar, tr_meds, w_bar, label="ARFF train", color="steelblue", alpha=0.8)
axes[0].bar(x,         te_meds, w_bar, label="ARFF test", color="darkorange", alpha=0.8)
axes[0].bar(x + w_bar, h7_meds, w_bar, label="H743 actual (n=5 windows)", color="crimson", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"ch{c:02d}" for c in chs], rotation=0)
axes[0].set_ylabel("median")
axes[0].set_title("Per-channel median: ARFF train vs ARFF test vs H743 actual (5 windows)")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Drift ranking by IQR-normalized median distance
rank_chs = [d["ch"] for d in drift_sorted]
rank_iqr = [d["iqr_md"] for d in drift_sorted]
axes[1].bar(range(23), rank_iqr, color="crimson")
axes[1].set_xticks(range(23))
axes[1].set_xticklabels([f"ch{c:02d}" for c in rank_chs], rotation=0)
axes[1].set_ylabel("IQR-normalized median distance (sigma units)")
axes[1].set_title("H743↔ARFF-train drift by median distance (sorted desc)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=110)
print(f"  saved {OUT_PNG}")

print()
print("STEP 6: write markdown report ...")
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("# F 路线 Step 2 — H743 实际数据 vs ARFF 分布对比\n\n")
    f.write(f"## 数据规模\n")
    f.write(f"- H743 windows parsed: **{N}** (5 valid windows from logs/h743_swdbg2_dump.txt)\n")
    f.write(f"- H743 samples/channel: **{len(H743_f)}** (very small — affects KS reliability)\n")
    f.write(f"- ARFF train: 192224 samples/channel\n")
    f.write(f"- ARFF test: 54912 samples/channel\n\n")

    f.write(f"## ⚠️ 警告: H743 样本极少\n\n")
    f.write("只有 5 windows × 16 timesteps = 80 samples per channel, 统计检验力低.\n")
    f.write("本报告主要看 median/IQR 而非 mean/std, 因为单个 attack 样本的极端值会扭曲 mean.\n\n")

    f.write(f"## H743 windows 详情\n\n")
    f.write("| w# | prob | 标签 (跟 memory 比对) |\n")
    f.write("|---|---|---|\n")
    for w in sorted(valid_windows.keys()):
        prob = valid_windows[w]["prob"]
        # Memory: 5 FN (2,3,7,15,51) + 3 TP (0,1,8)
        if w in [2,3,7,15,51]:
            label = "FN (预期 attack, 模型 miss)"
        elif w in [0,1,8]:
            label = "TP (预期 attack, 模型 catch)"
        else:
            label = "其他"
        f.write(f"| w#{w} | {prob:.4f} | {label} |\n")

    f.write("\n## Top 10 H743↔ARFF-train drift (IQR-normalized median distance)\n\n")
    f.write("| Rank | ch | IQR-MD (σ) | W | KS | median_diff | ARFF_tr_med [p25,p75] | H743_med [p25,p75] |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for i, d in enumerate(drift_sorted[:10]):
        c = d["ch"]
        s = stats[c]
        f.write(f"| {i+1} | ch{c:02d} | {d['iqr_md']:.2f} | {d['w_h7_tr']:.4f} | {d['ks_h7_tr']:.4f} | "
                f"{d['median_diff']:.3f} | {s['tr_med']:+.3f} [{s['tr_p25']:+.3f},{s['tr_p75']:+.3f}] | "
                f"{s['h7_med']:+.3f} [{s['h7_p25']:+.3f},{s['h7_p75']:+.3f}] |\n")

    f.write("\n## 全 channel 表 (median + IQR)\n\n")
    f.write("| ch | IQR-MD | ARFF_tr_med | ARFF_tr_IQR | H743_med | H743_IQR | H743 range |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for d in drift_sorted:
        c = d["ch"]
        s = stats[c]
        iqr_tr = s["tr_p75"] - s["tr_p25"]
        iqr_h7 = s["h7_p75"] - s["h7_p25"]
        f.write(f"| ch{c:02d} | {d['iqr_md']:.2f}σ | {s['tr_med']:+.3f} | {iqr_tr:.3f} | "
                f"{s['h7_med']:+.3f} | {iqr_h7:.3f} | [{s['h7_min']:+.2f}, {s['h7_max']:+.2f}] |\n")

    f.write("\n## 解读 & 下一步\n\n")
    top3 = drift_sorted[:3]
    f.write(f"### Top-3 漂移 channel:\n\n")
    for i, d in enumerate(top3):
        c = d["ch"]
        s = stats[c]
        f.write(f"- **ch{c:02d}** (IQR-MD={d['iqr_md']:.2f}σ):\n")
        f.write(f"  - ARFF train median: {s['tr_med']:+.3f} [p25={s['tr_p25']:+.3f}, p75={s['tr_p75']:+.3f}]\n")
        f.write(f"  - H743 median: {s['h7_med']:+.3f} [p25={s['h7_p25']:+.3f}, p75={s['h7_p75']:+.3f}]\n")
        f.write(f"  - H743 range: [{s['h7_min']:+.2f}, {s['h7_max']:+.2f}] (注意 max 可能受 attack sample 极端值影响)\n\n")

    f.write("### 跟 ARFF 内部漂移对比 (Step 1)\n\n")
    f.write("- Step 1 找的 train↔test 漂移最大是 ch14 (W=3.69)\n")
    f.write("- Step 2 找的 H743↔train 漂移大 channel 看是否跟 ch14 重合\n")
    f.write("- 若**不重合**, H743 跟 train/test 都不同, 走 **路线 E (augmentation)**\n")
    f.write("- 若**部分重合**, ARFF 时序分布确实有结构问题, 走 **路线 D (random split retrain)**\n\n")

    f.write("### 决策建议\n\n")
    f.write("- 若 H743↔train drift < 1σ for most channels → FPR 9.07% 不是分布漂移, 是别的原因 (e.g. 标签噪声, 阈值)\n")
    f.write("- 若 H743↔train drift > 2σ for most channels → 真因是 H743 实际数据跟 ARFF 训练分布不一致, 走 D 或 E\n")
    f.write("- 若只有少数 channel drift 大 → 排查这些 channel 是什么特征, 决定是 D (整体分布) 还是 E (特定 feature)\n\n")

print(f"  saved {OUT_MD}")
print()
print("DONE.")