#!/usr/bin/env python3
"""
Round 63 — FP distribution by attack category.

Hypothesis: FPR 23.98% is real, but may be concentrated in specific attack categories
(or in a few long-burst normal regions). Per-window distribution can tell us whether to
(a) retrain with more diverse normal data, (b) switch to higher threshold, or (c) accept.

Output: analyze_fp_type_v1.json + a markdown summary.
"""
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

LOG = Path("logs/test_20261012-002126_auto-40ms.csv")
CSV = Path("IanArffDatasetF32.csv")

# ARFF 'categorized result' mapping (0..7) per i.md Project dataset memory
CAT_NAMES = {
    0: "Normal",
    1: "DoS",
    2: "Recon",
    3: "MPCI",
    4: "MSCI",
    5: "NMRI",
    6: "CMRI",
    7: "MFCI",
}


def load_arff_labels(csv_path: Path):
    """Return list of (binary, categorized) by 0-indexed frame position.
    ARFF CSV is 1:1 with sent frames: row i (0-indexed after header) corresponds to frame_idx=i+1.
    """
    labels = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        # fields are 20: 18=binary, 19=categorized, 20=specific
        for row in r:
            try:
                bin_v = int(row[17])
                cat_v = int(row[18])
                labels.append((bin_v, cat_v))
            except (ValueError, IndexError):
                labels.append((0, 0))
    return labels


def analyze():
    labels = load_arff_labels(CSV)
    print(f"[load] {len(labels):,} rows from {CSV.name}")

    # Track alarm positions and their FP/TP status
    # Per-window analysis: alarms are per-frame in this log, but per-window eval
    # was done on the test. We can leverage the running_tp / running_fp columns.
    #
    # Simpler approach: For each ALARM row, look at its frame_idx in ARFF
    # (the frame that just got sent). Then compute distribution.

    alarms = []
    with LOG.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["event"] == "ALARM":
                try:
                    fi = int(row["frame_idx"])
                    p1000 = int(row["prob_x1000"])
                    is_fp = row["is_fp"] == "1"
                    n_bytes_s = row["n_bytes"]
                    # n_bytes in alarm = packet length of the alarmed frame
                    alarms.append((fi, p1000, is_fp, n_bytes_s))
                except (ValueError, KeyError):
                    continue

    print(f"[load] {len(alarms):,} alarms from {LOG.name}")

    # Per-category TP/FP
    by_cat = defaultdict(lambda: {"tp": 0, "fp": 0, "probs": []})
    # Per-category alarm count (regardless of fp)
    alarm_by_cat = Counter()
    # n_bytes distribution for FP
    fp_nbytes = Counter()
    tp_nbytes = Counter()

    out_of_range = 0
    for frame_idx, p1000, is_fp, n_bytes_s in alarms:
        row0 = frame_idx - 1  # 0-indexed
        if row0 < 0 or row0 >= len(labels):
            out_of_range += 1
            continue
        bin_v, cat_v = labels[row0]
        cat_name = CAT_NAMES.get(cat_v, f"UNK_{cat_v}")
        by_cat[cat_name]["tp" if not is_fp else "fp"] += 1
        by_cat[cat_name]["probs"].append(p1000 / 1000.0)

        # n_bytes may be empty in alarm rows
        if n_bytes_s:
            try:
                nb = int(n_bytes_s)
                if is_fp:
                    fp_nbytes[nb] += 1
                else:
                    tp_nbytes[nb] += 1
            except ValueError:
                pass

    # Compute summary stats
    summary = {
        "log": str(LOG.name),
        "total_alarms": len(alarms),
        "out_of_range_alarms": out_of_range,
    }
    print("\n=== Alarm distribution by attack category ===")
    print(f"{'Category':<10} {'TP':>6} {'FP':>6} {'FP%':>6} {'MedProb':>8} {'MeanProb':>9}")
    breakdown = []
    for cat_name in CAT_NAMES.values():
        s = by_cat.get(cat_name)
        if not s:
            continue
        tp, fp = s["tp"], s["fp"]
        total = tp + fp
        fp_pct = 100.0 * fp / total if total else 0
        probs = s["probs"]
        med = sorted(probs)[len(probs) // 2] if probs else 0
        mean = sum(probs) / len(probs) if probs else 0
        print(f"{cat_name:<10} {tp:>6} {fp:>6} {fp_pct:>5.1f}% {med:>8.3f} {mean:>9.3f}")
        breakdown.append({
            "category": cat_name,
            "tp": tp,
            "fp": fp,
            "total_alarms": total,
            "fp_pct": round(fp_pct, 2),
            "median_prob": round(med, 4),
            "mean_prob": round(mean, 4),
        })

    summary["by_category"] = breakdown

    # Normal ALARM = FP by definition (PC dedup logic)
    norm = by_cat.get("Normal", {"tp": 0, "fp": 0})
    summary["normal_alarms_as_fp"] = norm["tp"] + norm["fp"]
    print(f"\n[NOTE] Normal-source alarms: {norm['tp'] + norm['fp']:,} "
          f"(these are pure FP since labelled normal)")

    # FP ratio per category
    fp_total = sum(s["fp"] for s in by_cat.values())
    tp_total = sum(s["tp"] for s in by_cat.values())
    print(f"\n[TOTAL] TP={tp_total:,} FP={fp_total:,} Precision={tp_total/(tp_total+fp_total)*100:.2f}%")

    # n_bytes distribution
    print("\n=== Top 10 n_bytes (FP vs TP) ===")
    print(f"{'n_bytes':>8} {'FP':>6} {'TP':>6}")
    fp_all = sorted(fp_nbytes.items(), key=lambda x: -x[1])[:10]
    for nb, c in fp_all:
        t = tp_nbytes.get(nb, 0)
        print(f"{nb:>8} {c:>6} {t:>6}")

    summary["n_bytes_top_fp"] = [{"n_bytes": nb, "fp": c, "tp": tp_nbytes.get(nb, 0)}
                                  for nb, c in fp_all]

    # Save
    out_json = Path("logs/test_20261012-002126_auto-40ms.fp_by_category.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    analyze()
