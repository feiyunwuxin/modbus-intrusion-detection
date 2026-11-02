"""Offline replay of rtu_replay_gui.py's WindowEvaluator over a test-log CSV.

The evaluator class is exec'd from the real source file (no Python re-implementation),
per the "parity test must bind to real source" rule.

Usage: python replay_window_eval.py <log.csv> [more.csv ...]
"""
import ast
import csv
import sys
from collections import deque

SRC = r"C:/work/Claude/Issue/rtu_replay_gui.py"


def load_evaluator(src_path):
    source = open(src_path, encoding="utf-8").read()
    tree = ast.parse(source)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "WindowEvaluator")
    ns = {"deque": deque}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), src_path, "exec"), ns)
    return ns["WindowEvaluator"]


def replay(evaluator_cls, csv_path):
    class Traced(evaluator_cls):
        """Keeps a live reference to each window's state dict, so late rescues
        (Round 64/69 FN->TP / TN->FP rollback) are visible in the final labels."""

        def __init__(self):
            super().__init__()
            self.final = {}

        def _evaluate_window(self):
            super()._evaluate_window()
            W = self._n_windows - 1
            self.final[W] = self._recent_evals[-1]

    ev = Traced()
    prev_count = None
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            kind = row["event"]
            if kind == "SEND":
                ev.on_frame(row["binary"] == "1")
            elif kind == "ALARM":
                count = int(row["alarm_count"])
                K = int(row["frame_idx"]) if row["frame_idx"] else 0
                delta = count if prev_count is None else count - prev_count
                if delta < 1:
                    delta = 1
                prev_count = count
                if K > 0:
                    for i in range(delta):
                        fr = K - (delta - 1) + i
                        ev.on_alarm_with_frame(max(fr, 1))
                else:
                    for _ in range(delta):
                        ev.on_alarm()
    per_window = [(w, s["attack"], s["alarm"]) for w, s in sorted(ev.final.items())]
    return ev, per_window


def main():
    src = SRC
    if len(sys.argv) > 3:
        src = sys.argv[1]
        paths = sys.argv[2:]
    else:
        paths = sys.argv[1:]
    cls = load_evaluator(src)
    results = []
    for p in paths:
        ev, pw = replay(cls, p)
        m = ev.get_metrics()
        print(f"\n=== {p}")
        print(f"  windows={m['n_windows']} TP={m['tp']} FP={m['fp']} "
              f"TN={m['tn']} FN={m['fn']} F1={m['f1']:.4f} "
              f"R={m['recall']:.4f} P={m['precision']:.4f}")
        results.append((p, m, pw))
    if len(results) == 2:
        (_, m1, pw1), (_, m2, pw2) = results
        n = min(len(pw1), len(pw2))
        flips = [(i, pw1[i], pw2[i]) for i in range(n)
                 if pw1[i][1:] != pw2[i][1:]]
        print(f"\n--- window-level diff (n={n}) ---")
        print(f"  windows compared: {n}  (len1={len(pw1)} len2={len(pw2)})")
        print(f"  differing windows: {len(flips)}")
        for f in flips[:20]:
            print(f"    W={f[0]}: {f[1]} vs {f[2]}")
        # aggregate label flip counts
        from collections import Counter
        c = Counter((a[1], a[2], b[1], b[2]) for _, a, b in flips)
        for k, v in c.items():
            print(f"    attack/alarm {k[0]}/{k[1]} -> {k[2]}/{k[3]}: {v}")


if __name__ == "__main__":
    main()
