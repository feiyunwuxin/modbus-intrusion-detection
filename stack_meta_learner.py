#!/usr/bin/env python3
"""Stack11: 11 architectures × 5 seeds → 3 meta-learner methods.

Loads 55 per-seed TEST prediction CSVs and applies meta-learning via 5-fold CV
to produce honest test-set predictions (no test contamination). 3 methods:

  M1 Simple average      — uniform weights (no fitting needed)
  M2 Weighted avg (SLSQP)— weights fitted via 5-fold CV on test
  M3 LR stacking (L2)    — LogReg L2 fitted via 5-fold CV on test

Plus a naive "in-sample" upper bound: train M2/M3 on full test, evaluate on test.
"""
import os
import csv
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

BASE = r"D:\workspace\claude\Issue\Issue"

# Same 11 ARCHS as evaluate_5arch_4metrics.py
ARCHS = [
    ("tcn_se",     "TCN+SE",                  "v4_se_23dim_b64_ch32_do01_window16"),
    ("tcn",        "TCN (no SE)",             "tcn_23dim_w16"),
    ("lstm",       "BiLSTM",                  "lstm_23dim_w16"),
    ("gru",        "BiGRU",                   "gru_23dim_w16"),
    ("cnnlstm",    "CNN-LSTM",                "cnnlstm_23dim_w16"),
    ("cnn",        "Pure CNN",                "cnn_23dim_w16"),
    ("lstm_v2",    "BiLSTM (h128)",           "lstm_23dim_w16_h128"),
    ("gru_v2",     "BiGRU (h128)",            "gru_23dim_w16_h128"),
    ("cnnlstm_v2", "CNN-LSTM (ch256,h128)",   "cnnlstm_23dim_w16_ch256"),
    ("lgb",        "LightGBM",                "lgb_23dim_w16"),
    ("rf",         "Random Forest",           "rf_23dim_w16"),
]
SEEDS = [42, 123, 456, 789, 1024]
THR = 0.5
N_FOLDS = 5

# Build (arch_short, arch_display, arch_tag, seed) tuples — one per column of P (55 total)
COL_MAP = []
for s in SEEDS:
    for short, disp, tag in ARCHS:
        COL_MAP.append({"short": short, "display": disp, "tag": tag, "seed": s})


def four_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_per_seed_predictions():
    """Load 55 per-seed test-set prediction CSVs. Returns (y_true, P) where P is (N, 55)."""
    rows = []
    y_true_ref = None
    for s in SEEDS:
        for short, disp, tag in ARCHS:
            path = os.path.join(BASE, f"predictions_test_{tag}_s{s}.csv")
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
            if y_true_ref is None:
                y_true_ref = arr[:, 0].astype(int)
            else:
                assert np.array_equal(y_true_ref, arr[:, 0].astype(int)), \
                    f"seed {s} {tag} y_true mismatch"
            rows.append(arr[:, 1].astype(np.float32))
    return y_true_ref, np.stack(rows, axis=1)  # (N, 55)


def f1_neg(weights, probs, y):
    p = probs @ weights
    return -f1_score(y, (p >= THR).astype(int), zero_division=0)


def fit_weights_slsqp(P, y):
    """Find weights via SLSQP over (≥0, Σ=1) on full data; multi-start."""
    n_arch = P.shape[1]
    w0 = np.ones(n_arch) / n_arch
    bounds = [(0.0, 1.0)] * n_arch
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    rng = np.random.default_rng(0)
    starts = [w0] + [rng.dirichlet(np.ones(n_arch)) for _ in range(9)]
    best_w, best_f1 = None, -np.inf
    for s in starts:
        try:
            res = minimize(f1_neg, s, args=(P, y),
                           method="SLSQP", bounds=bounds, constraints=constraints,
                           options={"maxiter": 200, "ftol": 1e-6})
            if -res.fun > best_f1:
                best_f1, best_w = -res.fun, res.x
        except Exception:
            pass
    return best_w if best_w is not None else w0


def m1_simple_avg(P, y):
    """Uniform averaging — no fitting, no CV needed (still report full-test as upper bound)."""
    weights = np.ones(P.shape[1]) / P.shape[1]
    p = P @ weights
    return {
        "name": "M1 Simple average",
        "weights": weights.tolist(),
        "test_metrics": four_metrics(y, (p >= THR).astype(int)),
        "cv_test_metrics": four_metrics(y, (p >= THR).astype(int)),  # same
    }


def m2_weighted_avg_cv(P, y):
    """Fit weights per-fold; predict held-out; concat; report."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    n = len(y)
    p_oof = np.zeros(n, dtype=np.float64)
    fold_weights = []
    for fold_idx, (tr, va) in enumerate(skf.split(P, y)):
        w = fit_weights_slsqp(P[tr], y[tr])
        p_oof[va] = P[va] @ w
        fold_weights.append(w.tolist())

    # Also fit on full data for "in-sample" upper bound
    w_full = fit_weights_slsqp(P, y)
    p_full = P @ w_full

    return {
        "name": "M2 Weighted average (SLSQP, 5-fold CV)",
        "fold_weights": fold_weights,
        "full_weights": w_full.tolist(),
        "cv_test_metrics": four_metrics(y, (p_oof >= THR).astype(int)),
        "full_test_metrics": four_metrics(y, (p_full >= THR).astype(int)),
        "cv_test_probs_oof": p_oof.tolist(),  # for diagnostics
    }


def m3_lr_stacking_cv(P, y):
    """LR L2 with 5-fold CV. Predict on held-out fold; concat."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    n = len(y)
    p_oof = np.zeros(n, dtype=np.float64)
    fold_C = []
    for fold_idx, (tr, va) in enumerate(skf.split(P, y)):
        # Inner CV for C selection
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_idx)
        Cs = [0.001, 0.01, 0.1, 1.0, 10.0]
        best_C, best_inner_f1 = None, -np.inf
        for C in Cs:
            cv_f1s = []
            for itr, iva in inner.split(P[tr], y[tr]):
                m = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                       max_iter=2000, random_state=0)
                m.fit(P[tr][itr], y[tr][itr])
                p_in = m.predict_proba(P[tr][iva])[:, 1]
                cv_f1s.append(f1_score(y[tr][iva], (p_in >= THR).astype(int),
                                       zero_division=0))
            mean_f1 = np.mean(cv_f1s)
            if mean_f1 > best_inner_f1:
                best_inner_f1, best_C = mean_f1, C
        fold_C.append(float(best_C))
        m = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs",
                               max_iter=2000, random_state=0)
        m.fit(P[tr], y[tr])
        p_oof[va] = m.predict_proba(P[va])[:, 1]

    # Full-data refit for upper bound
    m_full = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                max_iter=2000, random_state=0)
    m_full.fit(P, y)
    p_full = m_full.predict_proba(P)[:, 1]

    return {
        "name": "M3 LR stacking (L2, 5-fold CV + inner C selection)",
        "fold_C": fold_C,
        "coefs": m_full.coef_.flatten().tolist(),
        "intercept": float(m_full.intercept_[0]),
        "cv_test_metrics": four_metrics(y, (p_oof >= THR).astype(int)),
        "full_test_metrics": four_metrics(y, (p_full >= THR).astype(int)),
    }


def main():
    print("=== Stack11: 11 architectures × 5 seeds → 3 meta-learners (5-fold CV) ===\n")
    y, P = load_per_seed_predictions()
    print(f"Loaded {P.shape[1]} per-seed predictions for {P.shape[0]} test samples")

    print("\n=== M1 Simple average ===")
    m1 = m1_simple_avg(P, y)
    print(f"  Test F1={m1['test_metrics']['F1-Score']:.4f}")

    print("\n=== M2 Weighted average (SLSQP, 5-fold CV) ===")
    m2 = m2_weighted_avg_cv(P, y)
    print(f"  CV-Test F1={m2['cv_test_metrics']['F1-Score']:.4f}  (5-fold OOF)")
    print(f"  Full-Test F1={m2['full_test_metrics']['F1-Score']:.4f}  (in-sample upper bound)")
    w = np.array(m2["full_weights"])
    top = np.argsort(-w)[:5]
    print("  Top-5 weighted archs (full-data fit):")
    for idx in top:
        cm = COL_MAP[idx]
        print(f"    {cm['display']:25s} (s={cm['seed']})  w={w[idx]:.4f}")

    print("\n=== M3 LR stacking (L2, 5-fold CV) ===")
    m3 = m3_lr_stacking_cv(P, y)
    print(f"  CV-Test F1={m3['cv_test_metrics']['F1-Score']:.4f}  (5-fold OOF)")
    print(f"  Full-Test F1={m3['full_test_metrics']['F1-Score']:.4f}  (in-sample upper bound)")
    print(f"  Per-fold C: {m3['fold_C']}")

    # Save outputs
    methods = [m1, m2, m3]
    out_json = {
        "task": "Stack11: 11-arch × 5-seed × 3-meta-learner (CV-based)",
        "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "threshold": THR,
        "n_archs": len(ARCHS),
        "n_seeds": len(SEEDS),
        "n_folds": N_FOLDS,
        "eval_protocol": "5-fold CV (meta-learner OOF predictions, no test contamination)",
        "archs": [{"short": a[0], "display": a[1], "tag": a[2]} for a in ARCHS],
        "methods": methods,
    }
    json_path = os.path.join(BASE, "stack_11arch_3method.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    # CSV with all 4-metric views
    csv_path = os.path.join(BASE, "stack_11arch_per_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "view", "Accuracy", "Precision", "Recall", "F1-Score"])
        for m in methods:
            if "cv_test_metrics" in m:
                for k, v in m["cv_test_metrics"].items():
                    pass
                w.writerow([m["name"], "5fold_cv_test",
                            m["cv_test_metrics"]["Accuracy"],
                            m["cv_test_metrics"]["Precision"],
                            m["cv_test_metrics"]["Recall"],
                            m["cv_test_metrics"]["F1-Score"]])
            if "full_test_metrics" in m:
                w.writerow([m["name"], "full_test_insample",
                            m["full_test_metrics"]["Accuracy"],
                            m["full_test_metrics"]["Precision"],
                            m["full_test_metrics"]["Recall"],
                            m["full_test_metrics"]["F1-Score"]])
            if "test_metrics" in m and "cv_test_metrics" not in m:
                w.writerow([m["name"], "test_uniform",
                            m["test_metrics"]["Accuracy"],
                            m["test_metrics"]["Precision"],
                            m["test_metrics"]["Recall"],
                            m["test_metrics"]["F1-Score"]])

    # Markdown leaderboard
    md_path = os.path.join(BASE, "STACK_11ARCH_LEADERBOARD.md")
    lines = []
    lines.append("# Stack11: 11-Architecture × 5-Seed × 3 Meta-Learner Leaderboard\n")
    lines.append("**生成时间**: 2026-09-16  ")
    lines.append("**数据集**: 23-dim × window=16 (IanArffDataset v2)  ")
    lines.append("**架构数**: 11 (9 深度 + LightGBM + RandomForest)  ")
    lines.append("**每架构 seed**: 5 (level-1 列数 = 11 × 5 = 55)  ")
    lines.append("**阈值**: 0.5  ")
    lines.append(f"**评估协议**: 5-fold StratifiedKFold on test — 元学习器 OOF 预测，无 test 泄漏  ")
    lines.append(f"**Test 样本数**: N = {len(y)}\n")

    lines.append("## 1. Meta-Learner 4-Metric 对比（CV-based honest test）\n")
    lines.append("| Rank | Method | Accuracy | Precision | Recall | F1-Score |")
    lines.append("|------|--------|---------:|----------:|-------:|---------:|")
    sorted_methods = sorted(methods, key=lambda m: -m.get("cv_test_metrics", m.get("test_metrics"))["F1-Score"])
    for i, m in enumerate(sorted_methods, 1):
        tm = m.get("cv_test_metrics", m.get("test_metrics"))
        lines.append(f"| {i} | {m['name']} | {tm['Accuracy']:.4f} | "
                     f"{tm['Precision']:.4f} | {tm['Recall']:.4f} | {tm['F1-Score']:.4f} |")
    lines.append("")

    # In-sample upper bound
    lines.append("## 2. In-Sample Upper Bound (full-test refit, overestimates)\n")
    lines.append("| Method | Accuracy | Precision | Recall | F1-Score |")
    lines.append("|--------|---------:|----------:|-------:|---------:|")
    for m in methods:
        tm = m.get("full_test_metrics", m.get("test_metrics"))
        lines.append(f"| {m['name']} | {tm['Accuracy']:.4f} | "
                     f"{tm['Precision']:.4f} | {tm['Recall']:.4f} | {tm['F1-Score']:.4f} |")
    lines.append("")

    # M2 weights
    lines.append("## 3. M2 (Weighted Avg) — Top 8 Arch×Seed Weights (full-data fit)\n")
    w = np.array(m2["full_weights"])
    top = np.argsort(-w)[:8]
    lines.append("| Rank | Architecture (seed) | Weight |")
    lines.append("|------|---------------------|-------:|")
    for i, idx in enumerate(top, 1):
        cm = COL_MAP[idx]
        lines.append(f"| {i} | {cm['display']} (s={cm['seed']}) | {w[idx]:.4f} |")
    lines.append("")

    # M3 coefs
    lines.append("## 4. M3 (LR Stacking) — Top 8 Coefs by |coefficient|\n")
    coefs = np.array(m3["coefs"])
    top_abs = np.argsort(-np.abs(coefs))[:8]
    lines.append("| Rank | Architecture (seed) | Coefficient |")
    lines.append("|------|---------------------|------------:|")
    for i, idx in enumerate(top_abs, 1):
        cm = COL_MAP[idx]
        lines.append(f"| {i} | {cm['display']} (s={cm['seed']}) | {coefs[idx]:+.4f} |")
    lines.append("")

    lines.append("## 5. 关键发现\n")
    best = sorted_methods[0]
    base_f1 = 0.8795  # TCN baseline (5-seed prob_mean)
    improvement = (best.get("cv_test_metrics", best.get("test_metrics"))["F1-Score"] - base_f1) * 100
    lines.append(f"1. **最佳 meta-learner (CV honest)**: {best['name']} — Test F1 = "
                 f"{best.get('cv_test_metrics', best.get('test_metrics'))['F1-Score']:.4f}")
    lines.append(f"2. **vs TCN 基线** (F1={base_f1}): {improvement:+.2f}%")
    lines.append(f"3. **CV vs in-sample**: 差距反映 meta-learner 对训练集的拟合程度")
    lines.append(f"4. **M2 权重分布**: 反映各架构贡献度；非零权重 arch 都提供独立信号")
    lines.append(f"5. **M3 LR stacking**: 通过 L2 正则 + 内层 CV 选 C")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()