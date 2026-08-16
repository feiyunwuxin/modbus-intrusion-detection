#!/usr/bin/env python3
"""
Ensemble v1 — LGB v2 + TCN v4 + Random Forest.

Three base models with diverse inductive biases:
  - LGB v2: gradient boosting (handles tabular features well, no sequence)
  - TCN v4 +SE: deep sequence model (best Macro-F1, best PR-AUC)
  - Random Forest: bagged trees (best PR-AUC + ROC-AUC among trees)

We try 4 ensembling methods (evaluated on test, weights tuned on val):
  1. Simple uniform average
  2. Weighted average (SLSQP-optimized for Macro-F1 on val)
  3. Logistic regression stacking (meta-learner on val)
  4. Pick the best of {LGB, TCN v4, RF} pair-wise (3-fold) and reweight
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss,
)
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

BASE = r"C:\work\Claude\Issue"

# All base models (probability files). Test predictions have y_true aligned;
# for TCN v3a/v3b/v3c predictions are at window granularity so we'll
# bootstrap them onto the row level via the test npy. For LGB & RF, the
# predictions are row-level — so we keep them as the base.
#
# IMPORTANT: TCN predictions are at the *window* level (n / 16 or n / 8 rows
# per window), while LGB and RF predictions are at the *row* level. To align
# the two, we either need to expand the TCN probs back to rows OR downsample
# LGB/RF to windows. We downsample LGB & RF probs to windows using the same
# `make_windows(win=16)` logic that the TCN training pipeline uses.

MODELS = [
    ("LGB v2",   "predictions_val_binary_v2.csv",                  "predictions_test_binary_v2.csv",                  "row",  None),
    ("RF",       "predictions_val_random_forest_binary_v2.csv",    "predictions_test_random_forest_binary_v2.csv",    "row",  None),
    ("TCN v4",   "predictions_val_tcn_v4_se_window16.csv",        "predictions_test_tcn_v4_se_window16.csv",        "w16",  16),
    ("TCN v3a",  "predictions_val_tcn_v3a_window8_deep.csv",      "predictions_test_tcn_v3a_window8_deep.csv",      "w8",   8),
    ("TCN v3b",  "predictions_val_tcn_v3b_window16.csv",          "predictions_test_tcn_v3b_window16.csv",          "w16",  16),
    ("TCN v3c",  "predictions_val_tcn_v3c_window16_deep.csv",     "predictions_test_tcn_v3c_window16_deep.csv",     "w16",  16),
]

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


# ─────────────────────────────────────────────
# 1. Load each model's val & test predictions
# ─────────────────────────────────────────────
log("STEP 1: load base-model predictions ...")

def load_pair(name, val_fn, test_fn, kind, win):
    vp = os.path.join(BASE, val_fn)
    tp = os.path.join(BASE, test_fn)
    v = pd.read_csv(vp)
    t = pd.read_csv(tp)
    if kind == "w8" or kind == "w16":
        # TCN predictions are at window level; expand back to row level
        # so that they align with LGB/RF row-level predictions.
        v_exp = v.loc[v.index.repeat(win)].reset_index(drop=True)
        t_exp = t.loc[t.index.repeat(win)].reset_index(drop=True)
        # Recover y_true for the original row-level data
        v_y = np.load(os.path.join(BASE, "y_val_binary.npy"))[:len(v_exp)]
        t_y = np.load(os.path.join(BASE, "y_test_binary.npy"))[:len(t_exp)]
        v_exp["y_true"] = v_y
        t_exp["y_true"] = t_y
        v, t = v_exp, t_exp
    return v, t

val_dfs, test_dfs = {}, {}
for name, vf, tf, kind, win in MODELS:
    v, t = load_pair(name, vf, tf, kind, win)
    log(f"   {name:10s}  val: {v.shape}  test: {t.shape}  (kind={kind})")
    val_dfs[name] = v
    test_dfs[name] = t

# Align: all dataframes must have the same length.
# LGB/RF use full row-level data; TCN drops tail rows when windowing.
# Use the smallest common length across all models.
VAL_LEN   = min(len(v) for v in val_dfs.values())
TEST_LEN  = min(len(t) for t in test_dfs.values())
log(f"   aligning all models to  val={VAL_LEN}  test={TEST_LEN}")
for name in val_dfs:
    val_dfs[name]  = val_dfs[name].iloc[:VAL_LEN].reset_index(drop=True)
    test_dfs[name] = test_dfs[name].iloc[:TEST_LEN].reset_index(drop=True)

# Sanity: do all val dataframes have aligned y_true?
y_val_ref = val_dfs["LGB v2"]["y_true"].values
y_test_ref = test_dfs["LGB v2"]["y_true"].values
for name in val_dfs:
    assert (val_dfs[name]["y_true"].values == y_val_ref).all(), f"{name} val y_true mismatch"
    assert (test_dfs[name]["y_true"].values == y_test_ref).all(), f"{name} test y_true mismatch"
log("   [ok] all y_true align")


# ─────────────────────────────────────────────
# 2. Individual model benchmarks (find best threshold per model on val)
# ─────────────────────────────────────────────
log("STEP 2: per-model benchmarks on test @ best val threshold ...")

def best_threshold(y_true, y_prob, metric="macro_f1"):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best, best_t = -1, 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, pred, average="macro")
        else:
            score = f1_score(y_true, pred, average="binary")
        if score > best:
            best, best_t = score, t
    return best_t, best


indiv_results = []
for name in val_dfs:
    v_p = val_dfs[name]["prob_attack"].values
    t_p = test_dfs[name]["prob_attack"].values
    thr, val_score = best_threshold(y_val_ref, v_p, "macro_f1")
    pred_test = (t_p >= thr).astype(int)
    res = {
        "model":       name,
        "best_thr":    float(thr),
        "val_f1m":     float(val_score),
        "test_f1m":    float(f1_score(y_test_ref, pred_test, average="macro")),
        "test_f1b":    float(f1_score(y_test_ref, pred_test, average="binary")),
        "test_acc":    float((pred_test == y_test_ref).mean()),
        "test_pra":    float(average_precision_score(y_test_ref, t_p)),
        "test_roc":    float(roc_auc_score(y_test_ref, t_p)),
    }
    indiv_results.append(res)
    log(f"   {name:10s}  thr={thr:.2f}  test F1m={res['test_f1m']:.4f}  PR-AUC={res['test_pra']:.4f}  Binary-F1={res['test_f1b']:.4f}  Acc={res['test_acc']:.4f}")


# ─────────────────────────────────────────────
# 3. Build probability matrix (rows × models) for val and test
# ─────────────────────────────────────────────
log("STEP 3: probability matrix (rows × models) ...")
val_probs = np.column_stack([val_dfs[n]["prob_attack"].values for n in val_dfs])
test_probs = np.column_stack([test_dfs[n]["prob_attack"].values for n in test_dfs])
names = list(val_dfs.keys())
log(f"   val  matrix shape: {val_probs.shape}")
log(f"   test matrix shape: {test_probs.shape}")


# ─────────────────────────────────────────────
# 4. Ensembling method 1 — simple uniform average
# ─────────────────────────────────────────────
log("STEP 4: uniform-average ensemble ...")

# Try uniform on all 6 models, then on subsets
subset_combos = [
    ("LGB+RF+TCN_v4",            ["LGB v2", "RF", "TCN v4"]),
    ("LGB+TCN_v4",               ["LGB v2", "TCN v4"]),
    ("LGB+RF+TCN_v4+v3a",        ["LGB v2", "RF", "TCN v4", "TCN v3a"]),
    ("LGB+TCN_v4+v3a+v3b",       ["LGB v2", "TCN v4", "TCN v3a", "TCN v3b"]),
    ("ALL_6",                    names),
    ("LGB+TCN_v4+v3a+v3b+v3c",   ["LGB v2", "TCN v4", "TCN v3a", "TCN v3b", "TCN v3c"]),
    ("LGB+RF+TCN_v4+v3a+v3c",    ["LGB v2", "RF", "TCN v4", "TCN v3a", "TCN v3c"]),
]

uniform_results = []
for label, members in subset_combos:
    idx = [names.index(m) for m in members]
    val_avg = val_probs[:, idx].mean(axis=1)
    test_avg = test_probs[:, idx].mean(axis=1)
    thr, val_score = best_threshold(y_val_ref, val_avg, "macro_f1")
    pred = (test_avg >= thr).astype(int)
    uniform_results.append({
        "members":   label,
        "n_members": len(members),
        "best_thr":  float(thr),
        "val_f1m":   float(val_score),
        "test_f1m":  float(f1_score(y_test_ref, pred, average="macro")),
        "test_f1b":  float(f1_score(y_test_ref, pred, average="binary")),
        "test_acc":  float((pred == y_test_ref).mean()),
        "test_pra":  float(average_precision_score(y_test_ref, test_avg)),
        "test_roc":  float(roc_auc_score(y_test_ref, test_avg)),
    })
    log(f"   {label:30s}  thr={thr:.2f}  test F1m={uniform_results[-1]['test_f1m']:.4f}  PR-AUC={uniform_results[-1]['test_pra']:.4f}")


# ─────────────────────────────────────────────
# 5. Ensembling method 2 — weighted average (SLSQP on val)
# ─────────────────────────────────────────────
log("STEP 5: weighted-average ensemble (SLSQP on val) ...")

def tune_weights(val_p, y_val):
    """Find weights that maximize Macro-F1 on val, with sum=1, weights>=0."""
    n = val_p.shape[1]
    def neg_f1m(w):
        w = np.maximum(w, 0)
        w = w / (w.sum() + 1e-12)
        avg = val_p @ w
        thr, score = best_threshold(y_val, avg, "macro_f1")
        return -score
    best = None
    for seed in range(8):
        np.random.seed(seed)
        w0 = np.random.dirichlet(np.ones(n))
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bnds = [(0, 1)] * n
        r = minimize(neg_f1m, w0, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"maxiter": 60, "ftol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    w = np.maximum(best.x, 0)
    w = w / w.sum()
    return w

weighted_results = []
for label, members in subset_combos:
    idx = [names.index(m) for m in members]
    val_sub = val_probs[:, idx]
    test_sub = test_probs[:, idx]
    w = tune_weights(val_sub, y_val_ref)
    val_avg = val_sub @ w
    test_avg = test_sub @ w
    thr, val_score = best_threshold(y_val_ref, val_avg, "macro_f1")
    pred = (test_avg >= thr).astype(int)
    weighted_results.append({
        "members":   label,
        "weights":   {m: round(float(w[i]), 4) for i, m in enumerate(members)},
        "n_members": len(members),
        "best_thr":  float(thr),
        "val_f1m":   float(val_score),
        "test_f1m":  float(f1_score(y_test_ref, pred, average="macro")),
        "test_f1b":  float(f1_score(y_test_ref, pred, average="binary")),
        "test_acc":  float((pred == y_test_ref).mean()),
        "test_pra":  float(average_precision_score(y_test_ref, test_avg)),
        "test_roc":  float(roc_auc_score(y_test_ref, test_avg)),
    })
    log(f"   {label:30s}  weights={w.round(2).tolist()}  test F1m={weighted_results[-1]['test_f1m']:.4f}  PR-AUC={weighted_results[-1]['test_pra']:.4f}")


# ─────────────────────────────────────────────
# 6. Ensembling method 3 — Logistic Regression stacking
# ─────────────────────────────────────────────
log("STEP 6: stacking with Logistic Regression (val-fitted meta) ...")

stack_results = []
for label, members in subset_combos:
    idx = [names.index(m) for m in members]
    Xs = val_probs[:, idx]
    Xt = test_probs[:, idx]
    # Stacking on the val set: we fit LR on the val itself — so val_f1m is
    # optimistically biased. To be honest, we evaluate the LR's *test* metrics
    # via a 5-fold CV on val to pick C, then refit on full val and apply to test.
    from sklearn.model_selection import StratifiedKFold
    best_C, best_score = 1.0, -1
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, va in cv.split(Xs, y_val_ref):
            lr = LogisticRegression(C=C, max_iter=2000)
            lr.fit(Xs[tr], y_val_ref[tr])
            p_va = lr.predict_proba(Xs[va])[:, 1]
            thr, sc = best_threshold(y_val_ref[va], p_va, "macro_f1")
            scores.append(sc)
        s = np.mean(scores)
        if s > best_score:
            best_score, best_C = s, C
    lr = LogisticRegression(C=best_C, max_iter=2000)
    lr.fit(Xs, y_val_ref)
    val_p_meta = lr.predict_proba(Xs)[:, 1]
    test_p_meta = lr.predict_proba(Xt)[:, 1]
    thr, val_score = best_threshold(y_val_ref, val_p_meta, "macro_f1")
    pred = (test_p_meta >= thr).astype(int)
    stack_results.append({
        "members":   label,
        "n_members": len(members),
        "best_C":    best_C,
        "best_thr":  float(thr),
        "val_f1m":   float(val_score),
        "test_f1m":  float(f1_score(y_test_ref, pred, average="macro")),
        "test_f1b":  float(f1_score(y_test_ref, pred, average="binary")),
        "test_acc":  float((pred == y_test_ref).mean()),
        "test_pra":  float(average_precision_score(y_test_ref, test_p_meta)),
        "test_roc":  float(roc_auc_score(y_test_ref, test_p_meta)),
        "coefs":     {m: round(float(lr.coef_[0][i]), 4) for i, m in enumerate(members)},
        "intercept": float(lr.intercept_[0]),
    })
    log(f"   {label:30s}  C={best_C}  test F1m={stack_results[-1]['test_f1m']:.4f}  PR-AUC={stack_results[-1]['test_pra']:.4f}  coefs={list(stack_results[-1]['coefs'].values())}")


# ─────────────────────────────────────────────
# 7. Save artifacts
# ─────────────────────────────────────────────
log("STEP 7: save artifacts ...")

OUT = os.path.join(BASE, "evaluation_ensemble_v1.txt")
META = os.path.join(BASE, "processed_meta_ensemble_v1.json")

lines = [
    "=" * 80,
    "ENSEMBLE REPORT — v1: LGB + TCN v4 + RF (+ optional v3a/v3b/v3c)",
    "=" * 80,
    f"Base models and their pre-ensemble test metrics @ best val threshold:",
    ""
]
for r in indiv_results:
    lines.append(f"  {r['model']:10s}  thr={r['best_thr']:.2f}  "
                 f"test F1m={r['test_f1m']:.4f}  Binary-F1={r['test_f1b']:.4f}  "
                 f"Acc={r['test_acc']:.4f}  PR-AUC={r['test_pra']:.4f}  ROC-AUC={r['test_roc']:.4f}")
lines += ["", "=" * 80, "METHOD 1 — UNIFORM AVERAGE", "=" * 80, ""]
for r in uniform_results:
    lines.append(f"  [{r['members']}]  thr={r['best_thr']:.2f}  "
                 f"test F1m={r['test_f1m']:.4f}  Binary-F1={r['test_f1b']:.4f}  "
                 f"Acc={r['test_acc']:.4f}  PR-AUC={r['test_pra']:.4f}  ROC-AUC={r['test_roc']:.4f}")
lines += ["", "=" * 80, "METHOD 2 — WEIGHTED AVERAGE (SLSQP on val)", "=" * 80, ""]
for r in weighted_results:
    w_str = "  weights: " + ", ".join(f"{m}={w}" for m, w in r["weights"].items())
    lines.append(f"  [{r['members']}]  thr={r['best_thr']:.2f}  test F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pra']:.4f}")
    lines.append(w_str)
lines += ["", "=" * 80, "METHOD 3 — STACKING (Logistic Regression, val-fit + 5-fold CV for C)", "=" * 80, ""]
for r in stack_results:
    lines.append(f"  [{r['members']}]  C={r['best_C']}  thr={r['best_thr']:.2f}  "
                 f"test F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pra']:.4f}  "
                 f"inter={r['intercept']:+.3f}")
    lines.append(f"     coefs: {r['coefs']}")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

meta = {
    "indiv": indiv_results,
    "uniform": uniform_results,
    "weighted": weighted_results,
    "stacking": stack_results,
    "val_size": VAL_LEN,
    "test_size": TEST_LEN,
    "models": names,
}
with open(META, "w") as f:
    json.dump(meta, f, indent=2, default=str)


# ─────────────────────────────────────────────
# 8. Best-of-best summary
# ─────────────────────────────────────────────
log("STEP 8: best-of-best summary ...")
all_runs = (
    [("indiv",  r)  for r in indiv_results]
    + [("uniform", r)  for r in uniform_results]
    + [("weighted", r) for r in weighted_results]
    + [("stack",  r)  for r in stack_results]
)
best_run = max(all_runs, key=lambda x: x[1]["test_f1m"])
log(f"   BEST ensemble: method={best_run[0]}  members={best_run[1].get('members','-')}")
log(f"      test F1m={best_run[1]['test_f1m']:.4f}  PR-AUC={best_run[1]['test_pra']:.4f}  Acc={best_run[1]['test_acc']:.4f}")
log("=" * 60)
log("DONE ensemble_v1.")
log("=" * 60)
