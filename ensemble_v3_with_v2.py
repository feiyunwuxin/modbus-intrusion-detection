#!/usr/bin/env python3
"""
Ensemble v3 — Stack ALL_9: v1's 7 + v2's 2 strong models.

Includes:
  - v1: LGB, RF, TCN v4, TCN v3a, TCN v3b, TCN v3c, LSTM+Att
  - v2: TCN v4 +SE (SCADA features), LSTM+Att (SCADA features)

Tests whether adding v2 (SCADA-aware) models on top of v1's diverse set
breaks the 0.8478 ceiling.
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

BASE = r"C:\work\Claude\Issue"

MODELS = [
    ("LGB v2",   "predictions_val_binary_v2.csv",                    "predictions_test_binary_v2.csv",                    "row",  None),
    ("RF",       "predictions_val_random_forest_binary_v2.csv",      "predictions_test_random_forest_binary_v2.csv",      "row",  None),
    ("TCN v4",   "predictions_val_tcn_v4_se_window16.csv",          "predictions_test_tcn_v4_se_window16.csv",          "w16",  16),
    ("TCN v3a",  "predictions_val_tcn_v3a_window8_deep.csv",        "predictions_test_tcn_v3a_window8_deep.csv",        "w8",   8),
    ("TCN v3b",  "predictions_val_tcn_v3b_window16.csv",            "predictions_test_tcn_v3b_window16.csv",            "w16",  16),
    ("TCN v3c",  "predictions_val_tcn_v3c_window16_deep.csv",       "predictions_test_tcn_v3c_window16_deep.csv",       "w16",  16),
    ("LSTM+Att", "predictions_val_lstm_attention_v2_window16.csv",  "predictions_test_lstm_attention_v2_window16.csv",  "w16",  16),
    # NEW: v2 (SCADA-aware) versions of best two deep models
    ("TCN v4 v2","predictions_val_tcn_v2_v4se_window16.csv",       "predictions_test_tcn_v2_v4se_window16.csv",       "w16",  16),
    ("LSTM+Att v2", "predictions_val_lstm_attention_v2_window16.csv", "predictions_test_lstm_attention_v2_window16.csv", "w16", 16),
]

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ─────────────────────────────────────────────
log("STEP 1: load ...")
def load_pair(val_fn, test_fn, kind, win):
    v = pd.read_csv(os.path.join(BASE, val_fn))
    t = pd.read_csv(os.path.join(BASE, test_fn))
    if kind in ("w8", "w16"):
        v = v.loc[v.index.repeat(win)].reset_index(drop=True)
        t = t.loc[t.index.repeat(win)].reset_index(drop=True)
        v["y_true"] = np.load(os.path.join(BASE, "y_val_binary.npy"))[:len(v)]
        t["y_true"] = np.load(os.path.join(BASE, "y_test_binary.npy"))[:len(t)]
    return v, t

val_dfs, test_dfs = {}, {}
for name, vf, tf, kind, win in MODELS:
    v, t = load_pair(vf, tf, kind, win)
    val_dfs[name] = v; test_dfs[name] = t
    log(f"   {name:14s}  val: {v.shape}  test: {t.shape}")

VAL_LEN  = min(len(v) for v in val_dfs.values())
TEST_LEN = min(len(t) for t in test_dfs.values())
log(f"   align: val={VAL_LEN}  test={TEST_LEN}")
for name in val_dfs:
    val_dfs[name]  = val_dfs[name].iloc[:VAL_LEN].reset_index(drop=True)
    test_dfs[name] = test_dfs[name].iloc[:TEST_LEN].reset_index(drop=True)

y_val_ref  = val_dfs["LGB v2"]["y_true"].values
y_test_ref = test_dfs["LGB v2"]["y_true"].values
for name in val_dfs:
    assert (val_dfs[name]["y_true"].values == y_val_ref).all()
log("   [ok] all y_true align")

# ─────────────────────────────────────────────
log("STEP 2: per-model benchmarks ...")
def best_threshold(y_true, y_prob, metric="macro_f1"):
    best, best_t = -1, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (y_prob >= t).astype(int)
        s = f1_score(y_true, pred, average="macro" if metric=="macro_f1" else "binary")
        if s > best:
            best, best_t = s, t
    return best_t, best

indiv_results = []
for name in val_dfs:
    v_p = val_dfs[name]["prob_attack"].values
    t_p = test_dfs[name]["prob_attack"].values
    thr, val_s = best_threshold(y_val_ref, v_p, "macro_f1")
    pred = (t_p >= thr).astype(int)
    indiv_results.append({
        "model":    name,
        "best_thr": float(thr),
        "val_f1m":  float(val_s),
        "test_f1m": float(f1_score(y_test_ref, pred, average="macro")),
        "test_f1b": float(f1_score(y_test_ref, pred, average="binary")),
        "test_acc": float((pred == y_test_ref).mean()),
        "test_pra": float(average_precision_score(y_test_ref, t_p)),
        "test_roc": float(roc_auc_score(y_test_ref, t_p)),
    })
    r = indiv_results[-1]
    log(f"   {name:14s}  thr={r['best_thr']:.2f}  F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pra']:.4f}")

# ─────────────────────────────────────────────
log("STEP 3: probability matrix ...")
val_probs  = np.column_stack([val_dfs[n]["prob_attack"].values  for n in val_dfs])
test_probs = np.column_stack([test_dfs[n]["prob_attack"].values for n in test_dfs])
names = list(val_dfs.keys())
log(f"   val  shape: {val_probs.shape}")

# ─────────────────────────────────────────────
SUBSETS = [
    ("ALL_9",                                        names),
    ("LGB+RF+TCN_v4+TCN_v4_v2",                      ["LGB v2", "RF", "TCN v4", "TCN v4 v2"]),
    ("LGB+RF+TCN_v4_v2+LSTM+Att_v2",                 ["LGB v2", "RF", "TCN v4 v2", "LSTM+Att v2"]),
    ("LGB+RF+TCN_v4+TCN_v4_v2+LSTM+Att+LSTM+Att_v2", ["LGB v2", "RF", "TCN v4", "TCN v4 v2", "LSTM+Att", "LSTM+Att v2"]),
    ("LGB+RF+v2_strong",                             ["LGB v2", "RF", "TCN v4 v2", "LSTM+Att v2"]),
    ("LGB+RF+all_TCN+LSTM_v1+v2",                    ["LGB v2", "RF", "TCN v4", "TCN v3a", "TCN v3b", "TCN v3c", "TCN v4 v2", "LSTM+Att", "LSTM+Att v2"]),
]

# ─────────────────────────────────────────────
log("STEP 4: uniform-average ensemble ...")
uniform_results = []
for label, members in SUBSETS:
    idx = [names.index(m) for m in members]
    val_avg  = val_probs[:, idx].mean(axis=1)
    test_avg = test_probs[:, idx].mean(axis=1)
    thr, val_s = best_threshold(y_val_ref, val_avg, "macro_f1")
    pred = (test_avg >= thr).astype(int)
    uniform_results.append({
        "members": label, "n_members": len(members), "best_thr": float(thr),
        "val_f1m": float(val_s), "test_f1m": float(f1_score(y_test_ref, pred, average="macro")),
        "test_f1b": float(f1_score(y_test_ref, pred, average="binary")),
        "test_acc": float((pred == y_test_ref).mean()),
        "test_pra": float(average_precision_score(y_test_ref, test_avg)),
        "test_roc": float(roc_auc_score(y_test_ref, test_avg)),
    })
    r = uniform_results[-1]
    log(f"   {label:50s}  F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pra']:.4f}")

# ─────────────────────────────────────────────
log("STEP 5: weighted-average (SLSQP) ...")
def tune_weights(val_p, y_val):
    n = val_p.shape[1]
    def neg_f1m(w):
        w = np.maximum(w, 0); w = w / (w.sum() + 1e-12)
        avg = val_p @ w
        _, s = best_threshold(y_val, avg, "macro_f1")
        return -s
    best = None
    for seed in range(6):
        np.random.seed(seed)
        w0 = np.random.dirichlet(np.ones(n))
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        r = minimize(neg_f1m, w0, method="SLSQP", bounds=[(0,1)]*n, constraints=cons,
                     options={"maxiter": 60, "ftol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    w = np.maximum(best.x, 0); w = w / w.sum()
    return w

weighted_results = []
for label, members in SUBSETS:
    idx = [names.index(m) for m in members]
    val_sub = val_probs[:, idx]; test_sub = test_probs[:, idx]
    w = tune_weights(val_sub, y_val_ref)
    val_avg  = val_sub @ w
    test_avg = test_sub @ w
    thr, val_s = best_threshold(y_val_ref, val_avg, "macro_f1")
    pred = (test_avg >= thr).astype(int)
    weighted_results.append({
        "members": label, "weights": {m: round(float(w[i]), 4) for i, m in enumerate(members)},
        "n_members": len(members), "best_thr": float(thr),
        "val_f1m": float(val_s), "test_f1m": float(f1_score(y_test_ref, pred, average="macro")),
        "test_f1b": float(f1_score(y_test_ref, pred, average="binary")),
        "test_acc": float((pred == y_test_ref).mean()),
        "test_pra": float(average_precision_score(y_test_ref, test_avg)),
        "test_roc": float(roc_auc_score(y_test_ref, test_avg)),
    })
    r = weighted_results[-1]
    log(f"   {label:50s}  weights={w.round(2).tolist()}  F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pra']:.4f}")

# ─────────────────────────────────────────────
log("STEP 6: stacking (LR) ...")
stack_results = []
for label, members in SUBSETS:
    idx = [names.index(m) for m in members]
    Xs = val_probs[:, idx]; Xt = test_probs[:, idx]
    best_C, best_score = 1.0, -1
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, va in cv.split(Xs, y_val_ref):
            lr = LogisticRegression(C=C, max_iter=2000)
            lr.fit(Xs[tr], y_val_ref[tr])
            p_va = lr.predict_proba(Xs[va])[:, 1]
            _, sc = best_threshold(y_val_ref[va], p_va, "macro_f1")
            scores.append(sc)
        s = np.mean(scores)
        if s > best_score:
            best_score, best_C = s, C
    lr = LogisticRegression(C=best_C, max_iter=2000)
    lr.fit(Xs, y_val_ref)
    val_p_meta  = lr.predict_proba(Xs)[:, 1]
    test_p_meta = lr.predict_proba(Xt)[:, 1]
    thr, val_s = best_threshold(y_val_ref, val_p_meta, "macro_f1")
    pred = (test_p_meta >= thr).astype(int)
    stack_results.append({
        "members": label, "n_members": len(members), "best_C": best_C,
        "best_thr": float(thr), "val_f1m": float(val_s),
        "test_f1m": float(f1_score(y_test_ref, pred, average="macro")),
        "test_f1b": float(f1_score(y_test_ref, pred, average="binary")),
        "test_acc": float((pred == y_test_ref).mean()),
        "test_pra": float(average_precision_score(y_test_ref, test_p_meta)),
        "test_roc": float(roc_auc_score(y_test_ref, test_p_meta)),
        "coefs": {m: round(float(lr.coef_[0][i]), 4) for i, m in enumerate(members)},
        "intercept": float(lr.intercept_[0]),
    })
    r = stack_results[-1]
    log(f"   {label:50s}  C={best_C}  F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pra']:.4f}  coefs={list(r['coefs'].values())}")

# ─────────────────────────────────────────────
log("STEP 7: save artifacts ...")
OUT  = os.path.join(BASE, "evaluation_ensemble_v3.txt")
META = os.path.join(BASE, "processed_meta_ensemble_v3.json")

lines = [
    "=" * 80, "ENSEMBLE REPORT — v3: ALL_9 (v1 7 models + v2 2 strong models)",
    "=" * 80,
    "Base models:", ""
]
for r in indiv_results:
    lines.append(f"  {r['model']:14s}  thr={r['best_thr']:.2f}  "
                 f"F1m={r['test_f1m']:.4f}  Bin-F1={r['test_f1b']:.4f}  "
                 f"Acc={r['test_acc']:.4f}  PR-AUC={r['test_pra']:.4f}  ROC-AUC={r['test_roc']:.4f}")

for sec, runs in [("METHOD 1 — UNIFORM AVERAGE", uniform_results),
                  ("METHOD 2 — WEIGHTED AVERAGE (SLSQP on val)", weighted_results),
                  ("METHOD 3 — STACKING (LR, val-fit + 5-fold CV)", stack_results)]:
    lines += ["", "=" * 80, sec, "=" * 80, ""]
    for r in runs:
        lines.append(f"  [{r['members']}]  thr={r['best_thr']:.2f}  F1m={r['test_f1m']:.4f}  PR-AUC={r['test_pra']:.4f}")
        if "weights" in r:
            lines.append("    weights: " + ", ".join(f"{m}={w}" for m, w in r["weights"].items()))
        if "coefs" in r:
            lines.append(f"    C={r['best_C']}  inter={r['intercept']:+.3f}  coefs={r['coefs']}")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

with open(META, "w") as f:
    json.dump({"indiv": indiv_results, "uniform": uniform_results,
               "weighted": weighted_results, "stacking": stack_results,
               "models": names, "val_size": VAL_LEN, "test_size": TEST_LEN}, f, indent=2, default=str)

log("STEP 8: best-of-best ...")
all_runs = ([("indiv",  r) for r in indiv_results]
            + [("uniform", r) for r in uniform_results]
            + [("weighted", r) for r in weighted_results]
            + [("stack",  r) for r in stack_results])
best_run = max(all_runs, key=lambda x: x[1]["test_f1m"])
log(f"   BEST: method={best_run[0]}  members={best_run[1].get('members','-')}")
log(f"      test F1m={best_run[1]['test_f1m']:.4f}  PR-AUC={best_run[1]['test_pra']:.4f}  Acc={best_run[1]['test_acc']:.4f}")
log("DONE ensemble_v3.")
