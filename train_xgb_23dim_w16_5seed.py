#!/usr/bin/env python3
"""XGBoost on flattened 23-dim × window=16, 5 seeds.

Same input format as LGB/RF — flatten 23×16 window to 368-d feature vector.
Uses XGBClassifier (sklearn API) for early stopping on validation set.

Outputs predictions in same format (y_true,prob_attack) so it can be consumed
by stack_meta_learner.py as a 12th base learner.

Hyperparams chosen for diversity vs LGB:
  - LGB: n=200, lr=0.05, max_depth=6, num_leaves=63
  - XGB: n=300, lr=0.05, max_depth=5, hist grow policy (vs depth-wise for LGB)
"""
import os
import time
import csv
import numpy as np
from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS

TAG = "xgb_23dim_w16"
N_FEATURES_RAW = 23 * 16


def flatten_windows(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def train_one_seed_xgb(seed: int):
    import xgboost as xgb
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data_23dim_w16()
    Xtr = flatten_windows(X_tr)
    Xva = flatten_windows(X_va)
    Xte = flatten_windows(X_te)

    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=20,
        verbosity=0,
    )
    model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
    train_time = time.time() - t0

    test_prob = model.predict_proba(Xte)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "seed": int(seed),
        "best_iter": int(model.best_iteration) if hasattr(model, "best_iteration") else -1,
        "train_time_s": float(train_time),
        "test_macro_f1": float(f1_score(y_te, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(y_te, test_pred, average="binary")),
        "test_accuracy": float((test_pred == y_te).mean()),
        "test_roc_auc": float(roc_auc_score(y_te, test_prob)),
        "test_pr_auc": float(average_precision_score(y_te, test_prob)),
    }

    pred_path = os.path.join(BASE_PATH, f"predictions_test_{TAG}_s{seed}.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack"])
        for y, p in zip(y_te.tolist(), test_prob.tolist()):
            w.writerow([y, p])

    print(f"  seed={seed}: F1m={metrics['test_macro_f1']:.4f}  "
          f"best_iter={metrics['best_iter']}  time={train_time:.1f}s")
    return metrics


def main():
    print(f"=== XGBoost 23-dim × window=16 (flattened 368 features) × 5 seeds ===\n"
          f"  Hyperparams: n=300, lr=0.05, max_depth=5, gamma=0.1, reg_alpha=0.1")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_xgb(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="XGBoost", tag=TAG,
                   config={"n_estimators": 300, "lr": 0.05, "max_depth": 5,
                           "min_child_weight": 3, "subsample": 0.8,
                           "colsample_bytree": 0.8, "gamma": 0.1,
                           "reg_alpha": 0.1, "reg_lambda": 1.0,
                           "tree_method": "hist",
                           "n_features_input": N_FEATURES_RAW})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()