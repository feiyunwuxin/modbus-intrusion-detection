#!/usr/bin/env python3
"""Random Forest on flattened 23-dim × window=16, 5 seeds.

Each window (16 timesteps × 23 features) is flattened to a 368-d feature vector.

Outputs predictions in same format as deep learning models
(y_true,prob_attack) so they can be consumed by stack_meta_learner.py.
"""
import os
import time
import csv
import numpy as np
from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS

TAG = "rf_23dim_w16"
N_FEATURES_RAW = 23 * 16  # 368 flattened features


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """X: (N, 23, 16) -> (N, 368). Clip already applied in load_data."""
    return X.reshape(X.shape[0], -1)


def train_one_seed_rf(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data_23dim_w16()
    Xtr = flatten_windows(X_tr)
    Xva = flatten_windows(X_va)
    Xte = flatten_windows(X_te)

    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(Xtr, y_tr)
    train_time = time.time() - t0

    # Predict probabilities for test set
    test_prob = model.predict_proba(Xte)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    metrics = {
        "seed": int(seed),
        "n_estimators": 200,
        "max_depth": 12,
        "train_time_s": float(train_time),
        "test_macro_f1": float(f1_score(y_te, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(y_te, test_pred, average="binary")),
        "test_accuracy": float((test_pred == y_te).mean()),
        "test_roc_auc": float(roc_auc_score(y_te, test_prob)),
        "test_pr_auc": float(average_precision_score(y_te, test_prob)),
    }

    # Save predictions CSV
    pred_path = os.path.join(BASE_PATH, f"predictions_test_{TAG}_s{seed}.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "prob_attack"])
        for y, p in zip(y_te.tolist(), test_prob.tolist()):
            w.writerow([y, p])

    print(f"  seed={seed}: F1m={metrics['test_macro_f1']:.4f}  "
          f"max_depth={metrics['max_depth']}  time={train_time:.1f}s")
    return metrics


def main():
    print(f"=== Random Forest 23-dim × window=16 (flattened 368 features) × 5 seeds ===")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_rf(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="Random Forest", tag=TAG,
                   config={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 10,
                           "max_features": "sqrt", "n_features_input": N_FEATURES_RAW})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()