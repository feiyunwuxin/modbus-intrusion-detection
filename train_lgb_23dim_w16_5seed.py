#!/usr/bin/env python3
"""LightGBM on flattened 23-dim × window=16, 5 seeds.

Each window (16 timesteps × 23 features) is flattened to a 368-d feature vector
to match the input shape used by sklearn-style tree models.

Outputs predictions in same format as deep learning models
(y_true,prob_attack) so they can be consumed by stack_meta_learner.py.
"""
import os
import time
import csv
import numpy as np
from _common_train import load_data_23dim_w16, save_meta_json, BASE_PATH, SEEDS, KEEP_23, CLIP_VAL

TAG = "lgb_23dim_w16"
N_FEATURES_RAW = 23 * 16  # 368 flattened features


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """X: (N, 23, 16) -> (N, 368). Clip already applied in load_data."""
    return X.reshape(X.shape[0], -1)


def train_one_seed_lgb(seed: int):
    import lightgbm as lgb
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data_23dim_w16()
    Xtr = flatten_windows(X_tr)
    Xva = flatten_windows(X_va)
    Xte = flatten_windows(X_te)

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    train_time = time.time() - t0

    # Predict probabilities for test set
    test_prob = model.predict_proba(Xte)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    metrics = {
        "seed": int(seed),
        "best_iter": int(model.best_iteration_),
        "train_time_s": float(train_time),
        "test_macro_f1": float(f1_score(y_te, test_pred, average="macro")),
        "test_binary_f1": float(f1_score(y_te, test_pred, average="binary")),
        "test_accuracy": float((test_pred == y_te).mean()),
        "test_roc_auc": float(roc_auc_score(y_te, test_prob)),
        "test_pr_auc": float(average_precision_score(y_te, test_prob)),
    }

    # Save predictions CSV (same format as deep learning)
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
    print(f"=== LightGBM 23-dim × window=16 (flattened 368 features) × 5 seeds ===")
    results = []
    for seed in SEEDS:
        results.append(train_one_seed_lgb(seed))

    save_meta_json(results, os.path.join(BASE_PATH, f"processed_meta_{TAG}.json"),
                   model_name="LightGBM", tag=TAG,
                   config={"n_estimators": 200, "lr": 0.05, "max_depth": 6,
                           "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
                           "n_features_input": N_FEATURES_RAW})
    print(f"\n[saved meta] processed_meta_{TAG}.json")


if __name__ == "__main__":
    main()