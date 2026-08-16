#!/usr/bin/env python3
"""
OCC-eSNN (One-Class Classification eSNN) on 19-dim SCADA features.

eSNN (Evolving Spiking Neural Network, Kasabov et al.):
  - Each training sample creates a spiking neuron (rank-order)
  - Synaptic weights = pre-synaptic spike times
  - Output = winner-takes-all (neuron with earliest spike)
  - For one-class: train on NORMAL only, anomaly = large distance to all neurons

Implementation (simplified, window=16, 19-dim):
  1. Train only on y=0 (normal) samples in windowed form
  2. Encode each window as Poisson spike train (rate = normalized feature)
  3. Each training sample → a neuron with weights = first-spike time per channel
  4. Anomaly score = 1 - min distance to any neuron (in normalized feature space)
  5. Threshold tuning on val (using both normal and attack for monitoring)
"""

import os, time, json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

BASE = r"C:\work\Claude\Issue"

X_TR = os.path.join(BASE, "X_train_binary_v2_scada.npy")
X_VA = os.path.join(BASE, "X_val_binary_v2_scada.npy")
X_TE = os.path.join(BASE, "X_test_binary_v2_scada.npy")
Y_TR = os.path.join(BASE, "y_train_binary_v2_scada.npy")
Y_VA = os.path.join(BASE, "y_val_binary_v2_scada.npy")
Y_TE = os.path.join(BASE, "y_test_binary_v2_scada.npy")

EVAL_OUT  = os.path.join(BASE, "evaluation_occ_esnn_v3_19dim.txt")
META_OUT  = os.path.join(BASE, "processed_meta_occ_esnn_v3_19dim.json")
THR_OUT   = os.path.join(BASE, "best_threshold_occ_esnn_v3_19dim.json")
PRED_VAL  = os.path.join(BASE, "predictions_val_occ_esnn_v3_19dim.csv")
PRED_TEST = os.path.join(BASE, "predictions_test_occ_esnn_v3_19dim.csv")
CM_PNG    = os.path.join(BASE, "confusion_matrix_occ_esnn_v3_19dim.png")
PRROC_PNG = os.path.join(BASE, "pr_roc_occ_esnn_v3_19dim.png")

SEED = 42
WINDOW = 16
SIM_THRESH = 0.5   # similarity threshold for adding new neuron (lower = more neurons)
np.random.seed(SEED)

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def make_windows(X, y, win):
    n = (len(X) // win) * win
    Xw = X[:n].reshape(n // win, win, -1)
    yw = (y[:n].reshape(n // win, win).max(axis=1)).astype(np.int64)
    return Xw, yw


class OCCeSNN:
    """One-Class Classification evolving SNN.
    Each neuron = (weight_vector, label) where weight = rank-order spike time per channel.
    Simplified: weights are first-spike latency = 1 - normalized_feature (Poisson rate).
    """
    def __init__(self, sim_thresh=SIM_THRESH, max_neurons=2000):
        self.sim_thresh = sim_thresh
        self.max_neurons = max_neurons
        self.neurons = []  # list of (weight_vec, label)

    def _encode(self, x):
        """Rank-order encoding: center features at 0, then convert to latency in [-1, 1]."""
        x_feat = x.mean(axis=0)  # (F,)
        # center at 0 with tanh (output ∈ [-1, 1])
        latency = np.tanh(x_feat * 0.3)
        return latency

    def _similarity(self, a, b):
        """Cosine similarity."""
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def fit(self, X_normal):
        """Train on normal data only."""
        log(f"   training on {len(X_normal)} normal windows, max_neurons={self.max_neurons}")
        for i, x in enumerate(X_normal):
            w = self._encode(x)
            # check if similar to existing neuron
            added = False
            for nw, _ in self.neurons:
                if self._similarity(w, nw) >= self.sim_thresh:
                    added = True
                    break
            if not added:
                self.neurons.append((w, 0))
                if len(self.neurons) >= self.max_neurons:
                    log(f"   reached max_neurons={self.max_neurons}, stopping")
                    break
        log(f"   neurons created: {len(self.neurons)}")

    def decision_function(self, X):
        """Higher score = more anomalous."""
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            w = self._encode(x)
            if not self.neurons:
                return scores
            sims = np.array([self._similarity(w, nw) for nw, _ in self.neurons])
            scores[i] = 1.0 - sims.max()  # anomaly = max distance
        return scores


# ─────────────────────────────────────────────
log("STEP 1: load + window 19-dim ...")
X_train = np.load(X_TR).astype(np.float32)
X_val   = np.load(X_VA).astype(np.float32)
X_test  = np.load(X_TE).astype(np.float32)
y_train = np.load(Y_TR).astype(np.int64)
y_val   = np.load(Y_VA).astype(np.int64)
y_test  = np.load(Y_TE).astype(np.int64)
log(f"   shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

X_tr_w, y_tr_w = make_windows(X_train, y_train, WINDOW)
X_va_w, y_va_w = make_windows(X_val,   y_val,   WINDOW)
X_te_w, y_te_w = make_windows(X_test,  y_test,  WINDOW)
log(f"   windowed: train={X_tr_w.shape} val={X_va_w.shape} test={X_te_w.shape}")

# use only NORMAL windows for training
X_normal = X_tr_w[y_tr_w == 0]
log(f"   normal-only training windows: {len(X_normal)}")

# ─────────────────────────────────────────────
log("STEP 2: train OCC-eSNN ...")
model = OCCeSNN(sim_thresh=SIM_THRESH, max_neurons=2000)
model.fit(X_normal)

# ─────────────────────────────────────────────
log("STEP 3: scoring ...")
val_scores  = model.decision_function(X_va_w)
test_scores = model.decision_function(X_te_w)
log(f"   val score stats: min={val_scores.min():.4f} max={val_scores.max():.4f} mean={val_scores.mean():.4f}")
log(f"   test score stats: min={test_scores.min():.4f} max={test_scores.max():.4f} mean={test_scores.mean():.4f}")

# ─────────────────────────────────────────────
log("STEP 4: threshold tuning on val ...")
thresholds = np.linspace(0.0, val_scores.max() + 0.01, 100)
sweep = []
for t in thresholds:
    pred = (val_scores >= t).astype(int)
    f1m  = f1_score(y_va_w, pred, average="macro")
    sweep.append((float(t), float(f1m)))
sweep_df = pd.DataFrame(sweep, columns=["threshold","val_macro_f1"])
best_idx  = int(sweep_df["val_macro_f1"].idxmax())
best_thr  = float(sweep_df.loc[best_idx, "threshold"])
best_f1m  = float(sweep_df.loc[best_idx, "val_macro_f1"])
log(f"   best threshold = {best_thr:.4f}  →  val Macro-F1 = {best_f1m:.4f}")

with open(THR_OUT, "w") as f:
    json.dump({"best_threshold": best_thr, "val_macro_f1_at_best": best_f1m,
               "n_neurons": len(model.neurons), "sim_threshold": SIM_THRESH,
               "metric": "macro_f1"}, f, indent=2)

# ─────────────────────────────────────────────
log("STEP 5: evaluation ...")
def evaluate(scores, y, thr, label):
    pred = (scores >= thr).astype(int)
    return {
        "label": label, "threshold": float(thr),
        "macro_f1":  float(f1_score(y, pred, average="macro")),
        "binary_f1": float(f1_score(y, pred, average="binary")),
        "accuracy":  float((pred == y).mean()),
        "roc_auc":   float(roc_auc_score(y, scores)),
        "pr_auc":    float(average_precision_score(y, scores)),
        "report":    classification_report(y, pred, digits=4),
        "cm":        confusion_matrix(y, pred).tolist(),
    }

val_thr  = evaluate(val_scores,  y_va_w, best_thr, f"val@{best_thr:.4f}")
test_thr = evaluate(test_scores, y_te_w, best_thr, f"test@{best_thr:.4f}")

# ─────────────────────────────────────────────
log("STEP 6: save artifacts ...")
pd.DataFrame({"y_true": y_va_w, "anomaly_score": val_scores,
              "pred": (val_scores >= best_thr).astype(int)}).to_csv(PRED_VAL, index=False)
pd.DataFrame({"y_true": y_te_w, "anomaly_score": test_scores,
              "pred": (test_scores >= best_thr).astype(int)}).to_csv(PRED_TEST, index=False)

# train "size" = neuron count (not standard params)
n_neurons = len(model.neurons)
meta = {
    "model": "OCC-eSNN v3 19-dim", "n_features": 19, "n_neurons": n_neurons,
    "sim_threshold": SIM_THRESH, "best_threshold": best_thr,
    "metrics": {
        "val_at_thr":  {k: val_thr[k]  for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
        "test_at_thr": {k: test_thr[k] for k in ["macro_f1","binary_f1","accuracy","roc_auc","pr_auc"]},
    },
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

lines = [
    "=" * 72, "EVALUATION REPORT — OCC-eSNN v3 19-dim SCADA",
    "=" * 72,
    f"Window: {WINDOW}  |  Neurons: {n_neurons}  |  Sim threshold: {SIM_THRESH}",
    f"Tuned threshold: {best_thr:.4f}", "",
]
for ev in [val_thr, test_thr]:
    lines.append("-" * 72)
    lines.append(f"[{ev['label']}]  thr={ev['threshold']:.4f}  "
                 f"Macro-F1={ev['macro_f1']:.4f}  Binary-F1={ev['binary_f1']:.4f}  "
                 f"Acc={ev['accuracy']:.4f}  ROC-AUC={ev['roc_auc']:.4f}  PR-AUC={ev['pr_auc']:.4f}")
    lines.append(ev["report"])
    lines.append(f"CM: {ev['cm']}")
    lines.append("")
with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(test_thr["cm"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Attack"]); ax.set_yticklabels(["Normal","Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center", fontsize=14,
                    color="red" if cm[i,j] < cm.max()/2 else "white")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)

    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for sc, yy, lbl in [(val_scores, y_va_w, "val"), (test_scores, y_te_w, "test")]:
        p, r, _ = precision_recall_curve(yy, sc); ax1.plot(r, p, label=f"{lbl} AP={average_precision_score(yy, sc):.3f}")
        fpr, tpr, _ = roc_curve(yy, sc); ax2.plot(fpr, tpr, label=f"{lbl} AUC={roc_auc_score(yy, sc):.3f}")
    ax1.legend(); ax1.grid(alpha=0.3); ax2.legend(); ax2.grid(alpha=0.3); ax2.plot([0,1],[0,1],"k--",alpha=0.4)
    fig.tight_layout(); fig.savefig(PRROC_PNG, dpi=120); plt.close(fig)
    log("   plots saved")
except Exception as e:
    log(f"   plots skipped: {e}")

log("=" * 60)
log(f"DONE OCC-eSNN v3 19-dim.")
log(f"   test@{best_thr:.4f}: Macro-F1={test_thr['macro_f1']:.4f}  PR-AUC={test_thr['pr_auc']:.4f}  Bin-F1={test_thr['binary_f1']:.4f}  Acc={test_thr['accuracy']:.4f}")
log(f"   n_neurons (≈params): {n_neurons}")
log(f"   train time: {time.time()-t0:.1f}s")
log("=" * 60)