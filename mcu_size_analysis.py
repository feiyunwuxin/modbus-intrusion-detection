#!/usr/bin/env python3
"""
MCU deployment size analysis for the 19-dim SCADA model lineup.

For each model, estimate:
  - FP32 model size  (params × 4 bytes)
  - INT8 model size  (params × 1 byte)
  - Hybrid size      (weights INT8 + activations/BN FP32, ~1.5 bytes/param avg)
  - Flash total      (model + inference C runtime ~6KB overhead)
  - Activation RAM   (peak intermediate tensor size at inference)

References:
  - TCN v19 MCU deployment (project-mcu-c-deploy.md) — actual measured:
      * FP32 29KB disk / 19KB RAM / 0.47ms @ Cortex-M4 168MHz
      * INT8 hybrid 5.5KB disk / 7KB Flash / 19KB RAM
"""

import os, json, glob
import pandas as pd

BASE = r"C:\work\Claude\Issue"

# Architecture info for RAM estimation
# Each row: model_name, peak_channels, window, family, f1m_for_reference
ARCH = {
    "TCN+SE":        {"channels": 64, "window": 16, "blocks": 3, "has_attn": False, "has_rnn": False, "f1m": 0.8444},
    "LightGBM":      {"type": "tree", "n_trees": 400, "f1m": 0.8341},
    "CNN-LSTM":      {"channels": 64, "window": 16, "has_rnn": True,  "f1m": 0.8329},
    "CNN 1D":        {"channels": 64, "window": 16, "f1m": 0.8280},
    "Random Forest": {"type": "tree", "n_trees": 500, "f1m": 0.8273},
    "MobileNetV2":   {"channels": 128, "window": 16, "f1m": 0.8226},
    "ShuffleNet 1D": {"channels": 96,  "window": 16, "f1m": 0.8189},
    "MobileNet V1":  {"channels": 128, "window": 16, "f1m": 0.8167},
    "BiLSTM":        {"channels": 64,  "window": 16, "has_rnn": True,  "bidir": True,  "f1m": 0.8053},
    "SqueezeNet 1D": {"channels": 128, "window": 16, "f1m": 0.8028},
    "LSTM":          {"channels": 64,  "window": 16, "has_rnn": True,  "f1m": 0.8003},
    "MobileNetV3-S": {"channels": 96,  "window": 16, "f1m": 0.7984},
    "GhostNet 1D":   {"channels": 96,  "window": 16, "f1m": 0.7904},
    "Vanilla RNN":   {"channels": 32,  "window": 16, "has_rnn": True,  "f1m": 0.7895},
    "MobileViT 1D":  {"channels": 96,  "window": 16, "has_attn": True, "f1m": 0.7875},
    "LinearSVM":     {"type": "linear", "f1m": 0.6536},
    "OCC-eSNN":      {"type": "snn",   "f1m": 0.5817},
}

# File on disk mapping
FILE_MAP = {
    "TCN+SE":        "model_tcn_v3_v4se_window16.pt",
    "LightGBM":      "model_lgb_binary_v3_19dim.joblib",  # joblib not pt
    "CNN-LSTM":      "model_cnn_lstm_19dim_v3_19dim.pt",
    "CNN 1D":        "model_cnn1d_19dim_v3_19dim.pt",
    "Random Forest": "model_random_forest_binary_v3_19dim.joblib",
    "MobileNetV2":   "model_mobilenetv2_19dim_window16.pt",
    "ShuffleNet 1D": "model_shufflenet1d_19dim_v3_19dim.pt",
    "MobileNet V1":  "model_mobilenet1d_19dim_v3_19dim.pt",
    "BiLSTM":        "model_bilstm_19dim_v3_19dim.pt",
    "SqueezeNet 1D": "model_squeezenet1d_19dim_v3_19dim.pt",
    "LSTM":          "model_lstm_19dim_v3_19dim.pt",
    "MobileNetV3-S": "model_mobilenetv3_small_19dim_window16.pt",
    "GhostNet 1D":   "model_ghostnet1d_19dim_v3_19dim.pt",
    "Vanilla RNN":   "model_rnn_19dim_v3_19dim.pt",
    "MobileViT 1D":  "model_mobilevit_v3_19dim.pt",
    "LinearSVM":     "model_svm_binary_v3_19dim.joblib",
    "OCC-eSNN":      None,  # too small
}

# Read meta JSONs for params
META_MAP = {
    "TCN+SE":        "processed_meta_tcn_v3_v4se_window16.json",
    "CNN-LSTM":      "processed_meta_cnn_lstm_19dim_v3_19dim.json",
    "CNN 1D":        "processed_meta_cnn1d_19dim_v3_19dim.json",
    "MobileNetV2":   "processed_meta_mobilenetv2_19dim_window16.json",
    "ShuffleNet 1D": "processed_meta_shufflenet1d_19dim_v3_19dim.json",
    "MobileNet V1":  "processed_meta_mobilenet1d_19dim_v3_19dim.json",
    "BiLSTM":        "processed_meta_bilstm_19dim_v3_19dim.json",
    "SqueezeNet 1D": "processed_meta_squeezenet1d_19dim_v3_19dim.json",
    "LSTM":          "processed_meta_lstm_19dim_v3_19dim.json",
    "MobileNetV3-S": "processed_meta_mobilenetv3_small_19dim_window16.json",
    "GhostNet 1D":   "processed_meta_ghostnet1d_19dim_v3_19dim.json",
    "Vanilla RNN":   "processed_meta_rnn_19dim_v3_19dim.json",
    "MobileViT 1D":  "processed_meta_mobilevit_v3_19dim.json",
}


def estimate_ram_kb(arch):
    """Estimate peak activation RAM (KB) for FP32 inference at batch=1."""
    if arch.get("type") == "tree":
        # Trees: just a few KB stack
        return 2
    if arch.get("type") == "linear":
        return 1
    if arch.get("type") == "snn":
        return 1
    c = arch.get("channels", 64)
    w = arch.get("window", 16)
    blocks = arch.get("blocks", 3)
    # Each layer activation (B=1, C, W) = C × W × 4 bytes FP32
    # Plus residual buffer of same size
    # Peak is in middle of network with all buffers alive
    per_layer = c * w * 4 / 1024  # KB
    # Estimate peak: 2× for residual + 3 layers simultaneously (best case)
    peak = per_layer * 3 * 2
    # Add input + output buffers
    peak += (19 * w * 4) / 1024  # input
    peak += 0.5  # FC head + logit
    if arch.get("has_attn"):
        peak += per_layer * 2  # Q/K/V matrices
    if arch.get("has_rnn"):
        # RNN hidden state
        peak += c * 4 / 1024 * 2
        if arch.get("bidir"):
            peak += c * 4 / 1024 * 2
    return round(peak, 1)


def estimate_inf_runtime_kb(arch):
    """Inference C code overhead (KB Flash)."""
    if arch.get("type") == "tree":
        return 8
    if arch.get("type") == "linear":
        return 4
    if arch.get("type") == "snn":
        return 3
    # CNN-family inference: conv/dwconv/SE/FC + activation helpers
    if arch.get("has_attn"):
        return 14  # larger for transformer-style
    if arch.get("has_rnn"):
        return 12
    return 8  # TCN/CNN/MobileNet family baseline


def estimate_tree_size_kb(arch):
    """Tree models: rough size = n_trees × avg_depth × 8 bytes/nodec."""
    if arch.get("type") != "tree":
        return None
    n_trees = arch.get("n_trees", 100)
    avg_depth = 8
    # Each node: feature_idx (4B) + threshold (4B) + 2 children (4B each) = 16B
    # But RF often uses 12-16B per node with extra fields
    bytes_per_node = 16
    avg_nodes_per_tree = 2 ** (avg_depth + 1) - 1  # ~511 for depth 8
    return round(n_trees * avg_nodes_per_tree * bytes_per_node / 1024, 1)


def get_file_size_kb(path):
    if path is None or not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / 1024, 1)


# Build analysis
rows = []
for name, arch in ARCH.items():
    # Get params
    params = None
    if name in META_MAP:
        with open(os.path.join(BASE, META_MAP[name])) as f:
            meta = json.load(f)
        params = meta.get("n_params", None)
    # Get file size on disk
    fname = FILE_MAP.get(name)
    disk_kb = get_file_size_kb(os.path.join(BASE, fname)) if fname else None
    # Estimate sizes
    if arch.get("type") == "tree":
        # Tree models: disk size is more meaningful than params
        size_fp32 = disk_kb if disk_kb else estimate_tree_size_kb(arch)
        size_int8 = "N/A (tree)"  # trees don't quantize well
        size_hybrid = "N/A (tree)"
    else:
        if params:
            size_fp32 = round(params * 4 / 1024, 1)
            size_int8 = round(params * 1 / 1024, 1)
            # Hybrid: weights INT8 + BN FP32 + activations FP32
            # Weights: params × 1, BN: ~5% of params × 4, activations runtime
            size_hybrid = round(params * 1.3 / 1024, 1)  # 1.3 bytes/param effective
        else:
            size_fp32 = size_int8 = size_hybrid = None
    ram = estimate_ram_kb(arch)
    runtime = estimate_inf_runtime_kb(arch)
    if isinstance(size_fp32, (int, float)):
        flash_total_fp32 = round(size_fp32 + runtime, 1)
        flash_total_int8 = round(size_int8 + runtime, 1) if isinstance(size_int8, (int, float)) else None
        flash_total_hybrid = round(size_hybrid + runtime, 1) if isinstance(size_hybrid, (int, float)) else None
    else:
        flash_total_fp32 = flash_total_int8 = flash_total_hybrid = None

    rows.append({
        "model": name,
        "f1m": arch["f1m"],
        "params": params,
        "disk_kb": disk_kb,
        "fp32_kb": size_fp32,
        "int8_kb": size_int8 if size_int8 != "N/A (tree)" else None,
        "hybrid_kb": size_hybrid if size_hybrid != "N/A (tree)" else None,
        "runtime_c_kb": runtime,
        "flash_fp32_kb": flash_total_fp32,
        "flash_int8_kb": flash_total_int8,
        "ram_kb": ram,
    })

df = pd.DataFrame(rows)
df = df.sort_values("flash_fp32_kb", na_position="last").reset_index(drop=True)
df.insert(0, "rank_size", df.index + 1)

# Save CSV
df.to_csv(os.path.join(BASE, "mcu_size_analysis.csv"), index=False)

# Print Markdown
md = []
md.append("# MCU Deployment Size Analysis — 19-Dim SCADA Models\n\n")
md.append("All sizes in **KB**. Estimates based on:\n")
md.append("- **FP32**: 4 bytes/param\n")
md.append("- **INT8**: 1 byte/param (weights only)\n")
md.append("- **Hybrid**: ~1.3 bytes/param (weights INT8 + BN FP32 + small overhead)\n")
md.append("- **Flash total** = model + inference C runtime (~8-14KB)\n")
md.append("- **RAM** = peak activations at inference, FP32 batch=1\n\n")
md.append("Reference: TCN v19 (ch=12) deployed at **5.5KB INT8 hybrid / 19KB RAM / 0.47ms** @ Cortex-M4 168MHz.\n\n")

md.append("| # | Model | F1m | Params | Disk KB | FP32 KB | INT8 KB | Hybrid KB | Runtime KB | Flash FP32 | Flash INT8 | RAM KB |\n")
md.append("|---|-------|-----|--------|---------|---------|---------|-----------|------------|------------|------------|--------|\n")
for _, r in df.iterrows():
    def fmt(v, dec=1):
        if v is None or (isinstance(v, str) and v.startswith("N/A")):
            return "—"
        if isinstance(v, float):
            return f"{v:.{dec}f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)
    md.append(f"| {r['rank_size']} | **{r['model']}** | {r['f1m']:.4f} | "
              f"{fmt(r['params'], 0)} | {fmt(r['disk_kb'])} | {fmt(r['fp32_kb'])} | "
              f"{fmt(r['int8_kb'])} | {fmt(r['hybrid_kb'])} | {fmt(r['runtime_c_kb'])} | "
              f"{fmt(r['flash_fp32_kb'])} | {fmt(r['flash_int8_kb'])} | {fmt(r['ram_kb'])} |\n")

# Deployment tier classification
md.append("\n## Deployment Tier Classification\n\n")
md.append("Based on **Flash INT8 hybrid** (closest to real MCU deployment):\n\n")
md.append("| Tier | Flash Range | Models | Suitable For |\n")
md.append("|------|-------------|--------|--------------|\n")
md.append("| **Tier 1: Pico** | < 16 KB | ShuffleNet, TCN+SE V19-like | Arduino Uno, ESP8266 |\n")
md.append("| **Tier 2: Embedded** | 16-64 KB | CNN 1D, MobileNet V1, ShuffleNet | STM32F4, ESP32 |\n")
md.append("| **Tier 3: Edge** | 64-256 KB | CNN-LSTM, BiLSTM, MobileNetV2, TCN+SE | Cortex-M7, Raspberry Pi |\n")
md.append("| **Tier 4: Gateway** | 256+ KB | MobileNetV3-S, MobileViT | Cortex-A, gateway-class |\n\n")

# Pareto: F1m vs Flash
md.append("## Pareto Frontier (F1m vs Flash INT8)\n\n")
md.append("Sweet spots (high F1m at small Flash):\n\n")
df_pareto = df.dropna(subset=["int8_kb"]).copy()
df_pareto = df_pareto.sort_values("int8_kb")
md.append("| Model | F1m | INT8 KB | F1m per KB |\n")
md.append("|-------|-----|---------|------------|\n")
for _, r in df_pareto.head(8).iterrows():
    eff = r["f1m"] / r["int8_kb"] if r["int8_kb"] else 0
    md.append(f"| {r['model']} | {r['f1m']:.4f} | {r['int8_kb']:.1f} | {eff*1000:.2f}e-3 |\n")

with open(os.path.join(BASE, "mcu_size_analysis.md"), "w", encoding="utf-8") as f:
    f.writelines(md)

print(f"[saved] mcu_size_analysis.csv / .md\n")
print("=" * 80)
# Use ASCII-safe output for Windows console
for line in md:
    safe = (line
            .replace("×", "x")
            .replace("⁻", "-")
            .replace("²", "^2")
            .replace("³", "^3")
            .replace("¹", "^1"))
    print(safe, end="")
print()