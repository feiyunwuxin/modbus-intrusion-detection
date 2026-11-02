#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keras TCN+SE 23-dim ch=32 — model definition for TFLite-Micro H743 deployment.

This is the canonical layer-by-layer definition that exactly mirrors:
  - C-side  ch32_inference_full_int8_qat.c (QAT INT8 path)
  - PC-side verify_qat_pc.py  (QAT fake-quant forward pass)

Architecture (BN folded inside Conv kernel at export time):
    Input x: (B, T=16, C=23)
      |
    TCN Block 0 (d=1):
      Conv2D(23→32, kernel=(3,1), dilation=(1,1), padding='same')
      BN-folded: y = (W*x + b) * gamma + beta
      ReLU → Conv2D(32→32, kernel=(3,1), dilation=(1,1), padding='same') → ReLU
      SE(GAP → FC(32→4) → ReLU → FC(4→32) → Sigmoid) → channel-wise mul
      + Residual: Conv2D(23→32, kernel=(1,1))
      ReLU
    TCN Block 1 (d=2): same but identity residual
    TCN Block 2 (d=4): same but identity residual
      |
    GAP (mean over time)
      |
    FC1(32→32) + ReLU + Dropout(0.3)
      |
    FC2(32→1) → logit (sigmoid at inference after temperature scaling)

Layer-by-layer alignment with verify_qat_pc.py:
  - Conv: Conv2D kernel (K,1), dilation_rate (1,d), padding='same' (manual)
  - SE:   GAP → FC1 → ReLU → FC2 → Sigmoid → mul
  - Residual: Conv2D(1,1) for first block (23→32), Identity for blocks 1/2
  - Classifier: GAP → FC1 → ReLU → FC2 → Sigmoid

Quantization expectation (QAT via TFMOT):
  - Per-channel symmetric INT8 weights
  - Per-tensor symmetric INT8 activations (zp=0)
  - INT32 bias (BN-folded)
  - Requant: M, N per-channel (CMSIS-NN style)
  - Sigmoid: 256-entry INT8 LUT in TFLite via ReferenceOp + custom LUT
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# === Hyperparameters matching PC trainer =====================================
N_FEATURES = 23      # input feature dim (KEEP_23 from 27-dim ARFF)
WINDOW = 16         # time steps per window
CHANNELS = 32       # TCN channel count
N_BLOCKS = 3        # TCN blocks
KERNEL_SIZE = 3     # conv kernel size
DILATIONS = [1, 2, 4]
SE_REDUCTION = 8
DROPOUT = 0.1       # block-level dropout
FC_DROPOUT = 0.3    # classifier dropout


# ===========================================================================
# Helper: BN-folded Conv2D
# ===========================================================================
# PyTorch Conv1d (BN folded at deploy) becomes:
#   y = (W_conv * x) * scale + bias
# We implement this with a custom layer that takes precomputed (W_fused, b_fused)
# and applies: tf.nn.conv2d(x, W_fused, ...) + b_fused
# For Keras training (FP32, with BN), we use Conv2D + manual BatchNorm layer.

class ConvBNSame2D(layers.Layer):
    """Conv2D with batch-normalization, BN will be folded at QAT export time.

    Uses `kernel=(K, 1)` and `dilation_rate=(1, d)` to behave like Conv1d.

    NOTE: TFLite Conv2D supports dilation natively. We avoid Conv1d here
    because TFLite Micro did NOT support Conv1d until r2.20+ and even then
    performance is poor. Conv2D(K, 1) is universally supported.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, name=None):
        super().__init__(name=name)
        # Compute SAME padding (causal-style) for 'same' on PyTorch
        # PyTorch Conv1d(SAME): pad = (K-1)*d // 2 on each side
        pad_total = (kernel_size - 1) * dilation
        self.pad_left = pad_total // 2
        self.pad_right = pad_total - self.pad_left

        # Conv2D kernel shape: (K, 1) → TFLite-friendly layout
        self.conv = layers.Conv2D(
            out_ch,
            kernel_size=(kernel_size, 1),
            dilation_rate=(1, dilation),
            padding='valid',           # we add padding manually for SAME symmetry
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
        )
        self.bn = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)

    def call(self, x, training=False):
        # x: (B, 1, T, C)  NHWC-like 2D layout
        # pad time dim
        if self.pad_left > 0 or self.pad_right > 0:
            # tf.pad with [[0,0],[0,0],[left,right],[0,0]] on (B,1,T,C)
            x = tf.pad(x, [[0,0], [0,0], [self.pad_left, self.pad_right], [0,0]])
        y = self.conv(x)               # (B, 1, T, out_ch)
        y = self.bn(y, training=training)
        return y


# ===========================================================================
# SE Block (channel attention)
# ===========================================================================
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block.

    forward:  GAP → FC1(C→C//8) → ReLU → FC2(C//8→C) → Sigmoid → scale

    Reduction = max(C // 8, 4), matching PyTorch SEBlock.
    """
    def __init__(self, channels, reduction=SE_REDUCTION, name=None):
        super().__init__(name=name)
        hidden = max(channels // reduction, 4)
        self.gap = layers.GlobalAveragePooling1D()  # (B, T, C) → (B, C)
        self.fc1 = layers.Dense(hidden, activation=None, use_bias=True)
        self.fc2 = layers.Dense(channels, activation=None, use_bias=True)

    def call(self, x):
        # x: (B, 1, T, C)  -- NHWC-style 2D
        b = tf.shape(x)[0]
        t = tf.shape(x)[2]
        c = x.shape[3]                # static (known)
        # Reshape to (B, T, C) for GAP
        x_squeeze = tf.reshape(x, [b, t, c])
        s = self.gap(x_squeeze)       # (B, C)
        s = tf.nn.relu(self.fc1(s))   # (B, hidden)
        s = tf.nn.sigmoid(self.fc2(s))  # (B, C)
        # Channel-wise multiply: x * s[:, None, None, :]
        s = tf.reshape(s, [b, 1, 1, c])
        return x * s


# ===========================================================================
# TCN Block with SE and residual
# ===========================================================================
class TCNBlockSE(layers.Layer):
    """One TCN block: 2× Conv2d (K,1) dilated + ReLU + SE + residual."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation,
                 dropout=DROPOUT, se_reduction=SE_REDUCTION, name=None):
        super().__init__(name=name)
        self.conv1 = ConvBNSame2D(in_ch, out_ch, kernel_size, dilation=dilation)
        self.relu1 = layers.ReLU()
        self.drop1 = layers.Dropout(dropout)
        self.conv2 = ConvBNSame2D(out_ch, out_ch, kernel_size, dilation=dilation)
        self.relu2 = layers.ReLU()
        self.drop2 = layers.Dropout(dropout)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        # Residual: 1×1 conv if in_ch != out_ch else Identity
        if in_ch != out_ch:
            self.residual = layers.Conv2D(out_ch, kernel_size=(1, 1),
                                          padding='same', use_bias=True)
        else:
            self.residual = layers.Lambda(lambda x: x)
        self.out_relu = layers.ReLU()

    def call(self, x, training=False):
        # x: (B, 1, T, in_ch)
        residual = self.residual(x)
        y = self.conv1(x, training=training)
        y = self.relu1(y)
        y = self.drop1(y, training=training)
        y = self.conv2(y, training=training)
        y = self.relu2(y)
        y = self.drop2(y, training=training)
        y = self.se(y)
        y = self.out_relu(y + residual)
        return y


# ===========================================================================
# TCN+SE Classifier
# ===========================================================================
class TCNClassifierSE(keras.Model):
    """Full TCN+SE 23-dim ch=32 binary IDS classifier.

    Input  : (B, T=16, C=23)   → reshape to (B, 1, 16, 23) for Conv2D
    Output : (B,) logit (raw, pre-sigmoid)
    """
    def __init__(self, in_ch=N_FEATURES, channels=CHANNELS,
                 n_blocks=N_BLOCKS, kernel_size=KERNEL_SIZE,
                 dilations=DILATIONS, dropout=DROPOUT,
                 fc_dropout=FC_DROPOUT, se_reduction=SE_REDUCTION,
                 name="tcn_se_classifier"):
        super().__init__(name=name)
        self.in_ch = in_ch
        # TCN stack: first block has 1×1 residual (23→32), others identity
        blocks = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0],
                             dropout=dropout, se_reduction=se_reduction,
                             name=f"tcn_block_0_d{dilations[0]}")]
        for i, d in enumerate(dilations[1:n_blocks], start=1):
            blocks.append(TCNBlockSE(channels, channels, kernel_size, d,
                                     dropout=dropout, se_reduction=se_reduction,
                                     name=f"tcn_block_{i}_d{d}"))
        self.tcn = keras.Sequential(blocks, name="tcn_stack")
        # Classifier head
        self.gap = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(channels, activation=None, use_bias=True)
        self.fc1_relu = layers.ReLU()
        self.fc_drop = layers.Dropout(fc_dropout)
        self.fc2 = layers.Dense(1, activation=None, use_bias=True)
        # Sigmoid for inference; during training, output raw logit for BCEWithLogitsLoss

    def call(self, x, training=False):
        """x: (B, T, C_in=23). Output: (B,) logit or probability."""
        # Reshape (B, T, C) → (B, 1, T, C) for Conv2D
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        c = x.shape[2]
        x = tf.reshape(x, [b, 1, t, c])

        # TCN forward
        y = self.tcn(x, training=training)        # (B, 1, T, C=32)

        # Reshape back to (B, T, C) for GAP
        t_out = tf.shape(y)[2]
        c_out = y.shape[3]
        y = tf.reshape(y, [b, t_out, c_out])

        # Classifier
        y = self.gap(y)                           # (B, C=32)
        y = self.fc1_relu(self.fc1(y))            # (B, 32)
        y = self.fc_drop(y, training=training)
        logit = self.fc2(y)                        # (B, 1)
        return tf.squeeze(logit, axis=-1)         # (B,)


# ===========================================================================
# Factory
# ===========================================================================
def build_model(seed=None, dtype=tf.float32):
    """Build a fresh TCN+SE 23-dim ch=32 model.

    Args:
        seed: Optional int for weight initialization.
    Returns:
        TCNClassifierSE instance.
    """
    if seed is not None:
        tf.random.set_seed(seed)
    model = TCNClassifierSE()
    # Build with known input shape to allocate weights eagerly
    dummy = tf.zeros([1, WINDOW, N_FEATURES], dtype=dtype)
    _ = model(dummy, training=False)
    return model


# ===========================================================================
# Input transpose helper (PC raw data → Keras input)
# ===========================================================================
def data_to_keras_input(X_window_level):
    """Convert (N, T, C) to Keras input. TFLite needs same layout.

    PC raw data is already (N, T, C) with KEEP_23 sliced and clipped ±10.
    Args:
        X_window_level: numpy array (N, T, C=23)
    Returns:
        Same array, Keras/TFLite-compatible.
    """
    import numpy as np
    assert X_window_level.ndim == 3
    assert X_window_level.shape[1] == WINDOW
    assert X_window_level.shape[2] == N_FEATURES
    return X_window_level.astype('float32')


# ===========================================================================
# Smoke test (runs without TF installed test set, just init)
# ===========================================================================
if __name__ == "__main__":
    print("=== Building TCN+SE 23-dim ch=32 Keras model ===")
    model = build_model(seed=42)
    n_params = model.count_params()
    print(f"  Total params: {n_params:,}")
    print(f"  Input  shape: (B, {WINDOW}, {N_FEATURES})")
    print(f"  Output shape: (B,)")

    # Quick forward
    import numpy as np
    x_test = np.random.randn(2, WINDOW, N_FEATURES).astype('float32')
    y_test = model(x_test, training=False).numpy()
    print(f"  Forward OK: y_test shape = {y_test.shape}, values ~ {y_test}")
    print("  (Sigmoid not applied; these are raw logits.)")

    print("\nDone. Use model.save('path.h5') or model.save('path') to export.")
