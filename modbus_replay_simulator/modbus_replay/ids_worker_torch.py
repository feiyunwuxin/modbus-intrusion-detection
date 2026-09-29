"""Torch model definition for the PC-side IDS.

This module holds the verbatim ``SEBlock`` / ``TCNBlockSE`` /
``TCNClassifierSE`` classes from
``scripts/pc_inference_765_vs_h293_h743.py:29-82`` (the reference
implementation used to train ``model_v4_se_23dim_b64_ch32_do01_window16_s42.pt``)
plus a small factory :func:`build_tcn_se_model`.

Splitting this off into its own module is intentional:

* The IDS worker's run-loop imports this lazily so the rest of the GUI
  can launch without ``[ids]`` installed.
* Tests can ``monkeypatch`` ``modbus_replay.ids_worker.build_tcn_se_model``
  (or ``modbus_replay.ids_worker_torch.build_tcn_se_model``) without
  ever importing ``torch`` themselves.

Architecture defaults (``in_ch=23, n_blocks=3, channels=32,
kernel_size=3, dilations=(1,2,4), dropout=0.1``) match the canonical
23-dim TCN+SE checkpoint used by both the H743 firmware (via the
CMSIS-NN graph) and the PC inference script.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-excitation block (channel-wise attention).

    Verbatim from ``pc_inference_765_vs_h743.py`` lines 29-41.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        s = self.gap(x).squeeze(-1)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class TCNBlockSE(nn.Module):
    """Two-conv residual TCN block with a trailing SE attention layer.

    Verbatim from ``pc_inference_765_vs_h743.py`` lines 44-63.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        se_reduction: int = 8,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=pad, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, padding=pad, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.se(x)
        return torch.relu(x + residual)


class TCNClassifierSE(nn.Module):
    """23-dim TCN+SE classifier (matches the canonical .pt checkpoint).

    Verbatim from ``pc_inference_765_vs_h743.py`` lines 66-82. The
    head is GAP → Linear(32, 32) → ReLU → dropout(0.3) → Linear(32, 1).
    The final ``squeeze(-1)`` collapses the trailing 1-dim logit so
    the model output is shape ``(batch,)`` — caller applies sigmoid.
    """

    def __init__(
        self,
        in_ch: int = 23,
        n_blocks: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = [TCNBlockSE(in_ch, channels, kernel_size, dilations[0], dropout)]
        for d in dilations[1:n_blocks]:
            layers.append(TCNBlockSE(channels, channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
        return self.fc2(x).squeeze(-1)


def build_tcn_se_model(
    in_ch: int = 23,
    n_blocks: int = 3,
    channels: int = 32,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = (1, 2, 4),
    dropout: float = 0.1,
    se_reduction: int = 8,
) -> TCNClassifierSE:
    """Build a fresh :class:`TCNClassifierSE` matching the canonical .pt.

    Tests monkeypatch this so they never load torch. Production code
    uses this to build the model and then loads the checkpoint state
    dict via ``model.load_state_dict(...)``.
    """
    return TCNClassifierSE(
        in_ch=in_ch,
        n_blocks=n_blocks,
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    )


__all__ = ["SEBlock", "TCNBlockSE", "TCNClassifierSE", "build_tcn_se_model"]