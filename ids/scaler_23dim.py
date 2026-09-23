"""SCADA RobustScaler 包装器，供 23-dim IDS 模型使用。

训练脚本（``_common_train.py`` / ``preprocess_v2_scada.py``）把
``scaler_binary_v2_scada.joblib`` 序列化到项目根目录，IDS 面板加载
23-dim 模型时按 19 行级特征顺序做 ``scaler.transform(raw_19)``。

设计要点：

* **Lazy load**：构造时不读盘，第一次 ``transform()`` 才加载；之后
  缓存 ``self._scaler``，避免每帧 I/O。
* **找不到 / 损坏** : 不阻塞推理 —— ``transform()`` 返回原值的 float32
  拷贝（带一次性 warning），这样 IDS 面板在缺 scaler 的开发机上也能
  把模型跑起来。代价是预测概率会偏向错误，metrics 数字不准。
* **支持 ndarray 任意前置维度**：(N, 19) / (W, 19) / (19,) 都接受，
  由 ``scaler.transform`` 自身的 reshape 协议决定。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

# 项目根下的 scaler 文件（preprocess_v2_scada.py:147 写出位置）。
DEFAULT_SCALER_PATH = "scaler_binary_v2_scada.joblib"


class Scaler23:
    """Lazy-loaded RobustScaler for 19-dim SCADA row-level features."""

    def __init__(self, scaler_path: str | Path = DEFAULT_SCALER_PATH):
        self._path = Path(scaler_path)
        self._scaler = None  # type: ignore[var-annotated]
        self._warned_missing = False

    def _ensure_loaded(self) -> None:
        """Load on first use; idempotent after that."""
        if self._scaler is not None:
            return
        try:
            import joblib  # noqa: WPS433 (local import keeps deps optional)
        except ImportError:
            self._scaler = None
            return
        if not self._path.exists():
            self._scaler = None
            return
        try:
            self._scaler = joblib.load(self._path)
        except Exception as exc:  # corrupt file, version mismatch, etc.
            warnings.warn(
                f"scaler_binary_v2_scada.joblib failed to load ({exc!r}); "
                f"23-dim predictions will be on raw features (inaccurate).",
                stacklevel=2,
            )
            self._scaler = None

    def transform(self, raw_19: np.ndarray) -> np.ndarray:
        """Apply the loaded RobustScaler; return same shape as input.

        If no scaler is available, return ``raw_19`` as float32 (a copy if
        needed) and warn once. The 23-dim model still runs — predictions
        may be wrong, but the eval pipeline keeps working so the panel
        does not silently drop these models.
        """
        self._ensure_loaded()
        if self._scaler is None:
            if not self._warned_missing:
                warnings.warn(
                    "scaler_binary_v2_scada.joblib not loaded; "
                    "23-dim features are raw. Predictions inaccurate.",
                    stacklevel=2,
                )
                self._warned_missing = True
            return np.asarray(raw_19, dtype=np.float32)
        # sklearn's RobustScaler.transform handles (N, 19) etc. natively.
        out = self._scaler.transform(np.asarray(raw_19, dtype=np.float64))
        return np.asarray(out, dtype=np.float32)

    @property
    def is_loaded(self) -> bool:
        """True if the scaler file was loaded successfully."""
        return self._scaler is not None