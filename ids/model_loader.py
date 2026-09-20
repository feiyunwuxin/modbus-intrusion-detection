"""模型加载与推理封装。

支持的格式:
    - .joblib: sklearn 模型 (joblib + sklearn)
    - .pt:     PyTorch 模型 (torch)

提供统一接口:
    ModelWrapper.infer(features) -> (label, prob)
    load_model(path) -> ModelWrapper
    list_available_models(directory) -> list[str]
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ModelWrapper(Protocol):
    """统一推理接口。"""

    @property
    def input_features(self) -> int:
        """输入特征数（固定 17）。"""
        ...

    def infer(self, features) -> tuple[int, float]:
        """推理。返回 (predicted_label, probability_of_attack)。"""
        ...


class _SklearnWrapper:
    """sklearn joblib 模型包装。"""

    def __init__(self, model, n_features: int):
        self._model = model
        self._n = n_features

    @property
    def input_features(self) -> int:
        return self._n

    def infer(self, features) -> tuple[int, float]:
        import numpy as np
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(x)[0]
            # 二分类：取 attack 列（label=1）
            if len(proba) == 2:
                prob_attack = float(proba[1])
            else:
                prob_attack = float(proba[-1])
        else:
            pred = int(self._model.predict(x)[0])
            return pred, 1.0 if pred == 1 else 0.0
        label = 1 if prob_attack >= 0.5 else 0
        return label, prob_attack


class _TorchWrapper:
    """PyTorch 模型包装（仅支持 Linear 输入）。"""

    def __init__(self, model, n_features: int):
        self._model = model
        self._n = n_features

    @property
    def input_features(self) -> int:
        return self._n

    def infer(self, features) -> tuple[int, float]:
        import numpy as np
        import torch
        x = torch.as_tensor(np.asarray(features, dtype=np.float32)).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)
        if logits.dim() == 2 and logits.shape[-1] >= 2:
            probs = torch.softmax(logits, dim=-1)[0]
            prob_attack = float(probs[-1])
        else:
            # 单输出 sigmoid
            prob_attack = float(torch.sigmoid(logits).flatten()[0])
        label = 1 if prob_attack >= 0.5 else 0
        return label, prob_attack


def load_model(path: str) -> ModelWrapper:
    """按扩展名加载模型。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 输入特征数不是 17
        RuntimeError: 依赖未安装（torch/sklearn）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    ext = p.suffix.lower()
    if ext == ".joblib":
        return _load_sklearn(p)
    if ext == ".pt":
        return _load_torch(p)
    raise ValueError(f"不支持的模型格式: {ext}（仅 .pt / .joblib）")


def _load_sklearn(p: Path) -> ModelWrapper:
    try:
        import joblib
    except ImportError as e:
        raise RuntimeError(
            "joblib 未安装，无法加载 .joblib 模型。请运行: pip install joblib scikit-learn"
        ) from e
    model = joblib.load(p)
    n = getattr(model, "n_features_in_", None)
    if n != 17:
        raise ValueError(
            f"模型期望 {n} 个特征，但 IDS 仅支持 17 特征。请选择其他模型。"
        )
    return _SklearnWrapper(model, 17)


def _detect_torch_input_features(model) -> int | None:
    """从 PyTorch 模型探测第一个 Linear 层的 in_features。"""
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.in_features
    return None


def _load_torch(p: Path) -> ModelWrapper:
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "torch 未安装，无法加载 .pt 模型。请运行: pip install torch"
        ) from e
    model = torch.load(p, map_location="cpu", weights_only=False)
    n = _detect_torch_input_features(model)
    if n != 17:
        raise ValueError(
            f"模型输入特征数为 {n}，但 IDS 仅支持 17 特征。"
            "3D 窗口模型（如 TCN/LSTM）请选择其他模型。"
        )
    return _TorchWrapper(model, 17)


def list_available_models(directory: str) -> list[str]:
    """扫描目录下 model_*.pt 和 model_*.joblib，按名字排序返回完整路径。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    files: list[Path] = []
    for ext in (".pt", ".joblib"):
        files.extend(d.glob(f"model_*{ext}"))
    return sorted(str(f) for f in files)