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


class _TorchStateDictWrapper:
    """通过重建 nn.Sequential 从 state_dict 推理。

    SCADA 训练项目多数保存为 `{"state_dict": {...}, ...}` 嵌套格式。
    对于纯 Linear 或 Linear+BatchNorm+ReLU Sequential 模式可自动重建。
    含 LSTM/GRU/Conv1d 等 3D 层则拒绝加载（需窗口输入）。
    """

    _FORBIDDEN_KEYS = ("lstm", "gru", "conv")

    def __init__(self, state_dict: dict, n_features: int):
        self._n = n_features
        self._model = self._reconstruct(state_dict, n_features)

    @property
    def input_features(self) -> int:
        return self._n

    @classmethod
    def _is_3d_compatible(cls, sd: dict) -> bool:
        """含 LSTM/GRU/Conv1d 关键字的 state_dict 需 3D 窗口输入，无法单帧推理。"""
        for key in sd:
            k = key.lower()
            if any(tok in k for tok in cls._FORBIDDEN_KEYS):
                return True
            # 3D 权重张量 (Conv1d 权重 shape [out, in, kernel])
            if not (key.endswith(".weight") or key.endswith(".bias")):
                continue
            # 仅通过 key 不足以判断，需要看 tensor
        return False

    @classmethod
    def _is_3d_compatible_from_tensors(cls, sd: dict) -> bool:
        """含 LSTM/GRU/Conv1d 关键字或 3D weight tensor 的 state_dict 需 3D 输入。"""
        for key, val in sd.items():
            k = key.lower()
            if any(tok in k for tok in cls._FORBIDDEN_KEYS):
                return True
            if hasattr(val, "dim") and val.dim() == 3:
                return True
        return False

    @classmethod
    def _parse_layer_index(cls, key: str) -> tuple[str, int] | None:
        """解析 state_dict key 的层级前缀。

        支持 'net.{i}.weight' 和 '{i}.weight' 两种命名。
        返回 (prefix, index)；无法解析返回 None。
        """
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "net":
            try:
                return ("net.", int(parts[1]))
            except ValueError:
                return None
        if len(parts) >= 2 and parts[0].isdigit():
            return ("", int(parts[0]))
        return None

    @classmethod
    def _reconstruct(cls, sd: dict, n_features: int):
        """重建 nn.Sequential。要求：仅含 Linear（+可选 BatchNorm1d + ReLU）。"""
        import torch.nn as nn

        # 1. 探测前缀（'net.' 或空）和层索引
        prefix = None
        layer_indices: set[int] = set()
        for key in sd:
            parsed = cls._parse_layer_index(key)
            if parsed is None:
                continue
            p, idx = parsed
            if prefix is None:
                prefix = p
            elif prefix != p:
                raise ValueError("state_dict 中层前缀不一致（混用 'net.' 与无前缀）")
            layer_indices.add(idx)
        if prefix is None or not layer_indices:
            raise ValueError(
                "state_dict 无法解析层结构（key 命名需为 'net.{i}.weight' 或 '{i}.weight'）"
            )

        # 2. 区分 Linear 层（2D weight）和 BN 层（有 running_mean）
        linear_indices: list[int] = []
        bn_indices: set[int] = set()
        for i in sorted(layer_indices):
            wkey = f"{prefix}{i}.weight"
            rmkey = f"{prefix}{i}.running_mean"
            if rmkey in sd:
                bn_indices.add(i)
            elif wkey in sd and hasattr(sd[wkey], "dim") and sd[wkey].dim() == 2:
                linear_indices.append(i)
            # 其他 tensor（num_batches_tracked 等）忽略

        if not linear_indices:
            raise ValueError("state_dict 中未找到任何 Linear 层")

        # 3. 按顺序构建 Sequential
        layers: list[nn.Module] = []
        last_linear_idx = linear_indices[-1]

        for i in linear_indices:
            wkey = f"{prefix}{i}.weight"
            bkey = f"{prefix}{i}.bias"
            w = sd[wkey]
            out_f, in_f = w.shape
            linear = nn.Linear(in_f, out_f, bias=bkey in sd)
            linear.weight.data = w.clone()
            if bkey in sd:
                linear.bias.data = sd[bkey].clone()
            layers.append(linear)

            # 紧随其后的 BatchNorm（如果存在）
            if i + 1 in bn_indices:
                bn_wkey = f"{prefix}{i + 1}.weight"
                bn_bkey = f"{prefix}{i + 1}.bias"
                bn_rmkey = f"{prefix}{i + 1}.running_mean"
                bn_rvkey = f"{prefix}{i + 1}.running_var"
                bn = nn.BatchNorm1d(out_f)
                bn.weight.data = sd[bn_wkey].clone()
                bn.bias.data = sd[bn_bkey].clone()
                bn.running_mean.copy_(sd[bn_rmkey])
                bn.running_var.copy_(sd[bn_rvkey])
                layers.append(bn)

            # 非最后 Linear 后接 ReLU
            if i != last_linear_idx:
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        model.eval()
        return model

    def infer(self, features) -> tuple[int, float]:
        import numpy as np
        import torch
        x = torch.as_tensor(np.asarray(features, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
        if logits.dim() == 2 and logits.shape[-1] >= 2:
            probs = torch.softmax(logits, dim=-1)[0]
            prob_attack = float(probs[-1])
        else:
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
    """从 PyTorch 模型或 state_dict 探测第一个 Linear 层的 in_features。

    支持:
        - nn.Module: 遍历 modules() 找 Linear
        - 嵌套 dict {"state_dict": {...}, ...}: 提取内层 state_dict
        - state_dict (dict of tensors): 找第一个非 LSTM/GRU/Conv 的 2D weight
    """
    import torch.nn as nn

    # nn.Module
    if isinstance(model, nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return None

    # 嵌套训练产物格式 {"state_dict": {...}, "best_threshold": ..., ...}
    if isinstance(model, dict) and "state_dict" in model and isinstance(model["state_dict"], dict):
        return _detect_torch_input_features(model["state_dict"])

    # state_dict 本身（dict of tensors）
    if isinstance(model, dict):
        for key, val in model.items():
            if not (hasattr(val, "dim") and val.dim() == 2):
                continue
            k = key.lower()
            if any(tok in k for tok in ("lstm", "gru", "conv")):
                continue
            # 跳过 BatchNorm 的 weight（1D）
            return int(val.shape[1])
    return None


def _load_torch(p: Path) -> ModelWrapper:
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "torch 未安装，无法加载 .pt 模型。请运行: pip install torch"
        ) from e
    obj = torch.load(p, map_location="cpu", weights_only=False)

    # 1. nn.Module（完整 pickled 模型）
    import torch.nn as nn
    if isinstance(obj, nn.Module):
        n = _detect_torch_input_features(obj)
        if n != 17:
            raise ValueError(
                f"模型期望 {n} 个特征，但 IDS 仅支持 17 特征。"
                "请选择其他模型。"
            )
        return _TorchWrapper(obj, 17)

    # 2. dict 格式：可能是嵌套训练产物或 state_dict
    if not isinstance(obj, dict):
        raise ValueError(f"不支持的 .pt 文件类型: {type(obj).__name__}")

    # 提取内层 state_dict
    sd = obj.get("state_dict", obj) if "state_dict" in obj else obj
    if not isinstance(sd, dict):
        raise ValueError("state_dict 格式错误")

    # 3. 检测 3D 窗口层 → 友好拒绝
    if _TorchStateDictWrapper._is_3d_compatible_from_tensors(sd):
        raise ValueError(
            "模型含 LSTM/GRU/Conv1d 层，需要 3D 窗口输入 (batch, seq, features)，"
            "但 IDS 仅支持 17 维单帧推理。请选择其他模型。"
        )

    # 4. 纯 Linear/MLP：从 state_dict 重建
    n = _detect_torch_input_features(sd)
    if n != 17:
        raise ValueError(
            f"模型期望 {n} 个特征，但 IDS 仅支持 17 特征。"
            "请选择其他模型。"
        )
    try:
        return _TorchStateDictWrapper(sd, 17)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"无法从 state_dict 重建模型: {e}") from e


def list_available_models(directory: str) -> list[str]:
    """扫描目录下 model_*.pt 和 model_*.joblib，按名字排序返回完整路径。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    files: list[Path] = []
    for ext in (".pt", ".joblib"):
        files.extend(d.glob(f"model_*{ext}"))
    return sorted(str(f) for f in files)