"""模型加载与推理封装。

支持的格式:
    - .joblib: sklearn 模型 (joblib + sklearn)
    - .pt:     PyTorch 模型 (torch)

提供统一接口:
    ModelWrapper.infer(features) -> (label, prob) | None
    load_model(path, *, window_size=1) -> ModelWrapper
    list_available_models(directory) -> list[str]

3D 窗口模型（LSTM/GRU/Conv1d）需指定 window_size；infer 在 warm-up 期返回 None。
"""
from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Protocol


class ModelWrapper(Protocol):
    """统一推理接口。"""

    @property
    def input_features(self) -> int:
        """输入特征数（固定 17）。"""
        ...

    @property
    def window_size(self) -> int:
        """窗口大小（1 表示单帧；>1 表示 3D 窗口模型）。"""
        ...

    @property
    def warmup_remaining(self) -> int:
        """还需多少帧才能产生第一个预测（0 表示 ready）。"""
        ...

    def infer(self, features) -> tuple[int, float] | None:
        """推理。返回 (label, prob)；warm-up 期返回 None。"""
        ...


class _SklearnWrapper:
    """sklearn joblib 模型包装。"""

    def __init__(self, model, n_features: int):
        self._model = model
        self._n = n_features

    @property
    def input_features(self) -> int:
        return self._n

    @property
    def window_size(self) -> int:
        return 1

    @property
    def warmup_remaining(self) -> int:
        return 0

    def infer(self, features) -> tuple[int, float] | None:
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

    @property
    def window_size(self) -> int:
        return 1

    @property
    def warmup_remaining(self) -> int:
        return 0

    def infer(self, features) -> tuple[int, float] | None:
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

    @property
    def window_size(self) -> int:
        return 1

    @property
    def warmup_remaining(self) -> int:
        return 0

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

    def infer(self, features) -> tuple[int, float] | None:
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


# ----------------------------------------------------------------------
# 3D 窗口模型（LSTM / GRU / Conv1d）-- 滑动窗口推理
# ----------------------------------------------------------------------

_LSTM_KEY_RE = re.compile(r"^(lstm|gru)\.(weight|bias)_(ih|hh)_l(\d+)(_reverse)?$")


def _parse_lstm_arch(sd: dict) -> dict | None:
    """从 LSTM/GRU state_dict 推断架构。

    返回 {"module": "lstm"|"gru", "input_size", "hidden_size",
          "num_layers", "bidirectional"} 或 None。
    """
    module = None
    max_layer = -1
    bidirectional = False
    input_size = None
    hidden_size = None

    for key in sd:
        m = _LSTM_KEY_RE.match(key)
        if not m:
            continue
        mod_name = m.group(1)
        is_reverse = m.group(5) is not None
        layer_num = int(m.group(4))
        param_type = m.group(2)  # weight or bias
        gate_type = m.group(3)   # ih or hh

        if module is None:
            module = mod_name
        elif module != mod_name:
            return None  # 混用 lstm/gru，不支持

        max_layer = max(max_layer, layer_num)
        if is_reverse:
            bidirectional = True

        # 仅取 layer 0 的 weight_ih 推断 input_size / hidden_size
        if layer_num == 0 and param_type == "weight" and gate_type == "ih":
            w = sd[key]
            # LSTM/GRU weight_ih shape: (gate_size, input_size)
            # gate_size = 4*hidden for LSTM, 3*hidden for GRU
            gate_total = w.shape[0]
            input_size = int(w.shape[1])
            # 区分 LSTM/GRU：gate_total / 4 vs / 3 应整除
            if gate_total % 4 == 0:
                hidden_size = gate_total // 4
                module = "lstm"
            elif gate_total % 3 == 0:
                hidden_size = gate_total // 3
                module = "gru"

    if module is None or input_size is None or hidden_size is None:
        return None

    return {
        "module": module,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": max_layer + 1,
        "bidirectional": bidirectional,
    }


def _reconstruct_lstm(sd: dict, arch: dict):
    """从 state_dict 构建 nn.LSTM / nn.GRU 并加载权重。

    Real SCADA checkpoints store RNN weights under ``lstm.`` / ``gru.``
    prefixes (e.g. ``lstm.weight_ih_l0``); strip that prefix before
    delegating to ``rnn.load_state_dict``. Classifier keys (e.g.
    ``net.0.weight``) are untouched here — they are handled by
    ``_reconstruct_classifier_from_state_dict`` in the caller.
    """
    import torch.nn as nn

    if arch["module"] == "lstm":
        rnn = nn.LSTM(
            input_size=arch["input_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            batch_first=True,
            bidirectional=arch["bidirectional"],
        )
    else:
        rnn = nn.GRU(
            input_size=arch["input_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            batch_first=True,
            bidirectional=arch["bidirectional"],
        )

    prefix = arch["module"] + "."
    stripped = {
        k.removeprefix(prefix): v
        for k, v in sd.items()
        if k.startswith(prefix)
    }
    missing, unexpected = rnn.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"RNN state_dict 加载不完整: missing={missing[:3]}..., "
            f"unexpected={unexpected[:3]}..."
        )
    rnn.eval()
    return rnn


def _reconstruct_classifier_from_state_dict(
    sd: dict, prefix: str, indices: list[int]
):
    """从 state_dict 提取 Linear 子序列构建 classifier Sequential。

    不含 BN / ReLU 间隔——仅按 Linear 出现顺序连接，最后一层不加 ReLU。
    """
    import torch.nn as nn

    layers: list[nn.Module] = []
    sorted_idx = sorted(indices)
    last = sorted_idx[-1] if sorted_idx else None

    for i in sorted_idx:
        wkey = f"{prefix}{i}.weight"
        bkey = f"{prefix}{i}.bias"
        if wkey not in sd:
            raise ValueError(f"classifier 缺少层 {i} 的权重")
        w = sd[wkey]
        if not (hasattr(w, "dim") and w.dim() == 2):
            raise ValueError(f"classifier 层 {i} 权重维度非 2D")
        out_f, in_f = w.shape
        linear = nn.Linear(in_f, out_f, bias=bkey in sd)
        linear.weight.data = w.clone()
        if bkey in sd:
            linear.bias.data = sd[bkey].clone()
        layers.append(linear)
        if i != last:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class _Torch3DStateDictWrapper:
    """滑动窗口推理：LSTM / GRU / Conv1d state_dict 重建。

    维护最近 window_size 帧的 buffer；每次 infer 入队一帧，buffer 满后
    用最近 N 帧做一次推理（前 N-1 帧返回 None 表示 warm-up）。

    LSTM/GRU 约定: 取最后时间步 output → classifier。
    Conv1d 约定: stack Conv1d + BatchNorm + ReLU，AdaptiveAvgPool1d(1) → squeeze
                 → classifier。
    """

    def __init__(self, state_dict: dict, n_features: int, window_size: int):
        if window_size < 1:
            raise ValueError(f"window_size 必须 >= 1，实际 {window_size}")
        self._n = n_features
        self._window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._model = self._reconstruct(state_dict, n_features)
        self._model.eval()

    @property
    def input_features(self) -> int:
        return self._n

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def warmup_remaining(self) -> int:
        return max(0, self._window_size - len(self._buffer))

    def infer(self, features) -> tuple[int, float] | None:
        import numpy as np
        import torch
        self._buffer.append(np.asarray(features, dtype=np.float32))
        if len(self._buffer) < self._window_size:
            return None
        window = np.stack(list(self._buffer), axis=0)  # (W, F)
        x = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, W, F)
        with torch.no_grad():
            logits = self._model(x)
        if logits.dim() == 2 and logits.shape[-1] >= 2:
            probs = torch.softmax(logits, dim=-1)[0]
            prob_attack = float(probs[-1])
        else:
            prob_attack = float(torch.sigmoid(logits).flatten()[0])
        label = 1 if prob_attack >= 0.5 else 0
        return label, prob_attack

    @classmethod
    def _reconstruct(cls, sd: dict, n_features: int):
        """根据 state_dict 内容自动选择 LSTM/GRU/Conv1d 重建路径。"""
        import torch.nn as nn

        # 1. 试 LSTM/GRU
        arch = _parse_lstm_arch(sd)
        if arch is not None:
            if arch["input_size"] != n_features:
                raise ValueError(
                    f"RNN 期望 {arch['input_size']} 特征，但 IDS 提供 {n_features}。"
                    "请确认 window 中每帧的特征数与训练时一致。"
                )
            rnn = _reconstruct_lstm(sd, arch)

            # 找 classifier (非 lstm./gru. 前缀的 Linear 层)
            classifier_prefix, classifier_indices = cls._find_linear_classifier(sd)
            if classifier_indices and classifier_prefix is not None:
                classifier = _reconstruct_classifier_from_state_dict(
                    sd, classifier_prefix, classifier_indices
                )
            else:
                # 无 classifier: RNN 直接输出（罕见）
                classifier = nn.Identity()

            # 验证 classifier 的 in_features 与 RNN 输出匹配
            rnn_out_dim = arch["hidden_size"] * (2 if arch["bidirectional"] else 1)
            if classifier_indices:
                first_linear = classifier[0]
                if hasattr(first_linear, "in_features") and first_linear.in_features != rnn_out_dim:
                    # 接受偏差：可能是用 final hidden state 而不是 last output
                    pass  # 不报错，让 forward 用 last timestep（最常见约定）

            class _RNN3DModel(nn.Module):
                def __init__(self, rnn, classifier):
                    super().__init__()
                    self.rnn = rnn
                    self.classifier = classifier

                def forward(self, x):
                    output, _ = self.rnn(x)
                    last = output[:, -1, :]  # (batch, hidden*directions)
                    return self.classifier(last)

            return _RNN3DModel(rnn, classifier)

        # 2. 试 Conv1d（3D weight tensor）
        has_conv1d = any(
            hasattr(v, "dim") and v.dim() == 3
            for v in sd.values()
        )
        if has_conv1d:
            return cls._reconstruct_conv1d(sd, n_features)

        raise ValueError(
            "state_dict 既无 LSTM/GRU 关键字也无 3D weight tensor，"
            "无法识别 3D 架构"
        )

    @classmethod
    def _find_linear_classifier(cls, sd: dict) -> tuple[str | None, list[int]]:
        """找非 RNN 前缀的 Linear 层（2D weight，不在 lstm/gru 命名空间下）。

        返回 (prefix, [indices])。
        """
        indices_by_prefix: dict[str, list[int]] = {}
        for key, val in sd.items():
            if not (hasattr(val, "dim") and val.dim() == 2):
                continue
            k_low = key.lower()
            if k_low.startswith(("lstm.", "gru.")):
                continue
            # 需要是 .weight 后缀
            if not key.endswith(".weight"):
                continue
            parts = key.split(".")
            if len(parts) < 3:
                continue
            prefix = parts[0] + "."
            if not parts[1].isdigit():
                continue
            i = int(parts[1])
            indices_by_prefix.setdefault(prefix, []).append(i)

        # 取数量最多的 prefix 作为 classifier
        if not indices_by_prefix:
            return None, []
        prefix = max(indices_by_prefix, key=lambda p: len(indices_by_prefix[p]))
        return prefix, indices_by_prefix[prefix]

    @classmethod
    def _reconstruct_conv1d(cls, sd: dict, n_features: int):
        """重建 Conv1d + BN + ReLU Sequential + AdaptiveAvgPool1d(1) + Linear。"""
        import torch.nn as nn

        # 收集所有层索引 + prefix
        prefix = None
        layer_indices: set[int] = set()
        for key in sd:
            parts = key.split(".")
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            if prefix is None:
                prefix = parts[0] + "."
            elif not key.startswith(prefix):
                continue
            layer_indices.add(int(parts[1]))
        if prefix is None:
            raise ValueError("Conv1d state_dict 无法解析层前缀")

        # 分类：Conv1d (3D weight), BN (有 running_mean), Linear (2D weight)
        conv_indices: list[int] = []
        bn_indices: set[int] = set()
        classifier_indices: list[int] = []
        for i in sorted(layer_indices):
            wkey = f"{prefix}{i}.weight"
            rmkey = f"{prefix}{i}.running_mean"
            if rmkey in sd:
                bn_indices.add(i)
            elif wkey in sd and hasattr(sd[wkey], "dim"):
                if sd[wkey].dim() == 3:
                    conv_indices.append(i)
                elif sd[wkey].dim() == 2:
                    classifier_indices.append(i)

        if not conv_indices:
            raise ValueError("未找到 Conv1d 层（应有 3D weight tensor）")
        if not classifier_indices:
            raise ValueError("未找到 classifier Linear 层")

        # 校验第一层 Conv1d 的 in_channels
        first_w = sd[f"{prefix}{conv_indices[0]}.weight"]
        if first_w.shape[1] != n_features:
            raise ValueError(
                f"Conv1d 期望 {first_w.shape[1]} 通道，但 IDS 提供 {n_features}。"
            )

        # 验证 classifier 索引在 Conv1d 之后（idx 大于最后 conv）
        last_conv_idx = max(conv_indices)
        valid_classifier = [i for i in classifier_indices if i > last_conv_idx]
        if not valid_classifier:
            raise ValueError("classifier Linear 索引位置异常（应在 Conv1d 之后）")

        # 构建 conv stack: Conv1d → (BN → ReLU) → ...
        conv_layers: list[nn.Module] = []
        for i in sorted(conv_indices):
            w = sd[f"{prefix}{i}.weight"]
            b = sd.get(f"{prefix}{i}.bias")
            out_c, in_c, kernel = w.shape
            conv = nn.Conv1d(in_c, out_c, kernel_size=kernel, bias=b is not None)
            conv.weight.data = w.clone()
            if b is not None:
                conv.bias.data = b.clone()
            conv_layers.append(conv)
            if i + 1 in bn_indices:
                bn = nn.BatchNorm1d(out_c)
                bn.weight.data = sd[f"{prefix}{i+1}.weight"].clone()
                bn.bias.data = sd[f"{prefix}{i+1}.bias"].clone()
                bn.running_mean.copy_(sd[f"{prefix}{i+1}.running_mean"])
                bn.running_var.copy_(sd[f"{prefix}{i+1}.running_var"])
                conv_layers.append(bn)
            conv_layers.append(nn.ReLU())

        conv_seq = nn.Sequential(*conv_layers)
        pool = nn.AdaptiveAvgPool1d(1)
        classifier = _reconstruct_classifier_from_state_dict(
            sd, prefix, valid_classifier
        )

        class _CNN3DModel(nn.Module):
            def __init__(self, convs, pool, classifier):
                super().__init__()
                self.convs = convs
                self.pool = pool
                self.classifier = classifier

            def forward(self, x):
                # x: (batch, seq, features) → Conv1d expects (batch, features, seq)
                x = x.transpose(1, 2)
                h = self.convs(x)
                h = self.pool(h).squeeze(-1)  # (batch, channels)
                return self.classifier(h)

        return _CNN3DModel(conv_seq, pool, classifier)


def load_model(path: str, *, window_size: int = 1) -> ModelWrapper:
    """按扩展名加载模型。

    Args:
        path: 模型文件路径（.joblib 或 .pt）
        window_size: 窗口大小。1=单帧推理（默认）；>1=3D 窗口推理
                     （仅 LSTM/GRU/Conv1d 有效；单帧模型忽略此参数）

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 输入特征数不是 17，或 window_size 非法，或模型不支持
        RuntimeError: 依赖未安装（torch/sklearn）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    ext = p.suffix.lower()
    if ext == ".joblib":
        return _load_sklearn(p)
    if ext == ".pt":
        return _load_torch(p, window_size=window_size)
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


def _load_torch(p: Path, *, window_size: int = 1) -> ModelWrapper:
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

    # 3. 检测 3D 窗口层 → 用 _Torch3DStateDictWrapper 重建
    is_3d = _TorchStateDictWrapper._is_3d_compatible_from_tensors(sd)
    if is_3d:
        if window_size < 2:
            raise ValueError(
                "模型为 3D 窗口模型（LSTM/GRU/Conv1d），"
                "加载时需指定 window_size >= 2（如 16、32）。"
                "请在 IDS 面板设置窗口大小后重试（建议从 16 开始）。"
            )
        try:
            return _Torch3DStateDictWrapper(sd, 17, window_size)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"无法从 state_dict 重建 3D 模型: {e}") from e

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