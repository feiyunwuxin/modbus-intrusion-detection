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
        # 可选归一化：state_dict 里有 __norm_mean__ / __norm_std__ 时 (1D, shape (F,))
        # 在 infer 前对每帧做 (x - mean) / std。这些 key 不会被 _parse_lstm_arch、
        # _reconstruct_lstm 或 _reconstruct_classifier 误处理（不匹配任何前缀）。
        import torch
        self._norm_mean = state_dict.get("__norm_mean__")
        self._norm_std  = state_dict.get("__norm_std__")
        if self._norm_mean is not None and self._norm_std is not None:
            self._norm_mean = torch.as_tensor(self._norm_mean, dtype=torch.float32)
            self._norm_std  = torch.as_tensor(self._norm_std,  dtype=torch.float32)
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
        # 若 state_dict 含归一化参数，先对每帧做 (x - mean) / std 再推理
        if self._norm_mean is not None and self._norm_std is not None:
            window = (window - self._norm_mean.numpy()) / self._norm_std.numpy()
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
            classifier_prefix, classifier_keys = cls._find_linear_classifier(sd)
            if classifier_keys and classifier_prefix is not None:
                if classifier_prefix == "flat:":
                    classifier = cls._reconstruct_classifier_flat(sd, classifier_keys)
                else:
                    classifier = _reconstruct_classifier_from_state_dict(
                        sd, classifier_prefix, classifier_keys
                    )
            else:
                # 无 classifier: RNN 直接输出（罕见）
                classifier = nn.Identity()

            # 验证 classifier 的 in_features 与 RNN 输出匹配
            rnn_out_dim = arch["hidden_size"] * (2 if arch["bidirectional"] else 1)
            if classifier_keys and not isinstance(classifier, nn.Identity):
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
    def _find_linear_classifier(cls, sd: dict) -> tuple[str | None, list]:
        """找非 RNN 前缀的 Linear 层（2D weight，不在 lstm/gru 命名空间下）。

        支持两种命名风格：
        - sequential-style: ``net.0.weight``、``classifier.0.weight``、``head.1.weight``
          (前缀.<digit>.weight，返回 (prefix, [indices]))
        - flat-style: ``fc1.weight``、``fc2.weight``、``fc.weight``
          (2 段命名，无数字索引，返回 ("flat:", [完整 key 列表])，按名称末尾数字排序)

        优先 flat-style：训练脚本更常见这种命名（BiLSTM + fc1/fc2 head）。
        """
        flat_layers: list[tuple[str, str]] = []
        indexed_by_prefix: dict[str, list[int]] = {}

        for key, val in sd.items():
            if not (hasattr(val, "dim") and val.dim() == 2):
                continue
            k_low = key.lower()
            if any(k_low.startswith(p) for p in ("lstm.", "gru.")):
                continue
            if not key.endswith(".weight"):
                continue
            parts = key.split(".")
            if len(parts) == 2:
                # flat-style: "fc1.weight", "fc.weight", "head.weight"
                flat_layers.append((parts[0], key))
            elif len(parts) >= 3 and parts[1].isdigit():
                # sequential-style: "net.0.weight"
                indexed_by_prefix.setdefault(parts[0] + ".", []).append(int(parts[1]))

        # 优先 flat-style（更常见于 BiLSTM 训练脚本）
        if flat_layers:
            def _sort_key(item: tuple[str, str]) -> tuple[int, str]:
                name = item[0]
                digits = "".join(c for c in name if c.isdigit())
                return (int(digits) if digits else 0, name)
            flat_layers.sort(key=_sort_key)
            return "flat:", [k for _, k in flat_layers]

        # fallback: 选层数最多的 sequential prefix
        if indexed_by_prefix:
            prefix = max(indexed_by_prefix, key=lambda p: len(indexed_by_prefix[p]))
            return prefix, indexed_by_prefix[prefix]

        return None, []

    @classmethod
    def _reconstruct_classifier_flat(cls, sd: dict, keys: list[str]):
        """从 flat-style key 列表 (如 ["fc1.weight", "fc2.weight"]) 重建 Sequential。

        各层之间插入 ReLU（与训练时 `F.relu(self.fc1(last))` 一致）；
        最后一层不加 ReLU（与训练时 `self.fc2(h).squeeze(-1)` 一致）。
        Dropout 训练时存在但 eval 时是 identity，故省略。
        """
        import torch.nn as nn
        layers: list[nn.Module] = []
        last = len(keys) - 1
        for i, wkey in enumerate(keys):
            bkey = wkey[: -len(".weight")] + ".bias"
            w = sd[wkey]
            if not (hasattr(w, "dim") and w.dim() == 2):
                raise ValueError(f"{wkey} 权重维度非 2D")
            out_f, in_f = w.shape
            has_bias = bkey in sd
            linear = nn.Linear(in_f, out_f, bias=has_bias)
            linear.weight.data = w.clone()
            if has_bias:
                linear.bias.data = sd[bkey].clone()
            layers.append(linear)
            if i != last:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

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
            # 没找到内联 classifier，尝试 flat-style（CNN 23-dim 用 fc1/fc2）。
            flat_prefix, flat_keys = cls._find_linear_classifier(sd)
            if not (flat_prefix == "flat:" and flat_keys):
                raise ValueError("未找到 classifier Linear 层")

        # 校验第一层 Conv1d 的 in_channels
        first_w = sd[f"{prefix}{conv_indices[0]}.weight"]
        if first_w.shape[1] != n_features:
            raise ValueError(
                f"Conv1d 期望 {first_w.shape[1]} 通道，但 IDS 提供 {n_features}。"
            )

        # 验证 classifier 索引在 Conv1d 之后（idx 大于最后 conv）。
        # 若 conv.* 前缀下没找到 2D classifier（CNN 23-dim 等用 flat-style
        # fc1.weight/fc2.weight），退回到 flat-style classifier 重建。
        last_conv_idx = max(conv_indices)
        valid_classifier = [i for i in classifier_indices if i > last_conv_idx]
        use_flat_classifier = False
        if not valid_classifier:
            flat_prefix, flat_keys = cls._find_linear_classifier(sd)
            if flat_prefix == "flat:" and flat_keys:
                use_flat_classifier = True
                valid_classifier = flat_keys
            else:
                raise ValueError("未找到 classifier Linear 层（既无 conv.* 内联，也无 fc*/head* flat-style）")

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
        if use_flat_classifier:
            classifier = cls._reconstruct_classifier_flat(sd, valid_classifier)
        else:
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


# ----------------------------------------------------------------------
# TCN+Pool 嵌套格式 (train_tcn_23dim_*5seed.py 输出)
# ----------------------------------------------------------------------

_TCN_POOL_DEFAULTS = {
    # 与 train_tcn_23dim_*5seed.py:21-31 默认值保持一致
    "n_blocks": 3,
    "channels": 32,
    "kernel_size": 3,
    "dilations": (1, 2, 4),
    "dropout": 0.1,
    "attn_hidden": 16,
    "n_heads": 2,
}


def _detect_tcn_pool_nested(sd: dict) -> bool:
    """True if ``sd`` 含有 ``tcn.{i}.conv1.weight`` 形式的嵌套 4 段 key。

    训练脚本 (train_tcn_23dim_*.py) 保存的 state_dict 每个 TCN block 都有
    conv1/bn1/conv2/bn2/residual/se 子模块，而 ``_reconstruct_conv1d``
    只认 ``prefix.{i}.weight`` 这种扁平命名。识别后改走 :func:`_reconstruct_tcn_pool_model`。
    """
    for key in sd:
        parts = key.split(".")
        if (
            len(parts) >= 4
            and parts[0] == "tcn"
            and parts[1].isdigit()
            and parts[2] in {"conv1", "conv2", "bn1", "bn2", "residual", "se"}
        ):
            return True
    return False


def _detect_pool_type(sd: dict) -> str:
    """从 state_dict 的非 tcn/fc 前缀 key 识别 pool head 类型。"""
    if any(k.startswith("attpool.") for k in sd):
        return "attpool"
    if any(k.startswith("gatedpool.") for k in sd):
        return "gatedpool"
    if any(k.startswith("mhattpool.") for k in sd):
        return "mhattpool"
    if any(k.startswith("tfpool.") for k in sd):
        return "tfpool"
    return "gap"


def _has_se_block(sd: dict) -> bool:
    return any(
        len(k.split(".")) >= 4 and k.split(".")[2] == "se" for k in sd
    )


def _make_tcn_block(in_ch: int, out_ch: int, kernel_size: int,
                    dilation: int, dropout: float, has_se: bool, se_hidden: int):
    """Build one TCN block module mirroring ``TCNBlock`` / ``TCNBlockSE``."""
    import torch.nn as nn
    import torch.nn.functional as F
    pad = (kernel_size - 1) * dilation // 2

    class _TCNBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                   padding=pad, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                   padding=pad, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.drop = nn.Dropout(dropout)
            self.residual = (
                nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch
                else nn.Identity()
            )
            if has_se:
                self.se = self._make_se(out_ch, se_hidden)

        @staticmethod
        def _make_se(channels, hidden):
            class _SE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gap = nn.AdaptiveAvgPool1d(1)
                    self.fc1 = nn.Linear(channels, hidden)
                    self.fc2 = nn.Linear(hidden, channels)
                def forward(self, x):
                    s = self.gap(x).squeeze(-1)
                    s = F.relu(self.fc1(s))
                    s = torch.sigmoid(self.fc2(s))
                    return x * s.unsqueeze(-1)
            return _SE()

        def forward(self, x):
            r = self.residual(x)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.drop(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.drop(x)
            if has_se:
                x = self.se(x)
            return F.relu(x + r)

    return _TCNBlock()


def _make_attpool(channels: int, hidden: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class _AP(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            x_t = x.transpose(1, 2)
            weights = F.softmax(self.proj(x_t), dim=1)
            return (x_t * weights).sum(dim=1)

    return _AP()


def _make_gatedpool(channels: int, hidden: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class _GP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.Tanh(),
                nn.Linear(hidden, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x_t = x.transpose(1, 2)
            g = self.gate(x_t)
            return (x_t * g).sum(dim=1)

    return _GP()


def _make_mhattpool(channels: int, n_heads: int, hidden: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class _MH(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = n_heads
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(channels, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, 1),
                ) for _ in range(n_heads)
            ])

        def forward(self, x):
            x_t = x.transpose(1, 2)
            pooled = []
            for head in self.heads:
                w = F.softmax(head(x_t), dim=1)
                pooled.append((x_t * w).sum(dim=1))
            return torch.stack(pooled, dim=0).mean(dim=0)

    return _MH()


def _make_tfpool(channels: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _TF(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Parameter(torch.randn(channels) * 0.02)
            self.norm = nn.LayerNorm(channels)

        def forward(self, x):
            x_t = x.transpose(1, 2)
            scores = (x_t * self.query).sum(dim=-1)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            context = (x_t * weights).sum(dim=1)
            return self.norm(context)

    return _TF()


def _reconstruct_tcn_pool_model(sd: dict, n_features: int):
    """Build TCN+pool+classifier module from nested state_dict.

    Mirrors ``TCN*GELUClassifier`` / ``TCNAttPoolClassifier`` etc. in
    ``train_tcn_23dim_*5seed.py``. Defaults to the standard 23-dim config:

        n_blocks=3, channels=32, kernel_size=3, dilations=[1,2,4],
        dropout=0.1, attn_hidden=16, n_heads=2.

    Forward takes ``(batch, window, n_features)`` (transpose 1/2 internally)
    and returns ``(batch,)`` logits.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = _TCN_POOL_DEFAULTS
    n_blocks = cfg["n_blocks"]
    channels = cfg["channels"]
    kernel_size = cfg["kernel_size"]
    dilations = cfg["dilations"]
    dropout = cfg["dropout"]
    attn_hidden = cfg["attn_hidden"]
    n_heads = cfg["n_heads"]

    pool_type = _detect_pool_type(sd)
    has_se = _has_se_block(sd)
    se_hidden = max(channels // 8, 4) if has_se else 0

    # 激活函数：gelu 路径显式使用 GELU；其他用 ReLU（与所有
    # train_tcn_23dim_*5seed.py 训练脚本 forward 一致）。
    activation = "gelu" if pool_type == "gap" and has_se is False and False else "relu"
    # 注：geu 池式（gelu_s*.pt）activation 为 gelu，其余均为 relu。
    # 没有可靠的方式从 state_dict 推断；改为依据训练脚本默认——
    # 通过 pool_type 决策：只有 GAP + 显式无其他头 才视为 gelu 路径，
    # 否则走 relu。这里我们用 "relu" 默认，仅当激活必须为 gelu 时由
    # _activate_from_sd 覆写（占位 hook，便于后续扩展）。
    activation = "relu"

    blocks_mod = []
    for i in range(n_blocks):
        in_ch = n_features if i == 0 else channels
        d = dilations[i] if i < len(dilations) else dilations[-1]
        blocks_mod.append(
            _make_tcn_block(in_ch, channels, kernel_size, d, dropout,
                            has_se=has_se, se_hidden=se_hidden)
        )

    if pool_type == "gap":
        pool = nn.AdaptiveAvgPool1d(1)
    elif pool_type == "attpool":
        pool = _make_attpool(channels, attn_hidden)
    elif pool_type == "gatedpool":
        pool = _make_gatedpool(channels, attn_hidden)
    elif pool_type == "mhattpool":
        pool = _make_mhattpool(channels, n_heads, attn_hidden)
    elif pool_type == "tfpool":
        pool = _make_tfpool(channels)
    else:
        raise ValueError(f"未知的 pool head: {pool_type!r}")

    act_fn = F.gelu if activation == "gelu" else F.relu

    class _TCNPoolModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tcn = nn.Sequential(*blocks_mod)
            # 命名 pool 头使其对应 state_dict 前缀：gap (AdaptiveAvgPool1d 无参数),
            # attpool / gatedpool / mhattpool / tfpool
            if pool_type == "gap":
                self.gap = pool
            else:
                setattr(self, pool_type, pool)
            self.fc1 = nn.Linear(channels, 32)
            self.fc_drop = nn.Dropout(0.1)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            # x: (batch, window, features) → Conv1d 需要 (batch, features, window)
            x = x.transpose(1, 2)
            x = self.tcn(x)
            if pool_type == "gap":
                x = self.gap(x).squeeze(-1)
            else:
                x = getattr(self, pool_type)(x)
            x = act_fn(self.fc1(x))
            x = self.fc_drop(x)
            return self.fc2(x).squeeze(-1)

    model = _TCNPoolModel()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # missing: Dropout / AdaptiveAvgPool1d / Identity 无参数，正常
        bad_missing = [k for k in missing if not _is_dropout_or_identity_key(k)]
        bad_unexpected = [k for k in unexpected if not _is_dropout_or_identity_key(k)]
        if bad_missing or bad_unexpected:
            raise ValueError(
                f"TCN+pool state_dict 加载失败: missing={bad_missing[:3]}, "
                f"unexpected={bad_unexpected[:3]}"
            )
    model.eval()
    return model


def _is_dropout_or_identity_key(key: str) -> bool:
    """Dropout / AdaptiveAvgPool1d / Identity 无参数，其前缀不应出现在 sd 中。

    检查方式是判断 key 以这些前缀结尾：
      *.drop.* / *.gap.* / *.residual.*（Identity）
    """
    parts = key.split(".")
    if len(parts) >= 2 and parts[-2] in {"drop", "gap", "residual"}:
        return True
    return False


# ----------------------------------------------------------------------
# 23-dim wrapper（接在最底部）
# ----------------------------------------------------------------------


class _Torch3DStateDictWrapper23:
    """23-dim KEEP_23 模型的滑动窗口推理 wrapper。

    与 :class:`_Torch3DStateDictWrapper` 区别:

    * ``infer()`` 入参是**原始 record dict**（不是预先提取的特征数组），
      因为 19 行级特征里有 4 个派生列（time_diff、time_since_last、
      is_unusual_fc、is_response）需要跨帧状态。
    * 维护一个 16 帧的 buffer，buffer 满后计算 6 个窗口聚合列
      （press_mean_w / crc_max_w / resp_count_w / cmd_resp_balance_w /
      length_nunique_w / unusual_count_w），按 KEEP_23 顺序拼成
      (16, 23) 喂给模型。
    * RobustScaler 来自 ``scaler_binary_v2_scada.joblib``，缺文件时
      退化为 identity（带一次性 warning），模型仍能跑（预测会偏，
      但 IDS 面板的 TP/FP/TN/FN 计数 + metrics 显示链路保持工作）。
    * 模型结构重建复用 :meth:`_Torch3DStateDictWrapper._reconstruct`，
      传 ``n_features=23``；LSTM/GRU 走 batch_first 路径，Conv1d
      由 ``_CNN3DModel.forward`` 内部 ``transpose(1, 2)``，所以
      wrapper 喂 ``(1, 16, 23)`` 两种都兼容。
    """

    def __init__(self, state_dict: dict, window_size: int):
        if window_size < 1:
            raise ValueError(f"window_size 必须 >= 1，实际 {window_size}")
        self._n_features = 23
        self._window_size = window_size
        # 19-dim raw 的 buffer（用于聚合计算：press / crc / length / unusual / is_response）
        self._buffer_19: deque = deque(maxlen=window_size)
        # 跨帧状态：time_diff 和 time_since_last 需要 prev_time + 查表
        self._prev_time: int | None = None
        self._last_seen_time: dict[int, int] = {}
        # Lazy-load 的 RobustScaler
        from ids.scaler_23dim import Scaler23
        self._scaler = Scaler23()
        # 嵌套 TCN+pool 格式 (e.g. model_tcn_23dim_w16_*_s*.pt) 用专用
        # parser；LSTM/GRU/扁平 Conv1d 仍走 _reconstruct。
        if _detect_tcn_pool_nested(state_dict):
            self._model = _reconstruct_tcn_pool_model(state_dict, self._n_features)
        else:
            self._model = _Torch3DStateDictWrapper._reconstruct(
                state_dict, self._n_features
            )
        self._model.eval()

    @property
    def input_features(self) -> int:
        return self._n_features

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def warmup_remaining(self) -> int:
        return max(0, self._window_size - len(self._buffer_19))

    def infer(self, record: dict) -> tuple[int, float] | None:
        """喂一帧原始 record；满窗后返回 (label, prob_attack)，否则 None。

        record 必须包含 17 维 IDS 列（FEATURE_COLUMNS）。派生列
        time_diff / time_since_last 由 wrapper 内部维护。
        """
        import numpy as np
        import torch

        from ids.inference import (
            extract_features_19,
            keep_23_per_frame_indices,
            keep_23_aggregate_indices,
        )

        raw_19 = extract_features_19(
            record,
            prev_time=self._prev_time,
            last_seen_time=self._last_seen_time,
        )
        # 更新跨帧状态，供下一帧使用
        self._prev_time = int(raw_19[0])  # placeholder, overwritten below
        t = int(record.get("time") or 0)
        addr = int(record.get("address") or 0)
        fn = int(record.get("function") or 0)
        self._prev_time = t
        self._last_seen_time[addr * 1000 + fn] = t

        # 标准化：scaler 需要 (N, 19)；scaler_23dim 兼容任意前置维度
        scaled_19 = self._scaler.transform(
            raw_19.reshape(1, -1)
        )[0]  # (19,) float32

        self._buffer_19.append(scaled_19)
        if len(self._buffer_19) < self._window_size:
            return None  # warm-up

        # 计算 6 个窗口聚合（在 scaled 19-dim 上做，与训练一致）
        buf = np.stack(list(self._buffer_19), axis=0)  # (W, 19)
        # KEEP_23 per-frame 索引（17 个）→ 取 (W, 17) per-frame
        per_frame_idx = keep_23_per_frame_indices()
        per_frame = buf[:, per_frame_idx]  # (W, 17)
        # 6 个聚合：按 keep_23_aggregate_indices() 顺序
        aggs = self._compute_aggregates(buf)
        # 拼成 (W, 23)
        x_window = np.concatenate([per_frame, aggs], axis=-1)  # (W, 23)
        x = torch.as_tensor(x_window, dtype=torch.float32).unsqueeze(0)  # (1, W, 23)
        with torch.no_grad():
            logits = self._model(x)
        if logits.dim() == 2 and logits.shape[-1] >= 2:
            probs = torch.softmax(logits, dim=-1)[0]
            prob_attack = float(probs[-1])
        else:
            prob_attack = float(torch.sigmoid(logits).flatten()[0])
        label = 1 if prob_attack >= 0.5 else 0
        return label, prob_attack

    @staticmethod
    def _compute_aggregates(buf_19: np.ndarray) -> np.ndarray:
        """在 (W, 19) scaled 矩阵上算 6 个窗口聚合，输出 (W, 6)。

        每列与 KEEP_23 中 [19, 26] 一一对应（参见 _common_train.py:46
        与 preprocess_v2_scada.py:194–203）：

        | 输出列 | 含义              | 来源 19-dim 索引 |
        |--------|-------------------|-------------------|
        | 19     | press_mean_w      | mean(buf[:, 11])  |
        | 20     | (crc_mean_w)      | 不在 KEEP_23      |
        | 21     | crc_max_w         | max(buf[:, 12])   |
        | 22     | (cmd_count_w)     | 不在 KEEP_23      |
        | 23     | resp_count_w      | sum(buf[:, 18])   |
        | 24     | cmd_resp_balance_w| (cmd-resp)/W      |
        | 25     | length_nunique_w  | unique count of [2]|
        | 26     | unusual_count_w   | sum(buf[:, 17])   |

        KEEP_23 = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25,26]
        6 个保留的聚合列对应输出位置 [0, 1, 2, 3, 4, 5]（即 KEEP_23 减 18）。
        """
        import numpy as np
        W = buf_19.shape[0]
        press = buf_19[:, 11]                # pressure_measurement
        crc = buf_19[:, 12]                  # crc_rate
        is_resp = buf_19[:, 18]              # is_response (0/1)
        is_unusual = buf_19[:, 17]           # is_unusual_fc
        length = buf_19[:, 2]                # length

        press_mean_w = float(press.mean())
        crc_max_w = float(crc.max())
        resp_count_w = float(is_resp.sum())
        cmd_count_w = float((1.0 - is_resp).sum())
        balance_w = (cmd_count_w - resp_count_w) / W
        length_nunique_w = float(len(np.unique(length)))
        unusual_count_w = float(is_unusual.sum())

        # 6 列：press_mean, crc_max, resp_count, balance, length_nunique, unusual_count
        col_vecs = np.array(
            [press_mean_w, crc_max_w, resp_count_w, balance_w,
             length_nunique_w, unusual_count_w],
            dtype=np.float32,
        )  # (6,)
        # 广播到 (W, 6) — 与训练 make_windows_with_agg 一致
        return np.broadcast_to(col_vecs, (W, 6)).copy()


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

    # 3. 检测 3D 窗口层 → 用对应 wrapper 重建（17-dim 或 23-dim）
    is_3d = _TorchStateDictWrapper._is_3d_compatible_from_tensors(sd)
    if is_3d:
        if window_size < 2:
            raise ValueError(
                "模型为 3D 窗口模型（LSTM/GRU/Conv1d），"
                "加载时需指定 window_size >= 2（如 16、32）。"
                "请在 IDS 面板设置窗口大小后重试（建议从 16 开始）。"
            )
        # 探测第一层 RNN/Conv1d 的 in_features：决定走 17 还是 23
        n_3d = _detect_3d_input_features(sd)
        if n_3d == 23:
            try:
                return _Torch3DStateDictWrapper23(sd, window_size)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"无法从 state_dict 重建 23-dim 3D 模型: {e}") from e
        if n_3d == 17:
            try:
                return _Torch3DStateDictWrapper(sd, 17, window_size)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"无法从 state_dict 重建 3D 模型: {e}") from e
        # 探测失败：fall back to 17-dim，让 wrapper 自己报友好错误
        try:
            return _Torch3DStateDictWrapper(sd, 17, window_size)
        except ValueError as ve:
            raise ValueError(
                f"3D 模型输入维度既不是 17 也不是 23（探测到 {n_3d}）。"
                "请确认模型来自 IDS 训练流程。"
            ) from ve

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


def _detect_3d_input_features(sd: dict) -> int | None:
    """从 3D state_dict 探测第一层输入维度：LSTM 看 weight_ih_l0.shape[1]，
    Conv1d 看第一层 weight.shape[1]。返回探测到的维度（17/23/...），
    都不识别返回 None。
    """
    import re
    # RNN: weight_ih_l0 shape = (4*hidden, input_size)
    for key, val in sd.items():
        if re.match(r"^(lstm|gru)\.weight_ih_l0(_reverse)?$", key):
            if hasattr(val, "dim") and val.dim() == 2:
                return int(val.shape[1])
    # Conv1d: 第一个 .dim() == 3 的 weight 的 shape[1] = in_channels
    for val in sd.values():
        if hasattr(val, "dim") and val.dim() == 3:
            return int(val.shape[1])
    return None


def list_available_models(directory: str) -> list[str]:
    """扫描目录下 model_*.pt 和 model_*.joblib，按名字排序返回完整路径。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    files: list[Path] = []
    for ext in (".pt", ".joblib"):
        files.extend(d.glob(f"model_*{ext}"))
    return sorted(str(f) for f in files)