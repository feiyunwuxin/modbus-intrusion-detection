"""Modbus 入侵检测子包。

公开接口:
    FEATURE_COLUMNS: 17 个特征列名（与 17-dim IDS 训练流程一致）
    extract_features: CSV 记录 → (17,) float32 数组
    FEATURE_COLUMNS_19: 19 个 SCADA v2 行级特征名（23-dim 路径用）
    extract_features_19: CSV 记录 → (19,) float32 数组（raw, pre-scale）
    ModelWrapper: 统一模型推理接口
    load_model: 按扩展名加载 .pt / .joblib
    list_available_models: 扫描目录下可用模型文件
"""
from .inference import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_19,
    extract_features,
    extract_features_19,
    keep_23_per_frame_indices,
    keep_23_aggregate_indices,
)
from .model_loader import ModelWrapper, load_model, list_available_models

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_COLUMNS_19",
    "extract_features",
    "extract_features_19",
    "ModelWrapper",
    "load_model",
    "list_available_models",
]