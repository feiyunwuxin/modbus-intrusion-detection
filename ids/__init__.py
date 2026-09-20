"""Modbus 入侵检测子包。

公开接口:
    FEATURE_COLUMNS: 17 个特征列名（与 SCADA 训练流程一致）
    extract_features: CSV 记录 → (17,) float32 数组
    ModelWrapper: 统一模型推理接口
    load_model: 按扩展名加载 .pt / .joblib
    list_available_models: 扫描目录下可用模型文件
"""
from .inference import FEATURE_COLUMNS, extract_features
from .model_loader import ModelWrapper, load_model, list_available_models

__all__ = [
    "FEATURE_COLUMNS",
    "extract_features",
    "ModelWrapper",
    "load_model",
    "list_available_models",
]