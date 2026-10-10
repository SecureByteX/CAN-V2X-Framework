"""Export the trained runtime classifiers to compact ONNX files.

Run with the development environment after training:
    python tools/export_onnx_models.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

import config


def _json_value(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to model metadata")


def _install_bool_attribute_fix(container_class):
    original_add_node = container_class.add_node

    def add_node_with_int_booleans(container, *args, **kwargs):
        fixed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                array = np.asarray(value)
                fixed_kwargs[key] = array.astype(np.int64).tolist() if array.dtype == np.bool_ else value
            else:
                fixed_kwargs[key] = value
        return original_add_node(container, *args, **fixed_kwargs)

    container_class.add_node = add_node_with_int_booleans
    return original_add_node


def _export(source_path, model_path, metadata_path):
    artifact = joblib.load(source_path)
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])
    initial_types = [("features", FloatTensorType([None, len(feature_names)]))]

    if model.__class__.__module__.startswith("xgboost"):
        booster = model.get_booster()
        original_names = booster.feature_names
        try:
            # The converter resolves XGBoost tree nodes by f0, f1, ... names.
            # The predictor always supplies columns in artifact feature order.
            booster.feature_names = [f"f{index}" for index in range(len(feature_names))]
            from onnxconverter_common.container import ModelComponentContainer
            original_add_node = _install_bool_attribute_fix(ModelComponentContainer)
            converted = convert_xgboost(
                model,
                initial_types=initial_types,
                target_opset=15,
            )
        finally:
            if 'original_add_node' in locals():
                ModelComponentContainer.add_node = original_add_node
            booster.feature_names = original_names
    else:
        from skl2onnx.common._container import ModelComponentContainer
        original_add_node = _install_bool_attribute_fix(ModelComponentContainer)
        try:
            converted = convert_sklearn(
                model,
                initial_types=initial_types,
                target_opset=15,
                options={id(model): {"zipmap": False}},
            )
        finally:
            ModelComponentContainer.add_node = original_add_node

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as model_file:
        model_file.write(converted.SerializeToString())

    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"model", "label_encoder", "source_path"}
    }
    metadata["feature_names"] = feature_names
    metadata["classes"] = artifact["label_encoder"].classes_.tolist()
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, default=_json_value, separators=(",", ":"))

    print(f"Exported {source_path} -> {model_path}")


if __name__ == "__main__":
    _export(config.CAN_MODEL_PATH, config.CAN_ONNX_MODEL_PATH, config.CAN_MODEL_METADATA_PATH)
    _export(config.V2X_MODEL_PATH, config.V2X_ONNX_MODEL_PATH, config.V2X_MODEL_METADATA_PATH)
    _export(
        config.V2X_VEREMI_MODEL_PATH,
        config.V2X_VEREMI_ONNX_MODEL_PATH,
        config.V2X_VEREMI_MODEL_METADATA_PATH,
    )
