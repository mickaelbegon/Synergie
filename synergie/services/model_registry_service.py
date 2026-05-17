from __future__ import annotations

from pathlib import Path

import constants
from synergie import pretrained_models


def list_pretrained_training_models(task: str | None = None, compatible_only: bool = False) -> list[dict]:
    """Return registered pretrained models with optional filtering."""
    return pretrained_models.list_pretrained_models(task=task, compatible_only=compatible_only)


def delete_pretrained_training_model(model_id: str) -> dict:
    """Delete one registered archived model and its checkpoint file."""
    return pretrained_models.delete_pretrained_model(model_id)


def format_pretrained_model_label(model_entry: dict) -> str:
    """Return the GUI label used for one pretrained model entry."""
    performance = model_entry.get("performance") or {}
    accuracy = performance.get("test_accuracy")
    accuracy_text = "acc n/a" if accuracy is None else f"acc {accuracy:.3f}"
    status = "compatible" if model_entry.get("compatible") and model_entry.get("path_exists") else "not compatible"
    return f"{model_entry['label']} | {model_entry['architecture']} | {accuracy_text} | {status}"


def latest_model_path_for_task(task: str) -> str:
    """Return the active model alias used by inference for one task."""
    if task == "type":
        return constants.modeltype_filepath
    if task == "success":
        return constants.modelsuccess_filepath
    raise ValueError(f"Unsupported training task: {task}")


def list_pretrained_models_by_performance(task: str) -> list[dict]:
    """Return compatible models ordered by reported test accuracy."""
    models = list_pretrained_training_models(task=task, compatible_only=True)
    return sorted(
        models,
        key=lambda item: (
            item.get("performance", {}).get("test_accuracy") is not None,
            item.get("performance", {}).get("test_accuracy") or -1.0,
        ),
        reverse=True,
    )


def evaluate_registered_model(model_entry: dict) -> dict:
    """Return stored or recomputed evaluation metrics for one registered model."""
    performance = dict(model_entry.get("performance") or {})
    if performance.get("confusion_matrix"):
        return {
            "task": model_entry["task"],
            "confusion_matrix": performance["confusion_matrix"],
            "test_accuracy": performance.get("test_accuracy"),
            "test_samples": performance.get("test_samples"),
        }

    from core.model import model
    from core.model.training.loader import Loader
    from sklearn.metrics import accuracy_score, confusion_matrix
    import numpy as np

    dataset_path = performance.get("dataset") or "data/annotated/total"
    dataset = Loader(dataset_path, augment_mirror=True)
    data = dataset.get_type_data() if model_entry["task"] == "type" else dataset.get_success_data()
    trained_model = model.load_model(model_entry["path"], for_training=False)
    predicted = trained_model.predict(
        {"temporal_input": data.temporal_features_test, "scalar_input": data.scalar_features_test},
        verbose=0,
    )
    predicted_labels = [int(np.argmax(row)) for row in predicted]
    true_labels = [int(np.argmax(row)) for row in data.labels_test]
    return {
        "task": model_entry["task"],
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
        "test_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "test_samples": int(len(true_labels)),
    }


def audit_saved_models() -> list[dict]:
    """Inspect active and registered models under the current environment."""
    from core.model import model

    candidates: list[dict] = [
        {"label": "Active type model", "task": "type", "path": latest_model_path_for_task("type")},
        {"label": "Active success model", "task": "success", "path": latest_model_path_for_task("success")},
        {"label": "Legacy type model alias", "task": "type", "path": str(Path(latest_model_path_for_task("type")).with_suffix(""))},
        {"label": "Legacy success model alias", "task": "success", "path": str(Path(latest_model_path_for_task("success")).with_suffix(""))},
    ]
    candidates.extend(
        {
            "label": entry["label"],
            "task": entry["task"],
            "architecture": entry["architecture"],
            "path": entry["path"],
            "registered": True,
        }
        for entry in list_pretrained_training_models()
    )

    audited: list[dict] = []
    seen_paths: set[str] = set()
    for candidate in candidates:
        path = str(candidate["path"]).replace("\\", "/")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        audited.append(_audit_model_candidate(candidate, path, model))
    return audited


def model_format(path: Path) -> str:
    """Return the storage format for one model path."""
    if path.is_file() and path.suffix.lower() == ".keras":
        return "keras_v3"
    if path.is_file() and path.suffix.lower() == ".h5":
        return "keras_h5"
    if path.is_dir() and (path / "saved_model.pb").exists():
        return "legacy_saved_model"
    return "unknown"


def _audit_model_candidate(candidate: dict, path: str, model_module) -> dict:
    model_path = Path(path)
    item = dict(candidate)
    item["path"] = path
    item["exists"] = model_path.exists()
    item["format"] = model_format(model_path)
    item["training_compatible"] = pretrained_models.supports_training_reload(model_path)
    item["can_infer"] = False
    item["can_resume_training"] = False
    item["input_shapes"] = []
    item["error"] = ""
    if not item["exists"]:
        item["error"] = "missing path"
        return item
    try:
        loaded = model_module.load_model(path, for_training=False)
        item["can_infer"] = True
        item["input_shapes"] = [
            tuple(dimension if dimension is None else int(dimension) for dimension in model_input.shape)
            for model_input in getattr(loaded, "inputs", [])
        ]
    except Exception as exc:
        item["error"] = f"inference load failed: {exc}"
    if item["training_compatible"]:
        try:
            model_module.load_model(path, for_training=True)
            item["can_resume_training"] = True
        except Exception as exc:
            item["error"] = f"{item['error']} | training reload failed: {exc}".strip(" |")
    return item
