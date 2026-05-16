from __future__ import annotations

from pathlib import Path

import constants
from synergie import pretrained_models


def list_pretrained_training_models(task: str | None = None, compatible_only: bool = False) -> list[dict]:
    """Return registered pretrained models with optional filtering."""
    return pretrained_models.list_pretrained_models(task=task, compatible_only=compatible_only)


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
