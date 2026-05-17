from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PRETRAINED_MODELS_FILE = Path("config") / "pretrained_models.json"
MODEL_ARCHIVE_ROOT = Path("core") / "model" / "saved_models" / "archive"


def supports_training_reload(path: str | Path) -> bool:
    model_path = Path(path)
    return model_path.is_file() and model_path.suffix.lower() in {".keras", ".h5"}


def load_pretrained_models() -> list[dict]:
    if not PRETRAINED_MODELS_FILE.exists():
        return []
    with PRETRAINED_MODELS_FILE.open("r", encoding="utf-8") as handle:
        models = json.load(handle)
    for model in models:
        model["path_exists"] = Path(model["path"]).exists()
        model["compatible"] = bool(model.get("compatible", True)) and supports_training_reload(model["path"])
    return models


def list_pretrained_models(task: str | None = None, compatible_only: bool = False) -> list[dict]:
    models = load_pretrained_models()
    filtered: list[dict] = []
    for model in models:
        if task is not None and model["task"] != task:
            continue
        if compatible_only and not (model.get("compatible") and model.get("path_exists")):
            continue
        filtered.append(model)
    filtered.sort(key=_model_recency_sort_key, reverse=True)
    return filtered


def get_pretrained_model(model_id: str) -> dict:
    for model in load_pretrained_models():
        if model["id"] == model_id:
            return model
    raise KeyError(f"Unknown pretrained model '{model_id}'.")


def save_pretrained_models(models: list[dict]) -> None:
    PRETRAINED_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialized: list[dict[str, Any]] = []
    for model in models:
        item = dict(model)
        item.pop("path_exists", None)
        serialized.append(item)
    with PRETRAINED_MODELS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def _model_recency_sort_key(model: dict) -> tuple[datetime, float, str]:
    model_id = str(model.get("id", ""))
    timestamp_match = re.search(r"(\d{8}-\d{6})", model_id)
    if timestamp_match:
        try:
            parsed = datetime.strptime(timestamp_match.group(1), "%Y%m%d-%H%M%S")
            return (parsed, 0.0, model_id)
        except ValueError:
            pass

    model_path = Path(model.get("path", ""))
    if model_path.exists():
        return (datetime.min, model_path.stat().st_mtime, model_id)
    return (datetime.min, 0.0, model_id)


def build_trained_model_id(task: str, architecture: str, trained_at: datetime | None = None) -> str:
    timestamp = (trained_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{_slugify(task)}-{_slugify(architecture)}-{timestamp}"


def build_trained_model_path(task: str, architecture: str, trained_at: datetime | None = None) -> Path:
    model_id = build_trained_model_id(task, architecture, trained_at=trained_at)
    return MODEL_ARCHIVE_ROOT / _slugify(task) / f"{model_id}.keras"


def register_trained_model(
    *,
    model_id: str,
    label: str,
    task: str,
    architecture: str,
    path: str,
    dataset: str,
    performance: dict,
    notes: str,
    compatible: bool = True,
) -> dict:
    models = load_pretrained_models()
    entry = {
        "id": model_id,
        "label": label,
        "task": task,
        "architecture": architecture,
        "path": path,
        "compatible": compatible,
        "notes": notes,
        "performance": {
            "dataset": dataset,
            "test_accuracy": performance.get("test_accuracy"),
            "test_samples": performance.get("test_samples"),
            "confusion_matrix": performance.get("confusion_matrix"),
        },
    }
    models = [model for model in models if model["id"] != model_id]
    models.append(entry)
    save_pretrained_models(models)
    entry["path_exists"] = Path(path).exists()
    return entry
