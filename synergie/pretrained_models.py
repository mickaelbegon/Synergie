from __future__ import annotations

import json
from pathlib import Path


PRETRAINED_MODELS_FILE = Path("config") / "pretrained_models.json"


def load_pretrained_models() -> list[dict]:
    if not PRETRAINED_MODELS_FILE.exists():
        return []
    with PRETRAINED_MODELS_FILE.open("r", encoding="utf-8") as handle:
        models = json.load(handle)
    for model in models:
        model["path_exists"] = Path(model["path"]).exists()
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
    return filtered


def get_pretrained_model(model_id: str) -> dict:
    for model in load_pretrained_models():
        if model["id"] == model_id:
            return model
    raise KeyError(f"Unknown pretrained model '{model_id}'.")
