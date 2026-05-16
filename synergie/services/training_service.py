from __future__ import annotations

import os
import shutil
from pathlib import Path

import constants
from synergie import pretrained_models
from synergie.services.hyperparameter_search_service import sample_hyperparameter_trials
from synergie.services.training_dataset_service import find_training_dataset_duplicates


def run_hyperparameter_search(
    task: str,
    dataset_path: str,
    architecture: str,
    *,
    max_trials: int = 6,
    epochs: int = 8,
    random_seed: int = 42,
) -> dict:
    """Run a compact exploratory validation search."""
    import keras
    from core.model import model
    from core.model.training.loader import Loader

    loader = Loader(dataset_path, augment_mirror=True)
    dataset = loader.get_type_data() if task == "type" else loader.get_success_data()
    trials = sample_hyperparameter_trials(task, architecture, max_trials, random_seed=random_seed)
    results: list[dict] = []
    validation_data = (
        {"temporal_input": dataset.temporal_features_test, "scalar_input": dataset.scalar_features_test},
        dataset.labels_test,
    )
    for trial_index, parameters in enumerate(trials, start=1):
        keras.backend.clear_session()
        build_parameters = dict(parameters)
        batch_size = int(build_parameters.pop("batch_size"))
        model_instance = model.build_model(task, architecture, **build_parameters)
        history = model_instance.fit(
            {"temporal_input": dataset.temporal_features_train, "scalar_input": dataset.scalar_features_train},
            dataset.labels_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            class_weight=(dataset.class_weight or None) if task == "success" else None,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=3,
                    restore_best_weights=True,
                )
            ],
        )
        history_data = history.history
        results.append(
            {
                "trial": trial_index,
                "parameters": parameters,
                "best_val_accuracy": float(max(history_data.get("val_accuracy", [0.0]))),
                "final_val_accuracy": float(history_data.get("val_accuracy", [0.0])[-1]),
                "epochs_ran": int(len(history_data.get("loss", []))),
            }
        )
    results.sort(key=lambda item: item["best_val_accuracy"], reverse=True)
    return {
        "task": task,
        "architecture": architecture,
        "dataset_path": dataset_path,
        "max_trials": int(max_trials),
        "epochs": int(epochs),
        "objective": "best_val_accuracy",
        "results": results,
        "best_trial": results[0] if results else None,
        "note": "Exploratory validation search only; use grouped cross-validation before publication.",
    }


def train_model(
    task: str,
    dataset_path: str,
    epochs: int | None = None,
    architecture: str | None = None,
    pretrained_model_id: str | None = None,
    model_overrides: dict | None = None,
    batch_size: int | None = None,
) -> dict:
    """Train one model, promote it as latest, and register it for reuse."""
    from core.model import model
    from core.model.training.loader import Loader
    from core.model.training.training import Trainer

    _raise_for_duplicate_training_paths(dataset_path)
    dataset = Loader(dataset_path, augment_mirror=True)
    pretrained_entry = _resolve_pretrained_entry(task, pretrained_model_id)
    selected_architecture = architecture or (
        pretrained_entry["architecture"] if pretrained_entry else model.default_architecture(task)
    )
    run_path = pretrained_models.build_trained_model_path(task, selected_architecture)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    model_instance = (
        model.load_model(pretrained_entry["path"], for_training=True)
        if pretrained_entry
        else model.build_model(task, selected_architecture, **(model_overrides or {}))
    )
    if task == "type":
        summary = Trainer(dataset.get_type_data(), model_instance, str(run_path)).train(
            epochs=epochs or 10,
            batch_size=batch_size,
        )
        return _finalize_training_run(task, selected_architecture, dataset_path, run_path, Path(constants.modeltype_filepath), summary)
    if task == "success":
        summary = Trainer(dataset.get_success_data(), model_instance, str(run_path)).train_success(
            epochs=epochs or 20,
            batch_size=batch_size,
        )
        return _finalize_training_run(task, selected_architecture, dataset_path, run_path, Path(constants.modelsuccess_filepath), summary)
    raise ValueError(f"Unsupported training task: {task}")


def promote_trained_model(source_path: Path, latest_path: Path) -> None:
    """Replace the active model alias with one newly trained model file."""
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_path.is_dir():
        shutil.rmtree(latest_path, onerror=_handle_remove_readonly)
    elif latest_path.exists():
        latest_path.unlink()
    shutil.copy2(source_path, latest_path)


def _raise_for_duplicate_training_paths(dataset_path: str) -> None:
    duplicate_report = find_training_dataset_duplicates(dataset_path)
    if not duplicate_report["has_duplicates"]:
        return
    preview = ", ".join(duplicate_report["duplicate_paths"][:3])
    suffix = "..." if len(duplicate_report["duplicate_paths"]) > 3 else ""
    raise ValueError(
        f"Training dataset contains duplicate jump paths ({duplicate_report['duplicate_rows']} rows): "
        f"{preview}{suffix}"
    )


def _resolve_pretrained_entry(task: str, pretrained_model_id: str | None) -> dict | None:
    if not pretrained_model_id:
        return None
    pretrained_entry = pretrained_models.get_pretrained_model(pretrained_model_id)
    if pretrained_entry["task"] != task:
        raise ValueError(f"Pretrained model '{pretrained_model_id}' is not compatible with task '{task}'.")
    if not pretrained_entry.get("compatible") or not pretrained_entry.get("path_exists"):
        raise ValueError(f"Pretrained model '{pretrained_model_id}' is not currently compatible.")
    return pretrained_entry


def _finalize_training_run(
    task: str,
    architecture: str,
    dataset_path: str,
    run_path: Path,
    latest_path: Path,
    summary: dict,
) -> dict:
    promote_trained_model(run_path, latest_path)
    label_prefix = "Type" if task == "type" else "Success"
    registered = pretrained_models.register_trained_model(
        model_id=run_path.stem,
        label=f"{label_prefix} {architecture} {run_path.stem}",
        task=task,
        architecture=architecture,
        path=str(run_path).replace("\\", "/"),
        dataset=dataset_path,
        performance=summary,
        notes="Automatically registered after GUI/CLI training.",
    )
    summary["saved_model"] = registered
    summary["latest_model_path"] = str(latest_path).replace("\\", "/")
    summary["task"] = task
    return summary


def _handle_remove_readonly(function, path, _excinfo) -> None:
    os.chmod(path, 0o700)
    function(path)
