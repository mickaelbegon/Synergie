from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import constants
from synergie import pretrained_models
from synergie import session_store
from synergie.services.annotation_service import (
    ANNOTATION_EDGE_JUMP_OPTIONS,
    ANNOTATION_JUMP_TYPE_OPTIONS,
    ANNOTATION_METADATA_DEFAULTS,
    ANNOTATION_REVIEW_STATUS_OPTIONS,
    ANNOTATION_TOE_JUMP_OPTIONS,
    JUMP_TYPE_LABELS,
    annotate_combination_flags,
    annotation_metadata_path,
    annotation_review_status_from_row,
    annotation_review_status_to_backend,
    annotation_turn_options,
    annotation_turn_value_for_storage,
    annotation_turn_value_for_ui,
    compute_annotation_jump_video_time_ms,
    get_annotation_sensor_sync_offset,
    load_annotation_metadata,
    save_annotation_metadata,
    set_annotation_sensor_sync_offset,
    set_annotation_video_directory,
    set_annotation_video_path,
    summarize_annotation_progress,
)
from synergie.services.annotation_finalization_service import finalize_annotation_file
from synergie.services.annotation_generation_service import (
    estimate_sensor_impact_offset_ms,
    process_new_imu_file_for_annotation,
    process_new_imu_session_for_annotation,
)
from synergie.services.hyperparameter_search_service import (
    hyperparameter_search_space,
    sample_hyperparameter_trials,
)
from synergie.services.detection_tuning_service import (
    analyze_detection_review_labels,
    optimize_detection_parameters,
)
from synergie.services.quality_service import analyze_jump_quality
from synergie.services.signal_importance_service import compute_signal_importance
from synergie.services.new_data_service import (
    files_for_new_imu_session,
    list_new_data_directories,
    list_new_imu_files,
    list_new_imu_sessions,
    new_imu_session_key,
    parse_new_imu_filename,
    suggest_for_annotation_output_path,
)
from synergie.services.training_dataset_service import (
    describe_training_dataset,
    find_training_dataset_duplicates,
    load_training_dataset_state,
    training_dataset_state_path,
)
from synergie.services.video_service import (
    annotation_reference_datetime,
    list_video_files,
    parse_datetime_from_text as _parse_datetime_from_text,
    parse_datetime_value as _parse_datetime_value,
    read_video_creation_time_with_ffprobe as _read_video_creation_time_with_ffprobe,
    select_best_video_datetime as _select_best_video_datetime,
)


def list_sessions() -> list[str]:
    constants.sessions = session_store.load_sessions()
    return sorted(constants.sessions)


def session_metadata(session_name: str) -> dict:
    constants.sessions = session_store.load_sessions()
    return constants.get_session(session_name)


def add_session(session_name: str, path: str, sample_time_fine_synchro: int) -> dict:
    metadata = session_store.add_session(session_name, path, sample_time_fine_synchro)
    constants.sessions = session_store.load_sessions()
    return metadata


def session_synchro(session_name: str) -> int:
    return int(session_metadata(session_name)["sample_time_fine_synchro"])


def list_session_csv_files(session_name: str, raw_root: str = "data/raw") -> list[Path]:
    metadata = session_metadata(session_name)
    session_dir = Path(raw_root) / metadata["path"]
    if not session_dir.exists():
        return []
    return sorted(session_dir.glob("*.csv"))


def list_directory_files(directory: str | Path) -> list[Path]:
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    return sorted(path for path in directory_path.iterdir() if path.is_file())


def session_directory(session_name: str, raw_root: str = "data/raw") -> Path:
    return Path(raw_root) / session_metadata(session_name)["path"]


def describe_file(path: str | Path) -> dict:
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "path": file_path,
        "name": file_path.name,
        "suffix": file_path.suffix.lower(),
        "size_bytes": stat.st_size,
    }


def next_jumplist_output_path(directory: str | Path) -> Path:
    directory_path = Path(directory)
    index = 1
    while True:
        candidate = directory_path / f"jumplist_partie{index}.csv"
        if not candidate.exists():
            return candidate
        index += 1


def read_video_metadata(video_path: str | Path) -> dict:
    """Read video timestamps while preserving patchable helper aliases."""
    path = Path(video_path)
    stat = path.stat()
    candidates: list[tuple[str, datetime]] = []
    ffprobe_datetime = _read_video_creation_time_with_ffprobe(path)
    if ffprobe_datetime is not None:
        candidates.append(("ffprobe.creation_time", ffprobe_datetime))
    filename_datetime = _parse_datetime_from_text(path.stem)
    if filename_datetime is not None:
        candidates.append(("filename", filename_datetime))
    candidates.append(("filesystem.modified", datetime.fromtimestamp(stat.st_mtime)))
    candidates.append(("filesystem.created", datetime.fromtimestamp(stat.st_ctime)))
    best_source, best_datetime = _select_best_video_datetime(candidates)
    return {
        "path": path,
        "name": path.name,
        "recorded_at": best_datetime,
        "recorded_at_source": best_source,
        "candidates": [{"source": source, "recorded_at": value} for source, value in candidates],
    }


def find_matching_videos(
    annotation_csv_path: str | Path,
    video_directory: str | Path,
    *,
    annotation_rows=None,
    recursive: bool = True,
    limit: int = 5,
) -> dict:
    """Find nearest videos while preserving the historical operations API."""
    reference_datetime = annotation_reference_datetime(annotation_csv_path, annotation_rows=annotation_rows)
    matches: list[dict] = []
    for video_path in list_video_files(video_directory, recursive=recursive):
        metadata = read_video_metadata(video_path)
        delta_seconds = None
        if reference_datetime is not None:
            delta_seconds = abs((metadata["recorded_at"] - reference_datetime).total_seconds())
        matches.append(
            {
                "path": metadata["path"],
                "name": metadata["name"],
                "recorded_at": metadata["recorded_at"],
                "recorded_at_source": metadata["recorded_at_source"],
                "delta_seconds": delta_seconds,
            }
        )
    matches.sort(
        key=lambda item: (
            float("inf") if item["delta_seconds"] is None else item["delta_seconds"],
            item["recorded_at"],
            item["name"].lower(),
        )
    )
    return {
        "reference_datetime": reference_datetime,
        "video_directory": Path(video_directory),
        "matches": matches[:limit],
        "match_count": len(matches),
    }


def parse_training_id_from_csv_path(csv_path: Path) -> str | None:
    training_parts = csv_path.stem.split("_")
    if len(training_parts) < 2:
        return None
    return training_parts[1]


def list_pretrained_training_models(task: str | None = None, compatible_only: bool = False) -> list[dict]:
    return pretrained_models.list_pretrained_models(task=task, compatible_only=compatible_only)


def format_pretrained_model_label(model_entry: dict) -> str:
    performance = model_entry.get("performance") or {}
    accuracy = performance.get("test_accuracy")
    if accuracy is None:
        accuracy_text = "acc n/a"
    else:
        accuracy_text = f"acc {accuracy:.3f}"
    status = "compatible" if model_entry.get("compatible") and model_entry.get("path_exists") else "not compatible"
    return (
        f"{model_entry['label']} | {model_entry['architecture']} | "
        f"{accuracy_text} | {status}"
    )


def latest_model_path_for_task(task: str) -> str:
    """Return the active model alias used by inference for one task."""
    if task == "type":
        return constants.modeltype_filepath
    if task == "success":
        return constants.modelsuccess_filepath
    raise ValueError(f"Unsupported training task: {task}")


def summarize_pending_annotation_files(root: str | Path = "data/pending") -> dict:
    """Return per-file and global pending annotation counts."""
    import pandas as pd

    file_summaries: list[dict] = []
    global_summary = {"total": 0, "pending": 0, "completed": 0}
    for path in list_pending_annotation_files(root):
        frame = pd.read_csv(path)
        summary = summarize_annotation_progress(frame)
        file_summaries.append({"path": path, **summary})
        for key in global_summary:
            global_summary[key] += summary[key]
    return {"files": file_summaries, **global_summary}


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


def suggest_turns_from_rotation(rotation_value) -> str:
    """Convert a measured absolute rotation into a conservative 1-4 turn guess."""
    turns = int(round(_safe_float(rotation_value, default=1.0)))
    return str(min(4, max(1, turns)))


def prefill_annotation_predictions(
    annotation_rows,
    *,
    type_model_path: str,
    success_model_path: str,
) -> dict:
    """
    Predict type/success labels for annotation rows and prefill turn guesses.

    Annotation candidates are not yet linked to a validated athlete profile, so
    this V1 uses neutral scalar inputs. These predictions are intended as review
    aids only; the human annotation remains authoritative.
    """
    from synergie.services.prediction_service import PredictionService

    frame = annotation_rows.copy()
    if frame.empty:
        return {"rows": frame, "updated": 0, "skipped": 0}

    service = PredictionService.from_paths(type_model_path, success_model_path)
    result = service.prefill_annotation_rows(frame)
    frame = result["rows"]
    for index in frame.index:
        if frame.at[index, "type"] == 8:
            continue
        frame.at[index, "turns"] = suggest_turns_from_rotation(frame.at[index, "rotations"])
        frame.at[index, "prediction_source"] = "batch_model_prefill"
    result["rows"] = frame
    return result


def audit_saved_models() -> list[dict]:
    """
    Inspect active and registered models under the current Python environment.

    The audit distinguishes Keras 3 training-compatible files from legacy
    TensorFlow SavedModel directories. Legacy directories can still be used for
    inference through ``LegacySavedModelPredictor`` but cannot be resumed for
    training with Keras 3.
    """
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
        model_path = Path(path)
        item = dict(candidate)
        item["path"] = path
        item["exists"] = model_path.exists()
        item["format"] = _model_format(model_path)
        item["training_compatible"] = pretrained_models.supports_training_reload(model_path)
        item["can_infer"] = False
        item["can_resume_training"] = False
        item["input_shapes"] = []
        item["error"] = ""

        if not item["exists"]:
            item["error"] = "missing path"
            audited.append(item)
            continue

        try:
            loaded = model.load_model(path, for_training=False)
            item["can_infer"] = True
            item["input_shapes"] = [
                tuple(dimension if dimension is None else int(dimension) for dimension in model_input.shape)
                for model_input in getattr(loaded, "inputs", [])
            ]
        except Exception as exc:
            item["error"] = f"inference load failed: {exc}"

        if item["training_compatible"]:
            try:
                model.load_model(path, for_training=True)
                item["can_resume_training"] = True
            except Exception as exc:
                if item["error"]:
                    item["error"] += f" | training reload failed: {exc}"
                else:
                    item["error"] = f"training reload failed: {exc}"
        audited.append(item)
    return audited


def _model_format(path: Path) -> str:
    if path.is_file() and path.suffix.lower() == ".keras":
        return "keras_v3"
    if path.is_file() and path.suffix.lower() == ".h5":
        return "keras_h5"
    if path.is_dir() and (path / "saved_model.pb").exists():
        return "legacy_saved_model"
    return "unknown"


def run_hyperparameter_search(
    task: str,
    dataset_path: str,
    architecture: str,
    *,
    max_trials: int = 6,
    epochs: int = 8,
    random_seed: int = 42,
) -> dict:
    """
    Run a compact exploratory hyperparameter search on the validation split.

    This is intentionally a practical first pass rather than a publication
    protocol: it ranks bounded candidates on the current validation split, uses
    early stopping, and does not claim an unbiased final test estimate.
    """
    import keras
    import numpy as np
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
    from core.model import model
    from core.model.training.loader import Loader
    from core.model.training.training import Trainer

    duplicate_report = find_training_dataset_duplicates(dataset_path)
    if duplicate_report["has_duplicates"]:
        preview = ", ".join(duplicate_report["duplicate_paths"][:3])
        suffix = "..." if len(duplicate_report["duplicate_paths"]) > 3 else ""
        raise ValueError(
            f"Training dataset contains duplicate jump paths ({duplicate_report['duplicate_rows']} rows): "
            f"{preview}{suffix}"
        )
    dataset = Loader(dataset_path, augment_mirror=True)
    pretrained_entry = None
    if pretrained_model_id:
        pretrained_entry = pretrained_models.get_pretrained_model(pretrained_model_id)
        if pretrained_entry["task"] != task:
            raise ValueError(f"Pretrained model '{pretrained_model_id}' is not compatible with task '{task}'.")
        if not pretrained_entry.get("compatible") or not pretrained_entry.get("path_exists"):
            raise ValueError(f"Pretrained model '{pretrained_model_id}' is not currently compatible.")

    selected_architecture = architecture or (
        pretrained_entry["architecture"] if pretrained_entry else model.default_architecture(task)
    )
    run_path = pretrained_models.build_trained_model_path(task, selected_architecture)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    if task == "type":
        model_instance = (
            model.load_model(pretrained_entry["path"], for_training=True)
            if pretrained_entry
            else model.build_model(task, selected_architecture, **(model_overrides or {}))
        )
        trainer = Trainer(
            dataset.get_type_data(),
            model_instance,
            str(run_path),
        )
        summary = trainer.train(epochs=epochs or 10, batch_size=batch_size)
        latest_path = Path(constants.modeltype_filepath)
        _promote_trained_model(run_path, latest_path)
        registered = pretrained_models.register_trained_model(
            model_id=run_path.stem,
            label=f"Type {selected_architecture} {run_path.stem}",
            task=task,
            architecture=selected_architecture,
            path=str(run_path).replace("\\", "/"),
            dataset=dataset_path,
            performance=summary,
            notes="Automatically registered after GUI/CLI training.",
        )
        summary["saved_model"] = registered
        summary["latest_model_path"] = str(latest_path).replace("\\", "/")
        summary["task"] = task
        return summary

    if task == "success":
        model_instance = (
            model.load_model(pretrained_entry["path"], for_training=True)
            if pretrained_entry
            else model.build_model(task, selected_architecture, **(model_overrides or {}))
        )
        trainer = Trainer(
            dataset.get_success_data(),
            model_instance,
            str(run_path),
        )
        summary = trainer.train_success(epochs=epochs or 20, batch_size=batch_size)
        latest_path = Path(constants.modelsuccess_filepath)
        _promote_trained_model(run_path, latest_path)
        registered = pretrained_models.register_trained_model(
            model_id=run_path.stem,
            label=f"Success {selected_architecture} {run_path.stem}",
            task=task,
            architecture=selected_architecture,
            path=str(run_path).replace("\\", "/"),
            dataset=dataset_path,
            performance=summary,
            notes="Automatically registered after GUI/CLI training.",
        )
        summary["saved_model"] = registered
        summary["latest_model_path"] = str(latest_path).replace("\\", "/")
        summary["task"] = task
        return summary

    raise ValueError(f"Unsupported training task: {task}")


def _promote_trained_model(source_path: Path, latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_path.is_dir():
        shutil.rmtree(latest_path, onerror=_handle_remove_readonly)
    elif latest_path.exists():
        latest_path.unlink()
    shutil.copy2(source_path, latest_path)


def _handle_remove_readonly(function, path, _excinfo) -> None:
    os.chmod(path, 0o700)
    function(path)


def process_csv_file(
    csv_path: str,
    synchro: int = 0,
    output_path: str | None = None,
    *,
    type_model_path: str | None = None,
    success_model_path: str | None = None,
) -> dict:
    from core.data_treatment.data_generation.exporter import export
    import pandas as pd

    input_path = Path(csv_path)
    dataframe = pd.read_csv(input_path)
    result = export(
        dataframe,
        sampleTimeFineSynchro=synchro,
        type_model_path=type_model_path,
        success_model_path=success_model_path,
    )
    destination = Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_jumps.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return {
        "path": destination,
        "prediction_status": result.attrs.get("prediction_status", "unknown"),
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def list_pending_annotation_files(root: str | Path = "data/pending") -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        path for path in root_path.glob("*for_annotation*.csv")
        if path.is_file()
    )


def repredict_raw_trainings(raw_root: str = "data/raw") -> int:
    from core.data_treatment.data_generation.exporter import export
    from core.database.DatabaseManager import DatabaseManager
    import pandas as pd
    from synergie.services.jump_predictions import build_training_jump_payload

    raw_root_path = Path(raw_root)
    database = DatabaseManager()
    processed = 0

    for csv_path in sorted(raw_root_path.rglob("*.csv")):
        training_id = parse_training_id_from_csv_path(csv_path)
        if training_id is None:
            continue
        dataframe = pd.read_csv(csv_path)
        predictions = export(dataframe)
        training_jumps = build_training_jump_payload(predictions, training_id)
        database.add_jumps_to_training(training_id, training_jumps)
        processed += 1

    return processed
