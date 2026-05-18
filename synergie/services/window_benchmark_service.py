from __future__ import annotations

import logging

from synergie.config import (
    SEGMENT_FRAMES_AFTER_TAKEOFF,
    SEGMENT_FRAMES_BEFORE_TAKEOFF,
    SUCCESS_WINDOW_FRAMES,
    SUCCESS_WINDOW_START,
    TYPE_WINDOW_FRAMES,
)

OFFSET_REFERENCE_FRAMES = 120


def benchmark_temporal_windows(
    task: str,
    dataset_path: str,
    architecture: str,
    *,
    windows: list[int] | None = None,
    epochs: int = 8,
    use_scalar_features: bool = True,
    progress_callback=None,
) -> dict:
    """Compare temporal window lengths with short validation runs."""
    import keras
    from sklearn.metrics import balanced_accuracy_score
    import numpy as np

    from core.model import model
    from core.model.training.loader import Loader

    candidates = windows or ([240, 200, 160, 120] if task == "type" else [180, 160, 140, 120])
    results: list[dict] = []
    for index, frame_count in enumerate(candidates, start=1):
        if progress_callback:
            progress_callback({"stage": "started", "index": index, "total": len(candidates), "frames": frame_count})
        _clear_keras_session(keras)
        loader = Loader(
            dataset_path,
            augment_mirror=True,
            use_scalar_features=use_scalar_features,
            type_window_frames=frame_count if task == "type" else TYPE_WINDOW_FRAMES,
            success_window_start=(
                SEGMENT_FRAMES_BEFORE_TAKEOFF + SEGMENT_FRAMES_AFTER_TAKEOFF - frame_count
                if task == "success"
                else SUCCESS_WINDOW_START
            ),
            success_window_frames=frame_count if task == "success" else SUCCESS_WINDOW_FRAMES,
        )
        dataset = loader.get_type_data() if task == "type" else loader.get_success_data()
        overrides = {"input_shape": (frame_count, 10)}
        candidate_model = model.build_model(task, architecture, **overrides)
        history = candidate_model.fit(
            {"temporal_input": dataset.temporal_features_train, "scalar_input": dataset.scalar_features_train},
            dataset.labels_train,
            validation_data=(
                {"temporal_input": dataset.temporal_features_test, "scalar_input": dataset.scalar_features_test},
                dataset.labels_test,
            ),
            epochs=epochs,
            verbose=0,
            class_weight=(dataset.class_weight or None) if task == "success" else None,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True)],
        )
        predictions = candidate_model.predict(
            {"temporal_input": dataset.temporal_features_test, "scalar_input": dataset.scalar_features_test},
            verbose=0,
        )
        true_labels = [int(np.argmax(row)) for row in dataset.labels_test]
        predicted_labels = [int(np.argmax(row)) for row in predictions]
        result = {
            "frames": int(frame_count),
            "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
            "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
            "epochs_ran": int(len(history.history.get("loss", []))),
        }
        results.append(result)
        if progress_callback:
            progress_callback({"stage": "completed", "index": index, "total": len(candidates), "result": result})
    ranked = sorted(results, key=lambda item: (item["balanced_accuracy"], item["best_val_accuracy"]), reverse=True)
    return {
        "task": task,
        "architecture": architecture,
        "results": results,
        "best": ranked[0] if ranked else None,
        "note": "Short ablation benchmark; retrain the chosen setting with the full training schedule before adopting it.",
    }


def benchmark_temporal_offsets(
    task: str,
    dataset_path: str,
    architecture: str,
    *,
    window_frames: int = TYPE_WINDOW_FRAMES,
    offsets: list[int] | None = None,
    epochs: int = 8,
    use_scalar_features: bool = True,
    auto_reexport: bool = True,
    progress_callback=None,
) -> dict:
    """Compare same-length type windows shifted inside the exported segment."""
    if task != "type":
        raise ValueError("Offset benchmark currently supports only task='type'.")

    candidates = offsets or [-60, -40, -20, 0]
    min_offset = min(candidates)
    exported_before_frames = OFFSET_REFERENCE_FRAMES - min_offset if min_offset < 0 else OFFSET_REFERENCE_FRAMES
    if min_offset < 0:
        if not auto_reexport:
            raise ValueError(
                "Negative offsets need segments re-exported with more pre-takeoff context. "
                f"Current annotated segments must begin at least {OFFSET_REFERENCE_FRAMES} frames before takeoff."
            )
        from synergie.services.segment_reexport_service import ensure_pre_takeoff_context

        def report_reexport(event: dict) -> None:
            if progress_callback:
                progress_callback(
                    {
                        "stage": "reexporting",
                        "index": event["current"],
                        "total": event["total"],
                        "reexported": event["reexported"],
                    }
                )

        ensure_pre_takeoff_context(dataset_path, OFFSET_REFERENCE_FRAMES - min_offset, progress_callback=report_reexport)

    import keras
    from sklearn.metrics import balanced_accuracy_score
    import numpy as np

    from core.model import model
    from core.model.training.loader import Loader

    results: list[dict] = []
    for index, offset in enumerate(candidates, start=1):
        if progress_callback:
            progress_callback({"stage": "started", "index": index, "total": len(candidates), "offset": offset})
        _clear_keras_session(keras)
        loader = Loader(
            dataset_path,
            augment_mirror=True,
            use_scalar_features=use_scalar_features,
            type_window_start=exported_before_frames - OFFSET_REFERENCE_FRAMES + offset,
            type_window_frames=window_frames,
        )
        dataset = loader.get_type_data()
        candidate_model = model.build_model(task, architecture, input_shape=(window_frames, 10))
        history = candidate_model.fit(
            {"temporal_input": dataset.temporal_features_train, "scalar_input": dataset.scalar_features_train},
            dataset.labels_train,
            validation_data=(
                {"temporal_input": dataset.temporal_features_test, "scalar_input": dataset.scalar_features_test},
                dataset.labels_test,
            ),
            epochs=epochs,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True)],
        )
        predictions = candidate_model.predict(
            {"temporal_input": dataset.temporal_features_test, "scalar_input": dataset.scalar_features_test},
            verbose=0,
        )
        true_labels = [int(np.argmax(row)) for row in dataset.labels_test]
        predicted_labels = [int(np.argmax(row)) for row in predictions]
        result = {
            "offset": int(offset),
            "start_relative_to_takeoff": int(offset - OFFSET_REFERENCE_FRAMES),
            "end_relative_to_takeoff": int(offset - OFFSET_REFERENCE_FRAMES + window_frames),
            "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
            "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
            "epochs_ran": int(len(history.history.get("loss", []))),
        }
        results.append(result)
        if progress_callback:
            progress_callback({"stage": "completed", "index": index, "total": len(candidates), "result": result})
    ranked = sorted(results, key=lambda item: (item["balanced_accuracy"], item["best_val_accuracy"]), reverse=True)
    return {
        "task": task,
        "architecture": architecture,
        "window_frames": int(window_frames),
        "results": results,
        "best": ranked[0] if ranked else None,
        "note": (
            "Offsets are relative to the existing exported segments. "
            "To test windows earlier than -120 frames, re-export longer segments with more pre-takeoff context."
        ),
    }


def _clear_keras_session(keras_module) -> None:
    """Clear model state while suppressing Keras' noisy TensorFlow deprecation log."""
    tensorflow_logger = logging.getLogger("tensorflow")
    previous_level = tensorflow_logger.level
    tensorflow_logger.setLevel(logging.ERROR)
    try:
        keras_module.backend.clear_session()
    finally:
        tensorflow_logger.setLevel(previous_level)
