from __future__ import annotations


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
        keras.backend.clear_session()
        loader = Loader(
            dataset_path,
            augment_mirror=True,
            use_scalar_features=use_scalar_features,
            type_window_frames=frame_count if task == "type" else 240,
            success_window_start=300 - frame_count if task == "success" else 120,
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
