from __future__ import annotations

import constants


def compute_signal_importance(
    task: str,
    dataset_path: str,
    *,
    model_path: str | None = None,
    repeats: int = 3,
    temporal_windows: int = 6,
    random_seed: int = 42,
) -> dict:
    """
    Estimate temporal-channel, scalar, and time-window importance by permutation.

    The metric is balanced accuracy so the success task is not dominated by its
    majority class.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from core.model import model
    from core.model.training.loader import Loader

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if temporal_windows < 1:
        raise ValueError("temporal_windows must be at least 1")

    loader = Loader(dataset_path, augment_mirror=False)
    dataset = loader.get_type_data() if task == "type" else loader.get_success_data()
    temporal = np.asarray(dataset.temporal_features_test)
    scalar = np.asarray(dataset.scalar_features_test)
    labels = np.asarray(dataset.labels_test)
    if len(temporal) < 2:
        raise ValueError("At least two validation samples are required for permutation importance.")

    selected_model_path = model_path or latest_model_path_for_task(task)
    predictor = model.load_model(selected_model_path, for_training=False)
    true_labels = np.argmax(labels, axis=1)
    baseline_predictions = _predict_class_labels(predictor, temporal, scalar)
    baseline_accuracy = float(accuracy_score(true_labels, baseline_predictions))
    baseline_balanced_accuracy = float(balanced_accuracy_score(true_labels, baseline_predictions))
    rng = np.random.default_rng(random_seed)

    channel_importance = _compute_channel_importance(
        predictor,
        temporal,
        scalar,
        true_labels,
        baseline_balanced_accuracy,
        repeats,
        rng,
    )
    scalar_importance = _compute_scalar_importance(
        predictor,
        temporal,
        scalar,
        true_labels,
        baseline_balanced_accuracy,
        repeats,
        rng,
    )
    temporal_importance = _compute_temporal_importance(
        predictor,
        temporal,
        scalar,
        true_labels,
        baseline_balanced_accuracy,
        repeats,
        temporal_windows,
        rng,
    )
    channel_importance.sort(key=lambda item: item["mean_drop"], reverse=True)
    scalar_importance.sort(key=lambda item: item["mean_drop"], reverse=True)
    return {
        "task": task,
        "dataset_path": dataset_path,
        "model_path": selected_model_path,
        "validation_samples": int(len(temporal)),
        "baseline_accuracy": baseline_accuracy,
        "baseline_balanced_accuracy": baseline_balanced_accuracy,
        "channel_importance": channel_importance,
        "scalar_importance": scalar_importance,
        "temporal_importance": temporal_importance,
        "method": "grouped_permutation_balanced_accuracy",
    }


def latest_model_path_for_task(task: str) -> str:
    """Return the active inference model alias for one task."""
    if task == "type":
        return constants.modeltype_filepath
    if task == "success":
        return constants.modelsuccess_filepath
    raise ValueError(f"Unsupported training task: {task}")


def _compute_channel_importance(predictor, temporal, scalar, true_labels, baseline_score, repeats, rng) -> list[dict]:
    from sklearn.metrics import balanced_accuracy_score

    importance = []
    for channel_index, channel_name in enumerate(constants.fields_to_keep):
        drops = []
        for _ in range(repeats):
            permuted = temporal.copy()
            permuted[:, :, channel_index] = permuted[rng.permutation(len(permuted)), :, channel_index]
            score = balanced_accuracy_score(true_labels, _predict_class_labels(predictor, permuted, scalar))
            drops.append(float(baseline_score - score))
        importance.append(_summarize_importance(channel_name, drops))
    return importance


def _compute_scalar_importance(predictor, temporal, scalar, true_labels, baseline_score, repeats, rng) -> list[dict]:
    from sklearn.metrics import balanced_accuracy_score

    importance = []
    for scalar_index, scalar_name in enumerate(("weight", "height")):
        drops = []
        for _ in range(repeats):
            permuted_scalar = scalar.copy()
            permuted_scalar[:, scalar_index] = permuted_scalar[rng.permutation(len(permuted_scalar)), scalar_index]
            score = balanced_accuracy_score(true_labels, _predict_class_labels(predictor, temporal, permuted_scalar))
            drops.append(float(baseline_score - score))
        importance.append(_summarize_importance(scalar_name, drops))
    return importance


def _compute_temporal_importance(
    predictor,
    temporal,
    scalar,
    true_labels,
    baseline_score,
    repeats,
    temporal_windows,
    rng,
) -> list[dict]:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score

    importance = []
    for window_index, indices in enumerate(np.array_split(np.arange(temporal.shape[1]), temporal_windows), start=1):
        if len(indices) == 0:
            continue
        drops = []
        for _ in range(repeats):
            permuted = temporal.copy()
            permuted[:, indices, :] = permuted[rng.permutation(len(permuted))][:, indices, :]
            score = balanced_accuracy_score(true_labels, _predict_class_labels(predictor, permuted, scalar))
            drops.append(float(baseline_score - score))
        importance.append(
            {
                **_summarize_importance(f"window_{window_index}", drops),
                "start_frame": int(indices[0]),
                "end_frame": int(indices[-1]),
            }
        )
    return importance


def _predict_class_labels(predictor, temporal, scalar):
    import numpy as np

    predictions = predictor.predict(
        {"temporal_input": temporal, "scalar_input": scalar},
        verbose=0,
    )
    return np.argmax(predictions, axis=1)


def _summarize_importance(label: str, drops: list[float]) -> dict:
    import numpy as np

    return {
        "label": label,
        "mean_drop": float(np.mean(drops)),
        "std_drop": float(np.std(drops)),
    }
