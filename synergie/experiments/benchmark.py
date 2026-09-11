from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from synergie.experiments.dataset import ClassificationDataset, load_classification_dataset


@dataclass
class BenchmarkResult:
    model_name: str
    task: str
    accuracy: float
    train_samples: int
    test_samples: int
    labels: list[int]
    report: str


def _split_dataset(
    dataset: ClassificationDataset,
    test_size: float,
    random_state: int,
):
    train_idx, test_idx = train_test_split(
        np.arange(dataset.n_samples),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=dataset.y,
    )

    return (
        dataset.X_series[train_idx],
        dataset.X_series[test_idx],
        dataset.X_scalar[train_idx],
        dataset.X_scalar[test_idx],
        dataset.y[train_idx],
        dataset.y[test_idx],
        train_idx,
        test_idx,
    )


def _augment_series_with_scalar_channels(X_series: np.ndarray, X_scalar: np.ndarray) -> np.ndarray:
    repeated_scalar = np.repeat(X_scalar[:, :, np.newaxis], X_series.shape[2], axis=2)
    return np.concatenate([X_series, repeated_scalar.astype(np.float32)], axis=1)


def run_minirocket_benchmark(
    task: str,
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BenchmarkResult:
    from aeon.transformations.collection.convolution_based import MiniRocket

    dataset = load_classification_dataset(task, dataset_path)
    (
        X_train_series,
        X_test_series,
        X_train_scalar,
        X_test_scalar,
        y_train,
        y_test,
        train_idx,
        test_idx,
    ) = _split_dataset(
        dataset,
        test_size=test_size,
        random_state=random_state,
    )
    X_train_series = _augment_series_with_scalar_channels(X_train_series, X_train_scalar)
    X_test_series = _augment_series_with_scalar_channels(X_test_series, X_test_scalar)

    transformer = MiniRocket(n_kernels=10000, random_state=random_state)
    transformer.fit(X_train_series)
    X_train_transformed = np.asarray(transformer.transform(X_train_series))
    X_test_transformed = np.asarray(transformer.transform(X_test_series))

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    classifier.fit(X_train_transformed, y_train)
    predictions = classifier.predict(X_test_transformed)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, digits=4)

    return BenchmarkResult(
        model_name="minirocket",
        task=task,
        accuracy=accuracy,
        train_samples=int(len(train_idx)),
        test_samples=int(len(test_idx)),
        labels=dataset.labels,
        report=report,
    )


def run_hydra_benchmark(
    task: str,
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BenchmarkResult:
    from aeon.classification.convolution_based import HydraClassifier

    dataset = load_classification_dataset(task, dataset_path)
    (
        X_train_series,
        X_test_series,
        X_train_scalar,
        X_test_scalar,
        y_train,
        y_test,
        train_idx,
        test_idx,
    ) = _split_dataset(
        dataset,
        test_size=test_size,
        random_state=random_state,
    )
    X_train_series = _augment_series_with_scalar_channels(X_train_series, X_train_scalar)
    X_test_series = _augment_series_with_scalar_channels(X_test_series, X_test_scalar)

    classifier = HydraClassifier(random_state=random_state)
    classifier.fit(X_train_series, y_train)
    predictions = classifier.predict(X_test_series)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, digits=4)

    return BenchmarkResult(
        model_name="hydra",
        task=task,
        accuracy=accuracy,
        train_samples=int(len(train_idx)),
        test_samples=int(len(test_idx)),
        labels=dataset.labels,
        report=report,
    )


def benchmark_summary(task: str, dataset_path: str) -> str:
    dataset = load_classification_dataset(task, dataset_path)
    return (
        f"task={task}\n"
        f"samples={dataset.n_samples}\n"
        f"labels={dataset.labels}\n"
        f"channels={dataset.n_channels}\n"
        f"sequence_length={dataset.sequence_length}"
    )
