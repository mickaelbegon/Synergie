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


def _build_feature_matrix(dataset: ClassificationDataset, transformed_series: np.ndarray) -> np.ndarray:
    return np.hstack([transformed_series, dataset.X_scalar[: transformed_series.shape[0]]])


def run_minirocket_benchmark(
    task: str,
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BenchmarkResult:
    from aeon.transformations.collection.convolution_based import MiniRocket

    dataset = load_classification_dataset(task, dataset_path)
    train_idx, test_idx = train_test_split(
        np.arange(dataset.n_samples),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=dataset.y,
    )

    X_train_series = dataset.X_series[train_idx]
    X_test_series = dataset.X_series[test_idx]
    X_train_scalar = dataset.X_scalar[train_idx]
    X_test_scalar = dataset.X_scalar[test_idx]
    y_train = dataset.y[train_idx]
    y_test = dataset.y[test_idx]

    transformer = MiniRocket(n_kernels=10000, random_state=random_state)
    transformer.fit(X_train_series)
    X_train_transformed = np.asarray(transformer.transform(X_train_series))
    X_test_transformed = np.asarray(transformer.transform(X_test_series))

    X_train = np.hstack([X_train_transformed, X_train_scalar])
    X_test = np.hstack([X_test_transformed, X_test_scalar])

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
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


def benchmark_summary(task: str, dataset_path: str) -> str:
    dataset = load_classification_dataset(task, dataset_path)
    return (
        f"task={task}\n"
        f"samples={dataset.n_samples}\n"
        f"labels={dataset.labels}\n"
        f"channels={dataset.n_channels}\n"
        f"sequence_length={dataset.sequence_length}"
    )
