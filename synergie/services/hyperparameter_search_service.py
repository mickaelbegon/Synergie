from __future__ import annotations

import itertools
import json
import random
from datetime import datetime
from pathlib import Path


OPTIMIZED_MODEL_PARAMETERS_FILE = Path("config") / "optimized_model_parameters.json"


def hyperparameter_search_space(task: str, architecture: str) -> list[dict]:
    """Return the bounded exploratory search space used by the GUI tuner."""
    if task == "success" and architecture == "tcn":
        return [
            {"filters": filters, "dropout": dropout, "learning_rate": learning_rate, "batch_size": batch_size}
            for filters, dropout, learning_rate, batch_size in itertools.product(
                [32, 64],
                [0.1, 0.2, 0.3],
                [0.00001, 0.00003, 0.0001],
                [16, 32],
            )
        ]
    if task == "success" and architecture == "lstm":
        return [
            {
                "first_units": first_units,
                "second_units": second_units,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            for first_units, second_units, dropout, learning_rate, batch_size in itertools.product(
                [64, 128],
                [32, 64],
                [0.3, 0.4],
                [0.00001, 0.00003],
                [16, 32],
            )
        ]
    if task == "type" and architecture == "inceptiontime":
        return [
            {
                "filters": filters,
                "bottleneck_filters": bottleneck_filters,
                "modules": modules,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            for filters, bottleneck_filters, modules, dropout, learning_rate, batch_size in itertools.product(
                [16, 32],
                [16, 32],
                [2, 3],
                [0.1, 0.2],
                [0.00001, 0.00003],
                [16, 32],
            )
        ]
    if task == "type" and architecture == "transformer":
        return [
            {
                "head_size": head_size,
                "num_heads": num_heads,
                "ff_dim": ff_dim,
                "num_transformer_blocks": blocks,
                "dropout": dropout,
                "mlp_dropout": 0.1,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            for head_size, num_heads, ff_dim, blocks, dropout, learning_rate, batch_size in itertools.product(
                [64, 128],
                [2, 4],
                [32, 64],
                [2, 4],
                [0.2, 0.3],
                [0.00001, 0.00005],
                [16, 32],
            )
        ]
    raise ValueError(f"Unsupported search combination: task={task}, architecture={architecture}")


def sample_hyperparameter_trials(
    task: str,
    architecture: str,
    max_trials: int,
    *,
    random_seed: int = 42,
) -> list[dict]:
    """Choose a reproducible subset of the bounded search space."""
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    candidates = hyperparameter_search_space(task, architecture)
    if max_trials >= len(candidates):
        return candidates
    return random.Random(random_seed).sample(candidates, k=max_trials)


def save_optimized_model_parameters(
    task: str,
    architecture: str,
    best_trial: dict,
    *,
    path: str | Path = OPTIMIZED_MODEL_PARAMETERS_FILE,
) -> Path:
    """Persist the best explored parameters for reuse by future training runs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_parameter_store(output_path)
    payload.setdefault(task, {})[architecture] = {
        "parameters": dict(best_trial["parameters"]),
        "best_val_accuracy": float(best_trial.get("best_val_accuracy", 0.0)),
        "trial": int(best_trial.get("trial", 0)),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_optimized_model_parameters(
    task: str,
    architecture: str,
    *,
    path: str | Path = OPTIMIZED_MODEL_PARAMETERS_FILE,
) -> dict | None:
    """Load persisted tuned parameters for one task/architecture pair."""
    entry = _load_parameter_store(Path(path)).get(task, {}).get(architecture)
    return dict(entry) if entry else None


def _load_parameter_store(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
