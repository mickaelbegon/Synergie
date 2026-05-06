from __future__ import annotations

from pathlib import Path

import constants


def list_sessions() -> list[str]:
    return sorted(constants.sessions)


def session_metadata(session_name: str) -> dict:
    return constants.get_session(session_name)


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


def parse_training_id_from_csv_path(csv_path: Path) -> str | None:
    training_parts = csv_path.stem.split("_")
    if len(training_parts) < 2:
        return None
    return training_parts[1]


def train_model(task: str, dataset_path: str, epochs: int | None = None, architecture: str | None = None) -> None:
    from core.model import model
    from core.model.training.loader import Loader
    from core.model.training.training import Trainer

    dataset = Loader(dataset_path, augment_mirror=True)
    selected_architecture = architecture or model.default_architecture(task)
    if task == "type":
        trainer = Trainer(
            dataset.get_type_data(),
            model.build_model(task, selected_architecture),
            constants.modeltype_filepath,
        )
        trainer.train(epochs=epochs or 10)
        return

    if task == "success":
        trainer = Trainer(
            dataset.get_success_data(),
            model.build_model(task, selected_architecture),
            constants.modelsuccess_filepath,
        )
        trainer.train_success(epochs=epochs or 20)
        return

    raise ValueError(f"Unsupported training task: {task}")


def process_csv_file(csv_path: str, synchro: int = 0, output_path: str | None = None) -> Path:
    from core.data_treatment.data_generation.exporter import export
    import pandas as pd

    input_path = Path(csv_path)
    dataframe = pd.read_csv(input_path)
    result = export(dataframe, sampleTimeFineSynchro=synchro)
    destination = Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_jumps.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return destination


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
