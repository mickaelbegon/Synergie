from __future__ import annotations

from pathlib import Path


def process_csv_file(
    csv_path: str,
    synchro: int = 0,
    output_path: str | None = None,
    *,
    type_model_path: str | None = None,
    success_model_path: str | None = None,
) -> dict:
    """Process one raw IMU CSV into a jump-list CSV."""
    import pandas as pd
    from core.data_treatment.data_generation.exporter import export

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
    return {"path": destination, "prediction_status": result.attrs.get("prediction_status", "unknown")}
