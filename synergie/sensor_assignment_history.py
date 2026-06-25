from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


HISTORY_FILE = Path(__file__).resolve().parents[1] / "local_state" / "sensor_assignment_history.json"


def load_history(path: Path = HISTORY_FILE) -> dict:
    if not path.is_file():
        return {"assignments": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"assignments": {}}
    if not isinstance(payload, dict):
        return {"assignments": {}}
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        payload["assignments"] = {}
    return payload


def save_history(history: dict, path: Path = HISTORY_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")


def record_assignment(sensor_id: str, skater_id: str, skater_name: str, path: Path = HISTORY_FILE) -> None:
    history = load_history(path)
    assignments = history.setdefault("assignments", {})
    sensor_assignments = assignments.setdefault(str(sensor_id), {})
    record = sensor_assignments.setdefault(
        str(skater_id),
        {"count": 0, "skater_name": skater_name, "last_used_at": ""},
    )
    record["count"] = int(record.get("count", 0)) + 1
    record["skater_name"] = skater_name
    record["last_used_at"] = datetime.now().isoformat(timespec="milliseconds")
    save_history(history, path)


def assignment_count(sensor_id: str, skater_id: str, path: Path = HISTORY_FILE) -> int:
    history = load_history(path)
    record = history.get("assignments", {}).get(str(sensor_id), {}).get(str(skater_id), {})
    try:
        return int(record.get("count", 0))
    except (TypeError, ValueError):
        return 0


def assignment_record(sensor_id: str, skater_id: str, path: Path = HISTORY_FILE) -> dict:
    history = load_history(path)
    record = history.get("assignments", {}).get(str(sensor_id), {}).get(str(skater_id), {})
    return record if isinstance(record, dict) else {}


def assignment_label(sensor_id: str, skater_id: str, path: Path = HISTORY_FILE) -> str:
    count = assignment_count(sensor_id, skater_id, path)
    if count <= 0:
        return ""
    return f"Déjà utilisé {count} fois avec ce capteur"


def sort_skaters_for_sensor(skaters: list, sensor_id: str, path: Path = HISTORY_FILE) -> list:
    indexed_skaters = list(enumerate(skaters))
    return [
        skater
        for _index, skater in sorted(
            indexed_skaters,
            key=lambda item: (
                -assignment_count(sensor_id, item[1].skater_id, path),
                -_last_used_timestamp(sensor_id, item[1].skater_id, path),
                item[0],
            ),
        )
    ]


def _last_used_timestamp(sensor_id: str, skater_id: str, path: Path = HISTORY_FILE) -> float:
    value = assignment_record(sensor_id, skater_id, path).get("last_used_at", "")
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0
