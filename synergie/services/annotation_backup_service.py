from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4


FINALIZATION_BACKUP_FORMAT = "synergie-finalization-backup-v1"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without changing the source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_finalization_backup(
    *,
    backup_root: str | Path,
    session_key: str,
    protected_files: Iterable[tuple[str, str | Path]],
    tracked_files: Iterable[tuple[str, str | Path]] = (),
    planned_moves: Iterable[tuple[str | Path, str | Path]] = (),
    context: dict | None = None,
    created_at: datetime | None = None,
) -> dict:
    """
    Copy mutable metadata into a unique recovery directory and write a manifest.

    ``protected_files`` are copied byte-for-byte. Potentially large segment/raw
    files can be passed as ``tracked_files``: their size and digest are recorded
    so a moved file can be verified without duplicating the payload.
    """
    generated = created_at or datetime.now(timezone.utc)
    transaction_id = f"{generated.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
    transaction_dir = Path(backup_root) / f"{_safe_name(session_key)}-{transaction_id}"
    transaction_dir.mkdir(parents=True, exist_ok=False)
    files_dir = transaction_dir / "files"
    files_dir.mkdir()

    protected_records: list[dict] = []
    try:
        for index, (role, original_value) in enumerate(protected_files):
            original = Path(original_value)
            if not original.is_file():
                raise FileNotFoundError(f"Unable to back up {role}: {original}")
            backup_path = files_dir / f"{index:03d}-{_safe_name(role)}-{original.name}"
            shutil.copy2(original, backup_path)
            original_digest = sha256_file(original)
            backup_digest = sha256_file(backup_path)
            if original_digest != backup_digest:
                raise OSError(f"Backup verification failed for {original}")
            protected_records.append(
                {
                    "role": str(role),
                    "original_path": _posix(original),
                    "backup_path": _posix(backup_path),
                    "size_bytes": int(original.stat().st_size),
                    "sha256": original_digest,
                }
            )

        tracked_records = [_file_record(role, Path(path)) for role, path in tracked_files]
        move_records = [
            {"source": _posix(source), "destination": _posix(destination)}
            for source, destination in planned_moves
        ]
        manifest = {
            "format": FINALIZATION_BACKUP_FORMAT,
            "transaction_id": transaction_id,
            "created_at": generated.isoformat(),
            "updated_at": generated.isoformat(),
            "status": "prepared",
            "session_key": str(session_key),
            "transaction_dir": _posix(transaction_dir),
            "protected_files": protected_records,
            "tracked_files": tracked_records,
            "planned_moves": move_records,
            "context": dict(context or {}),
        }
        manifest_path = transaction_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest)
    except Exception:
        # No project data has been mutated at this point. Keep a partial backup
        # directory for diagnosis instead of deleting potentially useful copies.
        raise

    return {
        "transaction_id": transaction_id,
        "transaction_dir": transaction_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def update_finalization_backup(
    manifest_path: str | Path,
    *,
    status: str,
    error: str | None = None,
    rollback_errors: Iterable[str] = (),
    outputs: dict | None = None,
) -> dict:
    """Atomically update the durable transaction status in a backup manifest."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != FINALIZATION_BACKUP_FORMAT:
        raise ValueError(f"Unsupported finalization backup manifest: {path}")
    manifest["status"] = str(status)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        manifest["error"] = str(error)
    errors = [str(item) for item in rollback_errors]
    if errors:
        manifest["rollback_errors"] = errors
    if outputs is not None:
        manifest["outputs"] = outputs
    _atomic_write_json(path, manifest)
    return manifest


def protected_backup_path(backup: dict, original_path: str | Path) -> Path | None:
    """Find the recovery copy corresponding to one original path."""
    target = _path_key(Path(original_path))
    for record in backup["manifest"].get("protected_files", []):
        if _path_key(Path(record["original_path"])) == target:
            return Path(record["backup_path"])
    return None


def _file_record(role: str, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Unable to track {role}: {path}")
    return {
        "role": str(role),
        "path": _posix(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value))
    return cleaned.strip("._") or "session"


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.absolute()))


def _posix(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")
