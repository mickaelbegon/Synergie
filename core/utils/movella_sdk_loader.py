from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


_SDK_SEARCH_ROOTS = (
    Path(r"C:\Program Files\Movella"),
    Path(r"C:\Program Files\Xsens"),
    Path(r"C:\Users\micka\Sync\Mickael\Download_2024\Synergie"),
    Path(r"C:\Users\micka\Downloads\Xsens DOT SDK_Windows"),
)

_PYTHON_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
_BINDING_CANDIDATES = (
    f"movelladot_pc_sdk.movelladot_pc_sdk_py{sys.version_info.major}{sys.version_info.minor}_64",
    "movelladot_pc_sdk.movelladot_pc_sdk_py310_64",
    "movelladot_pc_sdk.movelladot_pc_sdk_py39_64",
)


def _candidate_sdk_locations() -> list[dict]:
    locations: list[dict] = []
    seen: set[str] = set()

    for root in _SDK_SEARCH_ROOTS:
        if not root.exists():
            continue

        for wheel in root.rglob("movelladot_pc_sdk-*.whl"):
            record = {
                "kind": "wheel",
                "path": wheel,
                "dll_dir": wheel.parents[2] / "x64" / "lib" if len(wheel.parents) >= 3 else None,
            }
            normalized = str(wheel).lower()
            if normalized not in seen:
                locations.append(record)
                seen.add(normalized)

        for package_init in root.rglob("movelladot_pc_sdk/__init__.py"):
            package_root = package_init.parent.parent
            record = {
                "kind": "package",
                "path": package_root,
                "dll_dir": package_root / "movelladot_pc_sdk",
            }
            normalized = str(package_root).lower()
            if normalized not in seen:
                locations.append(record)
                seen.add(normalized)

    return locations


def _supported_python_tags(locations: list[dict]) -> list[str]:
    tags: set[str] = set()
    for location in locations:
        path = location["path"]
        if path.suffix.lower() == ".whl":
            parts = path.name.split("-")
            if len(parts) >= 3:
                tags.add(parts[2])
        else:
            package_dir = path / "movelladot_pc_sdk"
            for module in package_dir.glob("movelladot_pc_sdk_py*_64.py"):
                suffix = module.stem.replace("movelladot_pc_sdk_py", "").replace("_64", "")
                tags.add(f"cp{suffix}")
    return sorted(tags)


def _prepare_sdk_search_path() -> list[dict]:
    locations = _candidate_sdk_locations()
    for location in locations:
        path = location["path"]
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        dll_dir = location.get("dll_dir")
        if dll_dir and Path(dll_dir).exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except (AttributeError, FileNotFoundError, OSError):
                pass
    return locations


def load_sdk_bindings():
    locations = _prepare_sdk_search_path()
    errors = []
    for module_name in _BINDING_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except (ModuleNotFoundError, ImportError) as exc:
            errors.append(f"{module_name}: {exc}")

    supported_tags = _supported_python_tags(locations)
    searched_locations = "\n".join(f"- {location['path']}" for location in locations) or "- no SDK location found"
    supported_text = ", ".join(supported_tags) if supported_tags else "unknown"
    details = "\n".join(errors)
    raise ModuleNotFoundError(
        "Unable to import Movella DOT Python bindings.\n"
        f"Current Python tag: {_PYTHON_TAG}\n"
        f"SDK-supported tags found on this machine: {supported_text}\n"
        "Searched SDK locations:\n"
        f"{searched_locations}\n"
        "Import errors:\n"
        f"{details}"
    )


sdk_bindings = load_sdk_bindings()

XsDotDevice = sdk_bindings.XsDotDevice
XsDotUsbDevice = sdk_bindings.XsDotUsbDevice
XsDotConnectionManager = sdk_bindings.XsDotConnectionManager
XsDotCallback = sdk_bindings.XsDotCallback
XsPortInfo = sdk_bindings.XsPortInfo
XsDataPacket = sdk_bindings.XsDataPacket
