from __future__ import annotations

import importlib
import importlib.util
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


def _candidate_sdk_locations() -> tuple[list[dict], list[Path]]:
    importable_locations: list[dict] = []
    diagnostic_wheels: list[Path] = []
    seen: set[str] = set()

    for root in _SDK_SEARCH_ROOTS:
        if not root.exists():
            continue

        for wheel in root.rglob("movelladot_pc_sdk-*.whl"):
            normalized = str(wheel).lower()
            if normalized not in seen:
                diagnostic_wheels.append(wheel)
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
                importable_locations.append(record)
                seen.add(normalized)

    return importable_locations, diagnostic_wheels


def _supported_python_tags(importable_locations: list[dict], diagnostic_wheels: list[Path]) -> list[str]:
    tags: set[str] = set()
    for wheel in diagnostic_wheels:
        parts = wheel.name.split("-")
        if len(parts) >= 3:
            tags.add(parts[2])
    for location in importable_locations:
        package_dir = location["path"] / "movelladot_pc_sdk"
        for module in package_dir.glob("movelladot_pc_sdk_py*_64.py"):
            suffix = module.stem.replace("movelladot_pc_sdk_py", "").replace("_64", "")
            tags.add(f"cp{suffix}")
    return sorted(tags)


def _prepare_sdk_search_path() -> tuple[list[dict], list[Path]]:
    locations, wheels = _candidate_sdk_locations()
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
    return locations, wheels


def load_sdk_bindings():
    errors = []
    installed_spec = importlib.util.find_spec("movelladot_pc_sdk")
    if installed_spec is not None:
        for module_name in _BINDING_CANDIDATES:
            try:
                return importlib.import_module(module_name)
            except (ModuleNotFoundError, ImportError) as exc:
                errors.append(f"{module_name}: {exc}")

    locations, wheels = _prepare_sdk_search_path()
    for module_name in _BINDING_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except (ModuleNotFoundError, ImportError) as exc:
            if installed_spec is None or f"{module_name}: {exc}" not in errors:
                errors.append(f"{module_name}: {exc}")

    supported_tags = _supported_python_tags(locations, wheels)
    searched_entries = []
    if installed_spec is not None:
        installed_origin = installed_spec.origin or "installed package"
        searched_entries.append(f"- installed package: {installed_origin}")
    searched_entries.extend(f"- {location['path']}" for location in locations)
    searched_entries.extend(f"- {wheel}" for wheel in wheels)
    searched_locations = "\n".join(searched_entries) or "- no SDK location found"
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
