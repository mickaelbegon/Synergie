from __future__ import annotations

import importlib

import movelladot_pc_sdk


_CANDIDATE_BINDINGS = (
    "movelladot_pc_sdk.movelladot_pc_sdk_py311_64",
    "movelladot_pc_sdk.movelladot_pc_sdk_py310_64",
    "movelladot_pc_sdk.movelladot_pc_sdk_py39_64",
)


def load_sdk_bindings():
    errors = []
    for module_name in _CANDIDATE_BINDINGS:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")

    details = "\n".join(errors)
    raise ModuleNotFoundError(
        "Unable to import Movella DOT Python bindings. "
        "Expected one of the supported SDK modules for Python 3.11, 3.10 or 3.9.\n"
        f"{details}"
    )


sdk_bindings = load_sdk_bindings()

XsDotDevice = sdk_bindings.XsDotDevice
XsDotUsbDevice = sdk_bindings.XsDotUsbDevice
XsDotConnectionManager = sdk_bindings.XsDotConnectionManager
XsDotCallback = sdk_bindings.XsDotCallback
XsPortInfo = sdk_bindings.XsPortInfo
XsDataPacket = sdk_bindings.XsDataPacket

