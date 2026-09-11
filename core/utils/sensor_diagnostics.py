from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UsbProbeResult:
    ok: bool
    timed_out: bool
    detected_count: int = 0
    ports: list[str] | None = None
    output: str = ""
    error: str = ""


def probe_movella_usb_detection(timeout_seconds: int = 20) -> UsbProbeResult:
    script = r"""
import json
from core.utils.xdpchandler import XdpcHandler

handler = XdpcHandler()
try:
    initialized = handler.initialize()
    if not initialized:
        print(json.dumps({"ok": False, "error": "Unable to initialize Movella DOT manager"}), flush=True)
    else:
        handler.detectUsbDevices()
        ports = []
        for device in handler.connectedUsbDots():
            port = device.portInfo()
            ports.append({
                "port": port.portName(),
                "device_id": device.deviceId().toXsString(),
                "tag_name": device.deviceTagName(),
                "bluetooth_address": device.bluetoothAddress(),
                "is_bluetooth": bool(port.isBluetooth()),
            })
        print(json.dumps({"ok": True, "detected_count": len(ports), "ports": ports}), flush=True)
finally:
    try:
        handler.cleanup()
    except Exception:
        pass
"""
    command = [sys.executable, "-u", "-c", script]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        return UsbProbeResult(
            ok=False,
            timed_out=True,
            output=(exc.stdout or ""),
            error=(exc.stderr or ""),
        )

    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return UsbProbeResult(
            ok=bool(payload.get("ok")),
            timed_out=False,
            detected_count=int(payload.get("detected_count", 0)),
            ports=[str(item.get("port", "")) for item in payload.get("ports", [])],
            output=output,
            error=str(payload.get("error") or error),
        )

    return UsbProbeResult(ok=False, timed_out=False, output=output, error=error or "No diagnostic JSON returned")
