from __future__ import annotations

import re


BLUETOOTH_ADDRESS_PATTERN = re.compile(r"([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}")


def is_valid_bluetooth_address(address: str | None) -> bool:
    if not address:
        return False
    return bool(BLUETOOTH_ADDRESS_PATTERN.fullmatch(str(address).strip()))


def normalize_sample_time_fine(values) -> list[int]:
    sample_times = list(values)
    if not sample_times:
        return []

    start_sample_time = int(sample_times[0])
    normalized: list[int] = []
    for raw_value in sample_times:
        new_value = int(raw_value) - start_sample_time
        if new_value < 0:
            normalized.append(new_value + 2**32)
        else:
            normalized.append(new_value)
    return normalized


def join_sensor_names(sensor_names: list[str]) -> str:
    filtered = [str(sensor_name).strip() for sensor_name in sensor_names if str(sensor_name).strip()]
    return ", ".join(filtered)
