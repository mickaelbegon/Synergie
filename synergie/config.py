from __future__ import annotations

from dataclasses import dataclass

import constants


TYPE_WINDOW_FRAMES = 240
SEGMENT_FRAMES_BEFORE_TAKEOFF = 120
SEGMENT_FRAMES_AFTER_TAKEOFF = 180
SUCCESS_WINDOW_FRAMES = 140
SUCCESS_WINDOW_START = SEGMENT_FRAMES_BEFORE_TAKEOFF + (SEGMENT_FRAMES_AFTER_TAKEOFF - SUCCESS_WINDOW_FRAMES)
JUMP_WINDOW_FRAMES = 300

DEFAULT_DETECTION_THRESHOLD = constants.treshold
DEFAULT_SMOOTHING_SIGMA = 30
DEFAULT_COMBINATION_GAP_FRAMES = 180
GYRO_SATURATION_WARNING_THRESHOLD = 1900.0

ROTATION_MIRROR_COLUMNS = ("Euler_X", "Gyr_X")


@dataclass(frozen=True)
class DetectionConfig:
    threshold: float = DEFAULT_DETECTION_THRESHOLD
    smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA
    combination_gap_frames: int = DEFAULT_COMBINATION_GAP_FRAMES


@dataclass(frozen=True)
class TrainingConfig:
    train_ratio: float = 0.8
    augment_mirror: bool = True
