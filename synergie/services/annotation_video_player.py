from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoFrame:
    rgb_frame: object
    position_ms: float


class AnnotationVideoPlayer:
    """Small wrapper around OpenCV video playback for annotation review."""

    def __init__(self, path: str | Path) -> None:
        import cv2

        self.path = Path(path)
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise OSError(f"Unable to open video: {self.path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_ms = self.frame_count / self.fps * 1000.0 if self.fps > 0 and self.frame_count > 0 else 0.0
        self.current_ms = 0.0

    def release(self) -> None:
        self._capture.release()

    def frame_step_ms(self) -> float:
        return 1000.0 / self.fps if self.fps > 0 else 40.0

    def read_frame(self, milliseconds: float | None = None, *, seek: bool = True) -> VideoFrame | None:
        target_ms = self.current_ms if milliseconds is None else self._clamp_ms(float(milliseconds))
        if seek:
            self._seek(target_ms)
        ok, frame = self._capture.read()
        if not ok:
            return None
        if not seek:
            position_ms = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC) or 0.0)
            target_ms = position_ms if position_ms > 0 else self._clamp_ms(target_ms)
        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        self.current_ms = target_ms
        return VideoFrame(rgb_frame=rgb_frame, position_ms=target_ms)

    def _seek(self, target_ms: float) -> None:
        if self.fps > 0:
            frame_index = int(round((target_ms / 1000.0) * self.fps))
            frame_index = min(frame_index, max(self.frame_count - 1, 0))
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            self._capture.set(self._cv2.CAP_PROP_POS_MSEC, target_ms)

    def _clamp_ms(self, value: float) -> float:
        if self.duration_ms <= 0:
            return max(value, 0.0)
        return max(0.0, min(value, self.duration_ms))
