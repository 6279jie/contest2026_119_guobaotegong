from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameSnapshot:
    frame_id: int
    timestamp_ms: int
    frame: np.ndarray
    result: dict[str, Any]


class LatestFrameStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot: FrameSnapshot | None = None

    def update(
        self,
        frame_id: int,
        timestamp_ms: int,
        frame: np.ndarray,
        result: dict[str, Any],
    ) -> bool:
        with self._lock:
            if self._snapshot is not None and frame_id <= self._snapshot.frame_id:
                return False
            self._snapshot = FrameSnapshot(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                frame=frame.copy(),
                result=dict(result),
            )
            return True

    def snapshot(self) -> FrameSnapshot | None:
        with self._lock:
            if self._snapshot is None:
                return None
            current = self._snapshot
            return FrameSnapshot(
                frame_id=current.frame_id,
                timestamp_ms=current.timestamp_ms,
                frame=current.frame.copy(),
                result=dict(current.result),
            )
