import time

import cv2

from .frame_store import LatestFrameStore
from .overlay import draw_overlay


def prepare_display_frame(frame):
    """Flip the camera image vertically and horizontally for the mounted board."""
    return cv2.flip(frame, -1)


def render_viewer_frame(frame):
    """Return the camera content after display-only two-axis flipping."""
    return prepare_display_frame(frame)


def run_viewer(store: LatestFrameStore, window_name: str = "ESP32 Sign Edge") -> None:
    last_frame_id = -1
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        while True:
            snapshot = store.snapshot()
            if snapshot is None or snapshot.frame_id == last_frame_id:
                if cv2.waitKey(20) & 0xFF in (27, ord("q")):
                    break
                time.sleep(0.01)
                continue

            last_frame_id = snapshot.frame_id
            cv2.imshow(
                window_name,
                draw_overlay(render_viewer_frame(snapshot.frame), snapshot),
            )
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except cv2.error as exc:
        print(f"OpenCV viewer disabled: {exc}", flush=True)
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
