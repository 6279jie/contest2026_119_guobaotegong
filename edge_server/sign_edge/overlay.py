import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .frame_store import FrameSnapshot


OVERLAY_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)


def load_overlay_font(size: int = 15):
    for font_path in OVERLAY_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return None


def draw_overlay(frame: np.ndarray, snapshot: FrameSnapshot) -> np.ndarray:
    rendered = frame.copy()
    result = snapshot.result
    lines = [
        f"frame: {snapshot.frame_id}",
        f"status: {result.get('status', 'unknown')}",
        f"label: {result.get('label') or '-'}",
        f"confidence: {float(result.get('confidence', 0.0)):.3f}",
        f"processing: {float(result.get('processing_ms', 0.0)):.1f} ms",
    ]
    panel_width = min(rendered.shape[1], 280)
    panel_height = min(rendered.shape[0], 18 + 24 * len(lines))
    panel = rendered.copy()
    cv2.rectangle(
        panel,
        (0, 0),
        (panel_width - 1, panel_height - 1),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(panel, 0.45, rendered, 0.55, 0, rendered)
    font = load_overlay_font()
    if font is not None:
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            draw.text(
                (8, 3 + index * 24),
                line,
                font=font,
                fill=(60, 255, 60),
            )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    for index, line in enumerate(lines):
        cv2.putText(
            rendered,
            line,
            (8, 22 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 255, 60),
            1,
            cv2.LINE_AA,
        )
    return rendered
