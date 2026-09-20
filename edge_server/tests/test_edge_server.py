import sys
import struct
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_edge.app import create_app
from sign_edge.frame_store import LatestFrameStore
from sign_edge.overlay import draw_overlay, load_overlay_font
from sign_edge.protocol import ProtocolError, parse_non_negative_int
from sign_edge.processor import RichModelProcessor
from sign_edge.viewer import render_viewer_frame


class ProtocolTests(unittest.TestCase):
    def test_parse_non_negative_integer(self):
        self.assertEqual(parse_non_negative_int("12", "X-Frame-Id"), 12)

    def test_rejects_negative_integer(self):
        with self.assertRaises(ProtocolError):
            parse_non_negative_int("-1", "X-Frame-Id")

    def test_rejects_missing_integer(self):
        with self.assertRaises(ProtocolError):
            parse_non_negative_int(None, "X-Frame-Id")


class FrameStoreTests(unittest.TestCase):
    def test_only_newer_frame_replaces_snapshot(self):
        store = LatestFrameStore()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        result = {"status": "waiting_model"}

        self.assertTrue(store.update(2, 100, frame, result))
        self.assertFalse(store.update(2, 101, frame, result))
        self.assertFalse(store.update(1, 102, frame, result))
        self.assertTrue(store.update(3, 103, frame, result))
        self.assertEqual(store.snapshot().frame_id, 3)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.store = LatestFrameStore()
        self.client = create_app(store=self.store, gui_enabled=False).test_client()
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 1] = 180
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    def _stream_body(self, *frame_ids):
        return b"".join(
            struct.pack("!QQI", frame_id, 1000 + frame_id, len(self.jpeg))
            + self.jpeg
            for frame_id in frame_ids
        )

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
        self.assertIsNone(response.get_json()["latest_frame_id"])

    def test_viewer_page_is_available(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("ESP32-S3-EYE", page)
        self.assertIn("scale(-1, -1)", page)
        self.assertIn("board-event", page)
        self.assertIn("板端回传", page)

    def test_snapshot_returns_latest_jpeg(self):
        self.store.update(7, 1234, cv2.imdecode(
            np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        ), {"status": "waiting_model"})

        response = self.client.get("/snapshot")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        decoded = cv2.imdecode(
            np.frombuffer(response.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertIsNotNone(decoded)

    def test_snapshot_returns_no_content_before_first_frame(self):
        response = self.client.get("/snapshot")
        self.assertEqual(response.status_code, 503)

    def test_accepts_jpeg_and_returns_synchronized_result(self):
        response = self.client.post(
            "/api/v1/frame",
            data=self.jpeg,
            content_type="image/jpeg",
            headers={"X-Frame-Id": "7", "X-Timestamp-Ms": "1234"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["frame_id"], 7)
        self.assertEqual(payload["status"], "waiting_model")
        self.assertEqual(payload["label"], "")
        self.assertEqual(payload["confidence"], 0.0)
        self.assertEqual(self.store.snapshot().frame_id, 7)

    def test_rejects_corrupt_jpeg(self):
        response = self.client.post(
            "/api/v1/frame",
            data=b"not-a-jpeg",
            content_type="image/jpeg",
            headers={"X-Frame-Id": "1", "X-Timestamp-Ms": "10"},
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_missing_headers(self):
        response = self.client.post(
            "/api/v1/frame", data=self.jpeg, content_type="image/jpeg"
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_duplicate_frame(self):
        headers = {"X-Frame-Id": "4", "X-Timestamp-Ms": "10"}
        self.assertEqual(
            self.client.post(
                "/api/v1/frame",
                data=self.jpeg,
                content_type="image/jpeg",
                headers=headers,
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/frame",
                data=self.jpeg,
                content_type="image/jpeg",
                headers=headers,
            ).status_code,
            409,
        )

    def test_accepts_multiple_frames_on_stream_endpoint(self):
        response = self.client.post(
            "/api/v1/stream",
            data=self._stream_body(8, 9, 10),
            content_type="application/x-esp32-jpeg-stream",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["frames_received"], 3)
        self.assertEqual(self.store.snapshot().frame_id, 10)


class OverlayTests(unittest.TestCase):
    def test_overlay_font_contains_chinese_glyphs(self):
        font = load_overlay_font()

        self.assertIsNotNone(font)
        self.assertGreater(font.getmask("你").getbbox()[2], 0)

    def test_overlay_returns_copy(self):
        store = LatestFrameStore()
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        store.update(9, 20, frame, {"status": "waiting_model"})
        snapshot = store.snapshot()

        rendered = draw_overlay(frame, snapshot)

        self.assertFalse(np.array_equal(rendered, frame))
        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))

    def test_overlay_status_panel_is_translucent_and_does_not_cover_full_width(self):
        frame = np.full((200, 320, 3), 100, dtype=np.uint8)
        store = LatestFrameStore()
        store.update(9, 20, frame, {"status": "waiting_model"})

        rendered = draw_overlay(frame, store.snapshot())

        panel_pixel = rendered[10, 100]
        self.assertTrue(np.all(panel_pixel > 0))
        self.assertTrue(np.all(panel_pixel < 100))
        self.assertTrue(np.array_equal(rendered[10, 300], frame[10, 300]))


class ViewerTests(unittest.TestCase):
    def test_render_viewer_frame_only_flips_both_axes(self):
        frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

        displayed = render_viewer_frame(frame)

        expected = np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(displayed, expected))
        self.assertTrue(np.array_equal(frame, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)))


class RichModelProcessorTests(unittest.TestCase):
    def test_runtime_defaults_are_relative_to_the_server_package(self):
        package_root = Path(__file__).resolve().parents[1]

        self.assertEqual(
            RichModelProcessor.DEFAULT_FEATURE_SOURCE,
            package_root / "model_runtime",
        )

    def test_loads_25_class_rich_checkpoint(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "sign_rich_frames_25_final.pth"
        )

        processor = RichModelProcessor(model_path)

        self.assertEqual(processor.feature_dim, 380)
        self.assertEqual(processor.num_classes, 25)
        self.assertEqual(len(processor.class_names), 25)
        self.assertEqual(
            set(processor.pair_models),
            {"have_want", "good_thanks"},
        )
        self.assertEqual(processor.pair_models["have_want"].class_ids, (19, 22))
        self.assertEqual(
            processor.pair_models["good_thanks"].class_ids,
            (12, 24),
        )


if __name__ == "__main__":
    unittest.main()
