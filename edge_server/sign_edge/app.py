import json
import queue
import socket
import threading
import time
import urllib.request
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from .frame_store import LatestFrameStore
from .processor import WaitingModelProcessor
from .protocol import ProtocolError, parse_non_negative_int
from .stream_protocol import StreamProtocolError, iter_stream_frames


VIEWER_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-S3-EYE</title>
  <style>
    body { margin: 0; min-height: 100vh; display: grid; place-items: center;
           background: #f3f5f7; font-family: sans-serif; }
    main { background: white; padding: 22px; border-radius: 14px;
           box-shadow: 0 8px 28px #0002; text-align: center; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    #status { margin: 0 0 12px; color: #555; }
    #board-event { margin: 12px 0 0; min-height: 1.4em; color: #176b3a;
                   font-weight: 600; }
    #camera { display: block; width: 320px; height: 240px;
              object-fit: contain; background: #111;
              transform: scale(-1, -1); }
  </style>
</head>
<body>
  <main>
    <h1>ESP32-S3-EYE</h1>
    <p id="status">正在连接视频流…</p>
    <img id="camera" alt="camera stream">
    <p id="board-event" aria-live="polite">板端回传：暂无</p>
  </main>
  <script>
    const image = document.getElementById('camera');
    const status = document.getElementById('status');
    const boardEvent = document.getElementById('board-event');
    async function refresh() {
      try {
        const response = await fetch('/health?ts=' + Date.now(),
                                     {cache: 'no-store'});
        const health = await response.json();
        if (health.latest_frame_id === null) {
          status.textContent = '等待摄像头画面…';
        } else {
          status.textContent = '视频流已连接';
          image.src = '/snapshot?ts=' + Date.now();
        }

        const eventResponse = await fetch('/api/v1/board-event?ts=' + Date.now(),
                                          {cache: 'no-store'});
        const eventData = await eventResponse.json();
        if (eventData.event === 'help_confirmed') {
          boardEvent.textContent = '板端回传：已确认，需要帮助';
        } else if (eventData.event) {
          boardEvent.textContent = '板端回传：' + eventData.event;
        } else {
          boardEvent.textContent = '板端回传：暂无';
        }
      } catch (error) {
        status.textContent = '视频连接中断，正在重连…';
        boardEvent.textContent = '板端回传：连接中…';
      }
    }
    refresh();
    setInterval(refresh, 150);
  </script>
</body>
</html>"""


class LatestBoardTextSender:
    """Send only changed short labels without blocking frame inference."""

    RESEND_INTERVAL_SECONDS = 1.0

    LABELS = {
        "不": "不",
        "不是": "不是",
        "不知道": "不知道",
        "不行": "不行",
        "什么or哪里": "哪里",
        "他": "他",
        "你": "你",
        "去": "去",
        "可以": "可以",
        "有": "有",
        "要": "要",
        "吃饭": "吃饭",
        "喝水": "喝水",
        "帮助": "帮助",
        "帮我": "帮我",
        "回": "回",
        "家": "家",
        "想": "想",
        "我": "我",
        "是": "是",
        "来": "来",
        "没有": "没有",
        "请": "请",
        "谢谢": "谢谢",
    }

    def __init__(self, url: str | None) -> None:
        self.url = url
        self._pending: queue.Queue[str] = queue.Queue(maxsize=1)
        self._last_label = ""
        self._last_sent_at = 0.0
        if url:
            threading.Thread(target=self._run, daemon=True).start()

    def submit(self, label: str, status: str) -> None:
        if not self.url or status != "model_ready":
            return
        text = self.LABELS.get(label)
        now = time.monotonic()
        if text is None or (
            text == self._last_label and
            now - self._last_sent_at < self.RESEND_INTERVAL_SECONDS
        ):
            return
        self._last_label = text
        self._last_sent_at = now
        try:
            self._pending.put_nowait(text)
        except queue.Full:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pending.put_nowait(text)
            except queue.Full:
                pass

    def _run(self) -> None:
        while True:
            text = self._pending.get()
            try:
                target = urlparse(self.url)
                if target.scheme == "udp":
                    port = target.port
                    if not target.hostname or port is None:
                        raise ValueError("UDP board URL must include host and port")
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
                        client.sendto(text.encode("utf-8"), (target.hostname, port))
                else:
                    payload = json.dumps({"text": text}).encode("utf-8")
                    request = urllib.request.Request(
                        self.url,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=1.0):
                        pass
            except Exception:
                pass


class BoardEventReceiver:
    """Receive short non-blocking events sent back by the board."""

    EVENT_TTL_SECONDS = 3.0

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 28790,
        ttl_seconds: float = EVENT_TTL_SECONDS,
    ) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.settimeout(0.2)
        self.port = self._socket.getsockname()[1]
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: str | None = None
        self._latest_received_at = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload, _ = self._socket.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                return

            event = payload.decode("utf-8", errors="replace").strip()
            if not event:
                continue
            with self._lock:
                self._latest = event
                self._latest_received_at = time.monotonic()
            print(f"board_event: {event}", flush=True)

    def latest_event(self) -> str | None:
        with self._lock:
            if (
                self._latest is None
                or time.monotonic() - self._latest_received_at >= self._ttl_seconds
            ):
                return None
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._socket.close()


def create_app(
    store: LatestFrameStore | None = None,
    processor: WaitingModelProcessor | None = None,
    gui_enabled: bool = False,
    board_text_url: str | None = None,
) -> Flask:
    app = Flask(__name__)
    frame_store = store if store is not None else LatestFrameStore()
    frame_processor = processor if processor is not None else WaitingModelProcessor()
    app.config["FRAME_STORE"] = frame_store
    app.config["GUI_ENABLED"] = gui_enabled
    board_sender = LatestBoardTextSender(board_text_url)
    board_event_receiver = None
    if board_text_url:
        try:
            board_event_receiver = BoardEventReceiver()
            board_event_receiver.start()
        except OSError as exc:
            print(f"board_event: receiver unavailable: {exc}", flush=True)
    app.config["BOARD_EVENT_RECEIVER"] = board_event_receiver

    @app.get("/")
    def viewer():
        return Response(VIEWER_HTML, mimetype="text/html")

    @app.get("/snapshot")
    def snapshot():
        current = frame_store.snapshot()
        if current is None:
            return jsonify(error="no frame received yet"), 503

        ok, encoded = cv2.imencode(
            ".jpg", current.frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            return jsonify(error="failed to encode latest frame"), 500

        return Response(
            encoded.tobytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    def health():
        snapshot = frame_store.snapshot()
        return jsonify(
            status="ok",
            latest_frame_id=None if snapshot is None else snapshot.frame_id,
            gui_enabled=bool(app.config["GUI_ENABLED"]),
        )

    @app.get("/api/v1/board-event")
    def board_event():
        receiver = app.config["BOARD_EVENT_RECEIVER"]
        event = None if receiver is None else receiver.latest_event()
        return jsonify(event=event)

    @app.post("/api/v1/frame")
    def receive_frame():
        started = time.perf_counter()
        try:
            frame_id = parse_non_negative_int(
                request.headers.get("X-Frame-Id"), "X-Frame-Id"
            )
            timestamp_ms = parse_non_negative_int(
                request.headers.get("X-Timestamp-Ms"), "X-Timestamp-Ms"
            )
        except ProtocolError as exc:
            return jsonify(error=str(exc)), 400

        if request.mimetype != "image/jpeg" or not request.data:
            return jsonify(error="request body must be a non-empty image/jpeg"), 400

        encoded = np.frombuffer(request.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(error="invalid JPEG payload"), 400

        result = frame_processor.process(frame, frame_id).as_dict()
        board_sender.submit(result["label"], result["status"])
        server_timestamp_ms = int(time.time() * 1000)
        processing_ms = round((time.perf_counter() - started) * 1000, 3)
        response = {
            "frame_id": frame_id,
            **result,
            "server_timestamp_ms": server_timestamp_ms,
            "processing_ms": processing_ms,
        }
        if not frame_store.update(frame_id, timestamp_ms, frame, response):
            return jsonify(error="stale or duplicate frame", frame_id=frame_id), 409
        return jsonify(response)

    @app.post("/api/v1/stream")
    def receive_stream():
        if request.mimetype != "application/x-esp32-jpeg-stream":
            return jsonify(error="request body must be an ESP32 JPEG stream"), 400

        frames_received = 0
        try:
            for frame_id, timestamp_ms, jpeg_payload in iter_stream_frames(
                request.stream
            ):
                frame_started = time.perf_counter()
                encoded = np.frombuffer(jpeg_payload, dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify(error="invalid JPEG payload"), 400

                result = frame_processor.process(frame, frame_id).as_dict()
                board_sender.submit(result["label"], result["status"])
                server_timestamp_ms = int(time.time() * 1000)
                response = {
                    "frame_id": frame_id,
                    **result,
                    "server_timestamp_ms": server_timestamp_ms,
                    "processing_ms": round(
                        (time.perf_counter() - frame_started) * 1000, 3
                    ),
                }
                if not frame_store.update(
                    frame_id, timestamp_ms, frame, response
                ):
                    return (
                        jsonify(
                            error="stale or duplicate frame", frame_id=frame_id
                        ),
                        409,
                    )
                frames_received += 1
        except StreamProtocolError as exc:
            return jsonify(error=str(exc)), 400

        if frames_received == 0:
            return jsonify(error="stream contained no frames"), 400

        snapshot = frame_store.snapshot()
        return jsonify(
            status="ok",
            frames_received=frames_received,
            latest_frame_id=None if snapshot is None else snapshot.frame_id,
        )

    return app
