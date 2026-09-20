import struct
import sys
import unittest
from io import BytesIO
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_edge.stream_protocol import iter_stream_frames


class ShortReadStream:
    def __init__(self, payload: bytes):
        self._stream = BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(1 if size != 0 else 0)


class StreamProtocolTests(unittest.TestCase):
    def test_iterates_length_prefixed_frames(self):
        body = b"".join(
            struct.pack("!QQI", frame_id, 1000 + frame_id, len(payload))
            + payload
            for frame_id, payload in ((8, b"jpeg-a"), (9, b"jpeg-b"))
        )

        self.assertEqual(
            list(iter_stream_frames(BytesIO(body))),
            [(8, 1008, b"jpeg-a"), (9, 1009, b"jpeg-b")],
        )

    def test_handles_short_reads_between_stream_chunks(self):
        body = b"".join(
            struct.pack("!QQI", frame_id, 1000 + frame_id, len(payload))
            + payload
            for frame_id, payload in ((8, b"jpeg-a"), (9, b"jpeg-b"))
        )

        self.assertEqual(
            list(iter_stream_frames(ShortReadStream(body))),
            [(8, 1008, b"jpeg-a"), (9, 1009, b"jpeg-b")],
        )


if __name__ == "__main__":
    unittest.main()
