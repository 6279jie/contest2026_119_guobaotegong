import struct
from typing import BinaryIO, Iterator


FRAME_HEADER = struct.Struct("!QQI")
MAX_FRAME_BYTES = 512 * 1024


class StreamProtocolError(ValueError):
    """Raised when a length-prefixed JPEG stream is malformed."""


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    if len(chunks) != size:
        raise StreamProtocolError("truncated stream frame")
    return bytes(chunks)


def iter_stream_frames(stream: BinaryIO) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(frame_id, timestamp_ms, jpeg)`` records until EOF."""

    while True:
        first = stream.read(1)
        if not first:
            return
        header = first + _read_exact(stream, FRAME_HEADER.size - 1)

        frame_id, timestamp_ms, jpeg_size = FRAME_HEADER.unpack(header)
        if jpeg_size == 0 or jpeg_size > MAX_FRAME_BYTES:
            raise StreamProtocolError("invalid stream JPEG size")

        yield frame_id, timestamp_ms, _read_exact(stream, jpeg_size)
