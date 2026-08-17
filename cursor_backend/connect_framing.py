"""Connect protocol framing helpers for Cursor HTTP/2 streams."""

from __future__ import annotations

import json
from collections.abc import Iterator


def frame_connect_message(data: bytes, flags: int = 0) -> bytes:
    """Return a Connect envelope for a protobuf payload."""

    payload = bytes(data)
    return bytes([flags]) + len(payload).to_bytes(4, "big") + payload


def parse_connect_frames(buffer: bytes | bytearray) -> Iterator[tuple[int, bytes]]:
    """Yield complete Connect frames from ``buffer``.

    If ``buffer`` is a ``bytearray``, parsed bytes are removed in-place so the
    caller can keep appending network data to the same buffer.
    """

    offset = 0
    total = len(buffer)

    while total - offset >= 5:
        flags = buffer[offset]
        msg_len = int.from_bytes(buffer[offset + 1 : offset + 5], "big")
        frame_end = offset + 5 + msg_len
        if total < frame_end:
            break
        yield flags, bytes(buffer[offset + 5 : frame_end])
        offset = frame_end

    if isinstance(buffer, bytearray) and offset:
        del buffer[:offset]


def parse_connect_end_stream(data: bytes) -> Exception | None:
    """Parse a Connect end-stream JSON payload into an exception."""

    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return Exception("Failed to parse Connect end stream")

    error = payload.get("error")
    if not error:
        return None

    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return Exception(
        f"Connect error {code or 'unknown'}: {message or 'Unknown error'}"
    )