class ProtocolError(ValueError):
    """Raised when a frame request does not satisfy the wire protocol."""


def parse_non_negative_int(value: str | None, field_name: str) -> int:
    if value is None:
        raise ProtocolError(f"missing {field_name}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid {field_name}") from exc
    if parsed < 0:
        raise ProtocolError(f"invalid {field_name}")
    return parsed
