import time
import uuid

TTL_SECONDS = 30 * 60
MAX_ENTRIES = 20

_store: dict[str, tuple[bytes, str, float]] = {}


def _evict_expired() -> None:
    cutoff = time.time() - TTL_SECONDS
    for key in [k for k, (_, _, created_at) in _store.items() if created_at < cutoff]:
        del _store[key]
    while len(_store) > MAX_ENTRIES:
        oldest_key = min(_store, key=lambda k: _store[k][2])
        del _store[oldest_key]


def put(data: bytes, mime: str) -> str:
    _evict_expired()
    media_id = uuid.uuid4().hex
    _store[media_id] = (data, mime, time.time())
    return media_id


def get(media_id: str) -> tuple[bytes, str] | None:
    entry = _store.get(media_id)
    if entry is None:
        return None
    data, mime, created_at = entry
    if created_at < time.time() - TTL_SECONDS:
        del _store[media_id]
        return None
    return data, mime
