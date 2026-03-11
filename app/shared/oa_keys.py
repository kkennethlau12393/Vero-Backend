"""Round-robin OpenAlex API key rotation.

Loads keys from OPENALEX_API_KEYS (comma-separated) with fallback
to OPENALEX_API_KEY (single key, backward compatible).
"""

from __future__ import annotations

import itertools
import logging
import os
import threading

logger = logging.getLogger(__name__)


def _load_keys() -> list[str]:
    multi = os.environ.get("OPENALEX_API_KEYS", "")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("OPENALEX_API_KEY", "")
    if single.strip():
        return [single.strip()]
    return []


_keys = _load_keys()
_cycle = itertools.cycle(_keys) if _keys else None
_lock = threading.Lock()


def get_oa_api_key() -> str | None:
    """Return the next round-robin OpenAlex API key, or None if no keys configured."""
    if _cycle is None:
        return None
    with _lock:
        key = next(_cycle)
        idx = _keys.index(key) + 1
    logger.info(f"[OA-KEY] Using key {idx}/{len(_keys)} (...{key[-6:]})")
    return key
