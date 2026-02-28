"""Round-robin Semantic Scholar API key rotation.

Loads keys from SEMANTIC_SCHOLAR_API_KEYS (comma-separated) with fallback
to SEMANTIC_SCHOLAR_API_KEY (single key, backward compatible).
"""

from __future__ import annotations

import itertools
import os
import threading
from typing import Dict


def _load_keys() -> list[str]:
    multi = os.environ.get("SEMANTIC_SCHOLAR_API_KEYS", "")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if single.strip():
        return [single.strip()]
    return []


_keys = _load_keys()
_cycle = itertools.cycle(_keys) if _keys else None
_lock = threading.Lock()


def get_s2_headers() -> Dict[str, str]:
    """Return headers with the next round-robin S2 API key."""
    if _cycle is None:
        return {}
    with _lock:
        key = next(_cycle)
    return {"x-api-key": key}


def has_s2_key() -> bool:
    """Return True if at least one S2 API key is configured."""
    return bool(_keys)
