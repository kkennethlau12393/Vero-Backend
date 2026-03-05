"""Short integer <-> work_id mapping for LLM prompts.

Usage:
    mapper = IdMapper("R")  # prefix for references
    short = mapper.add("W2163605009")  # returns "R1"
    mapper.add("S2:204e3073870f")      # returns "R2"

    # After LLM response:
    text = mapper.resolve_text("Paper [R1] introduced...")
    # -> "Paper [W2163605009] introduced..."

    d = mapper.resolve_dict({"R1": {"score": 8.5}})
    # -> {"W2163605009": {"score": 8.5}}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class IdMapper:
    """Bidirectional mapping between work_ids and short integer keys."""

    def __init__(self, prefix: str = "P"):
        self.prefix = prefix
        self._to_short: Dict[str, str] = {}
        self._to_real: Dict[str, str] = {}
        self._counter = 0

    def add(self, work_id: str) -> str:
        """Register a work_id and return its short key. Idempotent."""
        if work_id in self._to_short:
            return self._to_short[work_id]
        self._counter += 1
        short = f"{self.prefix}{self._counter}"
        self._to_short[work_id] = short
        self._to_real[short] = work_id
        return short

    def add_all(self, work_ids: List[str]) -> List[str]:
        """Register multiple work_ids. Returns list of short keys."""
        return [self.add(wid) for wid in work_ids]

    def short(self, work_id: str) -> str:
        """Get short key for a registered work_id."""
        return self._to_short[work_id]

    def real(self, short_key: str) -> str:
        """Get real work_id for a short key."""
        return self._to_real[short_key]

    def get_short(self, work_id: str, default: str = "") -> str:
        """Get short key, returning default if not registered."""
        return self._to_short.get(work_id, default)

    def resolve_text(self, text: str) -> str:
        """Replace all short keys in text with real work_ids.

        Handles both bracketed [R1] and bare R1 references.
        Processes longer keys first to avoid R1 matching inside R10.
        """
        if not text:
            return text
        result = text
        for short_key in sorted(self._to_real.keys(), key=len, reverse=True):
            real_id = self._to_real[short_key]
            result = result.replace(f"[{short_key}]", f"[{real_id}]")
            result = re.sub(rf'\b{re.escape(short_key)}\b', real_id, result)
        return result

    def resolve_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Replace short keys used as dict keys with real work_ids."""
        return {self._to_real.get(k, k): v for k, v in d.items()}

    def resolve_list(self, items: List[str]) -> List[str]:
        """Replace short keys in a list with real work_ids."""
        return [self._to_real.get(item, item) for item in items]
