"""Unit tests for app.shared.s2_keys — round-robin S2 API key rotation."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestS2KeyRotation:

    def _reload(self):
        import app.shared.s2_keys as mod
        importlib.reload(mod)
        return mod

    def test_single_key_backward_compatible(self):
        with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "solo"}, clear=True):
            mod = self._reload()
            assert mod.get_s2_headers() == {"x-api-key": "solo"}
            assert mod.has_s2_key() is True

    def test_multi_key_round_robin(self):
        with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEYS": "a,b,c"}, clear=True):
            mod = self._reload()
            keys = [mod.get_s2_headers()["x-api-key"] for _ in range(6)]
            assert keys == ["a", "b", "c", "a", "b", "c"]

    def test_no_keys_returns_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            mod = self._reload()
            assert mod.get_s2_headers() == {}
            assert mod.has_s2_key() is False

    def test_multi_takes_precedence_over_singular(self):
        with patch.dict("os.environ", {
            "SEMANTIC_SCHOLAR_API_KEY": "old",
            "SEMANTIC_SCHOLAR_API_KEYS": "new1,new2",
        }, clear=True):
            mod = self._reload()
            assert mod.get_s2_headers()["x-api-key"] in ("new1", "new2")

    def test_thread_safety(self):
        with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEYS": "x,y,z"}, clear=True):
            mod = self._reload()
            with ThreadPoolExecutor(max_workers=10) as pool:
                results = list(pool.map(lambda _: mod.get_s2_headers(), range(30)))
            assert all("x-api-key" in r for r in results)
            assert all(r["x-api-key"] in ("x", "y", "z") for r in results)

    def test_whitespace_trimmed(self):
        with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEYS": " a , b , c "}, clear=True):
            mod = self._reload()
            keys = [mod.get_s2_headers()["x-api-key"] for _ in range(3)]
            assert keys == ["a", "b", "c"]

    def test_empty_entries_skipped(self):
        with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEYS": "a,,b,"}, clear=True):
            mod = self._reload()
            keys = [mod.get_s2_headers()["x-api-key"] for _ in range(4)]
            assert keys == ["a", "b", "a", "b"]
