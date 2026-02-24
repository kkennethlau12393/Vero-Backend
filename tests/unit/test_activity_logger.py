"""
Unit tests for Feature 5: Activity Logger.

Tests the log_activity() and get_activity_summary() functions
using mocked database connections.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
from uuid import UUID, uuid4

import pytest

from app.feature5.activity_logger import (
    ACTIVITY_TYPES,
    log_activity,
    get_activity_summary,
)


@pytest.mark.unit
class TestLogActivity:
    """Tests for log_activity()."""

    def _make_conn(self):
        conn = MagicMock()
        conn.execute = MagicMock()
        conn.commit = MagicMock()
        return conn

    def test_valid_activity_inserts_row(self):
        conn = self._make_conn()
        tid = uuid4()
        rjid = uuid4()

        log_activity(
            conn, tid, "rank_job_created",
            rank_job_id=rjid,
            node_count=25,
            metadata={"query_text": "transformers"},
        )

        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_unknown_activity_type_does_not_insert(self):
        conn = self._make_conn()
        tid = uuid4()

        log_activity(conn, tid, "unknown_type", rank_job_id=uuid4())

        conn.execute.assert_not_called()
        conn.commit.assert_not_called()

    def test_creation_type_allowed_without_map_or_rank(self):
        """citation_map_created and rank_job_created don't need map_id or rank_job_id."""
        conn = self._make_conn()
        tid = uuid4()

        log_activity(conn, tid, "citation_map_created", work_ids=["W123"], node_count=10)

        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_non_creation_type_requires_map_or_rank(self):
        """Non-creation activities without map_id or rank_job_id are skipped."""
        conn = self._make_conn()
        tid = uuid4()

        log_activity(conn, tid, "novelty_assessed", work_ids=["W123"])

        conn.execute.assert_not_called()

    def test_methodology_compared_with_node_count(self):
        conn = self._make_conn()
        tid = uuid4()
        rjid = uuid4()

        log_activity(
            conn, tid, "methodology_compared",
            rank_job_id=rjid,
            work_ids=["W1", "W2", "W3"],
            node_count=3,
        )

        conn.execute.assert_called_once()
        args = conn.execute.call_args
        params = args[0][1]
        assert params["nc"] == 3
        assert params["wids"] == ["W1", "W2", "W3"]

    def test_db_error_does_not_raise(self):
        """Activity logging must never break the primary API response."""
        conn = self._make_conn()
        conn.execute.side_effect = Exception("DB down")
        tid = uuid4()

        # Should not raise
        log_activity(conn, tid, "rank_job_created", rank_job_id=uuid4())

    def test_all_valid_activity_types(self):
        """Verify all declared activity types are accepted."""
        conn = self._make_conn()
        tid = uuid4()
        rjid = uuid4()

        creation_types = {"citation_map_created", "rank_job_created"}

        for atype in ACTIVITY_TYPES:
            conn.reset_mock()
            if atype in creation_types:
                log_activity(conn, tid, atype)
            else:
                log_activity(conn, tid, atype, rank_job_id=rjid, work_ids=["W1"])
            conn.execute.assert_called_once()

    def test_metadata_serialized_as_json(self):
        conn = self._make_conn()
        tid = uuid4()

        log_activity(
            conn, tid, "rank_job_created",
            rank_job_id=uuid4(),
            metadata={"key": "value", "count": 42},
        )

        args = conn.execute.call_args
        params = args[0][1]
        assert '"key"' in params["meta"]
        assert '"count"' in params["meta"]

    def test_null_metadata_defaults_to_empty_json(self):
        conn = self._make_conn()
        tid = uuid4()

        log_activity(conn, tid, "rank_job_created", rank_job_id=uuid4())

        args = conn.execute.call_args
        params = args[0][1]
        assert params["meta"] == "{}"

    def test_work_ids_default_to_empty_list(self):
        conn = self._make_conn()
        tid = uuid4()

        log_activity(conn, tid, "rank_job_created", rank_job_id=uuid4())

        args = conn.execute.call_args
        params = args[0][1]
        assert params["wids"] == []


@pytest.mark.unit
class TestGetActivitySummary:
    """Tests for get_activity_summary()."""

    def test_no_context_returns_empty(self):
        conn = MagicMock()
        result = get_activity_summary(conn)
        assert result == {"activities": {}}

    def test_no_rows_returns_zero_coverage(self):
        conn = MagicMock()
        conn.execute.return_value.mappings.return_value.all.return_value = []

        result = get_activity_summary(conn, rank_job_id=uuid4())

        assert result["coverage_pct"] == 0.0
        assert result["activities"] == {}

    def test_methodology_aggregation(self):
        """Methodology comparisons should track by_node_count and work_ids."""
        from datetime import datetime

        conn = MagicMock()
        rows = [
            {
                "activity_type": "rank_job_created",
                "work_ids": [],
                "node_count": 0,
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "methodology_compared",
                "work_ids": ["W1", "W2"],
                "node_count": 2,
                "created_at": datetime(2026, 2, 24, 10, 5),
            },
            {
                "activity_type": "methodology_compared",
                "work_ids": ["W1", "W2", "W3"],
                "node_count": 3,
                "created_at": datetime(2026, 2, 24, 10, 10),
            },
        ]
        conn.execute.return_value.mappings.return_value.all.return_value = rows

        result = get_activity_summary(conn, rank_job_id=uuid4())

        meth = result["activities"]["methodology_compared"]
        assert meth["count"] == 2
        assert meth["by_node_count"]["2"] == 1
        assert meth["by_node_count"]["3"] == 1
        assert "W1" in meth["work_ids_compared"]
        assert "W2" in meth["work_ids_compared"]
        assert "W3" in meth["work_ids_compared"]

    def test_novelty_tracks_work_ids(self):
        from datetime import datetime

        conn = MagicMock()
        rows = [
            {
                "activity_type": "novelty_assessed",
                "work_ids": ["W100"],
                "node_count": 1,
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "novelty_assessed",
                "work_ids": ["W200"],
                "node_count": 1,
                "created_at": datetime(2026, 2, 24, 10, 5),
            },
        ]
        conn.execute.return_value.mappings.return_value.all.return_value = rows

        result = get_activity_summary(conn, rank_job_id=uuid4())

        nov = result["activities"]["novelty_assessed"]
        assert nov["count"] == 2
        assert "W100" in nov["work_ids"]
        assert "W200" in nov["work_ids"]

    def test_entry_point_detection_rank(self):
        from datetime import datetime

        conn = MagicMock()
        rows = [
            {
                "activity_type": "rank_job_created",
                "work_ids": [],
                "node_count": 25,
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
        ]
        conn.execute.return_value.mappings.return_value.all.return_value = rows

        result = get_activity_summary(conn, rank_job_id=uuid4())
        assert result["entry_point"] == "rank"

    def test_entry_point_detection_citation_map(self):
        from datetime import datetime

        conn = MagicMock()
        rows = [
            {
                "activity_type": "citation_map_created",
                "work_ids": ["W555"],
                "node_count": 30,
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
        ]
        conn.execute.return_value.mappings.return_value.all.return_value = rows

        result = get_activity_summary(conn, map_id=uuid4())
        assert result["entry_point"] == "citation_map"

    def test_timestamps_tracked(self):
        from datetime import datetime

        conn = MagicMock()
        rows = [
            {
                "activity_type": "rank_job_created",
                "work_ids": [],
                "node_count": 0,
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "novelty_assessed",
                "work_ids": ["W1"],
                "node_count": 1,
                "created_at": datetime(2026, 2, 24, 11, 30),
            },
        ]
        conn.execute.return_value.mappings.return_value.all.return_value = rows

        result = get_activity_summary(conn, rank_job_id=uuid4())
        assert result["first_activity_at"] == "2026-02-24T10:00:00"
        assert result["last_activity_at"] == "2026-02-24T11:30:00"

    def test_duplicate_work_ids_deduplicated(self):
        """Same work_id assessed twice should appear once in work_ids list."""
        from datetime import datetime

        conn = MagicMock()
        rows = [
            {
                "activity_type": "novelty_assessed",
                "work_ids": ["W100"],
                "node_count": 1,
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "novelty_assessed",
                "work_ids": ["W100"],
                "node_count": 1,
                "created_at": datetime(2026, 2, 24, 10, 5),
            },
        ]
        conn.execute.return_value.mappings.return_value.all.return_value = rows

        result = get_activity_summary(conn, rank_job_id=uuid4())

        nov = result["activities"]["novelty_assessed"]
        assert nov["count"] == 2  # Two events
        assert nov["work_ids"].count("W100") == 1  # But only one unique work_id
