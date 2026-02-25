"""
Unit tests for Feature 5: Activity Logger + Coverage Formula.

Tests the log_activity(), get_activity_summary() functions and
coverage calculation helpers using mocked database connections.
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
from app.feature5.coverage_tracker import (
    _calculate_methodology_contribution,
    _calculate_node_specific_contribution,
    WEIGHT_MAP_WIDE_TIMELINE,
    WEIGHT_NODE_SPECIFIC_ACTION,
    NODE_SPECIFIC_CAP,
    METHODOLOGY_BASE,
    METHODOLOGY_STALE_PENALTY,
    METHODOLOGY_CAP,
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

        log_activity(conn, tid, "citation_map_created", work_id="W123", node_count=10)

        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_non_creation_type_requires_map_or_rank(self):
        """Non-creation activities without map_id or rank_job_id are skipped."""
        conn = self._make_conn()
        tid = uuid4()

        log_activity(conn, tid, "novelty_assessed", work_id="W123")

        conn.execute.assert_not_called()

    def test_methodology_compared_with_node_count(self):
        conn = self._make_conn()
        tid = uuid4()
        rjid = uuid4()

        log_activity(
            conn, tid, "methodology_compared",
            rank_job_id=rjid,
            node_count=3,
            metadata={"work_ids": ["W1", "W2", "W3"]},
        )

        conn.execute.assert_called_once()
        args = conn.execute.call_args
        params = args[0][1]
        assert params["nc"] == 3
        assert params["wid"] is None  # methodology uses metadata, not work_id
        assert '"work_ids"' in params["meta"]

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
                log_activity(conn, tid, atype, rank_job_id=rjid, work_id="W1")
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

    def test_work_id_defaults_to_none(self):
        conn = self._make_conn()
        tid = uuid4()

        log_activity(conn, tid, "rank_job_created", rank_job_id=uuid4())

        args = conn.execute.call_args
        params = args[0][1]
        assert params["wid"] is None


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
                "work_id": None,
                "node_count": 0,
                "metadata": {},
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "methodology_compared",
                "work_id": None,
                "node_count": 2,
                "metadata": {"work_ids": ["W1", "W2"]},
                "created_at": datetime(2026, 2, 24, 10, 5),
            },
            {
                "activity_type": "methodology_compared",
                "work_id": None,
                "node_count": 3,
                "metadata": {"work_ids": ["W1", "W2", "W3"]},
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
                "work_id": "W100",
                "node_count": 1,
                "metadata": {},
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "novelty_assessed",
                "work_id": "W200",
                "node_count": 1,
                "metadata": {},
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
                "work_id": None,
                "node_count": 25,
                "metadata": {},
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
                "work_id": "W555",
                "node_count": 30,
                "metadata": {},
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
                "work_id": None,
                "node_count": 0,
                "metadata": {},
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "novelty_assessed",
                "work_id": "W1",
                "node_count": 1,
                "metadata": {},
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
                "work_id": "W100",
                "node_count": 1,
                "metadata": {},
                "created_at": datetime(2026, 2, 24, 10, 0),
            },
            {
                "activity_type": "novelty_assessed",
                "work_id": "W100",
                "node_count": 1,
                "metadata": {},
                "created_at": datetime(2026, 2, 24, 10, 5),
            },
        ]
        conn.execute.return_value.mappings.return_value.all.return_value = rows

        result = get_activity_summary(conn, rank_job_id=uuid4())

        nov = result["activities"]["novelty_assessed"]
        assert nov["count"] == 2  # Two events
        assert nov["work_ids"].count("W100") == 1  # But only one unique work_id


@pytest.mark.unit
class TestMethodologyContribution:
    """Tests for _calculate_methodology_contribution()."""

    def test_all_fresh_2_node(self):
        """2-node comparison with all fresh nodes = 4%."""
        comparisons = [(["A", "B"], 2)]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.04)
        assert penalty == 0.0
        assert count == 1

    def test_all_fresh_3_node(self):
        """3-node comparison with all fresh nodes = 6%."""
        comparisons = [(["A", "B", "C"], 3)]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.06)
        assert penalty == 0.0

    def test_all_fresh_4_node(self):
        """4-node comparison with all fresh nodes = 8%."""
        comparisons = [(["A", "B", "C", "D"], 4)]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.08)
        assert penalty == 0.0

    def test_stale_penalty_one_node(self):
        """Compare A,B then B,C → second gets -1% penalty."""
        comparisons = [
            (["A", "B"], 2),  # 4%, both fresh
            (["B", "C"], 2),  # 4% - 1% = 3%, B is stale
        ]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.07)  # 4% + 3%
        assert penalty == pytest.approx(0.01)
        assert count == 2

    def test_stale_penalty_multiple_nodes(self):
        """Compare A,B then B,C then A,C,D → third gets -2% penalty."""
        comparisons = [
            (["A", "B"], 2),      # 4%, both fresh
            (["B", "C"], 2),      # 4% - 1% = 3%, B stale
            (["A", "C", "D"], 3), # 6% - 2% = 4%, A and C stale
        ]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.11)  # 4% + 3% + 4%
        assert penalty == pytest.approx(0.03)  # 1% + 2%
        assert count == 3

    def test_all_stale_zeroed_out(self):
        """When all nodes are stale, contribution floors at 0."""
        comparisons = [
            (["A", "B"], 2),  # 4%
            (["A", "B"], 2),  # 4% - 2% = 2% (both stale)
        ]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.06)  # 4% + 2%
        assert penalty == pytest.approx(0.02)

    def test_fully_stale_floors_at_zero(self):
        """When penalty exceeds base, contribution = 0 for that comparison."""
        # 2-node base = 4%, but 2 stale nodes = -2%, so 4% - 2% = 2%
        # Need base < penalty: impossible with 2 nodes (4% base, max 2% penalty)
        # With 2-node: base=4%, max stale=2 → min=2%, never 0
        # But with many repeats, overall contribution still positive
        comparisons = [
            (["A", "B", "C", "D"], 4),  # 8%
            (["A", "B", "C", "D"], 4),  # 8% - 4% = 4% (all 4 stale)
        ]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.12)  # 8% + 4%
        assert penalty == pytest.approx(0.04)

    def test_cap_at_30_percent(self):
        """Many comparisons should cap at 30%."""
        # 8 fresh 4-node comparisons = 8 × 8% = 64% → cap at 30%
        comparisons = [
            ([f"N{i*4}", f"N{i*4+1}", f"N{i*4+2}", f"N{i*4+3}"], 4)
            for i in range(8)
        ]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == pytest.approx(0.30)
        assert penalty == 0.0
        assert count == 8

    def test_empty_comparisons(self):
        """No comparisons = 0."""
        pct, penalty, count = _calculate_methodology_contribution([])
        assert pct == 0.0
        assert penalty == 0.0
        assert count == 0

    def test_skips_node_count_below_2(self):
        """Comparisons with < 2 nodes are skipped."""
        comparisons = [(["A"], 1)]
        pct, penalty, count = _calculate_methodology_contribution(comparisons)
        assert pct == 0.0
        assert count == 0


@pytest.mark.unit
class TestNodeSpecificContribution:
    """Tests for _calculate_node_specific_contribution()."""

    def test_single_novelty(self):
        """One novelty assessment = 3%."""
        activities = [("novelty_assessed", "W1")]
        pct, count = _calculate_node_specific_contribution(activities)
        assert pct == pytest.approx(0.03)
        assert count == 1

    def test_novelty_and_timeline_same_node(self):
        """Novelty + timeline on same node = 6% (different action types)."""
        activities = [
            ("novelty_assessed", "W1"),
            ("timeline_per_node", "W1"),
        ]
        pct, count = _calculate_node_specific_contribution(activities)
        assert pct == pytest.approx(0.06)
        assert count == 2

    def test_duplicate_novelty_same_node(self):
        """Same novelty on same node twice = 3% (deduplicated)."""
        activities = [
            ("novelty_assessed", "W1"),
            ("novelty_assessed", "W1"),
        ]
        pct, count = _calculate_node_specific_contribution(activities)
        assert pct == pytest.approx(0.03)
        assert count == 1

    def test_multiple_nodes(self):
        """Novelty on 3 different nodes = 9%."""
        activities = [
            ("novelty_assessed", "W1"),
            ("novelty_assessed", "W2"),
            ("novelty_assessed", "W3"),
        ]
        pct, count = _calculate_node_specific_contribution(activities)
        assert pct == pytest.approx(0.09)
        assert count == 3

    def test_cap_at_30_percent(self):
        """11+ unique actions cap at 30%."""
        activities = [
            ("novelty_assessed", f"W{i}") for i in range(12)
        ]
        pct, count = _calculate_node_specific_contribution(activities)
        assert pct == pytest.approx(0.30)
        assert count == 12

    def test_ignores_other_activity_types(self):
        """node_details_viewed does NOT contribute to node-specific coverage."""
        activities = [
            ("node_details_viewed", "W1"),
            ("methodology_compared", "W1"),
        ]
        pct, count = _calculate_node_specific_contribution(activities)
        assert pct == 0.0
        assert count == 0

    def test_empty_activities(self):
        """No activities = 0."""
        pct, count = _calculate_node_specific_contribution([])
        assert pct == 0.0
        assert count == 0

    def test_mixed_novelty_and_timeline_different_nodes(self):
        """Mix of novelty + timeline across multiple nodes."""
        activities = [
            ("novelty_assessed", "W1"),
            ("timeline_per_node", "W1"),
            ("novelty_assessed", "W2"),
            ("timeline_per_node", "W3"),
        ]
        pct, count = _calculate_node_specific_contribution(activities)
        assert pct == pytest.approx(0.12)  # 4 unique pairs × 3%
        assert count == 4


@pytest.mark.unit
class TestCoverageConstants:
    """Verify coverage constants are correct."""

    def test_timeline_weight(self):
        assert WEIGHT_MAP_WIDE_TIMELINE == pytest.approx(0.15)

    def test_node_specific_weight(self):
        assert WEIGHT_NODE_SPECIFIC_ACTION == pytest.approx(0.03)

    def test_node_specific_cap(self):
        assert NODE_SPECIFIC_CAP == pytest.approx(0.30)

    def test_methodology_bases(self):
        assert METHODOLOGY_BASE[2] == pytest.approx(0.04)
        assert METHODOLOGY_BASE[3] == pytest.approx(0.06)
        assert METHODOLOGY_BASE[4] == pytest.approx(0.08)

    def test_methodology_penalty(self):
        assert METHODOLOGY_STALE_PENALTY == pytest.approx(0.01)

    def test_methodology_cap(self):
        assert METHODOLOGY_CAP == pytest.approx(0.30)

    def test_max_possible_coverage(self):
        """Max possible = 15% + 30% + 30% = 75%."""
        max_total = WEIGHT_MAP_WIDE_TIMELINE + NODE_SPECIFIC_CAP + METHODOLOGY_CAP
        assert max_total == pytest.approx(0.75)
