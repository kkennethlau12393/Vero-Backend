"""
Unit tests for app/feature3/node_timeline.py pure functions.

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.node_timeline import (
    era_map_to_sections,
    get_era_label,
    group_papers_by_era,
    merge_era_maps,
)


@pytest.mark.unit
class TestGetEraLabel:
    def test_1990s(self):
        assert get_era_label(1995) == "1990s"

    def test_2000s(self):
        assert get_era_label(2005) == "2000s"

    def test_2010s(self):
        assert get_era_label(2017) == "2010s"

    def test_none_returns_unknown(self):
        assert get_era_label(None) == "Unknown"

    def test_1960s(self):
        assert get_era_label(1965) == "1960s"

    def test_exact_decade_boundary(self):
        assert get_era_label(2020) == "2020s"


@pytest.mark.unit
class TestGroupPapersByEra:
    def test_groups_correctly(self):
        papers = [
            {"work_id": "W1", "title": "Paper A", "year": 2015, "cited_by_count": 100},
            {"work_id": "W2", "title": "Paper B", "year": 2018, "cited_by_count": 200},
            {"work_id": "W3", "title": "Paper C", "year": 2005, "cited_by_count": 300},
        ]
        result = group_papers_by_era(papers, "reference")
        assert "2010s" in result
        assert "2000s" in result
        assert len(result["2010s"]) == 2
        assert len(result["2000s"]) == 1

    def test_empty_list(self):
        result = group_papers_by_era([], "reference")
        assert result == {}

    def test_adds_relationship_field(self):
        papers = [{"work_id": "W1", "title": "P", "year": 2020, "cited_by_count": 10}]
        result = group_papers_by_era(papers, "landmark")
        assert result["2020s"][0]["relationship"] == "landmark"

    def test_unknown_year(self):
        papers = [{"work_id": "W1", "title": "P", "year": None, "cited_by_count": 10}]
        result = group_papers_by_era(papers, "reference")
        assert "Unknown" in result


@pytest.mark.unit
class TestMergeEraMaps:
    def test_non_overlapping(self):
        map1 = {"2000s": [{"work_id": "W1"}]}
        map2 = {"2010s": [{"work_id": "W2"}]}
        merged = merge_era_maps(map1, map2)
        assert "2000s" in merged
        assert "2010s" in merged

    def test_overlapping_combines(self):
        map1 = {"2010s": [{"work_id": "W1"}]}
        map2 = {"2010s": [{"work_id": "W2"}]}
        merged = merge_era_maps(map1, map2)
        assert len(merged["2010s"]) == 2

    def test_empty_maps(self):
        merged = merge_era_maps({}, {})
        assert merged == {}

    def test_three_maps(self):
        map1 = {"2000s": [{"work_id": "W1"}]}
        map2 = {"2010s": [{"work_id": "W2"}]}
        map3 = {"2020s": [{"work_id": "W3"}]}
        merged = merge_era_maps(map1, map2, map3)
        assert len(merged) == 3


@pytest.mark.unit
class TestEraMapToSections:
    def test_chronological_sorting(self):
        era_map = {
            "2010s": [{"work_id": "W1", "cited_by_count": 10}],
            "1990s": [{"work_id": "W2", "cited_by_count": 20}],
            "2000s": [{"work_id": "W3", "cited_by_count": 30}],
        }
        sections = era_map_to_sections(era_map, sort_ascending=True)
        eras = [s["era"] for s in sections]
        assert eras == ["1990s", "2000s", "2010s"]

    def test_reverse_sorting(self):
        era_map = {
            "2010s": [{"work_id": "W1", "cited_by_count": 10}],
            "1990s": [{"work_id": "W2", "cited_by_count": 20}],
        }
        sections = era_map_to_sections(era_map, sort_ascending=False)
        eras = [s["era"] for s in sections]
        assert eras == ["2010s", "1990s"]

    def test_unknown_era_last(self):
        era_map = {
            "Unknown": [{"work_id": "W1", "cited_by_count": 10}],
            "2010s": [{"work_id": "W2", "cited_by_count": 20}],
        }
        sections = era_map_to_sections(era_map, sort_ascending=True)
        assert sections[-1]["era"] == "Unknown"

    def test_papers_sorted_by_citations(self):
        era_map = {
            "2010s": [
                {"work_id": "W1", "cited_by_count": 10},
                {"work_id": "W2", "cited_by_count": 100},
            ],
        }
        sections = era_map_to_sections(era_map)
        papers = sections[0]["papers"]
        assert papers[0]["cited_by_count"] == 100
        assert papers[1]["cited_by_count"] == 10

    def test_empty_era_map(self):
        assert era_map_to_sections({}) == []
