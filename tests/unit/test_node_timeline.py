"""
Unit tests for app/feature3/node_timeline.py pure functions.

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.node_timeline import (
    _era_sort_key,
    compute_era_width,
    era_map_to_sections,
    get_era_label,
    group_papers_by_era,
    merge_era_maps,
)


@pytest.mark.unit
class TestComputeEraWidth:
    def test_empty_years(self):
        assert compute_era_width([]) == 10

    def test_wide_span_returns_decades(self):
        # IQR spans 1970-2010 → span=41 → decades
        assert compute_era_width([1960, 1970, 1980, 1990, 2000, 2010, 2020]) == 10

    def test_medium_span_returns_5_year(self):
        # IQR ~21 → 5-year blocks (6 distinct eras ≤ cap of 6)
        years = [1990, 2000, 2005, 2010, 2015, 2020, 2024]
        assert compute_era_width(years) == 5

    def test_narrow_span_returns_3_year(self):
        # IQR ~7 → 3-year blocks
        years = [2015, 2017, 2019, 2021, 2023, 2025]
        assert compute_era_width(years) == 3

    def test_very_narrow_span_returns_yearly(self):
        # IQR ~3 → individual years (4 eras ≤ cap of 6)
        years = [2020, 2021, 2022, 2023]
        assert compute_era_width(years) == 1

    def test_outlier_landmarks_ignored(self):
        # Most papers 2019-2025 but one outlier at 1990 — IQR should be narrow
        years = [1990, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        width = compute_era_width(years)
        assert width <= 3  # Should not be decades due to IQR

    def test_posthoc_cap_prevents_too_many_eras(self):
        # IQR is narrow (span≤4) → initial=yearly, but 8 distinct years
        # would create 8 yearly eras > cap of 6, so bumps to 3-year
        years = [2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        width = compute_era_width(years)
        assert width >= 3  # yearly would give too many eras

    def test_diffusion_field_gets_3_year_blocks(self):
        # Typical diffusion models field: 2010-2024, most papers 2019-2024
        years = [2010, 2014, 2017, 2019, 2020, 2021, 2022, 2023, 2024]
        width = compute_era_width(years)
        assert width == 3  # 5 eras at width=3, not 7+ yearly eras

    def test_single_year(self):
        assert compute_era_width([2020]) == 1

    def test_two_years_close(self):
        assert compute_era_width([2020, 2021]) == 1

    def test_two_years_far(self):
        assert compute_era_width([1990, 2020]) == 10


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

    def test_yearly_era(self):
        assert get_era_label(2022, era_width=1) == "2022"

    def test_3_year_era(self):
        label = get_era_label(2022, era_width=3)
        assert label == "2022-2024"  # floor(2022/3)*3 = 2022

    def test_5_year_era(self):
        label = get_era_label(2017, era_width=5)
        assert label == "2015-2019"  # floor(2017/5)*5 = 2015

    def test_none_with_era_width(self):
        assert get_era_label(None, era_width=3) == "Unknown"


@pytest.mark.unit
class TestEraSortKey:
    def test_decade_format(self):
        assert _era_sort_key("2020s") == 2020

    def test_year_format(self):
        assert _era_sort_key("2022") == 2022

    def test_range_format(self):
        assert _era_sort_key("2020-2024") == 2020

    def test_unknown(self):
        assert _era_sort_key("Unknown") == 9999


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

    def test_yearly_era_width(self):
        papers = [
            {"work_id": "W1", "title": "P1", "year": 2020, "cited_by_count": 10},
            {"work_id": "W2", "title": "P2", "year": 2021, "cited_by_count": 20},
            {"work_id": "W3", "title": "P3", "year": 2020, "cited_by_count": 30},
        ]
        result = group_papers_by_era(papers, "reference", era_width=1)
        assert "2020" in result
        assert "2021" in result
        assert len(result["2020"]) == 2
        assert len(result["2021"]) == 1

    def test_3_year_era_width(self):
        papers = [
            {"work_id": "W1", "title": "P1", "year": 2020, "cited_by_count": 10},
            {"work_id": "W2", "title": "P2", "year": 2023, "cited_by_count": 20},
        ]
        result = group_papers_by_era(papers, "reference", era_width=3)
        # 2020 → "2019-2021" (floor(2020/3)*3=2019+2=2021) Wait no...
        # floor(2020/3)*3 = 673*3 = 2019, so "2019-2021"
        # floor(2023/3)*3 = 674*3 = 2022, so "2022-2024"
        assert len(result) == 2


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

    def test_yearly_era_sorting(self):
        era_map = {
            "2022": [{"work_id": "W1", "cited_by_count": 10}],
            "2020": [{"work_id": "W2", "cited_by_count": 20}],
            "2021": [{"work_id": "W3", "cited_by_count": 30}],
        }
        sections = era_map_to_sections(era_map, sort_ascending=True)
        eras = [s["era"] for s in sections]
        assert eras == ["2020", "2021", "2022"]

    def test_range_era_sorting(self):
        era_map = {
            "2020-2024": [{"work_id": "W1", "cited_by_count": 10}],
            "2010-2014": [{"work_id": "W2", "cited_by_count": 20}],
            "2015-2019": [{"work_id": "W3", "cited_by_count": 30}],
        }
        sections = era_map_to_sections(era_map, sort_ascending=True)
        eras = [s["era"] for s in sections]
        assert eras == ["2010-2014", "2015-2019", "2020-2024"]

    def test_mixed_era_formats_sorting(self):
        """Handles legacy decade + new range formats in same map."""
        era_map = {
            "1990s": [{"work_id": "W1", "cited_by_count": 10}],
            "2020-2024": [{"work_id": "W2", "cited_by_count": 20}],
        }
        sections = era_map_to_sections(era_map, sort_ascending=True)
        eras = [s["era"] for s in sections]
        assert eras == ["1990s", "2020-2024"]
