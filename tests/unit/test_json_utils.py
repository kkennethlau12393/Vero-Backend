"""
Unit tests for app/feature3/json_utils.py

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.json_utils import (
    extract_json_from_llm_response,
    extract_json_from_llm_response_with_repair,
    repair_json,
    safe_json_loads,
)


@pytest.mark.unit
class TestExtractJsonFromLlmResponse:
    def test_clean_json_object(self):
        result, err = extract_json_from_llm_response('{"key": "value"}')
        assert result == {"key": "value"}
        assert err == ""

    def test_clean_json_array(self):
        result, err = extract_json_from_llm_response('[1, 2, 3]')
        assert result == [1, 2, 3]
        assert err == ""

    def test_json_in_markdown_code_block(self):
        content = '```json\n{"key": "value"}\n```'
        result, err = extract_json_from_llm_response(content)
        assert result == {"key": "value"}

    def test_json_in_plain_code_block(self):
        content = '```\n{"key": "value"}\n```'
        result, err = extract_json_from_llm_response(content)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        content = 'Here is the result:\n{"key": "value"}\nDone.'
        result, err = extract_json_from_llm_response(content)
        assert result == {"key": "value"}

    def test_empty_response(self):
        result, err = extract_json_from_llm_response("")
        assert result is None
        assert "Empty" in err

    def test_none_response(self):
        result, err = extract_json_from_llm_response(None)
        assert result is None

    def test_expected_type_object_rejects_array(self):
        result, err = extract_json_from_llm_response('[1, 2]', expected_type="object")
        assert result is None

    def test_expected_type_array_rejects_object(self):
        result, err = extract_json_from_llm_response('{"a": 1}', expected_type="array")
        assert result is None

    def test_auto_type_accepts_both(self):
        result1, _ = extract_json_from_llm_response('{"a": 1}', expected_type="auto")
        result2, _ = extract_json_from_llm_response('[1]', expected_type="auto")
        assert result1 is not None
        assert result2 is not None

    def test_nested_structures(self):
        content = '{"a": {"b": [1, 2, {"c": 3}]}}'
        result, err = extract_json_from_llm_response(content)
        assert result["a"]["b"][2]["c"] == 3

    def test_whitespace_only(self):
        result, err = extract_json_from_llm_response("   \n\n   ")
        assert result is None


@pytest.mark.unit
class TestRepairJson:
    def test_unclosed_brackets(self):
        repaired = repair_json('{"key": "value"')
        assert '"key"' in repaired
        assert repaired.endswith("}")

    def test_unclosed_strings(self):
        repaired = repair_json('{"key": "value')
        assert repaired.count('"') % 2 == 0

    def test_empty_string(self):
        assert repair_json("") == ""

    def test_valid_json_unchanged(self):
        valid = '{"key": "value"}'
        repaired = repair_json(valid)
        import json
        assert json.loads(repaired) == {"key": "value"}

    def test_unclosed_array(self):
        repaired = repair_json('[1, 2, 3')
        assert repaired.endswith("]")


@pytest.mark.unit
class TestExtractJsonWithRepair:
    def test_clean_json_no_repair(self):
        result, err = extract_json_from_llm_response_with_repair('{"a": 1}')
        assert result == {"a": 1}
        assert err == ""

    def test_empty_returns_none(self):
        result, err = extract_json_from_llm_response_with_repair("")
        assert result is None

    def test_completely_invalid(self):
        result, err = extract_json_from_llm_response_with_repair("not json at all")
        assert result is None


@pytest.mark.unit
class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_default(self):
        assert safe_json_loads("not json") is None
        assert safe_json_loads("not json", default=[]) == []

    def test_none_input(self):
        assert safe_json_loads(None) is None
