"""
Robust JSON extraction utilities for LLM responses.

This module provides reliable JSON extraction from LLM outputs that may contain:
- Markdown code blocks (with or without language specifiers)
- Extra text before/after JSON
- Multiple JSON structures (extracts first valid one)
- Empty or malformed responses
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def extract_json_from_llm_response(
    content: str,
    expected_type: str = "auto",
) -> Tuple[Optional[Union[dict, list]], str]:
    """
    Extract JSON from an LLM response with robust error handling.

    Args:
        content: Raw LLM response text
        expected_type: "object" for dict, "array" for list, "auto" for either

    Returns:
        (parsed_json, error_message)
        - If successful: (dict_or_list, "")
        - If failed: (None, "descriptive error message")
    """
    if not content or not content.strip():
        return None, "Empty response from LLM"

    content = content.strip()

    # Step 1: Remove markdown code blocks (handles ```json, ```, etc.)
    content = _remove_code_blocks(content)

    if not content.strip():
        return None, "Empty content after removing code blocks"

    content = content.strip()

    # Step 2: Try direct JSON parse first (fast path for clean responses)
    try:
        result = json.loads(content)
        if _matches_expected_type(result, expected_type):
            return result, ""
    except json.JSONDecodeError:
        pass

    # Step 3: Extract JSON from mixed content
    result, error = _extract_json_structure(content, expected_type)
    if result is not None:
        return result, ""

    return None, error or "Could not extract valid JSON from response"


def _remove_code_blocks(content: str) -> str:
    """
    Remove markdown code blocks from content.

    Handles:
    - ```json ... ```
    - ``` ... ```
    - Multiple code blocks (extracts content from first one)
    """
    # Pattern for code block with optional language specifier
    # This matches ```json, ```JSON, ```, etc.
    code_block_pattern = r'```(?:json|JSON)?\s*\n?([\s\S]*?)```'

    match = re.search(code_block_pattern, content)
    if match:
        # Return content from inside the code block
        return match.group(1).strip()

    # No code block found - check if content starts/ends with ```
    # (handles malformed code blocks)
    if content.startswith("```"):
        lines = content.split("\n")
        # Skip first line (the ```) and find closing ```
        inner_lines = []
        started = False
        for line in lines:
            if not started:
                if line.startswith("```"):
                    started = True
                    # Skip this line (```json or just ```)
                    continue
            else:
                if line.strip() == "```":
                    break
                inner_lines.append(line)

        if inner_lines:
            return "\n".join(inner_lines).strip()

    return content


def _extract_json_structure(
    content: str,
    expected_type: str,
) -> Tuple[Optional[Union[dict, list]], str]:
    """
    Extract JSON object or array from mixed content.

    Uses balanced bracket matching to find the first complete JSON structure.
    """
    # Determine what we're looking for
    if expected_type == "object":
        start_chars = ["{"]
    elif expected_type == "array":
        start_chars = ["["]
    else:  # auto
        start_chars = ["{", "["]

    # Find the first occurrence of a start character
    best_pos = len(content)
    best_char = None

    for char in start_chars:
        pos = content.find(char)
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_char = char

    if best_char is None:
        return None, f"No JSON structure found (looking for {start_chars})"

    # Extract from the start character using balanced matching
    end_char = "}" if best_char == "{" else "]"
    json_str = _extract_balanced_json(content[best_pos:], best_char, end_char)

    if not json_str:
        return None, f"Could not find balanced {best_char}...{end_char}"

    try:
        result = json.loads(json_str)
        if _matches_expected_type(result, expected_type):
            return result, ""
        return None, f"Parsed JSON but got {type(result).__name__}, expected {expected_type}"
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def _extract_balanced_json(content: str, start_char: str, end_char: str) -> Optional[str]:
    """
    Extract a balanced JSON structure from content.

    Handles:
    - Nested brackets
    - Strings containing brackets
    - Escaped characters in strings
    """
    if not content or content[0] != start_char:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                # Found the matching closing bracket
                return content[:i + 1]

    return None


def _matches_expected_type(result: Any, expected_type: str) -> bool:
    """Check if result matches expected type."""
    if expected_type == "object":
        return isinstance(result, dict)
    elif expected_type == "array":
        return isinstance(result, list)
    else:  # auto
        return isinstance(result, (dict, list))


def safe_json_loads(content: str, default: Any = None) -> Any:
    """
    Safely parse JSON with a default fallback.

    Args:
        content: JSON string to parse
        default: Value to return if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return default


def repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON errors from LLM output.

    All repairs are string-aware to avoid corrupting string values.

    Handles:
    - Unquoted property names
    - Missing colons after property names
    - Missing commas between values
    - Missing values after colons
    - Trailing commas
    - Truncated JSON (closes open brackets/strings)

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string (may still be invalid)
    """
    if not json_str:
        return json_str

    # Do all repairs in a single pass that's string-aware
    result = _comprehensive_json_repair(json_str)

    return result


def _comprehensive_json_repair(json_str: str) -> str:
    """
    Comprehensive JSON repair that processes the entire string
    while respecting string boundaries.
    """
    chars = list(json_str)
    result = []
    i = 0
    in_string = False
    escape_next = False
    context_stack = []  # Stack of '{' or '[' to track context
    prev_was_value = False  # Track if previous token was a complete value
    prev_was_colon = False  # Track if previous non-ws char was ':'

    def peek_identifier():
        """Check if position i starts an identifier followed by colon."""
        nonlocal i
        if i >= len(chars):
            return None, 0
        if not (chars[i].isalpha() or chars[i] == '_'):
            return None, 0
        j = i
        while j < len(chars) and (chars[j].isalnum() or chars[j] == '_'):
            j += 1
        ident = ''.join(chars[i:j])
        # Skip whitespace
        k = j
        while k < len(chars) and chars[k] in ' \t\n\r':
            k += 1
        if k < len(chars) and chars[k] == ':':
            return ident, k - i + 1  # Length including the colon
        return None, 0

    while i < len(chars):
        char = chars[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char == '"':
            # Check if we need comma before this string
            if not in_string and prev_was_value:
                result.append(',')
            in_string = not in_string
            result.append(char)
            if not in_string:
                # Just closed a string - it's now a value
                prev_was_value = True
            else:
                prev_was_value = False
            prev_was_colon = False
            i += 1
            continue

        if in_string:
            result.append(char)
            i += 1
            continue

        # Outside string processing
        if char in ' \t\n\r':
            result.append(char)
            i += 1
            continue

        if char == '{':
            if prev_was_value:
                result.append(',')
            result.append(char)
            context_stack.append('{')
            prev_was_value = False
            prev_was_colon = False
            i += 1
            continue

        if char == '[':
            if prev_was_value:
                result.append(',')
            result.append(char)
            context_stack.append('[')
            prev_was_value = False
            prev_was_colon = False
            i += 1
            continue

        if char == '}':
            result.append(char)
            if context_stack and context_stack[-1] == '{':
                context_stack.pop()
            prev_was_value = True
            prev_was_colon = False
            i += 1
            continue

        if char == ']':
            result.append(char)
            if context_stack and context_stack[-1] == '[':
                context_stack.pop()
            prev_was_value = True
            prev_was_colon = False
            i += 1
            continue

        if char == ':':
            result.append(char)
            prev_was_value = False
            prev_was_colon = True
            i += 1
            continue

        if char == ',':
            result.append(char)
            prev_was_value = False
            prev_was_colon = False
            i += 1
            continue

        # Handle unquoted property names in object context
        if context_stack and context_stack[-1] == '{':
            ident, length = peek_identifier()
            if ident:
                if prev_was_value:
                    result.append(',')
                result.append('"')
                result.append(ident)
                result.append('"')
                # Skip to after the colon
                i += length
                # Find and append the colon
                while i > 0 and chars[i-1] != ':':
                    pass  # Should already be at colon
                result.append(':')
                prev_was_value = False
                prev_was_colon = True
                continue

        # Handle numbers
        if char.isdigit() or char == '-':
            if prev_was_value:
                result.append(',')
            # Collect the number
            while i < len(chars) and chars[i] in '0123456789.eE+-':
                result.append(chars[i])
                i += 1
            prev_was_value = True
            prev_was_colon = False
            continue

        # Handle true/false/null
        if char in 'tfn':
            if prev_was_value:
                result.append(',')
            # Check for keywords
            rest = ''.join(chars[i:i+5])
            if rest.startswith('true'):
                result.extend(['t', 'r', 'u', 'e'])
                i += 4
                prev_was_value = True
            elif rest.startswith('false'):
                result.extend(['f', 'a', 'l', 's', 'e'])
                i += 5
                prev_was_value = True
            elif rest.startswith('null'):
                result.extend(['n', 'u', 'l', 'l'])
                i += 4
                prev_was_value = True
            else:
                result.append(char)
                i += 1
            prev_was_colon = False
            continue

        result.append(char)
        i += 1

    # Close any unclosed strings
    if in_string:
        result.append('"')

    # Close any unclosed brackets
    while context_stack:
        ctx = context_stack.pop()
        if ctx == '{':
            result.append('}')
        else:
            result.append(']')

    return ''.join(result)


def _fix_unquoted_property_names(json_str: str) -> str:
    """
    Fix unquoted property names while respecting string boundaries.

    Handles cases like:
        {key: "value"} -> {"key": "value"}
        {"a": 1, key: "value"} -> {"a": 1, "key": "value"}
        {"a": 1
         key: "value"} -> {"a": 1, "key": "value"}

    Does NOT modify content inside strings.
    """
    chars = list(json_str)
    result = []
    i = 0
    in_string = False
    escape_next = False
    object_depth = 0  # Track if we're inside an object

    while i < len(chars):
        char = chars[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if in_string:
            result.append(char)
            i += 1
            continue

        # Track object depth
        if char == '{':
            object_depth += 1
            result.append(char)
            i += 1
            continue
        elif char == '}':
            object_depth = max(0, object_depth - 1)
            result.append(char)
            i += 1
            continue

        # Inside an object, check for unquoted identifier followed by :
        if object_depth > 0 and (char.isalpha() or char == '_'):
            # Check if this looks like an unquoted property name
            # Save current position
            ident_start = i
            j = i
            while j < len(chars) and (chars[j].isalnum() or chars[j] == '_'):
                j += 1

            identifier = ''.join(chars[ident_start:j])

            # Skip whitespace after identifier
            ws_start = j
            while j < len(chars) and chars[j] in ' \t\n\r':
                j += 1

            # Check if followed by colon
            if j < len(chars) and chars[j] == ':':
                # This is an unquoted property name - quote it
                result.append('"')
                result.append(identifier)
                result.append('"')
                # Add whitespace
                for k in range(ws_start, j):
                    result.append(chars[k])
                result.append(':')
                i = j + 1
                continue

        result.append(char)
        i += 1

    return ''.join(result)


def _fix_missing_colons(json_str: str) -> str:
    """
    Fix missing colons after property names.

    Handles cases like:
        {"key" "value"} -> {"key": "value"}
        {"key" 123} -> {"key": 123}
        {"key" {}} -> {"key": {}}

    Uses string-aware processing to only fix outside of string values.
    """
    chars = list(json_str)
    result = []
    i = 0
    in_string = False
    escape_next = False
    last_string_was_key = False  # Track if last string could be a property key
    after_open_brace_or_comma = False

    while i < len(chars):
        char = chars[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char == '"':
            if not in_string:
                # Starting a string
                in_string = True
                # Check if this could be a property key (after { or ,)
                last_string_was_key = after_open_brace_or_comma
            else:
                # Ending a string
                in_string = False
            result.append(char)
            i += 1
            continue

        if in_string:
            result.append(char)
            i += 1
            continue

        # Track context for key detection
        if char in '{,':
            after_open_brace_or_comma = True
            last_string_was_key = False
        elif char == ':':
            after_open_brace_or_comma = False
            last_string_was_key = False
        elif char not in ' \t\n\r':
            # Non-whitespace after a potential key string
            if last_string_was_key and char != ':':
                # Check if we just ended a string and there's no colon
                # Look back for the closing quote
                j = len(result) - 1
                while j >= 0 and result[j] in ' \t\n\r':
                    j -= 1
                if j >= 0 and result[j] == '"':
                    # Insert colon before whitespace
                    ws_start = j + 1
                    ws = result[ws_start:]
                    result = result[:ws_start]
                    result.append(':')
                    result.extend(ws)
            after_open_brace_or_comma = False
            last_string_was_key = False

        result.append(char)
        i += 1

    return ''.join(result)


def _repair_truncated_json(json_str: str) -> str:
    """
    Attempt to repair truncated JSON by closing open structures.

    Handles:
    - Unclosed strings
    - Unclosed brackets/braces
    """
    # Track state
    in_string = False
    escape_next = False
    open_brackets = []  # Stack of open brackets

    for char in json_str:
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            open_brackets.append('}')
        elif char == '[':
            open_brackets.append(']')
        elif char in '}]':
            if open_brackets and open_brackets[-1] == char:
                open_brackets.pop()

    result = json_str

    # If we're still in a string, close it
    if in_string:
        result += '"'

    # Close any open brackets (in reverse order)
    result += ''.join(reversed(open_brackets))

    return result


def _repair_missing_commas(json_str: str) -> str:
    """
    Insert missing commas between JSON values/elements.

    Uses a simpler approach: identify where values end and new values begin
    without proper comma separation.
    """
    # Track state while scanning
    chars = list(json_str)
    result = []
    i = 0
    in_string = False
    escape_next = False

    def get_prev_token_end():
        """Find the end position of the previous token (skipping whitespace)."""
        j = len(result) - 1
        while j >= 0 and result[j] in ' \t\n\r':
            j -= 1
        return j

    def prev_token_is_value_end():
        """Check if the previous token is the end of a JSON value."""
        j = get_prev_token_end()
        if j < 0:
            return False
        char = result[j]
        # Value ends: closing quote, digit, bracket, or literal keywords
        if char == '"':
            return True
        if char.isdigit() or char == '.':
            return True
        if char in '}]':
            return True
        # Check for true/false/null
        token = ''.join(result[max(0, j-4):j+1]).rstrip()
        if token.endswith('true') or token.endswith('false') or token.endswith('null'):
            return True
        return False

    def prev_is_colon_or_open():
        """Check if previous non-whitespace is : or { or [ or ,"""
        j = get_prev_token_end()
        if j < 0:
            return True  # Start of document
        return result[j] in ':,{['

    while i < len(chars):
        char = chars[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char == '"':
            # Check if we need to insert comma before this string
            if not in_string:
                # Starting a new string - check if comma needed
                if prev_token_is_value_end() and not prev_is_colon_or_open():
                    result.append(',')
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if in_string:
            result.append(char)
            i += 1
            continue

        # Outside string - handle structural characters
        if char in '{[':
            # Opening bracket - check if comma needed
            if prev_token_is_value_end() and not prev_is_colon_or_open():
                result.append(',')
            result.append(char)
            i += 1
            continue

        if char in '}]:,':
            # These don't need comma before them
            result.append(char)
            i += 1
            continue

        # Handle start of number, boolean, or null
        if char.isdigit() or char == '-':
            if prev_token_is_value_end() and not prev_is_colon_or_open():
                result.append(',')
            result.append(char)
            i += 1
            continue

        if char in 'tfn':  # true, false, null
            if prev_token_is_value_end() and not prev_is_colon_or_open():
                result.append(',')
            result.append(char)
            i += 1
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def extract_json_from_llm_response_with_repair(
    content: str,
    expected_type: str = "auto",
) -> Tuple[Optional[Union[dict, list]], str]:
    """
    Extract JSON from LLM response with repair attempt for malformed JSON.

    First tries standard extraction, then attempts repair if that fails.

    Args:
        content: Raw LLM response text
        expected_type: "object" for dict, "array" for list, "auto" for either

    Returns:
        (parsed_json, error_message)
    """
    # Try standard extraction first
    result, error = extract_json_from_llm_response(content, expected_type)
    if result is not None:
        return result, ""

    # If standard extraction failed, try repair
    if not content or not content.strip():
        return None, "Empty response from LLM"

    content = content.strip()
    content = _remove_code_blocks(content)

    if not content.strip():
        return None, "Empty content after removing code blocks"

    # Try to repair the JSON
    repaired = repair_json(content)

    # Try parsing repaired JSON
    try:
        result = json.loads(repaired)
        if _matches_expected_type(result, expected_type):
            logger.info("JSON repair successful")
            return result, ""
    except json.JSONDecodeError:
        pass

    # Try extracting structure from repaired content
    result, extract_error = _extract_json_structure(repaired, expected_type)
    if result is not None:
        logger.info("JSON repair + extraction successful")
        return result, ""

    return None, f"JSON repair failed. Original error: {error}"
