"""
LLM ranking prompt enhancement for structured queries.

Builds additional context strings to inject into the LLM relevance
scoring prompt, so it understands the user's intent (scope, focus, depth).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_ranking_context(
    structured: Dict[str, Optional[str]],
    scope: str,
    focus: str,
    depth: str,
) -> str:
    """Build additional context for the LLM ranking prompt.

    Parameters
    ----------
    structured : dict
        Decomposed query with keys: topic, domain, aspect.
    scope : str
        One of: intersection, broad, topic_focused, domain_focused.
    focus : str
        One of: foundational, recent, surveys, all_time.
    depth : str
        One of: high_level, comprehensive.

    Returns
    -------
    str
        Context string to inject into the LLM prompt.
    """
    topic = structured.get("topic", "")
    domain = structured.get("domain")
    aspect = structured.get("aspect")

    lines = [f"The user is researching: {topic}"]

    if domain:
        lines.append(f"Applied to domain: {domain}")
    if aspect:
        lines.append(f"Specific focus: {aspect}")

    # Scope instructions
    scope_instructions = {
        "intersection": (
            f"CRITICAL: The user wants papers at the INTERSECTION of {topic} and {domain}.\n"
            f"Papers that only cover {topic} without {domain} relevance should be ranked LOW.\n"
            f"Papers that only cover {domain} without {topic} methodology should be ranked LOW.\n"
            f"Papers combining BOTH {topic} AND {domain} should be ranked HIGH."
        ),
        "broad": (
            f"The user wants a broad overview covering both {topic} and "
            f"{domain or 'related areas'}. Include foundational and survey papers."
        ),
        "topic_focused": (
            f"The user is primarily interested in {topic}. "
            f"{domain or 'Application'} context is a bonus but not required."
        ),
        "domain_focused": (
            f"The user is primarily interested in {domain}. "
            f"Papers using {topic} as methodology for {domain} should rank highest."
        ),
    }

    lines.append("")
    lines.append(scope_instructions.get(scope, ""))

    # Focus instructions
    focus_instructions = {
        "foundational": "Prefer seminal, highly-cited papers that established the field.",
        "recent": "Prefer recent papers (last 3 years) showing current state of the art.",
        "surveys": "Prefer survey papers, reviews, and meta-analyses that provide comprehensive overviews.",
        "all_time": "Consider papers from all time periods equally.",
    }
    lines.append(focus_instructions.get(focus, ""))

    return "\n".join(lines)
