"""
External validation for Feature 5: Research Gap Analysis.

Uses GPT-5.2 + web_search to validate that identified gaps haven't
already been addressed by research outside our academic database.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.feature5.schemas import (
    ExternalValidationResult,
    GapCard,
)

logger = logging.getLogger(__name__)

# GPT-5.2 model for web search validation
GPT_MODEL = "gpt-5.2"

# Max parallel validation requests
MAX_VALIDATION_CONCURRENCY = 4


def get_openai_client() -> OpenAI:
    """Get OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def validate_gap(
    gap_title: str,
    gap_description: str,
    suggested_direction: str,
    gap_type: str,
) -> ExternalValidationResult:
    """
    Validate a single gap using GPT-5.2 + web_search.

    Searches the web to check if this gap has already been addressed
    by research outside our academic database.

    Args:
        gap_title: Title of the gap
        gap_description: Detailed description with evidence
        suggested_direction: Suggested research direction
        gap_type: Type of gap (structural, coverage, etc.)

    Returns:
        ExternalValidationResult with status, reasoning, and sources
    """
    client = get_openai_client()

    prompt = f"""You are a rigorous, unbiased research validation agent. Your job is to honestly assess whether a proposed research gap has already been addressed by existing work.

## CRITICAL INSTRUCTIONS ON HONESTY AND PRECISION

- You MUST be honest. If the gap has already been addressed, say so. Do NOT inflate the novelty of a gap to appear helpful.
- You have NO stake in the outcome. A gap being "addressed" is just as valid a finding as "open". There is no preferred answer.
- Do NOT cherry-pick sources that support the gap being open while ignoring sources that address it.
- Search BROADLY. Try multiple phrasings and adjacent fields. A gap in one field's terminology may be solved under different terminology in another field.
- If your search is inconclusive due to limited results, say "partial" and explain what you could and couldn't find. Do NOT default to "open" when uncertain.

## CRITICAL: Distinguish Topic-Level vs. Question-Level Coverage

- Finding papers on the BROAD TOPIC is NOT the same as addressing the SPECIFIC QUESTION.
- Example: A gap says "No formal theoretical analysis of why domain randomization enables sim-to-real transfer." If you find papers that USE domain randomization, that does NOT address this gap. Only papers that provide THEORETICAL ANALYSIS of WHY it works would count.
- "addressed" means the SPECIFIC question or knowledge gap has been answered with evidence. Not just that related work exists in the general area.
- If related work exists but doesn't answer the specific question posed, that is "partial" — set coverage_pct based on how close the existing work comes to answering the specific question.

## Gap to Validate

**Gap Type:** {gap_type}
**Gap Title:** {gap_title}
**Description:** {gap_description}
**Suggested Direction:** {suggested_direction}

## Task

Search for academic papers, preprints (ArXiv, bioRxiv, SSRN), technical reports, dissertations, or industry research that addresses this gap. Search using multiple query formulations including synonyms and related terminology from adjacent fields.

Respond with a JSON object. STRICT FORMAT RULES:
- "reasoning" should be a concise paragraph (4-8 sentences, ~150-250 words). Third person only (never "I searched" or "I found").
- State what exists, what doesn't, and the verdict. No bullet lists, no essays.

{{
    "status": "open" | "partial" | "addressed",
    "coverage_pct": 0-100,
    "reasoning": "Concise paragraph, third person, 150-250 words. E.g.: 'Several papers address X but none tackle Y. Work by Author (2024) partially covers Z by showing... However, the specific question of W remains open because no existing work provides...'",
    "sources": [
        {{"title": "Paper title", "url": "URL if available", "year": 2024, "relevance": "One sentence: how it relates"}}
    ]
}}

Status definitions:
- "open" = No work substantively addresses this gap (coverage_pct: 0-10)
- "partial" = Related work exists but does not fully address it (coverage_pct: 10-80)
- "addressed" = Existing work directly addresses this gap (coverage_pct: 80-100)
"""

    try:
        response = client.responses.create(
            model=GPT_MODEL,
            input=prompt,
            tools=[{"type": "web_search"}],
        )

        # Parse response
        output_text = response.output_text or ""

        # Robust JSON extraction (strip markdown fences, find balanced braces)
        cleaned = re.sub(r'```(?:json)?\s*', '', output_text)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        result_data = None

        # Try direct parse
        try:
            result_data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Find balanced JSON object
            start = cleaned.find('{')
            if start != -1:
                depth = 0
                for idx in range(start, len(cleaned)):
                    if cleaned[idx] == '{':
                        depth += 1
                    elif cleaned[idx] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                result_data = json.loads(cleaned[start:idx + 1])
                            except json.JSONDecodeError:
                                pass
                            break

        if result_data and isinstance(result_data, dict):
            raw_coverage = result_data.get("coverage_pct")
            coverage_pct = None
            if raw_coverage is not None:
                try:
                    coverage_pct = float(raw_coverage)
                except (TypeError, ValueError):
                    pass

            return ExternalValidationResult(
                status=result_data.get("status", "partial"),
                reasoning=result_data.get("reasoning", "Unable to parse reasoning"),
                sources=result_data.get("sources", []),
                coverage_pct=coverage_pct,
            )
        else:
            # Fallback: try to infer from text
            lower_text = output_text.lower()
            if "no work found" in lower_text or "not addressed" in lower_text:
                status = "open"
            elif "already addressed" in lower_text or "has been solved" in lower_text:
                status = "addressed"
            else:
                status = "partial"

            return ExternalValidationResult(
                status=status,
                reasoning=output_text[:500],
                sources=[],
            )

    except Exception as e:
        logger.exception(f"Error validating gap: {gap_title}")
        # On error, return partial (conservative)
        return ExternalValidationResult(
            status="partial",
            reasoning=f"Validation error: {str(e)}",
            sources=[],
        )


def calculate_confidence(
    detection_score: float,
    validation_result: ExternalValidationResult,
) -> float:
    """
    Calculate confidence for a gap.

    The formula weights validation status heavily:
    - "addressed" is near-fatal: the gap is likely not real
    - "partial" is neutral: some related work, not conclusive
    - "open" boosts confidence: validated as genuinely unaddressed
    """
    # Base confidence from detection (max 60%)
    base_confidence = min(detection_score * 0.6, 0.6)

    # Validation adjustment
    if validation_result.status == "open":
        confidence = base_confidence + 0.30
    elif validation_result.status == "partial":
        coverage = validation_result.coverage_pct
        if coverage is None:
            # Fallback: estimate from source count when GPT didn't return coverage_pct
            source_count = len(validation_result.sources or [])
            if source_count == 0:
                coverage = 10.0
            elif source_count <= 2:
                coverage = 30.0
            elif source_count <= 4:
                coverage = 50.0
            else:
                coverage = 70.0

        coverage = max(0.0, min(100.0, coverage))
        # Continuous: 0% coverage → +0.25, 100% coverage → +0.00
        partial_bonus = 0.25 * (1.0 - coverage / 100.0)
        confidence = base_confidence + partial_bonus
    else:  # "addressed"
        confidence = base_confidence * 0.2

    # Clamp to valid range
    return max(0.0, min(1.0, confidence))


def _validate_single(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single gap candidate. Used for parallel execution."""
    validation_result = validate_gap(
        gap_title=candidate.get("title", ""),
        gap_description=candidate.get("description", ""),
        suggested_direction=candidate.get("suggested_direction", ""),
        gap_type=candidate.get("type", "unknown"),
    )

    confidence = calculate_confidence(
        detection_score=candidate.get("detection_score", 0.5),
        validation_result=validation_result,
    )

    return {
        **candidate,
        "validation_result": validation_result,
        "confidence": confidence,
    }


def validate_gaps_batch(
    gap_candidates: List[Dict[str, Any]],
    min_confidence_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Validate a batch of gap candidates in parallel.

    Uses ThreadPoolExecutor to run up to MAX_VALIDATION_CONCURRENCY
    GPT-5.2 + web_search calls concurrently.
    """
    if not gap_candidates:
        return []

    validated_gaps = []

    logger.info(
        f"Validating {len(gap_candidates)} gaps "
        f"(concurrency={min(MAX_VALIDATION_CONCURRENCY, len(gap_candidates))})"
    )

    with ThreadPoolExecutor(max_workers=MAX_VALIDATION_CONCURRENCY) as executor:
        future_to_candidate = {
            executor.submit(_validate_single, candidate): candidate
            for candidate in gap_candidates
        }

        for future in as_completed(future_to_candidate):
            candidate = future_to_candidate[future]
            try:
                result = future.result()
                confidence = result["confidence"]

                if confidence >= min_confidence_threshold:
                    validated_gaps.append(result)
                    logger.info(
                        f"Gap '{candidate.get('title')}' validated: "
                        f"confidence={confidence:.2f}"
                    )
                else:
                    logger.info(
                        f"Gap '{candidate.get('title')}' below threshold "
                        f"({confidence:.2f} < {min_confidence_threshold})"
                    )
            except Exception as e:
                logger.exception(
                    f"Error validating gap '{candidate.get('title')}': {e}"
                )

    # Sort by confidence
    validated_gaps.sort(key=lambda x: x["confidence"], reverse=True)

    return validated_gaps
