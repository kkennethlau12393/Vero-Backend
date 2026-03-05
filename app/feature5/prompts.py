"""
LLM prompt templates for Feature 5: Research Gap Analysis.

Contains prompts for evaluating, filtering, and synthesizing gap
candidates into coherent gap descriptions with inline citations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.common.id_mapping import IdMapper

# Prompt for evaluating and synthesizing gap candidates into structured gap cards.
# The LLM acts as BOTH an evaluator (filtering noise) and a describer.
GAP_SYNTHESIS_PROMPT = """You are a critical research gap analyst. Your job has two parts:

1. **EVALUATE** each gap candidate: Is this a genuine research gap, or a heuristic artifact?
2. **DESCRIBE** only the genuine gaps as specific research questions.

## What Constitutes a Research Gap

A research gap is missing KNOWLEDGE or UNDERSTANDING — something we don't know, haven't proven, or haven't systematically studied.

A research gap is NOT:
- A project nobody has built (that's an engineering task)
- Two fields that could be "integrated" (that's the premise of the research area)
- A domain where a method hasn't been "applied" (that's an application idea)

A research gap IS:
- An assumption made by multiple papers that has never been tested
- A theoretical result that lacks empirical validation (or vice versa)
- A comparison that researchers would need but nobody has conducted
- A limitation acknowledged across papers but never addressed
- A phenomenon observed but not yet explained

Read the abstracts. Identify what the papers assume, what they acknowledge as limitations, and what questions they leave unanswered. Those unanswered questions are the gaps.

## CRITICAL: You MUST reject bad candidates

Our heuristic detectors give you BROAD signals (e.g., "these clusters have low citation density"). Do NOT just describe the broad signal. Instead, read the evidence papers' abstracts and identify the specific unanswered question that the signal points to.

REJECT candidates that are:

- **Nonsensical combinations**: A method-domain pairing that makes no practical sense
- **Peripheral topics**: A topic with few papers that is tangential to the research area
- **Normal field growth**: Early decades having fewer papers is NOT a gap
- **Mature subfields**: A well-established area does not automatically have a "novelty gap"
- **Circular reasoning**: "No pioneering work exists here" does not mean "pioneering work is needed here"
- **Already connected**: Clusters that are topically distinct and SHOULD be independent
- **Too broad**: "Integration of X and Y" or "applying X to Y" — these are field premises, not gaps

For each candidate, first decide: **ACCEPT** or **REJECT**. Only describe accepted candidates.

## Gap Candidates by Type

{candidates_by_type}

## Paper Reference Data

{paper_references}

## Instructions

For each ACCEPTED gap, create a structured description:

1. **Title**: Clear, concise (10-15 words max)
2. **Description**: 150-250 words with inline citations (Author, Year) explaining what the gap is, the evidence, and why it matters
3. **Why It Matters**: 2-4 sentences explaining why a researcher should pursue this direction. This is NOT a restatement of the gap type. It should answer: What new understanding would this unlock? What problems remain unsolvable without this knowledge? What would change in the field if this gap were filled? Frame it as motivation for a researcher considering this direction.
4. **Suggested Direction**: 2-3 specific, actionable sentences

## Output Format

Return a JSON array. For REJECTED candidates, include a brief rejection entry. For ACCEPTED candidates, include the full description:

```json
[
  {{
    "gap_id": "gap_1",
    "status": "accepted",
    "type": "structural|coverage|temporal|methodological|novelty",
    "title": "Clear gap title",
    "description": "Detailed description with inline citations (Author, Year). Be SPECIFIC about what each evidence paper assumes or leaves unanswered.",
    "why_it_matters": "2-4 sentences: What understanding would this unlock? Why should a researcher invest years pursuing this? What remains unsolvable without this knowledge?",
    "suggested_direction": "Specific research direction...",
    "evidence_work_ids": ["W123", "W456", "W789"],
    "data_sources_used": ["abstracts", "methodology_fingerprints"],
    "detection_score": 0.85
  }},
  {{
    "gap_id": "gap_2",
    "status": "rejected",
    "type": "methodological",
    "rejection_reason": "This method-domain combination is nonsensical..."
  }}
]
```

## CRITICAL: Valid Gap Types

You MUST use exactly one of these types: structural, coverage, temporal, methodological, novelty. Do NOT invent types like "theoretical", "empirical", "conceptual", or "integration".

## CRITICAL: Diversity Requirement

Each accepted gap MUST address a DIFFERENT aspect of the research area. If you find multiple gaps about the same underlying theme (e.g., three variations of "theory vs. practice"), you MUST merge them into ONE gap and reject the duplicates. Diverse gaps across different types are strongly preferred.

## Guidelines

- Be CRITICAL. Rejecting 30-70% of candidates is normal and expected.
- Only accept gaps that represent genuine, actionable research opportunities
- Merge similar gaps if they overlap significantly
- Use inline citations in (Author, Year) format — ONLY cite papers listed in evidence_work_ids
- Suggested directions must be concrete and specific
- A gap that passes your filter should be one a researcher would actually pursue

Return ONLY the JSON array, no additional text.
"""


def format_candidates_for_synthesis(
    candidates_by_type: Dict[str, List[Any]],
) -> str:
    """Format gap candidates for the synthesis prompt."""
    sections = []

    for gap_type, candidates in candidates_by_type.items():
        if not candidates:
            continue

        section = f"### {gap_type.upper()} GAPS\n\n"

        for i, c in enumerate(candidates[:10]):  # Limit to top 10 per type
            if gap_type == "structural":
                section += f"{i+1}. Clusters '{c.cluster_a_label}' and '{c.cluster_b_label}' have high topic similarity ({c.topic_similarity:.2f}) but low citation density ({c.citation_density:.2f}). Gap score: {c.gap_score:.2f}\n"
                section += f"   Representative papers A: {', '.join(c.representative_papers_a[:3])}\n"
                section += f"   Representative papers B: {', '.join(c.representative_papers_b[:3])}\n\n"

            elif gap_type == "coverage":
                section += f"{i+1}. Topic '{c.topic}' is underrepresented (coverage ratio: {c.coverage_ratio:.2f}). Gap score: {c.gap_score:.2f}\n"
                section += f"   Related papers: {', '.join(c.related_papers[:3])}\n\n"

            elif gap_type == "temporal":
                section += f"{i+1}. Period {c.period_label} has low publication rate ({c.publication_rate:.2f} vs avg {c.average_rate:.2f}). Gap score: {c.gap_score:.2f}\n"
                if c.is_comeback:
                    section += f"   Pattern: Activity-Stagnation-Activity (comeback pattern)\n\n"
                else:
                    section += "\n"

            elif gap_type == "methodological":
                section += f"{i+1}. Method '{c.method}' hasn't been applied to domain '{c.domain}'. Neighbor density: {c.neighbor_density:.2f}. Gap score: {c.gap_score:.2f}\n"
                section += f"   Papers using method: {', '.join(c.related_method_papers[:3])}\n"
                section += f"   Papers in domain: {', '.join(c.related_domain_papers[:3])}\n\n"

            elif gap_type == "novelty":
                section += f"{i+1}. Cluster '{c.cluster_label}' has low novelty (avg: {c.avg_novelty:.2f}, max: {c.max_novelty:.2f}). Gap score: {c.gap_score:.2f}\n"
                section += f"   Papers in cluster: {', '.join(c.papers_in_cluster[:3])}\n\n"

        sections.append(section)

    return "\n".join(sections) if sections else "No gap candidates detected."


def format_paper_references(
    paper_data: Dict[str, Dict[str, Any]],
) -> str:
    """Format paper reference data for the synthesis prompt."""
    lines = []

    for work_id, data in list(paper_data.items())[:50]:  # Limit to 50 papers
        authors = data.get("authors", "Unknown")
        if isinstance(authors, list):
            if len(authors) > 2:
                authors = f"{authors[0]} et al."
            else:
                authors = " & ".join(authors)

        year = data.get("year", "n.d.")
        title = data.get("title", "Untitled")[:80]

        line = f"- {work_id}: {authors} ({year}). {title}"
        if abstract := data.get("abstract"):
            line += f"\n  Abstract: {abstract[:500]}"
        lines.append(line)

    return "\n".join(lines) if lines else "No paper references available."


def build_synthesis_prompt(
    candidates_by_type: Dict[str, List[Any]],
    paper_data: Dict[str, Dict[str, Any]],
) -> str:
    """Build the complete synthesis prompt."""
    return GAP_SYNTHESIS_PROMPT.format(
        candidates_by_type=format_candidates_for_synthesis(candidates_by_type),
        paper_references=format_paper_references(paper_data),
    )


# Prompt for LLM-direct gap detection when heuristic detectors produce sparse results.
# This gives the LLM the full paper list and asks it to identify gaps directly.
LLM_DIRECT_GAP_PROMPT = """You are an expert research gap analyst. You have been given a collection of academic papers in a specific research area. Your job is to identify genuine, actionable research gaps based on what IS covered and what is MISSING.

## CRITICAL GROUNDING RULES

- ONLY identify gaps that are directly evidenced by the papers listed below
- DO NOT extrapolate to fields, domains, or applications not represented in the paper collection
- Every gap you identify MUST reference at least 4 specific work_ids from the list as evidence
- If a domain (e.g., "medical robotics") has ZERO papers in the list, it is NOT a gap — it is simply out of scope
- Gaps must be BETWEEN or WITHIN the topics covered by the actual papers, not about missing fields entirely

## What Constitutes a Research Gap

A research gap is missing KNOWLEDGE or UNDERSTANDING — something we don't know, haven't proven, or haven't systematically studied.

A research gap is NOT:
- A project nobody has built (that's an engineering task)
- Two fields that could be "integrated" (that's the premise of the research area)
- A domain where a method hasn't been "applied" (that's an application idea)

A research gap IS:
- An assumption made by multiple papers that has never been tested
- A theoretical result that lacks empirical validation (or vice versa)
- A comparison that researchers would need but nobody has conducted
- A limitation acknowledged across papers but never addressed
- A phenomenon observed but not yet explained

Read the abstracts. Identify what the papers assume, what they acknowledge as limitations, and what questions they leave unanswered. Those unanswered questions are the gaps.

## Research Area Papers

{paper_list}

## Citation Structure Summary

{citation_summary}

## Internal Analysis Data

{internal_context}

## Instructions

Analyze ALL the data above — papers, abstracts, methodology fingerprints, novelty assessments, ranking scores, citation structure, and cluster groupings — to identify 2-3 genuine research gaps. Prefer fewer, deeply-grounded gaps over many shallow ones.

Use the methodology fingerprints to understand what each paper DOES and ASSUMES. Use the novelty assessments to understand what's truly new vs. incremental. Use the abstracts to find acknowledged limitations and open questions. Use the cluster structure to find disconnects between subfields. Use timeline narratives to understand how the field evolved and where momentum stalled.

Consider:

1. **Structural gaps**: Clusters that share related goals but don't cite each other — what specific question connects them that nobody has asked?
2. **Coverage gaps**: Perspectives, assumptions, or conditions that papers in this area haven't tested — not just "more papers needed" but specific untested hypotheses
3. **Methodological gaps**: Limitations acknowledged in methodology fingerprints that no paper addresses — what assumptions remain unvalidated?
4. **Temporal gaps**: Questions that were raised in earlier papers but never followed up on
5. **Novelty gaps**: Areas where all papers are incremental — what foundational question would enable a breakthrough?
6. **Cross-disciplinary gaps** (use type "structural"): Where insights from one cluster/subfield could answer open questions in another — look for shared assumptions, complementary methods, or transferable theoretical frameworks that haven't been connected

## CRITICAL: Valid Gap Types

You MUST use exactly one of these types in the JSON: structural, coverage, temporal, methodological, novelty. Do NOT invent types like "theoretical", "empirical", "conceptual", "integration", or "cross-disciplinary". Map cross-disciplinary gaps to "structural".

## CRITICAL: Diversity Requirement

Each gap MUST address a DIFFERENT aspect of the research area. If you find multiple gaps about the same underlying theme (e.g., three variations of "theory vs. practice" or "RL bounds validation"), merge them into ONE gap. Diverse gaps across different types are strongly preferred. 3 diverse gaps > 5 variations on the same theme.

## Quality Standards

- Each gap must be ACTIONABLE — a researcher should be able to pursue it
- Reference specific papers from the list as evidence (use their short IDs like P1, P2)
- In the description, cite papers using (Author, Year) format. ONLY cite papers that are also listed in your evidence_work_ids — do NOT mention papers that aren't direct evidence for this specific gap
- Be specific about WHAT is missing, not just that "more research is needed"
- Consider whether apparent gaps are actually mature/completed areas
- Do NOT flag early decades having fewer papers as gaps — that is natural field maturation
- REJECT gaps that are just "integrate X with Y" or "apply method X to domain Y" — those are application ideas, not gaps in understanding
- A gap that passes your filter should be one a PhD researcher would write a thesis proposal around
- Use the internal analysis data to ground your gaps in specific evidence, not broad impressions

## Output Format

Return a JSON array:
```json
[
  {{
    "gap_id": "gap_1",
    "status": "accepted",
    "type": "structural|coverage|temporal|methodological|novelty",
    "title": "Clear, concise gap title (10-15 words max)",
    "description": "150-250 words with inline citations (Author, Year). Explain what the gap is, cite evidence from the papers above, and explain why it matters. Be SPECIFIC about what each evidence paper assumes or leaves unanswered.",
    "why_it_matters": "2-4 sentences answering: Why should a researcher pursue this direction? What new understanding would filling this gap unlock? What problems remain unsolvable without this knowledge? What would change in the field if this were addressed? This should validate a researcher's motivation for going in this direction — not just describe the gap type.",
    "suggested_direction": "2-3 specific, actionable sentences describing what research would fill this gap.",
    "evidence_work_ids": ["P1", "P4", "P7", "P12"],
    "data_sources_used": ["abstracts", "methodology_fingerprints", "citation_structure"],
    "detection_score": 0.75
  }}
]
```

For `data_sources_used`, list which internal data you actually relied on. Valid values: "abstracts", "methodology_fingerprints", "novelty_assessments", "citation_structure", "cluster_structure", "ranking_scores", "paper_summaries", "publication_timeline", "timeline_narratives", "methodology_comparisons", "full_text".

Return ONLY the JSON array, no additional text.
"""


def format_paper_list_for_direct_detection(
    paper_data: Dict[str, Dict[str, Any]],
    mapper: Optional[IdMapper] = None,
) -> str:
    """Format paper data for the LLM-direct gap detection prompt."""
    lines = []

    for work_id, data in list(paper_data.items())[:50]:
        short_id = mapper.add(work_id) if mapper else work_id
        authors = data.get("authors", "Unknown")
        if isinstance(authors, list):
            if len(authors) > 2:
                authors = f"{authors[0]} et al."
            else:
                authors = " & ".join(authors)

        year = data.get("year", "n.d.")
        title = data.get("title", "Untitled")
        cited_by = data.get("cited_by_count", 0) or 0
        topic = data.get("topic_name", "")

        line = f"- {short_id}: {authors} ({year}). \"{title}\""
        if cited_by:
            line += f" [cited {cited_by}x]"
        if topic:
            line += f" [topic: {topic}]"
        if abstract := data.get("abstract"):
            line += f"\n  Abstract: {abstract[:500]}"
        lines.append(line)

    return "\n".join(lines) if lines else "No papers available."


def build_direct_detection_prompt(
    paper_data: Dict[str, Dict[str, Any]],
    citation_summary: str,
    internal_context: str = "",
    mapper: Optional[IdMapper] = None,
) -> tuple[str, IdMapper]:
    """Build the LLM-direct gap detection prompt. Returns (prompt, mapper)."""
    if mapper is None:
        mapper = IdMapper("P")
    prompt = LLM_DIRECT_GAP_PROMPT.format(
        paper_list=format_paper_list_for_direct_detection(paper_data, mapper),
        citation_summary=citation_summary,
        internal_context=internal_context or "No additional internal data available.",
    )
    return prompt, mapper


# Prompt for generating evidence roles
EVIDENCE_ROLE_PROMPT = """Given a research gap and a list of papers with abstracts, explain each paper's specific role in evidencing this gap.

Gap: {gap_title}
Gap Type: {gap_type}

Papers:
{papers}

For each paper, read its abstract and write a brief role (8-15 words) explaining what SPECIFIC assumption, limitation, or finding in this paper relates to the gap. Reference concrete details from the abstract, not generic descriptions.

BAD roles (too generic): "Demonstrates the methodological approach", "Supporting evidence"
GOOD roles: "Assumes skip connections improve gradient flow without testing feature redundancy", "Acknowledges scale sensitivity as limitation but offers no solution"

Return as JSON:
{{
  "P1": "Role description grounded in paper's abstract",
  "P2": "Role description grounded in paper's abstract"
}}
"""
