"""
Topic Brief Generator — produces a cited overview with dynamic sections
for F1/F2 results. Uses GPT-5 mini via OpenAI API.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-5-mini"


def get_openai_client() -> OpenAI:
    """Get OpenAI client (reuses OPENAI_API_KEY from env)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for topic brief")
    return OpenAI(api_key=api_key)


def generate_topic_brief(
    query: str,
    papers: List[Dict[str, Any]],
    feature: str = "ranked_list",
    max_papers: int = 15,
) -> Optional[Dict[str, Any]]:
    """
    Generate a cited topic brief with dynamically titled sections.

    Returns
    -------
    dict with keys:
        - overview (str): Main synthesis paragraph with [N] citations
        - sections (list[dict]): Two sections, each with:
            - title (str): Dynamic, query-specific section title
            - content (str): 2-4 sentence paragraph with [N] citations
        - citations (list[dict]): Ordered list of {index, work_id, title, authors, year}
    """
    if not papers:
        return None

    top_papers = papers[:max_papers]

    # Build paper context for prompt
    paper_context = []
    for i, p in enumerate(top_papers, 1):
        authors = p.get("authors", "Unknown")
        if isinstance(authors, list):
            authors = ", ".join(authors[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
        abstract = p.get("abstract", "")
        if abstract and len(abstract) > 500:
            abstract = abstract[:500] + "..."

        paper_context.append(
            f"[{i}] {p.get('title', 'Untitled')} ({p.get('year', 'n.d.')})\n"
            f"    Authors: {authors}\n"
            f"    Abstract: {abstract or 'N/A'}"
        )

    papers_text = "\n\n".join(paper_context)

    if feature == "citation_map":
        feature_context = (
            "The user is exploring a citation map — a network of papers connected by citations. "
            "Frame the brief as an introduction to the research landscape around their query."
        )
    else:
        feature_context = (
            "The user is viewing a ranked list of the most relevant papers for their query. "
            "Frame the brief as a synthesis of what these top papers collectively reveal about the topic."
        )

    prompt = f"""You are an expert research synthesizer. Given a user's research query and the top {len(top_papers)} papers retrieved, write a structured topic brief.

{feature_context}

USER QUERY: {query}

PAPERS:
{papers_text}

Write exactly THREE parts. Use inline citations like [1], [2], etc. referencing the paper numbers above. Every factual claim must be cited. Cite MULTIPLE papers where relevant (e.g., [1, 3, 7]).

Respond in this exact JSON format:
{{
    "overview": "A 4-6 sentence paragraph giving a high-level synthesis of this research area. What is it about? Why does it matter? What are the main threads? What has been achieved? Must include citations.",
    "sections": [
        {{
            "title": "A short, specific, dynamic title that captures the core technical theme of these papers (e.g., 'Cas9 Engineering & Guide RNA Design' not 'Key Concepts')",
            "content": "A 3-4 sentence paragraph exploring this theme in depth. Reference specific methods, findings, and frameworks from the papers. Must include citations."
        }},
        {{
            "title": "A short, specific, dynamic title that captures where the field is heading (e.g., 'Beyond Cas9: Base Editing & Prime Editing Frontiers' not 'Current State')",
            "content": "A 3-4 sentence paragraph on recent developments, active debates, emerging techniques, or open challenges. Must include citations."
        }}
    ]
}}

Rules:
- The overview should be substantive — a full paragraph that someone could read and understand the field
- Section titles MUST be specific to the query topic — never use generic titles like "Key Concepts", "Current State", "Recent Developments", "Core Methodologies", or "Future Directions"
- Good title examples: "Diffusion Model Architectures & Sampling Strategies", "From Supervised to Self-Supervised: The Pretraining Paradigm Shift", "Immunotherapy Resistance Mechanisms & Combination Strategies"
- Bad title examples: "Key Technical Concepts", "Current State of the Field", "Recent Advances"
- Write in clear, authoritative academic prose
- Every sentence must cite at least one paper
- Use [N] format for citations, where N matches the paper number
- Be specific — reference actual findings, methods, and results from the papers
- Do NOT use generic filler phrases like "various studies have shown"
- Keep each section concise but dense with insight"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a research synthesis expert. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1500,
        )

        content = response.choices[0].message.content
        brief_data = json.loads(content)

        # Build citations list
        citations = []
        for i, p in enumerate(top_papers, 1):
            citations.append({
                "index": i,
                "work_id": p.get("work_id", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", ""),
                "year": p.get("year", None),
            })

        # Validate sections
        sections = brief_data.get("sections", [])
        if not isinstance(sections, list) or len(sections) < 2:
            sections = [
                {"title": "Key Themes", "content": sections[0].get("content", "") if sections else ""},
                {"title": "Recent Developments", "content": sections[1].get("content", "") if len(sections) > 1 else ""},
            ]

        return {
            "overview": brief_data.get("overview", ""),
            "sections": sections[:2],
            "citations": citations,
        }

    except Exception as e:
        logger.error(f"Topic brief generation failed: {e}")
        return None
