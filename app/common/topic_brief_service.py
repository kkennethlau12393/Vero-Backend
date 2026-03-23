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
        - sections (list[dict]): Three sections, each with:
            - title (str): Dynamic, query-specific section title
            - content (str): 2-4 sentence paragraph with [N] citations
        - citations (list[dict]): Ordered list of {index, work_id, title, authors, year}
    """
    if not papers:
        return None

    top_papers = [p for p in papers[:max_papers] if p.get("abstract") and len(p.get("abstract", "")) > 50]
    if len(top_papers) < 3:
        logger.info(f"Topic brief skipped: only {len(top_papers)} papers with abstracts")
        return None

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

    prompt = f"""You are an expert research synthesizer writing a review of a research area. Given a user's query and the top {len(top_papers)} papers, write a structured topic brief.

{feature_context}

USER QUERY: {query}

PAPERS:
{papers_text}

Write exactly FOUR parts: one overview and three sections. Use inline citations like [1], [2], etc. referencing the paper numbers above. Every factual claim must be cited.

Respond in this exact JSON format:
{{
    "overview": "A 5-7 sentence paragraph synthesizing what this research area is about, why it matters, and the major threads of work. Write with authority as if introducing the field to an informed reader.",
    "sections": [
        {{
            "title": "A short, specific title capturing the core technical methods and concepts (e.g., 'Cas9 Engineering & Guide RNA Optimization' not 'Key Concepts')",
            "content": "A 5-6 sentence paragraph exploring the foundational methods, architectures, or frameworks. Reference specific techniques, formulations, and findings."
        }},
        {{
            "title": "A short, specific title capturing recent results and emerging techniques (e.g., 'Base Editing & Prime Editing Frontiers' not 'Recent Developments')",
            "content": "A 5-6 sentence paragraph on key results, breakthroughs, and newer approaches. Reference specific experimental outcomes, performance gains, or novel formulations."
        }},
        {{
            "title": "A short, specific title capturing open problems and future directions (e.g., 'Off-Target Effects & In Vivo Delivery Barriers' not 'Challenges')",
            "content": "A 5-6 sentence paragraph on active debates, unresolved challenges, and promising directions. Reference specific limitations identified and proposed solutions."
        }}
    ]
}}

Rules:
- Write as if authoring a review article, not commenting on a paper collection. Never say "the papers", "this set", "the corpus", "a subset of papers", "across the collection", "the top papers", "empirical comparisons in the corpus", or any similar meta-commentary. The reader should feel they are learning about the topic, not reading an analysis of a reading list.
- Weave citations naturally into claims about the field. Good: "Message-passing networks iteratively update node features by aggregating neighborhood information [1, 3], with attention-weighted variants improving performance on relational tasks [4]." Bad: "Several papers in this set develop message-passing formulations [1, 3, 4]."
- Section titles MUST be specific to the query topic — never use generic titles like "Key Concepts", "Current State", "Recent Developments", "Core Methodologies", "Challenges", or "Future Directions"
- NEVER cite more than 4 papers in a single claim — each citation should add distinct information
- Only cite a paper when you reference something specific from it
- Each citation should add information — don't cite papers that say the same thing as ones already cited
- Be specific — reference actual methods, findings, metrics, and results
- Do NOT use generic filler phrases like "various studies have shown" or "research has demonstrated"
- Every sentence must include at least one citation"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a research synthesis expert. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=16000,
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
        if not isinstance(sections, list) or len(sections) < 3:
            # Pad to 3 if LLM returned fewer
            while len(sections) < 3:
                sections.append({"title": "", "content": ""})

        return {
            "overview": brief_data.get("overview", ""),
            "sections": sections[:3],
            "citations": citations,
        }

    except Exception as e:
        logger.error(f"Topic brief generation failed: {e}")
        return None
