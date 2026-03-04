"""
Gap analysis service for Feature 5: Research Gap Analysis.

Main orchestration module that coordinates gap detection, LLM evaluation/synthesis,
and external validation to produce comprehensive research gap analysis.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.feature5.coverage_tracker import (
    get_available_data_sources,
    get_gap_analysis_status,
)
from app.feature5.external_validation import validate_gaps_batch
from app.feature5.prompts import (
    EVIDENCE_ROLE_PROMPT,
    build_direct_detection_prompt,
)
from app.feature5.schemas import (
    Evidence,
    ExternalValidationResult,
    GAP_TYPE_EXPLANATIONS,
    GapAnalysisJobStatus,
    GapAnalysisResponse,
    GapCard,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Groq for fast LLM synthesis
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3-32b"
GROQ_MAX_RETRIES = 3
GROQ_RETRY_BACKOFF = 2.0  # seconds, doubles each retry

MIN_CONFIDENCE_THRESHOLD = 0.50
MIN_DETECTION_SCORE = 0.40  # Filter out weak heuristic candidates before validation
MAX_COVERAGE_PCT = 60.0  # Reject gaps where GPT says >60% is already addressed

# Cache TTL: results older than this are considered stale
CACHE_TTL_HOURS = 24


def get_groq_client() -> OpenAI:
    """Get Groq client for LLM synthesis."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def extract_json_from_llm(content: str, expect_array: bool = True) -> Any:
    """
    Robustly extract JSON from LLM response text.

    Handles markdown code fences, trailing text, and other common LLM quirks.
    """
    # Strip <think>...</think> tags from thinking models (e.g. Qwen3-32B)
    # Handle both closed tags and unclosed/truncated tags
    content = re.sub(r'<think>[\s\S]*?</think>', '', content)
    content = re.sub(r'<think>[\s\S]*$', '', content)  # Truncated think block
    # Strip markdown code fences
    content = re.sub(r'```(?:json)?\s*', '', content)
    content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
    content = content.strip()

    # Try direct parse first
    try:
        parsed = json.loads(content)
        if expect_array and isinstance(parsed, list):
            return parsed
        if not expect_array and isinstance(parsed, dict):
            return parsed
        return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON structure in response
    if expect_array:
        # Find outermost balanced array
        start = content.find('[')
        if start != -1:
            depth = 0
            for i in range(start, len(content)):
                if content[i] == '[':
                    depth += 1
                elif content[i] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[start:i + 1])
                        except json.JSONDecodeError:
                            break
    else:
        start = content.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[start:i + 1])
                        except json.JSONDecodeError:
                            break

    logger.warning("Failed to extract JSON from LLM response")
    return [] if expect_array else {}


def _parse_authors(authors_json: Any) -> List[str]:
    """Parse authors from various JSON formats into a list of name strings."""
    authors = authors_json or []
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except (json.JSONDecodeError, TypeError):
            authors = []

    author_names = []
    for a in authors[:3]:
        if isinstance(a, dict):
            author_names.append(
                a.get("name", a.get("author", {}).get("display_name", "Unknown"))
            )
        elif isinstance(a, str):
            author_names.append(a)

    return author_names


def get_paper_data(
    conn: Connection,
    work_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Get paper metadata for a list of work IDs."""
    if not work_ids:
        return {}

    result = conn.execute(
        text("""
            SELECT
                work_id, title, year, cited_by_count, authors_json, venue, abstract
            FROM works
            WHERE work_id = ANY(:work_ids)
        """),
        {"work_ids": work_ids},
    ).mappings().all()

    paper_data = {}
    for row in result:
        paper_data[row["work_id"]] = {
            "title": row["title"],
            "year": row["year"],
            "cited_by_count": row["cited_by_count"],
            "authors": _parse_authors(row["authors_json"]),
            "venue": row["venue"],
            "abstract": row["abstract"],
        }

    return paper_data


def _backfill_authors_from_openalex(
    conn: Connection,
    work_ids: List[str],
) -> Dict[str, List[str]]:
    """
    Fetch author names from OpenAlex for works with missing authors.

    Updates the DB so subsequent calls don't need to re-fetch.
    Returns {work_id: [author_names]} for the backfilled works.
    """
    import requests

    if not work_ids:
        return {}

    backfilled = {}
    BATCH_SIZE = 50

    for i in range(0, len(work_ids), BATCH_SIZE):
        batch = work_ids[i:i + BATCH_SIZE]
        ids_param = "|".join(f"https://openalex.org/{wid}" for wid in batch)
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"openalex:{ids_param}",
            "per-page": len(batch),
            "select": "id,authorships",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"OpenAlex author backfill failed: {e}")
            continue

        for w in resp.json().get("results", []):
            wid_full = w.get("id", "")
            wid = wid_full.rsplit("/", 1)[-1] if "/" in wid_full else wid_full
            authors = [
                au.get("author", {}).get("display_name") or au.get("display_name")
                for au in w.get("authorships", [])
                if au.get("author", {}).get("display_name") or au.get("display_name")
            ]
            if authors:
                backfilled[wid] = authors
                conn.execute(
                    text("UPDATE works SET authors_json = :authors WHERE work_id = :wid"),
                    {"authors": json.dumps(authors), "wid": wid},
                )

    if backfilled:
        conn.commit()
        logger.info(f"Backfilled authors for {len(backfilled)} papers from OpenAlex")

    return backfilled


def get_all_map_paper_data(
    conn: Connection,
    map_id: UUID,
) -> Dict[str, Dict[str, Any]]:
    """
    Get paper metadata for ALL papers in a map.

    Used by LLM-direct gap detection which needs the full paper list.
    Includes topic names resolved from openalex_topics.
    """
    result = conn.execute(
        text("""
            SELECT
                w.work_id, w.title, w.year, w.cited_by_count, w.authors_json,
                w.venue, w.abstract, w.primary_topic_id,
                ot.display_name AS topic_name
            FROM map_nodes mn
            JOIN works w ON w.work_id = mn.work_id
            LEFT JOIN openalex_topics ot ON ot.topic_id = w.primary_topic_id
            WHERE mn.map_id = :map_id
        """),
        {"map_id": map_id},
    ).mappings().all()

    # Identify papers missing authors for backfill
    missing_authors = []
    paper_data = {}
    for row in result:
        authors = _parse_authors(row["authors_json"])
        if not authors:
            missing_authors.append(row["work_id"])
        paper_data[row["work_id"]] = {
            "title": row["title"],
            "year": row["year"],
            "cited_by_count": row["cited_by_count"],
            "authors": authors,
            "venue": row["venue"],
            "abstract": row["abstract"],
            "topic_name": row["topic_name"] or "",
        }

    # Backfill missing authors from OpenAlex
    if missing_authors:
        backfilled = _backfill_authors_from_openalex(conn, missing_authors)
        for wid, authors in backfilled.items():
            if wid in paper_data:
                paper_data[wid]["authors"] = authors

    return paper_data


def get_all_rank_paper_data(
    conn: Connection,
    rank_job_id: UUID,
) -> Dict[str, Dict[str, Any]]:
    """
    Get paper metadata for ALL papers in a rank job's results.

    Same shape as get_all_map_paper_data() but queries rank_results.
    """
    result = conn.execute(
        text("""
            SELECT
                w.work_id, w.title, w.year, w.cited_by_count, w.authors_json,
                w.venue, w.abstract, w.primary_topic_id,
                ot.display_name AS topic_name
            FROM rank_results rr
            JOIN works w ON w.work_id = rr.work_id
            LEFT JOIN openalex_topics ot ON ot.topic_id = w.primary_topic_id
            WHERE rr.rank_job_id = :rjid
        """),
        {"rjid": rank_job_id},
    ).mappings().all()

    missing_authors = []
    paper_data = {}
    for row in result:
        authors = _parse_authors(row["authors_json"])
        if not authors:
            missing_authors.append(row["work_id"])
        paper_data[row["work_id"]] = {
            "title": row["title"],
            "year": row["year"],
            "cited_by_count": row["cited_by_count"],
            "authors": authors,
            "venue": row["venue"],
            "abstract": row["abstract"],
            "topic_name": row["topic_name"] or "",
        }

    if missing_authors:
        backfilled = _backfill_authors_from_openalex(conn, missing_authors)
        for wid, authors in backfilled.items():
            if wid in paper_data:
                paper_data[wid]["authors"] = authors

    return paper_data



def _gather_internal_context(
    conn: Connection,
    map_id: Optional[UUID],
    paper_data: Dict[str, Dict[str, Any]],
) -> str:
    """
    Gather rich internal data from methodology fingerprints, novelty
    assessments, and cluster structure to supplement the LLM-direct prompt.
    """
    work_ids = list(paper_data.keys())
    if not work_ids:
        return ""

    sections = []

    # --- Methodology fingerprints ---
    try:
        method_rows = conn.execute(
            text("""
                SELECT work_id, fingerprint_json
                FROM methodology_fingerprint_cache
                WHERE work_id = ANY(:wids)
                  AND fingerprint_json IS NOT NULL
            """),
            {"wids": work_ids},
        ).mappings().all()

        if method_rows:
            method_lines = ["### Methodology Fingerprints\n"]
            for r in method_rows:
                fp = r["fingerprint_json"]
                if isinstance(fp, str):
                    fp = json.loads(fp)
                approach = fp.get("approach", "")[:150]
                domain = fp.get("domain", "")
                assumptions = fp.get("assumptions", [])
                method_type = fp.get("methodology_type", "")
                line = f"- {r['work_id']}: [{method_type}] {approach}"
                if domain:
                    line += f" | Domain: {domain}"
                if assumptions:
                    line += f" | Assumptions: {'; '.join(str(a) for a in assumptions[:3])}"
                method_lines.append(line)
            sections.append("\n".join(method_lines))
    except Exception as e:
        logger.debug(f"Could not load methodology fingerprints: {e}")

    # --- Novelty assessments (persistent table) ---
    try:
        novelty_rows = conn.execute(
            text("""
                SELECT work_id, novelty_level, whats_new,
                       compared_to_prior_work, novelty_explanation
                FROM novelty_assessments
                WHERE work_id = ANY(:wids)
            """),
            {"wids": work_ids},
        ).mappings().all()

        if novelty_rows:
            novelty_lines = ["### Novelty Assessments\n"]
            for r in novelty_rows:
                level = r["novelty_level"] or "unknown"
                whats_new = (r["whats_new"] or "")[:120]
                compared_to = (r["compared_to_prior_work"] or "")[:120]
                explanation = (r["novelty_explanation"] or "")[:200]
                line = f"- {r['work_id']}: [{level}] {whats_new}"
                if compared_to:
                    line += f" (vs prior: {compared_to})"
                if explanation:
                    line += f" | {explanation}"
                novelty_lines.append(line)
            sections.append("\n".join(novelty_lines))
    except Exception as e:
        logger.debug(f"Could not load novelty assessments: {e}")

    # --- Cluster structure (topic-based grouping) ---
    try:
        if map_id:
            topic_rows = conn.execute(
                text("""
                    SELECT mn.work_id, w.primary_topic_id, ot.display_name
                    FROM map_nodes mn
                    JOIN works w ON w.work_id = mn.work_id
                    LEFT JOIN openalex_topics ot ON ot.topic_id = w.primary_topic_id
                    WHERE mn.map_id = :map_id
                """),
                {"map_id": map_id},
            ).mappings().all()
        else:
            topic_rows = conn.execute(
                text("""
                    SELECT w.work_id, w.primary_topic_id, ot.display_name
                    FROM works w
                    LEFT JOIN openalex_topics ot ON ot.topic_id = w.primary_topic_id
                    WHERE w.work_id = ANY(:wids)
                """),
                {"wids": work_ids},
            ).mappings().all()

        if topic_rows:
            clusters: Dict[str, List[str]] = {}
            topic_names: Dict[str, str] = {}
            for r in topic_rows:
                tid = r["primary_topic_id"] or "uncategorized"
                clusters.setdefault(tid, []).append(r["work_id"])
                if r["display_name"]:
                    topic_names[tid] = r["display_name"]

            cluster_lines = ["### Research Clusters (by primary topic)\n"]
            for tid, wids in sorted(clusters.items(), key=lambda x: -len(x[1])):
                name = topic_names.get(tid, tid)
                cluster_lines.append(
                    f"- {name} ({len(wids)} papers): {', '.join(wids[:5])}"
                    + (f" + {len(wids) - 5} more" if len(wids) > 5 else "")
                )
            sections.append("\n".join(cluster_lines))
    except Exception as e:
        logger.debug(f"Could not load cluster structure: {e}")

    # --- Ranked list scores (if map was built from Feature 2) ---
    try:
        rank_rows = conn.execute(
            text("""
                SELECT rr.work_id, rr.score, rr.score_breakdown_json,
                       rr.reasons_json
                FROM rank_results rr
                WHERE rr.work_id = ANY(:wids)
                ORDER BY rr.score DESC
                LIMIT 50
            """),
            {"wids": work_ids},
        ).mappings().all()

        if rank_rows:
            rank_lines = ["### Ranking Scores (relevance to research area)\n"]
            for r in rank_rows:
                breakdown = r["score_breakdown_json"]
                if isinstance(breakdown, str):
                    breakdown = json.loads(breakdown)
                norm = breakdown.get("norm", {}) if breakdown else {}
                llm_rel = norm.get("llm_relevance", "?")
                impact = norm.get("impact", "?")
                reasons = r["reasons_json"]
                if isinstance(reasons, str):
                    reasons = json.loads(reasons)
                reason_str = reasons[0][:100] if reasons else ""
                rank_lines.append(
                    f"- {r['work_id']}: score={r['score']:.2f} "
                    f"(relevance={llm_rel}, impact={impact}) {reason_str}"
                )
            sections.append("\n".join(rank_lines))
    except Exception as e:
        logger.debug(f"Could not load ranking scores: {e}")

    # --- Node summaries and keywords ---
    try:
        detail_rows = conn.execute(
            text("""
                SELECT work_id, summary, keywords
                FROM node_details_cache
                WHERE work_id = ANY(:wids)
                  AND summary IS NOT NULL
            """),
            {"wids": work_ids},
        ).mappings().all()

        if detail_rows:
            detail_lines = ["### Paper Summaries\n"]
            for r in detail_rows:
                summary = (r["summary"] or "")[:200]
                keywords = r["keywords"]
                if isinstance(keywords, str):
                    keywords = json.loads(keywords)
                kw_str = ", ".join(keywords[:5]) if keywords else ""
                line = f"- {r['work_id']}: {summary}"
                if kw_str:
                    line += f" [Keywords: {kw_str}]"
                detail_lines.append(line)
            sections.append("\n".join(detail_lines))
    except Exception as e:
        logger.debug(f"Could not load node summaries: {e}")

    # --- Timeline narratives (persistent) ---
    try:
        timeline_rows = conn.execute(
            text("""
                SELECT work_id, result
                FROM node_timelines
                WHERE work_id = ANY(:wids)
            """),
            {"wids": work_ids},
        ).mappings().all()

        if timeline_rows:
            tl_lines = ["### Timeline Narratives\n"]
            for r in timeline_rows:
                result = r["result"]
                if isinstance(result, str):
                    result = json.loads(result)
                narrative = result.get("narrative") or {}
                hist = (narrative.get("historical_context") or "")[:200]
                contrib = (narrative.get("contribution_statement") or "")[:150]
                downstream = (narrative.get("downstream_impact") or "")[:150]
                era_coms = narrative.get("era_commentaries") or []

                parts = [f"- {r['work_id']}:"]
                if hist:
                    parts.append(f"  Context: {hist}")
                if contrib:
                    parts.append(f"  Contribution: {contrib}")
                if downstream:
                    parts.append(f"  Impact: {downstream}")
                for ec in era_coms[:4]:
                    headline = ec.get("headline", "")
                    era_narr = (ec.get("narrative") or "")[:120]
                    parts.append(f"  [{ec.get('era', '?')}] {headline}: {era_narr}")
                tl_lines.append("\n".join(parts))
            sections.append("\n".join(tl_lines))
    except Exception as e:
        logger.debug(f"Could not load timeline narratives: {e}")

    # --- Full text for top-4 ranked papers ---
    try:
        # Get top 4 by rank score (fall back to citation count if no rank data)
        top_ranked = conn.execute(
            text("""
                SELECT work_id FROM rank_results
                WHERE work_id = ANY(:wids)
                ORDER BY score DESC
                LIMIT 4
            """),
            {"wids": work_ids},
        ).scalars().all()

        # Fall back to citation count if no rank results
        if not top_ranked:
            top_ranked = sorted(
                work_ids,
                key=lambda w: paper_data.get(w, {}).get("cited_by_count", 0),
                reverse=True,
            )[:4]

        if top_ranked:
            ft_rows = conn.execute(
                text("""
                    SELECT work_id, methods_text, full_text_available
                    FROM paper_full_text_cache
                    WHERE work_id = ANY(:wids)
                      AND methods_text IS NOT NULL
                """),
                {"wids": list(top_ranked)},
            ).mappings().all()

            if ft_rows:
                ft_lines = ["### Paper Full Text (top-4 ranked papers)\n"]
                for r in ft_rows:
                    source = "full text" if r["full_text_available"] else "abstract"
                    content = (r["methods_text"] or "")[:3000]
                    if content:
                        title = paper_data.get(r["work_id"], {}).get("title", "?")[:80]
                        ft_lines.append(
                            f"- {r['work_id']} ({title}) [{source}]:\n{content}"
                        )
                sections.append("\n".join(ft_lines))
    except Exception as e:
        logger.debug(f"Could not load paper full text: {e}")

    # --- Methodology comparisons (persistent) ---
    try:
        meth_rows = conn.execute(
            text("""
                SELECT work_ids, result
                FROM methodology_comparisons
                WHERE work_ids && :wids
                ORDER BY created_at DESC
                LIMIT 10
            """),
            {"wids": work_ids},
        ).mappings().all()

        if meth_rows:
            meth_lines = ["### Methodology Comparisons\n"]
            for r in meth_rows:
                result = r["result"]
                if isinstance(result, str):
                    result = json.loads(result)
                compared_wids = r["work_ids"] or []

                # Extract key insights
                conv_div = result.get("convergence_divergence") or {}
                convergences = conv_div.get("convergences") or []
                divergences = conv_div.get("divergences") or []
                rec = result.get("recommendation") or {}
                rec_text = (rec.get("summary") or rec.get("recommendation") or "")[:200]

                parts = [f"- Compared: {', '.join(compared_wids)}"]
                if convergences:
                    conv_strs = [c if isinstance(c, str) else (c.get("description") or "")[:80] for c in convergences[:3]]
                    parts.append(f"  Convergences: {'; '.join(conv_strs)}")
                if divergences:
                    div_strs = [d if isinstance(d, str) else (d.get("description") or "")[:80] for d in divergences[:3]]
                    parts.append(f"  Divergences: {'; '.join(div_strs)}")
                if rec_text:
                    parts.append(f"  Recommendation: {rec_text}")
                meth_lines.append("\n".join(parts))
            sections.append("\n".join(meth_lines))
    except Exception as e:
        logger.debug(f"Could not load methodology comparisons: {e}")

    # --- Year distribution ---
    year_counts: Dict[int, int] = {}
    for pd in paper_data.values():
        yr = pd.get("year")
        if yr:
            year_counts[yr] = year_counts.get(yr, 0) + 1

    if year_counts:
        timeline_lines = ["### Publication Timeline\n"]
        for yr in sorted(year_counts):
            bar = "#" * year_counts[yr]
            timeline_lines.append(f"- {yr}: {bar} ({year_counts[yr]} papers)")
        sections.append("\n".join(timeline_lines))

    return "\n\n".join(sections) if sections else ""


def detect_gaps_with_llm(
    conn: Connection,
    map_id: Optional[UUID],
    paper_data: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    LLM-direct gap detection fallback.

    When heuristic detectors produce sparse results (<3 candidates),
    directly ask the LLM to identify research gaps from the paper list.
    Supplements the prompt with rich internal data (methodology fingerprints,
    novelty assessments, cluster structure).

    Args:
        conn: Database connection
        map_id: Map ID (None for rank-based analysis)
        paper_data: Paper metadata keyed by work_id
    """
    client = get_groq_client()

    node_count = len(paper_data)

    # Build citation summary from edges (only for map-based)
    edge_count = 0
    in_degree: Dict[str, int] = {}
    out_degree: Dict[str, int] = {}

    if map_id:
        edges = conn.execute(
            text("""
                SELECT from_work_id, to_work_id
                FROM map_edges
                WHERE map_id = :map_id
            """),
            {"map_id": map_id},
        ).mappings().all()
        edge_count = len(edges)

        for e in edges:
            out_degree[e["from_work_id"]] = out_degree.get(e["from_work_id"], 0) + 1
            in_degree[e["to_work_id"]] = in_degree.get(e["to_work_id"], 0) + 1

    # Find bridge papers (cited by many, cite many) and isolated papers
    bridge_papers = []
    isolated_papers = []
    for wid in paper_data:
        i = in_degree.get(wid, 0)
        o = out_degree.get(wid, 0)
        title = paper_data[wid].get("title", "?")[:60]
        if i >= 3 and o >= 2:
            bridge_papers.append(f"{wid} ({title}) [in:{i}, out:{o}]")
        elif i == 0 and o == 0 and map_id:
            isolated_papers.append(f"{wid} ({title})")

    citation_lines = [
        f"Total papers: {node_count}, Total citation links: {edge_count}",
    ]
    if bridge_papers:
        citation_lines.append(f"Highly connected papers: {', '.join(bridge_papers[:5])}")
    if isolated_papers:
        citation_lines.append(f"Isolated papers (no citations): {', '.join(isolated_papers[:5])}")

    citation_summary = "\n".join(citation_lines)

    # Gather rich internal context
    internal_context = _gather_internal_context(conn, map_id, paper_data)

    prompt = build_direct_detection_prompt(paper_data, citation_summary, internal_context)

    for attempt in range(GROQ_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert research gap analyst. Analyze the "
                            "paper collection and identify genuine, actionable "
                            "research gaps. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=6000,
            )

            content = response.choices[0].message.content or "[]"
            gaps = extract_json_from_llm(content, expect_array=True)

            if not isinstance(gaps, list):
                logger.warning("LLM direct detection returned non-list")
                return []

            # Filter to accepted gaps
            accepted = [g for g in gaps if g.get("status") != "rejected"]
            logger.info(
                f"LLM direct detection: {len(accepted)} gaps identified "
                f"from {node_count} papers"
            )
            return accepted

        except Exception as e:
            if attempt < GROQ_MAX_RETRIES - 1:
                wait = GROQ_RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"Groq API error in direct detection "
                    f"(attempt {attempt + 1}/{GROQ_MAX_RETRIES}), "
                    f"retrying in {wait:.1f}s: {e}"
                )
                time.sleep(wait)
            else:
                logger.exception("Error in LLM direct detection after all retries")
                return []

    return []


def collect_work_ids_from_candidates(
    candidates_by_type: Dict[str, List[Any]],
) -> List[str]:
    """Collect all work IDs referenced in gap candidates."""
    work_ids = set()

    for gap_type, candidates in candidates_by_type.items():
        for c in candidates:
            if hasattr(c, 'representative_papers_a'):
                work_ids.update(c.representative_papers_a)
            if hasattr(c, 'representative_papers_b'):
                work_ids.update(c.representative_papers_b)
            if hasattr(c, 'related_papers'):
                work_ids.update(c.related_papers)
            if hasattr(c, 'related_method_papers'):
                work_ids.update(c.related_method_papers)
            if hasattr(c, 'related_domain_papers'):
                work_ids.update(c.related_domain_papers)
            if hasattr(c, 'papers_in_cluster'):
                work_ids.update(c.papers_in_cluster)

    return list(work_ids)


def _generate_evidence_roles(
    gap_title: str,
    gap_type: str,
    work_ids: List[str],
    paper_data: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generate specific roles for evidence papers using LLM.

    Falls back to generic roles if LLM call fails.
    """
    if not work_ids:
        return {}

    lines = []
    for wid in work_ids:
        pd = paper_data.get(wid)
        if not pd:
            continue
        line = f"- {wid}: {pd.get('title', 'Unknown')} ({pd.get('year', '?')})"
        abstract = pd.get("abstract", "")
        if abstract:
            line += f"\n  Abstract: {abstract[:300]}"
        lines.append(line)
    papers_text = "\n".join(lines)

    if not papers_text:
        return {}

    prompt = EVIDENCE_ROLE_PROMPT.format(
        gap_title=gap_title,
        gap_type=gap_type,
        papers=papers_text,
    )

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "{}"
        roles = extract_json_from_llm(content, expect_array=False)
        if isinstance(roles, dict):
            return roles
    except Exception as e:
        logger.warning(f"Failed to generate evidence roles: {e}")

    return {}


def _format_author_str(authors: Any) -> str:
    """Format author list into citation string."""
    if isinstance(authors, list):
        if len(authors) > 2:
            return f"{authors[0]} et al."
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        elif len(authors) == 1:
            return authors[0]
        return "Unknown"
    return str(authors) if authors else "Unknown"


def _ensure_str(val: Any) -> str:
    """Coerce a value to string, joining lists if needed."""
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val) if val else ""


def _clean_gap_title(title: str) -> str:
    """Clean up a gap title that looks like description text.

    Fixes cases where the LLM puts a full sentence with inline citations
    as the title instead of a concise phrase.
    """
    # Strip inline citations like (Author, Year) or (Author et al., Year)
    cleaned = re.sub(r'\s*\([^)]*\d{4}[^)]*\)', '', title)
    return cleaned.strip()


def create_gap_cards(
    synthesized_gaps: List[Dict[str, Any]],
    paper_data: Dict[str, Dict[str, Any]],
) -> List[GapCard]:
    """Convert synthesized gap dicts to GapCard objects with evidence roles."""
    cards = []

    VALID_GAP_TYPES = {"structural", "coverage", "temporal", "methodological", "novelty"}
    # Map common LLM variants to valid types
    TYPE_ALIASES = {
        "theoretical": "coverage",
        "empirical": "methodological",
        "conceptual": "coverage",
        "integration": "structural",
        "cross-disciplinary": "structural",
        "cross_disciplinary": "structural",
    }

    # Regex to catch engineering/integration-style titles
    # These are application ideas, not knowledge gaps
    INTEGRATION_PATTERN = re.compile(
        r"^(integration|combining|bridging|merging|unifying|unification|applying|application|develop)\s+(of\s+|a\s+)?",
        re.IGNORECASE,
    )

    for gap in synthesized_gaps:
        gap_type = gap.get("type", "coverage")
        if gap_type not in VALID_GAP_TYPES:
            gap_type = TYPE_ALIASES.get(gap_type, "coverage")
        gap_title = gap.get("title") or ""

        # Filter out "Integration of X and Y" style titles
        if gap_title and INTEGRATION_PATTERN.match(gap_title):
            logger.info(
                f"Filtered integration-style gap: '{gap_title}'"
            )
            continue

        if not gap_title or gap_title == "Untitled Gap":
            # Generate title from first sentence of description
            desc = gap.get("description", "")
            gap_title = desc.split(".")[0] if desc else f"Research gap in {gap_type}"

        # Clean up titles that look like description text
        # (LLM sometimes puts a full sentence with citations as the title)
        gap_title = _clean_gap_title(gap_title)
        evidence_ids = [
            wid for wid in gap.get("evidence_work_ids", [])[:10]
            if wid in paper_data
        ]

        # Generate specific roles for evidence papers
        roles = _generate_evidence_roles(gap_title, gap_type, evidence_ids, paper_data)

        # Build evidence list
        evidence = []
        for work_id in evidence_ids:
            pd = paper_data[work_id]
            evidence.append(Evidence(
                work_id=work_id,
                authors=_format_author_str(pd.get("authors", ["Unknown"])),
                year=pd.get("year", 0),
                title=pd.get("title", "Unknown"),
                role=roles.get(work_id, "Supporting evidence"),
            ))

        why_it_matters = _ensure_str(gap.get("why_it_matters", ""))
        type_explanation = GAP_TYPE_EXPLANATIONS.get(gap_type, "")

        # Always assign sequential IDs to avoid collisions between
        # heuristic and LLM-direct gaps that both start at gap_1
        cards.append(GapCard(
            gap_id=f"gap_{len(cards) + 1}",
            type=gap_type,
            type_explanation=type_explanation,
            title=gap_title,
            description=gap.get("description", ""),
            why_it_matters=why_it_matters,
            evidence=evidence,
            suggested_direction=_ensure_str(gap.get("suggested_direction", "")),
            confidence=0.0,  # Will be set after validation
            detection_score=gap.get("detection_score", 0.5),
            data_sources_used=gap.get("data_sources_used", []),
        ))

    return cards


def _title_word_similarity(title_a: str, title_b: str) -> float:
    """
    Compute word-level Jaccard similarity between two titles.

    Strips common stop words to focus on content words.
    """
    stop_words = {
        "a", "an", "the", "of", "in", "for", "and", "or", "to", "on",
        "with", "by", "from", "as", "at", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does",
        "that", "this", "these", "those", "it", "its", "between",
        "no", "not", "but", "into", "through", "during", "before",
        "after", "above", "below", "each", "all", "both", "few",
        "more", "most", "other", "some", "such", "than",
    }
    words_a = {w.lower() for w in re.findall(r'\w+', title_a)} - stop_words
    words_b = {w.lower() for w in re.findall(r'\w+', title_b)} - stop_words

    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)

    return intersection / union if union > 0 else 0.0


def _deduplicate_gap_cards(cards: List[GapCard]) -> List[GapCard]:
    """
    Remove duplicate gap cards based on evidence overlap OR title similarity.

    Checks both:
    1. Evidence overlap: > 50% shared work_ids
    2. Semantic overlap: > 50% title word similarity (Jaccard)

    Keeps the card with the higher detection score.
    """
    if len(cards) <= 1:
        return cards

    # Build evidence sets
    card_evidence = []
    for card in cards:
        evidence_set = {e.work_id for e in card.evidence}
        card_evidence.append((card, evidence_set))

    keep = []
    dropped = set()

    for i, (card_a, ev_a) in enumerate(card_evidence):
        if i in dropped:
            continue

        for j, (card_b, ev_b) in enumerate(card_evidence[i + 1:], start=i + 1):
            if j in dropped:
                continue

            # Check evidence overlap
            evidence_similar = False
            if ev_a and ev_b:
                overlap = len(ev_a & ev_b)
                min_size = min(len(ev_a), len(ev_b))
                if min_size > 0 and overlap / min_size > 0.5:
                    evidence_similar = True

            # Check title similarity
            title_similar = _title_word_similarity(card_a.title, card_b.title) > 0.5

            if evidence_similar or title_similar:
                # Keep the one with higher detection score
                if card_a.detection_score >= card_b.detection_score:
                    dropped.add(j)
                    logger.info(
                        f"Dedup: dropping '{card_b.title}' "
                        f"(similar to '{card_a.title}')"
                    )
                else:
                    dropped.add(i)
                    logger.info(
                        f"Dedup: dropping '{card_a.title}' "
                        f"(similar to '{card_b.title}')"
                    )
                    break

        if i not in dropped:
            keep.append(card_a)

    if dropped:
        logger.info(f"Deduplication removed {len(dropped)} overlapping gap cards")

    return keep


def run_gap_analysis(
    engine: Engine,
    map_id: UUID,
    tenant_id: UUID,
) -> GapAnalysisResponse:
    """
    Run gap analysis for a citation map.

    Same pipeline as run_rank_gap_analysis() but uses citation map papers.
    LLM-direct detection with citation graph context (edges, bridge papers).

    1. Check unlock status
    2. Get paper data from map_nodes
    3. Detect gaps via LLM-direct
    4. Validate evidence grounding
    5. External validation via GPT
    6. Store and return results
    """
    status = get_gap_analysis_status(engine, map_id=map_id)
    if not status.unlocked:
        raise ValueError(f"Gap analysis not unlocked: {status.message}")

    with engine.connect() as conn:
        # Step 1: Get available data sources
        available_sources = get_available_data_sources(conn, map_id=map_id)
        logger.info(f"Map gap analysis — available sources: {available_sources}")

        # Step 2: Get all paper data from map nodes
        all_paper_data = get_all_map_paper_data(conn, map_id)

        if not all_paper_data:
            logger.warning(f"No paper data found for map {map_id}")
            return GapAnalysisResponse(
                job_id=uuid4(),
                gaps=[],
                coverage_pct=status.coverage_pct,
                data_sources_used=available_sources,
                total_candidates_detected=0,
                candidates_validated=0,
            )

        # Step 3: LLM-direct gap detection (with citation graph context)
        logger.info("Map gap analysis: LLM-direct detection")
        llm_direct_gaps = detect_gaps_with_llm(conn, map_id, all_paper_data)

        # Validate evidence grounding
        all_work_ids = set(all_paper_data.keys())
        before_count = len(llm_direct_gaps)
        llm_direct_gaps = [
            g for g in llm_direct_gaps
            if len(set(g.get("evidence_work_ids", [])) & all_work_ids) >= 3
        ]
        dropped = before_count - len(llm_direct_gaps)
        if dropped:
            logger.info(f"Dropped {dropped} LLM-direct gaps with insufficient evidence grounding")

        if not llm_direct_gaps:
            logger.warning("No gaps identified by LLM-direct detection")
            return GapAnalysisResponse(
                job_id=uuid4(),
                gaps=[],
                coverage_pct=status.coverage_pct,
                data_sources_used=available_sources,
                total_candidates_detected=0,
                candidates_validated=0,
            )

        # Step 4: Create gap cards, filter, deduplicate
        gap_cards = create_gap_cards(llm_direct_gaps, all_paper_data)

        pre_filter = len(gap_cards)
        gap_cards = [g for g in gap_cards if g.detection_score >= MIN_DETECTION_SCORE]
        if pre_filter - len(gap_cards) > 0:
            logger.info(
                f"Filtered {pre_filter - len(gap_cards)} gap cards below "
                f"detection_score threshold ({MIN_DETECTION_SCORE})"
            )

        gap_cards = _deduplicate_gap_cards(gap_cards)

        # Step 5: External validation
        logger.info("Map gap analysis: external validation")
        gaps_for_validation = [
            {
                "gap_id": g.gap_id,
                "type": g.type,
                "title": g.title,
                "description": g.description,
                "suggested_direction": g.suggested_direction,
                "detection_score": g.detection_score,
            }
            for g in gap_cards
        ]

        validated_gaps = validate_gaps_batch(
            gaps_for_validation,
            min_confidence_threshold=MIN_CONFIDENCE_THRESHOLD,
        )

        # Update gap cards with validation results
        validated_gap_ids = {g["gap_id"]: g for g in validated_gaps}
        final_cards = []

        for card in gap_cards:
            if card.gap_id in validated_gap_ids:
                v = validated_gap_ids[card.gap_id]
                card.confidence = v["confidence"]
                card.validation_result = v["validation_result"]

                coverage = card.validation_result.coverage_pct
                if coverage is not None and coverage > MAX_COVERAGE_PCT:
                    logger.info(
                        f"Filtered gap '{card.title}': "
                        f"coverage_pct {coverage:.0f}% > {MAX_COVERAGE_PCT:.0f}% max"
                    )
                    continue

                final_cards.append(card)

        final_cards.sort(key=lambda x: x.confidence, reverse=True)

        # Step 6: Store results
        result_id = uuid4()
        job_id = uuid4()

        conn.execute(
            text("""
                INSERT INTO gap_analysis_results
                (id, map_id, tenant_id, gaps, data_sources_used, coverage_pct,
                 total_candidates_detected, candidates_validated)
                VALUES
                (:id, :map_id, :tenant_id, :gaps, :data_sources_used, :coverage_pct,
                 :total_candidates_detected, :candidates_validated)
            """),
            {
                "id": result_id,
                "map_id": map_id,
                "tenant_id": tenant_id,
                "gaps": json.dumps([c.model_dump() for c in final_cards]),
                "data_sources_used": available_sources,
                "coverage_pct": status.coverage_pct,
                "total_candidates_detected": len(llm_direct_gaps),
                "candidates_validated": len(final_cards),
            },
        )
        conn.commit()

        return GapAnalysisResponse(
            job_id=job_id,
            gaps=final_cards,
            coverage_pct=status.coverage_pct,
            data_sources_used=available_sources,
            total_candidates_detected=len(llm_direct_gaps),
            candidates_validated=len(final_cards),
        )


def get_cached_gap_analysis(
    engine: Engine,
    map_id: UUID,
    tenant_id: Optional[UUID] = None,
) -> Optional[GapAnalysisResponse]:
    """Get cached gap analysis results if available and not stale."""
    with engine.connect() as conn:
        params: Dict[str, Any] = {"map_id": map_id, "ttl_hours": CACHE_TTL_HOURS}

        query = """
            SELECT id, gaps, data_sources_used, coverage_pct,
                   total_candidates_detected, candidates_validated
            FROM gap_analysis_results
            WHERE map_id = :map_id
            AND created_at > NOW() - INTERVAL '1 hour' * :ttl_hours
        """
        if tenant_id is not None:
            query += " AND tenant_id = :tenant_id"
            params["tenant_id"] = tenant_id

        query += " ORDER BY created_at DESC LIMIT 1"

        result = conn.execute(text(query), params).mappings().first()

        if not result:
            return None

        gaps_data = result["gaps"]
        if isinstance(gaps_data, str):
            gaps_data = json.loads(gaps_data)

        gap_cards = [GapCard(**g) for g in gaps_data]

        return GapAnalysisResponse(
            job_id=result["id"],
            gaps=gap_cards,
            coverage_pct=result["coverage_pct"],
            data_sources_used=result["data_sources_used"],
            total_candidates_detected=result["total_candidates_detected"],
            candidates_validated=result["candidates_validated"],
        )


def run_rank_gap_analysis(
    engine: Engine,
    rank_job_id: UUID,
    tenant_id: UUID,
) -> GapAnalysisResponse:
    """
    Run gap analysis for a rank job.

    Simplified pipeline that skips heuristic detection (no graph) and goes
    straight to LLM-direct detection using the ranked paper list.

    1. Check unlock status
    2. Get paper data from rank_results
    3. Detect gaps via LLM-direct
    4. Validate evidence grounding
    5. External validation via GPT
    6. Store and return results
    """
    status = get_gap_analysis_status(engine, rank_job_id=rank_job_id)
    if not status.unlocked:
        raise ValueError(f"Gap analysis not unlocked: {status.message}")

    with engine.connect() as conn:
        # Step 1: Get available data sources
        available_sources = get_available_data_sources(conn, rank_job_id=rank_job_id)
        logger.info(f"Rank gap analysis — available sources: {available_sources}")

        # Step 2: Get all paper data from rank results
        all_paper_data = get_all_rank_paper_data(conn, rank_job_id)

        if not all_paper_data:
            logger.warning(f"No paper data found for rank job {rank_job_id}")
            return GapAnalysisResponse(
                job_id=uuid4(),
                gaps=[],
                coverage_pct=status.coverage_pct,
                data_sources_used=available_sources,
                total_candidates_detected=0,
                candidates_validated=0,
            )

        # Step 3: LLM-direct gap detection (no heuristics for rank jobs)
        logger.info("Rank gap analysis: LLM-direct detection")
        llm_direct_gaps = detect_gaps_with_llm(conn, None, all_paper_data)

        # Validate evidence grounding
        all_work_ids = set(all_paper_data.keys())
        before_count = len(llm_direct_gaps)
        llm_direct_gaps = [
            g for g in llm_direct_gaps
            if len(set(g.get("evidence_work_ids", [])) & all_work_ids) >= 3
        ]
        dropped = before_count - len(llm_direct_gaps)
        if dropped:
            logger.info(f"Dropped {dropped} LLM-direct gaps with insufficient evidence grounding")

        if not llm_direct_gaps:
            logger.warning("No gaps identified by LLM-direct detection")
            return GapAnalysisResponse(
                job_id=uuid4(),
                gaps=[],
                coverage_pct=status.coverage_pct,
                data_sources_used=available_sources,
                total_candidates_detected=0,
                candidates_validated=0,
            )

        # Step 4: Create gap cards, filter, deduplicate
        gap_cards = create_gap_cards(llm_direct_gaps, all_paper_data)

        pre_filter = len(gap_cards)
        gap_cards = [g for g in gap_cards if g.detection_score >= MIN_DETECTION_SCORE]
        if pre_filter - len(gap_cards) > 0:
            logger.info(
                f"Filtered {pre_filter - len(gap_cards)} gap cards below "
                f"detection_score threshold ({MIN_DETECTION_SCORE})"
            )

        gap_cards = _deduplicate_gap_cards(gap_cards)

        # Step 5: External validation
        logger.info("Rank gap analysis: external validation")
        gaps_for_validation = [
            {
                "gap_id": g.gap_id,
                "type": g.type,
                "title": g.title,
                "description": g.description,
                "suggested_direction": g.suggested_direction,
                "detection_score": g.detection_score,
            }
            for g in gap_cards
        ]

        validated_gaps = validate_gaps_batch(
            gaps_for_validation,
            min_confidence_threshold=MIN_CONFIDENCE_THRESHOLD,
        )

        # Update gap cards with validation results
        validated_gap_ids = {g["gap_id"]: g for g in validated_gaps}
        final_cards = []

        for card in gap_cards:
            if card.gap_id in validated_gap_ids:
                v = validated_gap_ids[card.gap_id]
                card.confidence = v["confidence"]
                card.validation_result = v["validation_result"]

                coverage = card.validation_result.coverage_pct
                if coverage is not None and coverage > MAX_COVERAGE_PCT:
                    logger.info(
                        f"Filtered gap '{card.title}': "
                        f"coverage_pct {coverage:.0f}% > {MAX_COVERAGE_PCT:.0f}% max"
                    )
                    continue

                final_cards.append(card)

        final_cards.sort(key=lambda x: x.confidence, reverse=True)

        # Step 6: Store results
        result_id = uuid4()
        job_id = uuid4()

        conn.execute(
            text("""
                INSERT INTO gap_analysis_results
                (id, rank_job_id, tenant_id, gaps, data_sources_used, coverage_pct,
                 total_candidates_detected, candidates_validated)
                VALUES
                (:id, :rank_job_id, :tenant_id, :gaps, :data_sources_used, :coverage_pct,
                 :total_candidates_detected, :candidates_validated)
            """),
            {
                "id": result_id,
                "rank_job_id": rank_job_id,
                "tenant_id": tenant_id,
                "gaps": json.dumps([c.model_dump() for c in final_cards]),
                "data_sources_used": available_sources,
                "coverage_pct": status.coverage_pct,
                "total_candidates_detected": len(llm_direct_gaps),
                "candidates_validated": len(final_cards),
            },
        )
        conn.commit()

        return GapAnalysisResponse(
            job_id=job_id,
            gaps=final_cards,
            coverage_pct=status.coverage_pct,
            data_sources_used=available_sources,
            total_candidates_detected=len(llm_direct_gaps),
            candidates_validated=len(final_cards),
        )


def get_cached_rank_gap_analysis(
    engine: Engine,
    rank_job_id: UUID,
    tenant_id: Optional[UUID] = None,
) -> Optional[GapAnalysisResponse]:
    """Get cached gap analysis results for a rank job if available and not stale."""
    with engine.connect() as conn:
        params: Dict[str, Any] = {"rank_job_id": rank_job_id, "ttl_hours": CACHE_TTL_HOURS}

        query = """
            SELECT id, gaps, data_sources_used, coverage_pct,
                   total_candidates_detected, candidates_validated
            FROM gap_analysis_results
            WHERE rank_job_id = :rank_job_id
            AND created_at > NOW() - INTERVAL '1 hour' * :ttl_hours
        """
        if tenant_id is not None:
            query += " AND tenant_id = :tenant_id"
            params["tenant_id"] = tenant_id

        query += " ORDER BY created_at DESC LIMIT 1"

        result = conn.execute(text(query), params).mappings().first()

        if not result:
            return None

        gaps_data = result["gaps"]
        if isinstance(gaps_data, str):
            gaps_data = json.loads(gaps_data)

        gap_cards = [GapCard(**g) for g in gaps_data]

        return GapAnalysisResponse(
            job_id=result["id"],
            gaps=gap_cards,
            coverage_pct=result["coverage_pct"],
            data_sources_used=result["data_sources_used"],
            total_candidates_detected=result["total_candidates_detected"],
            candidates_validated=result["candidates_validated"],
        )
