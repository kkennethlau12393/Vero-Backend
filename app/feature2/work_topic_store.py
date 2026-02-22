from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Iterable
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TopicScore:
    topic_id: str
    score: float

@dataclass(frozen=True)
class WorkForMap:
    work_id: str
    title: Optional[str]
    year: Optional[int]
    cited_by_count: int
    authors_json: Any
    venue: Optional[str]
    primary_topic_id: Optional[str]
    primary_topic_score: Optional[float]
    topics: list[TopicScore]  # scored topics
    is_retracted: bool
    abstract: Optional[str]
    category: Optional[str] = None  # foundational, methodological, applied, implementation, handbook
    category_confidence: Optional[float] = None
    referenced_works: Optional[list[str]] = None  # OpenAlex work IDs this paper cites

    def preview_json(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "title": self.title,
            "year": self.year,
            "cited_by_count": self.cited_by_count,
            "authors": self.authors_json,
            "venue": self.venue,
            "primary_topic_id": self.primary_topic_id,
            "category": self.category,
        }

@dataclass(frozen=True)
class WorkPreview:
    work_id: str
    title: Optional[str]
    year: Optional[int]
    cited_by_count: int
    authors: list[str]
    venue: Optional[str]
    doi: Optional[str] = None
    is_open_access: Optional[bool] = None
    oa_pdf_url: Optional[str] = None

class WorkStore:
    """
    Production rule: cache-first; only call OpenAlex when missing.
    This class shows the cache/DB path; plug OpenAlex fetching into ensure_works_present().
    """

    @staticmethod
    def load_many(conn: Connection, work_ids: list[str]) -> dict[str, WorkForMap]:
        if not work_ids:
            return {}

        rows = conn.execute(
            text("""
                SELECT
                  work_id, title, year, cited_by_count,
                  authors_json, venue,
                  primary_topic_id, primary_topic_score,
                  topics_json,
                  is_retracted,
                  abstract,
                  category,
                  category_confidence,
                  referenced_works_json
                FROM works
                WHERE work_id = ANY(:ids)
            """),
            {"ids": work_ids},
        ).mappings().all()

        out: dict[str, WorkForMap] = {}
        for r in rows:
            topics_list = r["topics_json"] or []
            topics = [TopicScore(t["topic_id"], float(t["score"])) for t in topics_list if "topic_id" in t and "score" in t]
            # Extract referenced_works as list of work IDs
            refs_raw = r["referenced_works_json"]
            referenced_works = None
            if refs_raw and isinstance(refs_raw, list):
                referenced_works = refs_raw
            out[r["work_id"]] = WorkForMap(
                work_id=r["work_id"],
                title=r["title"],
                year=r["year"],
                cited_by_count=int(r["cited_by_count"] or 0),
                authors_json=r["authors_json"] or [],
                venue=r["venue"],
                primary_topic_id=r["primary_topic_id"],
                primary_topic_score=float(r["primary_topic_score"]) if r["primary_topic_score"] is not None else None,
                topics=topics,
                is_retracted=bool(r["is_retracted"] or False),
                abstract=r["abstract"],
                category=r["category"],
                category_confidence=float(r["category_confidence"]) if r["category_confidence"] is not None else None,
                referenced_works=referenced_works,
            )
        return out

    @staticmethod
    def load_previews_many(conn: Connection, work_ids: list[str]) -> dict[str, WorkPreview]:
        """
        Render path: batch load preview fields to avoid N+1
        """
        if not work_ids:
            return {}

        rows = conn.execute(
            text("""
                SELECT work_id, title, year, cited_by_count, authors_json, venue,
                       doi, is_open_access, oa_pdf_url
                FROM works
                WHERE work_id = ANY(:ids)
            """),
            {"ids": work_ids},
        ).mappings().all()

        out: dict[str, WorkPreview] = {}
        for r in rows:
            authors = r["authors_json"] or []
            if not isinstance(authors, list):
                authors = []
            out[r["work_id"]] = WorkPreview(
                work_id=r["work_id"],
                title=r["title"],
                year=r["year"],
                cited_by_count=int(r["cited_by_count"] or 0),
                authors=[str(a) for a in authors],
                venue=r["venue"],
                doi=r["doi"],
                is_open_access=r["is_open_access"],
                oa_pdf_url=r["oa_pdf_url"],
            )
        return out

    @staticmethod
    def ensure_works_present(conn: Connection, missing_work_ids: list[str]) -> None:
        """
        Fetch missing works from OpenAlex and upsert them into the works table.

        The OpenAlex API is called in batches of up to 50 IDs.  Missing works
        do not cause the caller to fail; they simply won't be inserted.
        """
        if not missing_work_ids:
            return
        import requests
        # Determine which IDs are absent from the DB
        existing = conn.execute(
            text("SELECT work_id FROM works WHERE work_id = ANY(:ids)"),
            {"ids": missing_work_ids},
        ).scalars().all()
        existing_set = set(existing)
        to_fetch = [wid for wid in missing_work_ids if wid not in existing_set]
        if not to_fetch:
            return
        # Batch up to 50 IDs per request
        BATCH_SIZE = 50
        for i in range(0, len(to_fetch), BATCH_SIZE):
            batch = to_fetch[i: i + BATCH_SIZE]
            # OpenAlex expects full URLs separated by pipes
            ids_param = "|".join(f"https://openalex.org/{wid}" for wid in batch)
            url = "https://api.openalex.org/works"
            params = {"filter": f"openalex:{ids_param}", "per-page": len(batch)}
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
            except Exception:
                continue
            data = resp.json()
            works = data.get("results", [])
            rows = []
            for w in works:
                wid_full = w.get("id")
                if not wid_full or "/" not in wid_full:
                    continue
                wid = wid_full.rsplit("/", 1)[-1]
                # Basic fields
                title = w.get("title")
                year = w.get("publication_year")
                authors = [
                    au.get("author", {}).get("display_name") or au.get("display_name")
                    for au in w.get("authorships", [])
                    if au.get("author", {}).get("display_name") or au.get("display_name")
                ]
                cited_by_count = w.get("cited_by_count") or 0
                venue = (
                    ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
                    or (w.get("host_venue") or {}).get("display_name")
                )
                # Fallback: check locations array for a journal/conference source
                if not venue:
                    for loc in w.get("locations", []):
                        loc_src = (loc.get("source") or {}).get("display_name")
                        loc_type = (loc.get("source") or {}).get("type")
                        if loc_src and loc_type in ("journal", "conference"):
                            venue = loc_src
                            break
                # Abstract: OpenAlex provides an inverted index
                abstract = _decode_openalex_abstract(w.get("abstract_inverted_index"))
                # Primary topic
                primary_concept = None
                primary_location = w.get("primary_location")
                if primary_location:
                    primary_concept = primary_location.get("primary_concept")
                primary_topic_id = None
                primary_topic_score = None
                if primary_concept:
                    primary_topic_id = primary_concept.get("id")
                    primary_topic_score = primary_concept.get("score")
                # Topics list
                topic_entries = []
                for t in w.get("concepts", []):
                    tid = t.get("id")
                    score = t.get("score")
                    if tid and score is not None:
                        topic_entries.append({"topic_id": tid.split("/")[-1] if "/" in tid else tid, "score": score})
                # Referenced works (papers this work cites) - extract work IDs from URLs
                referenced_works_raw = w.get("referenced_works", [])
                referenced_works = [
                    ref_url.rsplit("/", 1)[-1]
                    for ref_url in referenced_works_raw
                    if ref_url and "/" in ref_url
                ]
                # Open access info from OpenAlex
                oa_info = w.get("open_access") or {}
                is_open_access = oa_info.get("is_oa", False)
                oa_status = oa_info.get("oa_status")  # gold, green, hybrid, bronze, closed
                oa_pdf_url = oa_info.get("oa_url")
                # DOI
                doi = w.get("doi")
                if doi and doi.startswith("https://doi.org/"):
                    doi = doi[len("https://doi.org/"):]
                # Upsert row dictionary
                rows.append({
                    "work_id": wid,
                    "title": title,
                    "year": year,
                    "cited_by_count": cited_by_count,
                    "authors_json": authors,
                    "venue": venue,
                    "primary_topic_id": (primary_topic_id.split("/")[-1] if primary_topic_id and "/" in primary_topic_id else primary_topic_id),
                    "primary_topic_score": primary_topic_score,
                    "topics_json": topic_entries,
                    "is_retracted": False,
                    "abstract": abstract,
                    "referenced_works_json": referenced_works,
                    "is_open_access": is_open_access,
                    "oa_status": oa_status,
                    "oa_pdf_url": oa_pdf_url,
                    "doi": doi,
                })
            if not rows:
                continue
            # Bulk upsert into works table
            # IMPORTANT: Only update if OpenAlex data is more credible
            # - Keep higher citation count (prevents overwriting correct S2/ArXiv data)
            # - Keep earlier year if citations are similar (prevents future-dated papers)
            conn.execute(
                text(
                    """
                    INSERT INTO works (
                        work_id, title, year, cited_by_count,
                        authors_json, venue, primary_topic_id,
                        primary_topic_score, topics_json, is_retracted,
                        abstract, referenced_works_json,
                        is_open_access, oa_status, oa_pdf_url, doi
                    ) VALUES (
                        :work_id, :title, :year, :cited_by_count,
                        :authors_json, :venue, :primary_topic_id,
                        :primary_topic_score, :topics_json, :is_retracted,
                        :abstract, :referenced_works_json,
                        :is_open_access, :oa_status, :oa_pdf_url, :doi
                    )
                    ON CONFLICT (work_id) DO UPDATE
                    SET
                        -- Only update title if we don't have one
                        title = COALESCE(works.title, EXCLUDED.title),
                        -- Keep the earlier year (prevents future-dated wrong papers)
                        -- unless existing year is NULL
                        year = CASE
                            WHEN works.year IS NULL THEN EXCLUDED.year
                            WHEN EXCLUDED.year IS NULL THEN works.year
                            WHEN works.year <= EXCLUDED.year THEN works.year
                            ELSE EXCLUDED.year
                        END,
                        -- Keep the HIGHER citation count (S2 often has more accurate counts)
                        cited_by_count = GREATEST(COALESCE(works.cited_by_count, 0), COALESCE(EXCLUDED.cited_by_count, 0)),
                        -- Only update authors if we don't have any
                        authors_json = CASE
                            WHEN works.authors_json IS NULL OR works.authors_json = '[]'::jsonb
                            THEN EXCLUDED.authors_json
                            ELSE works.authors_json
                        END,
                        -- Only update venue if we don't have one
                        venue = COALESCE(works.venue, EXCLUDED.venue),
                        -- Update topic info (OpenAlex is authoritative for this)
                        primary_topic_id = COALESCE(EXCLUDED.primary_topic_id, works.primary_topic_id),
                        primary_topic_score = COALESCE(EXCLUDED.primary_topic_score, works.primary_topic_score),
                        topics_json = CASE
                            WHEN EXCLUDED.topics_json IS NOT NULL AND EXCLUDED.topics_json != '[]'::jsonb
                            THEN EXCLUDED.topics_json
                            ELSE works.topics_json
                        END,
                        is_retracted = EXCLUDED.is_retracted OR works.is_retracted,
                        -- Only update abstract if we don't have one
                        abstract = COALESCE(works.abstract, EXCLUDED.abstract),
                        -- Only update referenced_works if we don't have them
                        referenced_works_json = COALESCE(works.referenced_works_json, EXCLUDED.referenced_works_json),
                        -- Open access info (OpenAlex is authoritative)
                        is_open_access = COALESCE(EXCLUDED.is_open_access, works.is_open_access),
                        oa_status = COALESCE(EXCLUDED.oa_status, works.oa_status),
                        oa_pdf_url = COALESCE(EXCLUDED.oa_pdf_url, works.oa_pdf_url),
                        doi = COALESCE(EXCLUDED.doi, works.doi)
                    """
                ).bindparams(
                    bindparam("authors_json", type_=JSONB),
                    bindparam("topics_json", type_=JSONB),
                    bindparam("referenced_works_json", type_=JSONB),
                ),
                rows,
            )


def _decode_openalex_abstract(inverted_index: Any) -> Optional[str]:
    """Reconstruct abstract text from OpenAlex inverted index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    position_map: dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            try:
                pos_int = int(pos)
            except Exception:
                continue
            position_map.setdefault(pos_int, word)
    if not position_map:
        return None
    ordered = [position_map[i] for i in sorted(position_map.keys())]
    return " ".join(ordered).strip() or None


class TopicHierarchyStore:
    @staticmethod
    def load_topic_to_subfield_and_field(conn: Connection, topic_ids: list[str]) -> dict[str, dict[str, str]]:
        """
        Returns mapping: topic_id -> {"subfield_id": ..., "field_id": ...}
        """
        if not topic_ids:
            return {}
        rows = conn.execute(
            text("""
                SELECT topic_id, subfield_id, field_id
                FROM openalex_topics
                WHERE topic_id = ANY(:ids)
            """),
            {"ids": topic_ids},
        ).mappings().all()
        return {r["topic_id"]: {"subfield_id": r["subfield_id"], "field_id": r["field_id"]} for r in rows}

class LabelStore:
    """
    Dev-safe label resolver.
    For now: return IDs as labels (stable, non-blocking).
    Later: replace with real tables (topics/subfields/fields with display_name).
    """

    @staticmethod
    def topic_labels(conn: Connection, topic_ids: list[str]) -> dict[str, str]:
        return {tid: tid for tid in topic_ids}

    @staticmethod
    def subfield_labels(conn: Connection, subfield_ids: list[str]) -> dict[str, str]:
        return {sid: sid for sid in subfield_ids}

    @staticmethod
    def field_labels(conn: Connection, field_ids: list[str]) -> dict[str, str]:
        return {fid: fid for fid in field_ids}
