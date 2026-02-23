from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection

@dataclass(frozen=True)
class GraphDraftRow:
    graph_draft_id: UUID
    tenant_id: UUID
    candidate_set_id: Optional[UUID]

@dataclass(frozen=True)
class GraphDraftData:
    header: GraphDraftRow
    node_work_ids: list[str]
    edges: list[tuple[str, str]]  # (from, to)

class GraphDraftRepo:
    @staticmethod
    def load(conn: Connection, tenant_id: UUID, graph_draft_id: UUID) -> GraphDraftData:
        hdr = conn.execute(
            text("""
                SELECT graph_draft_id, tenant_id, candidate_set_id
                FROM graph_drafts
                WHERE graph_draft_id = :gd
            """),
            {"gd": graph_draft_id},
        ).mappings().first()

        if not hdr:
            raise ValueError("graph_draft_not_found")
        if hdr["tenant_id"] != tenant_id:
            raise PermissionError("graph_draft_wrong_tenant")

        nodes = conn.execute(
            text("""
                SELECT work_id
                FROM graph_draft_nodes
                WHERE graph_draft_id = :gd
            """),
            {"gd": graph_draft_id},
        ).scalars().all()

        edges = conn.execute(
            text("""
                SELECT from_work_id, to_work_id
                FROM graph_draft_edges
                WHERE graph_draft_id = :gd
            """),
            {"gd": graph_draft_id},
        ).all()

        return GraphDraftData(
            header=GraphDraftRow(
                graph_draft_id=hdr["graph_draft_id"],
                tenant_id=hdr["tenant_id"],
                candidate_set_id=hdr["candidate_set_id"],
            ),
            node_work_ids=list(nodes),
            edges=[(a, b) for (a, b) in edges],
        )

class MapRepo:
    @staticmethod
    def find_existing_map_id(conn: Connection, tenant_id: UUID, params_hash: str) -> Optional[UUID]:
        row = conn.execute(
            text("""
                SELECT map_id
                FROM maps
                WHERE tenant_id = :t AND params_hash = :h
                LIMIT 1
            """),
            {"t": tenant_id, "h": params_hash},
        ).first()
        return row[0] if row else None

    @staticmethod
    def insert_map_header(
        conn: Connection,
        *,
        map_id: UUID,
        tenant_id: UUID,
        graph_draft_id: UUID,
        default_grouping: str,
        allowed_groupings: list[str],
        stats_json: dict[str, Any],
        params_json: dict[str, Any],
        params_hash: str,
    ) -> None:
        stmt = text("""
            INSERT INTO maps (
              map_id, tenant_id, graph_draft_id,
              default_grouping, allowed_groupings,
              stats_json, params_json, params_hash
            ) VALUES (
              :map_id, :tenant_id, :graph_draft_id,
              :default_grouping, :allowed_groupings,
              :stats_json, :params_json, :params_hash
            )
        """).bindparams(
            bindparam("allowed_groupings", type_=JSONB),
            bindparam("stats_json", type_=JSONB),
            bindparam("params_json", type_=JSONB),
        )

        conn.execute(
            stmt,
            {
                "map_id": map_id,
                "tenant_id": tenant_id,
                "graph_draft_id": graph_draft_id,
                "default_grouping": default_grouping,
                "allowed_groupings": allowed_groupings,
                "stats_json": stats_json,
                "params_json": params_json,
                "params_hash": params_hash,
            },
        )

    @staticmethod
    def bulk_insert_nodes(conn: Connection, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        stmt = text("""
            INSERT INTO map_nodes (
              map_id, work_id,
              best_topic_id, best_subfield_id, topic_score, connector_score,
              x, y, work_preview_json
            ) VALUES (
              :map_id, :work_id,
              :best_topic_id, :best_subfield_id, :topic_score, :connector_score,
              :x, :y, :work_preview_json
            )
        """).bindparams(
            bindparam("work_preview_json", type_=JSONB),
        )

        payload = []
        for r in rows:
            payload.append({**r, "work_preview_json": r.get("work_preview_json") or {}})

        conn.execute(stmt, payload)

    @staticmethod
    def bulk_insert_edges(conn: Connection, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        conn.execute(
            text("""
                INSERT INTO map_edges (map_id, from_work_id, to_work_id)
                VALUES (:map_id, :from_work_id, :to_work_id)
            """),
            rows,
        )

    @staticmethod
    def load_map_header(conn: Connection, tenant_id: UUID, map_id: UUID) -> dict[str, Any]:
        row = conn.execute(
            text("""
                SELECT map_id, tenant_id, default_grouping, allowed_groupings, stats_json
                FROM maps
                WHERE map_id = :m AND tenant_id = :t
                LIMIT 1
            """),
            {"m": map_id, "t": tenant_id},
        ).mappings().first()

        if not row:
            raise ValueError("map_not_found")

        return {
            "map_id": row["map_id"],
            "default_grouping": row["default_grouping"],
            "allowed_groupings": row["allowed_groupings"],
            "stats": row["stats_json"] or {},
        }

    @staticmethod
    def load_map_nodes(conn: Connection, map_id: UUID) -> list[dict[str, Any]]:
        rows = conn.execute(
            text("""
                SELECT work_id, best_topic_id, best_subfield_id, connector_score, x, y
                FROM map_nodes
                WHERE map_id = :m
                ORDER BY work_id ASC
            """),
            {"m": map_id},
        ).mappings().all()
        return [dict(r) for r in rows]

    @staticmethod
    def load_map_edges(conn: Connection, map_id: UUID) -> list[tuple[str, str]]:
        rows = conn.execute(
            text("""
                SELECT from_work_id, to_work_id
                FROM map_edges
                WHERE map_id = :m
            """),
            {"m": map_id},
        ).all()
        return [(a, b) for (a, b) in rows]

class CandidateSetRepo:
    @staticmethod
    def load_work_ids_and_provenance(conn: Connection, tenant_id: UUID, candidate_set_id: UUID) -> list[dict[str, Any]]:
        # enforce tenant ownership via candidate_sets header
        hdr = conn.execute(
            text("""
                SELECT candidate_set_id, tenant_id
                FROM candidate_sets
                WHERE candidate_set_id = :cs
                LIMIT 1
            """),
            {"cs": candidate_set_id},
        ).mappings().first()

        if not hdr:
            raise ValueError("candidate_set_not_found")
        if hdr["tenant_id"] != tenant_id:
            raise PermissionError("candidate_set_wrong_tenant")

        rows = conn.execute(
            text("""
                SELECT work_id, provenance_json
                FROM candidate_set_items
                WHERE candidate_set_id = :cs
                ORDER BY work_id ASC
            """),
            {"cs": candidate_set_id},
        ).mappings().all()

        out = []
        for r in rows:
            prov = r.get("provenance_json") or []
            if not isinstance(prov, list):
                prov = []
            out.append({"work_id": r["work_id"], "provenance": prov})
        return out

    @staticmethod
    def insert_or_get_candidate_set(
        conn: Connection,
        *,
        tenant_id: UUID,
        seed_type: str,
        seed_json: dict[str, Any],
        params_hash: str,
    ) -> UUID:
        """
        Insert a candidate set header if not present and return its ID.

        Candidate sets are uniquely identified per tenant by their
        `params_hash` (deterministic hash of query and filters) to allow
        idempotent reuse.  If a candidate set with the same params already
        exists, this function returns its ID.  Otherwise it creates a new
        row with a fresh UUID and the provided seed metadata.
        """
        new_id = uuid4()
        stmt = text(
            """
            INSERT INTO candidate_sets (
                candidate_set_id, tenant_id, seed_type, seed_json, params_hash
            ) VALUES (
                :cs_id, :t, :seed_type, :seed_json, :params_hash
            )
            ON CONFLICT (tenant_id, params_hash) DO NOTHING
            RETURNING candidate_set_id
            """
        ).bindparams(
            bindparam("seed_json", type_=JSONB)
        )
        inserted = conn.execute(
            stmt,
            {
                "cs_id": new_id,
                "t": tenant_id,
                "seed_type": seed_type,
                "seed_json": seed_json or {},
                "params_hash": params_hash,
            },
        ).scalar_one_or_none()
        if inserted is not None:
            return inserted
        # If conflict, fetch the exsting ID
        row = conn.execute(
            text(
                """
                SELECT candidate_set_id
                FROM candidate_sets
                WHERE tenant_id = :t AND params_hash = :params_hash
                LIMIT 1
                """
            ),
            {"t": tenant_id, "params_hash": params_hash},
        ).first()
        if not row:
            raise RuntimeError("candidate_set_insert_conflict_but_missing")
        return row[0]

    @staticmethod
    def bulk_insert_candidate_items(
        conn: Connection,
        candidate_set_id: UUID,
        items: list[dict[str, Any]],
    ) -> None:
        """
        Bulk insert candidate_set_items rows.

        Each item dict must contain `work_id` and `provenance` keys.  The
        provenance will be stored in the `provenance_json` column.
        Duplicate work IDs for the same candidate set are ignored.
        """
        if not items:
            return
        import json
        from psycopg2.extras import execute_batch

        # Prepare rows, converting provenance to JSON strings
        payload = []
        for itm in items:
            wid = itm.get("work_id")
            prov = itm.get("provenance") or []
            if wid is None:
                continue
            payload.append((
                str(candidate_set_id),
                wid,
                json.dumps(prov),
            ))
        if not payload:
            return

        raw_cursor = conn.connection.dbapi_connection.cursor()
        try:
            execute_batch(
                raw_cursor,
                """
                INSERT INTO candidate_set_items (
                    candidate_set_id, work_id, provenance_json
                ) VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (candidate_set_id, work_id) DO NOTHING
                """,
                payload,
                page_size=200,
            )
        finally:
            raw_cursor.close()

class RankRepo:

    @staticmethod
    def insert_pending_or_get_existing(
        conn: Connection,
        *,
        tenant_id: UUID,
        rank_type: str,
        candidate_set_id: UUID,
        context_json: dict[str, Any],
        filters_json: dict[str, Any],
        rank_params_json: dict[str, Any],
        params_hash: str,
    ) -> tuple[UUID, bool, str]:
        """
        Returns (rank_job_id, created_new, status)

        created_new=True means this request is the builder for this params_hash.
        created_new=False means job already exists (possibly completed/running/pending/failed).
        """
        new_id = uuid4()

        stmt = text("""
            INSERT INTO rank_jobs (
                rank_job_id, tenant_id, rank_type, candidate_set_id,
                context_json, filters_json, rank_params_json,
                params_hash, status
            ) VALUES (
                :rank_job_id, :tenant_id, :rank_type, :candidate_set_id,
                :context_json, :filters_json, :rank_params_json,
                :params_hash, 'pending'
            )
            ON CONFLICT (tenant_id, rank_type, params_hash) DO NOTHING
            RETURNING rank_job_id
        """).bindparams(
            bindparam("context_json", type_=JSONB),
            bindparam("filters_json", type_=JSONB),
            bindparam("rank_params_json", type_=JSONB),
        )

        inserted = conn.execute(stmt, {
            "rank_job_id": new_id,
            "tenant_id": tenant_id,
            "rank_type": rank_type,
            "candidate_set_id": candidate_set_id,
            "context_json": context_json or {},
            "filters_json": filters_json or {},
            "rank_params_json": rank_params_json or {},
            "params_hash": params_hash,
        }).scalar_one_or_none()

        if inserted is not None:
            return inserted, True, "pending"

        existing = conn.execute(
            text("""
                SELECT rank_job_id, status
                FROM rank_jobs
                WHERE tenant_id = :t AND rank_type = :rt AND params_hash = :h
                LIMIT 1
            """),
            {"t": tenant_id, "rt": rank_type, "h": params_hash},
        ).mappings().first()

        if not existing:
            raise RuntimeError("rank_job_conflict_but_missing")

        return existing["rank_job_id"], False, existing["status"]

    @staticmethod
    def mark_running(conn: Connection, rank_job_id: UUID) -> None:
        conn.execute(
            text("""
                UPDATE rank_jobs
                SET status='running', started_at=now()
                WHERE rank_job_id=:id
            """),
            {"id": rank_job_id},
        )

    @staticmethod
    def mark_completed(conn: Connection, rank_job_id: UUID) -> None:
        conn.execute(
            text("""
                UPDATE rank_jobs
                SET status='completed', completed_at=now()
                WHERE rank_job_id=:id
            """),
            {"id": rank_job_id},
        )

    @staticmethod
    def mark_failed(conn: Connection, rank_job_id: UUID, error_json: dict[str, Any]) -> None:
        stmt = text("""
            UPDATE rank_jobs
            SET status='failed', completed_at=now(), error_json=:err
            WHERE rank_job_id=:id
        """).bindparams(bindparam("err", type_=JSONB))
        conn.execute(stmt, {"id": rank_job_id, "err": error_json or {}})

    @staticmethod
    def bulk_insert_results(conn: Connection, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        stmt = text("""
            INSERT INTO rank_results (
                rank_job_id, rank_index, work_id, score,
                score_breakdown_json, reasons_json, work_preview_json, provenance_json
            ) VALUES (
                :rank_job_id, :rank_index, :work_id, :score,
                :score_breakdown_json, :reasons_json, :work_preview_json, :provenance_json
            )
        """).bindparams(
            bindparam("score_breakdown_json", type_=JSONB),
            bindparam("reasons_json", type_=JSONB),
            bindparam("work_preview_json", type_=JSONB),
            bindparam("provenance_json", type_=JSONB),
        )
        conn.execute(stmt, rows)

    @staticmethod
    def load_results(conn: Connection, tenant_id: UUID, rank_job_id: UUID) -> dict[str, Any]:
        hdr = conn.execute(
            text("""
                SELECT rank_job_id, tenant_id, rank_type, candidate_set_id, status, context_json, filters_json, rank_params_json
                FROM rank_jobs
                WHERE rank_job_id=:id
                LIMIT 1
            """),
            {"id": rank_job_id},
        ).mappings().first()
        if not hdr:
            raise ValueError("rank_job_not_found")
        if hdr["tenant_id"] != tenant_id:
            raise PermissionError("rank_job_wrong_tenant")
        items = conn.execute(
            text("""
                SELECT rank_index, work_id, score, score_breakdown_json, reasons_json, work_preview_json, provenance_json
                FROM rank_results
                WHERE rank_job_id=:id
                ORDER BY rank_index ASC
            """),
            {"id": rank_job_id},
        ).mappings().all()

        job = dict(hdr)

        out_items = []
        for r in items:
            d = dict(r)
            out_items.append({
                "rank_index": d["rank_index"],
                "work_id": d["work_id"],
                "score": float(d["score"]),
                "reasons": d.get("reasons_json") or [],
                "score_breakdown": d.get("score_breakdown_json") or {},
                "preview": d.get("work_preview_json") or {},
                "provenance": d.get("provenance_json") or [],
            })

        return {"job": job, "items": out_items}

    @staticmethod
    def find_job_by_params_hash(
        conn: Connection,
        *,
        tenant_id: UUID,
        rank_type: str,
        params_hash: str,
    ) -> Optional[dict[str, Any]]:
        row = conn.execute(
            text("""
                SELECT rank_job_id, status
                FROM rank_jobs
                WHERE tenant_id = :t AND rank_type = :rt AND params_hash = :h
                LIMIT 1
            """),
            {"t": tenant_id, "rt": rank_type, "h": params_hash},
        ).mappings().first()
        return dict(row) if row else None



