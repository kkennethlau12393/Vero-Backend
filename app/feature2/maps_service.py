from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.feature2.repos import GraphDraftRepo, MapRepo
from app.feature2.work_topic_store import WorkStore, TopicHierarchyStore, WorkForMap, LabelStore
from app.settings.access_links import resolve_access_link
from app.settings.store import load_workspace_settings

from collections import Counter

MAX_SYNC_NODES = 3000
MAX_SYNC_EDGES = 20000

def _stable_params_hash(*, workspace_id: UUID, graph_draft_id: UUID, layout_mode: str, connector_score_mode: str, grouping_policy_version: int) -> str:
    s = f"{workspace_id}|{graph_draft_id}|{layout_mode}|{connector_score_mode}|gpv:{grouping_policy_version}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _advisory_lock_key(workspace_id: UUID, params_hash: str) -> int:
    b = hashlib.sha256(f"{workspace_id}|{params_hash}".encode("utf-8")).digest()
    # fit into signed bigint range by masking to 63 bits
    return int.from_bytes(b[:8], "big", signed=False) & ((1 << 63) - 1)

def _acquire_advisory_lock(conn, lock_key: int) -> None:
    conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": lock_key})

def _release_advisory_lock(conn, lock_key: int) -> None:
    conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock_key})

def _normalize_graph(node_work_ids: list[str], edges: list[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]], dict[str, int]]:
    # deterministic node list
    node_ids = sorted(set(node_work_ids))
    node_set = set(node_ids)

    dropped_missing_endpoint = 0
    dropped_self_loops = 0

    edge_set: set[tuple[str, str]] = set()
    for a, b in edges:
        if a not in node_set or b not in node_set:
            dropped_missing_endpoint += 1
            continue
        if a == b:
            dropped_self_loops += 1
            continue
        edge_set.add((a, b))

    norm_edges = sorted(edge_set, key=lambda x: (x[0], x[1]))  # deterministic
    stats = {
        "dropped_edge_missing_endpoint": dropped_missing_endpoint,
        "dropped_self_loops": dropped_self_loops,
        "deduped_nodes": len(node_work_ids) - len(node_ids),
        "deduped_edges": len(edges) - len(norm_edges),
    }
    return node_ids, norm_edges, stats

def _pick_best_topic(w: WorkForMap) -> tuple[Optional[str], Optional[float]]:
    # primary_topic wins; else max score in topics with deterministic tie-break
    if w.primary_topic_id is not None:
        return w.primary_topic_id, (w.primary_topic_score if w.primary_topic_score is not None else None)

    if not w.topics:
        return None, None

    # max by score, tie-break by topic_id lexicographically
    best = sorted(w.topics, key=lambda t: (-t.score, t.topic_id))[0]
    return best.topic_id, float(best.score)

def _compute_default_grouping(best_subfield_ids: list[Optional[str]]) -> tuple[str, dict[str, Any]]:
    total = len(best_subfield_ids)
    counts: dict[str, int] = {}
    for sid in best_subfield_ids:
        if sid is None:
            continue
        counts[sid] = counts.get(sid, 0) + 1

    distinct_subfields = len(counts)
    max_share = 0.0
    if total > 0 and counts:
        max_share = max(counts.values()) / float(total)

    if distinct_subfields < 4:
        default_grouping = "topic"
    elif max_share >= 0.8:
        default_grouping = "topic"
    else:
        default_grouping = "subfield"

    return default_grouping, {
        "distinct_subfields": distinct_subfields,
        "max_subfield_share": max_share,
        "subfield_counts_top": sorted(
            [{"subfield_id": k, "node_count": v, "share": v / float(total)} for k, v in counts.items()],
            key=lambda x: (-x["node_count"], x["subfield_id"]),
        )[:10],
    }

def _compute_degree_connector_scores(node_ids: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    deg = {nid: 0 for nid in node_ids}
    for a, b in edges:
        # out + in
        deg[a] += 1
        deg[b] += 1
    return deg

def build_map(
    engine: Engine,
    *,
    workspace_id: UUID | None = None,
    tenant_id: UUID | None = None,
    graph_draft_id: UUID,
    connector_score_mode: str = "degree",
    layout_mode: str = "none",
    grouping_policy_version: int = 1,
) -> dict[str, Any]:
    workspace_id = workspace_id or tenant_id
    if workspace_id is None:
        raise ValueError("workspace_id is required")

    params_hash = _stable_params_hash(
        workspace_id=workspace_id,
        graph_draft_id=graph_draft_id,
        layout_mode=layout_mode,
        connector_score_mode=connector_score_mode,
        grouping_policy_version=grouping_policy_version,
    )

    # Step 0: cheap idempotency check
    with engine.connect() as conn:
        existing = MapRepo.find_existing_map_id(conn, workspace_id, params_hash)
        if existing:
            row = conn.execute(
                text("SELECT default_grouping, allowed_groupings, stats_json FROM maps WHERE map_id=:m AND workspace_id=:w"),
                {"m": existing, "w": workspace_id},
            ).mappings().first()
            return {
                "map_id": existing,
                "default_grouping": row["default_grouping"],
                "allowed_groupings": row["allowed_groupings"],
                "stats": row["stats_json"],
            }

    lock_key = _advisory_lock_key(workspace_id, params_hash)

    # Single-builder concurrency safety (session-level lock held during compute)
    with engine.connect() as conn:
        _acquire_advisory_lock(conn, lock_key)
        try:
            # Re-check after lock (prevents thundering herd duplicates)
            existing = MapRepo.find_existing_map_id(conn, workspace_id, params_hash)
            if existing:
                row = conn.execute(
                    text("SELECT default_grouping, allowed_groupings, stats_json FROM maps WHERE map_id=:m AND workspace_id=:w"),
                    {"m": existing, "w": workspace_id},
                ).mappings().first()
                return {
                    "map_id": existing,
                    "default_grouping": row["default_grouping"],
                    "allowed_groupings": row["allowed_groupings"],
                    "stats": row["stats_json"],
                }

            # Step 1: load + validate GraphDraft
            gd = GraphDraftRepo.load(conn, workspace_id, graph_draft_id)

            if not gd.node_work_ids:
                raise ValueError("graph_draft_empty")

            # size gate (sync vs async)
            if len(gd.node_work_ids) > MAX_SYNC_NODES or len(gd.edges) > MAX_SYNC_EDGES:
                raise ValueError("graph_too_large_for_sync")  # in prod: return 202 + enqueue async

            # Step 2: normalize graph
            node_ids, norm_edges, norm_stats = _normalize_graph(gd.node_work_ids, gd.edges)

            # Step 3: ensure Work + topic data exists (cache-first)
            works = WorkStore.load_many(conn, node_ids)
            missing = [wid for wid in node_ids if wid not in works]
            WorkStore.ensure_works_present(conn, missing)
            if missing:
                works = WorkStore.load_many(conn, node_ids)  # re-load after attempted fetch

            # Step 4: assign best_topic_id + best_subfield_id deterministically
            best_topic_by_work: dict[str, Optional[str]] = {}
            topic_score_by_work: dict[str, Optional[float]] = {}
            for wid in node_ids:
                w = works.get(wid)
                if not w:
                    best_topic_by_work[wid] = None
                    topic_score_by_work[wid] = None
                    continue
                tid, tscore = _pick_best_topic(w)
                best_topic_by_work[wid] = tid
                topic_score_by_work[wid] = tscore

            topic_ids = sorted({tid for tid in best_topic_by_work.values() if tid is not None})
            topic_map = TopicHierarchyStore.load_topic_to_subfield_and_field(conn, topic_ids)

            best_subfield_by_work: dict[str, Optional[str]] = {}
            field_by_work: dict[str, Optional[str]] = {}
            missing_topic_count = 0
            for wid in node_ids:
                tid = best_topic_by_work[wid]
                if tid is None:
                    missing_topic_count += 1
                    best_subfield_by_work[wid] = None
                    field_by_work[wid] = None
                    continue
                hier = topic_map.get(tid)
                if not hier:
                    missing_topic_count += 1
                    best_subfield_by_work[wid] = None
                    field_by_work[wid] = None
                    continue
                best_subfield_by_work[wid] = hier["subfield_id"]
                field_by_work[wid] = hier["field_id"]

            # Step 5: compute header fields breakdown
            total_nodes = len(node_ids)
            field_counts: dict[str, int] = {}
            unclassified_count = 0
            for wid in node_ids:
                fid = field_by_work.get(wid)
                if fid is None:
                    unclassified_count += 1
                    continue
                field_counts[fid] = field_counts.get(fid, 0) + 1

            field_breakdown = sorted(
                [{"field_id": fid, "node_count": c, "share": c / float(total_nodes)} for fid, c in field_counts.items()],
                key=lambda x: (-x["node_count"], x["field_id"]),
            )

            # Step 6: choose default grouping
            best_subfields = [best_subfield_by_work[wid] for wid in node_ids]
            default_grouping, grouping_stats = _compute_default_grouping(best_subfields)

            # Step 7: connector scores
            connector_score_by_work: dict[str, float] = {wid: 0.0 for wid in node_ids}
            if connector_score_mode == "degree":
                deg = _compute_degree_connector_scores(node_ids, norm_edges)
                connector_score_by_work = {k: float(v) for k, v in deg.items()}
            elif connector_score_mode == "none":
                pass
            else:
                # pagerank support can be added later; keep degree as default
                raise ValueError("pagerank_not_implemented")

            # Step 8: layout (none by default)
            xy_by_work: dict[str, tuple[Optional[float], Optional[float]]] = {wid: (None, None) for wid in node_ids}

            # Step 9: atomic save
            map_id = uuid4()
            stats_json = {
                "node_count": len(node_ids),
                "edge_count": len(norm_edges),
                **norm_stats,
                "missing_topic_count": missing_topic_count,
                "unclassified_count": unclassified_count,
                "field_breakdown": field_breakdown,
                **grouping_stats,
            }
            params_json = {
                "layout_mode": layout_mode,
                "connector_score_mode": connector_score_mode,
                "grouping_policy_version": grouping_policy_version,
            }

            allowed_groupings = ["topic", "subfield"]

            # transaction: header + nodes + edges all-or-nothing
            with engine.begin() as tx:
                # unique conflict handling (rare but correct)
                try:
                    MapRepo.insert_map_header(
                        tx,
                        map_id=map_id,
                        workspace_id=workspace_id,
                        graph_draft_id=graph_draft_id,
                        default_grouping=default_grouping,
                        allowed_groupings=allowed_groupings,
                        stats_json=stats_json,
                        params_json=params_json,
                        params_hash=params_hash,
                    )
                except Exception:
                    # If conflict, return the winner map_id and do not insert nodes/edges
                    winner = MapRepo.find_existing_map_id(tx, workspace_id, params_hash)
                    if not winner:
                        raise
                    row = tx.execute(
                        text("SELECT default_grouping, allowed_groupings, stats_json FROM maps WHERE map_id=:m AND workspace_id=:w"),
                        {"m": winner, "w": workspace_id},
                    ).mappings().first()
                    return {
                        "map_id": winner,
                        "default_grouping": row["default_grouping"],
                        "allowed_groupings": row["allowed_groupings"],
                        "stats": row["stats_json"],
                    }

                node_rows: list[dict[str, Any]] = []
                for wid in node_ids:
                    w = works.get(wid)
                    x, y = xy_by_work[wid]
                    node_rows.append(
                        {
                            "map_id": map_id,
                            "work_id": wid,
                            "best_topic_id": best_topic_by_work[wid],
                            "best_subfield_id": best_subfield_by_work[wid],
                            "topic_score": topic_score_by_work[wid],
                            "connector_score": connector_score_by_work.get(wid, 0.0),
                            "x": x,
                            "y": y,
                            "work_preview_json": (w.preview_json() if w else {"work_id": wid}),
                        }
                    )

                edge_rows = [{"map_id": map_id, "from_work_id": a, "to_work_id": b} for (a, b) in norm_edges]

                # bulk inserts (executemany)
                MapRepo.bulk_insert_nodes(tx, node_rows)
                MapRepo.bulk_insert_edges(tx, edge_rows)

            # Step 10: response
            return {
                "map_id": map_id,
                "default_grouping": default_grouping,
                "allowed_groupings": allowed_groupings,
                "stats": stats_json,
            }
        finally:
            _release_advisory_lock(conn, lock_key)

def get_map_render_payload(
    engine: Engine,
    *,
    workspace_id: UUID | None = None,
    tenant_id: UUID | None = None,
    map_id: UUID,
    group_by: Optional[str] = None,
) -> dict[str, Any]:
    """
    GET /v1/maps/{map_id}?group_by=topic|subfield
    Loads stored map + nodes + edges and returns a stable render payload.
    """

    workspace_id = workspace_id or tenant_id
    if workspace_id is None:
        raise ValueError("workspace_id is required")

    with engine.connect() as conn:
        # 1) Load map header (enforce tenant ownership)
        hdr = MapRepo.load_map_header(conn, workspace_id=workspace_id, map_id=map_id)

        # 2) Load nodes + edges
        node_rows = MapRepo.load_map_nodes(conn, map_id=map_id)
        edge_rows = MapRepo.load_map_edges(conn, map_id=map_id)

        # 3) Decide grouping
        if group_by is not None:
            if group_by not in hdr["allowed_groupings"]:
                raise ValueError("invalid_group_by")
            active_grouping = group_by
        else:
            active_grouping = hdr["default_grouping"]

        # 4) Batch load previews (NO N+1)
        work_ids = [n["work_id"] for n in node_rows]
        previews = WorkStore.load_previews_many(conn, work_ids)

        # 4b) Load workspace settings for access link resolution
        workspace_settings = load_workspace_settings(conn, workspace_id)
        proxy_prefix = workspace_settings.get("institutional_proxy_prefix")
        libkey_api_key = workspace_settings.get("libkey_api_key")
        libkey_library_id = workspace_settings.get("libkey_library_id")

        # 5) Build nodes[]
        nodes_out: list[dict[str, Any]] = []
        for n in node_rows:
            if active_grouping == "subfield":
                gid = n["best_subfield_id"]
            else:
                gid = n["best_topic_id"]

            group_id = gid if gid is not None else "unclassified"

            p = previews.get(n["work_id"])
            if p is None:
                node = {
                    "work_id": n["work_id"],
                    "group_id": group_id,
                    "title": None,
                    "year": None,
                    "cited_by_count": 0,
                    "authors": [],
                    "venue": None,
                    "access_status": "unknown",
                    "pdf_url": None,
                    "connector_score": n["connector_score"],
                    "x": n["x"],
                    "y": n["y"],
                }
            else:
                access = resolve_access_link(
                    doi=p.doi,
                    is_open_access=p.is_open_access,
                    oa_pdf_url=p.oa_pdf_url,
                    proxy_prefix=proxy_prefix,
                    libkey_api_key=libkey_api_key,
                    libkey_library_id=libkey_library_id,
                )
                node = {
                    "work_id": p.work_id,
                    "group_id": group_id,
                    "title": p.title,
                    "year": p.year,
                    "cited_by_count": int(p.cited_by_count or 0),
                    "authors": p.authors or [],
                    "venue": p.venue,
                    "access_status": access["access_status"],
                    "pdf_url": access["pdf_url"],
                    "connector_score": n["connector_score"],
                    "x": n["x"],
                    "y": n["y"],
                }

            nodes_out.append(node)

        total_nodes = len(nodes_out) if nodes_out else 1

        # 6) Group summaries (counts + labels)
        counts = Counter([n["group_id"] for n in nodes_out])
        group_ids_real = [gid for gid in counts.keys() if gid != "unclassified"]

        if active_grouping == "topic":
            label_map = LabelStore.topic_labels(conn, group_ids_real)
        else:
            label_map = LabelStore.subfield_labels(conn, group_ids_real)

        group_summaries: list[dict[str, Any]] = []
        for gid, c in counts.items():
            label = "Unclassified" if gid == "unclassified" else label_map.get(gid, gid)
            group_summaries.append(
                {
                    "group_id": gid,
                    "label": label,
                    "node_count": int(c),
                    "share": float(c) / float(total_nodes),
                }
            )

        group_summaries.sort(key=lambda x: (-x["node_count"], x["group_id"]))

        # 7) Field contexts header (use stored stats_json.field_breakdown)
        field_breakdown = (hdr["stats_json"] or {}).get("field_breakdown") or []
        field_ids_ordered = [fb.get("field_id") for fb in field_breakdown if fb.get("field_id")]

        field_label_map = LabelStore.field_labels(conn, field_ids_ordered)

        field_contexts: list[dict[str, Any]] = []
        for fb in field_breakdown:
            fid = fb.get("field_id")
            if not fid:
                continue
            field_contexts.append(
                {
                    "field_id": fid,
                    "label": field_label_map.get(fid, fid),
                    "node_count": int(fb.get("node_count", 0)),
                    "share": float(fb.get("share", 0.0)),
                }
            )

        # 8) Edges
        edges_out = [{"from_work_id": a, "to_work_id": b} for (a, b) in edge_rows]

        return {
            "map_id": map_id,
            "field_contexts": field_contexts,
            "active_grouping": active_grouping,
            "allowed_groupings": hdr["allowed_groupings"],
            "group_summaries": group_summaries,
            "nodes": nodes_out,
            "edges": edges_out,
        }
