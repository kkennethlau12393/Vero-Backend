"""
Topic lookup utilities for Feature 3.

This module fetches topic display names from OpenAlex API and caches them
locally in the openalex_topics table for future lookups.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

OPENALEX_API_BASE = "https://api.openalex.org"
OPENALEX_TIMEOUT = 10.0


def get_topic_display_name(
    conn: Connection,
    topic_id: str,
) -> Optional[str]:
    """
    Get the display name for an OpenAlex topic.

    First checks local cache (openalex_topics table), then falls back to
    OpenAlex API if not found locally.

    Args:
        conn: Database connection
        topic_id: OpenAlex topic ID (e.g., "T12345")

    Returns:
        Topic display name or None if not found
    """
    if not topic_id:
        return None

    # Check local cache first
    cached_name = _get_cached_topic_name(conn, topic_id)
    if cached_name:
        return cached_name

    # Fetch from OpenAlex API
    display_name = _fetch_topic_from_openalex(topic_id)

    # Cache locally if found
    if display_name:
        _cache_topic_name(conn, topic_id, display_name)

    return display_name


def _get_cached_topic_name(conn: Connection, topic_id: str) -> Optional[str]:
    """Check if topic display name is cached locally."""
    try:
        row = conn.execute(
            text("""
                SELECT display_name FROM openalex_topics
                WHERE topic_id = :topic_id AND display_name IS NOT NULL
            """),
            {"topic_id": topic_id},
        ).mappings().first()

        if row and row.get("display_name"):
            return row["display_name"]
        return None
    except Exception as e:
        logger.warning(f"Failed to check cached topic name: {e}")
        return None


def _cache_topic_name(conn: Connection, topic_id: str, display_name: str) -> None:
    """Cache topic display name locally."""
    try:
        # Use upsert to handle both insert and update cases
        conn.execute(
            text("""
                INSERT INTO openalex_topics (topic_id, display_name, subfield_id, field_id)
                VALUES (:topic_id, :display_name, '', '')
                ON CONFLICT (topic_id) DO UPDATE SET
                    display_name = EXCLUDED.display_name
            """),
            {"topic_id": topic_id, "display_name": display_name},
        )
        conn.commit()
        logger.debug(f"Cached topic display name: {topic_id} -> {display_name}")
    except Exception as e:
        logger.warning(f"Failed to cache topic name: {e}")


def _fetch_topic_from_openalex(topic_id: str) -> Optional[str]:
    """
    Fetch topic display name from OpenAlex API.

    Args:
        topic_id: OpenAlex topic ID (e.g., "T12345")

    Returns:
        Display name or None if not found
    """
    if not topic_id:
        return None

    try:
        # OpenAlex topic IDs are like "T12345", API expects full URL or just the ID
        # The topics endpoint uses the format: /topics/T12345
        url = f"{OPENALEX_API_BASE}/topics/{topic_id}"

        with httpx.Client(timeout=OPENALEX_TIMEOUT) as client:
            resp = client.get(url, params={"mailto": "api@alexandria.app"})

            if resp.status_code == 404:
                logger.debug(f"Topic not found in OpenAlex: {topic_id}")
                return None

            resp.raise_for_status()
            data = resp.json()

            display_name = data.get("display_name")
            if display_name:
                logger.info(f"Fetched topic from OpenAlex: {topic_id} -> {display_name}")
                return display_name

            return None

    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching topic from OpenAlex: {topic_id}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch topic from OpenAlex: {topic_id}, error: {e}")
        return None
