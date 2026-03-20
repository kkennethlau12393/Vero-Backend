"""Admin Analytics API — time-series data for dashboard charts."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.admin import require_admin
from app.db import make_engine
from app.admin.admin_service import ensure_admin_tables, COST_PER_CREDIT_GBP, PRO_SUBSCRIPTION_GBP
from app.admin.analytics_schemas import (
    UserGrowthResponse,
    RevenueResponse,
    CreditUsageResponse,
    RetentionResponse,
    OverviewResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin/analytics", tags=["admin-analytics"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


# ---------------------------------------------------------------------------
# GET /v1/admin/analytics/user-growth
# ---------------------------------------------------------------------------

@router.get("/user-growth", response_model=UserGrowthResponse)
def user_growth(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    interval_map = {"7d": "7 days", "30d": "30 days", "90d": "90 days", "1y": "365 days"}
    interval = interval_map[period]

    with engine.connect() as conn:
        ensure_admin_tables(conn)

        # Daily signups
        rows = conn.execute(text("""
            SELECT
                date_trunc('day', created_at)::date AS date,
                count(*) AS signups
            FROM auth.users
            WHERE created_at >= now() - :interval::interval
            GROUP BY 1
            ORDER BY 1
        """), {"interval": interval}).mappings().all()

        # Cumulative total
        total_before = conn.execute(text("""
            SELECT count(*) AS cnt FROM auth.users
            WHERE created_at < now() - :interval::interval
        """), {"interval": interval}).mappings().first()

        cumulative = int(total_before["cnt"])
        data_points = []
        for r in rows:
            cumulative += int(r["signups"])
            data_points.append({
                "date": r["date"].isoformat(),
                "signups": int(r["signups"]),
                "cumulative": cumulative,
            })

    return UserGrowthResponse(period=period, data=data_points)


# ---------------------------------------------------------------------------
# GET /v1/admin/analytics/revenue
# ---------------------------------------------------------------------------

@router.get("/revenue", response_model=RevenueResponse)
def revenue(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    interval_map = {"7d": "7 days", "30d": "30 days", "90d": "90 days", "1y": "365 days"}
    interval = interval_map[period]

    with engine.connect() as conn:
        ensure_admin_tables(conn)

        # Daily credit consumption (negative amounts = usage)
        rows = conn.execute(text("""
            SELECT
                date_trunc('day', created_at)::date AS date,
                COALESCE(SUM(ABS(amount)), 0) AS credits_used
            FROM credit_transactions
            WHERE amount < 0 AND created_at >= now() - :interval::interval
            GROUP BY 1
            ORDER BY 1
        """), {"interval": interval}).mappings().all()

        # Pro subscriber count per day (approximate: current count)
        pro_count = conn.execute(text("""
            SELECT count(*) AS cnt FROM user_billing WHERE plan = 'pro'
        """)).mappings().first()
        daily_sub_revenue = (int(pro_count["cnt"]) * PRO_SUBSCRIPTION_GBP) / 30.0

        data_points = []
        for r in rows:
            credits = float(r["credits_used"])
            cost = credits * COST_PER_CREDIT_GBP
            data_points.append({
                "date": r["date"].isoformat(),
                "credits_used": round(credits, 2),
                "estimated_cost": round(cost, 2),
                "estimated_revenue": round(daily_sub_revenue, 2),
            })

    return RevenueResponse(period=period, data=data_points)


# ---------------------------------------------------------------------------
# GET /v1/admin/analytics/credit-usage
# ---------------------------------------------------------------------------

@router.get("/credit-usage", response_model=CreditUsageResponse)
def credit_usage(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    interval_map = {"7d": "7 days", "30d": "30 days", "90d": "90 days", "1y": "365 days"}
    interval = interval_map[period]

    with engine.connect() as conn:
        ensure_admin_tables(conn)

        # By reason/feature
        by_feature = conn.execute(text("""
            SELECT
                reason,
                count(*) AS usage_count,
                COALESCE(SUM(ABS(amount)), 0) AS total_credits
            FROM credit_transactions
            WHERE amount < 0 AND created_at >= now() - :interval::interval
            GROUP BY reason
            ORDER BY total_credits DESC
        """), {"interval": interval}).mappings().all()

        # Daily breakdown
        daily = conn.execute(text("""
            SELECT
                date_trunc('day', created_at)::date AS date,
                reason,
                COALESCE(SUM(ABS(amount)), 0) AS credits
            FROM credit_transactions
            WHERE amount < 0 AND created_at >= now() - :interval::interval
            GROUP BY 1, 2
            ORDER BY 1
        """), {"interval": interval}).mappings().all()

    features = [
        {
            "reason": r["reason"],
            "usage_count": int(r["usage_count"]),
            "total_credits": round(float(r["total_credits"]), 2),
        }
        for r in by_feature
    ]

    daily_data = [
        {
            "date": r["date"].isoformat(),
            "reason": r["reason"],
            "credits": round(float(r["credits"]), 2),
        }
        for r in daily
    ]

    return CreditUsageResponse(period=period, by_feature=features, daily=daily_data)


# ---------------------------------------------------------------------------
# GET /v1/admin/analytics/retention
# ---------------------------------------------------------------------------

@router.get("/retention", response_model=RetentionResponse)
def retention(
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)

        # Cohort-based retention: what % of users who signed up N days ago
        # have been active (last_sign_in_at) since signup
        cohorts = conn.execute(text("""
            WITH cohort AS (
                SELECT
                    id,
                    created_at,
                    last_sign_in_at
                FROM auth.users
                WHERE created_at >= now() - interval '90 days'
            )
            SELECT
                period,
                total,
                retained,
                CASE WHEN total > 0 THEN round(100.0 * retained / total, 1) ELSE 0 END AS rate
            FROM (
                SELECT 'Day 1' AS period, 1 AS sort_order,
                    count(*) AS total,
                    count(*) FILTER (WHERE last_sign_in_at > created_at + interval '1 day') AS retained
                FROM cohort
                WHERE created_at <= now() - interval '1 day'
                UNION ALL
                SELECT 'Day 7', 2,
                    count(*),
                    count(*) FILTER (WHERE last_sign_in_at > created_at + interval '7 days') AS retained
                FROM cohort
                WHERE created_at <= now() - interval '7 days'
                UNION ALL
                SELECT 'Day 14', 3,
                    count(*),
                    count(*) FILTER (WHERE last_sign_in_at > created_at + interval '14 days') AS retained
                FROM cohort
                WHERE created_at <= now() - interval '14 days'
                UNION ALL
                SELECT 'Day 30', 4,
                    count(*),
                    count(*) FILTER (WHERE last_sign_in_at > created_at + interval '30 days') AS retained
                FROM cohort
                WHERE created_at <= now() - interval '30 days'
                UNION ALL
                SELECT 'Day 60', 5,
                    count(*),
                    count(*) FILTER (WHERE last_sign_in_at > created_at + interval '60 days') AS retained
                FROM cohort
                WHERE created_at <= now() - interval '60 days'
                UNION ALL
                SELECT 'Day 90', 6,
                    count(*),
                    count(*) FILTER (WHERE last_sign_in_at > created_at + interval '90 days') AS retained
                FROM cohort
                WHERE created_at <= now() - interval '90 days'
            ) sub
            ORDER BY sort_order
        """)).mappings().all()

    data = [
        {
            "period": r["period"],
            "total": int(r["total"]),
            "retained": int(r["retained"]),
            "rate": float(r["rate"]),
        }
        for r in cohorts
    ]

    return RetentionResponse(data=data)


# ---------------------------------------------------------------------------
# GET /v1/admin/analytics/overview
# ---------------------------------------------------------------------------

@router.get("/overview", response_model=OverviewResponse)
def overview(
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)

        row = conn.execute(text("""
            SELECT
                (SELECT count(*) FROM auth.users
                 WHERE created_at >= date_trunc('day', now())) AS signups_today,
                (SELECT count(*) FROM auth.users
                 WHERE last_sign_in_at >= now() - interval '24 hours') AS active_24h,
                (SELECT count(*) FROM user_billing WHERE plan = 'pro') AS pro_subscribers,
                (SELECT COALESCE(SUM(ABS(amount)), 0) FROM credit_transactions
                 WHERE amount < 0 AND created_at >= date_trunc('day', now())) AS credits_today,
                (SELECT COALESCE(SUM(ABS(amount)), 0) FROM credit_transactions
                 WHERE amount < 0 AND created_at >= now() - interval '24 hours') AS credits_24h,
                (SELECT COALESCE(SUM(ABS(amount)), 0) FROM credit_transactions
                 WHERE amount < 0 AND created_at >= now() - interval '7 days') AS credits_7d
        """)).mappings().first()

        pro_count = int(row["pro_subscribers"])
        credits_7d = float(row["credits_7d"])
        daily_burn_rate = credits_7d / 7.0 if credits_7d > 0 else 0

    return OverviewResponse(
        signups_today=int(row["signups_today"]),
        active_24h=int(row["active_24h"]),
        pro_subscribers=pro_count,
        credits_used_today=round(float(row["credits_today"]), 2),
        credits_used_24h=round(float(row["credits_24h"]), 2),
        daily_burn_rate=round(daily_burn_rate, 2),
        estimated_daily_revenue=round((pro_count * PRO_SUBSCRIPTION_GBP) / 30.0, 2),
    )
