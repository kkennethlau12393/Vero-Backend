"""Admin service — DB queries, Stripe integration, Supabase Admin API."""
from __future__ import annotations

import json
import logging
import os
from uuid import UUID

import httpx
import stripe
from sqlalchemy import text

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

PRO_CREDITS = 200
FREE_CREDITS = 10
PRO_MRR = 20.0  # estimated monthly revenue per pro subscriber
COST_PER_CREDIT_GBP = 0.05
PRO_SUBSCRIPTION_GBP = 7.99

# ---------------------------------------------------------------------------
# DDL — idempotent table setup
# ---------------------------------------------------------------------------

_ADMIN_DDL = """
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS public.admin_audit_log (
    id              bigserial PRIMARY KEY,
    admin_user_id   uuid NOT NULL REFERENCES auth.users(id),
    action          text NOT NULL,
    target_user_id  uuid REFERENCES auth.users(id),
    details         jsonb NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_created ON admin_audit_log(created_at DESC);

ALTER TABLE public.user_billing ADD COLUMN IF NOT EXISTS admin_access_expires_at timestamptz;
"""

_tables_ensured = False


def ensure_admin_tables(conn) -> None:
    """Create admin tables if they don't exist (idempotent)."""
    global _tables_ensured
    if _tables_ensured:
        return
    conn.execute(text(_ADMIN_DDL))
    conn.commit()
    _tables_ensured = True


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def log_audit(conn, admin_user_id: UUID, action: str, target_user_id: UUID | None, details: dict | None = None) -> None:
    conn.execute(
        text("""
            INSERT INTO admin_audit_log (admin_user_id, action, target_user_id, details)
            VALUES (:admin_id, :action, :target_id, :details)
        """),
        {
            "admin_id": admin_user_id,
            "action": action,
            "target_id": target_user_id,
            "details": json.dumps(details or {}),
        },
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_admin_stats(conn) -> dict:
    row = conn.execute(text("""
        SELECT
            (SELECT count(*) FROM auth.users) AS total_users,
            (SELECT count(*) FROM user_billing WHERE plan = 'pro') AS pro_subscribers,
            (SELECT count(*) FROM auth.users
             WHERE created_at >= now() - interval '7 days') AS signups_7d,
            (SELECT count(*) FROM auth.users
             WHERE last_sign_in_at >= now() - interval '7 days') AS active_7d,
            (SELECT count(*) FROM auth.users
             WHERE last_sign_in_at >= now() - interval '30 days') AS active_30d
    """)).mappings().first()

    stats = dict(row)
    stats["mrr_estimate"] = float(stats["pro_subscribers"]) * PRO_MRR

    recent = conn.execute(text("""
        SELECT email, created_at
        FROM auth.users
        ORDER BY created_at DESC
        LIMIT 10
    """)).mappings().all()

    stats["recent_signups"] = [
        {"email": r["email"], "created_at": r["created_at"].isoformat()}
        for r in recent
    ]

    # Cost stats (30d)
    cost_row = conn.execute(text("""
        SELECT COALESCE(SUM(ABS(amount)), 0) AS total_credits
        FROM credit_transactions
        WHERE amount < 0 AND created_at >= now() - interval '30 days'
    """)).mappings().first()
    total_credits_30d = float(cost_row["total_credits"])
    estimated_cost_30d = total_credits_30d * COST_PER_CREDIT_GBP
    estimated_revenue_30d = float(stats["pro_subscribers"]) * PRO_SUBSCRIPTION_GBP
    stats["total_credits_consumed_30d"] = total_credits_30d
    stats["estimated_cost_30d"] = round(estimated_cost_30d, 2)
    stats["estimated_margin_30d"] = round(estimated_revenue_30d - estimated_cost_30d, 2)

    return stats


# ---------------------------------------------------------------------------
# User listing
# ---------------------------------------------------------------------------

_SORT_COLUMNS = {
    "created_at": "au.created_at",
    "email": "au.email",
    "last_sign_in_at": "au.last_sign_in_at",
    "plan": "COALESCE(ub.plan, 'free')",
    "credits_remaining": "COALESCE(ub.credits_remaining, 0)",
}


def get_users_paginated(
    conn,
    search: str | None = None,
    page: int = 1,
    per_page: int = 25,
    plan: str | None = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    status: str | None = None,
    min_credits: float | None = None,
    max_credits: float | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    active_since: str | None = None,
    is_admin: bool | None = None,
) -> dict:
    where_clauses = []
    params: dict = {}

    if search:
        where_clauses.append("(au.email ILIKE :search OR au.raw_user_meta_data->>'full_name' ILIKE :search)")
        params["search"] = f"%{search}%"

    if plan:
        if plan == "free":
            where_clauses.append("(ub.plan IS NULL OR ub.plan = 'free')")
        else:
            where_clauses.append("ub.plan = :plan")
            params["plan"] = plan

    if status:
        if status == "none":
            where_clauses.append("(ub.subscription_status IS NULL OR ub.subscription_status = 'none')")
        else:
            where_clauses.append("ub.subscription_status = :status")
            params["status"] = status

    if min_credits is not None:
        where_clauses.append("COALESCE(ub.credits_remaining, 0) >= :min_credits")
        params["min_credits"] = min_credits

    if max_credits is not None:
        where_clauses.append("COALESCE(ub.credits_remaining, 0) <= :max_credits")
        params["max_credits"] = max_credits

    if created_after:
        where_clauses.append("au.created_at >= :created_after")
        params["created_after"] = created_after

    if created_before:
        where_clauses.append("au.created_at <= :created_before")
        params["created_before"] = created_before

    if active_since:
        where_clauses.append("au.last_sign_in_at >= :active_since")
        params["active_since"] = active_since

    if is_admin is not None:
        where_clauses.append("COALESCE(pu.is_admin, false) = :is_admin")
        params["is_admin"] = is_admin

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sort_col = _SORT_COLUMNS.get(sort_by, "au.created_at")
    sort_direction = "ASC" if sort_dir.lower() == "asc" else "DESC"

    # Count
    count_row = conn.execute(
        text(f"""
            SELECT count(*) AS total
            FROM auth.users au
            LEFT JOIN user_billing ub ON ub.user_id = au.id
            LEFT JOIN public.users pu ON pu.user_id = au.id
            {where_sql}
        """),
        params,
    ).mappings().first()
    total = count_row["total"]

    # Page
    offset = (page - 1) * per_page
    params["limit"] = per_page
    params["offset"] = offset

    rows = conn.execute(
        text(f"""
            SELECT
                au.id AS user_id,
                au.email,
                COALESCE(pu.display_name, au.raw_user_meta_data->>'full_name') AS name,
                COALESCE(ub.plan, 'free') AS plan,
                COALESCE(ub.credits_remaining, 0) AS credits_remaining,
                COALESCE(ub.subscription_status, 'none') AS subscription_status,
                COALESCE(pu.is_admin, false) AS is_admin,
                au.created_at,
                au.last_sign_in_at
            FROM auth.users au
            LEFT JOIN user_billing ub ON ub.user_id = au.id
            LEFT JOIN public.users pu ON pu.user_id = au.id
            {where_sql}
            ORDER BY {sort_col} {sort_direction} NULLS LAST
            LIMIT :limit OFFSET :offset
        """),
        params,
    ).mappings().all()

    users = []
    for r in rows:
        users.append({
            "user_id": str(r["user_id"]),
            "email": r["email"],
            "name": r["name"],
            "plan": r["plan"],
            "credits_remaining": float(r["credits_remaining"]),
            "subscription_status": r["subscription_status"],
            "is_admin": bool(r["is_admin"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "last_sign_in_at": r["last_sign_in_at"].isoformat() if r["last_sign_in_at"] else None,
        })

    return {"users": users, "total": total, "page": page, "per_page": per_page}


# ---------------------------------------------------------------------------
# User detail
# ---------------------------------------------------------------------------

def get_user_detail(conn, user_id: UUID) -> dict | None:
    row = conn.execute(
        text("""
            SELECT
                au.id AS user_id,
                au.email,
                COALESCE(pu.display_name, au.raw_user_meta_data->>'full_name') AS name,
                COALESCE(ub.plan, 'free') AS plan,
                COALESCE(ub.credits_remaining, 0) AS credits_remaining,
                COALESCE(ub.subscription_status, 'none') AS subscription_status,
                COALESCE(pu.is_admin, false) AS is_admin,
                au.created_at,
                au.last_sign_in_at,
                au.banned_until
            FROM auth.users au
            LEFT JOIN user_billing ub ON ub.user_id = au.id
            LEFT JOIN public.users pu ON pu.user_id = au.id
            WHERE au.id = :uid
        """),
        {"uid": user_id},
    ).mappings().first()

    if not row:
        return None

    result = {
        "user_id": str(row["user_id"]),
        "email": row["email"],
        "name": row["name"],
        "plan": row["plan"],
        "credits_remaining": float(row["credits_remaining"]),
        "subscription_status": row["subscription_status"],
        "is_admin": bool(row["is_admin"]),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "last_sign_in_at": row["last_sign_in_at"].isoformat() if row["last_sign_in_at"] else None,
        "banned_until": row["banned_until"].isoformat() if row["banned_until"] else None,
    }

    # Billing detail
    billing_row = conn.execute(
        text("SELECT * FROM user_billing WHERE user_id = :uid"),
        {"uid": user_id},
    ).mappings().first()

    if billing_row:
        result["billing"] = {
            "plan": billing_row["plan"],
            "credits_remaining": float(billing_row["credits_remaining"]),
            "credits_monthly": float(billing_row["credits_monthly"]),
            "subscription_status": billing_row["subscription_status"],
            "stripe_customer_id": billing_row.get("stripe_customer_id"),
            "stripe_subscription_id": billing_row.get("stripe_subscription_id"),
            "current_period_end": billing_row["current_period_end"].isoformat() if billing_row.get("current_period_end") else None,
            "admin_access_expires_at": billing_row["admin_access_expires_at"].isoformat() if billing_row.get("admin_access_expires_at") else None,
        }

    # Workspaces
    ws_rows = conn.execute(
        text("""
            SELECT w.workspace_id, w.workspace_name AS name, w.created_at
            FROM workspaces w
            WHERE w.owner_user_id = :uid
            ORDER BY w.created_at DESC
        """),
        {"uid": user_id},
    ).mappings().all()

    result["workspaces"] = [
        {
            "workspace_id": str(r["workspace_id"]),
            "name": r["name"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in ws_rows
    ]

    # Recent transactions
    tx_rows = conn.execute(
        text("""
            SELECT id, amount, balance_after, reason, created_at
            FROM credit_transactions
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 20
        """),
        {"uid": user_id},
    ).mappings().all()

    result["recent_transactions"] = [
        {
            "id": str(r["id"]),
            "amount": float(r["amount"]),
            "balance_after": float(r["balance_after"]),
            "reason": r["reason"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in tx_rows
    ]

    return result


# ---------------------------------------------------------------------------
# Subscription actions
# ---------------------------------------------------------------------------

def grant_pro(conn, admin_user_id: UUID, target_user_id: UUID) -> None:
    """Grant pro plan to a user."""
    conn.execute(
        text("""
            INSERT INTO user_billing (user_id, plan, credits_remaining, credits_monthly, subscription_status)
            VALUES (:uid, 'pro', :credits, :monthly, 'admin_granted')
            ON CONFLICT (user_id) DO UPDATE SET
                plan = 'pro',
                credits_remaining = :credits,
                credits_monthly = :monthly,
                subscription_status = 'admin_granted',
                updated_at = now()
        """),
        {"uid": target_user_id, "credits": PRO_CREDITS, "monthly": PRO_CREDITS},
    )
    conn.execute(
        text("""
            INSERT INTO credit_transactions (user_id, amount, balance_after, reason, metadata)
            VALUES (:uid, :amount, :bal, 'admin_grant_pro', :meta)
        """),
        {
            "uid": target_user_id,
            "amount": PRO_CREDITS,
            "bal": PRO_CREDITS,
            "reason": "admin_grant_pro",
            "meta": json.dumps({"admin_user_id": str(admin_user_id)}),
        },
    )
    log_audit(conn, admin_user_id, "grant_pro", target_user_id)


def revoke_to_free(conn, admin_user_id: UUID, target_user_id: UUID) -> None:
    """Revoke pro plan, downgrade to free."""
    # Check for Stripe subscription to cancel
    row = conn.execute(
        text("SELECT stripe_subscription_id, credits_remaining FROM user_billing WHERE user_id = :uid"),
        {"uid": target_user_id},
    ).mappings().first()

    if row and row.get("stripe_subscription_id"):
        try:
            stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
            stripe.Subscription.cancel(row["stripe_subscription_id"])
        except Exception:
            logger.warning("Failed to cancel Stripe subscription %s", row["stripe_subscription_id"])

    capped_credits = min(float(row["credits_remaining"]), FREE_CREDITS) if row else FREE_CREDITS

    conn.execute(
        text("""
            INSERT INTO user_billing (user_id, plan, credits_remaining, credits_monthly, subscription_status)
            VALUES (:uid, 'free', :credits, :free_monthly, 'none')
            ON CONFLICT (user_id) DO UPDATE SET
                plan = 'free',
                credits_remaining = :credits,
                credits_monthly = :free_monthly,
                subscription_status = 'none',
                stripe_subscription_id = NULL,
                current_period_end = NULL,
                updated_at = now()
        """),
        {"uid": target_user_id, "credits": capped_credits, "free_monthly": FREE_CREDITS},
    )
    log_audit(conn, admin_user_id, "revoke_to_free", target_user_id)


def grant_credits(conn, admin_user_id: UUID, target_user_id: UUID, amount: float) -> None:
    """Add credits to a user."""
    conn.execute(
        text("""
            INSERT INTO user_billing (user_id, credits_remaining)
            VALUES (:uid, :amount)
            ON CONFLICT (user_id) DO UPDATE SET
                credits_remaining = user_billing.credits_remaining + :amount,
                updated_at = now()
        """),
        {"uid": target_user_id, "amount": amount},
    )
    # Get new balance
    row = conn.execute(
        text("SELECT credits_remaining FROM user_billing WHERE user_id = :uid"),
        {"uid": target_user_id},
    ).mappings().first()
    new_balance = float(row["credits_remaining"]) if row else amount

    conn.execute(
        text("""
            INSERT INTO credit_transactions (user_id, amount, balance_after, reason, metadata)
            VALUES (:uid, :amount, :bal, 'admin_grant_credits', :meta)
        """),
        {
            "uid": target_user_id,
            "amount": amount,
            "bal": new_balance,
            "reason": "admin_grant_credits",
            "meta": json.dumps({"admin_user_id": str(admin_user_id), "credits_granted": amount}),
        },
    )
    log_audit(conn, admin_user_id, "grant_credits", target_user_id, {"amount": amount})


def grant_temporary_pro(conn, admin_user_id: UUID, target_user_id: UUID, expires_at: str) -> None:
    """Grant temporary pro access with an expiry date."""
    conn.execute(
        text("""
            INSERT INTO user_billing (user_id, plan, credits_remaining, credits_monthly, subscription_status, admin_access_expires_at)
            VALUES (:uid, 'pro', :credits, :monthly, 'admin_trial', :expires_at)
            ON CONFLICT (user_id) DO UPDATE SET
                plan = 'pro',
                credits_remaining = :credits,
                credits_monthly = :monthly,
                subscription_status = 'admin_trial',
                admin_access_expires_at = :expires_at,
                updated_at = now()
        """),
        {"uid": target_user_id, "credits": PRO_CREDITS, "monthly": PRO_CREDITS, "expires_at": expires_at},
    )
    conn.execute(
        text("""
            INSERT INTO credit_transactions (user_id, amount, balance_after, reason, metadata)
            VALUES (:uid, :amount, :bal, 'admin_grant_temporary_pro', :meta)
        """),
        {
            "uid": target_user_id,
            "amount": PRO_CREDITS,
            "bal": PRO_CREDITS,
            "reason": "admin_grant_temporary_pro",
            "meta": json.dumps({"admin_user_id": str(admin_user_id), "expires_at": expires_at}),
        },
    )
    log_audit(conn, admin_user_id, "grant_temporary_pro", target_user_id, {"expires_at": expires_at})


def process_refund(conn, admin_user_id: UUID, target_user_id: UUID, amount: float, reason: str) -> None:
    """Refund credits back to a user."""
    conn.execute(
        text("""
            INSERT INTO user_billing (user_id, credits_remaining)
            VALUES (:uid, :amount)
            ON CONFLICT (user_id) DO UPDATE SET
                credits_remaining = user_billing.credits_remaining + :amount,
                updated_at = now()
        """),
        {"uid": target_user_id, "amount": amount},
    )
    row = conn.execute(
        text("SELECT credits_remaining FROM user_billing WHERE user_id = :uid"),
        {"uid": target_user_id},
    ).mappings().first()
    new_balance = float(row["credits_remaining"]) if row else amount

    conn.execute(
        text("""
            INSERT INTO credit_transactions (user_id, amount, balance_after, reason, metadata)
            VALUES (:uid, :amount, :bal, 'admin_refund', :meta)
        """),
        {
            "uid": target_user_id,
            "amount": amount,
            "bal": new_balance,
            "reason": "admin_refund",
            "meta": json.dumps({"admin_user_id": str(admin_user_id), "refund_reason": reason}),
        },
    )
    log_audit(conn, admin_user_id, "process_refund", target_user_id, {"amount": amount, "reason": reason})


def extend_billing_cycle(conn, admin_user_id: UUID, target_user_id: UUID, new_end_date: str) -> None:
    """Extend billing cycle end date."""
    conn.execute(
        text("""
            UPDATE user_billing
            SET current_period_end = :new_end, updated_at = now()
            WHERE user_id = :uid
        """),
        {"uid": target_user_id, "new_end": new_end_date},
    )
    log_audit(conn, admin_user_id, "extend_billing", target_user_id, {"new_period_end": new_end_date})


def get_user_cost_breakdown(conn, user_id: UUID) -> dict:
    """Get per-workspace cost breakdown for a user."""
    rows = conn.execute(
        text("""
            SELECT
                ct.metadata->>'workspace_id' AS workspace_id,
                w.workspace_name,
                ct.reason,
                COUNT(*) AS usage_count,
                SUM(ABS(ct.amount)) AS total_credits_used
            FROM credit_transactions ct
            LEFT JOIN workspaces w ON w.workspace_id::text = ct.metadata->>'workspace_id'
            WHERE ct.user_id = :uid AND ct.amount < 0
            GROUP BY ct.metadata->>'workspace_id', w.workspace_name, ct.reason
            ORDER BY total_credits_used DESC
        """),
        {"uid": user_id},
    ).mappings().all()

    entries = []
    total_credits = 0.0
    for r in rows:
        credits_used = float(r["total_credits_used"])
        total_credits += credits_used
        entries.append({
            "workspace_id": r["workspace_id"],
            "workspace_name": r["workspace_name"],
            "reason": r["reason"],
            "usage_count": int(r["usage_count"]),
            "total_credits_used": credits_used,
            "estimated_cost_gbp": round(credits_used * COST_PER_CREDIT_GBP, 4),
        })

    # Determine subscription revenue
    billing_row = conn.execute(
        text("SELECT plan FROM user_billing WHERE user_id = :uid"),
        {"uid": user_id},
    ).mappings().first()
    is_pro = billing_row and billing_row["plan"] == "pro"
    revenue = PRO_SUBSCRIPTION_GBP if is_pro else 0.0
    total_cost = round(total_credits * COST_PER_CREDIT_GBP, 2)

    return {
        "entries": entries,
        "total_credits_used": total_credits,
        "total_estimated_cost_gbp": total_cost,
        "subscription_revenue_gbp": revenue,
        "margin_gbp": round(revenue - total_cost, 2),
    }


# ---------------------------------------------------------------------------
# Supabase Admin API actions
# ---------------------------------------------------------------------------

def _supabase_admin_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def reset_password(target_user_id: UUID) -> dict:
    """Generate a password recovery link via Supabase Admin API."""
    resp = httpx.post(
        f"{SUPABASE_URL}/auth/v1/admin/generate_link",
        headers=_supabase_admin_headers(),
        json={"type": "recovery", "user_id": str(target_user_id)},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def disable_user(target_user_id: UUID) -> None:
    """Ban user via Supabase Admin API."""
    resp = httpx.put(
        f"{SUPABASE_URL}/auth/v1/admin/users/{target_user_id}",
        headers=_supabase_admin_headers(),
        json={"ban_duration": "876000h"},
        timeout=10,
    )
    resp.raise_for_status()


def enable_user(target_user_id: UUID) -> None:
    """Unban user via Supabase Admin API."""
    resp = httpx.put(
        f"{SUPABASE_URL}/auth/v1/admin/users/{target_user_id}",
        headers=_supabase_admin_headers(),
        json={"ban_duration": "none"},
        timeout=10,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def get_audit_log(conn, page: int = 1, per_page: int = 25) -> dict:
    count_row = conn.execute(
        text("SELECT count(*) AS total FROM admin_audit_log"),
    ).mappings().first()
    total = count_row["total"]

    offset = (page - 1) * per_page
    rows = conn.execute(
        text("""
            SELECT
                al.id,
                admin_u.email AS admin_email,
                al.action,
                target_u.email AS target_email,
                al.details,
                al.created_at
            FROM admin_audit_log al
            LEFT JOIN auth.users admin_u ON admin_u.id = al.admin_user_id
            LEFT JOIN auth.users target_u ON target_u.id = al.target_user_id
            ORDER BY al.created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"limit": per_page, "offset": offset},
    ).mappings().all()

    entries = [
        {
            "id": r["id"],
            "admin_email": r["admin_email"],
            "action": r["action"],
            "target_email": r["target_email"],
            "details": r["details"] or {},
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]

    return {"entries": entries, "total": total, "page": page, "per_page": per_page}
